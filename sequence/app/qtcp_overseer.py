"""qTCP packet overseer: the layer above QTCPTransfer that owns packets.

QTCPTransfer handles single-qubit transfers. This layer owns packets:
encoding a secret into N_SHARES shares, dispatching them through QTCPTransfer
under a bounded-parallelism rule, and driving the per-packet outcome
(DELIVERED / LOST) based on the terminal events QTCPTransfer feeds back.

Phase 3a scope: parallelism up to K-1 shares in flight at once, no recursion
on shares that fail. Recursion (the paper's mechanism for surviving losses
at the moment Alice's held count would drop below K) is a separate step.
Local recovery (RECOVERED_LOCALLY outcome) is Phase 4.

Contract (Alice side):
    send_packet(data_memory_index, dst) -> packet_id
    get_packet_outcome(packet_id) -> PacketOutcome | None

Contract (Bob side):
    get_received_packet(src, packet_id) -> state | None

Terminal transitions on QTCPTransfer trigger on_alice_transfer_finished (Alice)
or on_bob_transfer_finished (Bob). Both handlers live here.
"""

from dataclasses import dataclass
from enum import Enum, auto

import sequence.app.qss as qss
from sequence.app.qtcp_transfer import (
    QTCPTransfer, Transfer, BobTransfer,
    QTCPMessage, QTCPMsgType,
    TransferStatus, BobState,
)
from sequence.utils import log


class PacketOutcome(Enum):
    IN_PROGRESS       = auto()
    DELIVERED         = auto()   # >= K_THRESHOLD shares reached Bob
    RECOVERED_LOCALLY = auto()   # Phase 4: Alice reconstructed from her held shares
    LOST              = auto()


class ShareStatus(Enum):
    HELD      = auto()   # never fired; qubit still in Alice's data slot
    IN_FLIGHT = auto()   # send_single_qubit called; no terminal yet
    DELIVERED = auto()   # Bob has it
    FAILED    = auto()   # terminal, did not reach Bob


@dataclass
class PacketRecord:
    """Alice-side per-packet state. Bob doesn't need one; his aggregation runs
    off the BobTransfer records in QTCPTransfer.

    share_slots, share_transfer_ids and share_status are each of length
    N_SHARES; index i corresponds to share_index i throughout.
    """
    packet_id: int
    dst: str
    source_slot: int                            # the caller's original slot
    share_slots: list                           # data slot index per share
    share_transfer_ids: list                    # transfer id per share, or None
    share_status: list                          # ShareStatus per share
    outcome: PacketOutcome = PacketOutcome.IN_PROGRESS


class QTCPOverseer:
    """Packet-lifecycle owner. Sits above QTCPTransfer on both endpoints.

    Attributes:
        app (QTCPTransfer): the transport tool below.
        packets (dict[int, PacketRecord]): Alice's packet table.
        received_packets (dict[(str, int), int]): Bob's delivered-packet table
            (src, packet_id) -> data slot index holding the reconstructed secret.
        next_packet_id (int): Alice's monotonic counter.
    """

    # Cap on simultaneously in-flight shares per packet. Sending more than this
    # would let a burst of failures drop Alice's held count below K_THRESHOLD,
    # at which point neither Alice nor Bob can recover the secret. K-1 keeps
    # the invariant `alice_held + bob_arrived >= K_THRESHOLD` even in the worst
    # case where every currently-in-flight share fails.
    _MAX_IN_FLIGHT = qss.K_THRESHOLD - 1

    def __init__(self, app: QTCPTransfer):
        self.app = app
        app.terminal_observers.append(self)

        # Alice-side
        self.packets: dict[int, PacketRecord] = {}
        self.next_packet_id: int = 0

        # Bob-side
        self.received_packets: dict[tuple[str, int], int] = {}

        log.logger.debug(f"QTCPOverseer for {app.name}: initialized")

    # ==================================================================
    # Alice side
    # ==================================================================

    def send_packet(self, data_memory_index: int, dst: str) -> int:
        """Encode a single-qubit secret into N_SHARES shares and start
        delivery. Returns the packet id.

        Encoding is in place across N_SHARES data slots. The caller's slot is
        one of them (at code position SECRET_INDEX); the rest are allocated.
        After encoding, all N_SHARES slots hold shares; the "original" no
        longer exists as a separate qubit -- it IS one of the shares.

        Delivery is bounded-parallel: up to _MAX_IN_FLIGHT shares are fired
        immediately. Additional shares fire from on_alice_transfer_finished
        as earlier ones terminate.
        """
        ancillas = []
        for _ in range(qss.N_SHARES - 1):
            slot = self.app.alloc_data_slot()
            if slot is None:
                # Roll back. The caller's slot is theirs and untouched --
                # nothing has been encoded yet.
                for a in ancillas:
                    self.app.free_data_slot(a)
                raise RuntimeError(
                    f"QTCPOverseer: only {len(ancillas)} of {qss.N_SHARES - 1} "
                    f"ancilla slots free; cannot encode a packet"
                )
            ancillas.append(slot)

        # Position in this list IS the code position: slots[i] becomes share i.
        # The secret must sit at SECRET_INDEX, which is where the encoder
        # expects it and where the decoder returns it.
        slots = list(ancillas)
        slots.insert(qss.SECRET_INDEX, data_memory_index)

        data_arr = self.app.node.get_component_by_name(self.app.node.data_memo_arr_name)
        keys = [data_arr[s].qstate_key for s in slots]

        rnd = self.app.node.get_generator().random()
        self.app.node.timeline.quantum_manager.run_circuit(qss.ENCODER, keys, rnd)

        packet_id = self.next_packet_id
        self.next_packet_id += 1

        record = PacketRecord(
            packet_id=packet_id,
            dst=dst,
            source_slot=data_memory_index,
            share_slots=slots,
            share_transfer_ids=[None] * qss.N_SHARES,
            share_status=[ShareStatus.HELD] * qss.N_SHARES,
        )
        self.packets[packet_id] = record

        log.logger.info(
            f"QTCPOverseer: packet {packet_id} encoded from slot "
            f"{data_memory_index} into {qss.N_SHARES} shares -> {dst} "
            f"(slots {slots})"
        )

        self._fire_shares(packet_id)
        return packet_id

    def get_packet_outcome(self, packet_id: int) -> PacketOutcome | None:
        """Poll for terminal outcome. None if the packet id is unknown."""
        record = self.packets.get(packet_id)
        return record.outcome if record else None

    def on_alice_transfer_finished(self, transfer: Transfer) -> None:
        """Called by QTCPTransfer._finish when an Alice-side transfer reaches
        terminal. Updates the share's status and advances the packet."""
        if transfer.packet_id is None:
            return  # standalone send, not part of a packet we track

        record = self.packets.get(transfer.packet_id)
        if record is None:
            return

        share_index = transfer.share_index

        if record.outcome is not PacketOutcome.IN_PROGRESS:
            # Packet already decided; this is a late straggler from a share
            # that was still IN_FLIGHT when the decision was made.
            # free_data_slot is idempotent -- for shares whose _finish already
            # freed the slot (DELIVERED, NO_ACK) it no-ops; for NO_ENTANGLEMENT
            # (which _finish preserves) it cleans up the leak.
            self.app.free_data_slot(record.share_slots[share_index])
            return

        if transfer.status is TransferStatus.DELIVERED:
            record.share_status[share_index] = ShareStatus.DELIVERED
        else:
            record.share_status[share_index] = ShareStatus.FAILED

        self._advance_packet(transfer.packet_id)

    def _advance_packet(self, packet_id: int) -> None:
        """Decide the next action for a packet based on current share states.

        Rules, checked in order:
          1. If Bob already has >= K arrivals: success. Clean up Alice's side
             (measure out held shares, cancel un-sent shares).
          2. If Bob can never reach K given what's still held or in flight:
             lost. Same cleanup path as success, different outcome.
          3. Otherwise: try to fire more held shares up to the parallelism cap.
        """
        record = self.packets[packet_id]
        if record.outcome is not PacketOutcome.IN_PROGRESS:
            return

        delivered = sum(1 for s in record.share_status
                        if s is ShareStatus.DELIVERED)
        held = sum(1 for s in record.share_status
                   if s is ShareStatus.HELD)
        in_flight = sum(1 for s in record.share_status
                        if s is ShareStatus.IN_FLIGHT)

        if delivered >= qss.K_THRESHOLD:
            self._finalize_delivered(record)
            return

        # Optimistic upper bound: if every held + in_flight share succeeded, could
        # we still reach K? If not, the packet is unrecoverable.
        if delivered + held + in_flight < qss.K_THRESHOLD:
            self._finalize_lost(record)
            return

        self._fire_shares(packet_id)

    def _fire_shares(self, packet_id: int) -> None:
        """Fire held shares up to the parallelism cap and the goal cap, in
        share_index order.

        Two limits govern how many shares are in flight at once:

          Safety cap (_MAX_IN_FLIGHT = K-1): more than this could let a burst
          of failures drop Alice's exposure below K_THRESHOLD.

          Goal cap: delivered + in_flight should not exceed K_THRESHOLD.
          Beyond that, an additional share is one that -- if it delivered --
          would just be surplus Bob measures out, and if it failed would have
          exposed a share we did not need to expose. Every share past K in
          play is unnecessary risk on Alice's side and unnecessary work on
          Bob's.

        The lower of the two limits is what caps firing at any moment.
        """
        record = self.packets[packet_id]

        while True:
            delivered = sum(1 for s in record.share_status
                            if s is ShareStatus.DELIVERED)
            in_flight = sum(1 for s in record.share_status
                            if s is ShareStatus.IN_FLIGHT)

            if in_flight >= self._MAX_IN_FLIGHT:
                return
            if delivered + in_flight >= qss.K_THRESHOLD:
                return

            held_indices = [i for i, s in enumerate(record.share_status)
                            if s is ShareStatus.HELD]
            if not held_indices:
                return

            share_index = held_indices[0]
            slot = record.share_slots[share_index]

            tid = self.app.send_single_qubit(
                data_memory_index=slot,
                dst=record.dst,
                share_index=share_index,
                packet_id=packet_id,
            )
            record.share_transfer_ids[share_index] = tid
            record.share_status[share_index] = ShareStatus.IN_FLIGHT

    def _finalize_delivered(self, record: PacketRecord) -> None:
        """Success path. Clean up Alice's held shares and signal Bob that no
        more shares are coming for this packet.

        Ordering matters: measure out held shares BEFORE sending their cancel
        messages. The measurement collapses the joint state locally on Alice's
        side; the classical cancel travels to Bob afterward. By the time Bob's
        aggregation reaches N_SHARES terminals and _reconstruct_packet runs,
        Alice's shares are in a definite state and Bob's decoder produces the
        correct secret.
        """
        self._cleanup_shares(record)
        record.outcome = PacketOutcome.DELIVERED
        log.logger.info(
            f"QTCPOverseer: packet {record.packet_id} DELIVERED "
            f"to {record.dst}"
        )

    def _finalize_lost(self, record: PacketRecord) -> None:
        """Loss path. Same cleanup as delivered -- Alice's slots freed, cancels
        sent so Bob's aggregation completes -- but the packet did not make it
        and the caller will see LOST.

        Recovery from this state (Alice reconstructing locally from any shares
        she still holds together with any Bob would return) is Phase 4.
        """
        self._cleanup_shares(record)
        record.outcome = PacketOutcome.LOST
        log.logger.warning(
            f"QTCPOverseer: packet {record.packet_id} LOST "
            f"to {record.dst}"
        )

    def _cleanup_shares(self, record: PacketRecord) -> None:
        """Shared teardown for both terminal outcomes. For each share:

          HELD: measure out the slot, mint a fake transfer id, and send CANCEL
                to Bob so his aggregation reaches N_SHARES terminals for this
                packet. Bob's _on_cancel synthesises a CANCELLED BobTransfer
                keyed on the fake id; no BobTransfer ever existed on his side.

          FAILED: the slot may still hold the qubit if the failure reason was
                NO_ENTANGLEMENT (QTCPTransfer._finish preserves the slot for
                that case, in case the layer above wants it). Free the slot;
                free_data_slot is idempotent for slots already back in the pool.

          DELIVERED: nothing. QTCPTransfer._finish freed the slot when the ACK
                landed; Bob has the share.

          IN_FLIGHT: leave alone. QTCPTransfer will terminate the share later;
                on_alice_transfer_finished's late-straggler branch will free
                the slot then.
        """
        for share_index, status in enumerate(record.share_status):
            slot = record.share_slots[share_index]

            if status is ShareStatus.HELD:
                self.app.free_data_slot(slot)

                tid = self.app.mint_transfer_id()
                record.share_transfer_ids[share_index] = tid
                self.app.node.send_message(
                    record.dst,
                    QTCPMessage(
                        QTCPMsgType.CANCEL,
                        transfer_id=tid,
                        packet_id=record.packet_id,
                        share_index=share_index,
                    ),
                )
                record.share_status[share_index] = ShareStatus.FAILED

            elif status is ShareStatus.FAILED:
                self.app.free_data_slot(slot)

    # ==================================================================
    # Bob side
    # ==================================================================

    def on_bob_transfer_finished(self, bob_transfer: BobTransfer) -> None:
        """Called by QTCPTransfer when a Bob-side transfer reaches terminal
        (ARRIVED, CANCELLED, or CONSUMED)."""
        self._check_packet_complete(bob_transfer.src, bob_transfer.packet_id)

    def get_received_packet(self, src: str, packet_id: int):
        """Return the quantum state of the reconstructed secret, or None if
        that packet has not been reconstructed."""
        data_index = self.received_packets.get((src, packet_id))
        if data_index is None:
            return None
        data_arr = self.app.node.get_component_by_name(self.app.node.data_memo_arr_name)
        key = data_arr[data_index].qstate_key
        return self.app.node.timeline.quantum_manager.get(key).state

    def _check_packet_complete(self, src: str, packet_id: int) -> None:
        """Called whenever one of Bob's records reaches a terminal state.

        A packet is settled once every share has resolved -- ARRIVED or
        CANCELLED. Bob cannot reconstruct early even when he already holds
        K: an unresolved share is still entangled with the ones he has, so
        decoding around it gives a joint state rather than the secret.

        Once settled, >= K_THRESHOLD arrivals means reconstruct; fewer means
        the packet is unrecoverable on this side.
        """
        if packet_id is None:
            # Standalone transfer, not part of a packet -- packet_id is the
            # aggregation key, a None key would collide across unrelated
            # standalone sends.
            return

        records = self.app._bob_transfers_for_packet(src, packet_id)

        if len(records) < qss.N_SHARES:
            # Not every share has been heard of yet. A share that never fired
            # produces no SEND_NOTICE; its record only appears when Alice's
            # CANCEL arrives.
            return

        unsettled = [r for r in records
                     if r.state not in (BobState.ARRIVED, BobState.CANCELLED,
                                        BobState.CONSUMED)]
        if unsettled:
            return

        if any(r.state is BobState.CONSUMED for r in records):
            return  # already reconstructed

        arrived = [r for r in records if r.state is BobState.ARRIVED]

        if len(arrived) < qss.K_THRESHOLD:
            log.logger.warning(
                f"QTCPOverseer: packet {packet_id} from {src} settled with only "
                f"{len(arrived)}/{qss.N_SHARES} shares; cannot reconstruct"
            )
            return

        log.logger.info(
            f"QTCPOverseer: packet {packet_id} from {src} settled with "
            f"{len(arrived)}/{qss.N_SHARES} shares; reconstructing"
        )
        self._reconstruct_packet(src, packet_id)

    def _reconstruct_packet(self, src: str, packet_id: int) -> None:
        """Decode a settled packet and deliver the recovered secret.

        Preconditions: every share has settled, >= K_THRESHOLD ARRIVED.

        The decoder is built for exactly N_SHARES - K_THRESHOLD erasures, so
        Bob uses exactly K_THRESHOLD shares and treats the rest as erased --
        even the ones that arrived fine. Surplus arrivals must be measured out
        *before* decoding: they are still entangled with the shares in use,
        and decoding around a live share yields a joint state rather than the
        secret.
        """
        records = self.app._bob_transfers_for_packet(src, packet_id)

        arrived = sorted(
            (r for r in records if r.state is BobState.ARRIVED),
            key=lambda r: r.share_index,
        )
        used = arrived[:qss.K_THRESHOLD]
        surplus = arrived[qss.K_THRESHOLD:]

        # 1) get rid of shares we will not use. free_data_slot measures the
        #    qubit out and resets the slot -- exactly the erasure the decoder
        #    assumes for those positions.
        for record in surplus:
            log.logger.debug(
                f"QTCPOverseer: packet {packet_id} share {record.share_index} "
                f"arrived but is surplus; measuring out"
            )
            self.app.free_data_slot(record.data_index)
            record.data_index = None

        used_indices = {r.share_index for r in used}
        erased = [i for i in range(qss.N_SHARES) if i not in used_indices]

        # 2) fresh |0> qubits stand in at the erased positions
        spares = []
        for _ in erased:
            slot = self.app.alloc_data_slot()
            if slot is None:
                log.logger.warning(
                    f"QTCPOverseer: no free data slot to reconstruct packet "
                    f"{packet_id}; discarding shares"
                )
                for s in spares:
                    self.app.free_data_slot(s)
                for r in used:
                    self.app.free_data_slot(r.data_index)
                    r.data_index = None
                for r in records:
                    r.state = BobState.CONSUMED
                return
            spares.append(slot)

        # 3) build the key list. Position in this list IS the code position --
        #    share i must sit at index i or the syndrome is meaningless.
        slots = [None] * qss.N_SHARES
        for record in used:
            slots[record.share_index] = record.data_index
        for position, spare in zip(erased, spares):
            slots[position] = spare

        data_arr = self.app.node.get_component_by_name(self.app.node.data_memo_arr_name)
        keys = [data_arr[s].qstate_key for s in slots]

        # 4) decode. The decoder measures the four ancilla positions as part of
        #    the circuit, so this returns the syndrome.
        rnd = self.app.node.get_generator().random()
        meas = self.app.node.timeline.quantum_manager.run_circuit(
            qss.DECODER, keys, rnd
        )

        # Assemble in circuit-position order. Iterating the returned dict would
        # not give position order and would silently pick the wrong table row.
        syndrome = tuple(int(meas[keys[i]]) for i in range(qss.N_SHARES - 1))

        # 5) correct the secret at code position SECRET_INDEX
        correction = qss.correction_for(tuple(sorted(erased)), syndrome)
        secret_slot = slots[qss.SECRET_INDEX]
        if correction is not None:
            rnd = self.app.node.get_generator().random()
            self.app.node.timeline.quantum_manager.run_circuit(
                correction, [data_arr[secret_slot].qstate_key], rnd
            )

        log.logger.info(
            f"QTCPOverseer: packet {packet_id} reconstructed from shares "
            f"{sorted(used_indices)} (erased {erased}, syndrome {syndrome}, "
            f"correction {correction is not None}) -> data memory {secret_slot}"
        )

        # 6) release everything except the slot holding the secret. The four
        #    ancilla positions were consumed by the decoder's measurements.
        for position, slot in enumerate(slots):
            if position != qss.SECRET_INDEX:
                self.app.free_data_slot(slot)

        for record in records:
            record.state = BobState.CONSUMED
            record.data_index = None

        self._deliver_packet(src, packet_id, secret_slot)

    def _deliver_packet(self, src: str, packet_id: int, data_index: int) -> None:
        """Hand off a reconstructed packet. Stores it for retrieval; the slot
        is not returned to the free pool -- it holds the recovered secret."""
        self.received_packets[(src, packet_id)] = data_index