"""qTCP packet overseer: the layer above QTCPTransfer that owns packets.
QTCPTransfer handles single-qubit transfers. This layer owns packets:
encoding a secret into N_SHARES shares, dispatching them through QTCPTransfer
under a bounded-parallelism rule, and driving the per-packet outcome
(DELIVERED / LOST / RECOVERED_LOCALLY) based on the terminal events
QTCPTransfer feeds back.
Delivery model:
  - Up to K-1 shares in flight at once (the parallelism cap).
  - Sequential decisions from that point: when the recovery potential
    (delivered + held) drops to K, the next share is recursed into instead
    of fired -- encoded as its own sub-packet, tagged with a parent link so
    Bob routes the reconstructed share back into the parent aggregation.
  - Recursion is depth-capped (state grows per level under the simulator's
    quantum representation).
  - If every fired share fails before any deliver, Alice still holds K
    shares -- full recovery power. Instead of K sequential recursions
    (which the simulator cannot represent) she decodes locally and restarts
    the packet with a fresh encoding under the same packet id, bounded by
    max_restarts.
  - When the restart budget is exhausted and the packet is still stuck,
    dispatch on on_exhaust: "recover" finalizes as RECOVERED_LOCALLY and
    hands the intact secret back to the caller; "fire" fires the share
    directly and accepts the risk; "raise" halts.
Contract (Alice side):
    send_packet(data_memory_index, dst) -> packet_id
    get_packet_outcome(packet_id) -> PacketOutcome | None
    get_recovered_slot(packet_id) -> int | None
Contract (Bob side):
    get_received_packet(src, packet_id) -> state | None
Terminal transitions on QTCPTransfer trigger on_alice_transfer_finished (Alice)
or on_bob_transfer_finished (Bob). Both handlers live here.
"""
from dataclasses import dataclass
from enum import Enum, auto
import sequence.app.qtcp.qec as qec
import sequence.app.qtcp.qss as qss
from sequence.app.qtcp.qtcp_transfer import (
    QTCPTransfer, Transfer, BobTransfer,
    QTCPMessage, QTCPMsgType,
    TransferStatus, BobState,
)
from sequence.utils import log
class PacketOutcome(Enum):
    IN_PROGRESS       = auto()
    DELIVERED         = auto()   # >= K_THRESHOLD shares reached Bob
    RECOVERED_LOCALLY = auto()   # packet failed to reach Bob but Alice
                                 # decoded the secret locally; see
                                 # get_recovered_slot for its data slot
    LOST              = auto()
class ShareStatus(Enum):
    HELD      = auto()   # never fired; qubit still in Alice's data slot
    IN_FLIGHT = auto()   # send_single_qubit called; no terminal yet
    RECURSING = auto()   # recursed into a sub-packet; awaiting sub-packet outcome
    DELIVERED = auto()   # Bob has it
    FAILED    = auto()   # terminal, did not reach Bob
@dataclass
class PacketRecord:
    """Alice-side per-packet state. Bob doesn't need one; his aggregation runs
    off the BobTransfer records in QTCPTransfer.
    share_slots, share_transfer_ids and share_status are each of length
    N_SHARES; index i corresponds to share_index i throughout.
    parent_packet_id / parent_share_index link this record to a parent packet
    when it is a sub-packet spawned by recursion. Both None for top-level
    packets.
    depth is the recursion depth of this packet: 0 for top-level, 1 for a
    packet spawned by recursion on a top-level share, and so on. Used to cap
    unbounded recursion (state grows exponentially in depth).
    """
    packet_id: int
    dst: str
    source_slot: int                            # the caller's original slot
    share_slots: list                           # data slot index per share
    share_transfer_ids: list                    # transfer id (or sub-packet id) per share, or None
    share_status: list                          # ShareStatus per share
    outcome: PacketOutcome = PacketOutcome.IN_PROGRESS
    parent_packet_id: int = None
    parent_share_index: int = None
    depth: int = 0
    is_qec_layer: bool = False   # True for a leaf QEC block; False for QSS nodes
    restarts: int = 0
    # Slot holding the locally-recovered secret when outcome is
    # RECOVERED_LOCALLY. None otherwise.
    recovered_slot: int = None
class QTCPOverseer:
    """Packet-lifecycle owner. Sits above QTCPTransfer on both endpoints.
    Attributes:
        app (QTCPTransfer): the transport tool below.
        max_recursion_depth (int): cap on recursion depth per packet family.
        max_restarts (int): cap on local restarts per packet.
        on_exhaust (str): policy when the restart budget runs out --
            "recover" (default), "fire", or "raise".
        packets (dict[int, PacketRecord]): Alice's packet table.
        next_packet_id (int): Alice's monotonic packet-id counter.
        received_packets (dict[(str, int), int]): Bob's delivered-packet
            table (src, packet_id) -> data slot holding the reconstructed
            secret.
        next_synth_id (int): Bob-side counter for synthesized BobTransfer
            records (sub-packet reconstructions). Counts DOWN from -1 so
            these ids cannot collide with sender-minted ids (which count up
            from 0 and key the same table).
    """
    # Cap on simultaneously in-flight shares per packet. Sending more than this
    # would let a burst of failures drop Alice's held count below K_THRESHOLD,
    # at which point neither Alice nor Bob can recover the secret. K-1 keeps
    # the invariant `alice_held + bob_arrived >= K_THRESHOLD` even in the worst
    # case where every currently-in-flight share fails.
    _MAX_IN_FLIGHT = qss.K_THRESHOLD - 1
    def __init__(self, app: QTCPTransfer, max_recursion_depth: int = 1,
                 max_restarts: int = 1, on_exhaust: str = "recover"):
        self.app = app
        app.terminal_observers.append(self)
        # Cap on recursion depth. Recursion re-encodes a share as its own
        # sub-packet under QSS -- state grows exponentially per level. At the
        # cap, _fire_share_at fires the share directly instead of recursing.
        self.max_recursion_depth = max_recursion_depth
        # Cap on local restarts per packet. When every fired share has failed
        # and none delivered (delivered == 0 in recursion territory), Alice
        # still holds K shares -- full recovery power -- so instead of K
        # sequential recursions (whose accumulated joint state is beyond what
        # the simulator can represent) she decodes locally and starts the
        # packet over with a fresh encoding. This bounds how many times.
        self.max_restarts = max_restarts
        # Policy when the restart budget is exhausted and the packet is again
        # in the delivered=0-in-recursion-territory state:
        #   "recover" (default): decode locally, finalize outcome as
        #       RECOVERED_LOCALLY, notify Bob to purge. Caller sees an
        #       intact secret in a data slot and decides what to do next.
        #   "fire": fire the share directly, accept the risk of losing the
        #       secret (if it fails and no other holds the state) in exchange
        #       for a chance at delivery. For callers willing to gamble.
        #   "raise": halt with an exception. For tests and dev environments
        #       where hitting this branch is itself a bug.
        # See _fire_share_at's exhausted branch.
        if on_exhaust not in ("recover", "fire", "raise"):
            raise ValueError(
                f"on_exhaust must be 'recover', 'fire', or 'raise'; "
                f"got {on_exhaust!r}"
            )
        self.on_exhaust = on_exhaust
        # Alice-side
        self.packets: dict[int, PacketRecord] = {}
        self.next_packet_id: int = 0
        # Bob-side
        self.received_packets: dict[tuple[str, int], int] = {}
        # Ids for Bob-side synthesized BobTransfer records (a reconstructed
        # sub-packet becoming an ARRIVED share of its parent). These must not
        # collide with transfer ids minted by the SENDER, which key the same
        # (src, tid) table and count up from 0 -- Bob's own mint counter also
        # starts at 0 and would overwrite the sender's records. Negative ids
        # cannot collide and are self-documenting in logs.
        self.next_synth_id: int = -1
        log.logger.debug(
            f"QTCPOverseer for {app.name}: initialized "
            f"(max_recursion_depth={max_recursion_depth}, "
            f"max_restarts={max_restarts}, on_exhaust={on_exhaust!r})"
        )
    # ==================================================================
    # Alice side
    # ==================================================================
    # ------------------------------------------------------------------
    # QSS helpers used by both send/restart (encode) and Bob's reconstruct
    # + Alice's restart/recover (decode). Kept here so the QSS layout
    # invariants (SECRET_INDEX position, N-1 ancillas, position-order
    # syndrome assembly) live in one place.
    # ------------------------------------------------------------------
    def _encode_at(self, secret_slot: int) -> list:
        """Allocate N_SHARES-1 fresh ancilla slots and QSS-encode the qubit
        currently in `secret_slot` in place across all N_SHARES slots.
        Returns the position-ordered slot list (slots[i] is the data slot
        holding share i). slots[SECRET_INDEX] == secret_slot on entry and
        on exit; only the state at each slot changes.
        Alloc failures halt the process. Under the intended memory config
        the pool is always sufficient; hitting a None from alloc_data_slot
        means the config is broken, and silent recovery would just hide the
        misconfig.
        """
        ancillas = []
        for _ in range(qss.N_SHARES - 1):
            slot = self.app.alloc_data_slot()
            assert slot is not None, (
                f"QTCPOverseer: pool exhausted in _encode_at "
                f"({len(ancillas)}/{qss.N_SHARES - 1} ancillas allocated) -- "
                f"data memory config broken"
            )
            ancillas.append(slot)
        # Position in this list IS the code position: slots[i] becomes share i.
        # The secret must sit at SECRET_INDEX, where the encoder expects it
        # and where the decoder returns it.
        slots = list(ancillas)
        slots.insert(qss.SECRET_INDEX, secret_slot)
        data_arr = self.app.node.get_component_by_name(
            self.app.node.data_memo_arr_name)
        keys = [data_arr[s].qstate_key for s in slots]
        rnd = self.app.node.get_generator().random()
        self.app.node.timeline.quantum_manager.run_circuit(
            qss.ENCODER, keys, rnd)
        return slots
    def _decode_at(self, used_positions: list, used_slots: list,
                   erased_positions: list, desc: str) -> int:
        """Run the QSS decoder given a position -> slot layout.
        Allocs fresh |0> spares for the erased positions, builds the
        position-ordered slot list, runs DECODER, extracts the syndrome in
        circuit-position order, applies the appropriate correction to the
        SECRET_INDEX slot, then frees every slot except the one holding the
        recovered secret. Returns that slot.
        Args:
            used_positions: code positions where the caller has live shares.
                len == K_THRESHOLD.
            used_slots: data slot indices holding those shares, in the same
                order as used_positions.
            erased_positions: code positions to treat as erasures. len ==
                N_SHARES - K_THRESHOLD.
            desc: log-line description of who called (packet/context).
                The rest of the log format is uniform across callers.
        NOTE: any live entanglement on erased positions must be measured out
        by the caller BEFORE this call. Alice's local decode has to handle
        NO_ENTANGLEMENT-preserved qubits explicitly; Bob's reconstruct
        handles surplus arrivals explicitly. This helper assumes clean
        erased positions.
        """
        
        assert len(used_positions) == qss.K_THRESHOLD
        assert len(used_slots) == qss.K_THRESHOLD
        assert len(erased_positions) == qss.N_SHARES - qss.K_THRESHOLD
        # 1) fresh |0> spares stand in at the erased positions
        spares = []
        for _ in erased_positions:
            slot = self.app.alloc_data_slot()
            assert slot is not None, (
                f"QTCPOverseer: pool exhausted in _decode_at "
                f"({len(spares)}/{len(erased_positions)} spares allocated) "
                f"-- data memory config broken"
            )
            spares.append(slot)
        # 2) build the position-ordered slot list; share i sits at index i
        slots = [None] * qss.N_SHARES
        for position, slot in zip(used_positions, used_slots):
            slots[position] = slot
        for position, spare in zip(erased_positions, spares):
            slots[position] = spare
        data_arr = self.app.node.get_component_by_name(
            self.app.node.data_memo_arr_name)
        keys = [data_arr[s].qstate_key for s in slots]
        # 3) decode; the DECODER circuit measures the N-1 non-SECRET_INDEX
        # positions and produces the syndrome
        rnd = self.app.node.get_generator().random()
        meas = self.app.node.timeline.quantum_manager.run_circuit(
            qss.DECODER, keys, rnd)
        # Assemble in circuit-position order. Iterating the returned dict
        # would not give position order and would silently pick the wrong
        # table row.
        syndrome = tuple(int(meas[keys[i]]) for i in range(qss.N_SHARES - 1))
        # 4) apply correction at SECRET_INDEX
        erased_tuple = tuple(sorted(erased_positions))
        correction = qss.correction_for(erased_tuple, syndrome)
        secret_slot = slots[qss.SECRET_INDEX]
        if correction is not None:
            rnd = self.app.node.get_generator().random()
            self.app.node.timeline.quantum_manager.run_circuit(
                correction, [data_arr[secret_slot].qstate_key], rnd)
        log.logger.info(
            f"QTCPOverseer: {desc} from positions {sorted(used_positions)} "
            f"(erased {list(erased_tuple)}, syndrome {syndrome}, "
            f"correction {correction is not None}) -> data memory {secret_slot}"
        )
        # 5) release everything except the slot holding the secret. The
        # ancilla positions were consumed by the decoder's measurements.
        for position, slot in enumerate(slots):
            if position != qss.SECRET_INDEX:
                self.app.free_data_slot(slot)
        return secret_slot
    def _decode_corrected(self, share_positions: list, share_slots: list,
                          desc: str) -> int:
        """Correction-mode decode of a full [[5,1,3]] codeword.
        Unlike _decode_at (erasure mode: reconstruct from K survivors, treat
        the rest as erased), this needs ALL N_SHARES shares present and finds
        + fixes a single unknown Pauli error via stabiliser syndrome
        extraction. Flow, distinct from erasure decode:
            1. Order the 5 shares by CODE POSITION (not arrival order) and
               allocate ONE ancilla slot in |0>.
            2. For each of the 4 stabiliser generators: run its measurement
               circuit on [5 shares in code order, ancilla]; read the ancilla
               bit; reset the ancilla to |0> (X iff it read 1). This yields
               the 4-bit syndrome. The single ancilla is reused across all
               four -- peak width here is 5 shares + 1 ancilla = 6.
            3. Look up syndrome -> (position, Pauli). If not clean, apply that
               Pauli to the flagged share's slot.
            4. Run the QSS DECODER (U-dagger) to extract the secret at
               SECRET_INDEX, exactly as erasure decode does.
            5. Free the ancilla and every slot except the secret's.
        No fresh |0> spares are allocated (there are no erasures). No erasure
        table is consulted -- the correction table is keyed on syndrome alone.
        Args:
            share_positions: the 5 code positions (should be 0..4). len == N.
            share_slots: data slot per position, same order as share_positions.
            desc: log-line description of the caller.
        Returns the data slot holding the recovered secret.
        """
        assert len(share_positions) == qss.N_SHARES, (
            f"correction mode needs all {qss.N_SHARES} shares; got "
            f"{len(share_positions)} positions"
        )
        assert len(share_slots) == qss.N_SHARES
        assert sorted(share_positions) == list(range(qss.N_SHARES)), (
            f"correction mode expects positions 0..{qss.N_SHARES - 1}, got "
            f"{sorted(share_positions)}"
        )
        qm = self.app.node.timeline.quantum_manager
        data_arr = self.app.node.get_component_by_name(
            self.app.node.data_memo_arr_name)
        # 1) code-position-ordered slot list: slots[i] holds the share for
        # code position i. Feeding the syndrome circuits in arrival order
        # instead would compute the syndrome over the wrong qubits and
        # silently miscorrect.
        slots = [None] * qss.N_SHARES
        for position, slot in zip(share_positions, share_slots):
            slots[position] = slot
        # one reused ancilla, allocated from the data pool (fresh slots are |0>)
        ancilla_slot = self.app.alloc_data_slot()
        assert ancilla_slot is not None, (
            f"QTCPOverseer: pool exhausted allocating syndrome ancilla in "
            f"_decode_corrected -- data memory config broken"
        )
        share_keys = [data_arr[s].qstate_key for s in slots]
        ancilla_key = data_arr[ancilla_slot].qstate_key
        # 2) extract the 4-bit syndrome, one generator at a time, reusing the
        # single ancilla. The circuit acts on [share0..share4, ancilla]; its
        # measurement targets the ancilla (index N_SHARES in the circuit).
        syndrome_bits = []
        for gen_index, circuit in enumerate(qec.STABILIZER_CIRCUITS):
            circuit_keys = share_keys + [ancilla_key]
            rnd = self.app.node.get_generator().random()
            meas = qm.run_circuit(circuit, circuit_keys, rnd)
            bit = int(meas[ancilla_key])
            syndrome_bits.append(bit)
            # reset ancilla to |0> for the next generator (it holds |bit> now)
            if bit == 1:
                rnd = self.app.node.get_generator().random()
                qm.run_circuit(qec.ANCILLA_RESET, [ancilla_key], rnd)
        syndrome = tuple(syndrome_bits)
        # 3) look up and apply the correction to the flagged CODE qubit
        # (before decoding -- correction mode fixes the codeword, then decodes)
        correction = qec.correction_for(syndrome)
        if correction is not None:
            corr_position, corr_circuit = correction
            corr_slot = slots[corr_position]
            rnd = self.app.node.get_generator().random()
            qm.run_circuit(
                corr_circuit, [data_arr[corr_slot].qstate_key], rnd)
        # ancilla is spent (holds |0> after the last reset, or |0>/|1> from the
        # final measurement); free it back to the pool. free_data_slot measures
        # it out and resets to |0>.
        self.app.free_data_slot(ancilla_slot)
        # 4) decode the corrected codeword with the QSS decoder. The DECODER
        # measures the N-1 non-secret positions; we ignore its syndrome (the
        # correction already happened above) and keep the secret at
        # SECRET_INDEX.
        rnd = self.app.node.get_generator().random()
        qm.run_circuit(qss.DECODER, share_keys, rnd)
        secret_slot = slots[qss.SECRET_INDEX]
        log.logger.info(
            f"QTCPOverseer: {desc} corrected-mode decode "
            f"(syndrome {syndrome}, "
            f"correction {correction is not None}"
            + (f" at position {correction[0]}" if correction is not None else "")
            + f") -> data memory {secret_slot}"
        )
        # 5) free every slot except the secret's
        for position, slot in enumerate(slots):
            if position != qss.SECRET_INDEX:
                self.app.free_data_slot(slot)
        return secret_slot
    def _send_qec_leaf(self, data_memory_index: int, dst: str,
                       parent: tuple = None, packet_id: int = None) -> int:
        """Deliver a single qubit as a QEC leaf block: encode into N_SHARES
        [[5,1,3]] shares, deliver each directly (sequential, fail-stop), Bob
        correction-decodes. This is the innermost layer (Model A).
        parent: (parent_packet_id, parent_share_index) when this leaf is a
        share of a QSS node above it; None when it IS the whole packet
        (max_recursion_depth == 0, teleportation).
        Marked is_qec_layer=True so Alice routing (on_alice_transfer_finished)
        and Bob (_reconstruct_packet) select correction mode.
        """
        slots = self._encode_at(data_memory_index)
        if packet_id is None:
            packet_id = self.next_packet_id
            self.next_packet_id += 1
        parent_pid = parent[0] if parent is not None else None
        parent_share = parent[1] if parent is not None else None
        depth = (self.packets[parent_pid].depth + 1) if parent is not None else 0
        record = PacketRecord(
            packet_id=packet_id,
            dst=dst,
            source_slot=data_memory_index,
            share_slots=slots,
            share_transfer_ids=[None] * qss.N_SHARES,
            share_status=[ShareStatus.HELD] * qss.N_SHARES,
            parent_packet_id=parent_pid,
            parent_share_index=parent_share,
            depth=depth,
            is_qec_layer=True,
        )
        self.packets[packet_id] = record
        parent_desc = (f" [QEC leaf of packet {parent_pid} share {parent_share}, "
                       f"depth {depth}]") if parent is not None else " [QEC-only root]"
        log.logger.info(
            f"QTCPOverseer: QEC leaf {packet_id} encoded from slot "
            f"{data_memory_index} into {qss.N_SHARES} shares -> {dst} "
            f"(slots {slots}){parent_desc}"
        )
        self._fire_next_qec_share(record)
        return packet_id
    
    def send_share(self, data_memory_index: int, dst: str,
                    parent: tuple = None,  packet_id: int = None) -> int:
        """Encode a single-qubit secret into N_SHARES shares and start
        delivery. Returns the packet id.
        Encoding is in place across N_SHARES data slots. The caller's slot is
        one of them (at code position SECRET_INDEX); the rest are allocated.
        After encoding, all N_SHARES slots hold shares; the "original" no
        longer exists as a separate qubit -- it IS one of the shares.
        Delivery is bounded-parallel: up to _MAX_IN_FLIGHT shares are fired
        immediately. Additional shares fire from on_alice_transfer_finished
        as earlier ones terminate. If firing a share would drop Alice's
        recovery potential below K_THRESHOLD, the share is recursed into
        (encoded again as a sub-packet) instead.
        parent: (parent_packet_id, parent_share_index) when this call is a
        recursion spawned by _fire_share_at; None for top-level packets.
        The parent link travels with sub-share messages so Bob can route the
        reconstructed sub-packet back into the parent packet's aggregation.
        """
        slots = self._encode_at(data_memory_index)
        if packet_id is None:
            packet_id = self.next_packet_id
            self.next_packet_id += 1
        parent_pid = parent[0] if parent is not None else None
        parent_share = parent[1] if parent is not None else None
        depth = (self.packets[parent_pid].depth + 1) if parent is not None else 0
        record = PacketRecord(
            packet_id=packet_id,
            dst=dst,
            source_slot=data_memory_index,
            share_slots=slots,
            share_transfer_ids=[None] * qss.N_SHARES,
            share_status=[ShareStatus.HELD] * qss.N_SHARES,
            parent_packet_id=parent_pid,
            parent_share_index=parent_share,
            depth=depth,
        )
        self.packets[packet_id] = record
        parent_desc = ""
        if parent is not None:
            parent_desc = (
                f" [sub-packet of packet {parent_pid} share {parent_share}, "
                f"depth {depth}]"
            )
        log.logger.info(
            f"QTCPOverseer: packet {packet_id} encoded from slot "
            f"{data_memory_index} into {qss.N_SHARES} shares -> {dst} "
            f"(slots {slots}){parent_desc}"
        )
    
        self._fire_shares(packet_id)
        return packet_id
    def send_packet(self, data_memory_index: int, dst: str,
                    packet_id: int = None) -> int:
        """Top-level send. QSS-outer / QEC-at-leaf (Model A).
        max_recursion_depth counts QSS layers above the leaf:
          0 -> no QSS; the packet is a single QEC leaf block (teleportation).
          1 -> one QSS layer, QEC at each of its 5 leaves.
          n -> n QSS layers, QEC at the deepest leaves.
        The QSS tree is driven by send_share/_fire_shares/_fire_share_at
        (unchanged). The only inversion is at the leaf: _fire_share_at's
        depth-cap branch now spawns a QEC leaf block (_send_qec_leaf) instead
        of firing a bare qubit. This record is the QSS root: parent-None,
        is_qec_layer False.
        """
        if packet_id is None:
            packet_id = self.next_packet_id
            self.next_packet_id += 1
        if self.max_recursion_depth == 0:
            # No QSS layer at all: the whole packet is a single QEC leaf block.
            # Spawn it as a parent-None QEC block that delivers as the top.
            return self._send_qec_leaf(
                data_memory_index=data_memory_index,
                dst=dst,
                parent=None,
                packet_id=packet_id,
            )
        # QSS-outer: encode into the QSS tree and drive it. Identical to the
        # old send_share body, but this is the parent-None root.
        slots = self._encode_at(data_memory_index)
        record = PacketRecord(
            packet_id=packet_id,
            dst=dst,
            source_slot=data_memory_index,
            share_slots=slots,
            share_transfer_ids=[None] * qss.N_SHARES,
            share_status=[ShareStatus.HELD] * qss.N_SHARES,
            parent_packet_id=None,
            parent_share_index=None,
            depth=0,
            is_qec_layer=False,
        )
        self.packets[packet_id] = record
        log.logger.info(
            f"QTCPOverseer: QSS packet {packet_id} encoded from slot "
            f"{data_memory_index} into {qss.N_SHARES} shares -> {dst} "
            f"(slots {slots}) [outermost QSS layer]"
        )
        self._fire_shares(packet_id)
        return packet_id
 
    def _fire_next_qec_share(self, record: PacketRecord) -> None:
        """Fire the next still-HELD physical share of a QEC leaf block, one at
        a time.
        Sequential by construction: only ever one share IN_FLIGHT. We start the
        next share only when none is currently IN_FLIGHT (or RECURSING, which a
        leaf block never has).
        The IN_PROGRESS guard here is load-bearing, not redundant with the
        caller: this method is re-entered from _advance_qec_packet after each
        share resolves, and the leaf may have gone LOST (fail-stop) on a failed
        share in between.
        """
        if record.outcome is not PacketOutcome.IN_PROGRESS:
            return
 
        # if a share is still live (in flight), wait for it to resolve
        if any(s in (ShareStatus.IN_FLIGHT, ShareStatus.RECURSING)
               for s in record.share_status):
            return
        held = [i for i, s in enumerate(record.share_status)
                if s is ShareStatus.HELD]
        if not held:
            return
        share_index = held[0]
        slot = record.share_slots[share_index]
        # Leaf QEC shares are the physical qubits of the block: fire each
        # directly. Its parent link is THIS QEC block's own parentage (so a
        # sub-packet reconstruction routes up to the QSS share above), carried
        # on the transfer so Bob aggregates under this leaf's packet_id.
        tid = self.app.send_single_qubit(
            data_memory_index=slot,
            dst=record.dst,
            share_index=share_index,
            packet_id=record.packet_id,
            parent_packet_id=record.parent_packet_id,
            parent_share_index=record.parent_share_index,
            is_qec_layer=True,
        )
        record.share_transfer_ids[share_index] = tid
        record.share_status[share_index] = ShareStatus.IN_FLIGHT
        log.logger.info(
            f"QTCPOverseer: QEC leaf {record.packet_id} firing share "
            f"{share_index} directly (slot {slot})"
        )
 
    def _advance_qec_packet(self, packet_id: int) -> None:
        """Drive a QEC leaf block forward after one of its physical shares
        resolves (from on_alice_transfer_finished).
        Fail-stop: any FAILED share => the whole leaf is LOST (correction needs
        all N_SHARES). Otherwise, once all N_SHARES are DELIVERED, Alice's side
        of the leaf is done -- finalize (which cascades to the parent QSS share)
        and let Bob correction-decode his aggregation independently. Until then,
        fire the next share.
        The IN_PROGRESS guard here is load-bearing: this is the entry point
        from a terminal event, and the leaf may already have been finalized by
        a window-close teardown before this fires.
        """
        record = self.packets[packet_id]
        if record.outcome is not PacketOutcome.IN_PROGRESS:
            return
 
        # Any failed share is fatal in correction mode.
        if any(s is ShareStatus.FAILED for s in record.share_status):
            log.logger.warning(
                f"QTCPOverseer: QEC packet {packet_id} has a failed share; "
                f"correction mode needs all {qss.N_SHARES} -> LOST (fail-stop)"
            )
            self._finalize_lost(record)
            return
 
        delivered = sum(1 for s in record.share_status
                        if s is ShareStatus.DELIVERED)
        if delivered == qss.N_SHARES:
            # All physical shares are on Bob's side. Alice's side of this leaf
            # is done; finalize (which cascades to the parent QSS share via
            # _finalize_delivered, un-sticking the next leaf). Bob correction-
            # decodes his aggregation independently and feeds the recovered
            # qubit up his own aggregation.
            log.logger.info(
                f"QTCPOverseer: QEC leaf {packet_id} all {qss.N_SHARES} shares "
                f"sent; finalizing (Bob correction-decodes on his side)"
            )
            self._finalize_delivered(record)
            return
 
        # Not done, none failed -> start the next share.
        self._fire_next_qec_share(record)
    def get_packet_outcome(self, packet_id: int) -> PacketOutcome | None:
        """Poll for terminal outcome. None if the packet id is unknown."""
        record = self.packets.get(packet_id)
        return record.outcome if record else None
    def get_recovered_slot(self, packet_id: int) -> int | None:
        """For a packet whose outcome is RECOVERED_LOCALLY, return the data
        memory slot holding the recovered secret. Returns None for unknown
        packet ids and for packets in any other outcome (the slot is only
        meaningful in the RECOVERED_LOCALLY state)."""
        record = self.packets.get(packet_id)
        if record is None:
            return None
        if record.outcome is not PacketOutcome.RECOVERED_LOCALLY:
            return None
        return record.recovered_slot
    def on_alice_transfer_finished(self, transfer: Transfer) -> None:
        """Called by QTCPTransfer._finish when an Alice-side transfer reaches
        terminal. Updates the share's status and advances the packet -- but
        only once the initial parallel-fired batch has fully drained.
        Under the sequential-after-parallel-pair model, both shares fired at
        packet start need to terminate before the next decision is made. This
        keeps recursion decisions honest (based on complete information about
        the initial batch) and prevents a new share from being fired or
        recursed into while another share is still in flight and possibly
        entangled with what would be encoded.
        Routes to the QEC-leaf driver (_advance_qec_packet: fail-stop, all-N)
        or the QSS driver (_advance_packet: erasure, K-of-N) by the record's
        is_qec_layer flag.
        """
        if transfer.packet_id is None:
            return  # standalone send, not part of a packet we track
        record = self.packets.get(transfer.packet_id)
        if record is None:
            return
        share_index = transfer.share_index
        if record.outcome is not PacketOutcome.IN_PROGRESS:
            # Packet already decided; late straggler from a share that was
            # still IN_FLIGHT when the decision was made. free_data_slot is
            # idempotent -- for shares whose _finish already freed the slot
            # (DELIVERED, NO_ACK) it no-ops; for NO_ENTANGLEMENT (which
            # _finish preserves) it cleans up the leak.
            self.app.free_data_slot(record.share_slots[share_index])
            return
        if transfer.status is TransferStatus.DELIVERED:
            record.share_status[share_index] = ShareStatus.DELIVERED
        else:
            record.share_status[share_index] = ShareStatus.FAILED
        # Wait for the in-flight batch to fully drain before advancing.
        in_flight = sum(1 for s in record.share_status
                        if s is ShareStatus.IN_FLIGHT)
        if in_flight > 0:
            return
        log.logger.info(
                    f"transfer {transfer.transfer_id} share: {transfer.share_index} of packet:{transfer.packet_id}"
                )
        if record.is_qec_layer:
            self._advance_qec_packet(transfer.packet_id)
        else:
            self._advance_packet(transfer.packet_id)
    def _advance_packet(self, packet_id: int) -> None:
        """Decide the next action for a QSS packet based on current share
        states.
        Rules, checked in order:
          1. If Bob already has >= K arrivals: success. Clean up Alice's side
             (measure out held shares, cancel un-sent shares).
          2. If Bob can never reach K given what's still held / in flight /
             recursing: lost. Same cleanup path as success, different outcome.
          3. Otherwise: try to fire (or recurse into) more held shares up to
             the parallelism cap.
        The IN_PROGRESS guard here is load-bearing: this is an entry point from
        a terminal event and from the parent cascade, either of which may have
        already finalized this packet.
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
        recursing = sum(1 for s in record.share_status
                        if s is ShareStatus.RECURSING)
        if delivered >= qss.K_THRESHOLD:
            self._finalize_delivered(record)
            return
        # Optimistic upper bound: if every uncertain share (held, in_flight,
        # or recursing) resolved as DELIVERED, could we still reach K? If not,
        # the packet is unrecoverable.
        if delivered + held + in_flight + recursing < qss.K_THRESHOLD:
            self._finalize_lost(record)
            return
        self._fire_shares(packet_id)
    def _fire_shares(self, packet_id: int) -> None:
        """Fire (or recurse into) held shares up to the parallelism cap and
        the goal cap, in share_index order.
        Two limits govern how many shares are committed at once:
          Safety cap (_MAX_IN_FLIGHT = K-1): more than this in flight could
          let a burst of failures drop Alice's exposure below K_THRESHOLD.
          Goal cap: delivered + in_flight should not exceed K_THRESHOLD.
          Beyond that, an additional share is one that -- if it delivered --
          would just be surplus Bob measures out, and if it failed would have
          exposed a share we did not need to expose.
        For each share picked, _fire_share_at decides whether it can be fired
        directly, must be recursed into, or should wait for a still-live
        commitment (in-flight transfer or unresolved sub-packet) to settle.
        When it decides to wait, this loop exits -- another terminal event
        will re-drive the decision.
        """
        record = self.packets[packet_id]
        while True:
            # Re-checked each iteration (NOT redundant with the entry guard):
            # a _fire_share_at that restarts or recovers locally sets the
            # outcome terminal mid-loop, and we must stop firing after that.
            if record.outcome is not PacketOutcome.IN_PROGRESS:
                return
            delivered = sum(1 for s in record.share_status
                            if s is ShareStatus.DELIVERED)
            in_flight = sum(1 for s in record.share_status
                            if s is ShareStatus.IN_FLIGHT)
            # Sequential-leaf RAM constraint: never commit a new share while
            # any share is still live. Each committed share is a heavy QEC leaf
            # block; two live at once doubles the peak entangled width. This
            # guard also GUARANTEES _fire_share_at is only ever entered with
            # nothing IN_FLIGHT or RECURSING -- an invariant _fire_share_at
            # relies on (it no longer re-checks for live commitments itself).
            if any(s in (ShareStatus.IN_FLIGHT, ShareStatus.RECURSING)
                   for s in record.share_status):
                return
            if in_flight >= self._MAX_IN_FLIGHT:
                return
            if delivered + in_flight >= qss.K_THRESHOLD:
                return
            held_indices = [i for i, s in enumerate(record.share_status)
                            if s is ShareStatus.HELD]
            if not held_indices:
                return
            if not self._fire_share_at(record, held_indices[0]):
                return  # wait for something else to settle
    def _fire_share_at(self, record: PacketRecord, share_index: int) -> bool:
        """Send share `share_index` of `record` -- either as a QEC leaf block
        or, if losing it would drop the recovery potential below K_THRESHOLD,
        by recursing into a QSS sub-packet.
        Returns True if the share was fired or recursed into. Returns False if
        the decision was to wait (share left in HELD state); the caller
        should stop iterating and rely on the next terminal event to re-drive.
        Recursion rule: fire directly iff `delivered + held > K_THRESHOLD`.
        When delivered + held == K_THRESHOLD, this share is one of exactly K
        shares Alice can still contribute; losing it in transit would leave
        Alice with K-1 recovery potential, unrecoverable if the remaining
        in-flight shares also fail. Encoding this share into a sub-packet
        instead spreads the risk: a majority of sub-shares must be lost to
        lose the parent share.
        INVARIANT (relied on below): the sole caller _fire_shares only enters
        this method when NOTHING is IN_FLIGHT or RECURSING (its own guard
        returns otherwise). So there is no "wait for a live commitment to
        settle" case to re-check here -- that wait is expressed by
        _fire_shares' guard.
        Guards:
          - At max_recursion_depth, deliver as a QEC leaf instead of recursing.
            State grows exponentially per level; unbounded recursion is not
            simulable and the paper's protection is asymptotic anyway.
        """
        delivered = sum(1 for s in record.share_status
                        if s is ShareStatus.DELIVERED)
        held = sum(1 for s in record.share_status
                   if s is ShareStatus.HELD)
        slot = record.share_slots[share_index]
        if delivered + held > qss.K_THRESHOLD:
            # safe to lose this share directly; deliver as a QEC leaf block.
            self._fire_leaf(record, share_index, slot)
            return True
        # delivered + held <= K_THRESHOLD: recursion territory.
        # delivered == 0 in recursion territory: every fired share died and
        # Bob holds nothing. Nothing of value has left Alice -- she still
        # holds K shares, full recovery power. Recursing from here would mean
        # K sequential recursions (each sub-packet success only lifts
        # delivered by 1, keeping d+h == K), whose accumulated joint state is
        # beyond what the simulator can represent. Decode locally and start
        # the packet over with a fresh encoding instead; on exhaustion,
        # dispatch to on_exhaust.
        if delivered == 0:
            if record.restarts < self.max_restarts:
                self._restart_packet_locally(record)
                return True
            log.logger.info(
                f"QTCPOverseer: packet {record.packet_id} share {share_index} "
                f"-- delivered=0 in recursion territory, restart budget "
                f"exhausted ({record.restarts}/{self.max_restarts}); "
                f"on_exhaust={self.on_exhaust}"
            )
            if self.on_exhaust == "recover":
                # Decode locally into Alice's hand, finalize as
                # RECOVERED_LOCALLY, notify Bob to purge. The caller sees the
                # intact secret via get_recovered_slot and decides what to do.
                held_positions = [i for i, s in enumerate(record.share_status)
                                  if s is ShareStatus.HELD]
                failed_positions = [i for i, s in enumerate(record.share_status)
                                    if s is ShareStatus.FAILED]
                # Preserved failed-share qubits must be measured out before
                # decoding: live entanglement on erased positions corrupts
                # the local decode.
                for i in failed_positions:
                    self.app.free_data_slot(record.share_slots[i])
                held_slots = [record.share_slots[i] for i in held_positions]
                secret_slot = self._decode_at(
                    held_positions, held_slots, failed_positions,
                    desc=(f"packet {record.packet_id} recovered locally at "
                          f"exhaustion"))
                self._finalize_recovered(record, secret_slot)
                return True
            if self.on_exhaust == "fire":
                log.logger.warning(
                    f"QTCPOverseer: packet {record.packet_id} share "
                    f"{share_index} -- firing directly under on_exhaust=fire "
                    f"(may lose the secret if this or a following share fails)"
                )
                self._fire_leaf(record, share_index, slot)
                return True
            # on_exhaust == "raise"
            raise RuntimeError(
                f"QTCPOverseer: packet {record.packet_id} recovery exhausted "
                f"(on_exhaust=raise)"
            )
        # Depth cap reached: recursion would exceed the memory budget. Deliver
        # as a QEC leaf instead; if it fails, the packet just loses at this
        # level and cascades upward.
        if record.depth >= self.max_recursion_depth - 1:
            log.logger.info(
                f"QTCPOverseer: packet {record.packet_id} share {share_index} "
                f"at max QSS depth {record.depth}; delivering as QEC leaf block "
                f"(delivered={delivered}, held={held})"
            )
            self._fire_leaf(record, share_index, slot)
            return True
        # Recurse: encode this share as the secret of a QSS sub-packet;
        # sub-shares are its own new packet, tagged with a parent link so Bob
        # can route the reconstruction back into the parent packet's aggregation.
        log.logger.info(
            f"QTCPOverseer: packet {record.packet_id} share {share_index} "
            f"recursing at depth {record.depth} "
            f"(delivered={delivered}, held={held})"
        )
        sub_packet_id = self.send_share(
            data_memory_index=slot,
            dst=record.dst,
            parent=(record.packet_id, share_index),
        )
        record.share_transfer_ids[share_index] = sub_packet_id
        record.share_status[share_index] = ShareStatus.RECURSING
        return True
    def _fire_leaf(self, record: PacketRecord, share_index: int,
                   slot: int) -> None:
        """Deliver share `share_index` of `record` as a QEC leaf block.
        In Model A (QSS-outer / QEC-at-leaf) every leaf of the QSS tree is a
        QEC block, not a bare qubit. This spawns that block as a sub-packet
        parented to this QSS share, so its DELIVERED/LOST cascades back up
        through _finalize_delivered / _finalize_lost -> _advance_packet: a
        QEC-block loss becomes an erasure the QSS layer above recovers. The
        share is left RECURSING (the QSS layer treats a QEC leaf the same as a
        recursion sub-packet: it waits for the cascade).
        Shared by three call sites -- the "safe" branch and the depth-cap
        branch of _fire_share_at, and the recovery cascade in
        _finalize_recovered -- all of which turn one held share into a
        delivered QEC leaf.
        """
        sub_packet_id = self._send_qec_leaf(
            data_memory_index=slot,
            dst=record.dst,
            parent=(record.packet_id, share_index),
        )
        record.share_transfer_ids[share_index] = sub_packet_id
        record.share_status[share_index] = ShareStatus.RECURSING
    def _restart_packet_locally(self, record: PacketRecord) -> None:
        """Decode the packet from Alice's held shares and start it over with
        a fresh encoding, keeping the same packet id.
        Precondition: delivered == 0 and held == K_THRESHOLD (checked by the
        caller). Every fired share died; Bob holds nothing for this packet.
        Alice's K held shares are full recovery power.
        Mirrors Bob's _reconstruct_packet via _decode_at: the failed positions
        are erasures. Failed shares whose slots were preserved (NO_ENTANGLEMENT
        never fires the Bell measurement) still hold live qubits entangled
        with the held ones -- measured out first, which is mandatory before
        decoding (a live entangled share corrupts the local decode).
        After recovery: CANCELs go out for all N_SHARES old share identities
        so any records Bob may hold settle and purge (his _check_packet_
        complete purges a packet that settles with zero arrivals), then the
        recovered secret is re-encoded in place and delivery starts over.
        The packet id is unchanged -- the caller's handle stays valid, and
        for a sub-packet the parent's share_transfer_ids entry stays valid.
        FIFO classical ordering guarantees Bob's purge completes before any
        new SEND_NOTICE for this packet id arrives.
        """
        held_positions = [i for i, s in enumerate(record.share_status)
                          if s is ShareStatus.HELD]
        failed_positions = [i for i, s in enumerate(record.share_status)
                            if s is ShareStatus.FAILED]
        log.logger.info(
            f"QTCPOverseer: packet {record.packet_id} restarting locally "
            f"(restart {record.restarts + 1}/{self.max_restarts}, "
            f"held {held_positions}, failed {failed_positions})"
        )
        # Measure out preserved failed-share qubits before decoding.
        for i in failed_positions:
            self.app.free_data_slot(record.share_slots[i])
        # Decode via helper. Non-secret slots are freed inside.
        held_slots = [record.share_slots[i] for i in held_positions]
        secret_slot = self._decode_at(
            held_positions, held_slots, failed_positions,
            desc=f"packet {record.packet_id} recovered locally")
        # Settle Bob's side of the old attempt. Under a faithful
        # NO_ENTANGLEMENT history Bob has no records for this packet; the
        # CANCELs synthesize CANCELLED records, his aggregation settles
        # with zero arrivals and purges. Under histories where Bob holds
        # something, the ARRIVED branch of _on_cancel frees his slots.
        for share_index in range(qss.N_SHARES):
            tid = self.app.mint_transfer_id()
            self.app.node.send_message(
                record.dst,
                QTCPMessage(
                    QTCPMsgType.CANCEL,
                    transfer_id=tid,
                    packet_id=record.packet_id,
                    share_index=share_index,
                    parent_packet_id=record.parent_packet_id,
                    parent_share_index=record.parent_share_index,
                ),
            )
        # Re-encode the recovered secret in place, same packet id.
        new_slots = self._encode_at(secret_slot)
        record.share_slots = new_slots
        record.share_transfer_ids = [None] * qss.N_SHARES
        record.share_status = [ShareStatus.HELD] * qss.N_SHARES
        record.restarts += 1
        log.logger.info(
            f"QTCPOverseer: packet {record.packet_id} re-encoded "
            f"(slots {new_slots}); restarting delivery"
        )
    def _finalize_delivered(self, record: PacketRecord) -> None:
        """Success path. Clean up Alice's held shares and signal Bob that no
        more shares are coming for this packet.
        Ordering matters: measure out held shares BEFORE sending their cancel
        messages. The measurement collapses the joint state locally on Alice's
        side; the classical cancel travels to Bob afterward. By the time Bob's
        aggregation reaches N_SHARES terminals and _reconstruct_packet runs,
        Alice's shares are in a definite state and Bob's decoder produces the
        correct secret.
        If this is a sub-packet, cascade upward: the parent share this
        sub-packet was recursing on now counts as DELIVERED, and the parent
        needs to advance.
        """
        self._cleanup_shares(record)
        record.outcome = PacketOutcome.DELIVERED
        log.logger.info(
            f"QTCPOverseer: packet {record.packet_id} DELIVERED "
            f"to {record.dst}"
        )
        if record.parent_packet_id is not None:
            parent = self.packets[record.parent_packet_id]
            parent.share_status[record.parent_share_index] = ShareStatus.DELIVERED
            log.logger.info(
                f"QTCPOverseer: sub-packet {record.packet_id} DELIVERED "
                f"-> parent packet {parent.packet_id} share "
                f"{record.parent_share_index} counts as DELIVERED"
            )
            self._advance_packet(record.parent_packet_id)
    def _finalize_lost(self, record: PacketRecord) -> None:
        """Loss path. Same cleanup as delivered -- Alice's slots freed, cancels
        sent so Bob's aggregation completes -- but the packet did not make it
        and the caller will see LOST.
        LOST is the "genuinely unrecoverable" outcome: the optimistic ceiling
        (delivered + held + in_flight + recursing) dropped below K, so Bob
        cannot reach K arrivals even if everything still uncertain succeeded.
        This is distinct from RECOVERED_LOCALLY (delivered=0 but Alice still
        holds K -- she has the secret) and does not compose upward the same
        way: a sub-packet's LOST cascade marks parent-share FAILED, not HELD.
        If this is a sub-packet, cascade upward: the parent share this
        sub-packet was recursing on has now failed, and Bob's aggregation for
        the parent packet is still waiting for a terminal on that share (Alice
        never sent a CANCEL for it at recursion time). Send that CANCEL now,
        mark the parent-share FAILED, and advance the parent.
        """
        self._cleanup_shares(record)
        record.outcome = PacketOutcome.LOST
        log.logger.warning(
            f"QTCPOverseer: packet {record.packet_id} LOST "
            f"to {record.dst}"
        )
        if record.parent_packet_id is not None:
            parent = self.packets[record.parent_packet_id]
            parent.share_status[record.parent_share_index] = ShareStatus.FAILED
            # Bob's parent aggregation is waiting for a terminal on
            # parent_share_index. We never sent a CANCEL for it (Alice held
            # off at recursion time in case the sub-packet delivered). Now
            # that the sub-packet failed, send the CANCEL so parent's
            # aggregation reaches N_SHARES terminals.
            #
            # NOTE: parent_packet_id / parent_share_index on any message are
            # RELATIVE TO THAT MESSAGE'S OWN packet_id, not relative to the
            # sender. This CANCEL is about share `record.parent_share_index`
            # of packet `parent.packet_id`, so its parent link is parent's
            # own parentage (parent.parent_packet_id / parent.parent_share_
            # index). Only non-None at depth >= 2, but writing it consistently
            # keeps the invariant local -- readers should not have to reason
            # about which packet the fields "really" refer to.
            tid = self.app.mint_transfer_id()
            self.app.node.send_message(
                parent.dst,
                QTCPMessage(
                    QTCPMsgType.CANCEL,
                    transfer_id=tid,
                    packet_id=parent.packet_id,
                    share_index=record.parent_share_index,
                    parent_packet_id=parent.parent_packet_id,
                    parent_share_index=parent.parent_share_index,
                ),
            )
            log.logger.info(
                f"QTCPOverseer: sub-packet {record.packet_id} LOST "
                f"-> parent packet {parent.packet_id} share "
                f"{record.parent_share_index} counts as FAILED; CANCEL sent"
            )
            self._advance_packet(record.parent_packet_id)
    def _finalize_recovered(self, record: PacketRecord,
                            secret_slot: int) -> None:
        """Recovery path. The packet did not reach Bob, but Alice has the
        secret intact in `secret_slot`. Records the outcome and notifies Bob
        to purge any records he may hold for this packet.
        If this is a sub-packet, cascade upward per (ii): revert the parent-
        share to HELD at the recovered slot and let the parent's advance
        loop decide what to do next (fire the recovered share directly, or
        recurse again if depth permits, or exhaust into its own recovery).
        If this is a top-level packet, the caller sees the outcome via
        get_packet_outcome and the slot via get_recovered_slot. They decide
        whether to re-send, use the qubit elsewhere, or discard it.
        """
        record.recovered_slot = secret_slot
        record.outcome = PacketOutcome.RECOVERED_LOCALLY
        record.share_status = [ShareStatus.FAILED] * qss.N_SHARES
        record.share_slots = [None] * qss.N_SHARES
        log.logger.warning(
            f"QTCPOverseer: packet {record.packet_id} RECOVERED_LOCALLY "
            f"-> data memory {secret_slot} "
            f"(depth {record.depth}, restarts {record.restarts})"
        )
        # Notify Bob to purge any records for this packet. Under a faithful
        # NO_ENTANGLEMENT history he has none; the CANCELs synthesize
        # CANCELLED records that trigger the settled-with-zero-arrivals
        # purge. Under histories where he holds something, _on_cancel frees
        # his slots.
        for share_index in range(qss.N_SHARES):
            tid = self.app.mint_transfer_id()
            self.app.node.send_message(
                record.dst,
                QTCPMessage(
                    QTCPMsgType.CANCEL,
                    transfer_id=tid,
                    packet_id=record.packet_id,
                    share_index=share_index,
                    parent_packet_id=record.parent_packet_id,
                    parent_share_index=record.parent_share_index,
                ),
            )
        if record.parent_packet_id is not None:
            parent = self.packets[record.parent_packet_id]
            parent.share_slots[record.parent_share_index] = secret_slot
            log.logger.info(
                f"QTCPOverseer: sub-packet {record.packet_id} RECOVERED_LOCALLY "
                f"-> parent packet {parent.packet_id} share "
                f"{record.parent_share_index} firing directly at slot {secret_slot}"
            )
            self._fire_leaf(parent, record.parent_share_index, secret_slot)
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
                        parent_packet_id=record.parent_packet_id,
                        parent_share_index=record.parent_share_index,
                    ),
                )
                record.share_status[share_index] = ShareStatus.FAILED
            elif status is ShareStatus.FAILED:
                self.app.free_data_slot(slot)
                # Also send CANCEL. Under some histories Bob already got one (sweep,
                # timeout); his _on_cancel treats duplicates as no-ops. Under histories
                # where the share failed without ever sending a CANCEL (fire suppressed
                # at get_memory time -- no SEND_NOTICE, no timeout, no sweep), this is
                # the first and only signal Bob will get about this share, and without
                # it his aggregation stays one record short forever.
                tid = self.app.mint_transfer_id()
                self.app.node.send_message(
                    record.dst,
                    QTCPMessage(
                        QTCPMsgType.CANCEL,
                        transfer_id=tid,
                        packet_id=record.packet_id,
                        share_index=share_index,
                        parent_packet_id=record.parent_packet_id,
                        parent_share_index=record.parent_share_index,
                    ),
                )
    # ==================================================================
    # Bob side
    # ==================================================================
    def on_bob_transfer_finished(self, bob_transfer: BobTransfer) -> None:
        """Called by QTCPTransfer when a Bob-side transfer reaches terminal
        (ARRIVED, CANCELLED, or CONSUMED).
        Kept as its own method (not inlined) because it is the observer
        interface QTCPTransfer invokes by name via terminal_observers -- the
        transfer layer looks up this exact method on each registered observer.
        """
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
                # QEC leaves need the FULL codeword (correction mode); QSS nodes need
        # only K (erasure mode). Discriminate the same way _reconstruct_packet
        # does, so a partially-arrived QEC leaf is declared unrecoverable here
        # rather than being sent to reconstruct and tripping its full-codeword
        # assertion.
        is_qec_layer = any(getattr(r, "is_qec_layer", False) for r in records)
        required = qss.N_SHARES if is_qec_layer else qss.K_THRESHOLD
        if len(arrived) < required:
            log.logger.warning(
                f"QTCPOverseer: packet {packet_id} from {src} settled with only "
                f"{len(arrived)}/{qss.N_SHARES} shares; cannot reconstruct"
            )
            if not arrived:
                # Nothing physically held for this packet -- every record is
                # CANCELLED with no data slot. Purge them so a locally
                # restarted attempt under the SAME packet id starts with a
                # clean aggregation. (Alice's restart CANCELs arrive before
                # any of her new SEND_NOTICEs -- same FIFO channel -- so the
                # purge always completes before new records land.)
                stale_keys = [key for key, r in self.app.bob_transfers.items()
                              if r.src == src and r.packet_id == packet_id]
                for key in stale_keys:
                    del self.app.bob_transfers[key]
                log.logger.info(
                    f"QTCPOverseer: purged {len(stale_keys)} settled records "
                    f"for packet {packet_id} from {src}"
                )
            else:
                for r in arrived:
                    self.app.free_data_slot(r.data_index)
                    r.data_index = None
                # purge the records too, same as the zero-arrival case
                stale_keys = [key for key, rec in self.app.bob_transfers.items()
                              if rec.src == src and rec.packet_id == packet_id]
                for key in stale_keys:
                    del self.app.bob_transfers[key]
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
        used = arrived
        used_positions = [r.share_index for r in used]
        used_slots = [r.data_index for r in used]
 
        # Correction mode iff these records are a QEC leaf block. The leaf's
        # shares carry is_qec_layer=True (threaded from Alice via SEND_NOTICE);
        # QSS nodes (including the parent-None root) carry False.
        is_qec_layer = any(getattr(r, "is_qec_layer", False) for r in records)
 
        if is_qec_layer:
            # Correction mode needs the FULL codeword: all N_SHARES arrived.
            # The QEC layer guarantees this (fail-stop: it goes LOST on any
            # failed share, so it only ever reconstructs with all N_SHARES).
            # If we are correction-decoding with fewer, the QEC invariant broke
            # upstream -- trip loudly rather than silently miscorrect a partial
            # codeword.
            assert len(arrived) == qss.N_SHARES, (
                f"packet {packet_id}: correction-mode decode reached with "
                f"{len(arrived)}/{qss.N_SHARES} arrived shares. A QEC-leaf "
                f"packet must reconstruct with a full codeword. Either a "
                f"non-QEC packet leaked in (discriminator broken) or the QEC "
                f"fail-stop policy was bypassed."
            )
            secret_slot = self._decode_corrected(
                used_positions, used_slots,
                desc=f"packet {packet_id} from {src}")
        else:
            # Erasure mode (unchanged): reconstruct from K, treat rest as erased.
            assert len(arrived) == qss.K_THRESHOLD, (
            f"packet {packet_id}: expected exactly K={qss.K_THRESHOLD} arrivals, "
            f"got {len(arrived)} -- fire policy invariant (<=K arrivals) violated"
        )
            used_indices_set = set(used_positions)
            erased = [i for i in range(qss.N_SHARES)
                      if i not in used_indices_set]
            secret_slot = self._decode_at(
                used_positions, used_slots, erased,
                desc=f"packet {packet_id} from {src} reconstructed")
        for record in records:
            record.state = BobState.CONSUMED
            record.data_index = None
        # If this was a sub-packet, the reconstructed qubit is a share of a
        # parent packet, not a top-level secret. Synthesize an ARRIVED
        # BobTransfer for the parent-share so parent's aggregation gets a
        # record that counts as an arrival, then trigger parent's
        # _check_packet_complete.
        #
        # Parent-link detection scans all records rather than trusting
        # records[0]: records synthesized from CANCELs may lack parent fields
        # (dict ordering makes "first record" unreliable), but any record
        # created from a SEND_NOTICE carries them.
        parented = next((r for r in records
                         if r.parent_packet_id is not None), None)
        if parented is not None:
            synth_tid = self.next_synth_id
            self.next_synth_id -= 1
            self.app.bob_transfers[(src, synth_tid)] = BobTransfer(
                transfer_id=synth_tid,
                src=src,
                comm_memory_name=None,
                state=BobState.ARRIVED,
                data_index=secret_slot,
                share_index=parented.parent_share_index,
                packet_id=parented.parent_packet_id,
            )
            log.logger.info(
                f"QTCPOverseer: sub-packet {packet_id} from {src} "
                f"reconstructed -> parent packet {parented.parent_packet_id} "
                f"share {parented.parent_share_index} ARRIVED "
                f"(slot {secret_slot}, synth id {synth_tid})"
            )
            
            self._check_packet_complete(src, parented.parent_packet_id)
            return
        # Top-level packet: hand off the reconstructed secret. The slot is not
        # returned to the free pool -- it holds the recovered secret, retrieved
        # via get_received_packet.
        self.received_packets[(src, packet_id)] = secret_slot
    def mint_packet_id(self) -> int:
        """Reserve the next packet id without encoding or dispatching. The app
        calls this at send-request time so it can hand the user a stable handle
        before the packet is actually dispatched (which may be deferred until
        the handshake establishes)."""
        packet_id = self.next_packet_id
        self.next_packet_id += 1
        return packet_id
    def on_connection_closed(self, responder: str) -> None:
            """Tear down every in-progress packet bound for `responder` at window
            close, at PACKET granularity.
    
            For each in-progress packet to `responder`:
            - ALL N_SHARES still HELD  -> safe to recover locally. Nothing has
                left Alice, so her held shares are a self-contained entangled set;
                decode them and finalize RECOVERED_LOCALLY (existing path).
            - otherwise                -> LOST (ruthless). Some share is in
                flight / recursing / delivered; recovering would require decoding
                around state that has physically left Alice (entangled at Bob),
                which is unsafe. A mid-flight window close is a sender-side
                misconfiguration (end_t is the sender's choice), not something the
                protocol recovers from.
    
            Finalizing here sets each packet's outcome terminal BEFORE the transfer
            layer's sweep runs its per-share _finish. That makes the sweep's
            cascade (and any trailing ACK) inert via the `outcome is not
            IN_PROGRESS` guards in on_alice_transfer_finished / _advance_packet --
            so no recursion, restart, or re-queue is triggered.
    
            Bob is cleaned up entirely by CANCELs: _finalize_lost / the local-
            recover path CANCEL the held/failed shares; the sweep CANCELs the
            in-flight/pending ones. No Bob-side change is needed.
    
            Iterate a snapshot (list(...)) because finalizing a packet can cascade
            into its parent (mutating self.packets) mid-iteration.
            """
            for packet_id, record in list(self.packets.items()):
                if record.dst != responder:
                    continue
                if record.outcome is not PacketOutcome.IN_PROGRESS:
                    continue
    
                all_held = all(s is ShareStatus.HELD
                            for s in record.share_status)
    
                if all_held:
                    # Safe local recovery: every share is still HELD, so the whole
                    # codeword is present, live, and clean in Alice's memory.
                    #
                    # Decode the FULL intact codeword: feed all N real shares to
                    # the DECODER (U-dagger, the exact inverse of _encode_at's
                    # ENCODER). The secret lands at SECRET_INDEX; the other N-1
                    # positions measure out as ancillas. There are NO erasures and
                    # NO |0> stand-ins -- an "erasure" in QSS means a genuinely
                    # destroyed share replaced by a classical |0>, and the decoder
                    # is a full N-qubit circuit that always takes N inputs. Faking
                    # an erasure here by measuring out a live share would collapse
                    # the entanglement of the ones we keep and damage the decode.
                    # Since we hold the complete codeword, the clean move is the
                    # direct inverse, not an erasure decode.
                    slots = list(record.share_slots)   # code-position order: slots[i] is share i
                    data_arr = self.app.node.get_component_by_name(
                        self.app.node.data_memo_arr_name)
                    keys = [data_arr[s].qstate_key for s in slots]
                    rnd = self.app.node.get_generator().random()
                    self.app.node.timeline.quantum_manager.run_circuit(
                        qss.DECODER, keys, rnd)
                    secret_slot = slots[qss.SECRET_INDEX]
                    # DECODER measured the N-1 non-secret positions; free them.
                    for i, s in enumerate(slots):
                        if i != qss.SECRET_INDEX:
                            self.app.free_data_slot(s)
                    log.logger.warning(
                        f"QTCPOverseer: packet {record.packet_id} recovered at "
                        f"window close to {responder} (all shares still held) "
                        f"-> data memory {secret_slot}"
                    )
                    self._finalize_recovered(record, secret_slot)
                else:
                    # Ruthless LOST: some share already left Alice.
                    log.logger.warning(
                        f"QTCPOverseer: packet {record.packet_id} LOST at window "
                        f"close to {responder} (shares in flight/recursing/"
                        f"delivered; no safe local recovery)"
                    )
                    self._finalize_lost(record)