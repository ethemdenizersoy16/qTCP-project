"""qTCP transfer layer: reliable single-qubit delivery over teleportation.
 
This layer is a *detector*, not a recoverer. Teleportation consumes Alice's
data qubit at the Bell measurement, so a lost qubit cannot be resent -- that
is the no-cloning wall, and QSS is what gets you past it. What this layer
provides is a clean per-transfer outcome (DELIVERED / FAILED-with-reason) that
higher layers (QSS batching, the handshake) use to decide what to do next.
 
Contract:
    send_single_qubit(data_memory_index, dst) -> transfer_id
    ...later, exactly once per transfer, the layer records a terminal outcome.
"""
 
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from sequence.components.circuit import Circuit
from sequence.app.request_app import RequestApp
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.message import Message
from sequence.resource_management.memory_manager import MemoryInfo
from sequence.utils import log
from sequence.entanglement_management.qtcp_teleportation import QTCPTeleportMessage, QTCPTeleportProtocol
 
QTCP_APP = "qtcp_app"   # the Message.receiver string; keep as a constant so a
                        # typo is a NameError rather than a silently dropped message
 



class QTCPMsgType(Enum):
    SEND_NOTICE = auto()   # Alice -> Bob: "transfer N is arriving on comm memory X"
    ACK         = auto()   # Bob -> Alice: "transfer N received"
    NACK        = auto()   # Bob -> Alice: "transfer N is lost/corrupt on my side"
    CANCEL      = auto()   # Alice -> Bob: "I gave up, no more bits coming"
 
 
class QTCPMessage(Message):
    """Message for the qTCP transport layer."""
 
    __slots__ = ["transfer_id", "share_index","packet_id", "comm_memory_name", "reason"]
 
    def __init__(self, msg_type: QTCPMsgType, transfer_id: int,
                 share_index: int = None, packet_id: int = None,
                 comm_memory_name: str = None, reason: "FailureReason" = None):
        super().__init__(msg_type, QTCP_APP)
        self.transfer_id = transfer_id
        self.share_index = share_index
        self.packet_id = packet_id
        self.comm_memory_name = comm_memory_name   # Bob's memory, for SEND_NOTICE
        self.reason = reason                       # for NACK
 
    def __str__(self) -> str:
        return f"QTCPMessage({self.msg_type.name}, transfer={self.transfer_id})"
 
 
class TransferStatus(Enum):
    PENDING   = auto()   # created, waiting for an entangled pair
    IN_FLIGHT = auto()   # pair bound, Bell measurement fired, awaiting ACK
    DELIVERED = auto()   # terminal: Bob has it
    FAILED    = auto()   # terminal: gone
 
 
class FailureReason(Enum):
    NO_ENTANGLEMENT = auto()   # no pair arrived within the reservation window
    NO_ACK          = auto()   # Bell measurement fired, Bob never confirmed
    # NACK reserved for QEC integration -- FailureReason value TBD when QEC lands.
 
class BobState(Enum):
    NOTIFIED = auto()
    ARRIVED = auto()
    CANCELLED = auto()
    CONSUMED = auto()

@dataclass
class Transfer:
    """Sender-side record of one qubit's journey. Lives on QTCPTransfer, not
    in a method: get_memory(), received_message() and on_timeout() are three
    separate events that all need to see the same transfer."""
    transfer_id: int
    data_memory_index: int
    dst: str
    status: TransferStatus = TransferStatus.PENDING
    send_time: int = None
    comm_memory = None                 # sequence Memory, bound when a pair lands
    protocol: QTCPTeleportProtocol = None  # kept so on_timeout can clean it up
    reason: FailureReason = None
    share_index: int = None            # passthrough; this layer never interprets it
    packet_id: int = None

@dataclass
class BobTransfer:
    transfer_id: int
    src: str
    comm_memory_name: str
    state: BobState                 # NOTIFIED / ARRIVED / CANCELLED
    data_index: int = None
    protocol: QTCPTeleportProtocol = None
    share_index: int = None
    packet_id: int = None

 
class QTCPTransfer(RequestApp):
    """Reliable single-qubit delivery over teleportation.

    A *detector*, not a recoverer: teleportation consumes Alice's data qubit at
    the Bell measurement, so a lost qubit cannot be resent. This layer reports a
    terminal outcome per transfer (DELIVERED, or FAILED with a reason); higher
    layers (QSS batching, the handshake) decide what to do about it.

    One instance per node. Both endpoints run the same class -- Alice populates
    the sender fields, Bob the receiver fields.

    NOTE: QTCPTransfer squats in DQCNode's `teleport_app` slot, because
    TeleportProtocol.bob_handle_correction() hardcodes a callback to
    `owner.teleport_app.teleport_complete()`. A real TeleportApp cannot coexist
    on the same node.

    Attributes:
        node (QTCPNode): the quantum node this layer is attached to.
        name (str): name of this instance.

        # --- sender (Alice) ---
        transfers (dict[int, Transfer]): transfer table, keyed by transfer id.
            Lives on QTCPTransfer rather than in a method because get_memory(),
            received_message() and on_timeout() are three separate events that
            must all see the same record.
        next_transfer_id (int): monotonic counter; Alice stamps outgoing transfers.
        pending (list[int]): FIFO of transfer ids queued but not yet bound to an
            entangled pair.
        rto (int): retransmission timeout, in ps. Currently a constant;
            should be derived from path latency.
        # --- receiver (Bob) ---
        self.bob_transfers: dict[int, BobTransfer]: transfer id -> transfer information. BobTransfer holds information about
        the state of the transfer

        free_data_slots (list[int]): data memory indices available to receive an
            incoming state. Empty means a share has nowhere to land.
        

        # --- instrumentation ---
        metrics (list[dict]): one record per terminal transfer, for later analysis.

        # --- circuits ---
        _SWAP_CIRCUIT = Circuit(2)
        _SWAP_CIRCUIT.swap(0, 1): Circuit for swapping the teleported bit into data memory
    """
 
    # --- circuits ---
    _SWAP_CIRCUIT = Circuit(2)
    _SWAP_CIRCUIT.swap(0, 1)

    _MEASURE_CIRCUIT = Circuit(1)
    _MEASURE_CIRCUIT.measure(0)

    def __init__(self, node, rto: int):
        super().__init__(node)
        self.name = f"{node.name}.QTCPTransfer"
 
        node.register_app(QTCP_APP, self)
 
        # NOTE: we deliberately squat in DQCNode's teleport_app slot.
        # TeleportProtocol.bob_handle_correction() ends with a hardcoded
        # `self.owner.teleport_app.teleport_complete(comm_key)`. Taking the slot
        # is how we receive that callback without patching library code.
        # Consequence: a real TeleportApp cannot coexist on this node.
        node.teleport_app = self

        self.terminal_observers: list = []
        # --- sender state ---
        self.transfers: dict[int, Transfer] = {}
        self.next_transfer_id: int = 0
        self.pending: list[int] = []      # FIFO of transfer ids awaiting a pair
        self.rto: int = rto              # ps. TODO: derive from path latency
        
        # --- receiver state ---
        self.bob_transfers: dict[tuple[str, int], BobTransfer] = {}

        data_arr = node.get_component_by_name(node.data_memo_arr_name)
        self.free_data_slots: (list[int]) = list(range(len(data_arr)))

        # --- instrumentation ---
        self.metrics: list[dict] = []


 
        log.logger.debug(f"{self.name}: initialized")
 
    def start(self, responder, start_t, end_t, memory_size, fidelity):
        super().start(responder, start_t, end_t, memory_size, fidelity)

        # Schedule the reservation-end sweep. Any transfer still queued in
        # self.pending at end_t never got a pair and never will -- the sweep
        # terminates each one as NO_ENTANGLEMENT. Without this, an unfired
        # transfer has no on_timeout (scheduled in _fire, which never ran)
        # and hangs forever.
        process = Process(self, "on_reservation_end", [])
        event = Event(end_t, process, self.node.timeline.schedule_counter)
        self.node.timeline.schedule(event)


    # ------------------------------------------------------------------
    # sender: the primitive
    # ------------------------------------------------------------------
 
    def send_single_qubit(self, data_memory_index: int, dst: str,
                          share_index: int = None, packet_id: int = None) -> int:
        """Queue one data qubit for delivery to `dst`. Returns a transfer id.
 
        This does NOT teleport immediately -- there may be no entangled pair
        yet. The transfer sits in `pending` until get_memory() reports one, at
        which point it gets bound and fired.
        """
        transfer_id = self.next_transfer_id
        self.next_transfer_id += 1
 
        transfer = Transfer(
            transfer_id=transfer_id,
            data_memory_index=data_memory_index,
            dst=dst,
            share_index=share_index,
            packet_id = packet_id
        )
        self.transfers[transfer_id] = transfer
        self.pending.append(transfer_id)

        log.logger.info(
            f"{self.name}: queued transfer {transfer_id} "
            f"(data_memory={data_memory_index} -> {dst})"
        )
        return transfer_id

    def mint_transfer_id(self) -> int:
        """Reserve a fresh transfer id without creating a Transfer record.
 
        Used by the layer above for CANCEL messages on shares that were never
        actually fired -- e.g. a packet's un-sent shares once Bob has enough
        arrivals for reconstruction. Bob's _on_cancel synthesises a CANCELLED
        BobTransfer keyed on this id, which lets his aggregation reach the
        expected terminal count without those shares ever having been queued.
        """
        transfer_id = self.next_transfer_id
        self.next_transfer_id += 1
        return transfer_id
    # ------------------------------------------------------------------
    # sender: pair arrives -> bind and fire
    # ------------------------------------------------------------------
 
    def get_memory(self, info: MemoryInfo) -> None:
        """Called when a memory changes state. An ENTANGLED comm memory is the
        resource a pending transfer has been waiting for.
 
        Unlike TeleportApp, we do NOT fire on any entangled pair -- we fire only
        if a transfer is actually waiting. A pair with no pending transfer sits
        idle and available, which is exactly what qTCP needs (handshake probes,
        reserve pairs, shares not yet encoded).
        """
        log.logger.debug(f"{self.name}: get_memory index={info.index} state={info.state}")
        if info.index not in self.memo_to_reservation:
            return
        if info.state != "ENTANGLED":
            return
 
        for i, tid in enumerate(self.pending):
            if self.transfers[tid].dst == info.remote_node:
                self.pending.pop(i)
                self._fire(self.transfers[tid], info)
                return

 
    def _fire(self, transfer: Transfer, info: MemoryInfo) -> None:
        """Bind a pair to a transfer, tell Bob it's coming, teleport, arm the timer."""
        transfer.comm_memory = info.memory
        transfer.status = TransferStatus.IN_FLIGHT
        transfer.send_time = self.node.timeline.now()
 
        # Tell Bob which transfer is arriving on which of his memories, BEFORE
        # the teleportation's MEASUREMENT_RESULT lands. Both messages are sent
        # at the same simulated time over the same classical channel, so they
        # arrive together and the tie is broken by schedule_counter -- this one
        # is sent first, so it is processed first.
        # ORDERING DEPENDENCY: if this ever inverts, Bob's teleport_complete()
        # fires with a comm_key he cannot map to a transfer.
        notice = QTCPMessage(
            QTCPMsgType.SEND_NOTICE,
            transfer_id=transfer.transfer_id,
            share_index=transfer.share_index,
            packet_id=transfer.packet_id,
            comm_memory_name=info.remote_memo,
        )
        self.node.send_message(transfer.dst, notice)
 
        protocol = QTCPTeleportProtocol(
            self.node,
            alice=True,
            transfer_id = transfer.transfer_id,
            data_memory_index=transfer.data_memory_index,
            remote_node_name=transfer.dst,
        )
        protocol.set_alice_comm_memory_name(info.memory.name)
        protocol.set_alice_comm_memory(info.memory)
        protocol.set_bob_comm_memory_name(info.remote_memo)
        transfer.protocol = protocol
 
        reservation = self.memo_to_reservation[info.index]
 
        # Let Bob's generation protocol finish before Alice measures -- same
        # scheduling dance TeleportApp does.
        now = self.node.timeline.now()
        process = Process(protocol, "alice_bell_measurement", [reservation])
        event = Event(now, process, self.node.timeline.schedule_counter)
        self.node.timeline.schedule(event)
 
        # Arm the retransmission timer. If the ACK lands first, on_timeout()
        # sees a terminal status and no-ops.
        timeout_process = Process(self, "on_timeout", [transfer.transfer_id])
        timeout_event = Event(now + self.rto, timeout_process,
                              self.node.timeline.schedule_counter)
        self.node.timeline.schedule(timeout_event)
 
        log.logger.info(
            f"{self.name}: transfer {transfer.transfer_id} in flight "
            f"(comm={info.memory.name}, bob_comm={info.remote_memo})"
        )
    #------------------------------------------------------------------
    #Helpers
    #------------------------------------------------------------------

    def _bob_transfers_for_packet(self, src: str, packet_id: int) -> list[BobTransfer]:
        return [t for t in self.bob_transfers.values()
                if t.src == src and t.packet_id == packet_id]
    

    def alloc_data_slot(self) -> int | None:
        """Claim a free data memory index, or None if the array is full."""
        if not self.free_data_slots:
          return None
        return self.free_data_slots.pop(0)
 
 
    def free_data_slot(self, index: int) -> None:
        """Release a data slot. Measures the qubit out first: it may still be
        entangled with qubits held elsewhere (an abandoned share is entangled with
        the shares that did arrive), and returning it to the pool without
        collapsing it leaves that correlation live. Reset to |0> so the pool's
        invariant holds -- a free slot holds |0>, which the encoder relies on for
        its ancillas."""
        if index is None or index in self.free_data_slots:
            return

        data_arr = self.node.get_component_by_name(self.node.data_memo_arr_name)
        memory = data_arr[index]

        rnd = self.node.get_generator().random()
        self.node.timeline.quantum_manager.run_circuit(
            self._MEASURE_CIRCUIT, [memory.qstate_key], rnd)

        memory.update_state(np.array([1, 0], dtype=complex))
        self.free_data_slots.append(index)


   
    def _memory_by_qstate_key(self, key: int):
        """Find the comm memory holding a given qstate key.
    
        TODO: this is a linear scan. If it shows up in profiling, keep a
        {qstate_key: memory} map instead.
        """
        memo_arr = self.node.get_component_by_name(self.node.memo_arr_name)
        for memory in memo_arr:
            if memory.qstate_key == key:
                return memory
        return None
    
    def _memory_by_name(self, name):
        memo_arr = self.node.get_component_by_name(self.node.memo_arr_name)
        for m in memo_arr:
            if m.name == name:
                return m
        return None
    

    #------------------------------------------------------------------
    #Message handling
    #------------------------------------------------------------------

    #Bob:
    def _on_send_notice(self, src: str, msg) -> None:
        """Alice announces that transfer N is arriving on one of Bob's comm memories.

        Bob records the transfer in NOTIFIED so that when teleport_complete() later
        fires with a bare qstate key, the arriving state can be tied back to a
        transfer id and a sender to ACK.

        Must land before the teleportation's MEASUREMENT_RESULT. Both are sent at
        the same simulated instant over the same channel; SEND_NOTICE is sent first,
        so it takes the lower schedule_counter and is processed first.
        """
        tid = msg.transfer_id


        
        existing = self.bob_transfers.get((src, tid))
        if existing is not None:
            # Duplicate notice. Never overwrite -- an existing record may already be
            # ARRIVED or CANCELLED, and resetting it to NOTIFIED would resurrect a
            # terminal transfer.
            log.logger.debug(
                f"{self.name}: duplicate SEND_NOTICE for transfer {tid} "
                f"(state {existing.state}); ignoring"
            )

            return
        
        tp = QTCPTeleportProtocol(self.node, alice=False, transfer_id = tid, remote_node_name=src)
        tp.set_bob_comm_memory_name(msg.comm_memory_name)
        tp.set_bob_comm_memory( self._memory_by_name(msg.comm_memory_name))

        self.bob_transfers[(src, tid)] = BobTransfer(
            transfer_id=tid,
            src=src,
            comm_memory_name=msg.comm_memory_name,
            state=BobState.NOTIFIED,
            share_index=msg.share_index,
            packet_id=msg.packet_id,
            protocol = tp
        )
    
        
        log.logger.debug(
            f"notified of transfer {tid} (packet {msg.packet_id}, share {msg.share_index}) "
            f"arriving on {msg.comm_memory_name} from {src}"
        )

    def _on_cancel(self, src: str, msg) -> None:
        """Alice has given up on this transfer (timeout, or swept at reservation end).

        Bob marks it terminal and releases any data slot he was holding. One-way
        message -- no reply.
        """
        tid = msg.transfer_id
        transfer = self.bob_transfers.get((src , tid))

        if transfer is None:
            # Never heard of it -- notice never arrived, or Alice never fired
            # (swept from PENDING at reservation end). Synthesize a CANCELLED
            # record so observers see a terminal event and can settle any
            # per-packet aggregation.
            self.bob_transfers[(src, tid)] = BobTransfer(
                transfer_id=tid,
                src=src,
                comm_memory_name=None,      # no SEND_NOTICE ever arrived
                state=BobState.CANCELLED,
                share_index=msg.share_index,
                packet_id=msg.packet_id,
            )
            log.logger.debug(f"{self.name}: transfer {tid} cancelled (synthetic record)")
            for obs in self.terminal_observers:
                obs.on_bob_transfer_finished(self.bob_transfers[(src, tid)])
            return

        if transfer.state is BobState.CANCELLED:
            return  # duplicate cancel

        if transfer.state is BobState.CONSUMED:
            return

        if transfer.state is BobState.ARRIVED:
            # Qubit already landed (ACK may have been lost or crossed the cancel).
            # Alice considers it dead, so free the slot or it leaks.
            self.free_data_slot(transfer.data_index)

        transfer.state = BobState.CANCELLED
        for obs in self.terminal_observers:
            obs.on_bob_transfer_finished(transfer)

        log.logger.debug(f"{self.name}: transfer {tid} cancelled")
    
    #Alice:
    def _on_ack(self, src: str, msg) -> None:
        transfer = self.transfers.get(msg.transfer_id)
        if transfer is None or transfer.status in (TransferStatus.DELIVERED,
                                               TransferStatus.FAILED):
            return
        self._finish(transfer, TransferStatus.DELIVERED)

    def _on_nack(self, src: str, msg) -> None:
        transfer = self.transfers.get(msg.transfer_id)
        if transfer is None or transfer.status in (TransferStatus.DELIVERED,
                                               TransferStatus.FAILED):
            return
        self._finish(transfer, TransferStatus.FAILED, msg.reason)
    
    
    def received_message(self, src: str, msg) -> None:
        if isinstance(msg, QTCPTeleportMessage):
            record = self.bob_transfers.get((src, msg.transfer_id))
            if record and record.protocol:
                record.protocol.received_message(src, msg)
            return
        
        #Bob side:
        if msg.msg_type is QTCPMsgType.SEND_NOTICE:
            self._on_send_notice(src, msg)
        elif msg.msg_type is QTCPMsgType.CANCEL:
            self._on_cancel(src,msg)

        #Alice side:
        elif msg.msg_type is QTCPMsgType.ACK:
            self._on_ack(src, msg)
        elif msg.msg_type is QTCPMsgType.NACK:
            self._on_nack(src, msg)
        
        else:
            log.logger.debug(f"{self.name}: unknown message type")
   
    def on_timeout(self, transfer_id: int) -> None:
        """Diagnostic failsafe. Fires rto after _fire.

        Under the intended model (perfect classical channel, handshake-verified
        quality), a fired transfer resolves via ACK or NACK before this timer
        expires. Reaching this branch means an assumption was violated: RTO too
        short for actual path latency, or Bob's teleport_complete path failed
        silently.

        If it does fire on a still-unresolved transfer: send CANCEL so Bob does
        not sit on a NOTIFIED record forever, and _finish as NO_ACK. The alter-
        native is a hang, which is harder to diagnose than a warning line.

        If the ACK already landed, this is a no-op: the event fires anyway (we
        do not cancel it), sees a terminal status, and returns.
        """
        transfer = self.transfers.get(transfer_id)
        if transfer is None or transfer.status in (TransferStatus.DELIVERED,
                                                TransferStatus.FAILED):
            return  # already resolved -- ACK/NACK beat the clock

        log.logger.warning(
            f"{self.name}: transfer {transfer_id} timed out without ACK/NACK "
            f"-- this should not happen under the intended model. Cancelling."
        )
        self.node.send_message(
            transfer.dst,
            QTCPMessage(
                QTCPMsgType.CANCEL,
                transfer_id=transfer_id,
                packet_id=transfer.packet_id,
                share_index=transfer.share_index,
            ),
        )
        self._finish(transfer, TransferStatus.FAILED, FailureReason.NO_ACK)

    #------------------------------------------------------------------
    #Teleportation Complete (To be used by Bob)
    #------------------------------------------------------------------

    def teleport_complete(self, comm_key: int, transfer_id: int, src:str) -> None:
        """Called locally by TeleportProtocol once corrections have been applied.

        At this point |psi> is sitting in Bob's COMM memory -- the entanglement was
        consumed and that memory now holds the payload. It must be swapped into data
        memory before the comm memory is released, or the resource manager will hand
        it straight back to entanglement generation and overwrite the state.

        """
        comm_memory = self._memory_by_qstate_key(comm_key)
        if comm_memory is None:
            log.logger.warning(f"{self.name}: teleport_complete for unknown key {comm_key}")
            return

        transfer = self.bob_transfers.get((src, transfer_id))
        if transfer is None:
            # Unannounced qubit: no SEND_NOTICE for this memory. Garbage -- discard.
            log.logger.warning(
                f"{self.name}: teleport landed on {comm_memory.name} with no matching "
                f"transfer; discarding"
            )
            self.node.resource_manager.update(None, comm_memory, MemoryInfo.RAW)
            return

        if transfer.state is BobState.CANCELLED:
            # Race: Alice gave up, the qubit arrived anyway. Drop it. No data slot
            # was allocated (allocation happens here, at arrival), so nothing to free.
            log.logger.info(
                f"{self.name}: transfer {transfer.transfer_id} arrived after cancel; "
                f"discarding"
            )
            rnd = self.node.get_generator().random()
    
            self.node.timeline.quantum_manager.run_circuit(
                self._MEASURE_CIRCUIT, [comm_key], rnd)
    
            self.node.resource_manager.update(None, comm_memory, MemoryInfo.RAW)

            return

        if transfer.state is BobState.ARRIVED:
            # Duplicate completion -- one teleport per pair, so this shouldn't happen.
            log.logger.warning(
                f"{self.name}: duplicate teleport_complete for transfer "
                f"{transfer.transfer_id}; ignoring"
            )
            return

        # --- normal path: state is NOTIFIED ---

        data_index = self.alloc_data_slot()
        assert data_index is not None, (
            f"{self.name}: no free data slot for transfer {transfer.transfer_id} -- "
            f"window accounting is broken. Memory reservation is meant to guarantee "
            f"a slot for every incoming share; if this fires, the guarantee has been "
            f"violated upstream and the fix belongs there, not here."
        )

        # Move comm -> data. Must happen BEFORE releasing the comm memory.
        data_arr = self.node.get_component_by_name(self.node.data_memo_arr_name)
        data_key = data_arr[data_index].qstate_key
        rnd = self.node.get_generator().random()
        self.node.timeline.quantum_manager.run_circuit(
            self._SWAP_CIRCUIT, [comm_key, data_key], rnd
        )

        rnd = self.node.get_generator().random()
        self.node.timeline.quantum_manager.run_circuit(self._MEASURE_CIRCUIT, [comm_key], rnd)

        transfer.data_index = data_index
        
        transfer.state = BobState.ARRIVED
        
        # Comm memory is spent -- release it. We do NOT expire the reservation rules
        # (unlike TeleportApp): other transfers on this connection ride the same
        # reservation.
        self.node.resource_manager.update(None, comm_memory, MemoryInfo.RAW)

        log.logger.info(
            f"{self.name}: transfer {transfer.transfer_id} received, swapped into "
            f"data memory {data_index}"
        )

        self.node.send_message(
            transfer.src,
            QTCPMessage(QTCPMsgType.ACK, transfer_id=transfer.transfer_id),
        )
        for obs in self.terminal_observers:
            obs.on_bob_transfer_finished(transfer)

    def _finish(self, transfer, status, reason=None) -> None:
        """The one place a transfer reaches a terminal state.
    
        Everything that must happen exactly once per transfer -- releasing the
        source data slot, releasing the comm memory, recording metrics -- happens
        here, so no caller can forget a step.
        """
        transfer.status = status
        transfer.reason = reason
    
        # The Bell measurement consumed the source qubit; the slot is free again
        # unless the reason was NO_ENTANGLEMENT (in which case the qubit was
        # never sent and the slot still holds it -- the layer above may still
        # want it, e.g. for QSS recursion on a share Alice still holds).
        if reason is not FailureReason.NO_ENTANGLEMENT:
            self.free_data_slot(transfer.data_memory_index)
    
        if transfer.comm_memory is not None:
            self.node.resource_manager.update(None, transfer.comm_memory, MemoryInfo.RAW)
    
        self.metrics.append({
            "transfer_id": transfer.transfer_id,
            "status": status.name,
            "reason": reason.name if reason else None,
            "send_time": transfer.send_time,
            "finish_time": self.node.timeline.now(),
            "latency": (self.node.timeline.now() - transfer.send_time
                        if transfer.send_time else None),
        })
    
        log.logger.info(
            f"{self.name}: transfer {transfer.transfer_id} -> {status.name}"
            + (f" ({reason.name})" if reason else "")
        )
    
        for obs in self.terminal_observers:
            obs.on_alice_transfer_finished(transfer)
    
    #------------------------------------------------------------------
    #Reservation sweep
    #------------------------------------------------------------------
    def on_reservation_end(self) -> None:
        """Diagnostic failsafe. Fires at end_t.

        Under the intended model (handshake-verified channel quality, single
        connection at a time, reservation windows long enough for the work
        queued into them), self.pending should be empty at end_t. Any transfer
        still queued signals that the assumed model does not hold in this run:
        connection was too short, channel quality is worse than the handshake
        indicated, or the layer above miscounted somewhere.

        If it does fire, terminate the stuck transfers as NO_ENTANGLEMENT (they
        never got a pair and never will now) and CANCEL to Bob so his
        aggregation does not deadlock waiting on shares that will not arrive.

        Kept as a failsafe because the cost is zero when self.pending is empty
        and a hang is much harder to diagnose than a warning line.
        """
        if not self.pending:
            return

        log.logger.warning(
            f"{self.name}: reservation-end sweep found {len(self.pending)} "
            f"PENDING transfers -- this should not happen under the intended "
            f"model. Terminating as NO_ENTANGLEMENT."
        )

        to_sweep = list(self.pending)
        self.pending.clear()

        for transfer_id in to_sweep:
            transfer = self.transfers.get(transfer_id)
            if transfer is None or transfer.status is not TransferStatus.PENDING:
                # defensive; PENDING <-> self.pending should be 1:1
                continue

            self.node.send_message(
                transfer.dst,
                QTCPMessage(
                    QTCPMsgType.CANCEL,
                    transfer_id=transfer_id,
                    packet_id=transfer.packet_id,
                    share_index=transfer.share_index,
                ),
            )

            self._finish(transfer, TransferStatus.FAILED,
                         FailureReason.NO_ENTANGLEMENT)


    #This function is for TESTING ONLY
    def get_received_state(self, src:str, transfer_id: int):
        """Return the quantum state Bob holds for a completed transfer, or None
        if nothing has arrived for that transfer."""
        record = self.bob_transfers.get((src, transfer_id))
        if record is None or record.state is not BobState.ARRIVED:
            return None
        
        data_arr = self.node.get_component_by_name(self.node.data_memo_arr_name)
        key = data_arr[record.data_index].qstate_key
        

        return self.node.timeline.quantum_manager.get(key).state