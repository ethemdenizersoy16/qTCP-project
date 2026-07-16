"""qTCP application: reliable single-qubit delivery over teleportation.
 
This layer is a *detector*, not a recoverer. Teleportation consumes Alice's
data qubit at the Bell measurement, so a lost qubit cannot be resent -- that
is the no-cloning wall, and QSS is what gets you past it. What this layer
provides is a clean per-transfer outcome (DELIVERED / FAILED-with-reason) that
higher layers (QSS batching, the handshake) use to decide what to do next.
 
Contract:
    send_single_qubit(data_memory_index, dst) -> transfer_id
    ...later, exactly once per transfer, the app records a terminal outcome.
"""
 
from dataclasses import dataclass, field
from enum import Enum, auto
 
from sequence.app.teleport_app import TeleportMessage, TeleportProtocol
from sequence.components.circuit import Circuit
from sequence.app.request_app import RequestApp
from sequence.entanglement_management.teleportation import TeleportProtocol
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.message import Message
from sequence.resource_management.memory_manager import MemoryInfo
from sequence.utils import log
 
 
QTCP_APP = "qtcp_app"   # the Message.receiver string; keep as a constant so a
                        # typo is a NameError rather than a silently dropped message
 



class QTCPMsgType(Enum):
    SEND_NOTICE = auto()   # Alice -> Bob: "transfer N is arriving on comm memory X"
    ACK         = auto()   # Bob -> Alice: "transfer N received"
    NACK        = auto()   # Bob -> Alice: "transfer N is lost/corrupt on my side"
    PENDING     = auto()   # Bob -> Alice: "transfer N didn't arrive yet but I am aware of it"
    PROBE       = auto()   # Alice -> Bob: "did you ever get transfer N?"
    CANCEL      = auto()   # Alice -> Bob: "I gave up, no more bits coming"
 
 
class QTCPMessage(Message):
    """Message for the qTCP transport layer.
 
    conn_id and share_index are carried but unused at this stage. They cost one
    field each now and are painful to thread through five handlers later.
    """
 
    __slots__ = ["transfer_id", "conn_id", "share_index", "comm_memory_name", "reason"]
 
    def __init__(self, msg_type: QTCPMsgType, transfer_id: int,
                 conn_id: int = 0, share_index: int = 0,
                 comm_memory_name: str = None, reason: "FailureReason" = None):
        super().__init__(msg_type, QTCP_APP)
        self.transfer_id = transfer_id
        self.conn_id = conn_id
        self.share_index = share_index
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
    RECEIVER_LOST   = auto()   # Bob explicitly NACKed (e.g. his memory decohered)
    RECEIVER_FULL   = auto()   # Bob doesn't have enough memory
 
class BobState(Enum):
    NOTIFIED = auto()
    ARRIVED = auto()
    CANCELLED = auto()

@dataclass
class Transfer:
    """Sender-side record of one qubit's journey. Lives on the app, not in a
    method: get_memory(), received_message() and on_timeout() are three
    separate events that all need to see the same transfer."""
    transfer_id: int
    data_memory_index: int
    dst: str
    status: TransferStatus = TransferStatus.PENDING
    attempts: int = 0
    probes: int = 0
    send_time: int = None
    comm_memory = None                 # sequence Memory, bound when a pair lands
    protocol: TeleportProtocol = None  # kept so on_timeout can clean it up
    reason: FailureReason = None
    share_index: int = 0               # passthrough; this layer never interprets it

@dataclass
class BobTransfer:
    transfer_id: int
    src: str
    comm_memory_name: str
    state: BobState                 # NOTIFIED / ARRIVED / CANCELLED
    data_index: int = None
    protocol: TeleportProtocol = None

 
class QTCPApp(RequestApp):
    """Reliable single-qubit delivery over teleportation.

    A *detector*, not a recoverer: teleportation consumes Alice's data qubit at
    the Bell measurement, so a lost qubit cannot be resent. This layer reports a
    terminal outcome per transfer (DELIVERED, or FAILED with a reason); higher
    layers (QSS batching, the handshake) decide what to do about it.

    One instance per node. Both endpoints run the same class -- Alice populates
    the sender fields, Bob the receiver fields.

    NOTE: this app squats in DQCNode's `teleport_app` slot, because
    TeleportProtocol.bob_handle_correction() hardcodes a callback to
    `owner.teleport_app.teleport_complete()`. A real TeleportApp cannot coexist
    on the same node.

    Attributes:
        node (QTCPNode): the quantum node this app is attached to.
        name (str): name of this app instance.

        # --- sender (Alice) ---
        transfers (dict[int, Transfer]): transfer table, keyed by transfer id.
            Lives on the app rather than in a method because get_memory(),
            received_message() and on_timeout() are three separate events that
            must all see the same record.
        next_transfer_id (int): monotonic counter; Alice stamps outgoing transfers.
        pending (list[int]): FIFO of transfer ids queued but not yet bound to an
            entangled pair.
        rto (int): retransmission timeout, in ps. Currently a constant;
            should be derived from path latency.
        max_attempts (int | None): give up after this many attempts. None = unbounded.
        max_probes (int | None): give up after this many probes. None = unbounded.
        peer (str | None): who the connectio is setup with
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

    def __init__(self, node, rto: int, max_attempts: int = None, max_probes: int = None):
        super().__init__(node)
        self.name = f"{node.name}.QTCPApp"
 
        node.register_app(QTCP_APP, self)
 
        # NOTE: we deliberately squat in DQCNode's teleport_app slot.
        # TeleportProtocol.bob_handle_correction() ends with a hardcoded
        # `self.owner.teleport_app.teleport_complete(comm_key)`. Taking the slot
        # is how we receive that callback without patching library code.
        # Consequence: a real TeleportApp cannot coexist on this node.
        node.teleport_app = self
 
        # --- sender state ---
        self.transfers: dict[int, Transfer] = {}
        self.next_transfer_id: int = 0
        self.pending: list[int] = []      # FIFO of transfer ids awaiting a pair
        self.rto: int = rto              # ps. TODO: derive from path latency
        
        self.max_attempts :int = max_attempts  # None = unbounded
        self.max_probes:int = 1               # 1 for it to be easier during testing
        
        self.peer: str | None = None
        # --- receiver state ---
        self.bob_transfers: dict[int, BobTransfer] = {}

        data_arr = node.get_component_by_name(node.data_memo_arr_name)
        self.free_data_slots: (list[int]) = list(range(len(data_arr)))
 
        # --- instrumentation ---
        self.metrics: list[dict] = []


 
        log.logger.debug(f"{self.name}: initialized")
 
    def start(self, responder, start_t, end_t, memory_size, fidelity):
        self.peer = responder
        super().start(responder, start_t, end_t, memory_size, fidelity)


    # ------------------------------------------------------------------
    # sender: the primitive
    # ------------------------------------------------------------------
 
    def send_single_qubit(self, data_memory_index: int, dst: str,
                          share_index: int = 0) -> int:
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
        )
        self.transfers[transfer_id] = transfer
        self.pending.append(transfer_id)
 
        log.logger.info(
            f"{self.name}: queued transfer {transfer_id} "
            f"(data_memory={data_memory_index} -> {dst})"
        )
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
 
        # Bob's side: he never called send_single_qubit(), so he has nothing
        # pending. He learns the transfer id from Alice's SEND_NOTICE instead.
        if not self.pending:
            return
 
        transfer_id = self.pending.pop(0)
        transfer = self.transfers[transfer_id]
 
        if info.remote_node != transfer.dst:
            # pair is with the wrong peer -- put it back and leave the pair alone
            self.pending.insert(0, transfer_id)
            return
 
        self._fire(transfer, info)
 
    def _fire(self, transfer: Transfer, info: MemoryInfo) -> None:
        """Bind a pair to a transfer, tell Bob it's coming, teleport, arm the timer."""
        transfer.comm_memory = info.memory
        transfer.attempts += 1
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
            comm_memory_name=info.remote_memo,   # Bob's half of the pair
        )
        self.node.send_message(transfer.dst, notice)
 
        protocol = TeleportProtocol(
            self.node,
            alice=True,
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
            f"(attempt {transfer.attempts}, comm={info.memory.name}, "
            f"bob_comm={info.remote_memo})"
        )
    #------------------------------------------------------------------
    #Helpers
    #------------------------------------------------------------------
    def get_received_state(self, transfer_id: int):
        """Return the quantum state Bob holds for a completed transfer, or None
        if nothing has arrived for that transfer."""
        record = self.bob_transfers.get(transfer_id)
        if record is None or record.state is not BobState.ARRIVED:
            return None
        
        data_arr = self.node.get_component_by_name(self.node.data_memo_arr_name)
        key = data_arr[record.data_index].qstate_key
        

        return self.node.timeline.quantum_manager.get(key).state
    
    def _bob_transfer_by_memory(self, comm_memory_name: str) -> BobTransfer | None:
        """Find Bob's transfer record for a given comm memory name.

        teleport_complete() enters holding a bare qstate key, resolves it to a
        comm memory name, and needs the corresponding transfer record. The records
        are keyed by transfer_id, so this scans on the comm_memory_name field.

        Relies on the invariant that a comm memory name maps to at most one live
        transfer (one entangled pair per memory at a time). O(n) in the number of
        Bob's tracked transfers, which is bounded by the receiver window.
        """
        for transfer in self.bob_transfers.values():
            if transfer.comm_memory_name == comm_memory_name and transfer.state is BobState.NOTIFIED:
                return transfer
        return None

    def alloc_data_slot(self) -> int | None:
        """Claim a free data memory index, or None if the array is full."""
        if not self.free_data_slots:
          return None
        return self.free_data_slots.pop(0)
 
 
    def free_data_slot(self, index: int) -> None:
        """Return a data memory index to the pool."""
        if index is not None and index not in self.free_data_slots:
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

        if tid in self.bob_transfers:
            # Duplicate notice. Never overwrite -- an existing record may already be
            # ARRIVED or CANCELLED, and resetting it to NOTIFIED would resurrect a
            # terminal transfer.
            log.logger.debug(
                f"{self.name}: duplicate SEND_NOTICE for transfer {tid} "
                f"(state {self.bob_transfers[tid].state.name}); ignoring"
            )
            return
        
        tp = TeleportProtocol(self.node, alice=False, remote_node_name=src)
        tp.set_bob_comm_memory_name(msg.comm_memory_name)
        tp.set_bob_comm_memory( self._memory_by_name(msg.comm_memory_name))

        self.bob_transfers[tid] = BobTransfer(
            transfer_id=tid,
            src=src,
            comm_memory_name=msg.comm_memory_name,
            state=BobState.NOTIFIED,
            protocol = tp
        )
        
        log.logger.debug(
            f"{self.name}: notified of transfer {tid} arriving on "
            f"{msg.comm_memory_name} from {src}"
        )

    def _on_probe(self, src: str, msg) -> None:
        """Alice is asking whether transfer N ever reached Bob.

        A probe never changes state -- it only reports. The reply is a pure
        function of the record's state:

            missing    -> silence   (never heard of it; Alice's timeout handles it)
            NOTIFIED   -> PENDING   (alive, was told, hasn't landed -- "still waiting")
            ARRIVED    -> ACK       (have it; original ACK was presumably lost)
            CANCELLED  -> NACK      (told to drop it; lost from Alice's view)

        Silence and PENDING are deliberately different: silence means "unknown",
        PENDING means "known and in progress". That distinction is what lets Alice
        tell a dead connection from a merely slow qubit.
        """
        tid = msg.transfer_id
        transfer = self.bob_transfers.get(tid)

        if transfer is None:
            # Unknown transfer -- stay silent. Do NOT NACK: the notice may simply
            # be in flight, and a NACK would make Alice give up on a live transfer.
            log.logger.debug(f"{self.name}: PROBE for unknown transfer {tid}; no reply")
            return

        if transfer.state is BobState.NOTIFIED:
            reply = QTCPMessage(QTCPMsgType.PENDING, transfer_id=tid)
        elif transfer.state is BobState.ARRIVED:
            reply = QTCPMessage(QTCPMsgType.ACK, transfer_id=tid)
        elif transfer.state is BobState.CANCELLED: # Won't come up under normal circumstances, here as a failsafe
            reply = QTCPMessage(QTCPMsgType.NACK, transfer_id=tid,
                                reason=FailureReason.RECEIVER_LOST)
        else:
            log.logger.warning(f"{self.name}: PROBE for transfer {tid} in "
                            f"unexpected state {transfer.state}; no reply")
            return

        self.node.send_message(src, reply)
        log.logger.debug(f"{self.name}: PROBE for transfer {tid} -> {reply.msg_type.name}")


    def _on_cancel(self, src: str, msg) -> None:
        """Alice has given up on this transfer (timeout / probes exhausted).

        Bob marks it terminal and releases any data slot he was holding. One-way
        message -- no reply.
        """
        tid = msg.transfer_id
        transfer = self.bob_transfers.get(tid)

        if transfer is None:
            # Never heard of it -- notice lost, or a stray cancel. Nothing to do.
            log.logger.debug(f"{self.name}: CANCEL for unknown transfer {tid}; ignoring")
            return

        if transfer.state is BobState.CANCELLED:
            return  # duplicate cancel

        if transfer.state is BobState.ARRIVED:
            # Qubit already landed (ACK may have been lost or crossed the cancel).
            # Alice considers it dead, so free the slot or it leaks.
            self.free_data_slot(transfer.data_index)

        transfer.state = BobState.CANCELLED
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
        if isinstance(msg, TeleportMessage):
            record = self._bob_transfer_by_memory(msg.bob_comm_memory_name)
            if record and record.protocol:
                record.protocol.received_message(src, msg)
            return
        
        #Bob side:
        if msg.msg_type is QTCPMsgType.SEND_NOTICE:
            self._on_send_notice(src, msg)
        elif msg.msg_type is QTCPMsgType.PROBE:
            self._on_probe(src, msg)
        elif msg.msg_type is QTCPMsgType.CANCEL:
            self._on_cancel(src,msg)

        #Alice side:
        elif msg.msg_type is QTCPMsgType.ACK:
            self._on_ack(src, msg)
        elif msg.msg_type is QTCPMsgType.NACK:
            self._on_nack(src, msg)
        elif msg.msg_type is QTCPMsgType.PENDING:
             log.logger.debug(f"{self.name}: transfer {msg.transfer_id} still pending at receiver")
        
        else:
            log.logger.debug(f"{self.name}: unknown message type")
   
    def on_timeout(self, transfer_id: int) -> None:
        """Fires rto after a send (and after each probe). Alice cannot resend --
        the Bell measurement consumed her qubit -- so on timeout she probes Bob to
        learn whether it arrived, rather than retransmitting.

        If the ACK already landed, the transfer is terminal and this is a no-op:
        the event fires anyway (we don't cancel it), sees a terminal status, and
        returns.
        """
        transfer = self.transfers.get(transfer_id)
        if transfer is None or transfer.status in (TransferStatus.DELIVERED,
                                                TransferStatus.FAILED):
            return  # already resolved -- ACK/NACK beat the clock

        if transfer.probes < self.max_probes:
            transfer.probes += 1
            log.logger.info(
                f"{self.name}: transfer {transfer_id} timed out, probing "
                f"({transfer.probes}/{self.max_probes})"
            )
            self.node.send_message(
                transfer.dst, QTCPMessage(QTCPMsgType.PROBE, transfer_id=transfer_id)
            )
            now = self.node.timeline.now()
            process = Process(self, "on_timeout", [transfer_id])
            event = Event(now + self.rto, process, self.node.timeline.schedule_counter)
            self.node.timeline.schedule(event)
        else:
            # Probes exhausted with no ACK/NACK. Bob is unreachable or the qubit is
            # gone. Declare failure locally, and tell Bob to release his record/slot.
            log.logger.info(
                f"{self.name}: transfer {transfer_id} failed after "
                f"{transfer.probes} probes"
            )
            self.node.send_message(
                transfer.dst, QTCPMessage(QTCPMsgType.CANCEL, transfer_id=transfer_id)
            )
            self._finish(transfer, TransferStatus.FAILED, FailureReason.NO_ACK)

    #------------------------------------------------------------------
    #Teleportation Complete (To be used by Bob)
    #------------------------------------------------------------------

    def teleport_complete(self, comm_key: int) -> None:
        """Called locally by TeleportProtocol once corrections have been applied.

        At this point |psi> is sitting in Bob's COMM memory -- the entanglement was
        consumed and that memory now holds the payload. It must be swapped into data
        memory before the comm memory is released, or the resource manager will hand
        it straight back to entanglement generation and overwrite the state.

        Not a message handler: there is no `src`. The sender to ACK is recorded on
        the transfer (BobTransfer.src, stored at notice time).
        """
        comm_memory = self._memory_by_qstate_key(comm_key)
        if comm_memory is None:
            log.logger.warning(f"{self.name}: teleport_complete for unknown key {comm_key}")
            return

        transfer = self._bob_transfer_by_memory(comm_memory.name)
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
        if data_index is None:
            # Failsafe: window should have guaranteed room. If this fires, the
            # window accounting is broken upstream. NACK so Alice learns.
            log.logger.warning(
                f"{self.name}: no free data slot for transfer {transfer.transfer_id} "
                f"(window accounting broken?)"
            )
            self.node.send_message(
                transfer.src,
                QTCPMessage(QTCPMsgType.NACK, transfer_id=transfer.transfer_id,
                            reason=FailureReason.RECEIVER_FULL),
            )
            self.node.resource_manager.update(None, comm_memory, MemoryInfo.RAW)
            return

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
    def _finish(self, transfer, status, reason=None) -> None:
        """The one place a transfer reaches a terminal state.
    
        Everything that must happen exactly once per transfer -- releasing the
        source data slot, releasing the comm memory, recording metrics -- happens
        here, so no caller can forget a step.
        """
        transfer.status = status
        transfer.reason = reason
    
        # The Bell measurement consumed the source qubit; the slot is free again.
        self.free_data_slot(transfer.data_memory_index)
    
        if transfer.comm_memory is not None:
            self.node.resource_manager.update(None, transfer.comm_memory, MemoryInfo.RAW)
    
        self.metrics.append({
            "transfer_id": transfer.transfer_id,
            "status": status.name,
            "reason": reason.name if reason else None,
            "attempts": transfer.attempts,
            "send_time": transfer.send_time,
            "finish_time": self.node.timeline.now(),
            "latency": (self.node.timeline.now() - transfer.send_time
                        if transfer.send_time else None),
        })
    
        log.logger.info(
            f"{self.name}: transfer {transfer.transfer_id} -> {status.name}"
            + (f" ({reason.name})" if reason else "")
        )
    
        # TODO: notify the batching layer (QSS / handshake) that this transfer
        # reached a terminal state, so it can evaluate its predicate.