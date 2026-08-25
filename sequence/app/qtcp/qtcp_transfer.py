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

import sequence.app.qtcp.qec as qec
import numpy as np

from sequence.kernel.quantum_state import KetState
from sequence.constants import MILLISECOND
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

    MEM_REQ   = auto()
    MEM_ACCEPT = auto()
    MEM_REJECT = auto()

    CLOSE = auto()

    QPING_BASIS   = auto()   # Alice -> Bob: measure the pair on comm memory <comm_memory_name> in <basis>
    QPING_OUTCOME = auto()   # Bob -> Alice: outcome bit for that pair


    
class QTCPMessage(Message):
    """Message for the qTCP transport layer."""
 
    __slots__ = ["transfer_id", "share_index", "packet_id",
                 "parent_packet_id", "parent_share_index",
                 "comm_memory_name", "reason", "payload","basis","outcome",
                 "is_qec_layer"]
 
    def __init__(self, msg_type: QTCPMsgType, transfer_id: int,
                 share_index: int = None, packet_id: int = None,
                 parent_packet_id: int = None, parent_share_index: int = None,
                 comm_memory_name: str = None, payload: int = None,
                 reason: "FailureReason" = None,
                 basis: int = None, outcome: int = None,
                  is_qec_layer: bool = False):
        super().__init__(msg_type, QTCP_APP)
        self.transfer_id = transfer_id
        self.share_index = share_index
        self.packet_id = packet_id
        # parent link: set on messages carrying shares of a sub-packet
        # (recursion). None for top-level packets. Lets Bob route reconstructed
        # sub-packets back into the parent packet's aggregation.
        self.parent_packet_id = parent_packet_id
        self.parent_share_index = parent_share_index
        self.comm_memory_name = comm_memory_name   # Bob's memory, for SEND_NOTICE
        self.reason = reason                       # for NACK

        self.payload = payload

        self.basis = basis       # QPING_BASIS: which Pauli basis Bob should measure in
        self.outcome = outcome   # QPING_OUTCOME: Bob's measured bit
        self.is_qec_layer = is_qec_layer   # True: this share is a QEC leaf block
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
    # parent link: set when this transfer is a share of a sub-packet
    # (recursion). None for top-level packet shares.
    parent_packet_id: int = None
    parent_share_index: int = None
    is_qec_layer: bool = False

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
    # parent link: set on records for shares of a sub-packet (recursion).
    # None for top-level packets.
    parent_packet_id: int = None
    parent_share_index: int = None
    is_qec_layer: bool = False

 
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


            # module-level constants
    _RTO_MARGIN = 5          # safety multiple over the bare round-trip; RTO is
                            # diagnostic, so err generous -- it must never fire on
                            # a healthy transfer, only catch a genuine hang
    _RTO_FLOOR = 1_000_000   # ps; floor so a near-zero-latency test topology still
                            # gets a non-trivial timeout instead of ~0
    def __init__(self, node):
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
        self.rto: dict[str,int] = {}            
        self.reserved_at: dict[str, int] = {}

        self.is_testing: dict[str,bool] = {}


        self._last_fire_t = -1

        # --- receiver state ---
        self.bob_transfers: dict[tuple[str, int], BobTransfer] = {}
        self.reserved: int = 0
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
        process = Process(self, "on_reservation_end", [responder])
        event = Event(end_t, process, self.node.timeline.schedule_counter)
        self.node.timeline.schedule(event)


    # ------------------------------------------------------------------
    # sender: the primitive
    # ------------------------------------------------------------------
    def find_available_pair(self, dst: str):
        """Proactively find an already-ENTANGLED comm pair for `dst` that a
        consumer can use right now, without waiting for a fresh get_memory
        edge.

        Needed because get_memory is edge-triggered: a pair that became
        ENTANGLED while no consumer was ready fired its edge already and won't
        fire again. A consumer becoming ready (QPing wanting its next pair, or
        a data transfer just queued) calls this to pick up such a standing
        pair; on None it falls back to waiting for the next edge.

        We do NOT release-to-RAW unwanted pairs (that desyncs the two ends and
        stalls the simulator's time advance) -- we leave them standing and
        fetch them on demand instead.

        Filters: ENTANGLED, in one of our reservations, initiated by us, remote
        end is `dst`, and NOT already bound to an in-flight transfer. The
        last check matters for the parallel initial pair (two shares can be in
        flight at once): without it a second fetch could re-pick a memory the
        first fetch just bound but hasn't yet driven non-ENTANGLED.
        """
        mem_arr = self.node.get_component_by_name(self.node.memo_arr_name)
        mgr = self.node.resource_manager.memory_manager

        in_flight = {t.comm_memory for t in self.transfers.values()
                     if t.status is TransferStatus.IN_FLIGHT
                     and t.comm_memory is not None}

        for memory in mem_arr:
            info = mgr.get_info_by_memory(memory)
            if info.state != "ENTANGLED":
                continue
            if info.index not in self.memo_to_reservation:
                continue
            if self._get_initiator_from_memory_info(self.node, info) != self.node.name:
                continue
            if info.remote_node != dst:
                continue
            if memory in in_flight:
                continue

            return info
        return None
    
    def send_single_qubit(self, data_memory_index: int, dst: str,
                          share_index: int = None, packet_id: int = None,
                          parent_packet_id: int = None,
                          parent_share_index: int = None,
                          is_qec_layer: bool = False) -> int:
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
            packet_id=packet_id,
            parent_packet_id=parent_packet_id,
            parent_share_index=parent_share_index,
            is_qec_layer=is_qec_layer,
        )
        


        self.transfers[transfer_id] = transfer
        self.pending.append(transfer_id)

        log.logger.info(
            f"{self.name}: queued transfer {transfer_id} "
            f"(data_memory={data_memory_index} -> {dst})"
        )

      
        info = self.find_available_pair(dst)

        
        if info is not None:
            self.pending.remove(transfer_id)
            transfer.comm_memory = info.memory          # synchronous claim
            transfer.status = TransferStatus.IN_FLIGHT  # excluded by in_flight set now

            now = self.node.timeline.now()

            # stagger fires within a bounded window past now, so bursts spread
            # across a few ticks but the schedule can't run away from sim time
            # (which under a small memory_size starves the pool and storms edges)
            if self._last_fire_t < now:
                self._last_fire_t = now
            fire_t = self._last_fire_t + 1

            
            self._last_fire_t = fire_t

            proc = Process(self, "_fire", [transfer, info])
            evt = Event(fire_t, proc, self.node.timeline.schedule_counter)

            self.node.timeline.schedule(evt)
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
    def _get_initiator_from_memory_info(self, node, info: "MemoryInfo") -> str:
        """Extracts the reservation initiator's name for a given entangled memory."""
        for rule in node.resource_manager.rule_manager.rules:
            res = getattr(rule, 'reservation', None)
            if not res:
                continue
                
            mem_indices = getattr(rule, 'condition_args', {}).get('memory_indices', [])
            
            if info.index in mem_indices:
                return res.initiator
                
        return None



    def get_memory(self, info: MemoryInfo) -> None:
        """Called when a comm memory changes state. An ENTANGLED comm memory
        belonging to one of our reservations is a resource that either QPing
        (during the quality test) or a pending data transfer can consume.

        A pair that nobody is ready to use right now is NOT left sitting: it is
        released back to RAW so entanglement generation regenerates it and
        fires this handler again later. Leaving it idle would strand the slot --
        an idle ENTANGLED memory does not regenerate (nothing released it) and
        does not re-fire get_memory (the edge already passed), so a slot full of
        declined-idle pairs would stall generation and could deadlock a consumer
        waiting for a fresh edge. Bouncing to RAW keeps every slot cycling, so a
        pair is always moments away whenever someone becomes ready.

        Priority when a pair arrives:
          1. QPing, if the quality test is running and waiting for a pair.
          2. A pending data transfer for this remote node.
          3. Nobody ready -> release to RAW so it regenerates.
        """
        #log.logger.debug(f"{self.name}: get_memory index={info.index} state={info.state}")

        if info.index not in self.memo_to_reservation:
            return
        if info.state != "ENTANGLED":
            return

        node = self.node
        initiator = self._get_initiator_from_memory_info(node, info)
        if initiator is None or initiator != self.node.name:
            return

        # 1. QPing quality test: hand the pair over only if the test is
        #    actively waiting for one. If the test is running but currently
        #    awaiting Bob's outcome (loop is one-pair-at-a-time), this pair is
        #    not wanted yet -- fall through to the release path so it does not
        #    sit idle.
        if self.is_testing.get(info.remote_node, False):
            for obs in self.terminal_observers:
                if hasattr(obs, "qping_wants_pair") and obs.qping_wants_pair(info.remote_node):
                    obs.on_qping_pair(info)
                    return
            # testing, but nobody is ready for a pair right now -> release below

        # 2. Data phase: fire a pending transfer for this remote node.
        else:
            
            for t in self.transfers.values(): #NOTE: Redundant?
                if (t.status is TransferStatus.IN_FLIGHT
                        and t.comm_memory is info.memory):
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
            parent_packet_id=transfer.parent_packet_id,
            parent_share_index=transfer.parent_share_index,
            comm_memory_name=info.remote_memo,
            is_qec_layer=transfer.is_qec_layer,
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
        timeout_event = Event(now + self.rto.get(transfer.dst,self._RTO_FLOOR), timeout_process,
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

        if self.reserved > 0:
            self.reserved -= 1
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
            parent_packet_id=msg.parent_packet_id,
            parent_share_index=msg.parent_share_index,
            is_qec_layer=msg.is_qec_layer,
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

        if msg.packet_id is not None:
            existing = next(
                (r for r in self.bob_transfers.values()
                if r.src == src
                and r.packet_id == msg.packet_id
                and r.share_index == msg.share_index),
                None,
            )
            if existing is not None:
                # Duplicate CANCEL for a share we already have. If the existing record
                # is not yet terminal, transition it to CANCELLED; otherwise no-op.
                if existing.state not in (BobState.CANCELLED, BobState.CONSUMED):
                    if existing.state is BobState.ARRIVED:
                        self.free_data_slot(existing.data_index)
                        existing.data_index = None
                    existing.state = BobState.CANCELLED
                    for obs in self.terminal_observers:
                        if hasattr(obs, "on_bob_transfer_finished"):
                            obs.on_bob_transfer_finished(existing)
                return        
        if transfer is None:
            # Never heard of it -- notice never arrived, or Alice never fired
            # (swept from PENDING at reservation end). Synthesize a CANCELLED
            # record so observers see a terminal event and can settle any
            # per-packet aggregation.
            self.bob_transfers[(src, tid)] = BobTransfer(
                transfer_id=tid,
                src=src,
                comm_memory_name=None,
                state=BobState.CANCELLED,
                share_index=msg.share_index,
                packet_id=msg.packet_id,
                parent_packet_id=msg.parent_packet_id,
                parent_share_index=msg.parent_share_index,
                is_qec_layer=msg.is_qec_layer,
            )
            for obs in self.terminal_observers:
                if hasattr(obs, "on_bob_transfer_finished"):
                    obs.on_bob_transfer_finished(self.bob_transfers.get((src, tid)))
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
                if hasattr(obs, "on_bob_transfer_finished"):
                    obs.on_bob_transfer_finished(self.bob_transfers.get((src, tid)))
            

        log.logger.debug(f"{self.name}: transfer {tid} cancelled")
    
    #Alice:
    def _on_ack(self, src: str, msg) -> None:
        transfer = self.transfers.get(msg.transfer_id)
        if transfer is None or transfer.status in (TransferStatus.DELIVERED,
                                               TransferStatus.FAILED):
            return

        if src in self.reserved_at:
            self.reserved_at[src] = max(0, self.reserved_at[src] - 1)

        self._finish(transfer, TransferStatus.DELIVERED)

    def _on_nack(self, src: str, msg) -> None:
        transfer = self.transfers.get(msg.transfer_id)
        if transfer is None or transfer.status in (TransferStatus.DELIVERED,
                                               TransferStatus.FAILED):
            return
        self._finish(transfer, TransferStatus.FAILED, msg.reason)

    def _on_mem_req(self, src: str, msg: QTCPMessage) -> None:
        requested = msg.payload
        if requested is None:
            log.logger.warning(f"{self.name}: MEM_REQ from {src} missing payload")
            return

        if len(self.free_data_slots) - self.reserved >= requested:
            self.reserved += requested
            
            # Auto-reply with ACCEPT
            tid = self.mint_transfer_id()
            self.node.send_message(
                src, 
                QTCPMessage(QTCPMsgType.MEM_ACCEPT, transfer_id=tid)
            )
            
            # Wake up Bob's handshake to start the Quality Assessment
            for obs in self.terminal_observers:
                if hasattr(obs, "on_inbound_connection_accepted"):
                    obs.on_inbound_connection_accepted(src, requested)
        else:
            # Auto-reply with REJECT
            tid = self.mint_transfer_id()
            self.node.send_message(
                src, 
                QTCPMessage(QTCPMsgType.MEM_REJECT, transfer_id=tid)
            )

    def _on_mem_accept(self, src: str, msg: QTCPMessage) -> None:
        # Wake up Alice's handshake layer to trigger the reservation / fidelity test
        for obs in self.terminal_observers:
            if hasattr(obs, "on_mem_accept"):
                obs.on_mem_accept(src)

    def _on_mem_reject(self, src: str, msg: QTCPMessage) -> None:
        # Wake up Alice's handshake layer to gracefully abort
        for obs in self.terminal_observers:
            if hasattr(obs, "on_mem_reject"):
                obs.on_mem_reject(src)

    

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


        # --- New Dispatches ---
        elif msg.msg_type is QTCPMsgType.MEM_REQ:
            self._on_mem_req(src, msg)
        elif msg.msg_type is QTCPMsgType.MEM_ACCEPT:
            self._on_mem_accept(src, msg)
        elif msg.msg_type is QTCPMsgType.MEM_REJECT:
            self._on_mem_reject(src, msg)

        elif msg.msg_type is QTCPMsgType.CLOSE:
            if msg.payload is not None:
                self.reserved = max(0, self.reserved - msg.payload)
            log.logger.info(
                        f"Window coming from {src} is closed."
                        f" Current reserved: {self.reserved}"
                    )

        elif msg.msg_type is QTCPMsgType.QPING_BASIS:
            for obs in self.terminal_observers:
                if hasattr(obs, "on_qping_basis"):
                    obs.on_qping_basis(src, msg)

        elif msg.msg_type is QTCPMsgType.QPING_OUTCOME:
            for obs in self.terminal_observers:
                if hasattr(obs, "on_qping_outcome"):
                    obs.on_qping_outcome(src, msg)

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
                parent_packet_id=transfer.parent_packet_id,
                parent_share_index=transfer.parent_share_index,
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
            f"data memory {data_index}. Remaining data slots: {len(self.free_data_slots)}"
        )

        self.node.send_message(
            transfer.src,
            QTCPMessage(QTCPMsgType.ACK, transfer_id=transfer.transfer_id),
        )
        for obs in self.terminal_observers:
            if hasattr(obs, "on_bob_transfer_finished"):
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
            if hasattr(obs, "on_alice_transfer_finished"):
                obs.on_alice_transfer_finished(transfer)
    
    #------------------------------------------------------------------
    #Reservation sweep
    #------------------------------------------------------------------
    def on_reservation_end(self, responder: str) -> None:
        """Diagnostic failsafe. Fires at end_t.

        Under the intended model (handshake-verified channel quality, single
        connection at a time, reservation windows long enough for the work
        queued into them), self.pending should hold no transfers for this
        responder at end_t. Any still queued for it signals that the assumed
        model does not hold in this run: connection was too short, channel
        quality is worse than the handshake indicated, or the layer above
        miscounted somewhere.

        If it does fire, terminate the stuck transfers as NO_ENTANGLEMENT (they
        never got a pair and never will now) and CANCEL to Bob so his
        aggregation does not deadlock waiting on shares that will not arrive.

        Only transfers bound for `responder` are swept. self.pending is shared
        across all of this node's reservations, so clearing it wholesale would
        strand still-valid transfers queued for other destinations whose
        windows are still open.

        Kept as a failsafe because the cost is zero when nothing is pending for
        this responder and a hang is much harder to diagnose than a warning
        line.
        """
        if responder not in self.reserved_at:
            # Already torn down by the reject path -- no reservation left to close.
            return

        unused = self.reserved_at.pop(responder)
        tid = self.mint_transfer_id()
        log.logger.info(
                    f"{unused} slots of reservation unused."
                    f" Sending message to {responder}"
                )
        self.node.send_message(
            responder,
            QTCPMessage(QTCPMsgType.CLOSE, transfer_id=tid, payload=unused),
        )

        # Sweep ONLY transfers for this responder -- not the whole shared queue.
        to_sweep = [tid for tid in self.pending
                    if self.transfers[tid].dst == responder]
        to_sweep += [tid for tid, t in self.transfers.items()
             if t.dst == responder
             and t.status is TransferStatus.IN_FLIGHT
             and tid not in to_sweep]

        for obs in self.terminal_observers:
            if hasattr(obs, "on_connection_closed"):
                obs.on_connection_closed(responder)


        
        if to_sweep:
            log.logger.warning(
                f"{self.name}: reservation-end sweep found {len(to_sweep)} "
                f"PENDING transfer(s) for {responder} -- this should not happen "
                f"under the intended model. Terminating as NO_ENTANGLEMENT."
            )
            for transfer_id in to_sweep:
                if transfer_id in self.pending:
                    self.pending.remove(transfer_id)
                transfer = self.transfers.get(transfer_id)
                if transfer is None or transfer.status not in (TransferStatus.PENDING, TransferStatus.IN_FLIGHT):
                    continue
                self.node.send_message(
                    transfer.dst,
                    QTCPMessage(
                        QTCPMsgType.CANCEL,
                        transfer_id=transfer_id,
                        packet_id=transfer.packet_id,
                        share_index=transfer.share_index,
                        parent_packet_id=transfer.parent_packet_id,
                        parent_share_index=transfer.parent_share_index,
                    ),
                )
                self._finish(transfer, TransferStatus.FAILED,
                             FailureReason.NO_ENTANGLEMENT)

        # Normal-path teardown notification: let observers reset per-dst state so
        # `responder` can be connected to again. (The reject path resets its own
        # state and never reaches here, thanks to the reserved_at guard above.)

    def measure_comm_in_basis(self, memory, circuit) -> int:
        """Measure a comm memory's qubit with the given single-qubit basis
        circuit (from qping.measurement_circuit) and return the outcome bit.
        Consumes the pair -- the entanglement is destroyed by the measurement
        -- then releases the comm memory so the window regenerates it.
        Used by the handshake's QPing loop; not part of the data path."""
        key = memory.qstate_key
        rnd = self.node.get_generator().random()
        meas = self.node.timeline.quantum_manager.run_circuit(circuit, [key], rnd)
        outcome = int(meas[key])
        self.node.resource_manager.update(None, memory, MemoryInfo.RAW)
        return outcome





    def _one_way_classical_delay(self, dst: str) -> int:
        """One-way classical propagation delay from this node to `dst`, in ps.

        Current model: requires a DIRECT classical channel to `dst` and returns its
        delay. qTCP's classical traffic (SEND_NOTICE, ACK, MEM_*, QPING_*) is
        end-to-end, so a direct channel is assumed to exist; connect() validates
        this. To support multi-hop classical forwarding later, this is the ONLY
        function that changes: walk the classical route summing per-hop delays.
        """
        cchannels = self.node.cchannels
        ch = cchannels.get(dst)
        if ch is None:
            raise ValueError(
                f"{self.name}: no direct classical channel to {dst}; qTCP currently "
                f"requires one. (Multi-hop classical routing is future work.)"
            )
        return ch.delay


    def compute_rto(self, dst: str) -> int:
        """Diagnostic retransmission timeout for transfers to `dst`, in ps.

        NOT a retransmit timer -- this layer cannot resend (the qubit is consumed at
        the Bell measurement; recovery is the overseer's QSS recursion/restart).
        RTO only bounds how long a fired-but-unresolved transfer waits before
        on_timeout reports it as NO_ACK, which the overseer then treats as a failed
        share. So it must be comfortably longer than a healthy round-trip and short
        enough to surface a real hang.

        Round-trip a healthy transfer needs: SEND_NOTICE (Alice->Bob) + teleport
        MEASUREMENT_RESULT (Alice->Bob) + ACK (Bob->Alice). That's ~2x the one-way
        classical delay plus teleport-protocol overhead and Bob's processing, none
        of which we model precisely -- the margin absorbs them.
        """
        one_way = self._one_way_classical_delay(dst)
        rto = self._RTO_MARGIN * 2 * one_way
        self.rto[dst] =  max(rto, self._RTO_FLOOR)
        return self.rto[dst]

    
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