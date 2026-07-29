from dataclasses import dataclass, field
from enum import Enum, auto

from sequence.app.qtcp_transfer import (
    QTCPTransfer, Transfer, BobTransfer,
    QTCPMessage, QTCPMsgType,
    TransferStatus, BobState,
)
import sequence.app.qping as qping
from sequence.utils import log

@dataclass
class ConnectionConfig:
    """Holds reservation parameters while waiting for MEM_ACCEPT."""
    start_t: int
    end_t: int
    memory_size: int
    target_fidelity: float
    payload: int = 20


class HandshakeState(Enum):
    CLOSED = auto()
    MEM_REQ_SENT = auto()               # Initiator asked for memory, waiting for reply
    FIDELITY_TESTING = auto()           # Initiator is firing test qubits
    ESTABLISHED = auto()                # Window is open, token is actively passing

class QPingPhase(Enum):
    WAITING_FOR_PAIR = auto()      # ready to consume the next entangled pair
    WAITING_FOR_OUTCOME = auto()   # measured a pair, announced basis, awaiting Bob's outcome

@dataclass
class QPingSession:
    """Alice-side state for one connection's sequential fidelity test.

    Loop shape (b): exactly one pair outstanding at a time. On each pair Alice
    picks a random basis, measures her half immediately, announces the basis to
    Bob, and waits for his outcome. When it arrives she scores the correlation,
    updates the Beta-Bernoulli counts, and asks qping.decide whether to accept,
    reject, or sample another pair.
    """
    dst: str
    phase: QPingPhase = QPingPhase.WAITING_FOR_PAIR
    passes: int = 0
    trials: int = 0
    # the one outstanding pair's data, set at announce, cleared at score
    current_basis: "qping.PauliBasis" = None
    current_alice_outcome: int = None
    # config (constants for the first cut)
    f0: float = 0.5
    eta: float = 0.95
    max_pairs: int = 100



class QTCPHandshake:
    """State machine and session manager for qTCP.
    
    Orchestrates connection establishment, fidelity testing, and token passing.
    Sits directly above QTCPTransfer, acting as an observer for transfer events.
    """
    def __init__(self, transfer_layer: "QTCPTransfer"):
        self.transfer = transfer_layer
        self.node = self.transfer.node
        self.name = f"{self.node.name}.QTCPHandshake"
        

        # Register as an observer to catch the MEM_ACCEPT / REJECT callbacks
        self.transfer.terminal_observers.append(self)
        
        self.states: dict[str, HandshakeState] = {}
        self.pending_configs: dict[str, ConnectionConfig] = {}
        self.qping_sessions: dict[str, QPingSession] = {}

        log.logger.debug(f"{self.name}: initialized")

    # ------------------------------------------------------------------
    # Application Interface
    # ------------------------------------------------------------------
    def connect(self, dst: str, start_t: int, end_t: int, 
                memory_size: int, payload: int) -> None:
        """Called by the wrapper layer (QTCP) to initiate a session."""
        current_state = self.states.get(dst, HandshakeState.CLOSED)
        
        if current_state != HandshakeState.CLOSED:
            log.logger.info(
                f"{self.name}: Connection to {dst} already in state {current_state.name}. "
                f"Absorbing redundant request."
            )
            return
            
        self.states[dst] = HandshakeState.MEM_REQ_SENT
        
        # Store the reservation parameters to use later in on_mem_accept
        self.pending_configs[dst] = ConnectionConfig(
            start_t, end_t, memory_size, 0.01, payload
        )

        
        tid = self.transfer.mint_transfer_id()
        msg = QTCPMessage(QTCPMsgType.MEM_REQ, transfer_id=tid, payload=payload)
    

        self.node.send_message(dst, msg)
        
        log.logger.debug(f"{self.name}: Sent MEM_REQ for {memory_size} slots to {dst}")

    # ------------------------------------------------------------------
    # Observer Callbacks (Driven by QTCPTransfer)
    # ------------------------------------------------------------------
    def on_inbound_connection_accepted(self, src: str, requested_memory: int) -> None:
        """Bob's side: QTCPTransfer automatically accepted a MEM_REQ.
        
        Bob must transition to FIDELITY_TESTING_RECEIVER and wait for Alice's test qubits.
        """
        log.logger.info(f"{self.name}: Accepted inbound connection from {src}. Awaiting preamble.")

    def on_mem_accept(self, src: str) -> None:
        """Alice's side: Bob accepted. Retrieve the config and request the window."""
        self.states[src] = HandshakeState.FIDELITY_TESTING
        
        # Pop the config out of the dictionary (we only need it once)
        config = self.pending_configs.pop(src, None)
        if config is None:
            log.logger.error(f"{self.name}: MEM_ACCEPT from {src} but no pending config found.")
            return

        log.logger.info(f"{self.name}: {src} accepted MEM_REQ. Requesting reservation window.")

        self.transfer.reserved_at[src] = config.payload
        # Tell RequestApp to submit the reservation to the network
        self.transfer.start(
            src, 
            config.start_t, 
            config.end_t, 
            config.memory_size, 
            config.target_fidelity
        )
       
        self._start_fidelity_test(src)
   

    def on_mem_reject(self, src: str) -> None:
        """Alice's side: Bob denied the request. Abort session."""
        self.states[src] = HandshakeState.CLOSED
        self.pending_configs.pop(src, None)  # Clean up memory
        log.logger.warning(f"{self.name}: {src} rejected MEM_REQ. Session aborted.")


    def on_alice_transfer_finished(self, transfer: "Transfer") -> None:
        """Observer callback from QTCPTransfer. Evaluates if we must yield the token."""
        pass

    def on_bob_transfer_finished(self, transfer: "BobTransfer") -> None:
        """Triggered when Bob successfully receives a qubit.
        
        If Bob is in FIDELITY_TESTING_RECEIVER state, he intercepts this qubit, 
        measures it for the fidelity test, and checks if he has received enough to score.
        Otherwise, he passes it up to the application.
        """
        pass


    # ------------------------------------------------------------------
    # Fidelity Internal Helpers
    # ------------------------------------------------------------------
    def _qping_try_next_pair(self, dst: str) -> None:
        """QPing became ready for its next pair. Grab a standing entangled pair
        if one exists; otherwise stay in WAITING_FOR_PAIR and let the next
        get_memory edge deliver one via on_qping_pair."""
        info = self.transfer.find_available_pair(dst)
        if info is not None:
            self.on_qping_pair(info)


    def _start_fidelity_test(self, dst: str) -> None:

        """Begin the QPing quality test. Flip the transfer layer into testing
        mode and open a session in WAITING_FOR_PAIR. Nothing is sent now -- the
        test is passive until the first entangled pair arrives via get_memory,
        which (seeing is_testing and qping_wants_pair) routes it to
        on_qping_pair. Bob needs no start signal; he answers QPING_BASIS
        messages reactively.
        """



        self.states[dst] = HandshakeState.FIDELITY_TESTING
        self.qping_sessions[dst] = QPingSession(dst=dst)
        self.transfer.is_testing = True
        log.logger.info(f"{self.name}: QPing quality test started for {dst}")
        self._qping_try_next_pair(dst)

    def qping_wants_pair(self, dst: str) -> bool:
        """get_memory asks this before handing a pair to on_qping_pair. True
        only if a session for dst is actively waiting for its next pair (loop
        is one-pair-at-a-time; while awaiting Bob's outcome we do not want a new
        pair, and get_memory releases it to RAW instead)."""
        session = self.qping_sessions.get(dst)
        return (session is not None
                and session.phase is QPingPhase.WAITING_FOR_PAIR)

    def on_qping_pair(self, info) -> None:
        dst = info.remote_node
        session = self.qping_sessions.get(dst)
        if session is None or session.phase is not QPingPhase.WAITING_FOR_PAIR:
            return

        # Capture the remote memory name BEFORE measuring -- measure_comm_in_basis
        # releases the pair to RAW, which clears info.remote_memo.
        remote_memo = info.remote_memo

        gen = self.node.get_generator()
        basis = gen.choice(list(qping.PauliBasis))
        circuit = qping.measurement_circuit(basis)
        alice_outcome = self.transfer.measure_comm_in_basis(info.memory, circuit)

        session.current_basis = basis
        session.current_alice_outcome = alice_outcome
        session.phase = QPingPhase.WAITING_FOR_OUTCOME

        tid = self.transfer.mint_transfer_id()
        self.node.send_message(
            dst,
            QTCPMessage(
                QTCPMsgType.QPING_BASIS,
                transfer_id=tid,
                comm_memory_name=remote_memo,   # captured before the release
                basis=basis.value,
            ),
        )
        log.logger.debug(
            f"{self.name}: QPing announced basis {basis.name} to {dst} "
            f"(pair on {remote_memo}), alice_outcome={alice_outcome}"
        )

    def on_qping_basis(self, src: str, msg) -> None:
        """Bob: measure his half of the named pair in the announced basis and
        report the outcome. Stateless -- no session on Bob's side."""
        memory = self.transfer._memory_by_name(msg.comm_memory_name)
        if memory is None:
            log.logger.warning(
                f"{self.name}: QPING_BASIS for unknown memory "
                f"{msg.comm_memory_name} from {src}; ignoring"
            )
            return

        basis = qping.PauliBasis(msg.basis)
        circuit = qping.measurement_circuit(basis)
        bob_outcome = self.transfer.measure_comm_in_basis(memory, circuit)

        tid = self.transfer.mint_transfer_id()
        self.node.send_message(
            src,
            QTCPMessage(
                QTCPMsgType.QPING_OUTCOME,
                transfer_id=tid,
                outcome=bob_outcome,
            ),
        )
    def on_qping_outcome(self, src: str, msg) -> None:
        """Alice: Bob's outcome for the outstanding pair arrived. Score the
        correlation, update counts, and apply the sequential decision rule.
        CONTINUE -> flip back to WAITING_FOR_PAIR (the next get_memory edge
        supplies a pair). ACCEPT/REJECT -> conclude the test.
        """
        session = self.qping_sessions.get(src)
        if session is None or session.phase is not QPingPhase.WAITING_FOR_OUTCOME:
            # Late/stray outcome for a concluded test -- ignore.
            return

        passed = qping.is_pass(
            session.current_basis,
            session.current_alice_outcome,
            msg.outcome,
        )
        session.trials += 1
        if passed:
            session.passes += 1

        session.current_basis = None
        session.current_alice_outcome = None

        verdict = qping.decide(
            passes=session.passes,
            trials=session.trials,
            f0=session.f0,
            eta=session.eta,
            max_pairs=session.max_pairs,
        )

        log.logger.debug(
            f"{self.name}: QPing {src} trial {session.trials} "
            f"(passes {session.passes}) -> {verdict.name}"
        )

        if verdict is qping.QPingVerdict.CONTINUE:
            session.phase = QPingPhase.WAITING_FOR_PAIR
            self._qping_try_next_pair(src)
            return

        # terminal: stop testing, act on the verdict
        self.transfer.is_testing = False
        self.qping_sessions.pop(src, None)

        if verdict is qping.QPingVerdict.ACCEPT:
            self.states[src] = HandshakeState.ESTABLISHED
            log.logger.info(
                f"{self.name}: QPing ACCEPT for {src} after {session.trials} "
                f"pairs ({session.passes} passed). State -> ESTABLISHED"
            )
        else:
            self.states[src] = HandshakeState.CLOSED
            log.logger.warning(
                f"{self.name}: QPing REJECT for {src} after {session.trials} "
                f"pairs ({session.passes} passed). Closing."
            )
            self._reject_close(src)

    def _reject_close(self, dst: str) -> None:
        """Reject path: release Bob's reservation immediately rather than
        waiting for the sweep at end_t. No data was sent, so the full
        reservation is unused -- report it all. Reuses the CLOSE message the
        sweep would otherwise send; Bob's dispatch subtracts the payload from
        his reserved count.
        """
        unused = self.transfer.reserved_at.pop(dst, 0)
        tid = self.transfer.mint_transfer_id()
        self.node.send_message(
            dst,
            QTCPMessage(QTCPMsgType.CLOSE, transfer_id=tid, payload=unused),
        )
        for rule in self.node.resource_manager.rule_manager.rules:
            if getattr(rule, "reservation", None) and \
               rule.reservation.responder == dst:
                self.node.resource_manager.rule_manager.expire(rule)
                log.logger.debug(
                    f"{self.name}: expired reservation rule for {dst} on reject"
                )
                break