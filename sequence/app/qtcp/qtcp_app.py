"""qTCP application wrapper: the single entry point a user talks to.

This is the thin top of the qTCP stack. It owns one instance each of the three
layers and wires them together, then exposes three operations:

    connect(dst, start_t, end_t, memory_size, num_qubits)
    send_packet(data_memory_index, dst) -> packet_id
    get_received_packet(src, packet_id) -> state | None

Everything below is delegation. The wrapper's only real logic is:

  1. Translating the user's unit (qubits) into the reservation's unit (data
     slots) via the worst-case payload formula, so the reservation Bob grants
     is guaranteed large enough for every share the delivery can ever hold at
     once -- including recursion.

  2. Counting packets per destination so a user cannot send more qubits than
     they reserved for. The reservation was sized for `num_qubits`; the
     (num_qubits + 1)-th send is a programming error, caught here rather than
     blowing up later as a slot-exhaustion assertion deep in the transfer
     layer.

Design notes:
  * fidelity is fixed at 0.01 (a floor that never trips the reservation's own
    fidelity gate) -- qTCP's real channel-quality decision is QPing, inside the
    handshake, not this parameter. So it is not exposed.
  * rto is derived per-destination from the classical path delay inside the
    transfer layer (compute_rto), driven from the handshake at connect time --
    not a user parameter.
  * A send during the handshake is safe: shares queue in the transfer layer and
    fire only once the connection reaches the data phase (the is_testing gate).
    So the wrapper does NOT reject sends before ESTABLISHED -- it lets them
    queue.
"""

import sequence.app.qtcp.qss as qss
from sequence.app.qtcp.qtcp_transfer import QTCPTransfer
from sequence.app.qtcp.qtcp_overseer import QTCPOverseer
from sequence.app.qtcp.qtcp_handshake import QTCPHandshake
from sequence.utils import log


class QTCPApp:
    """Top-level qTCP interface for a node.

    Owns the three qTCP layers for one node and delegates to them. Both
    endpoints of a connection run their own QTCPApp; the initiator calls
    connect() then send_packet(), the responder answers reactively through the
    layers below.

    Attributes:
        node: the quantum node this app is attached to.
        transfer (QTCPTransfer): the delivery/detector layer.
        overseer (QTCPOverseer): the QSS encode/reconstruct/recovery layer.
        handshake (QTCPHandshake): connection setup + QPing quality gate.
        packets_reserved (dict[str, int]): per-destination qubit budget declared
            at connect().
        packets_sent (dict[str, int]): per-destination count of send_packet
            calls, checked against the budget.
    """

    def __init__(self, node, max_recursion_depth: int = 1,
                 max_restarts: int = 1, on_exhaust: str = "recover"):
        """Build and wire the three qTCP layers for `node`.

        Args:
            node: the quantum node.
            max_recursion_depth: cap on QSS recursion depth (passed to the
                overseer). Also sets the payload worst case: a single qubit can
                occupy N_SHARES + (N_SHARES - 1) * depth data slots at once.
            max_restarts: cap on local restarts per packet (overseer policy).
            on_exhaust: overseer policy when recursion + restart are exhausted;
                one of "recover" | "fire" | "raise".
        """
        self.node = node
        self.max_recursion_depth = max_recursion_depth

        # Build the stack bottom-up: transfer first, then the two layers that
        # observe it.
        self.transfer = QTCPTransfer(node)
        self.overseer = QTCPOverseer(
            self.transfer,
            max_recursion_depth=max_recursion_depth,
            max_restarts=max_restarts,
            on_exhaust=on_exhaust,
        )
        self.handshake = QTCPHandshake(self.transfer)

        # Per-destination send accounting (qubit units, user-facing).
        self.packets_reserved: dict[str, int] = {}
        self.packets_sent: dict[str, int] = {}
        self.handshake.state_observers.append(self)

        self.established: set[str] = set()
        self.pending_sends: dict[str, list[tuple[int,int]]] = {}

        log.logger.debug(f"{node.name}.QTCPApp: initialized")

    # ------------------------------------------------------------------
    # payload sizing
    # ------------------------------------------------------------------
    def _payload_for(self, num_qubits: int) -> int:
        """Worst-case data-slot footprint for delivering `num_qubits` qubits.

        Per qubit, peak simultaneous data-slot occupancy is:

            N_SHARES                     (the top-level packet's shares)
          + (N_SHARES - 1) * depth       (one active sub-packet per recursion
                                          level; the recursing share's own slot
                                          is reused, hence -1)

        Two shares of the same packet can each recurse, but never at the same
        time -- one sub-packet fully resolves and frees its slots before the
        next recurses -- so peak occupancy is one active sub-packet, not two.
        Local restart re-encodes within a sub-packet's own slot budget, so it
        does not add to the peak either.

        Packets are delivered bounded-parallel and multiple qubits can be in
        flight at once (if memory_size allows), so the reservation must cover
        all `num_qubits` peaking simultaneously: multiply.
        """
        per_qubit = 4+ qss.N_SHARES + (qss.N_SHARES - 1) * self.max_recursion_depth
        return num_qubits * per_qubit

    # ------------------------------------------------------------------
    # user interface
    # ------------------------------------------------------------------
    def connect(self, dst: str, start_t: int, end_t: int,
                memory_size: int, num_qubits: int) -> None:
        """Open a connection to `dst` sized for `num_qubits` qubits.

        Computes the worst-case data-slot payload and hands the reservation
        request to the handshake, which negotiates memory with Bob, opens the
        reservation window [start_t, end_t] of entanglement width memory_size,
        and runs the QPing quality test before entering the data phase.

        Returns immediately -- the connection is not ESTABLISHED until QPing
        accepts, which happens later in simulated time. It is safe to call
        send_packet right after connect(): shares queue in the transfer layer
        and fire once the data phase opens.

        Raises:
            ValueError: if there is no direct classical channel to `dst`
                (surfaced early by the handshake via compute_rto), since qTCP's
                classical traffic is end-to-end and currently requires one.
        """

        payload = self._payload_for(num_qubits)

        self.packets_reserved[dst] = num_qubits
        self.packets_sent[dst] = 0

        log.logger.info(
            f"{self.node.name}.QTCPApp: connect to {dst} for {num_qubits} "
            f"qubit(s) (payload {payload} slots, width {memory_size}, "
            f"window [{start_t}, {end_t}])"
        )

        self.handshake.connect(
            dst=dst,
            start_t=start_t,
            end_t=end_t,
            memory_size=memory_size,
            payload=payload,
        )

    def send_packet(self, data_memory_index: int, dst: str) -> int:
        """Send the single-qubit secret in `data_memory_index` to `dst`.

        Guards against overshooting the reservation: the connection to `dst`
        was sized for a fixed number of qubits at connect(); sending more than
        that would exhaust Bob's reserved slots and trip a slot-allocation
        assertion in the transfer layer. Refuse it here instead, with a clear
        error.

        Returns the overseer's packet id.

        Raises:
            RuntimeError: if send_packet is called for `dst` without a prior
                connect(), or beyond the reserved qubit budget.
        """
        if dst not in self.packets_reserved:
            raise RuntimeError(
                f"{self.node.name}.QTCPApp: send_packet to {dst} with no "
                f"connection; call connect() first."
            )
        if self.packets_sent[dst] >= self.packets_reserved[dst]:
            raise RuntimeError(
                f"{self.node.name}.QTCPApp: send_packet to {dst} exceeds the "
                f"reserved budget of {self.packets_reserved[dst]} qubit(s)."
            )

        # Budget is consumed at request time, whether we forward or hold, so the
        # (num_qubits + 1)-th request is caught even before the connection opens.
        self.packets_sent[dst] += 1
        packet_id = self.overseer.mint_packet_id()

        if dst in self.established:
            return self.overseer.send_packet(data_memory_index, dst, packet_id=packet_id)

        # Handshake still in progress -> hold until established.
        self.pending_sends.setdefault(dst, []).append((packet_id, data_memory_index))

        return packet_id

    def get_received_packet(self, src: str, packet_id: int):
        """Return the reconstructed secret state for (src, packet_id), or None
        if that packet has not been reconstructed. Passthrough to the overseer;
        primarily for testing."""
        return self.overseer.get_received_packet(src, packet_id)

    def on_connection_established(self, dst: str) -> None:
        """Handshake reached ESTABLISHED for dst: flush every held send, then
        mark dst so future sends forward immediately."""
        self.established.add(dst)
        held = self.pending_sends.pop(dst, [])
        for packet_id, data_memory_index in held:
            self.overseer.send_packet(data_memory_index, dst, packet_id=packet_id)
        if held:
            log.logger.info(
                f"{self.node.name}.QTCPApp: flushed {len(held)} held send(s) "
                f"to {dst} on connection established"
            )

    def on_connection_rejected(self, dst: str) -> None:
        """Handshake failed for dst (MEM_REJECT or QPing REJECT): discard held
        sends. They never entered the transfer layer, so nothing to cancel --
        just drop them and free the budget."""
        held = self.pending_sends.pop(dst, [])
        # Roll back the budget consumed by the discarded requests, and clear the
        # reservation record so a later connect() to dst starts clean.
        self.packets_sent.pop(dst, None)
        self.packets_reserved.pop(dst, None)
        self.established.discard(dst)
        if held:
            log.logger.warning(
                f"{self.node.name}.QTCPApp: connection to {dst} rejected; "
                f"discarded {len(held)} held send(s)"
            )
    def on_connection_closed(self, dst: str) -> None:
        """Normal end-of-window teardown (from the transfer layer at end_t).
        Drop dst from established so a future send_packet holds through the next
        handshake instead of forwarding immediately, and clear the send
        accounting so a reconnect starts with a fresh budget."""
        held = self.pending_sends.get(dst, [])
        n_held = len(held)
 
        if n_held > 0:
            was_established = dst in self.established  # adjust to your accessor
            if not was_established:
                # Never established -> QPing never completed -> the usual cause
                # is that entanglement never formed. The most common reason for
                # that is an over-large memory_size: the reservation asked for
                # more concurrent pairs than comm memory can supply, so it
                # produced NONE (all-or-nothing) and starved the channel. A
                # too-short window is the other possibility.
                log.logger.warning(
                    f"{self.node.name}: connection to {dst} closed with {n_held} "
                    f"unflushed send(s) -- it never established, so no qubits "
                    f"were sent. This usually means no entanglement was "
                    f"generated: check that the reservation's memory_size fits "
                    f"the available comm memory (an over-large memory_size "
                    f"produces no pairs at all rather than fewer), and that the "
                    f"window (start_t..end_t) is long enough for QPing to "
                    f"complete."
                )
            else:
                # Established but still holding sends at close -- shouldn't
                # normally happen (flush runs on establish). Report distinctly
                # so it is not misattributed to comm starvation.
                log.logger.warning(
                    f"{self.node.name}: connection to {dst} closed with {n_held} "
                    f"unflushed send(s) despite having established -- these "
                    f"sends did not go out before end_t. The window likely "
                    f"closed before delivery could start; consider a longer "
                    f"end_t."
                )



        self.established.discard(dst)
        self.packets_sent.pop(dst, None)
        self.packets_reserved.pop(dst, None)
        self.pending_sends.pop(dst, None)
        log.logger.debug(f"{self.node.name}.QTCPApp: connection to {dst} closed")


