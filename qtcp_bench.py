"""
qTCP benchmarking runner -- all five metrics from one set of trials.

USAGE
    python qtcp_bench.py --pilot          # ~5 min, validates plumbing, finds the knee
    python qtcp_bench.py --full           # the real run, resumable
    python qtcp_bench.py --full --workers 32

One trial = one send_packet through the full stack. Every trial writes one fat
row (see ROW_FIELDS); every metric is an aggregation over those rows. Rows are
appended to CSV as they complete, so a crash loses at most the in-flight trials
and a rerun skips what is already done.

TWO ARMS, per the realism table:
  ideal      gate/meas fidelity = 1.0, ent-gen swept   -> metric 2 (QPing knee)
  realistic  gate/meas fidelity < 1.0, ent-gen swept   -> metrics 1, 3, 4, 5

INPUT STATES: most trials use Haar-random states (correct for metric 1's
average). A block at each point uses FIXED stabiliser states, because a clean
logical Pauli error averaged over random inputs gives mean fidelity 1/3 with a
smooth spread -- which mimics "partial failure" and would make the metric-1 vs
metric-2 comparison unanswerable. Fixed states collapse it to a sharp 0/1.

ASSUMPTIONS -- all checked by preflight(), which fails loudly in the first
seconds rather than silently three hours in:
  A1  QPing threshold is set to 0 (passive) OUTSIDE this script. If QPing still
      gates, low-fidelity points measure where you configured it to refuse, not
      where delivery degrades, and the metric-2 knee becomes circular.
  A2  gate_fidelity / measurement_fidelity are per-NODE keys in the config dict
      (matching `node.get(Topo.GATE_FIDELITY, 1)` in _add_nodes).
  A3  entanglement generation fidelity is templates.<tpl>.MemoryArray.fidelity.
  A4  the cheat-swap has been reverted to a real circuit call, so gate fidelity
      actually reaches the swap.
  A5  one entry in transfer.metrics == one teleportation == one entangled pair.
  A6  get_received_packet returns a 2-amplitude vector for a delivered qubit.
"""

import argparse
import copy
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np

# ----------------------------------------------------------------------------
# CONFIGURATION -- edit these
# ----------------------------------------------------------------------------

BASE_CONFIG_PATH = "tmp/three_node.json"

ALICE = "router_0"
BOB = "router_1"          # ADJACENT. router_2 would be two hops via a swap.
CHARLIE = "router_2"

MAX_DEPTH = 1
RESTART_AMOUNT = 1
FALLBACK_MODE = "recover"

CONNECT_START_MS = 20
CONNECT_END_MS = 80
MEMORY_SIZE = 5
NUM_QUBITS = 1

# Gate/measurement noise mode.
#   "native"   -- pass gate_fid/meas_fid through the config and rely on
#                 SeQUeNCe to apply them. On ket_vector this is a NO-OP: those
#                 parameters are only read by the _bds swapping/purification
#                 paths, and we run neither. Confirmed by preflight [4].
#   "injected" -- apply the noise at the quantum manager using SeQUeNCe's own
#                 average-fidelity -> Pauli-probability conversion.
GATE_NOISE_MODE = "injected"

# True: one error draw per GATE per qubit (physically right -- a 10-gate
# circuit gets ~10x the error of a 1-gate circuit).
# False: one draw per run_circuit CALL, which understates deep circuits.
PER_GATE_NOISE = True

# Realistic-arm gate fidelity. 0.999 is today's best production two-qubit
# fidelity (Quantinuum H-series class) and sits mid-transition for this
# protocol: ~0.7 expected error events across a ~464-gate packet. Anything at
# 0.99 puts you at ~6 errors and flatlines the arm; 0.9999 puts you at ~0.06
# and nothing fails.
REALISTIC_GATE_FID = 0.999

# Measurement idealised. Justification for the report: readout contributes
# under 10% of error events at these gate fidelities (gate events outnumber
# misreads ~12:1), and repeated readout with majority vote is standard practice
# at the hardware layer. State it as an assumption, and note that QARQ applied
# depolarizing noise uniformly including measurement -- so this is slightly
# optimistic on an axis where they were not.
REALISTIC_MEAS_FID = 1.0

# Ent-gen fidelity sweep points.
# PILOT is deliberately wide and coarse: its job is to locate the knee.
PILOT_ENT_FIDS = [1.0, 0.99, 0.95, 0.90, 0.80, 0.70]
PILOT_N = 20

# FULL grid, revised from preflight: check [2] gave success 1.00 / 0.75 / 0.25
# at ent_fid 1.00 / 0.90 / 0.70, so the knee sits near 0.80. Points are dense
# through 0.75-0.95 where the curve actually moves, and QARQ-TP's published
# 0.988 / 0.979 / 0.962 are kept as cross-protocol anchors.
FULL_ENT_FIDS = [1.0, 0.988, 0.979, 0.962, 0.95, 0.92, 0.90, 0.88,
                 0.86, 0.84, 0.82, 0.80, 0.78, 0.75, 0.70, 0.65]

# Gate-fidelity sweep, run at PERFECT entanglement so the hardware requirement
# is isolated from the channel. This answers a limitation the source paper
# states but never quantifies: qTCP "requires low error rate of transmission,
# that is, it is more challenging in the hardware". The transition lives
# between ~0.997 and ~0.9999.
GATE_SWEEP_FIDS = [1.0, 0.9999, 0.9995, 0.999, 0.998, 0.997, 0.995, 0.99]

FULL_N = 2000              # random-state trials per point
FULL_N_FIXED = 400         # fixed-state trials per point

FIXED_STATES = {
    "fixed_0":     np.array([1, 0], dtype=complex),
    "fixed_plus":  np.array([1, 1], dtype=complex) / np.sqrt(2),
    "fixed_iplus": np.array([1, 1j], dtype=complex) / np.sqrt(2),
}
# All three, so no Pauli is invisible: Z is undetectable on |0>, X on |+>.

MASTER_SEED = 20260817
OUT_DIR = "benchmark_results"

ROW_FIELDS = [
    "arm", "ent_fid", "gate_fid", "meas_fid",
    "state_kind", "trial_idx",
    "delivered", "fidelity", "outcome",
    "send_time_us",
    "ent_pairs_consumed", "classical_msgs", "classical_msgs_total",
    "gate_count", "gate_count_total",
    "qubits_used", "memory_slot_time_us",
    "longest_share_us", "recursion_depth_max", "restart_count", "n_transfers",
    "paulis_injected", "meas_flips",
    "psi_sent", "psi_recv",
    "seed_base", "wall_s", "error",
]


# ----------------------------------------------------------------------------
# CONFIG BUILDING
# ----------------------------------------------------------------------------

def load_base_config(path=BASE_CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


def build_config(base, ent_fid, gate_fid, meas_fid, seed_base):
    """Deep-copy the base config and stamp in this trial's parameters.

    Every node gets a distinct seed derived from seed_base -- INCLUDING the BSM
    nodes. Seeding only the routers leaves part of the noise trajectory frozen
    across trials, which produces variance that is real but too small, and is
    much harder to spot than no variance at all.
    """
    cfg = copy.deepcopy(base)

    # Node seeds come from a SeedSequence rather than seed_base+i. Consecutive
    # integer seeds are only properly decorrelated if the RNG hashes them; if
    # SeQUeNCe uses a legacy seeding path anywhere, neighbouring nodes could
    # share structure in their streams. This costs nothing and removes the
    # question entirely.
    node_seeds = np.random.SeedSequence(int(seed_base)).generate_state(
        len(cfg["nodes"]))

    for i, node in enumerate(cfg["nodes"]):
        node["seed"] = int(node_seeds[i] % 2**31)
        if node.get("type") == "QTCPNode":
            node["gate_fidelity"] = gate_fid          # A2
            node["measurement_fidelity"] = meas_fid

    # Belt and braces: the loader shown to us reads these off the NODE dict,
    # but the same constants live on Topology alongside FORMALISM and
    # STOP_TIME, which are top-level. Setting both costs nothing and removes
    # one guess. If neither takes effect the parameter is not consumed at all,
    # and no amount of moving it in the JSON will help.
    cfg["gate_fidelity"] = gate_fid
    cfg["measurement_fidelity"] = meas_fid
    for tpl in cfg.get("templates", {}).values():
        tpl["gate_fidelity"] = gate_fid
        tpl["measurement_fidelity"] = meas_fid

    for tpl in cfg.get("templates", {}).values():     # A3
        if "MemoryArray" in tpl:
            tpl["MemoryArray"]["fidelity"] = ent_fid

    return cfg


# ----------------------------------------------------------------------------
# INSTRUMENTATION
# ----------------------------------------------------------------------------

class GateNoiseInjector:
    """Gate and measurement noise for the ket_vector data path.

    WHY THIS EXISTS. node.gate_fid / node.meas_fid are read in exactly two
    places in SeQUeNCe -- purification/bbpssw_bds.py and swapping/swapping_bds.py
    -- both of which are Bell-Diagonal-State analytic updates. On ket_vector,
    with purification off and adjacent nodes (no swap), none of that runs, so
    those parameters have no effect whatsoever. Confirmed by preflight [4].

    This injects the noise directly instead, using SeQUeNCe's OWN conversion
    from quantum_manager/stabilizer.py so the model matches what the toolkit
    does elsewhere rather than being invented here:

        p_error = 1.5  * (1 - one_qubit_gate_fid)     # single-qubit gates
        p_error = 1.25 * (1 - two_qubit_gate_fid)     # two-qubit gates

    That is the standard average-gate-fidelity -> depolarizing-probability
    conversion. On error, a uniformly random non-identity Pauli is applied to
    each qubit the circuit touched. Measurement infidelity flips each returned
    outcome bit with probability (1 - meas_fid).

    REPORT THIS HONESTLY: it is a channel model applied at the simulation
    boundary, not noise emerging from the protocol implementation. Say so, and
    cite the conversion. It is applied to every circuit the quantum manager
    executes, which includes entanglement-generation internals -- physically
    correct, since gate infidelity is a property of the hardware, not of which
    protocol happens to be calling.
    """

    def __init__(self, gate_fid=1.0, meas_fid=1.0, seed=0):
        self.gate_fid = gate_fid
        self.meas_fid = meas_fid
        self.rng = np.random.default_rng(seed)
        self.paulis_applied = 0
        self.meas_flips = 0
        self._undo = []

    def install(self, qm):
        if self.gate_fid >= 1.0 and self.meas_fid >= 1.0:
            return                                  # nothing to do
        from sequence.components.circuit import Circuit

        px, py, pz = Circuit(1), Circuit(1), Circuit(1)
        px.x(0)
        py.y(0)
        pz.z(0)
        paulis = [px, py, pz]

        original = qm.run_circuit
        gf, mf, rng = self.gate_fid, self.meas_fid, self.rng

        def make(orig_fn):
            def noisy(*args, **kwargs):
                circuit = args[0] if args else kwargs.get("circuit")
                keys = args[1] if len(args) > 1 else kwargs.get("keys", [])
                res = orig_fn(*args, **kwargs)

                # --- gate noise -------------------------------------------
                # One draw PER GATE per qubit, not per run_circuit call. A
                # circuit carrying 10 gates should get ~10x the error of a
                # circuit carrying 1; drawing once per call understates deep
                # circuits by exactly that factor.
                if gf < 1.0 and keys:
                    n = getattr(circuit, "size", None) or len(keys)
                    p = (1.5 * (1.0 - gf)) if n == 1 else (1.25 * (1.0 - gf))
                    p = min(1.0, p)
                    ngates = (len(getattr(circuit, "gates", None) or [1])
                              if PER_GATE_NOISE else 1)
                    for k in keys:
                        for _ in range(ngates):
                            if rng.random() < p:
                                # orig_fn, not the wrapper -- the error circuit
                                # must not itself be noised.
                                orig_fn(paulis[rng.integers(3)], [k],
                                        rng.random())
                                self.paulis_applied += 1

                # --- measurement noise ------------------------------------
                if mf < 1.0 and isinstance(res, dict) and res:
                    for k in list(res.keys()):
                        v = res[k]
                        if isinstance(v, (int, bool)) and rng.random() < (1.0 - mf):
                            res[k] = 1 - int(v)
                            self.meas_flips += 1
                return res
            return noisy

        qm.run_circuit = make(original)
        self._undo.append(lambda: setattr(qm, "run_circuit", original))

    def restore(self):
        for fn in reversed(self._undo):
            try:
                fn()
            except Exception:
                pass
        self._undo.clear()


class TrialCounters:
    """Counters wired in by wrapping live objects for the duration of one trial.

    Every wrapper is built by a FACTORY that closes over the original callable
    and forwards *args/**kwargs untouched. Binding the original as a default
    parameter (def f(a, b, _o=original)) breaks the moment the caller passes an
    extra positional argument -- it lands in _o and you get "int is not
    callable". Never assume a signature you have not read.

    Events are timestamped so counts can be reported both for the whole
    simulation and for the packet's own lifetime. Background entanglement
    generation keeps running after delivery, and including that traffic in
    metric 4 would swamp qTCP's actual protocol overhead.
    """

    def __init__(self, timeline=None):
        self.tl = timeline
        self.gate_times = []       # sim-time of each circuit execution
        self.gate_weights = []     # gates per execution
        self.msg_times = []        # sim-time of each classical transmit
        self.slot_events = []      # (sim_time, +1/-1)
        self.finishes = []         # (packet_id, share_index, status, reason)
        self._undo = []

    def _now(self):
        try:
            return self.tl.now() if self.tl is not None else 0
        except Exception:
            return 0

    # -- gate count ----------------------------------------------------------
    def wrap_quantum_manager(self, qm):
        original = qm.run_circuit

        def make(orig_fn):
            def counted(*args, **kwargs):
                circuit = args[0] if args else kwargs.get("circuit")
                n = len(getattr(circuit, "gates", None) or [1])
                self.gate_times.append(self._now())
                self.gate_weights.append(n)
                return orig_fn(*args, **kwargs)
            return counted

        qm.run_circuit = make(original)
        self._undo.append(lambda: setattr(qm, "run_circuit", original))

    # -- classical messages --------------------------------------------------
    def wrap_cchannels(self, cchannels):
        for ch in cchannels:
            original = ch.transmit

            def make(orig_fn):
                def counted(*args, **kwargs):
                    self.msg_times.append(self._now())
                    return orig_fn(*args, **kwargs)
                return counted

            ch.transmit = make(original)
            self._undo.append(
                lambda c=ch, o=original: setattr(c, "transmit", o))

    # -- slot occupancy ------------------------------------------------------
    def wrap_slots(self, transfer):
        orig_alloc = transfer.alloc_data_slot
        orig_free = getattr(transfer, "free_data_slot", None)

        def make_alloc(orig_fn):
            def alloc(*args, **kwargs):
                r = orig_fn(*args, **kwargs)
                self.slot_events.append((self._now(), +1))
                return r
            return alloc

        transfer.alloc_data_slot = make_alloc(orig_alloc)
        self._undo.append(
            lambda: setattr(transfer, "alloc_data_slot", orig_alloc))

        if orig_free is not None:
            def make_free(orig_fn):
                def free(*args, **kwargs):
                    self.slot_events.append((self._now(), -1))
                    return orig_fn(*args, **kwargs)
                return free

            transfer.free_data_slot = make_free(orig_free)
            self._undo.append(
                lambda: setattr(transfer, "free_data_slot", orig_free))

    # -- transfer structure --------------------------------------------------
    def wrap_finish(self, transfer):
        """packet_id / share_index per completed transfer -- transfer.metrics
        does not carry them."""
        original = transfer._finish

        def make(orig_fn):
            def finished(*args, **kwargs):
                t = args[0] if args else kwargs.get("transfer")
                status = args[1] if len(args) > 1 else kwargs.get("status")
                reason = args[2] if len(args) > 2 else kwargs.get("reason")
                self.finishes.append((
                    getattr(t, "packet_id", None),
                    getattr(t, "share_index", None),
                    getattr(status, "name", str(status)),
                    getattr(reason, "name", None) if reason else None,
                ))
                return orig_fn(*args, **kwargs)
            return finished

        transfer._finish = make(original)
        self._undo.append(lambda: setattr(transfer, "_finish", original))

    # -- derived quantities --------------------------------------------------
    def gate_count(self, t0=None, t1=None):
        if t0 is None:
            return sum(self.gate_weights)
        return sum(w for t, w in zip(self.gate_times, self.gate_weights)
                   if t0 <= t <= t1)

    def msg_count(self, t0=None, t1=None):
        if t0 is None:
            return len(self.msg_times)
        return sum(1 for t in self.msg_times if t0 <= t <= t1)

    def peak_slots(self):
        peak = cur = 0
        for _, d in sorted(self.slot_events):
            cur += d
            peak = max(peak, cur)
        return peak

    def slot_time_us(self):
        """Integral of occupied slots over time, in slot-microseconds."""
        if not self.slot_events:
            return 0.0
        ev = sorted(self.slot_events)
        total = cur = 0.0
        prev_t = ev[0][0]
        for t, d in ev:
            total += cur * (t - prev_t)
            prev_t = t
            cur += d
        return total / 1e6      # ps -> us

    def max_packet_id(self):
        ids = [p for p, _, _, _ in self.finishes if p is not None]
        return max(ids) if ids else 0

    def restore(self):
        for fn in reversed(self._undo):
            try:
                fn()
            except Exception:
                pass
        self._undo.clear()


# ----------------------------------------------------------------------------
# FIDELITY
# ----------------------------------------------------------------------------

def fidelity(psi_sent, psi_recv):
    """F = |<sent|recv>|^2. Global-phase invariant by construction, so none of
    the phase-aware machinery in verify_same_state_vector is needed here.

    Returns (F, note). A note means the received object was not a clean
    single-qubit ket and F should not be trusted -- e.g. the delivered qubit is
    still entangled with residual ancillas, which no amount of averaging fixes.
    """
    if psi_recv is None:
        return 0.0, "no_state"
    a = np.asarray(psi_sent, dtype=complex).ravel()
    b = np.asarray(psi_recv, dtype=complex).ravel()
    if b.size != 2:
        return float("nan"), f"recv_dim_{b.size}"
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0, "zero_norm"
    return float(abs(np.vdot(a / na, b / nb)) ** 2), ""


def random_state(rng):
    v = np.array([rng.normal() + 1j * rng.normal(),
                  rng.normal() + 1j * rng.normal()], dtype=complex)
    return v / np.linalg.norm(v)


def ser(v):
    if v is None:
        return ""
    v = np.asarray(v).ravel()
    return ";".join(f"{x.real:.6g}{x.imag:+.6g}j" for x in v)


# ----------------------------------------------------------------------------
# ONE TRIAL
# ----------------------------------------------------------------------------

def run_trial(base_cfg, arm, ent_fid, gate_fid, meas_fid,
              state_kind, trial_idx, seed_base):
    """Build a fresh topology, send one qubit, return one fat row."""
    from sequence.topology.qtcp_net_topo import QTCPNetTopo
    from sequence.app.qtcp.qtcp_app import QTCPApp
    from sequence.constants import MILLISECOND
    import sequence.utils.log as log

    t0 = time.time()
    row = {f: "" for f in ROW_FIELDS}
    row.update(arm=arm, ent_fid=ent_fid, gate_fid=gate_fid, meas_fid=meas_fid,
               state_kind=state_kind, trial_idx=trial_idx, seed_base=seed_base)

    counters = TrialCounters()
    injector = None
    try:
        cfg = build_config(base_cfg, ent_fid, gate_fid, meas_fid, seed_base)

        topo = QTCPNetTopo(cfg)
        tl = topo.tl
        nodes = topo.nodes[QTCPNetTopo.QTCP_NODE]
        alice = next(n for n in nodes if n.name == ALICE)
        bob = next(n for n in nodes if n.name == BOB)
        charlie = next(n for n in nodes if n.name == CHARLIE)

        app_alice = QTCPApp(alice, MAX_DEPTH, RESTART_AMOUNT, FALLBACK_MODE)
        app_bob = QTCPApp(bob, MAX_DEPTH, RESTART_AMOUNT, FALLBACK_MODE)
        app_charlie = QTCPApp(charlie, MAX_DEPTH, RESTART_AMOUNT, FALLBACK_MODE)

        # Input state. The per-trial RNG is seeded from seed_base so the state
        # is reproducible from the row alone.
        if state_kind == "random":
            psi_sent = random_state(np.random.default_rng(seed_base))
        else:
            psi_sent = FIXED_STATES[state_kind]

        slot = app_alice.transfer.alloc_data_slot()
        data_arr = alice.get_component_by_name(alice.data_memo_arr_name)
        data_arr[slot].update_state(psi_sent)

        app_alice.connect(dst=BOB,
                          start_t=CONNECT_START_MS * MILLISECOND,
                          end_t=CONNECT_END_MS * MILLISECOND,
                          memory_size=MEMORY_SIZE, num_qubits=NUM_QUBITS)

        pid = app_alice.send_packet(slot, BOB)

        # Instrument after send_packet, matching the order both existing
        # scripts use for their monkeypatches.
        injector = None
        if GATE_NOISE_MODE == "injected" and (gate_fid < 1.0 or meas_fid < 1.0):
            injector = GateNoiseInjector(gate_fid, meas_fid,
                                         seed=int(seed_base) ^ 0x5EED)
            injector.install(tl.quantum_manager)

        counters.tl = tl
        counters.wrap_quantum_manager(tl.quantum_manager)
        counters.wrap_cchannels(topo.cchannels)
        counters.wrap_slots(app_alice.transfer)
        counters.wrap_finish(app_alice.transfer)

        # Logging off: at DEBUG with tracked modules this dominates wall-clock
        # and produces unusable volumes of I/O over thousands of trials.
        setup_logging_once(tl)

        tl.init()
        tl.run()

        outcome = app_alice.overseer.get_packet_outcome(pid)
        psi_recv = app_bob.get_received_packet(ALICE, pid)

        F, note = fidelity(psi_sent, psi_recv)
        outcome_name = outcome.name if outcome is not None else "None"

        # delivered == the binary success for metric 2. Kept separate from
        # fidelity on purpose: do NOT assume F = 1 - delivered. Whether they
        # agree is the empirical question, not an assumption.
        delivered = (outcome_name == "DELIVERED" and psi_recv is not None
                     and F > 0.999)

        tmetrics = getattr(app_alice.transfer, "metrics", []) or []
        # Metric 5 wants send -> confirmed delivery for the WHOLE PACKET. Each
        # entry in transfer.metrics is one teleportation, so max(latency) is
        # the longest single share, not the packet. Span from the earliest
        # send_time to the latest finish_time instead.
        send_time_us = ""
        starts = [m.get("send_time") for m in tmetrics
                  if m.get("send_time") is not None]
        ends = [m.get("finish_time") for m in tmetrics
                if m.get("finish_time") is not None]
        if starts and ends:
            send_time_us = (max(ends) - min(starts)) / 1e6      # ps -> us
        longest_share_us = (max(m["latency"] for m in tmetrics
                                if m.get("latency") is not None) / 1e6
                            if any(m.get("latency") is not None
                                   for m in tmetrics) else "")

        # Window the resource counters to the packet's own lifetime. Background
        # entanglement generation keeps running after delivery; counting it
        # would swamp qTCP's protocol overhead in metric 4.
        if starts and ends:
            t0, t1 = min(starts), max(ends)
            gates_win, msgs_win = counters.gate_count(t0, t1), counters.msg_count(t0, t1)
        else:
            gates_win, msgs_win = counters.gate_count(), counters.msg_count()

        row.update(
            delivered=int(delivered),
            fidelity=F,
            outcome=outcome_name,
            send_time_us=send_time_us,
            longest_share_us=longest_share_us,
            ent_pairs_consumed=len(tmetrics),          # A5
            classical_msgs=msgs_win,
            classical_msgs_total=counters.msg_count(),
            gate_count=gates_win,
            gate_count_total=counters.gate_count(),
            qubits_used=counters.peak_slots(),
            memory_slot_time_us=round(counters.slot_time_us(), 3),
            recursion_depth_max=counters.max_packet_id(),
            restart_count=sum(1 for _, _, s, _ in counters.finishes
                              if s == "FAILED"),
            n_transfers=len(counters.finishes),
            paulis_injected=(injector.paulis_applied if injector else 0),
            meas_flips=(injector.meas_flips if injector else 0),
            psi_sent=ser(psi_sent),
            psi_recv=ser(psi_recv),
            error=note,
        )

    except Exception as e:
        # Compact, greppable form for the CSV; full traceback kept out-of-band
        # under a key DictWriter is told to ignore.
        row["error"] = f"EXC:{type(e).__name__}: {e}"[:300].replace("\n", " ")
        row["_traceback"] = traceback.format_exc()
    finally:
        counters.restore()
        if injector is not None:
            injector.restore()

    row["wall_s"] = round(time.time() - t0, 2)
    return row


# ----------------------------------------------------------------------------
# WORK PLAN
# ----------------------------------------------------------------------------

def make_jobs(ent_fids, n_random, n_fixed, gate_sweep=None):
    """Work plan as a list of arms.

    ideal      gate/meas perfect, ent-gen swept  -> metric 2 (QPing knee)
    realistic  gate 0.999, ent-gen swept         -> metrics 1, 3, 4, 5
    gatesweep  ent-gen perfect, gate swept       -> hardware requirement figure
    """
    arms = [
        dict(name="ideal", ent_fids=ent_fids,
             gate_fids=[1.0], meas_fids=[1.0]),
        dict(name="realistic", ent_fids=ent_fids,
             gate_fids=[REALISTIC_GATE_FID], meas_fids=[REALISTIC_MEAS_FID]),
    ]
    if gate_sweep:
        arms.append(dict(name="gatesweep", ent_fids=[1.0],
                         gate_fids=gate_sweep, meas_fids=[REALISTIC_MEAS_FID]))

    # seed_base is unique BY CONSTRUCTION, not by luck. Drawing ~100k random
    # 31-bit seeds gives ~2 expected birthday collisions, and a collision means
    # two trials replay an identical trajectory -- small, but it is exactly the
    # duplicate-trajectory failure this whole exercise is trying to avoid.
    # build_config runs SeedSequence(seed_base) to derive node seeds, and that
    # hashes properly, so a sequential counter here is safe.
    si = 0

    jobs = []
    for arm in arms:
        for ef in arm["ent_fids"]:
            for gf in arm["gate_fids"]:
                for mf in arm["meas_fids"]:
                    kinds = [("random", n_random)]
                    if n_fixed:
                        per = max(1, n_fixed // len(FIXED_STATES))
                        kinds += [(k, per) for k in FIXED_STATES]
                    for kind, n in kinds:
                        for i in range(n):
                            seed_base = (MASTER_SEED << 20) + si
                            si += 1
                            jobs.append(dict(
                                arm=arm["name"], ent_fid=ef, gate_fid=gf,
                                meas_fid=mf, state_kind=kind, trial_idx=i,
                                seed_base=seed_base))
    return jobs


def job_key(j):
    # gate_fid MUST be in the key. The gatesweep arm holds ent_fid constant and
    # varies gate_fid, so a key without it collapses every gate point onto one
    # another -- resume would skip work that was never done.
    return (j["arm"], f'{j["ent_fid"]:.6g}', f'{j["gate_fid"]:.6g}',
            j["state_kind"], j["trial_idx"])


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            done.add((r["arm"], f'{float(r["ent_fid"]):.6g}',
                      f'{float(r["gate_fid"]):.6g}',
                      r["state_kind"], int(r["trial_idx"])))
    return done


_BASE = None
_LOG_READY = False


def setup_logging_once(tl):
    """Configure SeQUeNCe logging ONCE per process, not once per trial.

    The implementation notes warn that per-trial log.set_logger does not
    cleanly rebind timelines. Over tens of thousands of trials in one worker
    that also risks accumulating handlers and leaking file descriptors. We
    silence it once and never touch it again; a failure here must not kill the
    run, so it is swallowed.
    """
    global _LOG_READY
    if _LOG_READY:
        return
    _LOG_READY = True
    try:
        import logging
        import sequence.utils.log as log
        log.set_logger(__name__, tl, os.devnull)
        log.set_logger_level("CRITICAL")
        logging.getLogger().addHandler(logging.NullHandler())
    except Exception:
        pass


def _init_worker(base_cfg):
    global _BASE
    _BASE = base_cfg


def _do(job):
    return run_trial(_BASE, **job)


# ----------------------------------------------------------------------------
# PREFLIGHT -- catch wrong assumptions in seconds, not hours
# ----------------------------------------------------------------------------

def probe_fidelity_plumbing(base_cfg, value=0.5):
    """Does the injected gate/measurement fidelity actually LAND on the node?

    This separates two failure modes that look identical from the outside:
      (a) the value never reaches the node object -> config plumbing is wrong,
          keep moving it around the JSON until it lands;
      (b) the value is sitting on the node but nothing reads it -> plumbing is
          fine and the fix is in the protocol code, not the config.
    Guessing between these is what burns preflight rounds.
    """
    from sequence.topology.qtcp_net_topo import QTCPNetTopo

    print("\n[0] does injected gate/meas fidelity reach the node object? ...")
    cfg = build_config(base_cfg, 1.0, value, value, 12345)
    topo = QTCPNetTopo(cfg)
    node = topo.nodes[QTCPNetTopo.QTCP_NODE][0]

    hits = {}
    for attr in dir(node):
        if attr.startswith("__"):
            continue
        low = attr.lower()
        if "fid" not in low:
            continue
        try:
            v = getattr(node, attr)
        except Exception:
            continue
        if isinstance(v, (int, float)):
            hits[attr] = v

    if not hits:
        print("    no fidelity-like attribute found on the node at all.")
    for k, v in sorted(hits.items()):
        mark = "  <-- injected value landed" if abs(v - value) < 1e-9 else ""
        print(f"    node.{k:28} = {v}{mark}")

    landed = any(abs(v - value) < 1e-9 for v in hits.values())
    if landed:
        print("\n    -> The value REACHES the node. Config plumbing is fine;")
        print("       moving it in the JSON will not help. Nothing downstream")
        print("       reads it -- the fix is in the protocol code.")
    else:
        print("\n    -> The value does NOT reach the node. This is a config")
        print("       plumbing problem; find the reader with:")
        print("       grep -rn 'gate_fid\\|GATE_FIDELITY' sequence/ --include=*.py")
    return landed


def preflight(base_cfg):
    print("=" * 70)
    print("PREFLIGHT")
    print("=" * 70)
    ok = True

    try:
        probe_fidelity_plumbing(base_cfg)
    except Exception as e:
        print(f"    probe failed: {type(e).__name__}: {e}")

    # 1. seeds actually vary outcomes  -- RANDOM states only.
    # A fixed stabiliser state is blind to one Pauli by construction (X|+> = |+>,
    # Z|0> = |0>), so using one here can report "no variation" on a channel that
    # is varying fine. Random states see every error class.
    print("\n[1] seeds vary trial outcomes ...")
    rows = [run_trial(base_cfg, "realistic", 0.85, REALISTIC_GATE_FID,
                      REALISTIC_MEAS_FID, "random", i, 1000 + i * 97)
            for i in range(8)]
    errs = [r for r in rows if str(r["error"]).startswith("EXC")]
    if errs:
        print(f"    FAIL - {len(errs)}/{len(rows)} trials raised.\n")
        print("    " + str(errs[0]["error"]))
        print("\n    full traceback:\n")
        for line in str(errs[0].get("_traceback", "")).splitlines():
            print("      " + line)
        return False
    fids = [float(r["fidelity"]) for r in rows]
    print(f"    fidelities: {[round(f, 4) for f in fids]}")
    if len(set(round(f, 9) for f in fids)) == 1:
        print("    WARN - every trial identical at ent_fid=0.85. Check [2].")
        ok = False
    else:
        print("    ok - outcomes differ")

    # 2. noise actually degrades
    print("\n[2] noise degrades delivery ...")
    for ef in [1.0, 0.90, 0.70]:
        rr = [run_trial(base_cfg, "realistic", ef, REALISTIC_GATE_FID,
                        REALISTIC_MEAS_FID, "random", i, 5000 + i * 31)
              for i in range(8)]
        good = [r for r in rr if not str(r["error"]).startswith("EXC")]
        if not good:
            print(f"    ent_fid={ef}: all trials errored")
            ok = False
            continue
        sr = np.mean([r["delivered"] for r in good])
        mf = np.nanmean([float(r["fidelity"]) for r in good])
        print(f"    ent_fid={ef:5.2f}  success={sr:5.2f}  mean F={mf:6.4f}")

    # 3. WHICH Paulis does the channel produce?
    # Each stabiliser state is immune to exactly one Pauli. Comparing all three
    # at the same noise level reads the channel's composition straight off:
    #   |0>  is blind to Z      |+>  is blind to X      |+i> is blind to Y
    # A state that stays at F=1.0 while the others degrade names the DOMINANT
    # error. This is worth recording in the report -- section 2 asserts noise
    # arrives as discrete Pauli events but does not say which.
    print("\n[3] Pauli composition of the channel (ent_fid=0.80) ...")
    comp, sr_by_kind = {}, {}
    for kind in FIXED_STATES:
        rr = [run_trial(base_cfg, "realistic", 0.80, REALISTIC_GATE_FID,
                        REALISTIC_MEAS_FID, kind, i, 7000 + i * 53)
              for i in range(10)]
        good = [r for r in rr if not str(r["error"]).startswith("EXC")]
        mf = np.nanmean([float(r["fidelity"]) for r in good]) if good else float("nan")
        sr = np.mean([r["delivered"] for r in good]) if good else float("nan")
        comp[kind] = mf
        sr_by_kind[kind] = sr
        blind = {"fixed_0": "Z", "fixed_plus": "X", "fixed_iplus": "Y"}[kind]
        print(f"    {kind:12} (blind to {blind})  success={sr:5.2f}  mean F={mf:6.4f}")
    clean = [k for k, v in comp.items() if v > 0.999]
    if clean and len(clean) < len(comp):
        dom = ", ".join({"fixed_0": "Z", "fixed_plus": "X",
                         "fixed_iplus": "Y"}[k] for k in clean)
        print(f"    -> channel is dominated by {dom} errors "
              f"(the state blind to them stayed perfect).")
    elif len(clean) == len(comp):
        print("    -> nothing degrades at 0.80. Inconsistent with [2]; investigate.")
        ok = False
    else:
        # Each state is blind to exactly one Pauli, so the three failure rates
        # are a solvable 3x3 system for the effective logical error rates:
        #   fail(|0>)  = pX + pY      fail(|+>)  = pY + pZ
        #   fail(|+i>) = pX + pZ
        f0 = 1 - comp["fixed_0"]
        fp = 1 - comp["fixed_plus"]
        fi = 1 - comp["fixed_iplus"]
        tot = (f0 + fp + fi) / 2.0
        pZ, pX, pY = tot - f0, tot - fp, tot - fi
        print(f"    -> effective logical error rates: "
              f"pX={pX:.3f}  pY={pY:.3f}  pZ={pZ:.3f}  (total {tot:.3f})")
        spread = max(pX, pY, pZ) / max(min(pX, pY, pZ), 1e-9)
        if spread < 1.5:
            print("       roughly balanced -> depolarizing-like.")
        else:
            worst = {"X": pX, "Y": pY, "Z": pZ}
            dom = max(worst, key=worst.get)
            print(f"       {dom}-biased by {spread:.1f}x -> NOT depolarizing.")
            print("       Worth confirming at higher N and stating in the")
            print("       report; section 2 says 'discrete Pauli events' but")
            print("       never says which.")
        # Does mean fidelity track success rate exactly?
        if all(abs(comp[k] - sr_by_kind.get(k, -1)) < 1e-6 for k in comp):
            print("       mean F == success rate for all three -> failures are")
            print("       ORTHOGONAL, so F = 1 - success holds (section 7).")

    # 4. gate and measurement fidelity, tested SEPARATELY and aggressively.
    # 0.5 is far past anything realistic on purpose: if the parameter is wired
    # in at all, it cannot fail to show at this level.
    print(f"\n[4] gate / measurement fidelity are consumed "
          f"(GATE_NOISE_MODE={GATE_NOISE_MODE!r}) ...")

    def mean_f(gate, meas):
        rr = [run_trial(base_cfg, "x", 1.0, gate, meas, "random", i, 9000 + i)
              for i in range(8)]
        good = [r for r in rr if not str(r["error"]).startswith("EXC")]
        return np.nanmean([float(r["fidelity"]) for r in good]) if good else float("nan")

    f_base = mean_f(1.0, 1.0)
    f_gate = mean_f(0.5, 1.0)
    f_meas = mean_f(1.0, 0.5)
    print(f"    gate=1.0 meas=1.0 -> F={f_base:.4f}   (baseline)")
    print(f"    gate=0.5 meas=1.0 -> F={f_gate:.4f}")
    print(f"    gate=1.0 meas=0.5 -> F={f_meas:.4f}")
    dead = []
    if abs(f_base - f_gate) < 1e-9:
        dead.append("gate_fidelity")
    if abs(f_base - f_meas) < 1e-9:
        dead.append("measurement_fidelity")
    if dead:
        print(f"    FAIL - {' and '.join(dead)} changes nothing even at 0.5.")
        if GATE_NOISE_MODE == "native":
            print("           Expected on ket_vector: those parameters are only")
            print("           read by the _bds swapping/purification paths, and")
            print("           you run neither. Set GATE_NOISE_MODE='injected'")
            print("           to model it, or drop to gate=meas=1.0 and scope")
            print("           it out of the report.")
        print("           Not the cheat-swap: with adjacent nodes there is no")
        print("           entanglement swap, so that revert cannot show here.")
        print("           The parameter is stored on QTCPNode but never reaches")
        print("           the data path. Until it is wired in, the realistic")
        print("           arm is identical to the ideal arm and metrics 1/3/4/5")
        print("           carry no gate/measurement realism at all.")
        ok = False
    else:
        print("    ok - both are live")

    # 5. counters populated
    print("\n[5] counters populated ...")
    r = rows[0]
    for f in ["ent_pairs_consumed", "classical_msgs", "classical_msgs_total",
              "gate_count", "gate_count_total", "qubits_used",
              "send_time_us", "longest_share_us"]:
        v = r[f]
        flag = "" if v not in ("", 0) else "   <-- zero/empty, check the wrapper"
        print(f"    {f:22s} = {v}{flag}")

    # 6. window is wide enough
    print("\n[6] connect window resolves every packet ...")
    ip = [r for r in rows if r["outcome"] == "IN_PROGRESS"]
    if ip:
        print(f"    FAIL - {len(ip)}/{len(rows)} ended IN_PROGRESS. The window")
        print(f"           closes at {CONNECT_END_MS} ms before the packet")
        print("           resolves; those trials would silently depress the")
        print("           success rate. Widen CONNECT_END_MS.")
        ok = False
    else:
        print(f"    ok - all resolved within {CONNECT_END_MS} ms")

    # 7. received state shape
    print("\n[7] received state is a clean single-qubit ket ...")
    notes = {r["error"] for r in rows if r["error"]
             and not str(r["error"]).startswith("EXC")}
    if notes:
        print(f"    WARN - notes seen: {notes}")
        print("           'recv_dim_N' means the delivered qubit is not a bare")
        print("           2-vector; fidelity cannot be trusted as computed.")
        ok = False
    else:
        print("    ok")

    print("\n" + "=" * 70)
    print("PREFLIGHT", "PASSED" if ok else "RAISED ISSUES -- read above")
    print("=" * 70)
    return ok


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--config", default=BASE_CONFIG_PATH)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not (args.pilot or args.full or args.preflight_only):
        ap.error("pick --pilot, --full, or --preflight-only")

    base_cfg = load_base_config(args.config)
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.preflight_only:
        sys.exit(0 if preflight(base_cfg) else 1)

    if args.pilot:
        ok = preflight(base_cfg)
        if not ok:
            print("\nPreflight raised issues. Continuing with the pilot anyway")
            print("so you can see the curves, but do not scale up until they")
            print("are resolved -- a flat curve here is the classic silent bug.\n")
        ent_fids, n_rand, n_fixed = PILOT_ENT_FIDS, PILOT_N, 0
        gsweep = None
        tag = "pilot"
    else:
        ent_fids, n_rand, n_fixed = FULL_ENT_FIDS, FULL_N, FULL_N_FIXED
        gsweep = GATE_SWEEP_FIDS
        tag = "full"

    out = args.out or os.path.join(OUT_DIR, f"trials_{tag}.csv")
    jobs = make_jobs(ent_fids, n_rand, n_fixed, gate_sweep=gsweep)
    done = load_done(out)
    todo = [j for j in jobs if job_key(j) not in done]

    print(f"\n{len(jobs)} trials planned, {len(done)} already done, "
          f"{len(todo)} to run")
    print(f"output: {out}")
    if not todo:
        print("nothing to do")
        summarise(out)
        return

    fresh = not os.path.exists(out)
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=ROW_FIELDS, extrasaction="ignore")
    if fresh:
        w.writeheader()
        fh.flush()

    t0 = time.time()
    n_err = 0

    def handle(i, row):
        nonlocal n_err
        w.writerow(row)
        fh.flush()                      # crash-safe: every row hits disk
        if str(row["error"]).startswith("EXC"):
            n_err += 1
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            el = time.time() - t0
            rate = (i + 1) / el
            eta = (len(todo) - i - 1) / rate if rate else 0
            print(f"  {i+1}/{len(todo)}  {rate*60:.1f}/min  "
                  f"ETA {eta/3600:.2f} h  errors {n_err}", flush=True)

    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(base_cfg,), maxtasksperchild=200) as pool:
            for i, row in enumerate(pool.imap_unordered(_do, todo,
                                                        chunksize=1)):
                handle(i, row)
    else:
        _init_worker(base_cfg)
        for i, job in enumerate(todo):
            handle(i, _do(job))

    fh.close()
    print(f"\ndone in {(time.time()-t0)/3600:.2f} h, {n_err} errored trials")
    summarise(out)


# ----------------------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------------------

def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial interval. The normal/Wald approximation is invalid near
    p=1 -- it returns bounds above 1.0 and gives a zero-width interval when
    zero failures are observed, which is a fake certainty."""
    from scipy.stats import beta
    if n == 0:
        return float("nan"), float("nan")
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def summarise(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if not str(r["error"]).startswith("EXC"):
                rows.append(r)
    if not rows:
        print("no usable rows")
        return

    print("\n" + "=" * 78)
    print("SUCCESS RATE (Clopper-Pearson, alpha=0.05) -- random-state trials")
    print("=" * 78)
    print(f'{"arm":10} {"ent_fid":>8} {"gate_fid":>9} {"n":>7} {"success":>9} '
          f'{"95% CI":>18} {"mean F":>8} {"send us":>9}')

    groups = {}
    for r in rows:
        if r["state_kind"] != "random":
            continue
        groups.setdefault((r["arm"], float(r["ent_fid"]),
                           float(r["gate_fid"])), []).append(r)

    for (arm, ef, gf) in sorted(groups, key=lambda x: (x[0], -x[1], -x[2])):
        g = groups[(arm, ef, gf)]
        n = len(g)
        k = sum(int(x["delivered"]) for x in g)
        lo, hi = clopper_pearson(k, n)
        fs = [float(x["fidelity"]) for x in g
              if x["fidelity"] not in ("", "nan")]
        st = [float(x["send_time_us"]) for x in g if x["send_time_us"] != ""]
        print(f"{arm:10} {ef:8.3f} {gf:9.5f} {n:7d} {k/n:9.4f} "
              f"[{lo:.4f},{hi:.4f}] {np.mean(fs) if fs else float('nan'):8.4f} "
              f"{np.mean(st) if st else float('nan'):9.1f}")
        if k == n:
            print(f'{"":10} {"":8} {"":9} zero failures -> success rate >= {lo:.4f} '
                  f"(one-sided; the point estimate of 1.000 is not a "
                  f"zero-uncertainty result)")

    # IN_PROGRESS = the connect window closed before the packet resolved. That
    # is a harness artifact, not a delivery failure, and must not be pooled
    # into the success rate. If this is ever non-zero, widen CONNECT_END_MS
    # and rerun the affected points.
    stuck = [r for r in rows if r["outcome"] == "IN_PROGRESS"]
    if stuck:
        print(f"\n!! {len(stuck)} trial(s) ended IN_PROGRESS -- the connect "
              f"window closed before resolution.")
        by = {}
        for r in stuck:
            by[(r["arm"], r["ent_fid"])] = by.get((r["arm"], r["ent_fid"]), 0) + 1
        for k, v in sorted(by.items()):
            print(f"     {k[0]:10} ent_fid={k[1]:>7}  {v} trial(s)")
        print("     These are counted as not-delivered above, which understates")
        print("     the success rate. Widen CONNECT_END_MS and rerun.")

    # metric-5 tail check
    st = [float(r["send_time_us"]) for r in rows if r["send_time_us"] != ""]
    if st:
        st = np.array(st)
        print("\n" + "=" * 78)
        print("SEND TIME distribution (metric 5)")
        print("=" * 78)
        print(f"  mean {st.mean():.1f}  sd {st.std():.1f}  "
              f"min {st.min():.1f}  p50 {np.percentile(st,50):.1f}  "
              f"p99 {np.percentile(st,99):.1f}  max {st.max():.1f}")
        if st.std() / max(st.mean(), 1e-9) < 0.01:
            print("  -> essentially constant. Expected here: with no loss and")
            print("     a distance-3 leaf in correction mode, nothing detects")
            print("     a failure, so recursion never fires. Report the mean")
            print("     and say why -- this is a result, not a missing tail.")

    # metric-7 distribution: fixed states only
    fx = [float(r["fidelity"]) for r in rows
          if r["state_kind"] != "random" and r["fidelity"] not in ("", "nan")]
    if fx:
        fx = np.array(fx)
        near01 = ((fx < 0.05) | (fx > 0.95)).mean()
        print("\n" + "=" * 78)
        print("FIDELITY DISTRIBUTION on FIXED states (the metric-1 vs 2 question)")
        print("=" * 78)
        print(f"  n={len(fx)}  mean {fx.mean():.4f}  "
              f"fraction near 0 or 1: {near01:.3f}")
        if near01 > 0.95:
            print("  -> bimodal: failures are effectively orthogonal, so")
            print("     success rate and mean fidelity tell the same story.")
        else:
            print("  -> spread: failures are partial. Report BOTH curves --")
            print("     how often it fails and how bad it is differ.")
        print("  (Read this on FIXED states only. On random inputs a clean")
        print("   logical Pauli gives mean F = 1/3 with a smooth spread, which")
        print("   would look like partial failure when it is not.)")


if __name__ == "__main__":
    main()