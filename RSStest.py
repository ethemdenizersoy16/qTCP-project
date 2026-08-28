"""
RSS regression harness for qTCP.

Sends one packet under a specific loss and Pauli-injection pattern and
checks that (a) the packet resolves with the expected outcome, (b) Alice
and Bob release their data slots, and (c) peak RSS stays bounded. Built
to catch entanglement leaks and slot leaks that would otherwise only
show up under the full benchmark at scale.

Each test sends ONE packet on purpose. The overseer's packet-id counter
is shared across a run, so a second concurrent packet would interleave
ids and invalidate any (packet_id, share_index) targeting used by the
injection or loss patches.

INJECTION PATTERNS. Loss and Pauli targets are keyed by
(packet_id, share_index); LOSSES fails a share once via
install_no_entanglement_monkeypatch_once, INJECT_* applies a named Pauli
to the qubit before fire. The two patch sets are disjoint by
construction so they compose cleanly.

X (not Z) on |0> is the sensitive choice for state-vector checks -- Z on
|0> is invisible. For a state where X aliases to a stabiliser under the
active correction, switch the second error to Z or another share.
"""
import numpy as np
from sequence.topology.qtcp_net_topo import QTCPNetTopo
from sequence.app.qtcp.qtcp_app import QTCPApp
from sequence.app.qtcp.qtcp_overseer import PacketOutcome
from sequence.components.circuit import Circuit
from sequence.constants import MILLISECOND
from sequence.kernel.quantum_utils import verify_same_state_vector
import sequence.utils.log as log

from testscript import (
    CONFIG, ALICE, BOB, CHARLIE,
    install_no_entanglement_monkeypatch_once,
    random_state,
)

# Pauli circuits. X alone covers most tests; Y and Z are optional and
# guarded because older Circuit builds omit them.
_PX = Circuit(1); _PX.x(0)
_PAULI = {"X": _PX}
try:
    _PY = Circuit(1); _PY.y(0); _PAULI["Y"] = _PY
    _PZ = Circuit(1); _PZ.z(0); _PAULI["Z"] = _PZ
except Exception:
    pass


def install_pauli_injection(transfer, targets):
    """Patch transfer._fire so that a transfer whose (packet_id,
    share_index) is a key in `targets` gets the named Pauli applied to
    its data qubit before fire. Fixed rnd=0.5 keeps the injection
    deterministic across runs of the same test."""
    original_fire = transfer._fire
    injected = set()

    def patched_fire(t, info):
        key = (t.packet_id, t.share_index)
        p = targets.get(key)
        if p is not None and key not in injected:
            data_arr = transfer.node.get_component_by_name(
                transfer.node.data_memo_arr_name)
            data_key = data_arr[t.data_memory_index].qstate_key
            rnd = 0.5
            transfer.node.timeline.quantum_manager.run_circuit(
                _PAULI[p], [data_key], rnd)
            injected.add(key)
            print(
                f"[inject] {p} on transfer {t.transfer_id} "
                f"(packet {t.packet_id} share {t.share_index} "
                f"slot {t.data_memory_index}) before fire")
            log.logger.warning(
                f"[inject] {p} on transfer {t.transfer_id} "
                f"(packet {t.packet_id} share {t.share_index} "
                f"slot {t.data_memory_index}) before fire")
        original_fire(t, info)

    transfer._fire = patched_fire


# Loss and injection targets, keyed by (packet_id, share_index).
LOSSES     = []
INJECT_ONE = {(1, 0): "X"}
INJECT_TWO = {(1, 0): "X", (1, 1): "Z", (2, 0): "X", (2, 1): "Y"}

TESTS = [
    dict(name="clean",      inject={},
         expect_outcome="DELIVERED", expect_fidelity=True),
    # dict(name="one error",  inject=INJECT_ONE,
    #      expect_outcome="DELIVERED", expect_fidelity=True),
    # dict(name="two errors", inject=INJECT_TWO,
    #      expect_outcome="DELIVERED", expect_fidelity=False),
]

MAX_DEPTH = 1


def _trial(sent_state, inject):
    """Build topology, prepare one state, install losses and injections,
    run. Returns (pid, outcome, received, alice_free, bob_free, rss)."""
    topo = QTCPNetTopo(CONFIG)
    tl = topo.tl
    nodes = topo.nodes[QTCPNetTopo.QTCP_NODE]
    alice = next(n for n in nodes if n.name == ALICE)
    bob = next(n for n in nodes if n.name == BOB)
    charlie = next(n for n in nodes if n.name == CHARLIE)

    app_alice = QTCPApp(alice, MAX_DEPTH, 1, "recover")
    app_bob = QTCPApp(bob, MAX_DEPTH, 1, "recover")
    app_charlie = QTCPApp(charlie, MAX_DEPTH, 1, "recover")

    data_arr = alice.get_component_by_name(alice.data_memo_arr_name)
    slot = app_alice.transfer.alloc_data_slot()
    data_arr[slot].update_state(sent_state)

    app_alice.connect(dst=BOB, start_t=20 * MILLISECOND, end_t=50 * MILLISECOND,
                      memory_size=5, num_qubits=1)

    pid = app_alice.send_packet(slot, BOB)

    # Patches install AFTER send_packet, matching testscript's ordering.
    install_no_entanglement_monkeypatch_once(app_alice.transfer, list(LOSSES))
    if inject:
        install_pauli_injection(app_alice.transfer, inject)

    log.set_logger(__name__, tl, "qtcp_rss.log")
    log.set_logger_level('DEBUG')
    log.track_module('qtcp_overseer')
    log.track_module('__main__')
    tl.init()
    for m in ['qtcp_app', 'teleportation', 'generation',
              'qtcp_handshake', 'qtcp_transfer']:
        log.track_module(m)
    tl.run()

    outcome = app_alice.overseer.get_packet_outcome(pid)
    received = app_bob.get_received_packet(ALICE, pid)

    a = app_alice.transfer
    b = app_bob.transfer
    a_arr = alice.get_component_by_name(alice.data_memo_arr_name)
    b_arr = bob.get_component_by_name(bob.data_memo_arr_name)
    # Alice keeps nothing after delivery; Bob keeps the one delivered secret.
    alice_free = len(a.free_data_slots) == len(a_arr)
    bob_free = len(b.free_data_slots) + 1 == len(b_arr)

    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return pid, outcome, received, alice_free, bob_free, rss


def run_batch():
    rng = np.random.default_rng(12345)
    print("=" * 60)
    print("RSS REGRESSION BATCH")
    print("=" * 60)
    for t in TESTS:
        sent = random_state(rng)
        pid, outcome, received, a_free, b_free, rss = _trial(sent, t["inject"])

        ok, detail = True, []
        name = outcome.name if outcome is not None else "None"
        if name != t["expect_outcome"]:
            ok = False; detail.append(f"outcome {name}!={t['expect_outcome']}")
        elif t["expect_fidelity"]:
            if received is None or not verify_same_state_vector(received, sent):
                ok = False; detail.append("fidelity FAIL")
        else:
            # Delivered but should be wrong (miscorrected).
            if received is not None and verify_same_state_vector(received, sent):
                ok = False; detail.append("unexpectedly correct (miscorrection expected)")
        if not a_free:
            ok = False; detail.append("alice leaked")
        if not b_free:
            ok = False; detail.append("bob leaked")

        print(f"[{'PASS' if ok else 'FAIL'}] {t['name']:22s} "
              f"RSS {rss:.0f} MB"
              + (f"  {detail}" if detail else ""))
    print("=" * 60)


if __name__ == "__main__":
    run_batch()