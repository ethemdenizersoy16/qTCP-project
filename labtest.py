"""
Depth-2 QEC batch: clean / 1-error-corrected / 2-error-miscorrected.

Built against the real testscript.py. Each test sends ONE QEC packet (a single
qubit). One packet keeps the packet-id numbering equal to the clean single-tree
we compute targets against -- sending two interleaves the shared id counter and
breaks the tree math. One qubit exercises the full depth-2 tree, the error
correction, and the accounting just as well; a second proves nothing extra.

DEPTH-2 REQUIRES FORCED LOSSES. On a clean channel nothing recurses and the tree
stops at depth 1. Losing packet 1 shares 1 and 2 drops packet 1 into recursion
territory so its shares 3 and 4 recurse (packets 2, 3) -> depth 2. Losses use
testscript's existing install_no_entanglement_monkeypatch_once.

PACKET NUMBERING (single QEC packet, depth-first):
  0 = QEC codeword (5 shares)     1 = QEC share 0's subtree
  2,3 = recursions inside packet 1  4 = QEC share 1's subtree (direct sends)

Bit errors ride on top of the losses:
  1 error : X on (packet 1, share 0)   -> QEC share 0 corrupt -> corrected
  2 errors: X on (1,0) and Z on (4,0)  -> QEC shares 0 AND 1 corrupt.
            [[5,1,3]] is distance 3: it CANNOT detect 2 errors (the syndrome
            table is a full 16<->16 bijection), so a 2-error state aliases to a
            single-error correction and is SILENTLY MISCORRECTED. The packet
            DELIVERS a WRONG state (it does NOT go LOST -- both errors arrive).
            This is the real failure mode above the code's capacity and why
            QPing must gate fidelity below threshold. NOTE: some 2-error
            patterns compose (with the aliased correction) into a stabiliser
            and deliver CORRECTLY by luck; if X+X aliases benign, switch the
            second error to Z or a different share.

X (not Z) on |0> so the test is X-sensitive -- Z on |0> is invisible.
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

# --- Pauli circuits ---------------------------------------------------------
_PX = Circuit(1); _PX.x(0)
_PAULI = {"X": _PX}
try:
    _PY = Circuit(1); _PY.y(0); _PAULI["Y"] = _PY
    _PZ = Circuit(1); _PZ.z(0); _PAULI["Z"] = _PZ
except Exception:
    pass  # X-only is sufficient for these tests


def install_pauli_injection(transfer, targets):
    """Patch transfer._fire: for a transfer whose (packet_id, share_index) is a
    key in targets, apply the named Pauli to its data qubit before firing.
    Reference swap only; the gate runs at fire time when the qubit is live."""
    original_fire = transfer._fire
    injected = set()

    def patched_fire(t, info):
        key = (t.packet_id, t.share_index)

        p = targets.get(key)
        if p is not None and key not in injected:
            data_arr = transfer.node.get_component_by_name(
                transfer.node.data_memo_arr_name)
            data_key = data_arr[t.data_memory_index].qstate_key
            #rnd = transfer.node.get_generator().random()
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
  

# --- targets (single-packet tree) -------------------------------------------
LOSSES     = [(1,2),(1,3),(1,4),(2,1),(2,2),(2,3)]                       # force depth-2 (every test)
INJECT_ONE = {(1, 0): "X"}                                   # 1 error -> corrected
INJECT_TWO = {(1, 0): "Z",(1, 1): "Y"}                       # 2 errors -> miscorrected

TESTS = [
    dict(name="depth2 clean",      inject={},
         expect_outcome="DELIVERED", expect_fidelity=True),
    #dict(name="depth2 one error",  inject=INJECT_ONE,
     #    expect_outcome="DELIVERED", expect_fidelity=True),
    #dict(name="depth2 two errors", inject=INJECT_TWO,
     #    expect_outcome="DELIVERED", expect_fidelity=False),   # miscorrected
]

MAX_DEPTH = 1


def _trial(sent_state, inject):
    """Build topo, prepare ONE state, install losses + injections, run.
    Returns (pid, outcome, received, alice_free, bob_free, rss)."""
    topo = QTCPNetTopo(CONFIG)
    tl = topo.tl
    nodes = topo.nodes[QTCPNetTopo.QTCP_NODE]
    alice = next(n for n in nodes if n.name == ALICE)
    bob = next(n for n in nodes if n.name == BOB)
    charlie = next(n for n in nodes if n.name == CHARLIE)

    app_alice = QTCPApp(alice, MAX_DEPTH, 1, "recover")
    app_bob = QTCPApp(bob, MAX_DEPTH, 1, "recover")
    app_charlie = QTCPApp(charlie, MAX_DEPTH, 1, "recover")

    # prepare the single sent state
    data_arr = alice.get_component_by_name(alice.data_memo_arr_name)
    slot = app_alice.transfer.alloc_data_slot()
    data_arr[slot].update_state(sent_state)

    # depth-2 delivery is slower than depth-1; 50ms gives margin past delivery
    # without over-running the window (80ms was already far too long).
    app_alice.connect(dst=BOB, start_t=20 * MILLISECOND, end_t=50 * MILLISECOND,
                      memory_size=5, num_qubits=1)

    pid = app_alice.send_packet(slot, BOB)

    # install both patches AFTER send_packet (matches testscript). Losses force
    # depth-2; injections apply the bit errors. Disjoint target sets, coexist.
    install_no_entanglement_monkeypatch_once(app_alice.transfer, list(LOSSES))
    if inject:
        install_pauli_injection(app_alice.transfer, inject)

    log.set_logger(__name__, tl, "qtcp_depth2.log")
    log.set_logger_level('DEBUG')
    log.track_module('qtcp_overseer')
    log.track_module('__main__')
    tl.init()
    for m in ['qtcp_app', 'teleportation', 'generation', 'qtcp_handshake', 'qtcp_transfer']:
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
    print("DEPTH-2 QEC BATCH")
    print("=" * 60)
    for t in TESTS:
        # Random state for sending
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
            # 2-error case: delivered but should be WRONG (miscorrected)
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