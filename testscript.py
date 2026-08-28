"""Manual trial harness for qTCP.

Runs one packet transmission end-to-end with configurable losses,
Pauli injections, and pool-cleanup assertions. Used interactively
for debugging protocol behaviour, reproducing killer seeds, and
sanity-checking new features before they hit the full benchmark.

Provides:
  - CONFIG / ALICE / BOB / CHARLIE constants used by RSStest.py too
  - install_no_entanglement_monkeypatch_once: fail specific shares once
  - install_no_entanglement_monkeypatch_times: fail specific shares N times
  - assert_bob_pool_clean / assert_alice_lost / assert_alice_clean /
    assert_bob_clean: pool-state assertions for after a run
  - random_state: pure single-qubit state for testing amplitude + phase

Not part of the benchmark. To use, edit run_trial() to set up the
scenario you want (state, patches, assertions) and run the script.
"""

import numpy as np

from sequence.topology.qtcp_net_topo import QTCPNetTopo
from sequence.app.qtcp.qtcp_app import QTCPApp
from sequence.app.qtcp.qtcp_transfer import BobState
from sequence.constants import MILLISECOND
from sequence.kernel.quantum_utils import verify_same_state_vector
import sequence.utils.log as log


# NOTE: no set_global_type(SINGLE_HERALDED) here.
# Single-heralded generation asserts bell_diagonal formalism, which
# cannot represent an arbitrary |psi> and so cannot support
# teleportation of a general state. Default generation type works.

CONFIG = "tmp/three_node.json"

ALICE = "router_0"
BOB = "router_2"
CHARLIE = "router_1"


def assert_pool_restored(node, held=0):
    """Data pool on `node` is fully free except for `held` retained slots."""
    t = node.teleport_app
    data_arr = node.get_component_by_name(node.data_memo_arr_name)
    assert len(t.free_data_slots) == len(data_arr) - held, (
        f"{node.name}: leaked {len(data_arr) - held - len(t.free_data_slots)} slot(s)"
    )


def install_no_entanglement_monkeypatch_times(qtcp_transfer_instance, fail_counts):
    """Fail a share N times before letting it succeed.

    Args:
        qtcp_transfer_instance: Alice's QTCPTransfer to patch.
        fail_counts: {(packet_id, share_index): n_fails}. Decremented
            on each matching fire; once a key hits 0 the share fires
            normally.

    Returns the live remaining-count dict (mutated in place) so the
    caller can assert every scheduled failure was consumed after the
    run.

    For scenarios where the SAME share identity must fail across
    multiple attempts -- e.g. exercising multiple local restarts (each
    restart re-fires the same (packet_id, share_index)) or driving a
    packet through fire-fails-to-LOST.
    """
    from sequence.app.qtcp.qtcp_transfer import TransferStatus, FailureReason

    remaining = dict(fail_counts)
    original_fire = qtcp_transfer_instance._fire

    def patched_fire(transfer, info):
        key = (transfer.packet_id, transfer.share_index)
        if remaining.get(key, 0) > 0:
            remaining[key] -= 1
            log.logger.info(
                f"MONKEYPATCH: suppressing fire of transfer "
                f"{transfer.transfer_id} (packet {transfer.packet_id} "
                f"share {transfer.share_index}) -> FAILED(NO_ENTANGLEMENT) "
                f"[{remaining[key]} fail(s) left for this share]"
            )
            # _finish(FAILED, NO_ENTANGLEMENT) matches what a genuine
            # no-pair-in-time failure looks like: slot recycles rather
            # than stranding the pair.
            qtcp_transfer_instance._finish(
                transfer, TransferStatus.FAILED,
                FailureReason.NO_ENTANGLEMENT)
            return
        original_fire(transfer, info)

    qtcp_transfer_instance._fire = patched_fire
    return remaining


def install_no_entanglement_monkeypatch_once(qtcp_transfer_instance, failure_set):
    """Fail each listed share exactly once. A restarted packet's second
    attempt at the same share succeeds normally.

    Args:
        qtcp_transfer_instance: Alice's QTCPTransfer to patch.
        failure_set: iterable of (packet_id, share_index) tuples.
    """
    from sequence.app.qtcp.qtcp_transfer import TransferStatus, FailureReason

    remaining = set(failure_set)
    original_fire = qtcp_transfer_instance._fire

    def patched_fire(transfer, info):
        key = (transfer.packet_id, transfer.share_index)
        if key in remaining:
            remaining.discard(key)
            log.logger.info(
                f"MONKEYPATCH: suppressing fire of transfer "
                f"{transfer.transfer_id} (packet {transfer.packet_id} "
                f"share {transfer.share_index}) -> FAILED(NO_ENTANGLEMENT) "
                f"[one-shot]"
            )
            qtcp_transfer_instance._finish(
                transfer, TransferStatus.FAILED,
                FailureReason.NO_ENTANGLEMENT)
            return
        original_fire(transfer, info)

    qtcp_transfer_instance._fire = patched_fire


def assert_bob_pool_clean(bob_app):
    """After a LOST packet, Bob keeps nothing: pool fully free, no
    ARRIVED records lingering, received_packets empty."""
    t = bob_app.transfer
    overseer = bob_app.overseer
    node = t.node
    data_arr = node.get_component_by_name(node.data_memo_arr_name)

    n_total = len(data_arr)
    n_free = len(t.free_data_slots)
    assert n_free == n_total, (
        f"{node.name}: LEAK -- {n_total - n_free} data slot(s) not freed "
        f"(free={n_free}, total={n_total}). The 0<arrived<K branch did "
        f"not release the orphaned arrival(s)."
    )

    leftover = [k for k, r in t.bob_transfers.items()
                if r.state is BobState.ARRIVED]
    assert not leftover, (
        f"{node.name}: {len(leftover)} ARRIVED record(s) still present "
        f"after LOST -- records not purged: {leftover}"
    )

    assert not overseer.received_packets, (
        f"{node.name}: received_packets should be empty for a LOST "
        f"packet, got {dict(overseer.received_packets)}"
    )

    print(f"PASS: {node.name} pool clean after LOST-with-partial "
          f"({n_free}/{n_total} slots free, no orphaned arrivals)")


def assert_alice_lost(alice_app, packet_id=0):
    """Alice registered `packet_id` as LOST and released her slots."""
    from sequence.app.qtcp.qtcp_overseer import PacketOutcome
    outcome = alice_app.overseer.get_packet_outcome(packet_id)
    assert outcome is PacketOutcome.LOST, (
        f"expected packet {packet_id} LOST, got {outcome}"
    )
    t = alice_app.transfer
    data_arr = t.node.get_component_by_name(t.node.data_memo_arr_name)
    assert len(t.free_data_slots) == len(data_arr), (
        f"{t.node.name}: Alice leaked slots on LOST "
        f"(free={len(t.free_data_slots)}, total={len(data_arr)})"
    )
    print(f"PASS: Alice packet {packet_id} correctly LOST, pool restored")


def assert_alice_clean(alice_app, held=0):
    """Alice's pool is fully free except for `held` retained slots."""
    t = alice_app.transfer
    data_arr = t.node.get_component_by_name(t.node.data_memo_arr_name)
    assert len(t.free_data_slots) + held == len(data_arr), (
        f"{t.node.name}: Alice leaked slots "
        f"(free={len(t.free_data_slots)}, total={len(data_arr)}, held={held})"
    )
    print(f"PASS: Alice has no leaked slots")


def assert_bob_clean(bob_app, held=0):
    """Bob's pool is fully free except for `held` retained slots
    (typically 1 -- the delivered secret)."""
    t = bob_app.transfer
    data_arr = t.node.get_component_by_name(t.node.data_memo_arr_name)
    assert len(t.free_data_slots) + held == len(data_arr), (
        f"{t.node.name}: Bob leaked slots "
        f"(free={len(t.free_data_slots)}, total={len(data_arr)}, held={held})"
    )
    print(f"PASS: Bob has no leaked slots")


def run_trial(psi) -> list:
    """Send psi[0] from ALICE to BOB. Returns [received_state].

    Edit this function to add more sends, install patches, or change
    the assertion suite for a specific scenario.
    """
    topo = QTCPNetTopo(CONFIG)
    tl = topo.tl

    qtcp_nodes = topo.nodes[QTCPNetTopo.QTCP_NODE]
    alice = next(n for n in qtcp_nodes if n.name == ALICE)
    bob = next(n for n in qtcp_nodes if n.name == BOB)
    charlie = next(n for n in qtcp_nodes if n.name == CHARLIE)

    app_alice = QTCPApp(alice, 1, 1, "recover")
    app_bob = QTCPApp(bob, 2, 1, "recover")
    app_charlie = QTCPApp(charlie, 2, 1, "recover")

    i = app_alice.transfer.alloc_data_slot()
    data_arr = alice.get_component_by_name(alice.data_memo_arr_name)
    data_arr[i].update_state(psi[0])

    app_alice.connect(dst=BOB, start_t=20 * MILLISECOND, end_t=80 * MILLISECOND,
                      memory_size=5, num_qubits=1)

    tid1 = app_alice.send_packet(i, BOB)

    log.set_logger(__name__, tl, "qtcp_test.log")
    log.set_logger_level('DEBUG')
    log.track_module('qtcp_overseer')

    tl.init()
    for m in ['qtcp_app', 'teleportation', 'generation',
              'qtcp_handshake', 'qtcp_transfer']:
        log.track_module(m)

    tl.run()

    import resource
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Peak RSS: {peak_kb / 1024:.1f} MB")

    return [app_bob.get_received_packet(ALICE, tid1)]


def random_state(rng: np.random.Generator) -> np.ndarray:
    """A random single-qubit pure state. Tests amplitude AND phase,
    unlike |0>."""
    a = rng.normal() + 1j * rng.normal()
    b = rng.normal() + 1j * rng.normal()
    v = np.array([a, b], dtype=complex)
    return v / np.linalg.norm(v)


if __name__ == "__main__":
    rng = np.random.default_rng(12345)

    # |+> is the minimal good test state: equal superposition, so a
    # broken correction (missing X or Z flip) shows up immediately.
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)

    states = [plus, random_state(rng), random_state(rng)]

    out = run_trial(states)

    for i, state in enumerate(out):
        if state is None:
            print("Send failed")
            continue
        ok = verify_same_state_vector(state, states[i])
        status = "PASS" if ok else "FAIL"
        print(f"[{status}]")
        print(f"    sent:     {states[i]}")
        print(f"    received: {state}")
        print()