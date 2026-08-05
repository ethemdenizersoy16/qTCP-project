"""Teleport a known state from router_0 to router_2 across a 3-node linear topology.

Substrate proof for the qTCP project: confirms that the entanglement pipeline
(generation -> swapping at router_1 -> end-to-end pair) plus the existing
TeleportProtocol delivers a data qubit intact.

Modeled on tests/app/test_teleport.py, which is the known-working reference.
Key differences from that test: linear 3-node chain (so entanglement swapping
at the middle router is actually exercised) rather than a star.
"""

import numpy as np

from sequence.topology.qtcp_net_topo import QTCPNetTopo
from sequence.app.qtcp.qtcp_app import QTCPApp
from sequence.app.qtcp.qtcp_transfer import BobState

from sequence.constants import MILLISECOND,MICROSECOND
from sequence.kernel.quantum_utils import verify_same_state_vector
import sequence.utils.log as log
from sequence.kernel.process import Process
from sequence.kernel.event import Event

def assert_pool_restored(node, held=0):
    t = node.teleport_app
    data_arr = node.get_component_by_name(node.data_memo_arr_name)
    assert len(t.free_data_slots) == len(data_arr) - held, (
        f"{node.name}: leaked {len(data_arr) - held - len(t.free_data_slots)} slot(s)"
    )



# NOTE: no set_global_type(SINGLE_HERALDED) here.
# Single-heralded generation asserts bell_diagonal formalism, which cannot
# represent an arbitrary |psi> and so cannot support teleportation.
# The working reference test leaves generation at its default. Do the same.


CONFIG = "tmp/three_node.json"

ALICE = "router_0"
BOB = "router_2"
CHARLIE = "router_1"

# Low target fidelity so the purification rule's condition never fires --
# keeps the pipeline minimal for the correctness gate. (Stolen from the
# reference test, where it's commented "not to activate distillation".)
TARGET_FIDELITY = 0.01

class _Deferred:
    def __init__(self, fn):
        self.fn = fn
        self.result = None
        self.error = None          # capture an exception raised inside the event
    def run(self):
        try:
            self.result = self.fn()
        except Exception as e:
            self.error = e

def install_no_entanglement_monkeypatch_times(qtcp_transfer_instance, fail_counts):
    """Like install_no_entanglement_monkeypatch_once, but each (packet_id,
    share_index) entry fails a SPECIFIED NUMBER of times before succeeding.

    For tests where the SAME share identity must fail across multiple attempts
    -- e.g. multiple local restarts (each restart re-fires the same
    (packet_id, share_index); triggering a second restart requires it to fail
    again), or fire-fails-to-LOST.

    Args:
        qtcp_transfer_instance: Alice's QTCPTransfer to patch.
        fail_counts: dict mapping (packet_id, share_index) -> number of times
            to fail that identity. Decremented on each matching fire; once a
            key's count reaches 0 the share fires normally.

    Returns the live remaining-count dict (mutated in place), so the caller can
    assert every scheduled failure was consumed after the run.
    """
    from sequence.app.qtcp.qtcp_transfer import TransferStatus, FailureReason
    from sequence.utils import log

    remaining = dict(fail_counts)
    original_fire = qtcp_transfer_instance._fire

    def patched_fire(transfer, info):
        key = (transfer.packet_id, transfer.share_index)
        if remaining.get(key, 0) > 0:
            remaining[key] -= 1
            log.logger.info(
                f"MONKEYPATCH: suppressing fire of transfer "
                f"{transfer.transfer_id} (packet {transfer.packet_id} share "
                f"{transfer.share_index}) -> FAILED(NO_ENTANGLEMENT) "
                f"[{remaining[key]} fail(s) left for this share]"
            )
            # Release the pair get_memory handed us -- same as the _once
            # variant. A suppressed fire that neither teleports nor releases
            # strands the pair; _finish(FAILED, NO_ENTANGLEMENT) recycles the
            # slot, matching a genuine no-pair-in-time failure.
            from sequence.resource_management.memory_manager import MemoryInfo

            qtcp_transfer_instance._finish(
                transfer, TransferStatus.FAILED,
                FailureReason.NO_ENTANGLEMENT)
            return
        original_fire(transfer, info)

    qtcp_transfer_instance._fire = patched_fire

    return remaining

def install_no_entanglement_monkeypatch_once(qtcp_transfer_instance, failure_set):
    """Same as install_no_entanglement_monkeypatch, but each (packet_id,
    share_index) entry fails only ONCE -- consumed on use. A restarted
    packet's second attempt at the same share succeeds normally.

    This is the variant for testing local restart end-to-end: fail the first
    attempt's shares, let the restarted attempt deliver.
    """
    from sequence.app.qtcp.qtcp_transfer import TransferStatus, FailureReason
    from sequence.utils import log

    remaining = set(failure_set)
    original_fire = qtcp_transfer_instance._fire

    def patched_fire(transfer, info):
        key = (transfer.packet_id, transfer.share_index)
        if key in remaining:            # (or `in failure_set` for the persistent variant)
            remaining.discard(key)
            log.logger.info(
                f"MONKEYPATCH: suppressing fire of transfer "
                f"{transfer.transfer_id} (packet {transfer.packet_id} share "
                f"{transfer.share_index}) -> FAILED(NO_ENTANGLEMENT) [one-shot]"
            )
            # Release the pair get_memory handed us. Under the current
            # get_memory (consume-or-release, no idle pairs), a suppressed
            # fire that neither teleports nor releases strands the pair --
            # the slot dies (no regeneration, no fresh edge) and later
            # transfers starve. Releasing to RAW matches what a genuine
            # no-pair-in-time failure looks like: the slot recycles.
            from sequence.resource_management.memory_manager import MemoryInfo

            qtcp_transfer_instance._finish(
                transfer, TransferStatus.FAILED,
                FailureReason.NO_ENTANGLEMENT)
            return
        original_fire(transfer, info)

    qtcp_transfer_instance._fire = patched_fire
def assert_bob_pool_clean(bob_app):
    """Bob's data pool must be fully restored -- a LOST packet leaves nothing
    behind. No delivered secret to keep (LOST, not delivered)."""
    t = bob_app.transfer          # QTCPTransfer: slots + bob_transfers live here
    overseer = bob_app.overseer   # received_packets lives here
    node = t.node
    data_arr = node.get_component_by_name(node.data_memo_arr_name)

    # 1. Every data slot back in the free pool.
    n_total = len(data_arr)
    n_free = len(t.free_data_slots)
    assert n_free == n_total, (
        f"{node.name}: LEAK -- {n_total - n_free} data slot(s) not freed "
        f"(free={n_free}, total={n_total}). The 0<arrived<K branch did not "
        f"release the orphaned arrival(s)."
    )

    # 2. No lingering ARRIVED records (all purged).
    leftover = [k for k, r in t.bob_transfers.items()
                if r.state is BobState.ARRIVED]
    assert not leftover, (
        f"{node.name}: {len(leftover)} ARRIVED record(s) still present "
        f"after LOST -- records not purged: {leftover}"
    )

    # 3. Bob received NO packet (it was LOST, not delivered).
    assert not overseer.received_packets, (
        f"{node.name}: received_packets should be empty for a LOST packet, "
        f"got {dict(overseer.received_packets)}"
    )

    print(f"PASS: {node.name} pool clean after LOST-with-partial "
          f"({n_free}/{n_total} slots free, no orphaned arrivals)")
 
 
# Also confirm Alice's side registered the LOST outcome and freed her slots:
def assert_alice_lost(alice_app, packet_id=0):
    outcome = alice_app.overseer.get_packet_outcome(packet_id)
    from sequence.app.qtcp.qtcp_overseer import PacketOutcome
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

def assert_alice_clean(alice_app, held = 0):

    t = alice_app.transfer
    data_arr = t.node.get_component_by_name(t.node.data_memo_arr_name)
    assert len(t.free_data_slots) + held == len(data_arr), (
        f"{t.node.name}: Alice leaked slots on LOST "
        f"(free={len(t.free_data_slots)}, total={len(data_arr)})"
    )
    print(f"PASS: Alice has no leaked slots")
def assert_bob_clean(bob_app, held=0):

    t = bob_app.transfer
    data_arr = t.node.get_component_by_name(t.node.data_memo_arr_name)
    assert len(t.free_data_slots) + held == len(data_arr), (
        f"{t.node.name}: Bob leaked slots on LOST "
        f"(free={len(t.free_data_slots)}, total={len(data_arr)})"
    )
    print(f"PASS: Bob has no leaked slots")


def run_trial(psi) -> np.ndarray:
    """Teleport psi from ALICE to BOB. Returns the state Bob ends up holding."""

    topo = QTCPNetTopo(CONFIG)
    tl = topo.tl

    qtcp_nodes = topo.nodes[QTCPNetTopo.QTCP_NODE]
    alice = next(n for n in qtcp_nodes if n.name == ALICE)
    bob = next(n for n in qtcp_nodes if n.name == BOB)
    charlie = next(n for n in qtcp_nodes if n.name == CHARLIE)
  

    app_alice = QTCPApp(alice, 2,1,"recover")
    app_bob = QTCPApp(bob, 2,1,"recover")
    app_charlie = QTCPApp(charlie, 2,1, "recover")



    # Attach QTCPApp to both endpoints.
    #    The constructor registers itself on the node (node.teleport_app = self),
    #    so no external bookkeeping dict is needed.
    





    #    Prepare |psi> in Alice's data memory.
    #    Memory.update_state() takes a state vector directly -- no circuit needed.
    #i = app_alice.transfer.alloc_data_slot()
    #data_arr = alice.get_component_by_name(alice.data_memo_arr_name)
    #data_arr[i].update_state(psi[0])
    i = app_alice.transfer.alloc_data_slot()
    data_arr = alice.get_component_by_name(alice.data_memo_arr_name)
    data_arr[i].update_state(psi[0])

    
    

    #j = app_alice.transfer.alloc_data_slot()
    #data_arr2 = bob.get_component_by_name(bob.data_memo_arr_name)
    #data_arr[j].update_state(psi[1])


    app_alice.connect(dst=BOB, start_t=20 *MILLISECOND, end_t=35* MILLISECOND ,  memory_size=5, num_qubits = 1)
    #app_alice.connect(dst=CHARLIE, start_t=20 *MILLISECOND, end_t=80*MILLISECOND ,  memory_size=5, num_qubits = 1)
    #app_bob.connect(dst=ALICE, start_t=20 *MILLISECOND, end_t=80*MILLISECOND , memory_size=5, num_qubits = 1)

  

    tid1 = app_alice.send_packet(i,BOB)
    #tid2 = app_alice.send_packet(j, CHARLIE)
    #tid2 = app_bob.send_packet(j,ALICE)
    #install_no_entanglement_monkeypatch_once(app_alice.transfer, [(1, 1),(1,2)])
    #counts = install_no_entanglement_monkeypatch_times( app_alice.transfer, {(0, 0): 3, (0, 1): 3})


    log.set_logger(__name__, tl, "qtcp_test.log")   
    log.set_logger_level('DEBUG')
    log.track_module('qtcp_overseer')     


    tl.init()
    for m in ['qtcp_app', 'teleportation', 'generation', "qtcp_handshake","qtcp_transfer"]:
        log.track_module(m)

    tl.run()

    #assert_alice_clean(app_alice,1)
    #assert_bob_clean(app_bob, 0)
    #assert all(v == 0 for v in counts.values()), f"unused failures: {counts}"
    import resource
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Peak RSS: {peak_kb / 1024:.1f} MB")

    state = [app_bob.get_received_packet(ALICE, tid1)]

    #assert_pool_restored(bob, held=1)
    return state


def random_state(rng: np.random.Generator) -> np.ndarray:
    """A random single-qubit pure state. Tests amplitude *and* phase, unlike |0>."""
    a = rng.normal() + 1j * rng.normal()
    b = rng.normal() + 1j * rng.normal()
    v = np.array([a, b], dtype=complex)
    return v / np.linalg.norm(v)


if __name__ == "__main__":
    rng = np.random.default_rng(12345)

    # |+> is the minimal good test state: equal superposition, so a broken
    # correction (missing X or Z flip) shows up immediately.
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)

    test_states = {
        "|+>": plus,
        "random_1": random_state(rng),
        "random_2": random_state(rng),
    }
    
    states = [plus, test_states["random_1"],test_states["random_2"]]

    # for label, psi in test_states.items():
    
    out = run_trial(states)
    i= 0
    for state in out:
        if state is None:
            print("Send failed")
            i = i +1
            continue
        ok = verify_same_state_vector(state, states[i])
        status = "PASS" if ok else "FAIL"
        print(f"[{status}]")
        print(f"    sent:     {states[i]}")
        print(f"    received: {state}")
        print()
        i = i +1
    
