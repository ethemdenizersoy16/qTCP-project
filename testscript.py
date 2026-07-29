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
from sequence.app.qtcp_transfer import QTCPTransfer, QTCPMsgType, QTCPMessage
from sequence.app.qtcp_overseer import QTCPOverseer
from sequence.app.qtcp_handshake import QTCPHandshake
from sequence.constants import MILLISECOND,MICROSECOND
from sequence.kernel.quantum_utils import verify_same_state_vector
import sequence.utils.log as log




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


def install_no_entanglement_monkeypatch(qtcp_transfer_instance, failure_set):
    """Faithful NO_ENTANGLEMENT failure injection: matching transfers never
    fire at all.

    Wraps _fire. When a matching (packet_id, share_index) transfer is about to
    fire, instead of running the Bell measurement it goes straight to
    _finish(FAILED, NO_ENTANGLEMENT). Consequences, all matching a real
    no-pair-in-time failure:

      - No Bell measurement: Alice's qubit is untouched, still in its slot
        (_finish preserves the slot for NO_ENTANGLEMENT).
      - No SEND_NOTICE: Bob never hears about the share, has no record.
      - No teleport: nothing arrives at Bob's memories.
      - The entangled pair that get_memory offered is left alone.

    This is the patch to use for testing local restart -- the earlier
    _finish-flipping patch let the qubit physically teleport to Bob first,
    which leaves live remote entanglement that corrupts Alice's local decode.

    Args:
        qtcp_transfer_instance: sender-side QTCPTransfer.
        failure_set: iterable of (packet_id, share_index) tuples.

    Note: each matching share fails EVERY time it is fired. A restarted
    packet reuses the same packet_id and share_index space, so (0, 0) will
    also kill share 0 of packet 0's second attempt. To fail only the first
    attempt, use install_no_entanglement_monkeypatch_once instead.
    """
    from sequence.app.qtcp_transfer import TransferStatus, FailureReason
    from sequence.utils import log

    failure_set = set(failure_set)
    original_fire = qtcp_transfer_instance._fire

    def patched_fire(transfer, info):
        key = (transfer.packet_id, transfer.share_index)
        if key in failure_set:
            log.logger.info(
                f"MONKEYPATCH: suppressing fire of transfer "
                f"{transfer.transfer_id} (packet {transfer.packet_id} share "
                f"{transfer.share_index}) -> FAILED(NO_ENTANGLEMENT), "
                f"qubit untouched, Bob not notified"
            )
            qtcp_transfer_instance._finish(
                transfer, TransferStatus.FAILED,
                FailureReason.NO_ENTANGLEMENT)
            return
        original_fire(transfer, info)

    qtcp_transfer_instance._fire = patched_fire


def install_no_entanglement_monkeypatch_once(qtcp_transfer_instance, failure_set):
    """Same as install_no_entanglement_monkeypatch, but each (packet_id,
    share_index) entry fails only ONCE -- consumed on use. A restarted
    packet's second attempt at the same share succeeds normally.

    This is the variant for testing local restart end-to-end: fail the first
    attempt's shares, let the restarted attempt deliver.
    """
    from sequence.app.qtcp_transfer import TransferStatus, FailureReason
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

def run_trial(psi) -> np.ndarray:
    """Teleport psi from ALICE to BOB. Returns the state Bob ends up holding."""

    topo = QTCPNetTopo(CONFIG)
    tl = topo.tl

    qtcp_nodes = topo.nodes[QTCPNetTopo.QTCP_NODE]
    alice = next(n for n in qtcp_nodes if n.name == ALICE)
    bob = next(n for n in qtcp_nodes if n.name == BOB)
    charlie = next(n for n in qtcp_nodes if n.name == CHARLIE)
  

 




    # Attach QTCPApp to both endpoints.
    #    The constructor registers itself on the node (node.teleport_app = self),
    #    so no external bookkeeping dict is needed.
    alice_transfer = QTCPTransfer(alice,rto=100_000_000)
    bob_transfer=QTCPTransfer(bob,rto=100_000_000)
    app_alice = QTCPOverseer(alice_transfer)
    app_bob = QTCPOverseer(bob_transfer)
    alice_handshake = QTCPHandshake( alice_transfer)
    bob_handshake = QTCPHandshake( bob_transfer)
    app_charlie = QTCPOverseer(QTCPTransfer(charlie,rto=100_000_000))





    #    Prepare |psi> in Alice's data memory.
    #    Memory.update_state() takes a state vector directly -- no circuit needed.
    i = app_alice.app.alloc_data_slot()
    data_arr = alice.get_component_by_name(alice.data_memo_arr_name)
    data_arr[i].update_state(psi[0])

    j = app_bob.app.alloc_data_slot()
    data_arr2 = bob.get_component_by_name(bob.data_memo_arr_name)
    data_arr2[j].update_state(psi[1])


    alice_handshake.connect(dst=BOB, start_t=20 *MILLISECOND, end_t=80*MILLISECOND ,  memory_size=5, payload = 10)
    #bob_handshake.connect(dst=ALICE, start_t=20 *MILLISECOND, end_t=500*MILLISECOND , memory_size=5, payload=10)


    tid1 = app_alice.send_packet(i, BOB)
   
    #tid2 = app_bob.send_packet(j,ALICE)
    install_no_entanglement_monkeypatch_once(app_alice.app, [(0, 1), (0, 2)])



    log.set_logger(__name__, tl, "qtcp_test.log")   
    log.set_logger_level('DEBUG')
    log.track_module('qtcp_overseer')     


    tl.init()
    for m in ['qtcp_transfer', 'teleportation', 'generation', "qtcp_handshake"]:
        log.track_module(m)

    tl.run()


    
    state = [app_bob.get_received_packet(ALICE, tid1)]
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

