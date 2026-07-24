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




def run_trial(psi) -> np.ndarray:
    """Teleport psi from ALICE to BOB. Returns the state Bob ends up holding."""

    topo = QTCPNetTopo(CONFIG)
    tl = topo.tl

    qtcp_nodes = topo.nodes[QTCPNetTopo.QTCP_NODE]
    alice = next(n for n in qtcp_nodes if n.name == ALICE)
    bob = next(n for n in qtcp_nodes if n.name == BOB)
  

 




    # Attach QTCPApp to both endpoints.
    #    The constructor registers itself on the node (node.teleport_app = self),
    #    so no external bookkeeping dict is needed.
    app_alice = QTCPOverseer(QTCPTransfer(alice,rto=100_000_000))
    app_bob = QTCPOverseer(QTCPTransfer(bob,rto=100_000_000))
   





    #    Prepare |psi> in Alice's data memory.
    #    Memory.update_state() takes a state vector directly -- no circuit needed.
    i = app_alice.app.alloc_data_slot()
    data_arr = alice.get_component_by_name(alice.data_memo_arr_name)
    data_arr[i].update_state(psi[0])

    j = app_alice.app.alloc_data_slot()
    data_arr[j].update_state(psi[1])


    app_alice.app.start(responder=BOB, start_t=20 *MILLISECOND, end_t=250*MILLISECOND , memory_size=5, fidelity=0.01)
    

    tid1 = app_alice.send_packet(i, BOB)





    log.set_logger(__name__, tl, "qtcp_test.log")   
    log.set_logger_level('DEBUG')
    log.track_module('qtcp_overseer')     


    tl.init()
    for m in ['qtcp_transfer', 'teleportation', 'generation']:
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