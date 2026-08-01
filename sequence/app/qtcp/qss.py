"""(3,5) quantum secret sharing on the [[5,1,3]] five-qubit code.

Encodes a single-qubit secret into five shares such that any three reconstruct
it exactly and any two reveal nothing. Used by qTCP as the loss-recovery layer:
teleportation consumes the sender's qubit, so a lost qubit cannot be resent --
instead the payload is split into five shares and any three suffice.

Construction follows Graves, Nelson & Chitambar, "Implementing Quantum Secret
Sharing on Current Hardware" (arXiv:2410.11640), which builds the encoding
unitary U from the ring-state stabilizers of the five-qubit code. The
correction tables in this module are computed against *that* U -- they are not
interchangeable with tables computed for a different encoder (e.g. Laflamme's),
since the two differ by a Clifford.

Stabilizer generators (Graves Table I):
    G1 = X Z Z X I
    G2 = I X Z Z X
    G3 = X I X Z Z
    G4 = Z X I X Z
    Zbar = Z Z Z Z Z

Qubit convention (from Graves Fig. 4):
    qubits 0..3  -- ancillas; measured after decoding to give the syndrome
    qubit  4     -- the secret; the correction R_k is applied here

Deliberately free of simulator state: this module builds Circuit objects and
looks up tables. It never touches nodes, timelines or the quantum manager --
that is the app's job. Keeping the boundary here is what makes the encode ->
erase -> decode round trip testable without running a simulation.
"""

from sequence.components.circuit import Circuit


# ======================================================================
# constants
# ======================================================================

N_SHARES = 5      # code length
K_THRESHOLD = 3   # shares needed to reconstruct
N_ANCILLA = 4     # qubits measured for the syndrome
SECRET_INDEX = 4  # position of the secret within the five


# Stabilizer generators, as Pauli strings. Kept for verifying the encoder:
# U|00000> should be a +1 eigenstate of all four.
STABILIZERS = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
LOGICAL_Z = "ZZZZZ"


# ======================================================================
# circuits
# ======================================================================

def _build_encoder() -> Circuit:
    """The encoding unitary U from Graves Fig. 4.

    Maps the secret in qubit 4 (with qubits 0-3 in |0>) to the five-qubit
    logical state. Per Eq. (6), read right-to-left: Hadamards on qubits 0-3,
    then CNOTs, then CZs on ring-adjacent pairs.

    TODO: transcribe from the figure. Watch the dot-dot vs dot-XOR
    distinction -- dot-dot is CZ (symmetric), dot-XOR is CNOT.
    """
    circuit = Circuit(N_SHARES)
    circuit.h(0)
    circuit.h(1)
    circuit.h(2)
    circuit.h(3)
    circuit.z(4)
    
    circuit.cx(0,4)
    
    circuit.cx(1,4)
    
    circuit.cz(0,1)
    circuit.cx(2,4)

    circuit.cz(1,2)
    circuit.cx(3,4)

    circuit.cz(2,3)
    circuit.cz(3,4)
    circuit.cz(0,4)



    return circuit


def _build_decoder() -> Circuit:
    """U-dagger: the encoder run backwards.

    H, CNOT and CZ are all self-inverse, so this is the same gate list in
    reverse order. Kept as a separate function rather than derived at runtime
    so the reversal is explicit and inspectable.
"""
    circuit = Circuit(5)
    for gate in reversed(ENCODER.gates):
        circuit.gates.append(gate)
    for q in range(4):
        circuit.measure(q)
    return circuit


ENCODER = _build_encoder()   # set to _build_encoder() once implemented
DECODER = _build_decoder()   # set to _build_decoder() once implemented


# ======================================================================
# correction tables
#
# One table per erasure pattern: {syndrome bits -> Pauli on the secret}.
#
# The syndrome is the computational-basis measurement of qubits 0-3 after
# decoding. The correction is a single-qubit Pauli (U is Clifford, so
# U-dagger E-dagger U restricted to the secret is always a Pauli).
#
# Each table has 16 entries: {I,X,Y,Z} on each of the two erased positions,
# giving 16 Pauli combinations, matched one-to-one with the 16 syndromes.

# ======================================================================




# keys are (i, j) with i < j, 0-indexed 
TABLES: dict[tuple[int, int], dict[tuple[int, ...], str]] = {
        (0, 1): {
        (0, 0, 0, 0): 'I',
        (0, 0, 0, 1): 'Z',
        (0, 0, 1, 0): 'X',
        (0, 0, 1, 1): 'Y',
        (0, 1, 0, 0): 'I',
        (0, 1, 0, 1): 'Z',
        (0, 1, 1, 0): 'X',
        (0, 1, 1, 1): 'Y',
        (1, 0, 0, 0): 'I',
        (1, 0, 0, 1): 'Z',
        (1, 0, 1, 0): 'X',
        (1, 0, 1, 1): 'Y',
        (1, 1, 0, 0): 'I',
        (1, 1, 0, 1): 'Z',
        (1, 1, 1, 0): 'X',
        (1, 1, 1, 1): 'Y',
    },
    (0, 2): {
        (0, 0, 0, 0): 'I',
        (0, 0, 0, 1): 'Y',
        (0, 0, 1, 0): 'I',
        (0, 0, 1, 1): 'Y',
        (0, 1, 0, 0): 'Z',
        (0, 1, 0, 1): 'X',
        (0, 1, 1, 0): 'Z',
        (0, 1, 1, 1): 'X',
        (1, 0, 0, 0): 'I',
        (1, 0, 0, 1): 'Y',
        (1, 0, 1, 0): 'I',
        (1, 0, 1, 1): 'Y',
        (1, 1, 0, 0): 'Z',
        (1, 1, 0, 1): 'X',
        (1, 1, 1, 0): 'Z',
        (1, 1, 1, 1): 'X',
    },
    (0, 3): {
        (0, 0, 0, 0): 'I',
        (0, 0, 0, 1): 'I',
        (0, 0, 1, 0): 'Y',
        (0, 0, 1, 1): 'Y',
        (0, 1, 0, 0): 'Y',
        (0, 1, 0, 1): 'Y',
        (0, 1, 1, 0): 'I',
        (0, 1, 1, 1): 'I',
        (1, 0, 0, 0): 'I',
        (1, 0, 0, 1): 'I',
        (1, 0, 1, 0): 'Y',
        (1, 0, 1, 1): 'Y',
        (1, 1, 0, 0): 'Y',
        (1, 1, 0, 1): 'Y',
        (1, 1, 1, 0): 'I',
        (1, 1, 1, 1): 'I',
    },
    (0, 4): {
        (0, 0, 0, 0): 'I',
        (0, 0, 0, 1): 'X',
        (0, 0, 1, 0): 'Z',
        (0, 0, 1, 1): 'Y',
        (0, 1, 0, 0): 'X',
        (0, 1, 0, 1): 'I',
        (0, 1, 1, 0): 'Y',
        (0, 1, 1, 1): 'Z',
        (1, 0, 0, 0): 'I',
        (1, 0, 0, 1): 'X',
        (1, 0, 1, 0): 'Z',
        (1, 0, 1, 1): 'Y',
        (1, 1, 0, 0): 'X',
        (1, 1, 0, 1): 'I',
        (1, 1, 1, 0): 'Y',
        (1, 1, 1, 1): 'Z',
    },
    (1, 2): {
        (0, 0, 0, 0): 'I',
        (0, 0, 0, 1): 'X',
        (0, 0, 1, 0): 'I',
        (0, 0, 1, 1): 'X',
        (0, 1, 0, 0): 'I',
        (0, 1, 0, 1): 'X',
        (0, 1, 1, 0): 'I',
        (0, 1, 1, 1): 'X',
        (1, 0, 0, 0): 'X',
        (1, 0, 0, 1): 'I',
        (1, 0, 1, 0): 'X',
        (1, 0, 1, 1): 'I',
        (1, 1, 0, 0): 'X',
        (1, 1, 0, 1): 'I',
        (1, 1, 1, 0): 'X',
        (1, 1, 1, 1): 'I',
    },
    (1, 3): {
        (0, 0, 0, 0): 'I',
        (0, 0, 0, 1): 'I',
        (0, 0, 1, 0): 'Z',
        (0, 0, 1, 1): 'Z',
        (0, 1, 0, 0): 'I',
        (0, 1, 0, 1): 'I',
        (0, 1, 1, 0): 'Z',
        (0, 1, 1, 1): 'Z',
        (1, 0, 0, 0): 'Y',
        (1, 0, 0, 1): 'Y',
        (1, 0, 1, 0): 'X',
        (1, 0, 1, 1): 'X',
        (1, 1, 0, 0): 'Y',
        (1, 1, 0, 1): 'Y',
        (1, 1, 1, 0): 'X',
        (1, 1, 1, 1): 'X',
    },
    (1, 4): {
        (0, 0, 0, 0): 'I',
        (0, 0, 0, 1): 'Y',
        (0, 0, 1, 0): 'Y',
        (0, 0, 1, 1): 'I',
        (0, 1, 0, 0): 'I',
        (0, 1, 0, 1): 'Y',
        (0, 1, 1, 0): 'Y',
        (0, 1, 1, 1): 'I',
        (1, 0, 0, 0): 'Z',
        (1, 0, 0, 1): 'X',
        (1, 0, 1, 0): 'X',
        (1, 0, 1, 1): 'Z',
        (1, 1, 0, 0): 'Z',
        (1, 1, 0, 1): 'X',
        (1, 1, 1, 0): 'X',
        (1, 1, 1, 1): 'Z',
    },
    (2, 3): {
        (0, 0, 0, 0): 'I',
        (0, 0, 0, 1): 'I',
        (0, 0, 1, 0): 'I',
        (0, 0, 1, 1): 'I',
        (0, 1, 0, 0): 'X',
        (0, 1, 0, 1): 'X',
        (0, 1, 1, 0): 'X',
        (0, 1, 1, 1): 'X',
        (1, 0, 0, 0): 'Z',
        (1, 0, 0, 1): 'Z',
        (1, 0, 1, 0): 'Z',
        (1, 0, 1, 1): 'Z',
        (1, 1, 0, 0): 'Y',
        (1, 1, 0, 1): 'Y',
        (1, 1, 1, 0): 'Y',
        (1, 1, 1, 1): 'Y',
    },
    (2, 4): {
        (0, 0, 0, 0): 'I',
        (0, 0, 0, 1): 'Z',
        (0, 0, 1, 0): 'I',
        (0, 0, 1, 1): 'Z',
        (0, 1, 0, 0): 'Y',
        (0, 1, 0, 1): 'X',
        (0, 1, 1, 0): 'Y',
        (0, 1, 1, 1): 'X',
        (1, 0, 0, 0): 'Y',
        (1, 0, 0, 1): 'X',
        (1, 0, 1, 0): 'Y',
        (1, 0, 1, 1): 'X',
        (1, 1, 0, 0): 'I',
        (1, 1, 0, 1): 'Z',
        (1, 1, 1, 0): 'I',
        (1, 1, 1, 1): 'Z',
    },
    (3, 4): {
        (0, 0, 0, 0): 'I',
        (0, 0, 0, 1): 'I',
        (0, 0, 1, 0): 'X',
        (0, 0, 1, 1): 'X',
        (0, 1, 0, 0): 'Z',
        (0, 1, 0, 1): 'Z',
        (0, 1, 1, 0): 'Y',
        (0, 1, 1, 1): 'Y',
        (1, 0, 0, 0): 'X',
        (1, 0, 0, 1): 'X',
        (1, 0, 1, 0): 'I',
        (1, 0, 1, 1): 'I',
        (1, 1, 0, 0): 'Y',
        (1, 1, 0, 1): 'Y',
        (1, 1, 1, 0): 'Z',
        (1, 1, 1, 1): 'Z',
    },
}
def _one_qubit_circuit(type: str):
    circuit = Circuit(1)
    if type == "x":
        circuit.x(0)
    elif type == "y":
        circuit.y(0)
    elif type == "z":
        circuit.z(0)
    else:
        raise ValueError(f"correction type {type} doesn't exist")

    return circuit


_CORRECTIONS = {
    "X": _one_qubit_circuit("x"),
    "Y": _one_qubit_circuit("y"),
    "Z": _one_qubit_circuit("z"),
    "I": None,          # nothing to run
}

def correction_for(missing: tuple[int, int], syndrome: tuple[int, ...]) -> Circuit | None:
    """Look up the Pauli correction for a given erasure pattern and syndrome.

    Args:
        missing: the two erased share indices, 0-indexed, sorted.
        syndrome: the four measured ancilla bits.
    """
    table = TABLES.get(tuple(sorted(missing)))
    if table is None:
        raise KeyError(f"no correction table for erasure {missing}")
    correction = table.get(tuple(syndrome))
    if correction is None:
        raise KeyError(f"syndrome {syndrome} not in table for {missing}")
    return _CORRECTIONS[correction]

