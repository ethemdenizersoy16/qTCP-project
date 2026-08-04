"""[[5,1,3]] error-correction (correction mode) for the qTCP outer layer.

The QSS module (qss.py) uses the SAME five-qubit code in ERASURE mode: it knows
which shares are missing and reconstructs from the survivors. This module uses
it in CORRECTION mode: all five shares are present, at most one carries an
unknown single-qubit Pauli error, and we find and fix it by measuring the code's
four stabiliser generators.

The two modes are not interchangeable. Erasure decode is decode-then-measure-
ancillas-then-correct-the-secret, keyed on which positions were erased.
Correction decode is measure-stabilisers-on-the-intact-codeword-then-correct-
the-flagged-code-qubit-then-decode, keyed on the syndrome alone. This file
provides the correction-mode pieces; qss.py keeps the erasure-mode pieces.

Everything here was validated in numpy against every single-qubit Pauli error on
every position (see qec_correction_mode.py): all 16 syndromes map to the correct
(position, Pauli) and recover the secret at fidelity 1. The syndrome-extraction
circuits below use only gates SeQUeNCe's Circuit exposes (SNOT/H, S, Sdg, CNOT,
CZ); controlled-Y is decomposed as Sdg . CX . S on the target.

Convention (must not drift -- the table is derived against it):
    STABILIZERS = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]  as (G1, G2, G3, G4)
    syndrome bit k == 1  iff generator G_k's ancilla measures 1
                         iff the codeword sits in G_k's -1 eigenspace
                         iff the error anticommutes with G_k
    positions 0..4 index the five code qubits in code order (NOT arrival order)
"""

from sequence.components.circuit import Circuit


# ======================================================================
# constants -- shared with qss.py's code, restated here for locality
# ======================================================================
N_SHARES = 5
SECRET_INDEX = 4          # position of the secret within the five (qss convention)
N_GENERATORS = 4          # stabiliser generators / syndrome bits

STABILIZERS = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]

# One reused ancilla sits at circuit index N_SHARES (= 5), appended after the
# five code qubits. Each stabiliser-measurement circuit therefore has 6 qubits.
ANCILLA_INDEX = N_SHARES


# ======================================================================
# syndrome -> (position, Pauli) correction table
#
# Derived from the stabilisers by commute/anticommute (see
# derive_correction_table.py). [[5,1,3]] is a perfect code, so the 15 nonzero
# 4-bit syndromes map one-to-one onto the 15 single-qubit Paulis; (0,0,0,0)
# means no detected error. The Pauli to APPLY equals the detected error (X,Y,Z
# are self-inverse).
# ======================================================================
CORRECTION_TABLE: dict[tuple[int, int, int, int], tuple[int, str] | None] = {
    (0, 0, 0, 0): None,
    (0, 0, 0, 1): (0, "X"),
    (0, 0, 1, 0): (2, "Z"),
    (0, 0, 1, 1): (4, "X"),
    (0, 1, 0, 0): (4, "Z"),
    (0, 1, 0, 1): (1, "Z"),
    (0, 1, 1, 0): (3, "X"),
    (0, 1, 1, 1): (4, "Y"),
    (1, 0, 0, 0): (1, "X"),
    (1, 0, 0, 1): (3, "Z"),
    (1, 0, 1, 0): (0, "Z"),
    (1, 0, 1, 1): (0, "Y"),
    (1, 1, 0, 0): (2, "X"),
    (1, 1, 0, 1): (1, "Y"),
    (1, 1, 1, 0): (2, "Y"),
    (1, 1, 1, 1): (3, "Y"),
}


# ======================================================================
# stabiliser-measurement circuits (one per generator)
#
# For generator g, measured with the ancilla at ANCILLA_INDEX:
#     H(anc)
#     for each non-I position p in g:
#         'X' at p : CX(anc, p)
#         'Z' at p : CZ(anc, p)
#         'Y' at p : Sdg(p); CX(anc, p); S(p)      # controlled-Y decomposition
#     H(anc)
#     measure(anc)
# The ancilla outcome is the syndrome bit for g. Measuring a stabiliser this way
# commutes with the code space, so it does not disturb the logical state.
# ======================================================================
def _build_stabilizer_circuit(generator: str) -> Circuit:
    """Build the 6-qubit measurement circuit for one stabiliser generator."""
    circuit = Circuit(N_SHARES + 1)   # 5 code qubits + 1 ancilla
    anc = ANCILLA_INDEX

    circuit.h(anc)
    for pos, ch in enumerate(generator):
        if ch == "I":
            continue
        if ch == "X":
            circuit.cx(anc, pos)
        elif ch == "Z":
            circuit.cz(anc, pos)
        elif ch == "Y":
            # controlled-Y(anc -> pos) = Sdg(pos) . CX(anc,pos) . S(pos)
            circuit.sdg(pos)
            circuit.cx(anc, pos)
            circuit.s(pos)
        else:
            raise ValueError(f"unexpected stabiliser char {ch!r} in {generator!r}")
    circuit.h(anc)
    circuit.measure(anc)
    return circuit


# One circuit per generator, in G1..G4 order -- the order the syndrome bits are
# assembled in. Index i of this list produces syndrome bit i.
STABILIZER_CIRCUITS = [_build_stabilizer_circuit(g) for g in STABILIZERS]


# ======================================================================
# ancilla reset
#
# The single ancilla is reused across the four measurements. After each
# measurement it holds |0> or |1> (a definite basis state, since it was just
# measured). Reset it to |0> before the next generator: apply X iff it measured
# 1. This is a 1-qubit circuit run on the ancilla's own memory.
# ======================================================================
def _build_reset_circuit() -> Circuit:
    """X on a lone qubit -- used to flip a measured-|1> ancilla back to |0>."""
    circuit = Circuit(1)
    circuit.x(0)
    return circuit


ANCILLA_RESET = _build_reset_circuit()


# ======================================================================
# correction lookup + application
# ======================================================================
def _one_qubit_pauli(kind: str) -> Circuit:
    circuit = Circuit(1)
    if kind == "X":
        circuit.x(0)
    elif kind == "Y":
        circuit.y(0)
    elif kind == "Z":
        circuit.z(0)
    else:
        raise ValueError(f"correction Pauli {kind!r} doesn't exist")
    return circuit


_CORRECTIONS = {
    "X": _one_qubit_pauli("X"),
    "Y": _one_qubit_pauli("Y"),
    "Z": _one_qubit_pauli("Z"),
}


def correction_for(syndrome: tuple[int, int, int, int]) -> tuple[int, Circuit] | None:
    """Given the 4-bit syndrome, return (code_position, pauli_circuit) to apply,
    or None if the syndrome is clean (no detected error).

    Raises KeyError on an unknown syndrome (should be impossible: all 16 4-bit
    patterns are in the table).
    """
    if syndrome not in CORRECTION_TABLE:
        raise KeyError(f"syndrome {syndrome} not in correction table")
    entry = CORRECTION_TABLE[syndrome]
    if entry is None:
        return None
    position, pauli = entry
    return position, _CORRECTIONS[pauli]