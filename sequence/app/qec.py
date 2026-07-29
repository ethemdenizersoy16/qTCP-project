"""[[3,1,1]] bit-flip detection code for the qTCP transfer layer.

Under the current simulator config, memory fidelity < 1 makes SeQUeNCe
sample: with probability (1 - fidelity), a memory operation produces a
different pure state than intended. Teleported qubits inherit that noise
via imperfect Bell pairs, so Bob's received share can be a corrupted
version of what Alice sent. Without a check, Bob accepts corrupted shares
silently and QSS reconstructs from garbage.

This code encodes a single data qubit into three code qubits by
entangling it with two |0> ancillas via CNOT:

    |psi> -> alpha |000> + beta |111>

Bob applies the inverse (same CNOTs -- they're self-inverse) and measures
the two ancilla positions. The syndrome (m1, m2) reveals bit-flip errors:

    (0, 0): no error detected -- ACK
    (1, 0): ancilla 1 flipped -- error detected, NACK
    (0, 1): ancilla 2 flipped -- error detected, NACK
    (1, 1): data qubit flipped, OR both ancillas flipped -- NACK

Detection only, not correction: the point isn't to salvage corrupted
shares (QSS's threshold structure already tolerates lost shares) but to
detect corruption so it becomes an ordinary FAILED transfer instead of
silent contamination of the QSS decode.

Limitations. This is a [[3,1]] repetition-based detection code, not a
proper stabiliser code with distance guarantees. Under a per-transmission
error rate of p:

  - Single X errors on any of the three code qubits produce non-zero
    syndromes; caught reliably.
  - Simultaneous X errors on both ancillas produce syndrome (1, 1) which
    is indistinguishable from a single X error on the data qubit; caught,
    but the "correction" a corrector would apply is wrong. Since we only
    detect and NACK, this is fine.
  - Simultaneous X errors on the data qubit and one ancilla produce a
    single-flip syndrome that reflects the ancilla but leaves the data
    qubit corrupted -- an undetected error. Probability O(p^2), which
    under low per-transmission noise is small compared to O(p) single
    errors.
  - Z errors are completely invisible to this code (repetition codes only
    protect against X). Under the current SeQUeNCe sampling noise model,
    which produces alternative pure states rather than phase noise, this
    is closer to protection for the actual error mode than it looks -- but
    if the noise model changes, this code will silently fail.

Code positions:

    Index 0 : the data qubit (the one that carries the payload)
    Index 1 : ancilla 1
    Index 2 : ancilla 2

The DATA_QUBIT_INDEX constant makes this explicit for callers that need
to know which slot to keep after a clean-syndrome ACK on Bob's side.
"""

from sequence.components.circuit import Circuit


CODE_SIZE = 3
DATA_QUBIT_INDEX = 0


# Encoder: data qubit at position 0, ancillas at 1 and 2.
# CNOT(0,1) and CNOT(0,2) entangle the ancillas with the data qubit.
ENCODER = Circuit(CODE_SIZE)
ENCODER.cx(0, 1)
ENCODER.cx(0, 2)


# Decoder: applies the same CNOTs (self-inverse) then measures the two
# ancilla positions. Position 0 (data qubit) is not measured -- it holds
# the recovered payload after decoding.
DECODER = Circuit(CODE_SIZE)
DECODER.cx(0, 1)
DECODER.cx(0, 2)
DECODER.measure(1)
DECODER.measure(2)


def syndrome_clean(syndrome: tuple) -> bool:
    """Given the two ancilla measurement outcomes as (m1, m2), return True
    if no error was detected (both zero).

    Any non-(0, 0) syndrome triggers NACK on Bob's side. Bob does not
    attempt correction: the point of this code is detection, not repair,
    because QSS at the packet level already tolerates lost shares.
    """
    return syndrome == (0, 0)