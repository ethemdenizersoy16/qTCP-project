"""QPing: sequential quantum-channel fidelity estimation for the qTCP handshake.

Estimates the Alice->Bob channel fidelity by measuring shared entangled pairs
in random matching Pauli bases and checking their correlations, then deciding
whether the fidelity clears a threshold F0 via sequential Bayesian hypothesis
testing.

Design (agreed with the transfer/handshake layers):

  - One-directional. This tests the fidelity of the pairs Alice's reservation
    produces; if Bob wants to send, he opens his own window and runs his own
    QPing. Fidelity is not assumed symmetric.

  - Runs on the live reservation, before the data phase. The window is already
    producing entangled pairs when QPing starts; QPing consumes the first
    several as they arrive (they are measured, not teleported, so they are
    destroyed -- fine, they are diagnostic traffic and the window keeps
    regenerating for the data phase that follows).

  - Alice drives, Bob is reactive. For each test pair Alice picks a random
    Pauli basis, measures her half immediately, and announces the basis (and
    which pair) to Bob. Bob measures his half in that basis and reports his
    outcome. Alice pairs the two outcomes, scores pass/fail, updates the
    posterior, and checks the decision rule.

  - Measurement primitive: random matching Pauli basis (X, Y, or Z). For a
    perfect |Phi+> the correlations are deterministic --
        ZZ : outcomes equal
        XX : outcomes equal
        YY : outcomes ANTI-correlated  (<YY> = -1 for |Phi+>)
    so "pass" means equal in X/Z, differing in Y. A depolarised/Werner
    component passes with probability 1/2 (uncorrelated), which is what makes
    the pass rate track fidelity.

  - Likelihood (Werner model): a pair of true fidelity F passes a random-basis
    correlation check with probability
        p_pass(F) = (1 + 2F) / 3
    (F=1 -> 1 always passes; F=1/4 -> 1/2 coin flip, i.e. no entanglement).
    p_pass is monotonic in F, so "F >= F0" is the same event as
    "theta >= p_pass(F0)" where theta is the Bernoulli pass-probability. That
    lets the whole test run on a Beta-Bernoulli posterior over theta with a
    closed-form decision via the Beta CDF -- no integral over F.

  - Sequential decision: Beta(alpha0, beta0) prior on theta; after k passes in
    n trials the posterior is Beta(alpha0 + k, beta0 + n - k). Accept as soon
    as P(theta >= theta0 | data) >= eta; reject if it can no longer get there
    (or at the max-pairs cap). No inconclusive band for now -- binary verdict.

This module is intentionally split: the CIRCUITS and the pure MATH
(p_pass, posterior, decision) live here as stateless helpers; the
orchestration (who announces, who measures, accumulating counts, driving the
per-pair loop, flipping is_testing, gating the handshake state) lives in
QTCPHandshake, which calls into here.
"""

from enum import Enum, auto

from scipy.stats import beta as _beta

from sequence.components.circuit import Circuit


# ----------------------------------------------------------------------
# Pauli bases
# ----------------------------------------------------------------------

class PauliBasis(Enum):
    X = auto()
    Y = auto()
    Z = auto()


# ----------------------------------------------------------------------
# Measurement circuits
# ----------------------------------------------------------------------
#
# Each measures a single qubit in a chosen Pauli basis by rotating that
# basis onto the computational (Z) basis, then measuring:
#
#   Z basis : measure directly. Eigenstates |0>,|1> already computational.
#   X basis : H maps |+>,|-> -> |0>,|1>, then measure.
#   Y basis : S^dagger then H maps |+i>,|-i> -> |0>,|1>, then measure.
#             (S^dagger = phase(-pi/2); H*S^dagger is the standard Y readout.)
#
# The measurement outcome (0/1) is the eigenvalue index in that basis:
# 0 <-> +1 eigenstate, 1 <-> -1 eigenstate. Correlation checks downstream
# compare Alice's and Bob's outcome bits per the |Phi+> correlation table.

_MEASURE_Z = Circuit(1)
_MEASURE_Z.measure(0)

_MEASURE_X = Circuit(1)
_MEASURE_X.h(0)
_MEASURE_X.measure(0)

_MEASURE_Y = Circuit(1)
_MEASURE_Y.sdg(0)
_MEASURE_Y.h(0)
_MEASURE_Y.measure(0)

_BASIS_CIRCUIT = {
    PauliBasis.X: _MEASURE_X,
    PauliBasis.Y: _MEASURE_Y,
    PauliBasis.Z: _MEASURE_Z,
}


def measurement_circuit(basis: "PauliBasis") -> Circuit:
    """The single-qubit circuit that reads out `basis`, mapping that basis's
    +1/-1 eigenstates to outcomes 0/1."""
    return _BASIS_CIRCUIT[basis]


# ----------------------------------------------------------------------
# Correlation check
# ----------------------------------------------------------------------

def is_pass(basis: "PauliBasis", alice_outcome: int, bob_outcome: int) -> bool:
    """Whether one measured pair passes its correlation check for |Phi+>.

    X, Z: outcomes should match (correlated).
    Y   : outcomes should differ (anti-correlated, <YY> = -1 for |Phi+>).

    A perfect pair always passes; an uncorrelated (fully mixed) pair passes
    with probability 1/2.
    """
    if basis is PauliBasis.Y:
        return alice_outcome != bob_outcome
    return alice_outcome == bob_outcome


# ----------------------------------------------------------------------
# Likelihood + decision math (stateless; Werner model)
# ----------------------------------------------------------------------

def p_pass(fidelity: float) -> float:
    """Werner-model pass probability for a random matching-basis correlation
    check: p_pass(F) = (1 + 2F) / 3. Monotonic increasing on [1/4, 1] ->
    [1/2, 1]."""
    return (1.0 + 2.0 * fidelity) / 3.0


def posterior_accept_prob(passes: int, trials: int, f0: float,
                          alpha0: float = 1.0, beta0: float = 1.0) -> float:
    """P(theta >= theta0 | data), where theta is the Bernoulli pass-probability
    and theta0 = p_pass(F0).

    With a Beta(alpha0, beta0) prior on theta and `passes` successes in
    `trials` Bernoulli trials, the posterior is
    Beta(alpha0 + passes, beta0 + trials - passes). The tail mass above theta0
    is 1 - CDF(theta0).

    Because p_pass is monotonic in F, {theta >= theta0} and {F >= F0} are the
    same event -- so this is an exact posterior probability for the fidelity
    hypothesis, not an approximation.
    """
    theta0 = p_pass(f0)
    a = alpha0 + passes
    b = beta0 + trials - passes
    return 1.0 - float(_beta.cdf(theta0, a, b))


class QPingVerdict(Enum):
    ACCEPT = auto()     # confident F >= F0
    REJECT = auto()     # confident F <  F0, or inconclusive at the cap
    CONTINUE = auto()   # not yet conclusive; sample another pair


def decide(passes: int, trials: int, f0: float, eta: float,
           max_pairs: int, alpha0: float = 1.0, beta0: float = 1.0
           ) -> "QPingVerdict":
    """Sequential decision after observing `passes` of `trials` test pairs.

    Let p = P(F >= F0 | data) (via posterior_accept_prob). Symmetric rule:

        p >= eta        -> ACCEPT   (eta-confident the channel clears F0)
        p <= 1 - eta    -> REJECT   (eta-confident it does not)
        otherwise       -> CONTINUE (sample another pair)

    At the cap (trials >= max_pairs) an unresolved test resolves to REJECT:
    inconclusive is treated as not-good-enough for admission. The caller loops
    on CONTINUE, announcing one more pair each time, and stops on ACCEPT or
    REJECT.

    Note trials should be >= 1 before calling; with trials == 0 the posterior
    is the prior and no evidence has accrued.
    """
    p = posterior_accept_prob(passes, trials, f0, alpha0, beta0)

    if p >= eta:
        return QPingVerdict.ACCEPT
    if p <= 1.0 - eta:
        return QPingVerdict.REJECT
    if trials >= max_pairs:
        # Cap reached without crossing either threshold: inconclusive, which
        # for admission control is not good enough -> reject.
        return QPingVerdict.REJECT
    return QPingVerdict.CONTINUE