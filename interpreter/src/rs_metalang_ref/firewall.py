"""Proof-firewall checks and probability-bound arithmetic (S6).

This module implements a conservative subset of the result-to-premise table and the
stated probability calculation; its scope is the result-type separation checks and
the finite probability arithmetic.
"""

from __future__ import annotations

import math
from numbers import Real

# -- S6.1: firewall rules for comparison results ----------------------------

DISCHARGEABLE_BY_NO_WITNESS = frozenset(
    {
        "SearchExecuted",
        "NoCounterexampleFoundInEnumeratedSet",
        "EmpiricalSearchStatistic",
    }
)

NEVER_DISCHARGEABLE_BY_NO_WITNESS = frozenset(
    {
        "GlobalResidualEquivalence",
        "ProfileRelativeOperationalAdequacy",
        "RuleLevelKernelSoundness",
        "ForwardSimulation",
        "EventLabelSoundness",
        "EffectOverapproximation",
        "CompleteMediation",
        "InvariantClosure",
        "InvariantSafety",
        "AbsenceOfDeploymentModelMiss",
        "UniversalShieldSoundness",
    }
)


# The specification's proof-firewall-and-shield rules (S6.1) and its result
# algebra define the closed comparison-result constructor vocabulary. The
# result algebra fixes what these results may fill: "A result in this
# algebra can never fill a premise of the shield theorem."
COMPARISON_RESULT_KINDS = frozenset(
    {
        "Distinguished",
        "ProvedEquivalent",
        "NoWitnessWithinBound",
        "SearchIncomplete",
        "Untested",
    }
)

# The proof premises a comparison result can never populate: the universal
# premises of S6.1 plus the simulation and labelling certificate that pairs
# ForwardSimulation with EventLabelSoundness ("a residual-equivalence
# certificate is not a simulation certificate ... no coercion exists").
PROOF_PREMISE_CERTIFICATES = NEVER_DISCHARGEABLE_BY_NO_WITNESS | frozenset(
    {"SimulationAndLabellingCertificate"}
)

# The premise vocabulary this reference subset recognises: the eleven universal
# premises (S6.1) and the simulation and
# labelling certificate that a
# comparison result may never populate, together with the only propositions a
# comparison result may discharge: the three NoWitnessWithinBound forms
# (S6.1) and the Untested meta-proposition
# TestStatus (S6.1; see also the result algebra's account of what a result
# kind may discharge). It omits the meta-reports that SearchIncomplete may make: that a search
# ran, what part of the scope it covered, and why it stopped. An unrecognised
# premise name is refused rather than admitted by default.
KNOWN_PREMISE_KINDS = (
    PROOF_PREMISE_CERTIFICATES | DISCHARGEABLE_BY_NO_WITNESS | frozenset({"TestStatus"})
)

# The pairs implemented by this reference subset: the three
# NoWitnessWithinBound search forms, and Untested against the TestStatus
# meta-proposition. SearchIncomplete's permitted meta-reporting is not
# implemented. Every other pair is refused, including pairs whose two names
# are each individually well-formed. ProvedEquivalent against
# GlobalResidualEquivalence is refused here too, and it is the pair most
# likely to look like an omission: a proved equivalence does bear on global
# residual equivalence, but only conditionally on the certificate that
# established it, and a pair of bare kind names carries no certificate. This
# entry point sees two strings, so it cannot admit a conditional truth.
DISCHARGEABLE_PAIRS = frozenset(
    {("NoWitnessWithinBound", proposition) for proposition in DISCHARGEABLE_BY_NO_WITNESS}
    | {("Untested", "TestStatus")}
)


class FirewallViolation(Exception):
    def __init__(self, code: str = "E-PROOF-FIREWALL", **details):
        self.code = code
        self.details = details
        super().__init__(f"{code}: {details}")


def discharge_no_witness(proposition: str) -> str:
    """S6.1: NoWitnessWithinBound may discharge only the three named
    proposition forms; it must not discharge any of the eleven universal
    premises."""
    if proposition in NEVER_DISCHARGEABLE_BY_NO_WITNESS:
        raise FirewallViolation(field=proposition, required="not dischargeable by NoWitnessWithinBound")
    if proposition in DISCHARGEABLE_BY_NO_WITNESS:
        return proposition
    raise FirewallViolation(field=proposition, required="unrecognized proposition form for NoWitnessWithinBound")


def discharge_search_incomplete(proposition: str) -> str:
    """Refuse SearchIncomplete propositions in this reference subset.

    The specification permits reports that a search ran, what part of the scope it
    covered, and why it stopped. This function does not implement those meta-reports.
    SearchIncomplete still cannot fill a simulation, semantic, statistical, causal,
    or safety premise.
    """
    raise FirewallViolation(
        field=proposition,
        required="SearchIncomplete cannot discharge any proposition",
    )


def discharge_untested(proposition: str) -> str:
    """S6.1: Untested(reason) may discharge only TestStatus = Untested(reason);
    it must not discharge any positive semantic/statistical/causal/safety
    premise."""
    if proposition == "TestStatus":
        return proposition
    raise FirewallViolation(field=proposition, required="Untested may only discharge TestStatus=Untested(reason)")


def typecheck(available_result_kind: str, required_certificate: str) -> bool:
    """S6.1: conservative comparison-result proof-firewall checks.

    DISCHARGEABLE_PAIRS holds exactly four implemented pairs:
    NoWitnessWithinBound against each of its three named search propositions,
    and Untested against TestStatus. Every other pair is refused. 'A
    residual-equivalence certificate is not a simulation certificate. The
    types are disjoint ... No coercion exists between them' (S6.1), and 'a
    result in this algebra can never fill a premise of the shield theorem'
    (result algebra). SearchIncomplete's permitted meta-reporting is
    not represented here, so it reaches no pair in this API. It still cannot fill
    any proof premise.

    The comparison-result vocabulary is closed. The premise vocabulary is limited
    to the implemented pairs and the proof premises whose rejection this module
    checks. An unrecognised name is refused before the table is consulted rather
    than admitted by default: the result algebra fixes
    the five comparison-result constructors and states that 'no sixth
    comparison status may be introduced', and S6.1 fixes the
    premises. Admitting an unlisted name instead would let an invented result
    kind populate a proof premise simply by being unnamed here."""
    if available_result_kind not in COMPARISON_RESULT_KINDS:
        return False
    if required_certificate not in KNOWN_PREMISE_KINDS:
        return False
    return (available_result_kind, required_certificate) in DISCHARGEABLE_PAIRS


def require_typecheck(available_result_kind: str, required_certificate: str) -> None:
    if not typecheck(available_result_kind, required_certificate):
        raise FirewallViolation(
            field=required_certificate, required=f"cannot be populated by {available_result_kind}"
        )


# -- S6.5: probabilistic theorem ----------------------------------------------


def probabilistic_bad_state_bound(delta: float, epsilons) -> float:
    """PROB-SHIELD (S6.5): Pr(encoded hard bad state by H) <= delta +
    sum(epsilon_t), clipped at one."""
    if (
        isinstance(delta, bool)
        or not isinstance(delta, Real)
        or not math.isfinite(delta)
        or not 0 <= delta <= 1
    ):
        raise ValueError("delta must be finite and between zero and one")

    try:
        epsilons = tuple(epsilons)
    except TypeError as error:
        raise ValueError("epsilons must be an iterable of probability terms") from error

    for epsilon in epsilons:
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, Real)
            or not math.isfinite(epsilon)
            or epsilon < 0
        ):
            raise ValueError("each epsilon must be finite and non-negative")

    return min(1.0, delta + sum(epsilons))
