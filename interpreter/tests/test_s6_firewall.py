"""S6 proof-firewall and probability-bound helper cases."""

from __future__ import annotations

import pytest

from rs_metalang_ref import firewall
from rs_metalang_ref.firewall import (
    FirewallViolation,
    probabilistic_bad_state_bound,
    require_typecheck,
    typecheck,
)

# -- S6.7, "acceptance tests":
#    S6-no-witness-cannot-fill-simulation-premise -----------------------------


def test_s6_no_witness_cannot_fill_simulation_premise():
    with pytest.raises(FirewallViolation) as excinfo:
        require_typecheck("NoWitnessWithinBound", "SimulationAndLabellingCertificate")
    assert excinfo.value.code == "E-PROOF-FIREWALL"


@pytest.mark.parametrize(
    ("available_result_kind", "required_certificate"),
    [
        ("Distinguished", "SimulationAndLabellingCertificate"),
        ("Distinguished", "InvariantClosure"),
        ("NoWitnessWithinBound", "InvariantClosure"),
        ("ProvedEquivalent", "InvariantClosure"),
        ("Untested", "InvariantClosure"),
    ],
)
def test_s6_comparison_results_cannot_fill_proof_premises(
    available_result_kind, required_certificate
):
    with pytest.raises(FirewallViolation) as excinfo:
        require_typecheck(available_result_kind, required_certificate)
    assert excinfo.value.code == "E-PROOF-FIREWALL"


def test_s6_unknown_available_result_kind_fails_closed():
    with pytest.raises(FirewallViolation) as excinfo:
        require_typecheck("InventedResult", "InvariantClosure")
    assert excinfo.value.code == "E-PROOF-FIREWALL"


def test_s6_unknown_required_premise_kind_fails_closed():
    with pytest.raises(FirewallViolation) as excinfo:
        require_typecheck("NoWitnessWithinBound", "InventedShieldPremise")
    assert excinfo.value.code == "E-PROOF-FIREWALL"


def test_s6_typecheck_matches_implemented_conservative_matrix():
    result_kinds = (
        "Distinguished",
        "ProvedEquivalent",
        "NoWitnessWithinBound",
        "SearchIncomplete",
        "Untested",
    )
    premise_kinds = (
        "SearchExecuted",
        "NoCounterexampleFoundInEnumeratedSet",
        "EmpiricalSearchStatistic",
        "TestStatus",
        "GlobalResidualEquivalence",
        "RuleLevelKernelSoundness",
        "ProfileRelativeOperationalAdequacy",
        "ForwardSimulation",
        "EventLabelSoundness",
        "EffectOverapproximation",
        "CompleteMediation",
        "InvariantClosure",
        "InvariantSafety",
        "AbsenceOfDeploymentModelMiss",
        "UniversalShieldSoundness",
        "SimulationAndLabellingCertificate",
    )
    allowed_pairs = frozenset(
        {
            ("NoWitnessWithinBound", "SearchExecuted"),
            ("NoWitnessWithinBound", "NoCounterexampleFoundInEnumeratedSet"),
            ("NoWitnessWithinBound", "EmpiricalSearchStatistic"),
            ("Untested", "TestStatus"),
        }
    )

    mismatches = []
    for result_kind in result_kinds:
        for premise_kind in premise_kinds:
            expected = (result_kind, premise_kind) in allowed_pairs
            actual = typecheck(result_kind, premise_kind)
            if actual is not expected:
                mismatches.append((result_kind, premise_kind, expected, actual))

    assert not mismatches, f"{len(mismatches)} typecheck matrix mismatches: {mismatches!r}"


def test_s6_module_has_no_executable_shield_surface():
    removed_names = {
        "MissingTerminalBranchProof",
        "OutOfCertifiedFragment",
        "ShieldCertificate",
        "ShieldOutcome",
        "certified_actions",
        "select_shield_action",
        "shield_theorem_holds",
    }

    present_names = removed_names.intersection(vars(firewall))
    assert present_names == set()


@pytest.mark.parametrize(
    ("delta", "epsilons"),
    [
        (-0.1, []),
        (1.1, []),
        (float("nan"), []),
        (0.1, [-0.2]),
        (0.1, [float("inf")]),
        (0.1, [float("nan")]),
        (True, []),
        (0.1, [False]),
        ("0.1", []),
        (0.1, ["0.1"]),
        (0.1 + 0.0j, []),
        (0.1, [0.1 + 0.0j]),
        (0.1, None),
    ],
    ids=(
        "negative-delta",
        "delta-above-one",
        "non-finite-delta",
        "negative-epsilon",
        "infinite-epsilon",
        "nan-epsilon",
        "boolean-delta",
        "boolean-epsilon",
        "string-delta",
        "string-epsilon",
        "complex-delta",
        "complex-epsilon",
        "non-iterable-epsilons",
    ),
)
def test_s6_probabilistic_bound_rejects_invalid_probability_terms(delta, epsilons):
    with pytest.raises(ValueError):
        probabilistic_bad_state_bound(delta, epsilons)


def test_s6_probabilistic_bound_clips_valid_sum_above_one():
    assert probabilistic_bad_state_bound(0.9, [0.2]) == 1.0


def test_s6_probabilistic_bound_consumes_one_shot_generator_once():
    epsilons = (epsilon for epsilon in [0.2, 0.3])

    assert probabilistic_bad_state_bound(0.1, epsilons) == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("delta", "epsilons", "expected"),
    [
        (0.0, [0.0], 0.0),
        (1.0, [], 1.0),
        (0.0, [1.0], 1.0),
    ],
    ids=("zero", "delta-one", "epsilon-one"),
)
def test_s6_probabilistic_bound_accepts_closed_interval_endpoints(
    delta, epsilons, expected
):
    assert probabilistic_bad_state_bound(delta, epsilons) == expected
