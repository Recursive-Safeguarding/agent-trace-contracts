"""S2: indexed evidence and claim system.

The S2.9 fixtures, S2.10 worked example, and S2.11 demonstrated failure are
normative examples. The later Python scalar, probability validation, and
tolerance checks state reference-interface policy, not normative S2 semantics.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import pytest

import rs_metalang_ref.evidence as evidence_module
from rs_metalang_ref.evidence import (
    CanonicalArtifactIdentity,
    ClaimRejected,
    Dependence,
    Evidence,
    EvidenceTypeError,
    FusionRejected,
    Grade,
    Index,
    IntervalEstimate,
    NonIdentifiable,
    SubjectId,
    Support,
    and_intro,
    conjoin_all,
    detected_claim,
    fuse,
    identifiability_gate,
    total_variation,
)


def _identity(artifact_key: str, locator_digest: str) -> CanonicalArtifactIdentity:
    return CanonicalArtifactIdentity(
        registry_identity="registry-main",
        artifact_key=artifact_key,
        locator_digest=locator_digest,
    )


def _subject(checkpoint: str) -> SubjectId:
    return SubjectId(
        model_family="m",
        checkpoint_identity=_identity(checkpoint, f"digest-{checkpoint}"),
        system_prompt_identity=_identity("prompt-p", "digest-p"),
        tool_manifest_identity=_identity("tools", "digest-tools"),
        scaffolding_identity=_identity("scaffold", "digest-scaffold"),
        deployment_config_identity=_identity("deploy", "digest-deploy"),
    )


def _index(checkpoint: str) -> Index:
    return Index(
        subject=_subject(checkpoint),
        version="v1",
        time="t1",
        intervention="i1",
        population="pop1",
    )


# -- S2.9 acceptance test 1: S2-conjunction-same-subject --------------------


def test_s2_conjunction_same_subject():
    idx = _index("h1")
    e1 = Evidence(proposition="A(m)", modality=Grade.P, index=idx)
    e2 = Evidence(proposition="B(m)", modality=Grade.X, index=idx)

    support = and_intro(e1, e2)

    assert support.proposition == "A(m) and B(m)"
    assert support.requirement == "P[A] AND X[B]"
    assert support.index == idx


def test_s2_subject_identity_does_not_collapse_on_locator_digest_collision():
    shared_identity = CanonicalArtifactIdentity(
        registry_identity="registry-main",
        artifact_key="shared-artifact",
        locator_digest="same-locator-digest",
    )
    checkpoint_a = CanonicalArtifactIdentity(
        registry_identity="registry-a",
        artifact_key="checkpoint-a",
        locator_digest="colliding-locator-digest",
    )
    checkpoint_b = CanonicalArtifactIdentity(
        registry_identity="registry-b",
        artifact_key="checkpoint-b",
        locator_digest="colliding-locator-digest",
    )
    left_subject = SubjectId(
        model_family="m",
        checkpoint_identity=checkpoint_a,
        system_prompt_identity=shared_identity,
        tool_manifest_identity=shared_identity,
        scaffolding_identity=shared_identity,
        deployment_config_identity=shared_identity,
    )
    right_subject = SubjectId(
        model_family="m",
        checkpoint_identity=checkpoint_b,
        system_prompt_identity=shared_identity,
        tool_manifest_identity=shared_identity,
        scaffolding_identity=shared_identity,
        deployment_config_identity=shared_identity,
    )
    left = Evidence(
        proposition="A(m)",
        modality=Grade.P,
        index=replace(_index("h1"), subject=left_subject),
    )
    right = Evidence(
        proposition="B(m)",
        modality=Grade.X,
        index=replace(_index("h1"), subject=right_subject),
    )

    assert checkpoint_a != checkpoint_b
    assert left_subject != right_subject
    with pytest.raises(EvidenceTypeError) as excinfo:
        and_intro(left, right)
    assert excinfo.value.code == "E-SUBJECT-UNIFICATION"
    assert excinfo.value.field == "checkpoint_identity"


def test_s2_locator_digest_does_not_define_canonical_artifact_equality():
    left = CanonicalArtifactIdentity(
        registry_identity="registry-main",
        artifact_key="checkpoint-a",
        locator_digest="locator-before-move",
    )
    right = CanonicalArtifactIdentity(
        registry_identity="registry-main",
        artifact_key="checkpoint-a",
        locator_digest="locator-after-move",
    )

    assert left == right
    assert hash(left) == hash(right)


def test_s2_conjunction_unifies_subject_subclass_by_declared_values():
    class EquivalentSubjectId(SubjectId):
        pass

    left_index = _index("h1")
    right_index = replace(
        left_index,
        subject=EquivalentSubjectId(
            model_family="m",
            checkpoint_identity=_identity("h1", "digest-h1"),
            system_prompt_identity=_identity("prompt-p", "digest-p"),
            tool_manifest_identity=_identity("tools", "digest-tools"),
            scaffolding_identity=_identity("scaffold", "digest-scaffold"),
            deployment_config_identity=_identity("deploy", "digest-deploy"),
        ),
    )
    assert left_index.subject != right_index.subject

    left = Evidence(proposition="A(m)", modality=Grade.P, index=left_index)
    right = Evidence(proposition="B(m)", modality=Grade.X, index=right_index)

    support = conjoin_all([left, right])

    assert isinstance(support, Support)
    assert support.proposition == "A(m) and B(m)"
    assert support.requirement == "P[A] AND X[B]"
    assert support.index == left_index


def test_s2_conjunction_returns_typed_unknown_for_inconsistent_assumptions():
    idx = _index("h1")
    e1 = Evidence(
        proposition="A(m)",
        modality=Grade.P,
        index=idx,
        assumptions=frozenset({"A"}),
    )
    e2 = Evidence(
        proposition="B(m)",
        modality=Grade.X,
        index=idx,
        assumptions=frozenset({"not A"}),
    )

    result = conjoin_all([e1, e2])

    assert type(result).__name__ == "AssumptionCompatibilityUnknown"
    assert isinstance(result, evidence_module.AssumptionCompatibilityUnknown)
    assert result.evidences == (e1, e2)
    assert result.assumptions == frozenset({"A", "not A"})


def test_s2_conjunction_returns_typed_unknown_for_non_empty_assumptions():
    idx = _index("h1")
    left = Evidence(proposition="A(m)", modality=Grade.P, index=idx)
    right = Evidence(proposition="B(m)", modality=Grade.X, index=idx)

    control = conjoin_all([left, right])

    assert isinstance(control, Support)

    assumed_left = Evidence(
        proposition="A(m)",
        modality=Grade.P,
        index=idx,
        assumptions=frozenset({"calibration is stable"}),
    )
    result = conjoin_all([assumed_left, right])

    assert type(result).__name__ == "AssumptionCompatibilityUnknown"
    assert isinstance(result, evidence_module.AssumptionCompatibilityUnknown)
    assert result.evidences == (assumed_left, right)
    assert result.assumptions == frozenset({"calibration is stable"})


def test_s2_index_mismatch_precedes_assumption_compatibility_unknown():
    left = Evidence(
        proposition="A(m)",
        modality=Grade.P,
        index=_index("h1"),
        assumptions=frozenset({"calibration is stable"}),
    )
    right = Evidence(
        proposition="B(m)",
        modality=Grade.X,
        index=_index("h2"),
    )

    with pytest.raises(EvidenceTypeError) as excinfo:
        conjoin_all([left, right])

    assert excinfo.value.code == "E-SUBJECT-UNIFICATION"


@pytest.mark.parametrize(
    "changed_field",
    [
        "model_family",
        "system_prompt_identity",
        "tool_manifest_identity",
        "scaffolding_identity",
        "deployment_config_identity",
    ],
)
def test_s2_subject_mismatch_reports_changed_subject_field(changed_field):
    left = Evidence(proposition="A(m)", modality=Grade.P, index=_index("h1"))
    changed_value = (
        f"changed-{changed_field}"
        if changed_field == "model_family"
        else _identity(f"changed-{changed_field}", f"digest-{changed_field}")
    )
    changed_subject = replace(
        left.index.subject,
        **{changed_field: changed_value},
    )
    right = Evidence(
        proposition="B(m)",
        modality=Grade.X,
        index=replace(left.index, subject=changed_subject),
    )

    with pytest.raises(EvidenceTypeError) as excinfo:
        conjoin_all([left, right])

    assert excinfo.value.code == "E-SUBJECT-UNIFICATION"
    assert excinfo.value.field == changed_field


# -- S2.9 acceptance test 2: S2-four-checkpoint-rejection -------------------


def test_s2_four_checkpoint_rejection():
    checkpoints = ["h0", "h1", "h2", "h3"]
    evidences = [
        Evidence(proposition=name, modality=Grade.P, index=_index(cp))
        for name, cp in zip(
            [
                "ConflictGoal",
                "RelevantKnowledge",
                "SideTaskAttempt",
                "OversightSensitivePolicy",
            ],
            checkpoints,
        )
    ]

    with pytest.raises(EvidenceTypeError) as excinfo:
        conjoin_all(evidences)

    assert excinfo.value.code == "E-SUBJECT-UNIFICATION"


# -- S2.9 acceptance test 3: S2-duplicate-statistical-reports ---------------


def test_s2_duplicate_statistical_reports():
    e1 = IntervalEstimate(lower=0.10, upper=0.20, provenance_root="dataset7")
    e2 = IntervalEstimate(lower=0.10, upper=0.20, provenance_root="dataset7")

    result = fuse([e1, e2], Dependence.DUPLICATE)

    assert result.fused_interval == (0.10, 0.20)
    assert result.effective_information_count == 1


@pytest.mark.parametrize(
    ("lower", "upper", "provenance_root", "episodes"),
    [
        pytest.param(float("nan"), 0.2, "dataset7", frozenset(), id="nan-lower"),
        pytest.param(0.1, float("inf"), "dataset7", frozenset(), id="infinite-upper"),
        pytest.param(True, 0.2, "dataset7", frozenset(), id="boolean-lower"),
        pytest.param(0.1, "0.2", "dataset7", frozenset(), id="string-upper"),
        pytest.param(0.3, 0.2, "dataset7", frozenset(), id="reversed-interval"),
        pytest.param(0.1, 0.2, "", frozenset(), id="empty-provenance-root"),
        pytest.param(0.1, 0.2, "dataset7", set(), id="mutable-episodes"),
    ],
)
def test_s2_interval_estimate_requires_a_finite_ordered_provenanced_interval(
    lower,
    upper,
    provenance_root,
    episodes,
):
    with pytest.raises(ValueError):
        IntervalEstimate(
            lower=lower,
            upper=upper,
            provenance_root=provenance_root,
            episodes=episodes,
        )


def test_s2_duplicate_fusion_rejects_an_empty_estimate_set():
    with pytest.raises(FusionRejected) as raised:
        fuse([], Dependence.DUPLICATE)

    assert raised.value.reason == "EmptyEstimateSet"


@pytest.mark.parametrize(
    "estimates",
    [
        pytest.param(
            [
                IntervalEstimate(0.10, 0.20, "dataset7"),
                IntervalEstimate(0.30, 0.40, "dataset7"),
            ],
            id="forward",
        ),
        pytest.param(
            [
                IntervalEstimate(0.30, 0.40, "dataset7"),
                IntervalEstimate(0.10, 0.20, "dataset7"),
            ],
            id="reverse",
        ),
    ],
)
def test_s2_duplicate_fusion_rejects_conflicting_reports_in_any_order(estimates):
    with pytest.raises(FusionRejected) as raised:
        fuse(estimates, Dependence.DUPLICATE)

    assert raised.value.reason == "InconsistentDuplicateReports"


def test_s2_identical_duplicate_fusion_is_order_independent():
    first = IntervalEstimate(0.10, 0.20, "dataset7", frozenset({"episode-1"}))
    second = IntervalEstimate(0.10, 0.20, "dataset7", frozenset({"episode-1"}))

    assert fuse([first, second], Dependence.DUPLICATE) == fuse(
        [second, first], Dependence.DUPLICATE
    )


# -- S2.9 acceptance test 4: S2-identical-mechanism-distributions -----------


def test_s2_identical_mechanism_distributions():
    """Check a focused result projection, not the complete S2.9 statement.

    This checks `NonIdentifiable` and the two rejected hard-detection claims. It
    does not assert the permitted `ObservationallyIndistinguishableUnderPolicy`
    statement.
    """
    dist_h = {True: 0.5, False: 0.5}
    dist_m = {True: 0.5, False: 0.5}

    result = identifiability_gate(dist_h, dist_m)

    assert isinstance(result, NonIdentifiable)
    assert result.bayes_error == 0.5

    with pytest.raises(ClaimRejected):
        detected_claim("H", result)
    with pytest.raises(ClaimRejected):
        detected_claim("M", result)


@pytest.mark.parametrize(
    "malformed_gate_result",
    [
        pytest.param(None, id="none"),
        pytest.param(object(), id="object"),
        pytest.param("0.25", id="string"),
        pytest.param(True, id="boolean"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinite"),
        pytest.param(-0.01, id="negative"),
        pytest.param(0.5, id="non-identifiable-boundary"),
    ],
)
def test_s2_detected_claim_fails_closed_on_a_malformed_gate_result(
    malformed_gate_result,
):
    with pytest.raises(ClaimRejected, match="invalid identifiability result"):
        detected_claim("H", malformed_gate_result)


@pytest.mark.parametrize(
    "invalid_hypothesis",
    [
        pytest.param("", id="empty"),
        pytest.param(None, id="none"),
        pytest.param(object(), id="object"),
    ],
)
def test_s2_detected_claim_fails_closed_on_an_invalid_hypothesis(
    invalid_hypothesis,
):
    with pytest.raises(ClaimRejected, match="hypothesis_name"):
        detected_claim(invalid_hypothesis, 0.25)


@pytest.mark.parametrize("hypothesis", ["H", "M"])
def test_s2_detected_claim_accepts_an_exact_finite_gate_result(hypothesis):
    assert detected_claim(hypothesis, 0.25) == f"Detected({hypothesis})"


# -- Non-normative reference-interface policy: Python scalars and tolerance -


def test_s2_equal_measures_with_explicit_zero_mass_are_non_identifiable():
    result = identifiability_gate(
        {"a": 1.0},
        {"a": 1.0, "b": 0.0},
        policy="fixed-policy",
        observation_space="outcomes",
    )

    assert result == NonIdentifiable(
        bayes_error=0.5,
        policy="fixed-policy",
        observation_space="outcomes",
    )

    with pytest.raises(ClaimRejected):
        detected_claim("H", result)
    with pytest.raises(ClaimRejected):
        detected_claim("M", result)


@pytest.mark.parametrize("invalid_argument", ["dist_h", "dist_m"])
@pytest.mark.parametrize(
    "invalid_distribution",
    [
        pytest.param({}, id="empty"),
        pytest.param({True: -0.25, False: 1.25}, id="negative-mass"),
        pytest.param({True: 0.25, False: 0.25}, id="mass-does-not-sum-to-one"),
        pytest.param({True: float("inf"), False: 0.0}, id="infinite-mass"),
        pytest.param({True: float("nan"), False: 1.0}, id="nan-mass"),
        pytest.param({True: "1.0"}, id="non-numeric-mass"),
        pytest.param({True: True}, id="boolean-mass"),
        pytest.param([("outcome", 1.0)], id="non-mapping"),
        pytest.param({"outcome": 1.0 + 2e-12}, id="outside-total-tolerance"),
        pytest.param({"outcome": 1.25}, id="mass-above-one"),
        pytest.param({"x": 1e308, "y": 1e308}, id="finite-masses-overflow-total"),
    ],
)
def test_s2_identifiability_rejects_invalid_probability_mass_function(
    invalid_argument, invalid_distribution
):
    valid_distribution = {True: 0.5, False: 0.5}
    dist_h = (
        invalid_distribution if invalid_argument == "dist_h" else valid_distribution
    )
    dist_m = (
        invalid_distribution if invalid_argument == "dist_m" else valid_distribution
    )

    with pytest.raises(ValueError):
        identifiability_gate(dist_h, dist_m)


def test_s2_distinct_valid_mechanism_distributions_have_bayes_error():
    dist_h = {True: 0.75, False: 0.25}
    dist_m = {True: 0.25, False: 0.75}

    result = identifiability_gate(dist_h, dist_m)

    assert result == 0.25


def test_s2_disjoint_near_normalized_distributions_have_zero_bayes_error():
    result = identifiability_gate(
        {"h": 1.0 + 5e-13},
        {"m": 1.0 + 5e-13},
    )

    assert result == 0.0
    assert 0.0 <= result <= 0.5


def test_s2_total_variation_accepts_valid_probability_mass_functions():
    assert total_variation({"same": 1.0}, {"same": 1.0}) == 0.0
    assert total_variation({"left": 1.0}, {"right": 1.0}) == 1.0
    assert total_variation({"a": 0.75, "b": 0.25}, {"a": 0.25, "b": 0.75}) == 0.5
    assert (
        total_variation(
            {"a": Fraction(3, 4), "b": Fraction(1, 4)},
            {"a": Fraction(1, 4), "b": Fraction(3, 4)},
        )
        == 0.5
    )


@pytest.mark.parametrize("invalid_argument", ["p", "q"])
@pytest.mark.parametrize(
    ("invalid_distribution", "message_template"),
    [
        pytest.param(
            {"outcome": True},
            "{argument_name} contains a non-numeric probability mass",
            id="boolean-mass",
        ),
        pytest.param(
            {"outcome": "1.0"},
            "{argument_name} contains a non-numeric probability mass",
            id="string-mass",
        ),
        pytest.param(
            {"outcome": Decimal("1.0")},
            (
                "{argument_name}['outcome'] has unsupported probability scalar "
                "type Decimal; expected a numbers.Real value other than bool"
            ),
            id="decimal-mass",
        ),
        pytest.param(
            [("outcome", 1.0)],
            "{argument_name} must be a non-empty probability mass function",
            id="non-mapping",
        ),
    ],
)
def test_s2_total_variation_rejects_invalid_probability_mass_function(
    invalid_argument,
    invalid_distribution,
    message_template,
):
    valid_distribution = {"outcome": 1.0}
    p = invalid_distribution if invalid_argument == "p" else valid_distribution
    q = invalid_distribution if invalid_argument == "q" else valid_distribution

    with pytest.raises(ValueError) as raised:
        total_variation(p, q)

    assert str(raised.value) == message_template.format(argument_name=invalid_argument)


# -- S2.10: fully worked four-checkpoint example ----------------------------


def test_s2_worked_four_checkpoint_example():
    idx = lambda cp: Index(
        subject=SubjectId(
            model_family="m",
            checkpoint_identity=_identity(cp, f"digest-{cp}"),
            system_prompt_identity=_identity("prompt-p7", "digest-p7"),
            tool_manifest_identity=_identity("tools4", "digest-tools4"),
            scaffolding_identity=_identity("agentloop9", "digest-agentloop9"),
            deployment_config_identity=_identity("deploy3", "digest-deploy3"),
        ),
        version="v1",
        time="t1",
        intervention="i1",
        population="pop1",
    )

    e0 = Evidence("ConflictGoal(m0)", Grade.P, idx("ckpt_A"))
    e1 = Evidence("RelevantKnowledge(m1)", Grade.P, idx("ckpt_B"))
    e2 = Evidence("SideTaskAttempt(m2)", Grade.P, idx("ckpt_C"))
    e3 = Evidence("OversightSensitivePolicy(m3)", Grade.X, idx("ckpt_D"))

    with pytest.raises(EvidenceTypeError) as excinfo:
        conjoin_all([e0, e1, e2, e3])

    assert excinfo.value.code == "E-SUBJECT-UNIFICATION"
    assert excinfo.value.field == "checkpoint_identity"


# -- S2.11: DEMONSTRATED-FAILURE (missing dependence declaration) -----------


def test_s2_demonstrated_failure_missing_dependence_declaration():
    e1 = IntervalEstimate(
        lower=0.21,
        upper=0.21,
        provenance_root="e1",
        episodes=frozenset({1, 2, 3, 4, 5}),
    )
    e2 = IntervalEstimate(
        lower=0.18,
        upper=0.18,
        provenance_root="e2",
        episodes=frozenset({4, 5, 6, 7, 8}),
    )

    with pytest.raises(FusionRejected) as excinfo:
        fuse([e1, e2], dependence=None)

    assert excinfo.value.reason == "MissingDependenceDeclaration"
    assert excinfo.value.details["detected_overlap"] == [4, 5]
