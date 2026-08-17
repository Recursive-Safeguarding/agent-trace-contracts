"""Indexed evidence and claim system (S2).

Covers: the Evidence[P; subject, version, time, intervention, population,
provenance, assumptions] type; SubjectId ("same model") equality; index
unification for conjunction-introduction with the typed E-SUBJECT-UNIFICATION
rejection (S2.8's four-checkpoint counterexample); dependence-mediated
fusion (S2.5, only the Duplicate case is numerically spelled out); the
identifiability gate (S2.7); and a minimal Requirement grammar (S2.6).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from enum import Enum
from numbers import Number, Real


class Grade(Enum):
    """The four incomparable evidence modalities (S2.1). No pair is ordered:
    none of the four ranks above or below any other."""

    P = "P"  # proof / replay / checker-validated deductive evidence
    X = "X"  # randomized or otherwise identified interventional evidence
    S = "S"  # calibrated statistical / observational / predictive evidence
    J = "J"  # human or model judgment evidence


@dataclasses.dataclass(frozen=True)
class CanonicalArtifactIdentity:
    """A resolving artifact identity from one canonical registry.

    The registry identity and artifact key identify the artifact. A valid
    registry rejects a different artifact under the same key. The locator
    digest is checked against the resolved artifact for storage and retrieval;
    it does not establish identity.
    """

    registry_identity: str
    artifact_key: str
    locator_digest: str = dataclasses.field(compare=False, hash=False)


@dataclasses.dataclass(frozen=True)
class SubjectId:
    """S2.2 subject identity for one exact model configuration."""

    model_family: str
    checkpoint_identity: CanonicalArtifactIdentity
    system_prompt_identity: CanonicalArtifactIdentity
    tool_manifest_identity: CanonicalArtifactIdentity
    scaffolding_identity: CanonicalArtifactIdentity
    deployment_config_identity: CanonicalArtifactIdentity


@dataclasses.dataclass(frozen=True)
class Index:
    """The semantic-scope indices that must unify for ordinary conjunction
    (S2.3): subject, version, time, intervention, population."""

    subject: SubjectId
    version: str
    time: str
    intervention: str
    population: str


@dataclasses.dataclass(frozen=True)
class Evidence:
    proposition: str
    modality: Grade
    index: Index
    provenance: frozenset = dataclasses.field(default_factory=frozenset)
    assumptions: frozenset = dataclasses.field(default_factory=frozenset)


@dataclasses.dataclass(frozen=True)
class Support:
    """The conclusion of andIntro/conjoin_all (S2.3): an evidence BUNDLE
    retaining every source modality, never a synthetic 'minimum' grade."""

    proposition: str
    index: Index
    provenance: frozenset
    assumptions: frozenset
    requirement: str


@dataclasses.dataclass(frozen=True)
class AssumptionCompatibilityUnknown:
    """Evidence whose non-empty assumptions require an external solver."""

    evidences: tuple[Evidence, ...]
    assumptions: frozenset


class EvidenceTypeError(Exception):
    """S2.8's typed rejection (E-SUBJECT-UNIFICATION and kin): raised instead
    of silently deriving a heterogeneous-subject conjunction."""

    def __init__(self, code: str, field: str, required, observed):
        self.code = code
        self.field = field
        self.required = required
        self.observed = observed
        super().__init__(f"{code}: field={field} required={required} observed={observed}")


def _schema(proposition: str) -> str:
    """The proposition-pattern name, e.g. 'A' for 'A(m)' -- a Requirement
    (S2.6) is stated over PATTERNS, not fully-applied instances."""
    return proposition.split("(", 1)[0]


def unify_index(i1: Index, i2: Index) -> Index:
    """S2.3 Unify: succeeds only under exact equality of every scope field.
    (Provenance MERGES and assumptions UNION separately, on the SUPPORT
    object built by the caller -- Unify itself only unifies the scope.)"""
    subject_differences = tuple(
        subject_field.name
        for subject_field in dataclasses.fields(SubjectId)
        if getattr(i1.subject, subject_field.name) != getattr(i2.subject, subject_field.name)
    )
    scope1 = (i1.version, i1.time, i1.intervention, i1.population)
    scope2 = (i2.version, i2.time, i2.intervention, i2.population)
    if subject_differences or scope1 != scope2:
        field = subject_differences[0] if subject_differences else "index"
        raise EvidenceTypeError(
            "E-SUBJECT-UNIFICATION", field, "one common SubjectId/index", {"left": i1, "right": i2}
        )
    return i1


def conjoin_all(
    evidences: list[Evidence],
) -> Support | AssumptionCompatibilityUnknown:
    """S2.3's conjunction-introduction rule, folded across >=2 evidence
    objects; S2.8's typed rejection when their indices do not unify to one
    common SubjectId (the four-checkpoint counterexample). Non-empty assumption
    sets return a typed unknown because this package has no compatibility solver."""
    if len(evidences) < 2:
        raise ValueError("conjoin_all requires at least two evidence objects")
    base_index = evidences[0].index
    for e in evidences[1:]:
        unify_index(base_index, e.index)
    proposition = " and ".join(e.proposition for e in evidences)
    requirement = " AND ".join(f"{e.modality.value}[{_schema(e.proposition)}]" for e in evidences)
    provenance: frozenset = frozenset()
    assumptions: frozenset = frozenset()
    for e in evidences:
        provenance = provenance | e.provenance
        assumptions = assumptions | e.assumptions
    if assumptions:
        return AssumptionCompatibilityUnknown(tuple(evidences), assumptions)
    return Support(proposition, base_index, provenance, assumptions, requirement)


def and_intro(
    e1: Evidence,
    e2: Evidence,
) -> Support | AssumptionCompatibilityUnknown:
    return conjoin_all([e1, e2])


# -- S2.5: dependence-mediated fusion --------------------------------------


class Dependence(Enum):
    DUPLICATE = "Duplicate"
    INDEPENDENT_BY_DESIGN = "IndependentByDesign"
    CONDITIONALLY_INDEPENDENT_GIVEN = "ConditionallyIndependentGiven"
    JOINT_LIKELIHOOD = "JointLikelihood"
    KNOWN_COVARIANCE = "KnownCovariance"
    CLUSTERED = "Clustered"
    SEQUENTIAL_VALID = "SequentialValid"
    WORST_CASE_UNKNOWN = "WorstCaseUnknown"


@dataclasses.dataclass(frozen=True)
class IntervalEstimate:
    lower: float
    upper: float
    provenance_root: str
    episodes: frozenset = dataclasses.field(default_factory=frozenset)

    def __post_init__(self) -> None:
        endpoints = (self.lower, self.upper)
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in endpoints
        ):
            raise ValueError("IntervalEstimate endpoints must be finite real numbers")
        if not all(math.isfinite(value) for value in endpoints):
            raise ValueError("IntervalEstimate endpoints must be finite real numbers")
        if self.lower > self.upper:
            raise ValueError(
                "IntervalEstimate lower endpoint must not exceed upper endpoint"
            )
        if not isinstance(self.provenance_root, str) or not self.provenance_root:
            raise ValueError(
                "IntervalEstimate provenance_root must be a non-empty string"
            )
        if not isinstance(self.episodes, frozenset):
            raise ValueError("IntervalEstimate episodes must be a frozenset")


@dataclasses.dataclass(frozen=True)
class FusionResult:
    fused_interval: tuple
    effective_information_count: int


class FusionRejected(Exception):
    def __init__(self, reason: str, **details):
        self.reason = reason
        self.details = details
        super().__init__(f"FusionRejected({reason})")


def _detect_overlap(estimates) -> frozenset:
    if not estimates or not all(e.episodes for e in estimates):
        return frozenset()
    overlap = estimates[0].episodes
    for e in estimates[1:]:
        overlap = overlap & e.episodes
    return overlap


def fuse(estimates: list[IntervalEstimate], dependence: Dependence | None) -> FusionResult:
    """S2.5 dependence-mediated fusion.

    Only `Duplicate` (dedupe, no confidence gain) is spelled
    out as a concrete numeric rule by the specification. The other seven
    declarations each name a meaning and a permitted operation (e.g.
    "product likelihood or registered independent combination", "the
    registered clustered estimator", "an anytime-valid e-process or
    confidence-sequence rule") but not one general combination formula, so
    this reference raises NotImplementedError for them rather than
    inventing a formula the spec does not state. `dependence=None`
    (S2.11's demonstrated failure) is fully specified: fusion is rejected,
    and the detected episode overlap is reported.
    """
    if dependence is None:
        overlap = _detect_overlap(estimates)
        raise FusionRejected("MissingDependenceDeclaration", detected_overlap=sorted(overlap))
    if dependence is Dependence.DUPLICATE:
        if not estimates:
            raise FusionRejected("EmptyEstimateSet")
        roots = {e.provenance_root for e in estimates}
        if len(roots) != 1:
            raise ValueError("Duplicate dependence declared but provenance roots differ")
        intervals = {(estimate.lower, estimate.upper) for estimate in estimates}
        if len(intervals) != 1:
            raise FusionRejected("InconsistentDuplicateReports")
        return FusionResult(
            fused_interval=next(iter(intervals)),
            effective_information_count=1,
        )
    raise NotImplementedError(
        f"S2.5 names the meaning of {dependence.value} but not a "
        "general numeric combination rule beyond Duplicate; implementing one "
        "would be guessing semantics the specification leaves to a "
        "'registered' external estimator/design."
    )


# -- S2.7: identifiability gate ---------------------------------------------


@dataclasses.dataclass(frozen=True)
class NonIdentifiable:
    bayes_error: float
    policy: str
    observation_space: str


_PROBABILITY_TOTAL_TOLERANCE = 1e-12  # Allow floating-point summation error only.


def _validate_probability_mass_function(distribution: dict, argument_name: str) -> None:
    if not isinstance(distribution, Mapping) or not distribution:
        raise ValueError(f"{argument_name} must be a non-empty probability mass function")

    masses = []
    for outcome, mass in distribution.items():
        if isinstance(mass, bool) or not isinstance(mass, Number):
            raise ValueError(  # noqa: TRY004 - the public seam normalizes boundary failures
                f"{argument_name} contains a non-numeric probability mass"
            )
        if not isinstance(mass, Real):
            raise ValueError(  # noqa: TRY004 - normalize all PMF boundary failures
                f"{argument_name}[{outcome!r}] has unsupported probability scalar "
                f"type {type(mass).__name__}; expected a numbers.Real value other than bool"
            )
        if not math.isfinite(mass):
            raise ValueError(f"{argument_name} contains a non-finite probability mass")
        if mass < 0 or mass > 1 + _PROBABILITY_TOTAL_TOLERANCE:
            raise ValueError(f"{argument_name} contains a probability mass outside zero and one")
        masses.append(mass)

    total_mass = math.fsum(masses)
    if not math.isclose(
        total_mass,
        1.0,
        rel_tol=_PROBABILITY_TOTAL_TOLERANCE,
        abs_tol=_PROBABILITY_TOTAL_TOLERANCE,
    ):
        raise ValueError(f"{argument_name} probability masses must sum to one")


def total_variation(p: dict, q: dict) -> float:
    _validate_probability_mass_function(p, "p")
    _validate_probability_mass_function(q, "q")
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def identifiability_gate(dist_h: dict, dist_m: dict, policy: str = "pi", observation_space: str = "O"):
    """S2.7: e*_pi = (1 - TV(P_H^pi, P_M^pi)) / 2. If the two policy-induced
    distributions coincide, the claim checker returns NonIdentifiable and
    rejects Detected(H)/Detected(M)."""
    _validate_probability_mass_function(dist_h, "dist_h")
    _validate_probability_mass_function(dist_m, "dist_m")

    tv = min(1.0, total_variation(dist_h, dist_m))
    if tv == 0.0:
        return NonIdentifiable(bayes_error=0.5, policy=policy, observation_space=observation_space)
    return (1 - tv) / 2


class ClaimRejected(Exception):
    pass


def detected_claim(hypothesis_name: str, gate_result) -> str:
    """S2.7: 'the claim checker ... rejects Detected(H) and Detected(M)' when
    the gate reports NonIdentifiable."""
    if not isinstance(hypothesis_name, str) or not hypothesis_name.strip():
        raise ClaimRejected("hypothesis_name must be a non-empty string")
    if isinstance(gate_result, NonIdentifiable):
        raise ClaimRejected(f"Detected({hypothesis_name}) rejected: {gate_result}")
    if (
        type(gate_result) is not float
        or not math.isfinite(gate_result)
        or not 0.0 <= gate_result < 0.5
    ):
        raise ClaimRejected("invalid identifiability result")
    return f"Detected({hypothesis_name})"


# -- S2.6: minimal claim-requirement grammar --------------------------------


@dataclasses.dataclass(frozen=True)
class ModalityRequirement:
    grade: Grade
    pattern: str


@dataclasses.dataclass(frozen=True)
class AndRequirement:
    left: object
    right: object


@dataclasses.dataclass(frozen=True)
class OrRequirement:
    left: object
    right: object


@dataclasses.dataclass(frozen=True)
class Same:
    fields: tuple


@dataclasses.dataclass(frozen=True)
class Identified:
    declaration: str


def check_requirement(requirement, evidence_bundle: list[Evidence]) -> bool:
    """S2.6 claim requirements are positive Boolean formulas over
    modality-indexed propositions. AND/OR over modality leaves and SAME(...)
    (index-field equality) are mechanical. IDENTIFIED(...) is NOT reduced by
    the specification to one general procedure beyond the specific S2.7
    identifiability-gate case -- see the NotImplementedError below
    (implementation boundary)."""
    if isinstance(requirement, ModalityRequirement):
        return any(
            e.modality is requirement.grade and _schema(e.proposition) == requirement.pattern
            for e in evidence_bundle
        )
    if isinstance(requirement, AndRequirement):
        return check_requirement(requirement.left, evidence_bundle) and check_requirement(
            requirement.right, evidence_bundle
        )
    if isinstance(requirement, OrRequirement):
        return check_requirement(requirement.left, evidence_bundle) or check_requirement(
            requirement.right, evidence_bundle
        )
    if isinstance(requirement, Same):
        values = {tuple(getattr(e.index, f) for f in requirement.fields) for e in evidence_bundle}
        return len(values) <= 1
    if isinstance(requirement, Identified):
        raise NotImplementedError(
            "IDENTIFIED(...) validity in general is not reduced "
            "to one procedure by the specification beyond S2.7's specific "
            "worked identifiability-gate case (identifiability_gate above)."
        )
    raise TypeError(f"unrecognized requirement node: {requirement!r}")
