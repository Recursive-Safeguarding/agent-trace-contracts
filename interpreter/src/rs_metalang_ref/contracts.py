"""Contract grammar and per-clause-form semantics (S1.1, S1.5).

    Contract ::= Clause | Contract and Contract
    Clause   ::= always | forbid | before | after | at_end | budget | flow

S1.1 requires pattern matching to be total, deterministic, and finite, but it
does not choose one concrete matcher or pattern syntax. This module therefore
does not implement a general multi-clause compiler. The executable fragment is
the one-clause form used by S1's worked examples:

    after TRIGGER when GUARD require RESPONSE within D
        discharge MODE on_agent_abort DISPOSITION

Proposition keys are opaque strings in this release fragment: the fragment
assigns no temporal interpretation to any key, including keys that resemble
as-of syntax such as `Sensitive(f)@0`. Predicate temporal kinds are not
represented.

The clause dataclasses model the grammar structurally. `always_check`,
`forbid_check`, `before_check`, `at_end_check`, `budget_check`, `flow_check`,
and `conjunction_check` implement S1.5's verdict mappings once the caller has
already supplied a K3 predicate value.
"""

from __future__ import annotations

import dataclasses
from enum import Enum

from .kleene import K3

Severity = str


class InvalidAfterClauseSpec(ValueError):
    """The executable after-clause fragment rejects this shape at construction."""


class InvalidBinding(ValueError):
    """A runtime event or direct guard query has an invalid binding."""


_RESERVED_BINDING_FIELDS = frozenset({"discharges"})


# -- discharge / abort dispositions (part of the `after` clause grammar) --


@dataclasses.dataclass(frozen=True)
class Linear:
    pass


@dataclasses.dataclass(frozen=True)
class Broadcast:
    key: str


DischargeMode = Linear | Broadcast


@dataclasses.dataclass(frozen=True)
class GroundGuard:
    proposition_key: str

    def __post_init__(self) -> None:
        if type(self.proposition_key) is not str or not self.proposition_key:
            raise InvalidAfterClauseSpec("GroundGuard requires a non-empty string proposition_key")


@dataclasses.dataclass(frozen=True)
class BoundUnaryGuard:
    predicate: str
    binding_field: str

    def __post_init__(self) -> None:
        if type(self.predicate) is not str or not self.predicate:
            raise InvalidAfterClauseSpec("BoundUnaryGuard requires a non-empty string predicate")
        if type(self.binding_field) is not str or not self.binding_field:
            raise InvalidAfterClauseSpec("BoundUnaryGuard requires a non-empty string binding_field")


Guard = GroundGuard | BoundUnaryGuard


@dataclasses.dataclass(frozen=True)
class AfterClauseSpec:
    """The executable unary-or-fieldless after-clause fragment."""

    clause_id: str
    trigger_tag: str
    response_tag: str
    binding_fields: tuple[str, ...]
    guard: Guard
    bound: int
    discharge: DischargeMode = dataclasses.field(default_factory=Linear)
    on_agent_abort: AbortDisposition = dataclasses.field(default_factory=lambda: Breach())

    def __post_init__(self) -> None:
        for field_name in ("clause_id", "trigger_tag", "response_tag"):
            value = getattr(self, field_name)
            if type(value) is not str or not value:
                raise InvalidAfterClauseSpec(f"{field_name} must be a non-empty string")
        if not isinstance(self.binding_fields, tuple):
            raise InvalidAfterClauseSpec("binding_fields must be a tuple of strings")
        if any(type(field) is not str for field in self.binding_fields):
            raise InvalidAfterClauseSpec("binding_fields must contain only strings")
        if any(not field for field in self.binding_fields):
            raise InvalidAfterClauseSpec("binding_fields must not contain empty names")
        if len(set(self.binding_fields)) != len(self.binding_fields):
            raise InvalidAfterClauseSpec("binding_fields must not contain duplicates")
        if any(field in _RESERVED_BINDING_FIELDS for field in self.binding_fields):
            raise InvalidAfterClauseSpec("binding_fields must not contain reserved metadata names")
        if len(self.binding_fields) > 1:
            raise InvalidAfterClauseSpec("this executable fragment admits at most one binding field")
        if type(self.bound) is not int or self.bound < 0:
            raise InvalidAfterClauseSpec("bound must be a natural number")
        if type(self.discharge) not in (Linear, Broadcast):
            raise InvalidAfterClauseSpec(
                "discharge must be exactly Linear or Broadcast"
            )
        if type(self.on_agent_abort) not in (Breach, Indeterminate, WaiveIf):
            raise InvalidAfterClauseSpec(
                "on_agent_abort must be exactly Breach, Indeterminate, or WaiveIf"
            )
        if type(self.guard) is GroundGuard:
            return
        if type(self.guard) is BoundUnaryGuard:
            if self.binding_fields != (self.guard.binding_field,):
                raise InvalidAfterClauseSpec(
                    "BoundUnaryGuard.binding_field must be the single declared binding field"
                )
            return
        raise InvalidAfterClauseSpec("guard must be exactly GroundGuard or BoundUnaryGuard")


@dataclasses.dataclass(frozen=True)
class Breach:
    pass


@dataclasses.dataclass(frozen=True)
class Indeterminate:
    pass


@dataclasses.dataclass(frozen=True)
class WaiveIf:
    authority_key: str

    def __post_init__(self) -> None:
        if type(self.authority_key) is not str or not self.authority_key:
            raise InvalidAfterClauseSpec(
                "WaiveIf requires a non-empty string authority_key"
            )


AbortDisposition = Breach | Indeterminate | WaiveIf


# -- clause grammar (structural) -----------------------------------------


@dataclasses.dataclass(frozen=True)
class AlwaysClause:
    clause_id: str
    predicate_key: str
    severity: Severity


@dataclasses.dataclass(frozen=True)
class ForbidClause:
    clause_id: str
    event_pattern: str
    predicate_key: str
    authority_key: str
    severity: Severity


@dataclasses.dataclass(frozen=True)
class BeforeClause:
    clause_id: str
    event_pattern: str
    evidence_pattern: str
    severity: Severity


@dataclasses.dataclass(frozen=True)
class AfterClause:
    clause_id: str
    trigger_pattern: str
    response_pattern: str
    binding_fields: tuple[str, ...]
    guard: Guard
    bound: int
    discharge: DischargeMode = dataclasses.field(default_factory=Linear)
    on_agent_abort: AbortDisposition = dataclasses.field(default_factory=Breach)
    severity: Severity = "high"


@dataclasses.dataclass(frozen=True)
class AtEndClause:
    clause_id: str
    predicate_key: str
    severity: Severity


@dataclasses.dataclass(frozen=True)
class BudgetClause:
    clause_id: str
    metric_key: str
    bound: float
    severity: Severity


@dataclasses.dataclass(frozen=True)
class FlowClause:
    clause_id: str
    label: str
    sink: str
    authority_key: str
    severity: Severity


@dataclasses.dataclass(frozen=True)
class AndContract:
    left: Clause
    right: Clause


Clause = (
    AlwaysClause
    | ForbidClause
    | BeforeClause
    | AfterClause
    | AtEndClause
    | BudgetClause
    | FlowClause
    | AndContract
)


# -- S1.5: per-clause-form semantics over an already-evaluated K3 value --


class ClauseCheck(Enum):
    HELD = "held"
    VIOLATION = "violation"
    PENDING = "pending"


def always_check(p: K3) -> ClauseCheck:
    """`always P` (S1.5): T -> held, F -> emit a violation, U -> pending."""
    return {K3.T: ClauseCheck.HELD, K3.F: ClauseCheck.VIOLATION, K3.U: ClauseCheck.PENDING}[p]


def forbid_check(p: K3, authority: K3) -> ClauseCheck:
    """`forbid E when P unless A` (S1.5): theta = P and not A;
    T -> violation, F -> no violation, U -> pending."""
    theta = p & (~authority)
    return {K3.T: ClauseCheck.VIOLATION, K3.F: ClauseCheck.HELD, K3.U: ClauseCheck.PENDING}[theta]


def before_check(evidence_status: str) -> ClauseCheck:
    """`before E require EvidencePattern` (S1.5): the monitor queries the S2
    indexed evidence ledger strictly before the event tick; `evidence_status`
    is the caller's already-computed classification of that query."""
    mapping = {
        "definite_match": ClauseCheck.HELD,
        "definite_absence": ClauseCheck.VIOLATION,
        "incomplete": ClauseCheck.PENDING,
    }
    if evidence_status not in mapping:
        raise ValueError(f"unrecognized evidence_status: {evidence_status!r}")
    return mapping[evidence_status]


def at_end_check(p: K3, is_actual_end: bool):
    """`at_end P` (S1.5): evaluated only at Complete / AgentAbort /
    ExternalCrash / Timeout; ObservationCut yields Unknown (unless a
    violation was already established elsewhere)."""
    from .verdict import Summary

    if not is_actual_end:
        return Summary.UNKNOWN
    return {K3.T: Summary.SATISFIED, K3.F: Summary.VIOLATED, K3.U: Summary.UNKNOWN}[p]


def budget_check(value: float | None, bound: float) -> ClauseCheck:
    """`budget M <= b` (S1.5): unavailable -> pending; else compare to bound."""
    if value is None:
        return ClauseCheck.PENDING
    return ClauseCheck.HELD if value <= bound else ClauseCheck.VIOLATION


def flow_check(has_label: K3, targets_sink: K3, authority: K3) -> ClauseCheck:
    """`flow L -> S only_if A` (S1.5): same T/F/U rule as `forbid`, over
    theta = HasLabel(L) and Targets(S) and not A."""
    theta = has_label & targets_sink & (~authority)
    return {K3.T: ClauseCheck.VIOLATION, K3.F: ClauseCheck.HELD, K3.U: ClauseCheck.PENDING}[theta]


def conjunction_check(left: ClauseCheck, right: ClauseCheck, at_closing_boundary: bool) -> ClauseCheck:
    """`C1 and C2` (S1.5): a proved violation in either component makes the
    conjunction violated; it is satisfied only at a closing terminal
    boundary when both components are satisfied; otherwise unknown."""
    if ClauseCheck.VIOLATION in (left, right):
        return ClauseCheck.VIOLATION
    if at_closing_boundary and left is ClauseCheck.HELD and right is ClauseCheck.HELD:
        return ClauseCheck.HELD
    return ClauseCheck.PENDING


# -- S1.1 static well-formedness ------------------------------------------


def unique_clause_ids(clause_ids) -> bool:
    """Well-formedness condition 8 ("Clause identifiers are unique."):
    fully checkable structurally, unlike conditions 1-7."""
    ids = list(clause_ids)
    return len(ids) == len(set(ids))


@dataclasses.dataclass(frozen=True)
class WellFormednessCertificate:
    """Each instance records seven caller attestations for S1.1.

    The caller supplies seven Boolean `attested_*` values to declare that
    conditions 1 to 7 hold. This class records those declarations. It does not
    inspect the external matchers or adapters, and it does not independently
    check the other declared conditions.
    `declared_well_formedness_premises_hold` combines the seven supplied
    Booleans and computes clause-ID uniqueness as condition 8.
    """

    attested_total_deterministic_pattern_matching: bool
    attested_variables_bound: bool
    attested_trigger_time_guard_only: bool
    attested_total_atomic_predicate_adapters: bool
    attested_total_effect_adapters: bool
    attested_decidable_broadcast_keys: bool
    attested_typed_waive_if_predicates: bool
    clause_ids: tuple

    def __post_init__(self) -> None:
        for field_name in (
            "attested_total_deterministic_pattern_matching",
            "attested_variables_bound",
            "attested_trigger_time_guard_only",
            "attested_total_atomic_predicate_adapters",
            "attested_total_effect_adapters",
            "attested_decidable_broadcast_keys",
            "attested_typed_waive_if_predicates",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"{field_name} must be a Boolean")

    def declared_well_formedness_premises_hold(self) -> bool:
        return (
            self.attested_total_deterministic_pattern_matching
            and self.attested_variables_bound
            and self.attested_trigger_time_guard_only
            and self.attested_total_atomic_predicate_adapters
            and self.attested_total_effect_adapters
            and self.attested_decidable_broadcast_keys
            and self.attested_typed_waive_if_predicates
            and unique_clause_ids(self.clause_ids)
        )
