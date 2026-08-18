"""Total residual definition and one quotient (S5).

This module implements the specification's totalised transition system,
residual construction, finite supplied-continuation agreement check, and
bounded witness-search result algebra.

Two complementary realizations are provided, matching the two families of S5
acceptance tests:

* `TotalizedLTS` -- a generic totalized transition system (S5.1's disabled
  sink for an unmatched action, plus agreement over an explicit finite
  continuation set) for tests that are abstract over the underlying monitor
  (enabled-action domains, canonicalization).
* `monitor_residual` -- realizes R_u^C concretely against monitor.py's
  SingleClauseMonitor engine, for tests that exercise S1's actual terminal
  conversion (S1.7) through the residual (e.g. Complete vs ObservationCut
  distinguishing an Open obligation).

`bounded_compare` implements S5.7's witness-search algebra
(ComparisonResult): it returns Distinguished, NoWitnessWithinBound, or
SearchIncomplete, never ProvedEquivalent. The symbolic proof that S5.7 requires
for ProvedEquivalent is not constructed here.

`canonical_observation_digest`, `bounded_compare`, and
`TotalizedLTS.agrees_on_supplied_continuations` are generic utilities over the
documented encodable domain. The monitor-driven path is stricter:
`monitor_residual` produces exact `TotalObservation` roots, and
conformance-fragment comparisons compare exact `TotalObservation` values only.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import hashlib
import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet

from .events import (
    AgentAbort,
    Complete,
    DomainEvent,
    ExternalCrash,
    MalformedEvent,
    ObservationCut,
    TerminalEvent,
    TickEvent,
    Timeout,
)
from .verdict import Mode, Summary, VerdictObject


def _structurally_equal(left, right) -> bool:
    """Compare typed observation structures without Python scalar coercions."""
    try:
        return _admitted_observation_encoding(left) == _admitted_observation_encoding(
            right
        )
    except ValueError:
        pass

    if type(left) is not type(right):
        return False

    if dataclasses.is_dataclass(left):
        return all(
            _structurally_equal(getattr(left, field.name), getattr(right, field.name))
            for field in dataclasses.fields(left)
        )

    model_fields = getattr(type(left), "model_fields", None)
    if model_fields is not None:
        return all(
            _structurally_equal(getattr(left, name), getattr(right, name))
            for name in model_fields
        )

    if isinstance(left, Mapping):
        if len(left) != len(right):
            return False
        unmatched = list(right.items())
        for left_key, left_value in left.items():
            for index, (right_key, right_value) in enumerate(unmatched):
                if _structurally_equal(left_key, right_key) and _structurally_equal(
                    left_value, right_value
                ):
                    unmatched.pop(index)
                    break
            else:
                return False
        return True

    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _structurally_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )

    if isinstance(left, AbstractSet) and not isinstance(left, (str, bytes)):
        if len(left) != len(right):
            return False
        unmatched = list(right)
        for left_item in left:
            for index, right_item in enumerate(unmatched):
                if _structurally_equal(left_item, right_item):
                    unmatched.pop(index)
                    break
            else:
                return False
        return True

    return left == right


class DisabledReason(enum.Enum):
    """Typed disabled-transition reasons from S5."""

    NOT_ENABLED = "NotEnabled"
    AFTER_TERMINAL = "AfterTerminal"
    AFTER_FAULT = "AfterFault"


@dataclasses.dataclass(frozen=True, eq=False)
class DisabledSink:
    """The first disabled attempt and its frozen source observation."""

    frozen_contract_observation: object
    attempted: object
    enabled_set: frozenset
    reason: DisabledReason

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return (
            _structurally_equal(
                self.frozen_contract_observation, other.frozen_contract_observation
            )
            and _structurally_equal(self.attempted, other.attempted)
            and _structurally_equal(self.enabled_set, other.enabled_set)
            and self.reason == other.reason
        )

    __hash__ = None  # type: ignore[assignment]


def _type_identity(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _canonical_sort_key(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _canonical_observation_value(
    value: object, active_container_ids: set[int] | None = None
) -> object:
    if active_container_ids is None:
        active_container_ids = set()

    value_type = type(value)
    type_identity = _type_identity(value)

    if value_type in {
        Disabled,
        DisabledSink,
        NoDisabledLabel,
        Complete,
        AgentAbort,
        ExternalCrash,
        Timeout,
        ObservationCut,
    }:
        return _canonical_composite_value(
            value,
            active_container_ids,
            lambda: [
                "dataclass",
                type_identity,
                [
                    [
                        field.name,
                        _canonical_observation_value(
                            getattr(value, field.name), active_container_ids
                        ),
                    ]
                    for field in dataclasses.fields(value)
                ],
            ],
        )

    if value_type is TotalObservation:
        return _canonical_composite_value(
            value,
            active_container_ids,
            lambda: [
                "dataclass",
                type_identity,
                [
                    [
                        field.name,
                        _canonical_observation_value(
                            getattr(value, field.name), active_container_ids
                        ),
                    ]
                    for field in dataclasses.fields(TotalObservation)
                ],
            ],
        )

    if value_type is VerdictObject:
        return _canonical_composite_value(
            value,
            active_container_ids,
            lambda: [
                "model",
                type_identity,
                [
                    [
                        name,
                        _canonical_observation_value(
                            getattr(value, name), active_container_ids
                        ),
                    ]
                    for name in VerdictObject.model_fields
                ],
            ],
        )

    if value_type in {DisabledReason, Summary, Mode}:
        return ["enum", type_identity, value.name]

    if value_type is dict:

        def canonical_mapping() -> object:
            items = [
                [
                    _canonical_observation_value(key, active_container_ids),
                    _canonical_observation_value(item_value, active_container_ids),
                ]
                for key, item_value in value.items()
            ]
            items.sort(key=_canonical_sort_key)
            return ["mapping", type_identity, items]

        return _canonical_composite_value(
            value, active_container_ids, canonical_mapping
        )

    if value_type in {tuple, list}:
        return _canonical_composite_value(
            value,
            active_container_ids,
            lambda: [
                "sequence",
                type_identity,
                [
                    _canonical_observation_value(item, active_container_ids)
                    for item in value
                ],
            ],
        )

    if value_type is frozenset:

        def canonical_set() -> object:
            items = [
                _canonical_observation_value(item, active_container_ids)
                for item in value
            ]
            items.sort(key=_canonical_sort_key)
            return ["set", type_identity, items]

        return _canonical_composite_value(value, active_container_ids, canonical_set)

    if value is None:
        return ["none", type_identity]
    if value_type is bool:
        return ["bool", type_identity, value]
    if value_type is int:
        return ["int", type_identity, str(value)]
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(
                "canonical observation digest does not support non-finite values "
                f"of type {type_identity}"
            )
        normalized = 0.0 if value == 0.0 else value
        return ["float", type_identity, normalized.hex()]
    if value_type is str:
        return ["str", type_identity, value]

    raise ValueError(
        f"canonical observation digest does not support values of type {type_identity}"
    )


def _canonical_composite_value(
    value: object,
    active_container_ids: set[int],
    canonicalize: Callable[[], object],
) -> object:
    value_id = id(value)
    if value_id in active_container_ids:
        raise ValueError(
            "canonical observation digest does not support cyclic values of type "
            f"{_type_identity(value)}"
        )

    active_container_ids.add(value_id)
    try:
        return canonicalize()
    finally:
        active_container_ids.remove(value_id)


def canonical_observation_digest(observation: object) -> str:
    """Return a deterministic SHA-256 identity for a structural observation."""
    return hashlib.sha256(_admitted_observation_encoding(observation)).hexdigest()


def _admitted_observation_encoding(observation: object) -> bytes:
    """Validate and canonically encode one admitted semantic observation."""
    canonical = _canonical_observation_value(observation)
    return json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclasses.dataclass(frozen=True)
class _TerminalState:
    """A totalised state whose source transition crossed a terminal label."""

    source_state: object


@dataclasses.dataclass(frozen=True)
class NoDisabledLabel:
    """No continuation label has been disabled."""


@dataclasses.dataclass(frozen=True, eq=False)
class Disabled:
    """The first disabled continuation label and its typed reason."""

    attempted: object
    enabled_set: frozenset
    reason: DisabledReason

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return (
            _structurally_equal(self.attempted, other.attempted)
            and _structurally_equal(self.enabled_set, other.enabled_set)
            and self.reason == other.reason
        )

    __hash__ = None  # type: ignore[assignment]


@dataclasses.dataclass(frozen=True, eq=False)
class TotalObservation:
    """The complete contract observation and its continuation status."""

    contract_observation: object
    continuation_status: NoDisabledLabel | Disabled

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return _structurally_equal(
            self.contract_observation, other.contract_observation
        ) and _structurally_equal(self.continuation_status, other.continuation_status)

    __hash__ = None  # type: ignore[assignment]


_ENABLED_ACTIONS_ABSENT = object()


def _declared_enabled_actions(observation):
    """The `enabled_actions` field of a contract observation, or the absent
    marker when the observation carries none. This reference `VerdictObject`
    declares that omission (verdict.py, divergence list), so the consistency
    invariant below has nothing to check against such an observation."""
    if isinstance(observation, Mapping):
        if "enabled_actions" in observation:
            return observation["enabled_actions"]
        return _ENABLED_ACTIONS_ABSENT
    return getattr(observation, "enabled_actions", _ENABLED_ACTIONS_ABSENT)


def _canonical_action_set(declared):
    """Canonicalise a declared enabled-action collection, or None when the
    value is not a collection of actions at all (which is itself an
    inconsistency, since no canonical action set can equal it)."""
    if isinstance(declared, (str, bytes)) or not isinstance(
        declared, (AbstractSet, Sequence)
    ):
        return None
    return frozenset(declared)


class TotalizedLTS:
    """A concrete realization of S5.1's totalized LTS: given a concrete
    transition function and per-state enabled-action sets, an unmatched
    label enters the canonical Disabled sink, and any label after a terminal
    label is likewise disabled (S5.1: 'A nonterminal label after a terminal
    boundary is disabled with reason AfterTerminal')."""

    def __init__(
        self,
        transition: Callable[[object, str], object],
        enabled: Callable[[object], frozenset],
        observe: Callable[[object], object],
        terminal_labels: frozenset = frozenset(
            {"Complete", "AgentAbort", "ExternalCrash", "Timeout", "ObservationCut"}
        ),
    ):
        self.transition = transition
        self.enabled = enabled
        self.observe = observe
        self.terminal_labels = terminal_labels

    def delta_bar(self, state, label: str):
        """Apply one totalised transition to a raw or already-totalised state."""
        if isinstance(state, DisabledSink):
            return state

        if isinstance(state, _TerminalState):
            return DisabledSink(
                frozen_contract_observation=self.observe(state.source_state),
                attempted=label,
                enabled_set=frozenset(),
                reason=DisabledReason.AFTER_TERMINAL,
            )

        allowed = frozenset(self.enabled(state))
        if label not in allowed:
            frozen_observation = self.observe(state)
            declared = _declared_enabled_actions(frozen_observation)
            if (
                declared is not _ENABLED_ACTIONS_ABSENT
                and _canonical_action_set(declared) != allowed
            ):
                raise ValueError(
                    "enabled-set consistency invariant (S5): a "
                    "disabled transition from Base(s) must record "
                    "Canonicalise(Enabled(s)) == Obs_C(s).enabled_actions, but "
                    f"the frozen contract observation declares {declared!r} "
                    f"against the enabled set {set(allowed)!r}; the totalisation "
                    "adapter must not emit an inconsistent DisabledSink"
                )
            return DisabledSink(
                frozen_contract_observation=frozen_observation,
                attempted=label,
                enabled_set=allowed,
                reason=DisabledReason.NOT_ENABLED,
            )

        next_state = self.transition(state, label)
        if label in self.terminal_labels:
            return _TerminalState(next_state)
        return next_state

    def residual(self, state, word: Sequence[str]):
        """R_u^C(w) from the residual construction.

        The first disabled label freezes Obs_C of the current state in an
        internal DisabledSink. Its offset remains replay metadata and is not
        part of residual equality. Every path projects the final totalised
        state to a TotalObservation.
        """
        current = state
        for label in word:
            current = self.delta_bar(current, label)

        if isinstance(current, DisabledSink):
            return TotalObservation(
                contract_observation=current.frozen_contract_observation,
                continuation_status=Disabled(
                    attempted=current.attempted,
                    enabled_set=current.enabled_set,
                    reason=current.reason,
                ),
            )
        if isinstance(current, _TerminalState):
            current = current.source_state
        return TotalObservation(
            contract_observation=self.observe(current),
            continuation_status=NoDisabledLabel(),
        )

    def agrees_on_supplied_continuations(
        self, u_state, v_state, continuations: Sequence[Sequence[str]]
    ) -> bool:
        """Return whether residual encodings agree for every supplied word.

        The result covers exactly the words supplied, which is weaker than
        the universally quantified congruence of S5.4: agreement on a finite
        set of supplied words is not a merge certificate. Comparisons inside
        the conformance fragment use exact TotalObservation residuals.
        """
        for word in continuations:
            left_observation = self.residual(u_state, word)
            right_observation = self.residual(v_state, word)
            left_encoding = _admitted_observation_encoding(left_observation)
            right_encoding = _admitted_observation_encoding(right_observation)
            if left_encoding != right_encoding:
                return False
        return True


def _attempted_label(event) -> object:
    if isinstance(event, TickEvent):
        return ("Tick", event.tick)
    tag = getattr(event, "tag", None)
    if tag is not None:
        return (str(tag), copy.deepcopy(getattr(event, "fields", {})))
    kind = getattr(event, "kind", None)
    if kind is not None:
        return copy.deepcopy(kind)
    if dataclasses.is_dataclass(event):
        structural_fields = tuple(
            (field.name, copy.deepcopy(getattr(event, field.name)))
            for field in dataclasses.fields(event)
            if field.name not in {"event_id", "tick"}
        )
        return (type(event).__name__, structural_fields)
    return type(event).__name__


def _monitor_observation(
    verdict: VerdictObject,
    continuation_status: NoDisabledLabel | Disabled,
) -> TotalObservation:
    return TotalObservation(
        contract_observation=verdict,
        continuation_status=continuation_status,
    )


def monitor_residual(monitor, continuation_events) -> TotalObservation:
    """Realizes R_u^C for monitor.py's SingleClauseMonitor: replays
    `continuation_events` on a DEEP COPY of the prefix's monitor state, so
    evaluating several continuations from the same prefix does not mutate it
    (S5.3's totalized-function requirement)."""
    clone = copy.deepcopy(monitor)
    verdict = clone.current_verdict()
    continuation_status: NoDisabledLabel | Disabled = NoDisabledLabel()
    for event in continuation_events:
        if isinstance(event, MalformedEvent):
            raise ValueError(  # noqa: TRY004 - one normalized continuation-boundary error
                "MalformedEvent is not a Closed Core continuation event"
            )
        if verdict.mode is Mode.FAULTED:
            continuation_status = Disabled(
                attempted=_attempted_label(event),
                enabled_set=frozenset(),
                reason=DisabledReason.AFTER_FAULT,
            )
            break
        if verdict.mode is not Mode.RUNNING:
            continuation_status = Disabled(
                attempted=_attempted_label(event),
                enabled_set=frozenset(),
                reason=DisabledReason.AFTER_TERMINAL,
            )
            break
        if isinstance(event, (DomainEvent, TickEvent, TerminalEvent)):
            event = dataclasses.replace(event, tick=clone.tick + event.tick)
        verdict = clone.step(event)
    return _monitor_observation(verdict, continuation_status)


# -- S5.7: witness-search algebra --------------------------------------------


class InvalidComparisonRequest(ValueError):
    """The comparison request is malformed; no search was run and no result exists."""


@dataclasses.dataclass(frozen=True)
class UnitDataDomain:
    kind: str = dataclasses.field(init=False, default="unit")
    note: str = dataclasses.field(
        init=False,
        default="contract declares no binding fields",
    )


@dataclasses.dataclass(frozen=True)
class UnitDataUnitAdvanceBound:
    continuation_length: int
    data_domain: UnitDataDomain = dataclasses.field(
        init=False,
        default_factory=UnitDataDomain,
    )
    timing_bound: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if type(self.continuation_length) is not int or self.continuation_length < 0:
            raise InvalidComparisonRequest(
                "continuation_length must be a non-negative integer, "
                f"got {self.continuation_length!r}"
            )
        object.__setattr__(self, "timing_bound", self.continuation_length)


@dataclasses.dataclass(frozen=True)
class ComparisonScope:
    contract_environment: object
    observation_projection: str
    continuation_family: str
    bound: object
    exclusions: tuple


@dataclasses.dataclass(frozen=True)
class ExhaustiveSearchReceipt:
    words_enumerated: int
    max_length: int
    alphabet: tuple


@dataclasses.dataclass(frozen=True)
class PartialCoverage:
    searched: str | int
    not_searched: str | int


@dataclasses.dataclass(frozen=True)
class PartialSearchReceipt:
    partial_coverage: PartialCoverage


@dataclasses.dataclass(frozen=True)
class Distinguished:
    scope: ComparisonScope
    witness: tuple
    left_observation: object
    right_observation: object


@dataclasses.dataclass(frozen=True)
class ProvedEquivalent:
    scope: ComparisonScope
    certificate: object


@dataclasses.dataclass(frozen=True)
class NoWitnessWithinBound:
    scope: ComparisonScope
    exhaustive_search_receipt: ExhaustiveSearchReceipt


@dataclasses.dataclass(frozen=True)
class SearchIncomplete:
    scope: ComparisonScope
    reason: str
    incomplete_comparison_receipt: PartialSearchReceipt


@dataclasses.dataclass(frozen=True)
class Untested:
    scope: ComparisonScope
    reason: str


def bounded_compare(
    residual_u: Callable[[tuple], object],
    residual_v: Callable[[tuple], object],
    alphabet: Sequence[str],
    bound: UnitDataUnitAdvanceBound,
    max_words: int | None = None,
    *,
    contract_environment: object,
    observation_projection: str,
    continuation_family: str,
) -> Distinguished | NoWitnessWithinBound | SearchIncomplete:
    """S5.7's ComparisonResult, realized as a bounded exhaustive search.

    This never returns ProvedEquivalent. S5.7 states that ProvedEquivalent(F,
    pi) "implies global equivalence only if pi also proves that F = Sigma_C* or
    otherwise covers the complete admissible domain symbolically", that is, it
    requires a symbolic proof over a possibly infinite alphabet, which this
    reference's brute-force enumeration does not construct. Distinguished
    records a witness, NoWitnessWithinBound records a completed exhaustive
    search, and SearchIncomplete records exhaustion of an explicit word
    budget before the declared domain was exhausted.

    This comparator covers words in an immutable unit-data domain. Each label
    advances time by one relative tick, so the timing bound is the continuation
    length. The conformance fragment's comparisons return exact TotalObservation
    values from both callbacks.
    """
    if contract_environment is None or (
        type(contract_environment) is str
        and (not contract_environment.strip() or contract_environment == "undeclared")
    ):
        raise InvalidComparisonRequest("contract_environment must be declared")
    if (
        type(observation_projection) is not str
        or not observation_projection.strip()
        or observation_projection == "undeclared"
    ):
        raise InvalidComparisonRequest("observation_projection must be declared")
    if (
        type(continuation_family) is not str
        or not continuation_family.strip()
        or continuation_family == "undeclared"
    ):
        raise InvalidComparisonRequest("continuation_family must be declared")

    if type(bound) is not UnitDataUnitAdvanceBound:
        raise InvalidComparisonRequest(
            "bound must be an exact UnitDataUnitAdvanceBound, "
            f"got {_type_identity(bound)}"
        )
    max_length = bound.continuation_length

    if max_words is not None and (
        isinstance(max_words, bool) or not isinstance(max_words, int) or max_words <= 0
    ):
        raise InvalidComparisonRequest(
            f"max_words must be a positive integer or None, got {max_words!r}"
        )

    if isinstance(
        alphabet, (str, bytes, bytearray, Mapping, AbstractSet)
    ) or not isinstance(alphabet, Sequence):
        raise InvalidComparisonRequest(
            f"alphabet must be a non-text Sequence, got {_type_identity(alphabet)}"
        )
    alphabet_tuple = tuple(alphabet)

    for index, symbol in enumerate(alphabet_tuple):
        if type(symbol) is not str:
            raise InvalidComparisonRequest(
                f"alphabet element {index} must be a str, got {_type_identity(symbol)}"
            )
    if len(set(alphabet_tuple)) != len(alphabet_tuple):
        raise InvalidComparisonRequest("alphabet elements must be pairwise distinct")

    scope = ComparisonScope(
        contract_environment=contract_environment,
        observation_projection=observation_projection,
        continuation_family=continuation_family,
        bound=bound,
        exclusions=(),
    )
    domain_size = sum(len(alphabet_tuple) ** length for length in range(max_length + 1))
    words_enumerated = 0

    for length in range(max_length + 1):
        for word in itertools.product(alphabet_tuple, repeat=length):
            if max_words is not None and words_enumerated == max_words:
                return SearchIncomplete(
                    scope=scope,
                    reason="WordBudgetExhausted",
                    incomplete_comparison_receipt=PartialSearchReceipt(
                        PartialCoverage(
                            searched=max_words,
                            not_searched=domain_size - max_words,
                        )
                    ),
                )
            ru = residual_u(word)
            rv = residual_v(word)
            try:
                ru_encoding = _admitted_observation_encoding(ru)
            except ValueError as exc:
                raise InvalidComparisonRequest(
                    "left residual callback returned an observation outside the "
                    f"admitted domain: {_type_identity(ru)}"
                ) from exc
            try:
                rv_encoding = _admitted_observation_encoding(rv)
            except ValueError as exc:
                raise InvalidComparisonRequest(
                    "right residual callback returned an observation outside the "
                    f"admitted domain: {_type_identity(rv)}"
                ) from exc
            words_enumerated += 1
            if ru_encoding != rv_encoding:
                return Distinguished(
                    scope=scope,
                    witness=word,
                    left_observation=ru,
                    right_observation=rv,
                )
    return NoWitnessWithinBound(
        scope=scope,
        exhaustive_search_receipt=ExhaustiveSearchReceipt(
            words_enumerated=words_enumerated,
            max_length=max_length,
            alphabet=alphabet_tuple,
        ),
    )


# -- S5.5-S5.6: quotient / gate projection -----------------------------------


def gate_projection(summary) -> str:
    """rho_gate (S5.5): Satisfied -> permit; Violated and Unknown -> deny.
    Deliberately coarser than the full three-valued quotient."""
    if type(summary) is not Summary:
        raise TypeError("summary must be a Summary")
    return "permit" if summary is Summary.SATISFIED else "deny"


def binary_accept(summary) -> bool:
    """S5.6's binary completed-trace acceptance: only Satisfied is
    'accepted'; both Violated and Unknown are 'not accepted'."""
    if type(summary) is not Summary:
        raise TypeError("summary must be a Summary")
    return summary is Summary.SATISFIED
