"""Single-`after`-clause lifecycle subset for the reference interpreter.

S1.5 gives the related lifecycle order ("transition order"). S1.6 gives the
terminal-conversion rules. The aggregate summary rule sits alongside them.

The monitor executes one clause of the shape
`after TRIGGER when GUARD require RESPONSE within D`, taking at most one
observation entry per event and rejecting effect receipts. Extending it to
several clauses at once would need a pattern-matching algorithm, which the
specification leaves open.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from enum import Enum

from .contracts import (
    AbortDisposition,
    AfterClauseSpec,
    BoundUnaryGuard,
    Breach,
    GroundGuard,
    Indeterminate,
    InvalidBinding,
    Linear,
    WaiveIf,
)
from .events import (
    AgentAbort,
    Complete,
    DomainEvent,
    Event,
    ExternalCrash,
    ObservationCut,
    TerminalEvent,
    TickEvent,
    Timeout,
)
from .kleene import K3, InconsistentObservation, refines
from .obligations import (
    Occurrence,
    OccurrenceStatus,
    ResponseToken,
    guard_becomes_false,
    guard_remains_unknown_at_deadline,
    guards_become_true,
    register_response_token,
    trigger_and_guard_rule,
)
from .verdict import Mode, Summary, VerdictObject


class MonitorMode(Enum):
    RUNNING = "Running"
    HALTED = "Halted"
    FAULTED = "Faulted"


class UnsupportedDischargeModeError(NotImplementedError):
    """The executable single-clause fragment does not define this mode."""


class UnsupportedEventTypeError(TypeError):
    """The executable single-clause fragment does not define this event form."""


class UnsupportedEffectReceiptError(NotImplementedError):
    """The event has an effect receipt that this fragment does not execute."""


class UnsupportedObservationBundleError(NotImplementedError):
    """The event has more than one observation entry."""


@dataclasses.dataclass(frozen=True)
class MonitorFault:
    code: str
    proposition_key: str
    prior_value: K3 | None
    attempted_value: K3 | None


class SingleClauseMonitor:
    """Execute the implemented single-`after`-clause lifecycle subset.

    `step` supports `DomainEvent`, `TickEvent`, and `TerminalEvent`, each
    carrying either no observation or a single-entry observation.
    An effect receipt, or a larger observation bundle, is rejected before
    `step` reads or changes monitor state, so a rejected event leaves the
    monitor as it was.

    `MalformedEvent` is rejected at the same point, with
    `UnsupportedEventTypeError`, because its reference type carries no event
    identifier or tick. What a
    malformed event should do to monitor state is a semantic choice the
    specification leaves open, and rejection here selects none of the options.
    """

    def __init__(self, spec: AfterClauseSpec):
        if type(spec.discharge) is not Linear:
            raise UnsupportedDischargeModeError(
                f"{type(spec.discharge).__name__} is outside the executable "
                "conformance fragment"
            )
        self.spec = spec
        self.mode = MonitorMode.RUNNING
        self.terminal_kind = None
        self.fault: MonitorFault | None = None
        self.tick = -1
        self.epistemic: dict[str, K3] = {}
        self.occurrences: dict[str, Occurrence] = {}
        self._oid_counter = 0

    # -- epistemic store (S1.2) -------------------------------------------

    def set_initial(self, key: str, value: K3) -> None:
        """Establish an epistemic fact before the first event (tick -1)."""
        if type(value) is not K3:
            raise ValueError("initial observation must be a K3 value")
        if self.mode is not MonitorMode.RUNNING or self.tick != -1:
            raise ValueError(
                "initial observations are allowed only before the first event"
            )
        prior = self.epistemic.get(key)
        if prior is not None and not refines(prior, value):
            raise ValueError("initial observation is already established incompatibly")
        self.epistemic[key] = value

    @classmethod
    def restore_open_state_card(
        cls,
        spec: AfterClauseSpec,
        *,
        initial_guard_value: K3,
        retained_events: Sequence[Event],
        current_tick: int,
        open_occurrences: Sequence[tuple[str, int, int]],
    ) -> SingleClauseMonitor:
        """Restore the declared ground-clause open state-card format."""
        if type(initial_guard_value) is not K3:
            raise ValueError("initial_guard_value must be a K3 value")
        if type(current_tick) is not int or current_tick < -1:
            raise ValueError("current_tick must be an integer no smaller than -1")
        if type(spec) is not AfterClauseSpec:
            raise ValueError("spec must be an AfterClauseSpec")
        if spec.binding_fields or type(spec.guard) is not GroundGuard:
            raise ValueError("state-card restoration requires a ground-clause spec")
        if not isinstance(retained_events, Sequence) or isinstance(
            retained_events, (str, bytes)
        ):
            raise ValueError("retained_events must be a sequence")
        if not isinstance(open_occurrences, Sequence) or isinstance(
            open_occurrences, (str, bytes)
        ):
            raise ValueError("open_occurrences must be a sequence")

        retained_event_copy = tuple(retained_events)
        prior_tick = -1
        for event in retained_event_copy:
            if type(event) not in (DomainEvent, TickEvent):
                raise ValueError("retained_events must contain running events")
            if type(event.tick) is not int or event.tick < 0:
                raise ValueError("retained_events must have natural-number ticks")
            if event.tick <= prior_tick or event.tick > current_tick:
                raise ValueError(
                    "retained_events must be strictly increasing and not exceed current_tick"
                )
            if type(event.observations) is not dict and not isinstance(
                event.observations, Mapping
            ):
                raise ValueError("retained_events must have mapping observations")
            observations = dict(event.observations)
            if len(observations) > 1:
                raise ValueError(
                    "retained_events cannot contain multi-entry observation bundles"
                )
            if any(
                type(key) is not str
                or not key
                or type(value) is not K3
                for key, value in observations.items()
            ):
                raise ValueError(
                    "retained_events must contain non-empty K3 observations"
                )
            if isinstance(event, DomainEvent) and event.effect_receipt is not None:
                raise ValueError("retained_events cannot contain effect receipts")
            prior_tick = event.tick

        occurrence_copy: list[tuple[str, int, int]] = []
        aliases: set[str] = set()
        for snapshot in open_occurrences:
            if type(snapshot) is not tuple or len(snapshot) != 3:
                raise ValueError(
                    "open_occurrences must contain (alias, trigger_tick, deadline) tuples"
                )
            alias, trigger_tick, deadline = snapshot
            if type(alias) is not str or not alias.startswith("o"):
                raise ValueError("open occurrence alias must use canonical oN form")
            suffix = alias[1:]
            if not suffix.isdigit() or int(suffix) < 1 or alias != f"o{int(suffix)}":
                raise ValueError("open occurrence alias must use canonical oN form")
            if alias in aliases:
                raise ValueError("open_occurrences must contain unique aliases")
            if type(trigger_tick) is not int or trigger_tick < 0:
                raise ValueError("trigger_tick must be a natural-number integer")
            if type(deadline) is not int or deadline < trigger_tick:
                raise ValueError("deadline must be an integer no earlier than trigger_tick")
            if trigger_tick > current_tick or deadline <= current_tick:
                raise ValueError(
                    "open occurrence must be live at current_tick within its deadline"
                )
            if deadline != trigger_tick + spec.bound:
                raise ValueError("deadline must equal trigger_tick plus the clause bound")
            aliases.add(alias)
            occurrence_copy.append((alias, trigger_tick, deadline))

        if occurrence_copy and initial_guard_value is not K3.T:
            raise ValueError("open_occurrences require an initially true guard")

        monitor = cls(spec)
        monitor.set_initial(monitor.guard_key_for({}), initial_guard_value)
        for event in retained_event_copy:
            monitor.step(event)
        if monitor.mode is not MonitorMode.RUNNING:
            raise ValueError("retained_events must preserve a running monitor")
        if occurrence_copy and monitor.guard_value_for({}) is not K3.T:
            raise ValueError("open_occurrences require a true restored guard")

        monitor.tick = current_tick
        if occurrence_copy:
            guard_key = monitor.guard_key_for({})
            monitor.occurrences = {
                alias: Occurrence(
                    oid=alias,
                    clause_id=spec.clause_id,
                    substitution={},
                    trigger_tick=trigger_tick,
                    deadline=deadline,
                    guard_key=guard_key,
                    status=OccurrenceStatus.OPEN,
                )
                for alias, trigger_tick, deadline in occurrence_copy
            }
            monitor._oid_counter = max(int(alias[1:]) for alias in aliases)
        return monitor

    def observe(self, key: str, value: K3, tick: int) -> None:
        if type(value) is not K3:
            raise ValueError("observation must be a K3 value")
        prior = self.epistemic.get(key)
        if prior is not None and not refines(prior, value):
            fault = InconsistentObservation(key, prior, value)
            self.fault = MonitorFault(
                "InconsistentObservation", key, fault.prior_value, fault.attempted_value
            )
            self.mode = MonitorMode.FAULTED
            return
        self.epistemic[key] = value

    def guard_key_for(self, substitution: Mapping[str, object]) -> str:
        projection = self._validate_exact_substitution(substitution)
        guard = self.spec.guard
        if type(guard) is GroundGuard:
            return guard.proposition_key
        if type(guard) is BoundUnaryGuard:
            return f"{guard.predicate}({projection[guard.binding_field]})"
        raise AssertionError(
            "AfterClauseSpec validation admits only known guard classes"
        )

    def guard_value_for(self, substitution: Mapping[str, object]) -> K3:
        return self.epistemic.get(self.guard_key_for(substitution), K3.U)

    # -- Monitor-local occurrence aliases ---------------------------------

    def _fresh_oid(self) -> str:
        self._oid_counter += 1
        return f"o{self._oid_counter}"

    # -- Implemented single-clause transition order ------------------------

    def step(self, event: Event) -> VerdictObject:
        if not isinstance(event, (DomainEvent, TickEvent, TerminalEvent)):
            raise UnsupportedEventTypeError(
                f"{type(event).__name__} is outside the executable conformance fragment"
            )
        if isinstance(event, TerminalEvent) and type(event.kind) not in (
            Complete,
            AgentAbort,
            ExternalCrash,
            Timeout,
            ObservationCut,
        ):
            raise UnsupportedEventTypeError(
                "terminal kind is outside the executable conformance fragment"
            )
        if isinstance(event, DomainEvent) and event.effect_receipt is not None:
            raise UnsupportedEffectReceiptError(
                "effect receipt is outside the executable reference fragment"
            )
        if not isinstance(event.observations, Mapping):
            raise ValueError("event observations must be a mapping")
        observations = dict(event.observations)
        for key, value in observations.items():
            if type(key) is not str or not key:
                raise ValueError(
                    "event observation keys must be non-empty strings"
                )
            if type(value) is not K3:
                raise ValueError("event observations must contain only K3 values")
        if len(event.observations) > 1:
            raise UnsupportedObservationBundleError(
                "observation bundles with more than one entry are outside the "
                "executable reference fragment"
            )
        if type(event.tick) is not int or event.tick < 0:
            raise ValueError(f"event tick must be a natural number, got {event.tick!r}")

        # 1. Mode check.
        if self.mode is not MonitorMode.RUNNING:
            return self._verdict(diagnostic="AfterTerminalOrFault")

        trigger_match = None
        response_match = None
        explicit_oids = frozenset()
        if isinstance(event, DomainEvent):
            trigger_match = self._project_match(event, self.spec.trigger_tag)
            response_match = self._project_match(event, self.spec.response_tag)
            if response_match is not None:
                explicit_oids = self._explicit_oids(event)

        # 2. Tick validation.
        tick = event.tick
        if tick <= self.tick:
            self.mode = MonitorMode.FAULTED
            self.fault = MonitorFault("NonMonotoneTick", "", None, None)
            return self._verdict(diagnostic="NonMonotoneTick")
        self.tick = tick

        # 3. Pre-expiry.
        self._expire(tick, strict=True)

        # 4-5. Apply the supported epistemic observation.
        for key, value in observations.items():
            self.observe(key, value, tick)
            if self.mode is MonitorMode.FAULTED:
                return self._verdict(diagnostic=self.fault.code)

        # 6. Resolve historical pending objects (conditional obligations).
        self._resolve_conditionals(tick)

        # 7-8. Instantiate new trigger occurrences; register/allocate the
        # current response token.
        if isinstance(event, DomainEvent):
            if trigger_match is not None:
                self._instantiate_trigger(event, tick, trigger_match)
            if response_match is not None:
                self._register_response(event, tick, response_match, explicit_oids)

        # 9. Evaluate instantaneous clauses: none in this single-clause
        # engine beyond the response obligation itself.

        # 10. Boundary expiry.
        self._expire(tick, strict=False)

        # 11. Terminal conversion.
        if isinstance(event, TerminalEvent):
            self._terminal_conversion(event)

        return self._verdict()

    # -- step helpers -------------------------------------------------------

    def _expire(self, tick: int, strict: bool) -> None:
        """Pre-expiry (strict=True, D < t_e) or boundary expiry
        (strict=False, D == t_e) (S1.6 steps 3 and 10)."""
        for occ in self.occurrences.values():
            expired = occ.deadline < tick if strict else occ.deadline == tick
            if not expired:
                continue
            if occ.status is OccurrenceStatus.OPEN:
                occ.status = OccurrenceStatus.BREACHED
                # The spec does not name this pre/boundary-expiry breach
                # reason distinctly from the terminal table's named reasons
                # (S1.7); "UnmetAtExpiry" is this reference's label for "the
                # deadline passed with the obligation still Open".
                occ.breach_reason = "UnmetAtExpiry"
                occ.effective_time = occ.deadline
                occ.discovery_time = tick
            elif occ.status is OccurrenceStatus.CONDITIONAL_OPEN:
                guard_remains_unknown_at_deadline(occ)

    def _resolve_conditionals(self, tick: int) -> None:
        activated = []
        for occ in self.occurrences.values():
            if occ.status not in (
                OccurrenceStatus.CONDITIONAL_OPEN,
                OccurrenceStatus.CONDITIONAL_EXPIRED,
            ):
                continue
            current = self.epistemic.get(occ.guard_key, K3.U)
            if type(current) is not K3:
                raise ValueError("epistemic store must contain only K3 values")
            if current is K3.F:
                guard_becomes_false(occ)
            elif current is K3.T:
                activated.append(occ)
        # Activation allocates once for the whole configuration, so the
        # occurrence and token orders belong to the pass, not to this caller.
        guards_become_true(activated, tick)

    def _validate_exact_substitution(
        self, substitution: Mapping[str, object]
    ) -> dict[str, str]:
        if not isinstance(substitution, Mapping):
            raise InvalidBinding("substitution must be a mapping")
        expected = set(self.spec.binding_fields)
        actual = set(substitution)
        if actual != expected:
            raise InvalidBinding(
                "substitution must contain exactly the declared binding fields"
            )
        projection: dict[str, str] = {}
        for field in self.spec.binding_fields:
            value = substitution[field]
            if type(value) is not str:
                raise InvalidBinding("bound values must be strings")
            projection[field] = value
        return projection

    def _project_match(self, event: DomainEvent, tag: str) -> dict[str, str] | None:
        if event.tag != tag:
            return None
        projection: dict[str, str] = {}
        for field in self.spec.binding_fields:
            if field not in event.fields:
                return None
            value = event.fields[field]
            if type(value) is not str:
                raise InvalidBinding("bound values must be strings")
            projection[field] = value
        return projection

    def _explicit_oids(self, event: DomainEvent) -> frozenset[str]:
        if "discharges" not in event.fields:
            return frozenset()
        raw = event.fields["discharges"]
        if not isinstance(raw, (tuple, list)):
            raise InvalidBinding(
                "discharges must be a tuple or list of non-empty strings"
            )
        if any(type(oid) is not str or not oid for oid in raw):
            raise InvalidBinding("discharges must contain only non-empty strings")
        return frozenset(raw)

    def _instantiate_trigger(
        self, event: DomainEvent, tick: int, substitution: dict[str, str]
    ) -> None:
        oid = self._fresh_oid()
        guard_key = self.guard_key_for(substitution)
        guard = self.epistemic.get(guard_key, K3.U)
        deadline = tick + self.spec.bound
        occ = trigger_and_guard_rule(
            oid,
            self.spec.clause_id,
            dict(substitution),
            tick,
            deadline,
            guard_key,
            guard,
        )
        self.occurrences[oid] = occ

    def _register_response(
        self,
        event: DomainEvent,
        tick: int,
        substitution: dict[str, str],
        explicit_oids: frozenset[str],
    ) -> None:
        token = ResponseToken(
            rid=event.event_id,
            response_tick=tick,
            substitution=dict(substitution),
            explicit_oids=explicit_oids,
        )
        register_response_token(token, self.occurrences)

    def _terminal_conversion(self, event) -> None:
        kind = event.kind
        self.mode = MonitorMode.HALTED
        self.terminal_kind = kind
        for occ in self.occurrences.values():
            if occ.status is OccurrenceStatus.OPEN:
                status, reason = _convert_open(
                    kind, self.spec.on_agent_abort, self.epistemic
                )
                occ.status = status
                occ.breach_reason = reason
                if status is OccurrenceStatus.BREACHED:
                    occ.effective_time = occ.deadline
                    occ.discovery_time = event.tick
            elif occ.status in (
                OccurrenceStatus.CONDITIONAL_OPEN,
                OccurrenceStatus.CONDITIONAL_EXPIRED,
            ):
                status, reason = _convert_conditional(
                    kind, self.spec.on_agent_abort, self.epistemic
                )
                occ.status = status
                occ.breach_reason = reason

    # -- S1.9 aggregate three-valued verdict, S1.3 public verdict object --

    def _public_mode(self) -> Mode:
        if self.mode is MonitorMode.FAULTED:
            return Mode.FAULTED
        if self.mode is MonitorMode.RUNNING:
            return Mode.RUNNING
        mapping = {
            Complete: Mode.COMPLETE,
            AgentAbort: Mode.AGENT_ABORT,
            ExternalCrash: Mode.EXTERNAL_CRASH,
            Timeout: Mode.TIMEOUT,
            ObservationCut: Mode.OBSERVATION_CUT,
        }
        return mapping[type(self.terminal_kind)]

    def _aggregate_summary(self) -> Summary:
        # S1.9, in the spec's own order:
        if any(
            occ.status is OccurrenceStatus.BREACHED for occ in self.occurrences.values()
        ):
            return Summary.VIOLATED
        if self.mode is MonitorMode.FAULTED:
            return Summary.UNKNOWN
        if self.mode is MonitorMode.RUNNING:
            return Summary.UNKNOWN
        if isinstance(self.terminal_kind, ObservationCut):
            return Summary.UNKNOWN
        settled = (
            OccurrenceStatus.DISCHARGED,
            OccurrenceStatus.INAPPLICABLE,
            OccurrenceStatus.WAIVED,
        )
        if all(occ.status in settled for occ in self.occurrences.values()):
            return Summary.SATISFIED
        return Summary.UNKNOWN

    def _verdict(self, diagnostic: str | None = None) -> VerdictObject:
        return VerdictObject(
            summary=self._aggregate_summary(),
            mode=self._public_mode(),
            occurrences={
                oid: occ.status.value for oid, occ in self.occurrences.items()
            },
            diagnostic=diagnostic or (self.fault.code if self.fault else None),
        )

    def current_verdict(self) -> VerdictObject:
        return self._verdict()

    # -- S1.8 prospective fail-closed permission --------------------------

    def permit_after_fault(self):
        """The one S1.8 case the demonstrated-failure test (S1.13) needs:
        once the monitor is Faulted, every action is denied for lack of
        information."""
        from .verdict import DenyUnknown

        if self.mode is MonitorMode.FAULTED:
            return DenyUnknown("MonitorSemanticFault")
        raise NotImplementedError(
            "The general S1.8 Permit_C(q,a) rule requires evaluating every "
            "applicable `before` requirement, authority predicate, and "
            "immediate forbid/flow/budget/invariant check across the whole "
            "contract. This reference engine compiles only the single "
            "AfterClauseSpec shape (see contracts.py), so it cannot evaluate "
            "Permit_C generically."
        )


def _convert_open(
    kind, disposition: AbortDisposition, epistemic: dict
) -> tuple[OccurrenceStatus, str]:
    """S1.7's "Open active obligations" table."""
    if isinstance(kind, Complete):
        return OccurrenceStatus.BREACHED, "UnmetAtComplete"
    if isinstance(kind, AgentAbort):
        return _convert_agent_abort_open(disposition, epistemic)
    if isinstance(kind, ExternalCrash):
        return OccurrenceStatus.UNKNOWN_FINAL, "ExternalInterruption"
    if isinstance(kind, Timeout):
        return OccurrenceStatus.UNKNOWN_FINAL, "TruncatedBeforeDeadline"
    if isinstance(kind, ObservationCut):
        return OccurrenceStatus.UNKNOWN_FINAL, "ObservationTruncated"
    raise ValueError(f"unrecognized terminal kind: {kind!r}")


def _convert_agent_abort_open(
    disposition: AbortDisposition, epistemic: dict
) -> tuple[OccurrenceStatus, str]:
    if isinstance(disposition, Breach):
        return OccurrenceStatus.BREACHED, "AbortWithOpenObligation"
    if isinstance(disposition, Indeterminate):
        return OccurrenceStatus.UNKNOWN_FINAL, "AbortDispositionIndeterminate"
    if isinstance(disposition, WaiveIf):
        authority = epistemic.get(disposition.authority_key, K3.U)
        if authority is K3.T:
            return OccurrenceStatus.WAIVED, "AuthorizedAbort"
        if authority is K3.F:
            return OccurrenceStatus.BREACHED, "UnauthorizedAbort"
        return OccurrenceStatus.UNKNOWN_FINAL, "AbortAuthorityUnknown"
    raise ValueError(f"unrecognized abort disposition: {disposition!r}")


def _convert_conditional(
    kind, disposition: AbortDisposition, epistemic: dict
) -> tuple[OccurrenceStatus, str]:
    """S1.7's "Conditional-unresolved obligations" table. No terminal
    conversion may turn a still-unknown guard into a proved breach."""
    if isinstance(kind, Complete):
        return OccurrenceStatus.UNKNOWN_FINAL, "GuardUnknownAtComplete"
    if isinstance(kind, AgentAbort):
        if (
            isinstance(disposition, WaiveIf)
            and epistemic.get(disposition.authority_key, K3.U) is K3.T
        ):
            return OccurrenceStatus.WAIVED, "AuthorizedAbortOfPotentialObligation"
        return OccurrenceStatus.UNKNOWN_FINAL, "GuardUnknownAtAgentAbort"
    if isinstance(kind, ExternalCrash):
        return OccurrenceStatus.UNKNOWN_FINAL, "GuardUnknownAtExternalCrash"
    if isinstance(kind, Timeout):
        return OccurrenceStatus.UNKNOWN_FINAL, "GuardUnknownAtTimeout"
    if isinstance(kind, ObservationCut):
        return OccurrenceStatus.UNKNOWN_FINAL, "GuardUnknownAtObservationCut"
    raise ValueError(f"unrecognized terminal kind: {kind!r}")
