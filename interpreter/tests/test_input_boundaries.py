"""Public input boundaries for the executable reference fragment."""

from __future__ import annotations

import copy
import inspect

import pytest

import rs_metalang_ref.monitor as monitor_module
from rs_metalang_ref import obligations
from rs_metalang_ref.contracts import (
    AfterClauseSpec,
    BoundUnaryGuard,
    GroundGuard,
    Linear,
)
from rs_metalang_ref.events import Complete, DomainEvent, TerminalEvent, TickEvent
from rs_metalang_ref.kleene import K3, refines
from rs_metalang_ref.monitor import (
    MonitorMode,
    SingleClauseMonitor,
    UnsupportedEventTypeError,
)
from rs_metalang_ref.obligations import OccurrenceStatus
from rs_metalang_ref.residual import (
    InvalidComparisonRequest,
    NoWitnessWithinBound,
    UnitDataUnitAdvanceBound,
    bounded_compare,
)
from rs_metalang_ref.robust import AmbiguitySet
from rs_metalang_ref.verdict import Mode, Summary, VerdictObject


def _spec() -> AfterClauseSpec:
    return AfterClauseSpec(
        clause_id="c1",
        trigger_tag="export",
        response_tag="approval",
        binding_fields=("x",),
        guard=BoundUnaryGuard("Sensitive", "x"),
        bound=1,
        discharge=Linear(),
    )


def _ground_spec() -> AfterClauseSpec:
    return AfterClauseSpec(
        clause_id="ground-c1",
        trigger_tag="export",
        response_tag="approval",
        binding_fields=(),
        guard=GroundGuard("approval-required"),
        bound=2,
        discharge=Linear(),
    )


def _restore_open_state_card(**overrides) -> SingleClauseMonitor:
    restore = getattr(SingleClauseMonitor, "restore_open_state_card", None)
    assert callable(restore), (
        "SingleClauseMonitor must own restore_open_state_card"
    )
    arguments = {
        "initial_guard_value": K3.T,
        "retained_events": (TickEvent(event_id="retained-0", tick=0),),
        "current_tick": 3,
        "open_occurrences": (("o2", 3, 5),),
    }
    arguments.update(overrides)
    return restore(_ground_spec(), **arguments)


def _monitor_with_open_obligation() -> SingleClauseMonitor:
    monitor = SingleClauseMonitor(_spec())
    monitor.set_initial("Sensitive(f)", K3.U)
    monitor.step(
        DomainEvent(
            event_id="trigger",
            tick=0,
            tag="export",
            fields={"x": "f"},
        )
    )
    return monitor


def _public_monitor_state(monitor: SingleClauseMonitor):
    return (
        monitor.tick,
        monitor.mode,
        copy.deepcopy(monitor.occurrences),
        dict(monitor.epistemic),
        monitor.current_verdict(),
    )


def _admission_error_type(name: str) -> type[BaseException]:
    candidate = getattr(monitor_module, name, NotImplementedError)
    return candidate if isinstance(candidate, type) else NotImplementedError


def test_restore_open_state_card_returns_a_complete_running_monitor():
    retained_events = [TickEvent(event_id="retained-0", tick=0)]
    open_occurrences = [("o2", 3, 5)]

    monitor = _restore_open_state_card(
        retained_events=retained_events,
        open_occurrences=open_occurrences,
    )

    assert type(monitor) is SingleClauseMonitor
    assert monitor.spec == _ground_spec()
    assert monitor.tick == 3
    assert monitor.mode is MonitorMode.RUNNING
    assert monitor.guard_value_for({}) is K3.T
    assert tuple(monitor.occurrences) == ("o2",)
    restored = monitor.occurrences["o2"]
    assert restored.oid == "o2"
    assert restored.clause_id == "ground-c1"
    assert restored.substitution == {}
    assert restored.trigger_tick == 3
    assert restored.deadline == 5
    assert restored.guard_key == "approval-required"
    assert restored.status is OccurrenceStatus.OPEN
    assert restored.candidates == []
    assert restored.response is None
    assert restored.breach_reason is None
    assert restored.effective_time is None
    assert restored.discovery_time is None

    retained_events.append(TickEvent(event_id="outside-1", tick=1))
    open_occurrences.append(("o9", 3, 5))
    assert tuple(monitor.occurrences) == ("o2",)


def test_restore_open_state_card_accepts_the_initial_tick_for_an_empty_card():
    monitor = _restore_open_state_card(
        retained_events=(),
        current_tick=-1,
        open_occurrences=(),
    )

    assert monitor.tick == -1
    assert monitor.mode is MonitorMode.RUNNING
    assert monitor.occurrences == {}
    assert monitor.guard_value_for({}) is K3.T


def test_restore_open_state_card_preserves_subsequent_monitor_behavior():
    monitor = _restore_open_state_card()

    monitor.step(DomainEvent(event_id="approval-4", tick=4, tag="approval"))
    monitor.step(DomainEvent(event_id="export-5", tick=5, tag="export"))

    assert monitor.tick == 5
    assert tuple(monitor.occurrences) == ("o2", "o3")
    assert monitor.occurrences["o2"].status is OccurrenceStatus.DISCHARGED
    assert monitor.occurrences["o3"].status is OccurrenceStatus.OPEN


@pytest.mark.parametrize(
    ("field", "overrides"),
    [
        pytest.param(
            "initial_guard_value",
            {"initial_guard_value": "T"},
            id="guard-outside-k3",
        ),
        pytest.param(
            "retained_events",
            {"retained_events": object()},
            id="retained-events-not-a-sequence",
        ),
        pytest.param(
            "retained_events",
            {"retained_events": (object(),)},
            id="retained-event-outside-event-grammar",
        ),
        pytest.param(
            "current_tick",
            {"current_tick": "3"},
            id="tick-string",
        ),
        pytest.param(
            "current_tick",
            {"current_tick": 3.0},
            id="tick-float",
        ),
        pytest.param(
            "current_tick",
            {"current_tick": True},
            id="tick-boolean",
        ),
        pytest.param(
            "current_tick",
            {"current_tick": -2},
            id="tick-below-initial",
        ),
        pytest.param(
            "open_occurrences",
            {"open_occurrences": object()},
            id="open-occurrences-not-a-sequence",
        ),
    ],
)
def test_restore_open_state_card_rejects_invalid_top_level_values(
    field,
    overrides,
):
    with pytest.raises(ValueError, match=field):
        _restore_open_state_card(**overrides)


@pytest.mark.parametrize(
    "snapshot",
    [
        pytest.param(("o1", 3), id="missing-deadline"),
        pytest.param(["o1", 3, 5], id="record-not-a-tuple"),
        pytest.param(("o1", True, 5), id="boolean-trigger-tick"),
        pytest.param(("o1", 3, 5.0), id="float-deadline"),
        pytest.param(("o1", -1, 5), id="negative-trigger-tick"),
        pytest.param(("o1", 3, 2), id="deadline-before-trigger"),
        pytest.param(("o1", 4, 6), id="trigger-after-current-tick"),
        pytest.param(("o1", 0, 2), id="expired-open-occurrence"),
    ],
)
def test_restore_open_state_card_rejects_malformed_open_occurrences(snapshot):
    with pytest.raises(
        ValueError,
        match="open_occurrences|trigger_tick|deadline",
    ):
        _restore_open_state_card(open_occurrences=(snapshot,))


@pytest.mark.parametrize(
    "alias",
    [
        pytest.param("", id="empty"),
        pytest.param("1", id="missing-prefix"),
        pytest.param("o0", id="zero"),
        pytest.param("o01", id="leading-zero"),
        pytest.param("o-1", id="negative"),
        pytest.param(1, id="non-string"),
    ],
)
def test_restore_open_state_card_rejects_noncanonical_aliases(alias):
    with pytest.raises(ValueError, match="alias|canonical"):
        _restore_open_state_card(open_occurrences=((alias, 3, 5),))


def test_restore_open_state_card_rejects_duplicate_aliases():
    with pytest.raises(ValueError, match="duplicate|unique"):
        _restore_open_state_card(
            open_occurrences=(
                ("o2", 3, 5),
                ("o2", 3, 5),
            )
        )


@pytest.mark.parametrize(
    "retained_events",
    [
        pytest.param(
            (TickEvent(event_id="future-4", tick=4),),
            id="event-after-current-tick",
        ),
        pytest.param(
            (
                TickEvent(event_id="first-1", tick=1),
                TickEvent(event_id="repeat-1", tick=1),
            ),
            id="nonmonotone-prefix",
        ),
        pytest.param(
            (TerminalEvent(event_id="complete-0", tick=0, kind=Complete()),),
            id="terminal-prefix",
        ),
    ],
)
def test_restore_open_state_card_rejects_an_invalid_retained_prefix(
    retained_events,
):
    with pytest.raises(ValueError, match="retained_events|current_tick|running"):
        _restore_open_state_card(retained_events=retained_events)


def test_restore_open_state_card_validates_before_monitor_construction():
    class ConstructionProbe(SingleClauseMonitor):
        construction_count = 0

        def __init__(self, spec):
            type(self).construction_count += 1
            super().__init__(spec)

    restore = getattr(ConstructionProbe, "restore_open_state_card", None)
    assert callable(restore), (
        "SingleClauseMonitor must own restore_open_state_card"
    )

    with pytest.raises(ValueError, match="alias|canonical"):
        restore(
            _ground_spec(),
            initial_guard_value=K3.T,
            retained_events=(),
            current_tick=3,
            open_occurrences=(
                ("o2", 3, 5),
                ("o01", 3, 5),
            ),
        )

    assert ConstructionProbe.construction_count == 0


def test_restore_open_state_card_constructs_the_bound_monitor_class():
    class MonitorSubclass(SingleClauseMonitor):
        pass

    restore = getattr(MonitorSubclass, "restore_open_state_card", None)
    assert callable(restore), (
        "SingleClauseMonitor must own restore_open_state_card"
    )

    monitor = restore(
        _ground_spec(),
        initial_guard_value=K3.T,
        retained_events=(),
        current_tick=3,
        open_occurrences=(("o2", 3, 5),),
    )

    assert type(monitor) is MonitorSubclass


@pytest.mark.parametrize(
    ("prior", "updated"),
    [
        pytest.param(K3.U, "T", id="updated-outside-k3"),
        pytest.param("U", K3.T, id="prior-outside-k3"),
    ],
)
def test_refines_refuses_values_outside_k3(prior, updated):
    with pytest.raises(ValueError, match="K3"):
        refines(prior, updated)


def test_refines_keeps_the_declared_k3_information_order():
    assert refines(K3.U, K3.T) is True
    assert refines(K3.U, K3.F) is True
    assert refines(K3.T, K3.T) is True
    assert refines(K3.T, K3.F) is False


def test_set_initial_refuses_non_k3_before_store_mutation():
    monitor = SingleClauseMonitor(_spec())

    with pytest.raises(ValueError, match="K3"):
        monitor.set_initial("Sensitive(f)", "not-k3")

    assert monitor.epistemic == {}
    assert monitor.mode is MonitorMode.RUNNING


@pytest.mark.parametrize(
    ("summary", "mode"),
    [
        pytest.param("Satisfied", Mode.COMPLETE, id="summary-string"),
        pytest.param(Summary.SATISFIED, "Complete", id="mode-string"),
    ],
)
def test_verdict_object_requires_exact_summary_and_mode_values(summary, mode):
    with pytest.raises(ValueError):
        VerdictObject(summary=summary, mode=mode)


def test_set_initial_rejects_a_contradictory_second_value_without_mutation():
    monitor = SingleClauseMonitor(_spec())
    monitor.set_initial("Sensitive(f)", K3.T)
    before = _public_monitor_state(monitor)

    with pytest.raises(ValueError, match="already established"):
        monitor.set_initial("Sensitive(f)", K3.F)

    assert _public_monitor_state(monitor) == before


def test_set_initial_rejects_a_post_event_call_without_mutation():
    monitor = SingleClauseMonitor(_spec())
    monitor.set_initial("Sensitive(f)", K3.T)
    monitor.step(TickEvent(event_id="tick-0", tick=0))
    before = _public_monitor_state(monitor)

    with pytest.raises(ValueError, match="only before the first event"):
        monitor.set_initial("Sensitive(f)", K3.F)

    assert _public_monitor_state(monitor) == before


def test_set_initial_rejects_a_post_terminal_call_without_mutation():
    monitor = SingleClauseMonitor(_spec())
    monitor.set_initial("Sensitive(f)", K3.T)
    monitor.step(TerminalEvent(event_id="complete-0", tick=0, kind=Complete()))
    before = _public_monitor_state(monitor)

    with pytest.raises(ValueError, match="only before the first event"):
        monitor.set_initial("Sensitive(f)", K3.F)

    assert _public_monitor_state(monitor) == before


def test_observe_refuses_non_k3_before_store_mutation():
    monitor = SingleClauseMonitor(_spec())
    monitor.set_initial("Sensitive(f)", K3.U)

    with pytest.raises(ValueError, match="K3"):
        monitor.observe("Sensitive(f)", "not-k3", tick=0)

    assert monitor.epistemic == {"Sensitive(f)": K3.U}
    assert monitor.mode is MonitorMode.RUNNING
    assert monitor.fault is None


def test_step_refuses_non_k3_observation_before_state_mutation():
    monitor = SingleClauseMonitor(_spec())
    monitor.set_initial("Sensitive(f)", K3.U)
    monitor.step(
        DomainEvent(
            event_id="trigger",
            tick=0,
            tag="export",
            fields={"x": "f"},
        )
    )
    assert monitor.occurrences["o1"].status is OccurrenceStatus.CONDITIONAL_OPEN
    before_epistemic = dict(monitor.epistemic)
    before_occurrences = copy.deepcopy(monitor.occurrences)
    before_verdict = monitor.current_verdict()

    with pytest.raises(ValueError, match="K3"):
        monitor.step(
            TickEvent(
                event_id="deadline",
                tick=1,
                observations={"Sensitive(f)": "not-k3"},
            )
        )

    assert monitor.tick == 0
    assert monitor.epistemic == before_epistemic
    assert monitor.occurrences == before_occurrences
    assert monitor.current_verdict() == before_verdict
    assert monitor.mode is MonitorMode.RUNNING


@pytest.mark.parametrize(
    "invalid_observations",
    [
        pytest.param([("Sensitive(f)", K3.T)], id="list-of-pairs"),
        pytest.param({"": K3.T}, id="empty-key"),
        pytest.param({1: K3.T}, id="non-string-key"),
    ],
)
def test_step_requires_a_mapping_with_nonempty_string_observation_keys(
    invalid_observations,
):
    monitor = SingleClauseMonitor(_spec())
    monitor.set_initial("Sensitive(f)", K3.U)
    before = _public_monitor_state(monitor)

    event = TickEvent(
        event_id="tick-0",
        tick=0,
        observations=invalid_observations,
    )
    with pytest.raises(ValueError, match="observations|observation keys"):
        monitor.step(event)

    assert _public_monitor_state(monitor) == before


def test_step_rejects_an_invalid_terminal_kind_before_state_mutation():
    monitor = _monitor_with_open_obligation()
    before = _public_monitor_state(monitor)

    event = TerminalEvent(event_id="invalid-terminal", tick=1, kind=object())
    with pytest.raises(UnsupportedEventTypeError, match="terminal kind"):
        monitor.step(event)

    assert _public_monitor_state(monitor) == before


def test_step_refuses_an_event_outside_the_supported_union_before_state_mutation():
    class DuckEvent:
        def __init__(self):
            self.tick = 0
            self.observations = {}

    monitor = SingleClauseMonitor(_spec())

    with pytest.raises(UnsupportedEventTypeError, match="DuckEvent"):
        monitor.step(DuckEvent())

    assert monitor.tick == -1
    assert monitor.epistemic == {}
    assert monitor.occurrences == {}
    assert monitor.mode is MonitorMode.RUNNING


@pytest.mark.parametrize(
    "invalid_tick",
    [
        pytest.param(True, id="bool"),
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="float"),
        pytest.param("1", id="string"),
    ],
)
def test_step_refuses_non_natural_tick_before_state_mutation(invalid_tick):
    monitor = SingleClauseMonitor(_spec())

    with pytest.raises(ValueError, match="tick"):
        monitor.step(TickEvent(event_id="e0", tick=invalid_tick))

    assert monitor.tick == -1
    assert monitor.epistemic == {}
    assert monitor.occurrences == {}
    assert monitor.mode is MonitorMode.RUNNING


def test_step_accepts_a_supported_event_with_a_natural_tick_and_k3_observation():
    monitor = SingleClauseMonitor(_spec())

    verdict = monitor.step(
        TickEvent(
            event_id="e0",
            tick=0,
            observations={"Sensitive(f)": K3.T},
        )
    )

    assert monitor.tick == 0
    assert monitor.epistemic == {"Sensitive(f)": K3.T}
    assert monitor.mode is MonitorMode.RUNNING
    assert verdict == monitor.current_verdict()


@pytest.mark.parametrize(
    "error_name",
    [
        pytest.param("UnsupportedEffectReceiptError", id="effect-receipt"),
        pytest.param("UnsupportedObservationBundleError", id="observation-bundle"),
    ],
)
def test_public_admission_errors_are_named_notimplemented_errors(error_name):
    error_type = getattr(monitor_module, error_name, None)

    assert isinstance(error_type, type), f"{error_name} must be a public error class"
    assert issubclass(error_type, NotImplementedError)


@pytest.mark.parametrize(
    "effect_receipt",
    [
        pytest.param(False, id="false"),
        pytest.param(0, id="zero"),
        pytest.param("", id="empty-string"),
        pytest.param({}, id="empty-mapping"),
    ],
)
def test_step_rejects_any_non_none_effect_receipt_before_state_mutation(
    effect_receipt,
):
    monitor = _monitor_with_open_obligation()
    before = _public_monitor_state(monitor)

    assert effect_receipt is not None
    with pytest.raises(_admission_error_type("UnsupportedEffectReceiptError")):
        monitor.step(
            DomainEvent(
                event_id="unsupported-receipt",
                tick=1,
                tag="export",
                fields={"x": "f"},
                effect_receipt=effect_receipt,
            )
        )

    assert _public_monitor_state(monitor) == before


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(
            DomainEvent(
                event_id="unsupported-domain-bundle",
                tick=1,
                tag="export",
                fields={"x": "f"},
                observations={"A": K3.T, "B": K3.F},
            ),
            id="domain-event",
        ),
        pytest.param(
            TickEvent(
                event_id="unsupported-tick-bundle",
                tick=1,
                observations={"A": K3.T, "B": K3.F},
            ),
            id="tick-event",
        ),
        pytest.param(
            TerminalEvent(
                event_id="unsupported-terminal-bundle",
                tick=1,
                kind=Complete(),
                observations={"A": K3.T, "B": K3.F},
            ),
            id="terminal-event",
        ),
    ],
)
def test_step_rejects_multi_entry_observation_bundles_before_state_mutation(event):
    monitor = _monitor_with_open_obligation()
    before = _public_monitor_state(monitor)

    assert len(event.observations) > 1
    with pytest.raises(_admission_error_type("UnsupportedObservationBundleError")):
        monitor.step(event)

    assert _public_monitor_state(monitor) == before


@pytest.mark.parametrize(
    ("rejected_event", "error_name"),
    [
        pytest.param(
            DomainEvent(
                event_id="rejected-receipt",
                tick=1,
                tag="export",
                fields={"x": "f"},
                effect_receipt=False,
            ),
            "UnsupportedEffectReceiptError",
            id="effect-receipt",
        ),
        pytest.param(
            DomainEvent(
                event_id="rejected-observation-bundle",
                tick=1,
                tag="export",
                fields={"x": "f"},
                observations={"A": K3.T, "B": K3.F},
            ),
            "UnsupportedObservationBundleError",
            id="observation-bundle",
        ),
    ],
)
def test_valid_step_after_rejection_matches_a_fresh_run(rejected_event, error_name):
    monitor = _monitor_with_open_obligation()

    with pytest.raises(_admission_error_type(error_name)):
        monitor.step(rejected_event)

    accepted_event = DomainEvent(
        event_id="accepted-after-rejection",
        tick=1,
        tag="export",
        fields={"x": "f"},
        effect_receipt=None,
        observations={"Sensitive(f)": K3.T},
    )
    actual_verdict = monitor.step(accepted_event)

    fresh_monitor = _monitor_with_open_obligation()
    expected_verdict = fresh_monitor.step(accepted_event)

    assert actual_verdict == expected_verdict
    assert _public_monitor_state(monitor) == _public_monitor_state(fresh_monitor)


def test_supported_none_receipt_and_single_observation_remain_normal():
    monitor = SingleClauseMonitor(_spec())
    monitor.set_initial("Sensitive(f)", K3.U)

    verdict = monitor.step(
        DomainEvent(
            event_id="supported-receipt-none",
            tick=0,
            tag="export",
            fields={"x": "f"},
            effect_receipt=None,
            observations={"Sensitive(f)": K3.T},
        )
    )

    assert monitor.tick == 0
    assert monitor.mode is MonitorMode.RUNNING
    assert monitor.epistemic == {"Sensitive(f)": K3.T}
    assert set(monitor.occurrences) == {"o1"}
    assert verdict == monitor.current_verdict()


def test_single_inconsistent_observation_remains_a_semantic_fault():
    monitor = SingleClauseMonitor(_spec())
    monitor.set_initial("Sensitive(f)", K3.T)

    monitor.step(
        TickEvent(
            event_id="inconsistent-single-observation",
            tick=0,
            observations={"Sensitive(f)": K3.F},
        )
    )

    assert monitor.mode is MonitorMode.FAULTED
    assert monitor.fault is not None
    assert monitor.fault.code == "InconsistentObservation"


def test_reused_event_id_has_distinct_monitor_local_canonical_aliases():
    monitor = SingleClauseMonitor(_spec())
    monitor.set_initial("Sensitive(f)", K3.T)

    for tick in (4, 9):
        monitor.step(DomainEvent("e7", tick=tick, tag="export", fields={"x": "f"}))

    assert tuple(monitor.occurrences) == ("o1", "o2")
    assert [monitor.occurrences[alias].trigger_tick for alias in ("o1", "o2")] == [4, 9]
    assert ("c1", ("e7", 4), 0) not in monitor.occurrences
    assert set(monitor.current_verdict().occurrences) == {"o1", "o2"}


def test_monitor_docstring_describes_the_supported_fragment():
    monitor_doc = (inspect.getdoc(SingleClauseMonitor) or "").lower()

    assert "twelve-step total transition" not in monitor_doc
    for expected in ("effect receipt", "single-entry observation", "reject"):
        assert expected in monitor_doc


def test_public_occurrence_docs_describe_the_alias_boundary():
    obligations_doc = (inspect.getdoc(obligations) or "").lower()
    verdict_doc = (inspect.getdoc(VerdictObject) or "").lower()

    assert "fresh oid per trigger match" not in obligations_doc
    assert "trace-local canonical alias" in obligations_doc
    assert "o1" in obligations_doc
    assert "(clause_id, (event_id, tick), match_ordinal)" in obligations_doc
    assert "outside this fragment" in obligations_doc
    assert "trace-local canonical alias" in verdict_doc


def test_linear_only_obligations_module_has_no_broadcast_registration_surface():
    assert not hasattr(obligations, "register_broadcast_token")


def _stateful_residual(monitor: SingleClauseMonitor):
    def residual(word):
        return monitor.step(
            TickEvent(
                event_id=f"comparison-{len(word)}",
                tick=len(word),
            )
        )

    return residual


def test_bounded_compare_refuses_omitted_scope_before_search():
    left = SingleClauseMonitor(_spec())
    right = SingleClauseMonitor(_spec())

    with pytest.raises(TypeError, match="contract_environment"):
        bounded_compare(
            _stateful_residual(left),
            _stateful_residual(right),
            alphabet=("a",),
            bound=UnitDataUnitAdvanceBound(continuation_length=0),
        )

    assert left.tick == -1
    assert right.tick == -1


@pytest.mark.parametrize(
    "scope_override",
    [
        pytest.param({"contract_environment": "undeclared"}, id="contract-environment"),
        pytest.param({"observation_projection": ""}, id="observation-projection"),
        pytest.param({"continuation_family": ""}, id="continuation-family"),
    ],
)
def test_bounded_compare_refuses_undeclared_scope_before_search(scope_override):
    scope = {
        "contract_environment": "C",
        "observation_projection": VerdictObject.__name__,
        "continuation_family": "all-words-through-declared-bound",
    }
    scope.update(scope_override)
    left = SingleClauseMonitor(_spec())
    right = SingleClauseMonitor(_spec())

    invalid_field = next(
        field for field, value in scope_override.items() if value in {"", "undeclared"}
    )
    with pytest.raises(InvalidComparisonRequest, match=invalid_field):
        bounded_compare(
            _stateful_residual(left),
            _stateful_residual(right),
            alphabet=("a",),
            bound=UnitDataUnitAdvanceBound(continuation_length=0),
            **scope,
        )

    assert left.tick == -1
    assert right.tick == -1


def test_bounded_compare_accepts_a_declared_scope():
    left = SingleClauseMonitor(_spec())
    right = SingleClauseMonitor(_spec())

    result = bounded_compare(
        _stateful_residual(left),
        _stateful_residual(right),
        alphabet=("a",),
        bound=UnitDataUnitAdvanceBound(continuation_length=0),
        contract_environment="C",
        observation_projection=VerdictObject.__name__,
        continuation_family="all-words-through-declared-bound",
    )

    assert isinstance(result, NoWitnessWithinBound)
    assert result.scope.contract_environment == "C"
    assert result.scope.observation_projection == VerdictObject.__name__
    assert result.scope.continuation_family == "all-words-through-declared-bound"
    assert left.tick == 0
    assert right.tick == 0


@pytest.mark.parametrize(
    "invalid_nonempty",
    [
        pytest.param("false", id="string"),
        pytest.param(1, id="integer"),
    ],
)
def test_ambiguity_set_refuses_non_boolean_nonempty(invalid_nonempty):
    with pytest.raises(ValueError, match="nonempty.*bool"):
        AmbiguitySet(
            core_bad_mass=0.0,
            gamma_min=0.0,
            gamma_max=0.0,
            outside_model_risk=0.0,
            nonempty=invalid_nonempty,
        )
