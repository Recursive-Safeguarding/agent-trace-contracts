"""The capsule uses one fixed profile for one small setting.

Python uses Profile to store the values that run the capsule: a contract
environment, continuation alphabet, observation projection, retained
environment manifest, continuation interface, and comparison bound. It does not
store every deployment-profile component from the specification. The
continuation interface has four operations: configuration type, initialise,
step, and observe.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import inspect
import os
from pathlib import Path

import pytest
import rs_capsule.profile as profile_module
from rs_capsule.capsule import SOURCE_TRACE, build_source_prefix
from rs_capsule.profile import OccurrenceRecord, Profile, StateCardTerm, build_profile
from rs_metalang_ref import residual as residual_module
from rs_metalang_ref.contracts import AfterClauseSpec, Linear
from rs_metalang_ref.events import TickEvent
from rs_metalang_ref.kleene import K3
from rs_metalang_ref.monitor import SingleClauseMonitor

# The default terminal labels used by TotalizedLTS.
TERMINAL_LABELS = frozenset(
    {"Complete", "AgentAbort", "ExternalCrash", "Timeout", "ObservationCut"}
)

# The public Profile fields used by run_capsule.
PROFILE_FIELDS = (
    "profile_id",
    "contract_environment",
    "continuation_alphabet",
    "retained_environment_manifest",
    "continuation_interface",
    "comparison_bound",
)
REMOVED_PROFILE_BOUND_FIELDS = (
    "continuation_length_bound",
    "data_domain",
    "timing_bound",
)

# the profile-relative operational-adequacy rule, the I_P declaration above.
INTERFACE_COMPONENTS = ("configuration_type", "initialise", "step", "observe")

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SIBLING_INTERPRETER_RESIDUAL = (
    PACKAGE_ROOT / "interpreter" / "src" / "rs_metalang_ref" / "residual.py"
).resolve()
NORMATIVE_TOTAL_OBSERVATION_FIELDS = (
    "contract_observation",
    "continuation_status",
)


def test_public_unit_bound_derives_the_complete_comparison_bound():
    unit_data_domain_type = getattr(residual_module, "UnitDataDomain", None)
    comparison_bound_type = getattr(residual_module, "UnitDataUnitAdvanceBound", None)

    assert unit_data_domain_type is not None, (
        "rs_metalang_ref.residual must expose UnitDataDomain"
    )
    assert comparison_bound_type is not None, (
        "rs_metalang_ref.residual must expose UnitDataUnitAdvanceBound"
    )

    comparison_bound = comparison_bound_type(continuation_length=2)

    assert dataclasses.is_dataclass(comparison_bound)
    assert type(comparison_bound).__dataclass_params__.frozen
    assert dataclasses.is_dataclass(comparison_bound.data_domain)
    assert type(comparison_bound.data_domain).__dataclass_params__.frozen
    assert isinstance(comparison_bound.data_domain, unit_data_domain_type)
    assert dataclasses.asdict(comparison_bound) == {
        "continuation_length": 2,
        "data_domain": {
            "kind": "unit",
            "note": "contract declares no binding fields",
        },
        "timing_bound": 2,
    }


def test_capsule_imports_the_live_total_observation_contract():
    if os.environ.get("PYTHONPATH"):
        raise AssertionError(
            "The documented capsule command must not use PYTHONPATH to select the interpreter"
        )

    residual = importlib.import_module("rs_metalang_ref.residual")

    imported_source = Path(residual.__file__).resolve()
    assert imported_source == SIBLING_INTERPRETER_RESIDUAL, (
        "The capsule must import the declared live interpreter source; "
        f"required {SIBLING_INTERPRETER_RESIDUAL}, imported {imported_source}"
    )

    assert dataclasses.is_dataclass(residual.TotalObservation)
    public_fields = tuple(
        field.name for field in dataclasses.fields(residual.TotalObservation)
    )
    assert public_fields == NORMATIVE_TOTAL_OBSERVATION_FIELDS


def test_profile_exposes_only_fields_used_by_run_capsule():
    profile = build_profile()
    comparison_bound_type = getattr(residual_module, "UnitDataUnitAdvanceBound", None)

    assert isinstance(profile, Profile)
    assert tuple(field.name for field in dataclasses.fields(Profile)) == PROFILE_FIELDS
    assert not hasattr(profile, "observation_projection")
    assert comparison_bound_type is not None
    assert isinstance(profile.comparison_bound, comparison_bound_type)
    assert profile.comparison_bound == comparison_bound_type(continuation_length=2)
    for old_field in REMOVED_PROFILE_BOUND_FIELDS:
        assert not hasattr(profile, old_field), f"Profile still exposes {old_field}"


def test_continuation_interface_declares_the_four_verbatim_components():
    interface = build_profile().continuation_interface

    for component in INTERFACE_COMPONENTS:
        assert hasattr(interface, component), f"I_P is missing {component}"
    assert isinstance(interface.configuration_type, type)
    for component in ("initialise", "step", "observe"):
        assert callable(getattr(interface, component))


def test_profile_carries_an_identifier():
    profile = build_profile()

    assert isinstance(profile.profile_id, str)
    assert profile.profile_id.strip() != ""


def test_the_declared_setting_is_the_single_export_approval_clause():
    """One small declared setting: a single clause, "export needs approval
    within a bound", realized by the interpreter's own AfterClauseSpec
    (interpreter/src/rs_metalang_ref/monitor.py) rather than reimplemented.
    """
    spec = build_profile().contract_environment.clause_spec

    assert isinstance(spec, AfterClauseSpec)
    assert spec.trigger_tag == "export"
    assert spec.response_tag == "approval"
    assert isinstance(spec.bound, int) and spec.bound >= 1
    assert isinstance(spec.discharge, Linear)


def test_continuation_alphabet_is_finite_and_can_close_a_continuation():
    """Sigma_P is finite, and at least one label is terminal so an admissible
    continuation can actually reach a completed contract observation.
    """
    alphabet = build_profile().continuation_alphabet

    assert isinstance(alphabet, tuple)
    assert 1 <= len(alphabet) <= 8, "the declared setting is small by construction"
    assert all(isinstance(label, str) for label in alphabet)
    assert len(set(alphabet)) == len(alphabet), "Sigma_P must not repeat a label"
    assert set(alphabet) & TERMINAL_LABELS, (
        "no terminal label: no continuation can close"
    )


def test_source_residual_replays_each_continuation_label_as_one_relative_advance(
    monkeypatch,
):
    """A string word label means "take the next continuation step" each time."""
    profile = build_profile()
    prefix = build_source_prefix(profile, SOURCE_TRACE)
    captured_ticks = []

    def capture_residual(monitor, events):
        event_tuple = tuple(events)
        captured_ticks.append(tuple(event.tick for event in event_tuple))
        return residual_module.monitor_residual(monitor, event_tuple)

    monkeypatch.setattr("rs_capsule.profile.monitor_residual", capture_residual)

    profile.contract_environment.source_residual(prefix, ("approval", "Complete"))

    assert captured_ticks == [(1, 1)]


def test_profile_is_fixed_before_any_candidate_is_seen():
    """Two gaming routes exist against a capsule like this one, the first
    being "choose an easy continuation family after seeing the candidate",
    closed by fixing the deployment profile, continuation family and
    observation projection before candidate assessment.

    Mechanically: build_profile() takes no candidate argument, and repeated
    construction yields the same declarations.
    """
    signature = inspect.signature(build_profile)
    required = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    ]
    assert required == [], (
        f"build_profile() must be candidate-independent, got {required}"
    )

    first, second = build_profile(), build_profile()
    assert first.profile_id == second.profile_id
    assert first.continuation_alphabet == second.continuation_alphabet
    assert first.comparison_bound == second.comparison_bound
    assert (
        first.retained_environment_manifest.fields
        == second.retained_environment_manifest.fields
    )


def test_retained_environment_manifest_is_enumerated_and_closed():
    """the profile-relative operational-adequacy rule requires the retained-environment
    manifest to identify every retained component. The extracted environment must
    expose exactly the declared fields, so E_P has no undeclared channel.
    """
    profile = build_profile()
    prefix = build_source_prefix(profile, SOURCE_TRACE)
    manifest = profile.retained_environment_manifest

    assert isinstance(manifest.fields, tuple)
    assert manifest.fields, "E_P must enumerate what it retains"
    assert all(isinstance(field, str) for field in manifest.fields)

    environment = manifest.extract(prefix)
    assert set(environment.keys()) == set(manifest.fields)


def test_occurrence_record_contains_only_an_open_snapshot():
    assert tuple(field.name for field in dataclasses.fields(OccurrenceRecord)) == (
        "oid",
        "trigger_tick",
        "deadline",
    )
    assert dataclasses.asdict(OccurrenceRecord("o1", 0, 2)) == {
        "oid": "o1",
        "trigger_tick": 0,
        "deadline": 2,
    }


def test_occurrence_record_rejects_a_lifecycle_status():
    assert "status" not in inspect.signature(OccurrenceRecord).parameters

    with pytest.raises(TypeError):
        OccurrenceRecord(
            "o1",
            trigger_tick=0,
            deadline=2,
            status="Open",
        )


def test_profile_initialise_uses_monitor_owned_state_card_restoration(monkeypatch):
    profile = build_profile()
    retained_events = (TickEvent(event_id="retained-0", tick=0),)
    term = StateCardTerm(
        retained_events=retained_events,
        occurrence_table=(OccurrenceRecord("o2", 3, 5),),
    )
    restore = getattr(SingleClauseMonitor, "restore_open_state_card", None)
    assert callable(restore), (
        "SingleClauseMonitor must own restore_open_state_card"
    )
    calls = []

    class SealedMonitor:
        def __init__(self, monitor):
            object.__setattr__(self, "_monitor", monitor)

        def __setattr__(self, name, value):
            if name in {"tick", "occurrences", "_oid_counter"}:
                raise AssertionError(
                    "The capsule must not assign monitor restoration state"
                )
            setattr(self._monitor, name, value)

        def __deepcopy__(self, memo):
            return copy.deepcopy(self._monitor, memo)

    class MonitorBoundary:
        def __init__(self, _spec):
            raise AssertionError(
                "The capsule must call restore_open_state_card"
            )

        @classmethod
        def restore_open_state_card(
            cls,
            spec,
            *,
            initial_guard_value,
            retained_events,
            current_tick,
            open_occurrences,
        ):
            calls.append(
                (
                    spec,
                    initial_guard_value,
                    retained_events,
                    current_tick,
                    open_occurrences,
                )
            )
            monitor = restore(
                spec,
                initial_guard_value=initial_guard_value,
                retained_events=retained_events,
                current_tick=current_tick,
                open_occurrences=open_occurrences,
            )
            return SealedMonitor(monitor)

    def reject_occurrence_construction(*_args, **_kwargs):
        raise AssertionError(
            "The capsule must not construct interpreter occurrences"
        )

    monkeypatch.setattr(profile_module, "SingleClauseMonitor", MonitorBoundary)
    monkeypatch.setattr(
        profile_module,
        "Occurrence",
        reject_occurrence_construction,
        raising=False,
    )

    configuration = profile.continuation_interface.initialise(
        term,
        {
            "current_tick": 3,
            "guard_value": K3.T,
        },
    )

    assert calls == [
        (
            profile.contract_environment.clause_spec,
            K3.T,
            retained_events,
            3,
            (("o2", 3, 5),),
        )
    ]
    assert type(configuration._snapshot()) is SingleClauseMonitor


def test_profile_restores_an_open_card_that_continues_normally():
    profile = build_profile()
    configuration = profile.continuation_interface.initialise(
        StateCardTerm(
            retained_events=(TickEvent(event_id="retained-0", tick=0),),
            occurrence_table=(OccurrenceRecord("o2", 3, 5),),
        ),
        {
            "current_tick": 3,
            "guard_value": K3.T,
        },
    )

    before = profile.continuation_interface.observe(configuration)
    continued = profile.continuation_interface.step(configuration, "approval")
    after = profile.continuation_interface.observe(continued)

    assert before.contract_observation.occurrences == {"o2": "Open"}
    assert before.contract_observation.mode.value == "Running"
    assert after.contract_observation.occurrences == {"o2": "Discharged"}
    assert after.contract_observation.mode.value == "Running"
    assert configuration._snapshot().occurrences["o2"].status.value == "Open"
    assert (
        continued._snapshot().occurrences["o2"].status.value
        == "Discharged"
    )


def test_retained_environment_carries_no_source_history(walk_values):
    """The retained environment must not carry the original source trace.

    Checked structurally, not by field name: no interpreter Event instance
    may appear anywhere inside E_P(u), and no source event identifier may
    appear in it.
    """
    from rs_metalang_ref.events import DomainEvent, TerminalEvent, TickEvent

    profile = build_profile()
    prefix = build_source_prefix(profile, SOURCE_TRACE)
    environment = profile.retained_environment_manifest.extract(prefix)

    values = list(walk_values(environment))
    source_events = [
        v for v in values if isinstance(v, (DomainEvent, TickEvent, TerminalEvent))
    ]
    assert source_events == [], f"E_P retains raw source events: {source_events}"

    source_ids = {
        event.event_id for event in SOURCE_TRACE if hasattr(event, "event_id")
    }
    retained_source_ids = {v for v in values if isinstance(v, str) and v in source_ids}
    assert retained_source_ids == set(), (
        f"E_P retains source event identifiers: {retained_source_ids}"
    )
