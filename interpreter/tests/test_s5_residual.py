"""Tests for the local S5 residual family.

They transcribe the contract-adequacy material's acceptance tests, worked
example, and failure case for contract-relative continuation equivalence.
"""

from __future__ import annotations

import dataclasses
from dataclasses import fields

import pytest

import rs_metalang_ref.residual as residual_module
from rs_metalang_ref.contracts import AfterClauseSpec, BoundUnaryGuard, Linear
from rs_metalang_ref.events import Complete, DomainEvent, ObservationCut, TerminalEvent
from rs_metalang_ref.kleene import K3
from rs_metalang_ref.monitor import SingleClauseMonitor
from rs_metalang_ref.residual import (
    DisabledReason,
    Distinguished,
    NoWitnessWithinBound,
    TotalizedLTS,
    UnitDataUnitAdvanceBound,
    binary_accept,
    bounded_compare,
    gate_projection,
    monitor_residual,
)
from rs_metalang_ref.verdict import Summary


def test_s5_residual_module_keeps_result_algebra_and_residual_behavior():
    result_constructors = {
        "Distinguished",
        "ProvedEquivalent",
        "NoWitnessWithinBound",
        "SearchIncomplete",
        "Untested",
    }
    assert result_constructors.difference(vars(residual_module)) == set()

    def transition(state, label):
        assert (state, label) == ("before", "step")
        return "after"

    lts = TotalizedLTS(
        transition=transition,
        enabled=lambda state: frozenset({"step"}),
        observe=lambda state: {"state": state},
    )

    assert lts.residual("before", ("step",)) == residual_module.TotalObservation(
        {"state": "after"},
        residual_module.NoDisabledLabel(),
    )


def test_s5_residual_module_has_no_executable_replay_receipt_producer():
    executable_producer_names = {
        "DisablementPoint",
        "ReplayReceipt",
        "replay_receipt",
    }

    assert executable_producer_names.intersection(vars(residual_module)) == set()


def test_s5_totalized_lts_not_enabled_sink_records_canonical_enabled_set():
    def transition(state, label):
        raise AssertionError("transition must not run for a disabled label")

    def enabled(state):
        if state == "s0":
            return frozenset({"approve", "reject", "delete"})
        return frozenset()

    def observe(state):
        return f"Obs({state})"

    lts = TotalizedLTS(transition, enabled, observe)

    sink = lts.delta_bar("s0", "not-a-real-label")

    assert isinstance(sink, residual_module.DisabledSink)
    assert sink.reason is DisabledReason.NOT_ENABLED
    assert sink.enabled_set == frozenset({"approve", "reject", "delete"})


def test_s5_totalized_lts_refuses_inconsistent_enabled_set_at_disablement():
    @dataclasses.dataclass(frozen=True)
    class Observation:
        enabled_actions: frozenset[str]

    def transition(state, label):
        raise AssertionError("transition must not run for a disabled label")

    def enabled(state):
        return frozenset({"approve"})

    def observe(state):
        return Observation(enabled_actions=frozenset({"reject"}))

    lts = TotalizedLTS(transition, enabled, observe)

    with pytest.raises(ValueError):
        lts.delta_bar("s0", "export")


def _spec(bound: int) -> AfterClauseSpec:
    return AfterClauseSpec(
        clause_id="c1",
        trigger_tag="export",
        response_tag="approval",
        binding_fields=("x",),
        guard=BoundUnaryGuard("Sensitive", "x"),
        bound=bound,
        discharge=Linear(),
    )


# -- Acceptance test 1: S5-enabled-domain-witness ---------------------------


def test_s5_enabled_domain_witness():
    o_u = {
        "summary": "Unknown",
        "mode": "Running",
        "clause_lifecycle": (),
        "observability": {},
        "enabled_actions": frozenset({"noop"}),
        "forbidden_actions": {},
        "evidence_classes": {},
        "evidence_delta": (),
        "certificates": (),
        "severity_maxima": (),
        "monitorability": {},
        "diagnostic": None,
    }
    o_v = {
        **o_u,
        "enabled_actions": frozenset(),
    }

    def transition(state, label):
        assert (state, label) == ("u", "noop")
        return "u_after"

    def enabled(state):
        return frozenset({"noop"}) if state in {"u", "u_after"} else frozenset()

    def observe(state):
        return {
            "u": o_u,
            "u_after": o_u,
            "v": o_v,
        }[state]

    lts = TotalizedLTS(transition, enabled, observe)

    no_disabled_label = residual_module.NoDisabledLabel()
    residual_empty = lts.residual("u", ())
    residual_u = lts.residual("u", ("noop",))
    residual_v = lts.residual("v", ("noop",))

    assert residual_empty == residual_module.TotalObservation(o_u, no_disabled_label)
    assert residual_u == residual_module.TotalObservation(o_u, no_disabled_label)
    assert tuple(field.name for field in fields(residual_empty)) == (
        "contract_observation",
        "continuation_status",
    )
    assert residual_v == residual_module.TotalObservation(
        o_v,
        residual_module.Disabled(
            attempted="noop",
            enabled_set=frozenset(),
            reason=DisabledReason.NOT_ENABLED,
        ),
    )
    assert lts.residual("v", ("noop", "ignored")) == residual_v

    assert residual_module.TotalObservation(
        {**o_u, "diagnostic": True},
        no_disabled_label,
    ) != residual_module.TotalObservation(
        {**o_u, "diagnostic": 1},
        no_disabled_label,
    )
    assert residual_module.TotalObservation(
        o_v,
        residual_module.Disabled(True, frozenset(), DisabledReason.NOT_ENABLED),
    ) != residual_module.TotalObservation(
        o_v,
        residual_module.Disabled(1, frozenset(), DisabledReason.NOT_ENABLED),
    )

    assert (
        lts.agrees_on_supplied_continuations("u", "v", [("noop",)]) is False
    )


def test_agreement_on_supplied_continuations_does_not_cover_an_unlisted_word():
    observations = {
        "u": "shared",
        "v": "shared",
        "u_revealed": "left",
        "v_revealed": "right",
    }

    lts = TotalizedLTS(
        transition=lambda state, label: f"{state}_revealed",
        enabled=lambda state: (
            frozenset({"reveal"}) if state in {"u", "v"} else frozenset()
        ),
        observe=lambda state: observations[state],
    )

    assert lts.agrees_on_supplied_continuations("u", "v", [()]) is True

    larger_search = bounded_compare(
        lambda word: lts.residual("u", word),
        lambda word: lts.residual("v", word),
        alphabet=("reveal",),
        bound=UnitDataUnitAdvanceBound(continuation_length=1),
        contract_environment="finite-agreement-example",
        observation_projection="TotalObservation_C",
        continuation_family="all-words-through-declared-bound",
    )

    assert isinstance(larger_search, Distinguished)
    assert larger_search.witness == ("reveal",)


# -- Acceptance test 2: S5-terminal-labels-distinguish ----------------------


def test_s5_terminal_labels_distinguish():
    monitor = SingleClauseMonitor(_spec(bound=5))
    monitor.set_initial("Sensitive(f)", K3.T)
    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    # The open obligation has deadline 5; branch at tick 1, well before it,
    # so the difference is purely the terminal-conversion table.

    complete_obs = monitor_residual(
        monitor, [TerminalEvent("c", tick=1, kind=Complete())]
    )
    cut_obs = monitor_residual(
        monitor, [TerminalEvent("k", tick=1, kind=ObservationCut("s"))]
    )

    assert complete_obs.contract_observation.summary is Summary.VIOLATED
    assert cut_obs.contract_observation.summary is Summary.UNKNOWN
    assert complete_obs != cut_obs


# -- Acceptance test 3: S5-canonical-event-identifiers ----------------------


def test_s5_canonical_event_identifiers():
    def build(event_ids):
        monitor = SingleClauseMonitor(_spec(bound=2))
        monitor.set_initial("Sensitive(f)", K3.T)
        monitor.step(DomainEvent(event_ids[0], tick=0, tag="export", fields={"x": "f"}))
        monitor.step(
            DomainEvent(event_ids[1], tick=1, tag="approval", fields={"x": "f"})
        )
        return monitor.current_verdict()

    left = build(["17", "22"])
    right = build(["203", "991"])

    # Canonical oids (o1, o2, ...) are assigned by a pure counter, independent
    # of the raw event_id, so structurally identical traces canonicalize to
    # the same observation.
    assert left.summary == right.summary
    assert left.mode == right.mode
    assert left.occurrences == right.occurrences


# -- Acceptance test 4: S5-two-versus-three-valued --------------------------


def test_s5_two_versus_three_valued():
    binary_u = binary_accept(Summary.VIOLATED)
    binary_v = binary_accept(Summary.UNKNOWN)
    assert binary_u == binary_v is False  # binary_equivalent: true

    assert Summary.VIOLATED != Summary.UNKNOWN  # three_valued_equivalent: false

    assert gate_projection(Summary.VIOLATED) == gate_projection(
        Summary.UNKNOWN
    )  # gate_projection_equivalent: true


# -- Worked residual example --------------------------------------------------


def test_s5_worked_residual_example():
    def build(with_approval: bool):
        monitor = SingleClauseMonitor(_spec(bound=2))
        monitor.set_initial("Sensitive(f)", K3.T)
        monitor.step(DomainEvent("export", tick=0, tag="export", fields={"x": "f"}))
        if with_approval:
            monitor.step(
                DomainEvent("approval", tick=1, tag="approval", fields={"x": "f"})
            )
        return monitor

    prefix_u = build(with_approval=False)
    prefix_v = build(with_approval=True)

    complete = [TerminalEvent("complete", tick=3, kind=Complete())]
    obs_u = monitor_residual(prefix_u, complete)
    obs_v = monitor_residual(prefix_v, complete)

    assert obs_u.contract_observation.summary is Summary.VIOLATED
    assert obs_v.contract_observation.summary is Summary.SATISFIED
    assert obs_u != obs_v  # Complete distinguishes u and v


# -- Demonstrated failure: NoWitnessWithinBound, then Distinguished ----------


def test_s5_demonstrated_failure_no_witness_then_distinguished():
    def make_residual(threshold):
        def residual(word):
            return "flagged" if len(word) >= threshold else "clear"

        return residual

    residual_u = make_residual(threshold=3)
    residual_v = make_residual(threshold=4)

    bounded = bounded_compare(
        residual_u,
        residual_v,
        alphabet=("a",),
        bound=UnitDataUnitAdvanceBound(continuation_length=2),
        contract_environment="C",
        observation_projection="identity[str]",
        continuation_family="all-words-through-declared-bound",
    )
    assert isinstance(bounded, NoWitnessWithinBound)
    assert bounded.scope.bound == UnitDataUnitAdvanceBound(continuation_length=2)

    found = bounded_compare(
        residual_u,
        residual_v,
        alphabet=("a",),
        bound=UnitDataUnitAdvanceBound(continuation_length=3),
        contract_environment="C",
        observation_projection="identity[str]",
        continuation_family="all-words-through-declared-bound",
    )
    assert isinstance(found, Distinguished)
    assert found.witness == ("a", "a", "a")
