"""S1: total three-valued bounded-response semantics.

This file tests the S1 acceptance tests, worked example, and failure case for
bounded-response semantics. The
public seam is `SingleClauseMonitor`.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from rs_metalang_ref.contracts import (
    AfterClauseSpec,
    BoundUnaryGuard,
    Breach,
    Broadcast,
    GroundGuard,
    Indeterminate,
    InvalidAfterClauseSpec,
    InvalidBinding,
    Linear,
    WaiveIf,
    WellFormednessCertificate,
)
from rs_metalang_ref.events import (
    Complete,
    DomainEvent,
    ObservationCut,
    TerminalEvent,
    TickEvent,
    Timeout,
)
from rs_metalang_ref.kleene import K3
from rs_metalang_ref.monitor import MonitorMode, SingleClauseMonitor
from rs_metalang_ref.obligations import OccurrenceStatus
from rs_metalang_ref.verdict import DenyUnknown, Mode, Summary


def _spec(bound: int, clause_id: str = "c1") -> AfterClauseSpec:
    return AfterClauseSpec(
        clause_id=clause_id,
        trigger_tag="export",
        response_tag="approval",
        binding_fields=("x",),
        guard=BoundUnaryGuard("Sensitive", "x"),
        bound=bound,
        discharge=Linear(),
    )


def test_s1_well_formedness_public_api_exposes_declared_premises_only():
    certificate = WellFormednessCertificate(
        attested_total_deterministic_pattern_matching=True,
        attested_variables_bound=True,
        attested_trigger_time_guard_only=True,
        attested_total_atomic_predicate_adapters=True,
        attested_total_effect_adapters=True,
        attested_decidable_broadcast_keys=True,
        attested_typed_waive_if_predicates=True,
        clause_ids=("c1", "c2"),
    )

    assert certificate.declared_well_formedness_premises_hold() is True
    assert not hasattr(certificate, "holds")


@pytest.mark.parametrize(
    "non_boolean_attestation",
    [
        "attested_total_deterministic_pattern_matching",
        "attested_variables_bound",
        "attested_trigger_time_guard_only",
        "attested_total_atomic_predicate_adapters",
        "attested_total_effect_adapters",
        "attested_decidable_broadcast_keys",
        "attested_typed_waive_if_predicates",
    ],
)
def test_s1_well_formedness_rejects_truthy_non_boolean_attestations_at_construction(
    non_boolean_attestation,
):
    attestations = {
        "attested_total_deterministic_pattern_matching": True,
        "attested_variables_bound": True,
        "attested_trigger_time_guard_only": True,
        "attested_total_atomic_predicate_adapters": True,
        "attested_total_effect_adapters": True,
        "attested_decidable_broadcast_keys": True,
        "attested_typed_waive_if_predicates": True,
    }
    attestations[non_boolean_attestation] = "yes"

    with pytest.raises(ValueError):
        WellFormednessCertificate(
            clause_ids=("c1", "c2"),
            **attestations,
        )


@pytest.mark.parametrize(
    "false_attestation",
    [
        "attested_total_deterministic_pattern_matching",
        "attested_variables_bound",
        "attested_trigger_time_guard_only",
        "attested_total_atomic_predicate_adapters",
        "attested_total_effect_adapters",
        "attested_decidable_broadcast_keys",
        "attested_typed_waive_if_predicates",
    ],
)
def test_s1_well_formedness_rejects_each_false_attestation(false_attestation):
    attestations = {
        "attested_total_deterministic_pattern_matching": True,
        "attested_variables_bound": True,
        "attested_trigger_time_guard_only": True,
        "attested_total_atomic_predicate_adapters": True,
        "attested_total_effect_adapters": True,
        "attested_decidable_broadcast_keys": True,
        "attested_typed_waive_if_predicates": True,
    }
    attestations[false_attestation] = False
    certificate = WellFormednessCertificate(
        clause_ids=("c1", "c2"),
        **attestations,
    )

    assert certificate.declared_well_formedness_premises_hold() is False


def test_s1_well_formedness_rejects_duplicate_clause_ids():
    certificate = WellFormednessCertificate(
        attested_total_deterministic_pattern_matching=True,
        attested_variables_bound=True,
        attested_trigger_time_guard_only=True,
        attested_total_atomic_predicate_adapters=True,
        attested_total_effect_adapters=True,
        attested_decidable_broadcast_keys=True,
        attested_typed_waive_if_predicates=True,
        clause_ids=("c1", "c1"),
    )

    assert certificate.declared_well_formedness_premises_hold() is False


# -- S1.10 acceptance test 1: S1-response-at-inclusive-deadline -------------


def test_s1_response_at_inclusive_deadline():
    monitor = SingleClauseMonitor(_spec(bound=1))
    monitor.set_initial("Sensitive(f)", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    verdict = monitor.step(DomainEvent("e1", tick=1, tag="approval", fields={"x": "f"}))
    verdict = monitor.step(TerminalEvent("e2", tick=2, kind=Complete()))

    assert monitor.occurrences["o1"].status is OccurrenceStatus.DISCHARGED
    assert monitor.occurrences["o1"].response.response_tick == 1
    assert verdict.summary is Summary.SATISFIED
    assert not any(
        occ.status is OccurrenceStatus.BREACHED for occ in monitor.occurrences.values()
    )


# -- S1.10 acceptance test 2: S1-observation-cut-does-not-breach ------------


def test_s1_observation_cut_does_not_breach():
    monitor = SingleClauseMonitor(_spec(bound=5))
    monitor.set_initial("Sensitive(f)", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    verdict = monitor.step(TerminalEvent("e1", tick=1, kind=ObservationCut("source")))

    occ = monitor.occurrences["o1"]
    assert occ.status is OccurrenceStatus.UNKNOWN_FINAL
    assert occ.breach_reason == "ObservationTruncated"
    assert verdict.summary is Summary.UNKNOWN
    assert not any(
        o.status is OccurrenceStatus.BREACHED for o in monitor.occurrences.values()
    )


def test_s1_timeout_before_deadline_is_unknown_final():
    monitor = SingleClauseMonitor(_spec(bound=5))
    monitor.set_initial("Sensitive(f)", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    verdict = monitor.step(TerminalEvent("e1", tick=1, kind=Timeout("horizon")))

    occurrence = monitor.occurrences["o1"]
    assert occurrence.status is OccurrenceStatus.UNKNOWN_FINAL
    assert occurrence.breach_reason == "TruncatedBeforeDeadline"
    assert verdict.summary is Summary.UNKNOWN
    assert verdict.mode is Mode.TIMEOUT


# -- S1.10 acceptance test 3: S1-linear-response-does-not-discharge-two -----


def test_s1_linear_response_does_not_discharge_two():
    monitor = SingleClauseMonitor(_spec(bound=2))
    monitor.set_initial("Sensitive(f)", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    monitor.step(DomainEvent("e1", tick=1, tag="export", fields={"x": "f"}))
    monitor.step(DomainEvent("e2", tick=2, tag="approval", fields={"x": "f"}))
    verdict = monitor.step(TerminalEvent("e3", tick=3, kind=Complete()))

    statuses = [occ.status for occ in monitor.occurrences.values()]
    assert statuses.count(OccurrenceStatus.DISCHARGED) == 1
    assert statuses.count(OccurrenceStatus.BREACHED) == 1
    assert verdict.summary is Summary.VIOLATED


def _ground_spec(bound: int, clause_id: str = "ground-c1") -> AfterClauseSpec:
    return AfterClauseSpec(
        clause_id=clause_id,
        trigger_tag="export",
        response_tag="approval",
        binding_fields=(),
        guard=GroundGuard("approval-required"),
        bound=bound,
        discharge=Linear(),
    )


def test_s1_a_response_field_named_discharges_does_not_pick_the_duty():
    # The clause binds no event data, so both duties carry the same bindings
    # and the response settles the one with the least canonical key: the
    # tick-0 duty, due first. An event field cannot name a different target,
    # because the core allocates by that key alone.
    monitor = SingleClauseMonitor(_ground_spec(bound=2))
    monitor.set_initial("approval-required", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export"))
    monitor.step(DomainEvent("e1", tick=1, tag="export"))
    verdict = monitor.step(
        DomainEvent("e2", tick=2, tag="approval", fields={"discharges": ("o2",)})
    )

    assert monitor.occurrences["o1"].status is OccurrenceStatus.DISCHARGED
    assert monitor.occurrences["o2"].status is OccurrenceStatus.OPEN
    assert (verdict.summary, verdict.mode) == (Summary.UNKNOWN, Mode.RUNNING)


def test_s1_a_discharges_field_of_any_shape_is_ordinary_event_data():
    # The core reserves no field name, so a value here changes nothing and is
    # not a reason to refuse the event.
    for value in (("o2",), "o2", 7, None, {"o2": 1}):
        monitor = SingleClauseMonitor(_ground_spec(bound=2))
        monitor.set_initial("approval-required", K3.T)
        monitor.step(DomainEvent("e0", tick=0, tag="export"))
        verdict = monitor.step(
            DomainEvent("e1", tick=1, tag="approval", fields={"discharges": value})
        )
        assert monitor.occurrences["o1"].status is OccurrenceStatus.DISCHARGED
        assert verdict.summary is Summary.UNKNOWN


_SEED_DRIVER = """
from rs_metalang_ref.contracts import AfterClauseSpec, GroundGuard, Linear
from rs_metalang_ref.events import Complete, DomainEvent, TerminalEvent
from rs_metalang_ref.kleene import K3
from rs_metalang_ref.monitor import SingleClauseMonitor
from rs_metalang_ref.obligations import OccurrenceStatus

spec = AfterClauseSpec(
    clause_id="ground-c1",
    trigger_tag="export",
    response_tag="approval",
    binding_fields=(),
    guard=GroundGuard("approval-required"),
    bound=5,
    discharge=Linear(),
)
monitor = SingleClauseMonitor(spec)
monitor.set_initial("approval-required", K3.T)
monitor.step(DomainEvent("e0", tick=0, tag="export"))
monitor.step(DomainEvent("e1", tick=1, tag="export"))
monitor.step(DomainEvent("e2", tick=2, tag="export"))
monitor.step(DomainEvent("e3", tick=3, tag="approval"))
verdict = monitor.step(TerminalEvent("e4", tick=4, kind=Complete()))
discharged = sorted(
    oid for oid, occ in monitor.occurrences.items()
    if occ.status is OccurrenceStatus.DISCHARGED
)
print(verdict.summary.name + "|" + ",".join(discharged))
"""


def test_s1_allocation_is_the_same_under_every_hash_seed():
    """One trace gives one verdict and one discharged duty, whatever the
    interpreter's hash randomisation.

    Three duties are open and one approval arrives, so the canonical key has
    to pick the same single winner every run. PYTHONHASHSEED is read once at
    interpreter start and cannot be varied inside a running process, so this
    runs the trace in subprocesses under different seeds.

    Two claims, in order: the seeds agree with each other, and what they agree
    on is one discharge. An implementation that settled every eligible duty
    would satisfy the first and fail the second.
    """
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    results = {}
    for seed in range(6):
        completed = subprocess.run(
            [sys.executable, "-c", _SEED_DRIVER],
            env={**env, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            check=True,
        )
        results[seed] = completed.stdout.strip()

    distinct = sorted(set(results.values()))
    assert len(distinct) == 1, f"outcome varies with PYTHONHASHSEED: {results}"

    summary, discharged = distinct[0].split("|")
    discharged_oids = [oid for oid in discharged.split(",") if oid]
    assert discharged_oids == ["o1"]
    assert summary == Summary.VIOLATED.name


@pytest.mark.parametrize(
    ("occurrence_fields", "expected_oid"),
    (
        ({"o1": {"deadline": 9}, "o2": {"deadline": 5}}, "o2"),
        (
            {
                "o1": {"deadline": 9, "trigger_tick": 4},
                "o2": {"deadline": 9, "trigger_tick": 3},
            },
            "o2",
        ),
        (
            {
                "o1": {"deadline": 9, "trigger_tick": 4, "clause_id": "c2"},
                "o2": {"deadline": 9, "trigger_tick": 4, "clause_id": "c1"},
            },
            "o2",
        ),
        (
            {
                "o1": {"deadline": 9, "trigger_tick": 4, "clause_id": "c1"},
                "o2": {"deadline": 9, "trigger_tick": 4, "clause_id": "c1"},
            },
            "o1",
        ),
    ),
    ids=("deadline", "trigger-tick", "clause-id", "obligation-id"),
)
def test_s1_public_step_allocates_one_shared_retroactive_token_by_canonical_order(
    occurrence_fields,
    expected_oid,
):
    monitor = SingleClauseMonitor(_spec(bound=9))
    monitor.set_initial("Sensitive(f)", K3.U)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    monitor.step(DomainEvent("e1", tick=1, tag="export", fields={"x": "f"}))
    monitor.step(DomainEvent("r2", tick=2, tag="approval", fields={"x": "f"}))

    shared_token = monitor.occurrences["o1"].candidates[0]
    assert monitor.occurrences["o2"].candidates == [shared_token]

    for oid, fields in occurrence_fields.items():
        occurrence = monitor.occurrences[oid]
        for field, value in fields.items():
            setattr(occurrence, field, value)
    monitor.occurrences = dict(reversed(tuple(monitor.occurrences.items())))

    verdict = monitor.step(
        TickEvent("resolve20", tick=20, observations={"Sensitive(f)": K3.T})
    )

    loser_oid = "o1" if expected_oid == "o2" else "o2"
    assert monitor.occurrences[expected_oid].status is OccurrenceStatus.DISCHARGED
    assert monitor.occurrences[expected_oid].response is shared_token
    assert monitor.occurrences[loser_oid].status is OccurrenceStatus.BREACHED
    assert monitor.occurrences[loser_oid].response is None
    assert shared_token.consumed is True
    assert verdict.summary is Summary.VIOLATED


# -- S1.10 acceptance test 4: S1-retroactive-activation-without-response ---


def test_s1_retroactive_activation_without_response():
    monitor = SingleClauseMonitor(_spec(bound=1))
    monitor.set_initial("Sensitive(f)", K3.U)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    monitor.step(TickEvent("e1", tick=1))
    verdict = monitor.step(TickEvent("e2", tick=2, observations={"Sensitive(f)": K3.T}))

    occ = monitor.occurrences["o1"]
    assert occ.status is OccurrenceStatus.BREACHED
    assert occ.effective_time == 1
    assert occ.discovery_time == 2
    assert occ.breach_reason == "RetroactiveActivationAfterDeadline"
    assert verdict.summary is Summary.VIOLATED


# -- S1.11: fully worked export/Sensitive/approval example ------------------


def test_s1_worked_example_export_sensitive_approval():
    monitor = SingleClauseMonitor(_spec(bound=1))
    monitor.set_initial("Sensitive(f)", K3.U)

    # Tick 10: export(f) -- guard unknown, ConditionalOpen with D=11.
    monitor.step(DomainEvent("e10", tick=10, tag="export", fields={"x": "f"}))
    occ = monitor.occurrences["o1"]
    assert occ.status is OccurrenceStatus.CONDITIONAL_OPEN
    assert occ.deadline == 11
    assert monitor.current_verdict().summary is Summary.UNKNOWN

    # Tick 11: approval(f), on time (11 == D), and reserved for the
    # occurrence while its guard is unknown. Boundary expiry then converts the
    # occurrence to ConditionalExpired, not a breach.
    monitor.step(DomainEvent("e11", tick=11, tag="approval", fields={"x": "f"}))
    occ = monitor.occurrences["o1"]
    assert occ.status is OccurrenceStatus.CONDITIONAL_EXPIRED
    assert len(occ.candidates) == 1
    assert occ.candidates[0].rid == "e11"
    assert not any(
        o.status is OccurrenceStatus.BREACHED for o in monitor.occurrences.values()
    )

    # Tick 12: the historical guard refines to true. Retroactive activation
    # with the UNCHANGED deadline (D=11); the reserved candidate satisfies
    # t_r=11 <= D=11, so the occurrence is Discharged, not breached.
    monitor.step(
        DomainEvent(
            "e12", tick=12, tag="Observation", observations={"Sensitive(f)": K3.T}
        )
    )
    occ = monitor.occurrences["o1"]
    assert occ.status is OccurrenceStatus.DISCHARGED
    assert occ.response.rid == "e11"

    # Tick 13: Complete -- no open or conditional occurrence remains, so the
    # terminal verdict is Satisfied with no certificates.
    verdict = monitor.step(TerminalEvent("e13", tick=13, kind=Complete()))
    assert verdict.summary is Summary.SATISFIED
    assert verdict.mode is Mode.COMPLETE


# -- S1.12: DEMONSTRATED-FAILURE (InconsistentObservation) -----------------


def test_s1_demonstrated_failure_inconsistent_observation():
    monitor = SingleClauseMonitor(_spec(bound=1))
    monitor.set_initial("Sensitive(f)@10", K3.U)

    monitor.step(TickEvent("e10", tick=10, observations={"Sensitive(f)@10": K3.T}))
    monitor.step(TickEvent("e11", tick=11, observations={"Sensitive(f)@10": K3.F}))

    assert monitor.mode is MonitorMode.FAULTED
    assert monitor.fault.code == "InconsistentObservation"
    assert monitor.fault.proposition_key == "Sensitive(f)@10"
    assert monitor.fault.prior_value is K3.T
    assert monitor.fault.attempted_value is K3.F

    verdict = monitor.current_verdict()
    assert verdict.summary is Summary.UNKNOWN
    assert verdict.mode is Mode.FAULTED

    permission = monitor.permit_after_fault()
    assert permission == DenyUnknown("MonitorSemanticFault")


# -- Binding semantics -----------------------------------------------------
#
# The clause is `after export(x) when Sensitive(x) require approval(x) within 1`.
# the static well-formedness rule, condition 2: "for `after`, the
# `TriggerPattern as Binding` site supplies them", where "them" is the free
# variables of every predicate and response pattern of that clause. So one trigger
# occurrence binds `x`, and both the guard and the response of that occurrence read
# that binding. A different value of `x` is a different occurrence.
#
# The tests below state that rule at the public seam. They do not inspect internals.


def _binding_spec(bound: int = 1) -> AfterClauseSpec:
    return AfterClauseSpec(
        clause_id="c1",
        trigger_tag="export",
        response_tag="approval",
        binding_fields=("x",),
        guard=BoundUnaryGuard("Sensitive", "x"),
        bound=bound,
        discharge=Linear(),
    )


def test_s1_response_for_a_different_binding_does_not_discharge():
    """A response about `g` does not discharge an obligation triggered on `f`."""
    monitor = SingleClauseMonitor(_binding_spec())
    monitor.set_initial("Sensitive(f)", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    monitor.step(DomainEvent("e1", tick=1, tag="approval", fields={"x": "g"}))
    verdict = monitor.step(TerminalEvent("e2", tick=2, kind=Complete()))

    assert monitor.occurrences["o1"].status is OccurrenceStatus.BREACHED
    assert verdict.summary is Summary.VIOLATED


def test_s1_response_for_the_same_binding_discharges():
    """Positive control: a response about `f` discharges the `f` obligation."""
    monitor = SingleClauseMonitor(_binding_spec())
    monitor.set_initial("Sensitive(f)", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    monitor.step(DomainEvent("e1", tick=1, tag="approval", fields={"x": "f"}))
    verdict = monitor.step(TerminalEvent("e2", tick=2, kind=Complete()))

    assert monitor.occurrences["o1"].status is OccurrenceStatus.DISCHARGED
    assert verdict.summary is Summary.SATISFIED


def test_s1_response_matching_no_open_binding_discharges_nothing():
    """Two triggers, and a response about a third value, discharge neither."""
    monitor = SingleClauseMonitor(_binding_spec(bound=5))
    monitor.set_initial("Sensitive(f)", K3.T)
    monitor.set_initial("Sensitive(h)", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    monitor.step(DomainEvent("e1", tick=1, tag="export", fields={"x": "h"}))
    monitor.step(DomainEvent("e2", tick=2, tag="approval", fields={"x": "unrelated"}))
    verdict = monitor.step(TerminalEvent("e3", tick=3, kind=Complete()))

    statuses = [occurrence.status for occurrence in monitor.occurrences.values()]
    assert OccurrenceStatus.DISCHARGED not in statuses
    assert verdict.summary is Summary.VIOLATED


def test_s1_guard_is_read_at_the_triggered_binding():
    """A false guard at the triggered binding makes the occurrence inapplicable.

    S1.4 trigger-and-guard rule: `F => Inapplicable`. The occurrence is
    recorded; it does not become a duty.
    """
    monitor = SingleClauseMonitor(_binding_spec())
    monitor.set_initial("Sensitive(f)", K3.T)
    monitor.set_initial("Sensitive(g)", K3.F)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "g"}))
    verdict = monitor.step(TerminalEvent("e1", tick=1, kind=Complete()))

    statuses = {occurrence.status for occurrence in monitor.occurrences.values()}
    assert statuses <= {OccurrenceStatus.INAPPLICABLE}
    assert verdict.summary is not Summary.VIOLATED


def test_s1_guard_opens_an_obligation_at_a_true_binding():
    """A true guard at the triggered binding opens a duty.

    The control for the test above. `Sensitive(f)` is false here, so a monitor
    that reads the constructor's fixed key rather than the triggered binding
    reports the occurrence inapplicable and the trace satisfied.
    """
    monitor = SingleClauseMonitor(_binding_spec())
    monitor.set_initial("Sensitive(f)", K3.F)
    monitor.set_initial("Sensitive(g)", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "g"}))
    verdict = monitor.step(TerminalEvent("e1", tick=1, kind=Complete()))

    statuses = {occurrence.status for occurrence in monitor.occurrences.values()}
    assert OccurrenceStatus.INAPPLICABLE not in statuses
    assert verdict.summary is Summary.VIOLATED


def test_s1_ground_guard_does_not_rebind_to_event_field():
    spec = AfterClauseSpec(
        clause_id="c1",
        trigger_tag="export",
        response_tag="approval",
        binding_fields=("x",),
        guard=GroundGuard("CountryAllowed(UK)"),
        bound=1,
        discharge=Linear(),
    )
    monitor = SingleClauseMonitor(spec)
    monitor.set_initial("CountryAllowed(UK)", K3.T)
    monitor.set_initial("CountryAllowed(f)", K3.F)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    monitor.step(DomainEvent("e1", tick=1, tag="approval", fields={"x": "f"}))
    verdict = monitor.step(TerminalEvent("e2", tick=2, kind=Complete()))

    assert monitor.occurrences["o1"].status is OccurrenceStatus.DISCHARGED
    assert verdict.summary is Summary.SATISFIED


def test_s1_missing_trigger_field_is_no_match():
    monitor = SingleClauseMonitor(_binding_spec())
    monitor.set_initial("Sensitive(f)", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"y": "f"}))
    verdict = monitor.step(TerminalEvent("e1", tick=1, kind=Complete()))

    assert monitor.occurrences == {}
    assert verdict.summary is Summary.SATISFIED


def test_s1_missing_response_field_is_no_match():
    monitor = SingleClauseMonitor(_binding_spec())
    monitor.set_initial("Sensitive(f)", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    monitor.step(DomainEvent("e1", tick=1, tag="approval", fields={"y": "f"}))
    verdict = monitor.step(TerminalEvent("e2", tick=2, kind=Complete()))

    assert monitor.occurrences["o1"].status is OccurrenceStatus.BREACHED
    assert verdict.summary is Summary.VIOLATED


def test_s1_same_event_trigger_then_response_can_discharge_new_occurrence():
    spec = AfterClauseSpec(
        clause_id="c1",
        trigger_tag="export",
        response_tag="export",
        binding_fields=("x",),
        guard=BoundUnaryGuard("Sensitive", "x"),
        bound=1,
        discharge=Linear(),
    )
    monitor = SingleClauseMonitor(spec)
    monitor.set_initial("Sensitive(f)", K3.T)

    verdict = monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))

    assert monitor.occurrences["o1"].status is OccurrenceStatus.DISCHARGED
    assert verdict.summary is Summary.UNKNOWN


def test_s1_response_for_another_binding_does_not_reserve_a_conditional():
    monitor = SingleClauseMonitor(_binding_spec())
    monitor.set_initial("Sensitive(f)", K3.U)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    monitor.step(DomainEvent("e1", tick=1, tag="approval", fields={"x": "g"}))
    verdict = monitor.step(
        DomainEvent(
            "e2", tick=2, tag="Observation", observations={"Sensitive(f)": K3.T}
        )
    )

    assert monitor.occurrences["o1"].status is OccurrenceStatus.BREACHED
    assert monitor.occurrences["o1"].candidates == []
    assert verdict.summary is Summary.VIOLATED


def test_s1_non_string_bound_value_is_rejected_before_state_change():
    monitor = SingleClauseMonitor(_binding_spec())
    monitor.set_initial("Sensitive(f)", K3.T)

    with pytest.raises(InvalidBinding):
        monitor.step(
            DomainEvent(
                "e0",
                tick=0,
                tag="export",
                fields={"x": 123},
                observations={"Other": K3.T},
            )
        )

    assert monitor.tick == -1
    assert monitor.occurrences == {}
    assert "Other" not in monitor.epistemic


def test_s1_guard_value_for_requires_the_exact_projected_substitution():
    monitor = SingleClauseMonitor(_binding_spec())
    monitor.set_initial("Sensitive(f)", K3.T)

    assert monitor.guard_value_for({"x": "f"}) is K3.T
    with pytest.raises(InvalidBinding):
        monitor.guard_value_for({})
    with pytest.raises(InvalidBinding):
        monitor.guard_value_for({"x": "f", "y": "extra"})
    with pytest.raises(InvalidBinding):
        monitor.guard_value_for({"x": 1})


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GroundGuard(""),
        lambda: BoundUnaryGuard("", "x"),
        lambda: BoundUnaryGuard("Sensitive", ""),
        lambda: AfterClauseSpec(
            clause_id="c1",
            trigger_tag="export",
            response_tag="approval",
            binding_fields=["x"],
            guard=BoundUnaryGuard("Sensitive", "x"),
            bound=1,
        ),
        lambda: AfterClauseSpec(
            clause_id="c1",
            trigger_tag="export",
            response_tag="approval",
            binding_fields=("x", "y"),
            guard=BoundUnaryGuard("Sensitive", "x"),
            bound=1,
        ),
        lambda: AfterClauseSpec(
            clause_id="c1",
            trigger_tag="export",
            response_tag="approval",
            binding_fields=("x",),
            guard=BoundUnaryGuard("Sensitive", "y"),
            bound=1,
        ),
        lambda: AfterClauseSpec(
            clause_id="c1",
            trigger_tag="export",
            response_tag="approval",
            binding_fields=("x",),
            guard="Sensitive(x)",
            bound=1,
        ),
    ],
)
def test_s1_invalid_after_clause_shapes_are_rejected(factory):
    with pytest.raises(InvalidAfterClauseSpec):
        factory()


@pytest.mark.parametrize(
    ("field_name", "invalid_variant"),
    [
        pytest.param("discharge", object(), id="discharge-object"),
        pytest.param(
            "discharge",
            type("LinearSubclass", (Linear,), {})(),
            id="discharge-subclass",
        ),
        pytest.param("on_agent_abort", object(), id="abort-object"),
        pytest.param(
            "on_agent_abort",
            type("BreachSubclass", (Breach,), {})(),
            id="abort-subclass",
        ),
    ],
)
def test_s1_after_clause_rejects_values_outside_exact_closed_variants(
    field_name,
    invalid_variant,
):
    fields = {
        "clause_id": "c1",
        "trigger_tag": "export",
        "response_tag": "approval",
        "binding_fields": (),
        "guard": GroundGuard("Sensitive"),
        "bound": 1,
        "discharge": Linear(),
        "on_agent_abort": Breach(),
    }
    fields[field_name] = invalid_variant

    with pytest.raises(InvalidAfterClauseSpec, match=field_name):
        AfterClauseSpec(**fields)


@pytest.mark.parametrize(
    ("discharge", "on_agent_abort"),
    [
        pytest.param(Linear(), Breach(), id="linear-breach"),
        pytest.param(Broadcast("group"), Indeterminate(), id="broadcast-indeterminate"),
        pytest.param(Linear(), WaiveIf("abort-authority"), id="linear-waive-if"),
    ],
)
def test_s1_after_clause_accepts_each_exact_closed_variant(
    discharge,
    on_agent_abort,
):
    spec = AfterClauseSpec(
        clause_id="c1",
        trigger_tag="export",
        response_tag="approval",
        binding_fields=(),
        guard=GroundGuard("Sensitive"),
        bound=1,
        discharge=discharge,
        on_agent_abort=on_agent_abort,
    )

    assert spec.discharge == discharge
    assert spec.on_agent_abort == on_agent_abort


def test_s1_waive_if_rejects_malformed_authority_before_monitor_construction():
    with pytest.raises(InvalidAfterClauseSpec):
        WaiveIf(authority_key=[])


@pytest.mark.parametrize(
    "bound",
    [
        pytest.param(-1, id="negative"),
        pytest.param(True, id="bool"),
        pytest.param(1.5, id="float"),
    ],
)
def test_s1_after_clause_bound_must_be_a_natural_number(bound):
    with pytest.raises(InvalidAfterClauseSpec):
        _spec(bound)
