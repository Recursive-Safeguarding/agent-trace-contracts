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

from rs_metalang_ref import contracts as contracts_module
from rs_metalang_ref.contracts import (
    AfterClauseSpec,
    Breach,
    GroundGuard,
    InvalidAfterClauseSpec,
    InvalidBinding,
    Linear,
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
        binding_fields=(),
        guard=GroundGuard("approval-required"),
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
    monitor.set_initial("approval-required", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export"))
    verdict = monitor.step(DomainEvent("e1", tick=1, tag="approval"))
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
    monitor.set_initial("approval-required", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export"))
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
    monitor.set_initial("approval-required", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export"))
    verdict = monitor.step(TerminalEvent("e1", tick=1, kind=Timeout("horizon")))

    occurrence = monitor.occurrences["o1"]
    assert occurrence.status is OccurrenceStatus.UNKNOWN_FINAL
    assert occurrence.breach_reason == "TruncatedBeforeDeadline"
    assert verdict.summary is Summary.UNKNOWN
    assert verdict.mode is Mode.TIMEOUT


# -- S1.10 acceptance test 3: S1-linear-response-does-not-discharge-two -----


def test_s1_linear_response_does_not_discharge_two():
    monitor = SingleClauseMonitor(_spec(bound=2))
    monitor.set_initial("approval-required", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export"))
    monitor.step(DomainEvent("e1", tick=1, tag="export"))
    monitor.step(DomainEvent("e2", tick=2, tag="approval"))
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
    monitor.set_initial("approval-required", K3.U)

    monitor.step(DomainEvent("e0", tick=0, tag="export"))
    monitor.step(DomainEvent("e1", tick=1, tag="export"))
    monitor.step(DomainEvent("r2", tick=2, tag="approval"))

    shared_token = monitor.occurrences["o1"].candidates[0]
    assert monitor.occurrences["o2"].candidates == [shared_token]

    for oid, fields in occurrence_fields.items():
        occurrence = monitor.occurrences[oid]
        for field, value in fields.items():
            setattr(occurrence, field, value)
    monitor.occurrences = dict(reversed(tuple(monitor.occurrences.items())))

    verdict = monitor.step(
        TickEvent("resolve20", tick=20, observations={"approval-required": K3.T})
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
    monitor.set_initial("approval-required", K3.U)

    monitor.step(DomainEvent("e0", tick=0, tag="export"))
    monitor.step(TickEvent("e1", tick=1))
    verdict = monitor.step(TickEvent("e2", tick=2, observations={"approval-required": K3.T}))

    occ = monitor.occurrences["o1"]
    assert occ.status is OccurrenceStatus.BREACHED
    assert occ.effective_time == 1
    assert occ.discovery_time == 2
    assert occ.breach_reason == "RetroactiveActivationAfterDeadline"
    assert verdict.summary is Summary.VIOLATED


# -- S1.11: fully worked export/Sensitive/approval example ------------------


def test_s1_worked_example_export_sensitive_approval():
    monitor = SingleClauseMonitor(_spec(bound=1))
    monitor.set_initial("approval-required", K3.U)

    # Tick 10: export(f) -- guard unknown, ConditionalOpen with D=11.
    monitor.step(DomainEvent("e10", tick=10, tag="export"))
    occ = monitor.occurrences["o1"]
    assert occ.status is OccurrenceStatus.CONDITIONAL_OPEN
    assert occ.deadline == 11
    assert monitor.current_verdict().summary is Summary.UNKNOWN

    # Tick 11: approval(f), on time (11 == D), and reserved for the
    # occurrence while its guard is unknown. Boundary expiry then converts the
    # occurrence to ConditionalExpired, not a breach.
    monitor.step(DomainEvent("e11", tick=11, tag="approval"))
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
            "e12", tick=12, tag="Observation", observations={"approval-required": K3.T}
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


# -- The core clause reads no event data ------------------------------------
#
# The clause in force is `after export when approval-required require approval
# within D`. It declares no binding fields, so an event matches on its tag
# alone and its `fields` map is data the clause never reads. A guard is a
# ground proposition, fixed before the comparison begins.


def _core_spec(bound: int = 1) -> AfterClauseSpec:
    return AfterClauseSpec(
        clause_id="c1",
        trigger_tag="export",
        response_tag="approval",
        binding_fields=(),
        guard=GroundGuard("approval-required"),
        bound=bound,
        discharge=Linear(),
    )


def test_s1_a_response_discharges_the_duty_a_trigger_raised():
    monitor = SingleClauseMonitor(_core_spec())
    monitor.set_initial("approval-required", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export"))
    monitor.step(DomainEvent("e1", tick=1, tag="approval"))
    verdict = monitor.step(TerminalEvent("e2", tick=2, kind=Complete()))

    assert monitor.occurrences["o1"].status is OccurrenceStatus.DISCHARGED
    assert verdict.summary is Summary.SATISFIED


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({}, id="no-fields"),
        pytest.param({"x": "f"}, id="one-field"),
        pytest.param({"x": 123, "y": None}, id="values-of-any-type"),
    ],
)
def test_s1_event_fields_do_not_change_the_reading(fields):
    # The clause binds nothing, so the same tags give the same reading
    # whatever data the events carry.
    monitor = SingleClauseMonitor(_core_spec())
    monitor.set_initial("approval-required", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields=fields))
    monitor.step(DomainEvent("e1", tick=1, tag="approval", fields=fields))
    verdict = monitor.step(TerminalEvent("e2", tick=2, kind=Complete()))

    assert monitor.occurrences["o1"].status is OccurrenceStatus.DISCHARGED
    assert verdict.summary is Summary.SATISFIED


def test_s1_an_unmatched_tag_raises_no_duty():
    monitor = SingleClauseMonitor(_core_spec())
    monitor.set_initial("approval-required", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="publish", fields={"x": "f"}))
    verdict = monitor.step(TerminalEvent("e1", tick=1, kind=Complete()))

    assert monitor.occurrences == {}
    assert verdict.summary is Summary.SATISFIED


def test_s1_a_false_guard_makes_the_occurrence_inapplicable():
    """The trigger-and-guard rule: a false guard records the occurrence
    without making it a duty."""
    monitor = SingleClauseMonitor(_core_spec())
    monitor.set_initial("approval-required", K3.F)

    monitor.step(DomainEvent("e0", tick=0, tag="export"))
    verdict = monitor.step(TerminalEvent("e1", tick=1, kind=Complete()))

    statuses = {occurrence.status for occurrence in monitor.occurrences.values()}
    assert statuses == {OccurrenceStatus.INAPPLICABLE}
    assert verdict.summary is Summary.SATISFIED


def test_s1_a_true_guard_opens_a_duty():
    """The control for the test above."""
    monitor = SingleClauseMonitor(_core_spec())
    monitor.set_initial("approval-required", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export"))
    verdict = monitor.step(TerminalEvent("e1", tick=1, kind=Complete()))

    statuses = {occurrence.status for occurrence in monitor.occurrences.values()}
    assert statuses == {OccurrenceStatus.BREACHED}
    assert verdict.summary is Summary.VIOLATED


def test_s1_the_guard_key_does_not_follow_an_event_field():
    monitor = SingleClauseMonitor(
        AfterClauseSpec(
            clause_id="c1",
            trigger_tag="export",
            response_tag="approval",
            binding_fields=(),
            guard=GroundGuard("CountryAllowed(UK)"),
            bound=1,
            discharge=Linear(),
        )
    )
    monitor.set_initial("CountryAllowed(UK)", K3.T)
    monitor.set_initial("CountryAllowed(f)", K3.F)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    monitor.step(DomainEvent("e1", tick=1, tag="approval", fields={"x": "f"}))
    verdict = monitor.step(TerminalEvent("e2", tick=2, kind=Complete()))

    assert monitor.occurrences["o1"].status is OccurrenceStatus.DISCHARGED
    assert verdict.summary is Summary.SATISFIED


def test_s1_one_event_matching_both_tags_settles_the_duty_it_raised():
    monitor = SingleClauseMonitor(
        AfterClauseSpec(
            clause_id="c1",
            trigger_tag="export",
            response_tag="export",
            binding_fields=(),
            guard=GroundGuard("approval-required"),
            bound=1,
            discharge=Linear(),
        )
    )
    monitor.set_initial("approval-required", K3.T)

    verdict = monitor.step(DomainEvent("e0", tick=0, tag="export"))

    assert monitor.occurrences["o1"].status is OccurrenceStatus.DISCHARGED
    assert verdict.summary is Summary.UNKNOWN


def test_s1_guard_value_for_admits_only_the_empty_substitution():
    monitor = SingleClauseMonitor(_core_spec())
    monitor.set_initial("approval-required", K3.T)

    assert monitor.guard_value_for({}) is K3.T
    for substitution in ({"x": "f"}, {"x": 1}, {"y": "extra"}):
        with pytest.raises(InvalidBinding):
            monitor.guard_value_for(substitution)
    with pytest.raises(InvalidBinding):
        monitor.guard_value_for("not-a-mapping")


def _core_clause_fields() -> dict:
    """The clause form Appendix B fixes: no binding fields, a ground guard,
    linear discharge, and Breach as the only abort disposition."""
    return {
        "clause_id": "c1",
        "trigger_tag": "export",
        "response_tag": "approval",
        "binding_fields": (),
        "guard": GroundGuard("approval-required"),
        "bound": 1,
        "discharge": Linear(),
        "on_agent_abort": Breach(),
    }


def test_s1_after_clause_accepts_the_core_form():
    spec = AfterClauseSpec(**_core_clause_fields())

    assert spec.binding_fields == ()
    assert type(spec.guard) is GroundGuard
    assert type(spec.discharge) is Linear
    assert type(spec.on_agent_abort) is Breach


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        # "An AfterClause with non-empty binding_fields is not supported in
        # this core."
        pytest.param("binding_fields", ("x",), id="one-binding-field"),
        pytest.param("binding_fields", ("x", "y"), id="two-binding-fields"),
        pytest.param("binding_fields", ["x"], id="binding-fields-not-a-tuple"),
        pytest.param("binding_fields", (1,), id="binding-field-not-a-string"),
        pytest.param("guard", "approval-required", id="guard-not-a-guard"),
        pytest.param("guard", None, id="guard-missing"),
        pytest.param("discharge", object(), id="discharge-object"),
        pytest.param(
            "discharge",
            type("LinearSubclass", (Linear,), {})(),
            id="discharge-subclass",
        ),
        # "This core admits only on_agent_abort=Breach."
        pytest.param("on_agent_abort", object(), id="abort-object"),
        pytest.param(
            "on_agent_abort",
            type("BreachSubclass", (Breach,), {})(),
            id="abort-subclass",
        ),
    ],
)
def test_s1_after_clause_rejects_what_the_core_leaves_undefined(field_name, value):
    fields = _core_clause_fields()
    fields[field_name] = value

    with pytest.raises(InvalidAfterClauseSpec):
        AfterClauseSpec(**fields)


def test_s1_the_package_offers_no_construct_the_core_leaves_undefined():
    # A construct outside this core is reported as not implemented rather than
    # given a reading here, so the package does not publish one at all.
    for name in ("BoundUnaryGuard", "Broadcast", "Indeterminate", "WaiveIf"):
        assert not hasattr(contracts_module, name), name


def test_s1_ground_guard_rejects_an_empty_proposition_key():
    with pytest.raises(InvalidAfterClauseSpec):
        GroundGuard("")


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
