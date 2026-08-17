"""Property tests for linear response allocation.

The expected winner is computed directly from the specification tuple
``(deadline, trigger_tick, clause_id, oid)``. These tests do not call the
production ``canonical_order_key`` helper as their oracle.

One test below is a white-box Python optimization check over the occurrence
store. It is an implementation-specific optimisation check, not a conformance
property stated by S1.4's response-token discipline.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rs_metalang_ref import obligations as obligations_module
from rs_metalang_ref.contracts import AfterClauseSpec, BoundUnaryGuard, Linear
from rs_metalang_ref.events import TickEvent
from rs_metalang_ref.kleene import K3
from rs_metalang_ref.monitor import SingleClauseMonitor
from rs_metalang_ref.obligations import (
    Occurrence,
    OccurrenceStatus,
    ResponseToken,
    register_response_token,
)

_OIDS = tuple(f"o{index}" for index in range(10))
_MISSING_OIDS = tuple(f"missing{index}" for index in range(3))
_CLAUSE_IDS = ("c0", "c1", "c2")
_GENERATED_STATUSES = (
    OccurrenceStatus.OPEN,
    OccurrenceStatus.CONDITIONAL_OPEN,
    OccurrenceStatus.CONDITIONAL_EXPIRED,
    OccurrenceStatus.DISCHARGED,
    OccurrenceStatus.INAPPLICABLE,
    OccurrenceStatus.BREACHED,
    OccurrenceStatus.WAIVED,
    OccurrenceStatus.UNKNOWN_FINAL,
)
_PROPERTY_SETTINGS = settings(max_examples=120, deadline=None, database=None)


def _occurrence(
    oid: str,
    *,
    clause_id: str,
    trigger_tick: int,
    deadline: int,
    status: OccurrenceStatus,
    guard_key: str = "Sensitive(f)",
) -> Occurrence:
    return Occurrence(
        oid=oid,
        clause_id=clause_id,
        substitution={"x": "f"},
        trigger_tick=trigger_tick,
        deadline=deadline,
        guard_key=guard_key,
        status=status,
    )


@st.composite
def _allocation_case(draw, *, explicit: bool):
    count = draw(st.integers(min_value=1, max_value=7))
    oids = draw(
        st.lists(
            st.sampled_from(_OIDS),
            min_size=count,
            max_size=count,
            unique=True,
        )
    )

    records = {}
    for oid in oids:
        trigger_tick = draw(st.integers(min_value=0, max_value=8))
        bound = draw(st.integers(min_value=0, max_value=8))
        records[oid] = _occurrence(
            oid,
            clause_id=draw(st.sampled_from(_CLAUSE_IDS)),
            trigger_tick=trigger_tick,
            deadline=trigger_tick + bound,
            status=draw(st.sampled_from(_GENERATED_STATUSES)),
        )

    insertion_order = draw(st.permutations(oids))
    occurrences = {oid: records[oid] for oid in insertion_order}
    response_tick = draw(st.integers(min_value=0, max_value=16))

    if explicit:
        targets = draw(
            st.sets(
                st.sampled_from(tuple(oids) + _MISSING_OIDS),
                min_size=1,
                max_size=count + len(_MISSING_OIDS),
            )
        )
    else:
        targets = set()

    token = ResponseToken(
        rid="response",
        response_tick=response_tick,
        substitution={"x": "f"},
        explicit_oids=frozenset(targets),
    )
    return occurrences, token


def _literal_specification_key(occurrence: Occurrence):
    return (
        occurrence.deadline,
        occurrence.trigger_tick,
        occurrence.clause_id,
        occurrence.oid,
    )


def _newly_discharged(before, occurrences):
    return sorted(
        oid
        for oid, occurrence in occurrences.items()
        if before[oid] is not OccurrenceStatus.DISCHARGED
        and occurrence.status is OccurrenceStatus.DISCHARGED
    )


@_PROPERTY_SETTINGS
@given(_allocation_case(explicit=True))
def test_explicit_linear_registration_uses_literal_specification_order(case):
    occurrences, token = case
    before = {oid: occurrence.status for oid, occurrence in occurrences.items()}
    eligible = [
        occurrence
        for oid, occurrence in occurrences.items()
        if oid in token.explicit_oids
        and occurrence.status is OccurrenceStatus.OPEN
        and token.response_tick <= occurrence.deadline
    ]
    expected = min(eligible, key=_literal_specification_key).oid if eligible else None

    register_response_token(token, occurrences)

    newly_discharged = _newly_discharged(before, occurrences)
    assert len(newly_discharged) <= 1
    assert newly_discharged == ([] if expected is None else [expected])
    assert all(oid in token.explicit_oids for oid in newly_discharged)
    assert token.consumed is (expected is not None)


@_PROPERTY_SETTINGS
@given(_allocation_case(explicit=False))
def test_generic_linear_registration_uses_literal_specification_order(case):
    occurrences, token = case
    before = {oid: occurrence.status for oid, occurrence in occurrences.items()}
    eligible = [
        occurrence
        for occurrence in occurrences.values()
        if occurrence.status is OccurrenceStatus.OPEN
        and token.response_tick <= occurrence.deadline
    ]
    expected = min(eligible, key=_literal_specification_key).oid if eligible else None

    register_response_token(token, occurrences)

    newly_discharged = _newly_discharged(before, occurrences)
    assert len(newly_discharged) <= 1
    assert newly_discharged == ([] if expected is None else [expected])
    assert token.consumed is (expected is not None)


def _allocation_snapshot(occurrences):
    return {
        oid: (
            occurrence.status,
            None if occurrence.response is None else occurrence.response.rid,
            tuple(candidate.rid for candidate in occurrence.candidates),
        )
        for oid, occurrence in occurrences.items()
    }


@pytest.mark.parametrize("explicit", [False, True], ids=["generic", "explicit"])
@pytest.mark.parametrize(
    "remaining_status",
    [OccurrenceStatus.OPEN, OccurrenceStatus.CONDITIONAL_OPEN],
    ids=["second-open", "conditional-candidate"],
)
def test_consumed_token_registration_is_idempotent(explicit, remaining_status):
    first = _occurrence(
        "o1",
        clause_id="c1",
        trigger_tick=0,
        deadline=5,
        status=OccurrenceStatus.OPEN,
    )
    remaining = _occurrence(
        "o2",
        clause_id="c1",
        trigger_tick=1,
        deadline=6,
        status=remaining_status,
    )
    occurrences = {first.oid: first, remaining.oid: remaining}
    token = ResponseToken(
        rid="r2",
        response_tick=2,
        substitution={"x": "f"},
        explicit_oids=frozenset(occurrences) if explicit else frozenset(),
    )

    register_response_token(token, occurrences)
    assert token.consumed is True
    after_first_registration = _allocation_snapshot(occurrences)

    register_response_token(token, occurrences)

    assert _allocation_snapshot(occurrences) == after_first_registration


def test_white_box_no_sort_without_eligible_occurrences(
    monkeypatch,
):
    monitor = SingleClauseMonitor(
        AfterClauseSpec(
            clause_id="c1",
            trigger_tag="export",
            response_tag="approval",
            binding_fields=("x",),
            guard=BoundUnaryGuard("Sensitive", "x"),
            bound=5,
            discharge=Linear(),
        )
    )
    monitor.set_initial("Sensitive(f)", K3.T)
    settled_statuses = (
        OccurrenceStatus.DISCHARGED,
        OccurrenceStatus.INAPPLICABLE,
        OccurrenceStatus.BREACHED,
        OccurrenceStatus.WAIVED,
        OccurrenceStatus.UNKNOWN_FINAL,
    )
    for index in range(1_000):
        if index % 6 == 5:
            status = OccurrenceStatus.CONDITIONAL_OPEN
            guard_key = "OtherGuard"
        else:
            status = settled_statuses[index % len(settled_statuses)]
            guard_key = "Sensitive(f)"
        oid = f"stored-{index:04d}"
        monitor.occurrences[oid] = _occurrence(
            oid,
            clause_id="c1",
            trigger_tick=index,
            deadline=2_000,
            status=status,
            guard_key=guard_key,
        )

    key_calls = 0

    def counted_key(occurrence):
        nonlocal key_calls
        key_calls += 1
        return _literal_specification_key(occurrence)

    monkeypatch.setattr(obligations_module, "canonical_order_key", counted_key)

    monitor.step(TickEvent("tick0", tick=0))

    assert key_calls == 0
