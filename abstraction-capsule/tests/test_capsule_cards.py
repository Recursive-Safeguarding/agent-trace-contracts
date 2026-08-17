"""Acceptance for this Python capsule: two candidate cards use one term format.

This non-normative capsule fixes the profile choices, represents both candidates
as ``StateCardTerm`` values, takes one nontrivial source trace through
transformation and continuation, and emits the specified comparison constructor.

Both candidates emit the same Python term type so that the profile's
continuation initializer can accept either one. Each candidate is a different
map into this profile-local format. The normative specification leaves ``h(u)``
abstract and does not define a general state-card grammar.

Card (a), truncation: drops the oldest events beyond a budget. Expected to be
rejected by a witness, because the dropped prefix held the live obligation.
Card (b), obligation-preserving: retains the open obligation record. Expected
to survive the declared bound with no witness.

Both must be valid constructions. The tests below check two conditions for
these candidates: candidate (a) must drop the trigger that opened the live
obligation, and candidate (b)'s term must not copy the source trace.
"""

from __future__ import annotations

import pytest
from rs_metalang_ref.events import DomainEvent
from rs_metalang_ref.obligations import OccurrenceStatus

from rs_capsule import cards
from rs_capsule.capsule import SOURCE_TRACE, build_source_prefix
from rs_capsule.cards import truncation_card
from rs_capsule.profile import OccurrenceRecord, StateCardTerm, build_profile


def _prefix():
    return build_source_prefix(build_profile(), SOURCE_TRACE)


def _open_occurrence(prefix):
    live = [
        occurrence
        for occurrence in prefix.monitor.occurrences.values()
        if occurrence.status is OccurrenceStatus.OPEN
    ]
    assert len(live) == 1, f"the declared source trace must leave exactly one open obligation, got {live}"
    return live[0]


def test_cards_module_exposes_only_open_obligation_card():
    assert (
        "open_obligation_card" in vars(cards),
        "obligation_preserving_card" in vars(cards),
    ) == (True, False)


def test_both_candidates_emit_the_one_declared_card_syntax():
    prefix = _prefix()

    for card in (truncation_card(budget=1), cards.open_obligation_card()):
        term = card(prefix)
        assert isinstance(term, StateCardTerm), f"{card.card_id} left the declared card syntax"


def test_every_candidate_carries_an_identifier():
    for card in (truncation_card(budget=1), cards.open_obligation_card()):
        assert isinstance(card.card_id, str)
        assert card.card_id.strip() != ""


def test_truncation_card_keeps_only_the_budgeted_window():
    prefix = _prefix()
    budget = 1

    term = truncation_card(budget=budget)(prefix)

    assert len(term.retained_events) == budget
    assert tuple(term.retained_events) == tuple(prefix.trace[-budget:]), (
        "a truncation card keeps the newest events; dropping the newest instead "
        "would be a different card"
    )


@pytest.mark.parametrize(
    "invalid_budget",
    (pytest.param(True, id="bool"), pytest.param(1.5, id="float")),
)
def test_truncation_card_rejects_a_non_integer_budget(invalid_budget):
    with pytest.raises(ValueError):
        truncation_card(budget=invalid_budget)


def test_truncation_card_genuinely_drops_the_live_obligation():
    """The validity condition on candidate (a). The card must be rejected
    because the dropped prefix held the live obligation, not because it was
    built to fail some other way.
    """
    prefix = _prefix()
    occurrence = _open_occurrence(prefix)

    term = truncation_card(budget=1)(prefix)
    retained_ticks = {event.tick for event in term.retained_events}

    assert occurrence.trigger_tick not in retained_ticks, (
        "the trigger that opened the live obligation survived truncation, so this "
        "candidate does not exercise the loss it is meant to demonstrate"
    )
    assert term.occurrence_table == (), "a truncation card retains events, not obligations"


def test_open_obligation_card_retains_the_open_obligation():
    prefix = _prefix()
    occurrence = _open_occurrence(prefix)

    term = cards.open_obligation_card()(prefix)

    assert term.occurrence_table, "candidate (b) must retain the obligation record"
    retained = {record.oid: record for record in term.occurrence_table}
    assert occurrence.oid in retained

    record = retained[occurrence.oid]
    assert record.deadline == occurrence.deadline
    assert record.trigger_tick == occurrence.trigger_tick


def test_open_obligation_card_is_not_a_copy_of_the_source_trace(walk_values):
    """The profile-local no-copy condition on candidate (b).

    This capsule declares ``current_tick`` and ``guard_value`` as its retained
    environment. Its candidate card term must not carry the original source
    trace as an undeclared extra input. The normative specification permits
    retention when a profile declares and costs it; this test checks only this
    capsule's chosen card boundary.

    Checked structurally: no interpreter Event instance and no source event
    identifier anywhere in the term.
    """
    from rs_metalang_ref.events import DomainEvent, TerminalEvent, TickEvent

    prefix = _prefix()
    term = cards.open_obligation_card()(prefix)

    assert term.retained_events == (), "candidate (b) abstracts the events away"

    values = list(walk_values(term))
    source_events = [
        v for v in values if isinstance(v, (DomainEvent, TickEvent, TerminalEvent))
    ]
    assert source_events == [], f"candidate (b) retains raw source events: {source_events}"

    source_ids = {e.event_id for e in SOURCE_TRACE if hasattr(e, "event_id")}
    retained_source_ids = {v for v in values if isinstance(v, str) and v in source_ids}
    assert retained_source_ids == set(), (
        f"candidate (b) retains source event identifiers: {retained_source_ids}"
    )


def test_the_two_candidates_are_actually_different_maps():
    prefix = _prefix()

    truncated = truncation_card(budget=1)(prefix)
    open_only = cards.open_obligation_card()(prefix)

    assert truncated != open_only
    assert truncation_card(budget=1).card_id != cards.open_obligation_card().card_id


def test_a_card_does_not_mutate_the_source_prefix():
    """Both card maps leave the source monitor's occurrences and tick unchanged."""
    prefix = _prefix()
    before = dict(prefix.monitor.current_verdict().occurrences)
    tick_before = prefix.monitor.tick

    truncation_card(budget=1)(prefix)
    cards.open_obligation_card()(prefix)

    assert dict(prefix.monitor.current_verdict().occurrences) == before
    assert prefix.monitor.tick == tick_before


def test_open_obligation_card_retains_only_open_records_from_a_mixed_prefix():
    profile = build_profile()
    prefix = build_source_prefix(
        profile,
        (
            DomainEvent(event_id="first-export", tick=0, tag="export"),
            DomainEvent(event_id="second-export", tick=3, tag="export"),
        ),
    )
    assert {
        occurrence.oid: occurrence.status
        for occurrence in prefix.monitor.occurrences.values()
    } == {
        "o1": OccurrenceStatus.BREACHED,
        "o2": OccurrenceStatus.OPEN,
    }

    card = cards.open_obligation_card()
    term = card(prefix)

    assert card.card_id == "obligation-table-v1"
    assert term.retained_events == ()
    assert term.occurrence_table == (
        OccurrenceRecord(
            oid="o2",
            trigger_tick=3,
            deadline=5,
        ),
    )
