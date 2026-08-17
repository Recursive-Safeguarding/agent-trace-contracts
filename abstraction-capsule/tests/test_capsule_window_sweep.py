"""Acceptance: the truncation-card comparison result depends on the window budget.

At the fixed profile, the budget-1 tail card is distinguished from the source
while the budget-2 and budget-3 cards are not: the retained export event
replays with its original tick, so one extra retained event re-derives the
open duty and its deadline. These tests pin the tail-window sweep the package documents.
"""

from __future__ import annotations

import pytest
from rs_capsule.capsule import SOURCE_TRACE, run_capsule
from rs_capsule.cards import open_obligation_card, truncation_card
from rs_capsule.profile import build_profile
from rs_metalang_ref.residual import Distinguished, NoWitnessWithinBound


def _run(card):
    return run_capsule(build_profile(), card, SOURCE_TRACE)


def test_budget_one_tail_card_is_distinguished():
    """At a window of one the tail card loses the duty and is separated."""
    result = _run(truncation_card(budget=1))

    assert isinstance(result.comparison, Distinguished)
    assert result.comparison.witness, "a separating continuation must be reported"


@pytest.mark.parametrize("budget", (2, 3))
def test_wider_tail_cards_are_not_distinguished(budget):
    """From a window of two the tail card retains the event that creates the duty."""
    result = _run(truncation_card(budget=budget))

    assert isinstance(result.comparison, NoWitnessWithinBound)


def test_obligation_card_matches_wider_tail_cards():
    """The obligation card's result is invariant where wider tail cards agree."""
    obligation = _run(open_obligation_card())
    wider_tail = _run(truncation_card(budget=2))

    assert isinstance(obligation.comparison, NoWitnessWithinBound)
    assert type(obligation.comparison) is type(wider_tail.comparison)
