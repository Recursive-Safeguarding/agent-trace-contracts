"""Acceptance: the capsule reports scoped agreement or rejection.

A bounded result cannot establish profile-level operational adequacy. The
public seam checks the current structural contract: result constructors,
declared scope, witnesses, and firewall behaviour.
"""

from __future__ import annotations

import dataclasses

import pytest
from rs_capsule.capsule import SOURCE_TRACE, run_capsule
from rs_capsule.cards import open_obligation_card, truncation_card
from rs_capsule.profile import build_profile
from rs_metalang_ref.firewall import FirewallViolation, require_typecheck
from rs_metalang_ref.residual import Distinguished, NoWitnessWithinBound


def _summaries():
    profile = build_profile()
    return profile, [
        run_capsule(profile, truncation_card(budget=1), SOURCE_TRACE),
        run_capsule(profile, open_obligation_card(), SOURCE_TRACE),
    ]


@pytest.mark.parametrize("identifier_field", ("profile_id", "card_id"))
def test_identifiers_remain_non_empty(identifier_field):
    """Profile and card identifiers must be non-empty strings."""
    profile = build_profile()
    card = truncation_card(budget=1)
    if identifier_field == "profile_id":
        profile = dataclasses.replace(profile, profile_id=" ")
    else:
        card = dataclasses.replace(card, card_id=" ")

    with pytest.raises(
        ValueError, match=rf"{identifier_field} must be a non-empty string"
    ):
        run_capsule(profile, card, SOURCE_TRACE)


def test_bounded_result_cannot_discharge_operational_adequacy():
    """A scoped bounded search cannot establish profile-level adequacy."""
    profile = build_profile()
    result = run_capsule(profile, open_obligation_card(), SOURCE_TRACE)

    assert isinstance(result.comparison, NoWitnessWithinBound)
    with pytest.raises(FirewallViolation) as excinfo:
        require_typecheck(
            type(result.comparison).__name__,
            "ProfileRelativeOperationalAdequacy",
        )
    assert excinfo.value.code == "E-PROOF-FIREWALL"


def test_summaries_carry_their_scope():
    """Scoped agreement or rejection: every summary names the profile, the
    candidate, and the bound it is relative to.
    """
    profile, results = _summaries()

    for result in results:
        summary = result.summary
        assert profile.profile_id in summary
        assert result.card_id in summary
        assert str(profile.comparison_bound.continuation_length) in summary
        assert "within the declared" in summary.lower(), (
            f"unscoped summary: {summary!r}"
        )


def test_rejection_summary_reports_the_witness():
    """A rejection summary must include enough information to replay the witness."""
    profile = build_profile()
    result = run_capsule(profile, truncation_card(budget=1), SOURCE_TRACE)

    assert isinstance(result.comparison, Distinguished)
    for label in result.comparison.witness:
        assert label in result.summary, (
            f"witness label {label!r} missing from {result.summary!r}"
        )


def test_no_witness_summary_states_exhaustion_not_absence_of_testing():
    """the result-algebra constructors and closed vocabulary: reporting an exhaustive
    negative as Untested erases the fact that a search ran, and reporting a
    partial search as NoWitnessWithinBound falsely claims exhaustion. The
    summary must say which one this was.
    """
    profile = build_profile()
    result = run_capsule(profile, open_obligation_card(), SOURCE_TRACE)

    assert isinstance(result.comparison, NoWitnessWithinBound)
    assert "exhaustive" in result.summary.lower()


def test_the_two_outcomes_do_not_read_alike():
    _profile, results = _summaries()
    rejected, survived = results

    assert rejected.summary != survived.summary
    assert isinstance(rejected.comparison, Distinguished)
    assert isinstance(survived.comparison, NoWitnessWithinBound)
