"""Conditional response candidates and executable-fragment boundaries.

Most tests in this file state contracts from S1.3, "retroactive activation",
and S1.4, "response-token discipline":

* a conditional record contains only response tokens whose ticks are at most
  that occurrence's deadline;
* retroactive activation allocates a shared linear token in canonical
  occurrence order;
* activation allocates once for the whole configuration, so two tokens
  discharge two occurrences and neither occurrence takes both, and the outcome
  is the same whichever order the occurrences arrive in; and
* this conformance fragment rejects ``Broadcast(key)`` explicitly until its
  event-to-key and event-to-scope mappings are specified.

One further test checks the event-union boundary from
S1.1's typed events: this conformance fragment
rejects ``MalformedEvent`` explicitly instead of reading its missing tick and
raising ``AttributeError``.

The ordinary single-clause event path cannot produce the deadline inversion
needed to distinguish conditional canonical order from insertion order. That
test therefore installs a synthetic obligations-layer state directly in the
monitor's public ``occurrences`` store as white-box setup, then drives
resolution through public ``step(...)``. The manually installed state is not
claimed to be a valid public state or one the current fixed-bound construction
can generate from trigger events alone.
"""

from __future__ import annotations

import pytest

from rs_metalang_ref import monitor as monitor_module
from rs_metalang_ref.contracts import (
    AfterClauseSpec,
    BoundUnaryGuard,
    Broadcast,
    Linear,
)
from rs_metalang_ref.events import DomainEvent, MalformedEvent, TickEvent
from rs_metalang_ref.kleene import K3
from rs_metalang_ref.monitor import SingleClauseMonitor
from rs_metalang_ref.obligations import (
    Occurrence,
    OccurrenceStatus,
    ResponseToken,
    canonical_order_key,
    guards_become_true,
    register_response_token,
)
from rs_metalang_ref.verdict import Summary


def _spec(discharge) -> AfterClauseSpec:
    return AfterClauseSpec(
        clause_id="c1",
        trigger_tag="export",
        response_tag="approval",
        binding_fields=("x",),
        guard=BoundUnaryGuard("Sensitive", "x"),
        bound=5,
        discharge=discharge,
    )


def _conditional(oid: str, deadline: int, candidates=()) -> Occurrence:
    return Occurrence(
        oid=oid,
        clause_id="c1",
        substitution={"x": "f"},
        trigger_tick=0,
        deadline=deadline,
        guard_key="Sensitive(f)",
        status=OccurrenceStatus.CONDITIONAL_OPEN,
        candidates=list(candidates),
    )


@pytest.mark.parametrize(
    "explicit_oids",
    [
        pytest.param(frozenset({"o1"}), id="explicit"),
        pytest.param(frozenset(), id="generic"),
    ],
)
def test_late_response_is_not_retained_as_conditional_candidate(explicit_oids):
    """Both token-registration paths reject a response after the deadline."""
    occurrence = _conditional("o1", deadline=5)
    token = ResponseToken(
        rid="r6",
        response_tick=6,
        substitution={"x": "f"},
        explicit_oids=explicit_oids,
    )

    register_response_token(token, {occurrence.oid: occurrence})

    assert occurrence.candidates == []
    assert token.consumed is False


def test_response_before_occurrence_is_not_retained_for_future_trigger():
    """The event API does not attach an earlier response to a future occurrence."""
    monitor = SingleClauseMonitor(_spec(Linear()))
    monitor.set_initial("Sensitive(f)", K3.U)

    monitor.step(DomainEvent("r0", tick=0, tag="approval"))
    monitor.step(DomainEvent("e1", tick=1, tag="export", fields={"x": "f"}))

    assert monitor.occurrences["o1"].candidates == []


def test_simultaneous_conditional_activation_uses_canonical_order():
    """A shared linear token goes to the canonical conditional occurrence."""
    monitor = SingleClauseMonitor(_spec(Linear()))
    shared_token = ResponseToken(
        rid="r4",
        response_tick=4,
        substitution={"x": "f"},
        explicit_oids=frozenset({"o1", "o2"}),
    )
    later = _conditional("o1", deadline=9, candidates=(shared_token,))
    canonical = _conditional("o2", deadline=5, candidates=(shared_token,))
    monitor.occurrences = {later.oid: later, canonical.oid: canonical}

    expected = min(monitor.occurrences.values(), key=canonical_order_key).oid
    assert expected == "o2"

    monitor.step(
        TickEvent(
            "resolve4",
            tick=4,
            observations={"Sensitive(f)": K3.T},
        )
    )

    discharged = sorted(
        oid
        for oid, occurrence in monitor.occurrences.items()
        if occurrence.status is OccurrenceStatus.DISCHARGED
    )
    assert discharged == [expected]
    assert monitor.occurrences["o1"].status is OccurrenceStatus.OPEN
    assert monitor.occurrences["o2"].response is shared_token


@pytest.mark.parametrize(
    "arrival_order",
    [("o1", "o2"), ("o2", "o1")],
    ids=["canonical-first", "canonical-last"],
)
def test_one_token_two_occurrences_gives_one_discharge_and_one_breach(arrival_order):
    """Two occurrences share one token, and both guards become true late.

    The pass allocates the token to the canonical occurrence and breaches the
    other, whichever order the two occurrences arrive in. The deadline does not
    move, so the breach records tick 2 as the effective time and tick 5 as the
    discovery time.
    """
    monitor = SingleClauseMonitor(_spec(Linear()))
    token = ResponseToken(rid="r2", response_tick=2, substitution={"x": "f"})
    canonical = _conditional("o1", deadline=2, candidates=(token,))
    other = _conditional("o2", deadline=2, candidates=(token,))
    records = {"o1": canonical, "o2": other}
    monitor.occurrences = {oid: records[oid] for oid in arrival_order}

    verdict = monitor.step(
        TickEvent("resolve5", tick=5, observations={"Sensitive(f)": K3.T})
    )

    assert canonical.status is OccurrenceStatus.DISCHARGED
    assert canonical.response is token
    assert other.status is OccurrenceStatus.BREACHED
    assert other.response is None
    assert other.breach_reason == "RetroactiveActivationAfterDeadline"
    assert other.effective_time == 2
    assert other.discovery_time == 5
    assert token.consumed is True
    assert verdict.summary is Summary.VIOLATED


@pytest.mark.parametrize(
    "arrival_order",
    [("o1", "o2"), ("o2", "o1")],
    ids=["canonical-first", "canonical-last"],
)
def test_two_tokens_two_occurrences_discharge_one_token_each(arrival_order):
    """Two occurrences share two tokens, and both guards become true late.

    Each occurrence receives at most one linear token in the pass, so the
    earlier token goes to the canonical occurrence and the later token goes to
    the other. Neither occurrence takes both.
    """
    monitor = SingleClauseMonitor(_spec(Linear()))
    earlier = ResponseToken(rid="r1", response_tick=1, substitution={"x": "f"})
    later = ResponseToken(rid="r2", response_tick=2, substitution={"x": "f"})
    canonical = _conditional("o1", deadline=2, candidates=(earlier, later))
    other = _conditional("o2", deadline=2, candidates=(earlier, later))
    records = {"o1": canonical, "o2": other}
    monitor.occurrences = {oid: records[oid] for oid in arrival_order}

    verdict = monitor.step(
        TickEvent("resolve5", tick=5, observations={"Sensitive(f)": K3.T})
    )

    assert canonical.status is OccurrenceStatus.DISCHARGED
    assert canonical.response is earlier
    assert other.status is OccurrenceStatus.DISCHARGED
    assert other.response is later
    assert verdict.summary is Summary.UNKNOWN


@pytest.mark.parametrize(
    "arrival_order",
    [("o1", "o2"), ("o2", "o1")],
    ids=["canonical-first", "canonical-last"],
)
def test_activation_pass_ignores_the_order_its_caller_supplies(arrival_order):
    """The pass orders its own occurrences, so the caller's order cannot decide."""
    token = ResponseToken(rid="r2", response_tick=2, substitution={"x": "f"})
    canonical = _conditional("o1", deadline=2, candidates=(token,))
    other = _conditional("o2", deadline=2, candidates=(token,))
    records = {"o1": canonical, "o2": other}

    guards_become_true([records[oid] for oid in arrival_order], 5)

    assert canonical.status is OccurrenceStatus.DISCHARGED
    assert canonical.response is token
    assert other.status is OccurrenceStatus.BREACHED
    assert other.response is None


def test_broadcast_is_rejected_at_monitor_construction():
    """Unsupported broadcast mode fails before any event can use linear rules."""
    error_type = getattr(monitor_module, "UnsupportedDischargeModeError", None)
    assert error_type is not None, (
        "monitor must expose UnsupportedDischargeModeError for discharge modes "
        "outside its conformance fragment"
    )

    with pytest.raises(error_type, match="Broadcast"):
        SingleClauseMonitor(_spec(Broadcast("batch")))


def test_malformed_event_is_rejected_at_monitor_step_boundary():
    """Unsupported malformed input fails before monitor state can change."""
    error_type = getattr(monitor_module, "UnsupportedEventTypeError", None)
    assert error_type is not None, (
        "monitor must expose UnsupportedEventTypeError for event forms outside "
        "its conformance fragment"
    )

    monitor = SingleClauseMonitor(_spec(Linear()))
    with pytest.raises(error_type, match="MalformedEvent"):
        monitor.step(MalformedEvent(raw_digest="sha256:deadbeef", error_code="E-PARSE"))

    assert monitor.tick == -1
    assert monitor.occurrences == {}


def test_linear_monitor_construction_remains_supported():
    """The exclusion boundary leaves the implemented linear mode available."""
    monitor = SingleClauseMonitor(_spec(Linear()))

    assert isinstance(monitor.spec.discharge, Linear)
