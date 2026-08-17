"""S5: the attempted entry of a DisabledSink is a canonical continuation LABEL.

The totalised transition system and residual definitions build the disabled sink out of the
continuation alphabet, not out of raw trace records: the totalised transition
system fixes `attempted = a` where `a` is a label of `Sigma`, and the residual
definition's `TotalObservation_C` carries `Disabled(attempted = label, ...)`. The alphabet
itself is fixed in the totalised transition system's first paragraph: "The continuation alphabet
`Sigma` contains domain-event labels, tick labels, all five terminal labels,
and action-attempt labels." Raw event identifiers and absolute timestamps are
not in that alphabet, and the canonical-event-identifiers rule pins the
consequence: structurally identical traces whose events are numbered
differently are the same to the checker.

These tests exercise the residual itself. After a terminal boundary, the disabled
sink must store the canonical continuation label, not a raw event record.
"""

from __future__ import annotations

import rs_metalang_ref.residual as residual_module
from rs_metalang_ref.contracts import AfterClauseSpec, BoundUnaryGuard, Linear
from rs_metalang_ref.events import Complete, DomainEvent, TerminalEvent, TickEvent
from rs_metalang_ref.kleene import K3
from rs_metalang_ref.monitor import SingleClauseMonitor
from rs_metalang_ref.residual import DisabledReason, monitor_residual


def _spec(bound: int = 2) -> AfterClauseSpec:
    return AfterClauseSpec(
        clause_id="c1",
        trigger_tag="export",
        response_tag="approval",
        binding_fields=("x",),
        guard=BoundUnaryGuard("Sensitive", "x"),
        bound=bound,
        discharge=Linear(),
    )


def _halted_prefix(terminal_tick: int = 3, event_ids=("e0", "t0")):
    """A prefix that has crossed a terminal boundary.

    Any further continuation label is therefore disabled with reason
    `AfterTerminal` (the totalised transition-system rule: "a non-terminal label after a terminal
    boundary has reason `AfterTerminal`"), which is the branch that records an
    attempted label in the sink.
    """
    monitor = SingleClauseMonitor(_spec())
    monitor.set_initial("Sensitive(f)", K3.T)
    monitor.step(DomainEvent(event_ids[0], tick=0, tag="export", fields={"x": "f"}))
    monitor.step(TerminalEvent(event_ids[1], tick=terminal_tick, kind=Complete()))
    return monitor


def _sink(observation) -> residual_module.Disabled:
    status = observation.continuation_status
    assert isinstance(status, residual_module.Disabled), f"expected Disabled, got {status!r}"
    assert status.reason is DisabledReason.AFTER_TERMINAL
    return status


def test_tick_label_is_canonical_under_event_identifier_renaming():
    """the relevant acceptance-test rule, acceptance test `canonical-event-identifiers`:
    `left_trace_event_ids: [17, 22]`, `right_trace_event_ids: [203, 991]`,
    `structural_events: identical`, `expected: canonical_observations_equal:
    true`.

    Deciding sentence, the contract-observation rule:
    "Canonicalisation prevents two structurally identical runs from being
    reported as different only because one numbered its events differently."

    Two continuations that advance the clock by the same amount from the same
    prefix and differ only in the raw `event_id` of the tick event are the same
    continuation, so their residuals are equal.
    """
    left = monitor_residual(_halted_prefix(), [TickEvent("17", tick=4)])
    right = monitor_residual(_halted_prefix(), [TickEvent("991", tick=4)])

    assert _sink(left).attempted == _sink(right).attempted
    assert left == right


def test_attempted_label_does_not_expose_event_identity_or_absolute_time():
    """the totalised transition-system rule, first paragraph: "The continuation alphabet `Sigma`
    contains domain-event labels, tick labels, all five terminal labels, and
    action-attempt labels. Time inside a continuation is represented by
    relative tick advances, so that concatenation does not depend on absolute
    timestamps."

    An `event_id` is not a label of `Sigma`, and an absolute timestamp is
    exactly what a continuation is defined not to depend on. A sink that stores
    either one makes residual equality turn on how the harness numbered and
    clocked the trace, which is what the canonical-event-identifiers rule's acceptance test forbids.

    The prefix here is clocked at a large absolute tick and the continuation
    advances it by one, so a label that named the relative advance would carry
    the digit `1` and pass; only a label carrying the raw record fails.
    """
    prefix = _halted_prefix(terminal_tick=1_000_000, event_ids=("e-4b7c9f", "t-4b7c9f"))
    attempt = TickEvent("tick-id-4b7c9f", tick=1)

    attempted = _sink(monitor_residual(prefix, [attempt])).attempted
    rendered = repr(attempted)

    assert "4b7c9f" not in rendered, f"attempted label exposes an event_id: {rendered}"
    assert "1000001" not in rendered, f"attempted label exposes an absolute tick: {rendered}"
    assert "1000000" not in rendered, f"attempted label exposes an absolute tick: {rendered}"


def test_domain_event_label_keeps_its_argument_and_drops_its_identifier():
    """the relevant acceptance-test rule renders an attempted domain-event label as
    `attempted: export(f)` (acceptance test
    `disabled-sink-retains-full-contract-observation`), and the specification's worked
    example distinguishes `export(f)` from `approval(f)`.

    So the argument is part of the label and the identifier is not. This checks
    both directions at once: canonicalising the identifier away must not also
    collapse the argument.
    """
    def attempted_for(event_id: str, argument: str):
        prefix = _halted_prefix()
        event = DomainEvent(event_id, tick=4, tag="export", fields={"x": argument})
        return _sink(monitor_residual(prefix, [event])).attempted

    assert attempted_for("17", "f") == attempted_for("991", "f")
    assert attempted_for("17", "f") != attempted_for("17", "g")


def test_distinct_labels_still_distinguish_residuals():
    """Positive control: distinct labels still distinguish residuals.

    the totalised transition-system rule: "Totalisation is what makes the
    enabled-action domain observable. Two prefixes that differ only in which
    actions are available are distinguished by attempting an action that one
    enables and the other does not."

    A canonicalisation that merged every attempted label into one value would
    destroy that observability. Labels drawn from different parts of `Sigma`
    (a domain-event label, a tick label, a terminal label) stay apart.
    """
    domain = _sink(
        monitor_residual(_halted_prefix(), [DomainEvent("a", tick=4, tag="export", fields={"x": "f"})])
    ).attempted
    tick = _sink(monitor_residual(_halted_prefix(), [TickEvent("b", tick=4)])).attempted
    terminal = _sink(
        monitor_residual(_halted_prefix(), [TerminalEvent("c", tick=4, kind=Complete())])
    ).attempted

    assert domain != tick
    assert tick != terminal
    assert domain != terminal
