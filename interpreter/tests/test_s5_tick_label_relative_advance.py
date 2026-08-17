"""S5: a tick label is a relative tick advance, and a disabled label carries
no payload the transition never applies.

Two axes, both fixed by the first paragraph of the totalised-transition-system
rule, quoted here once in full because every test
below cites part of it:

    "The continuation alphabet `Sigma` contains domain-event labels, tick
    labels, all five terminal labels, and action-attempt labels. Time inside a
    continuation is represented by relative tick advances, so that
    concatenation does not depend on absolute timestamps."

The traces, prefixes, and continuations rule states the same rule from the
trace side: "Time inside a continuation is represented by relative tick
advances, so that a continuation can be appended to any prefix without
depending on absolute timestamps."

Read together with the totalised transition system's `attempted = a`, where `a` is a label of
`Sigma`, this says a tick label represents an advance and nothing else. The
advance is the content: two continuations that move the clock by different
amounts are two different labels, and two that move it by the same amount from
differently clocked prefixes are one label. A projection that keeps the
absolute tick fails the second half; a projection that keeps no time at all
fails the first.

The observation map is the second axis. An attempted label is built only for an
event that the monitor never steps, so its bundled observation updates are
never applied. the typed-events rule: "Observation updates attached to an event
are bundled into that event and are applied before that event's contract
checks." the totalised transition-system rule, on the disabled label: "The disabled label is not
passed to the underlying contract transition and does not create a trigger
occurrence, response token, evidence item, clause check, terminal conversion,
or contract certificate. Its only new observable content is the disabled
transition record carried alongside the frozen contract observation." An update
that is never applied has no semantic effect, so making residual equality turn
on it reports a distinction the transition system does not have. The residual
definition requires the opposite: "Equality of total observations is structural equality
after the canonicalisation required there."

The path an observation map does take into the residual is the monitor state:
an event that is applied changes the clause lifecycle, and the sink then
freezes that. `test_observations_reach_the_residual_when_they_are_applied`
pins that direction so the canonicalisation above cannot be over-applied.

These tests exercise `monitor_residual` through its public surface only. In
continuation position, the tick field carries the relative advance. The tests
do not prescribe how the residual boundary converts that label into the
absolute event consumed by the monitor.
"""

from __future__ import annotations

import pytest

import rs_metalang_ref.residual as residual_module
from rs_metalang_ref.contracts import AfterClauseSpec, BoundUnaryGuard, Linear
from rs_metalang_ref.events import (
    Complete,
    DomainEvent,
    MalformedEvent,
    TerminalEvent,
    TickEvent,
)
from rs_metalang_ref.kleene import K3
from rs_metalang_ref.monitor import SingleClauseMonitor
from rs_metalang_ref.residual import DisabledReason, monitor_residual
from rs_metalang_ref.verdict import Mode, Summary


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


def _halted_prefix(terminal_tick: int = 3) -> SingleClauseMonitor:
    """A prefix that has crossed a terminal boundary, so the next label is
    disabled with reason `AfterTerminal` (the totalised transition-system rule: "a non-terminal
    label after a terminal boundary has reason `AfterTerminal`").

    The trigger fires at tick 0 with a bound of 2, so the obligation is already
    breached at expiry for every `terminal_tick` above 2. The frozen
    observation is therefore the same object whatever the terminal tick is,
    which is what lets the tests below vary the absolute clock and attribute
    any residual difference to the label alone.
    """
    monitor = SingleClauseMonitor(_spec())
    monitor.set_initial("Sensitive(f)", K3.T)
    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    monitor.step(TerminalEvent("t0", tick=terminal_tick, kind=Complete()))
    return monitor


def _faulted_prefix(fault_tick: int) -> SingleClauseMonitor:
    """A prefix whose monitor is faulted, so the next label is disabled with
    reason `AfterFault` (the totalised transition-system rule: "a label after a monitor fault has
    reason `AfterFault`").

    The contradiction is the `InconsistentObservation` path of the three-valued
    epistemic layer. No trigger fires here, so the frozen observation carries no
    occurrence and is again independent of the absolute clock.
    """
    monitor = SingleClauseMonitor(_spec())
    monitor.set_initial("Sensitive(f)", K3.T)
    monitor.step(TickEvent("f0", tick=fault_tick, observations={"Sensitive(f)": K3.F}))
    return monitor


def _running_prefix() -> SingleClauseMonitor:
    """A prefix still running, with one trigger occurrence whose guard is
    unknown. A continuation event here is applied, so its observation map
    resolves the guard."""
    monitor = SingleClauseMonitor(_spec(bound=10))
    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    return monitor


def _running_prefix_at_tick(prefix_tick: int, bound: int) -> SingleClauseMonitor:
    """A running prefix with one obligation due `bound` ticks after its origin.

    The prefixes used by the origin-independence test differ only in their
    absolute clock. Their contract structure and relative remaining deadline
    are the same.
    """
    monitor = SingleClauseMonitor(_spec(bound=bound))
    monitor.set_initial("Sensitive(f)", K3.T)
    monitor.step(DomainEvent("e0", tick=prefix_tick, tag="export", fields={"x": "f"}))
    return monitor


def _sink(
    observation,
    reason: DisabledReason = DisabledReason.AFTER_TERMINAL,
) -> residual_module.Disabled:
    status = observation.continuation_status
    assert isinstance(status, residual_module.Disabled), f"expected Disabled, got {status!r}"
    assert status.reason is reason
    return status


# -- the tick label is the relative advance -----------------------


def test_tick_label_distinguishes_different_relative_advances():
    """the totalised transition-system rule: "Time inside a continuation is represented by
    relative tick advances", which also fixes `attempted = a` where `a` is a
    label of `Sigma`, whose members include tick labels.

    A continuation that advances the clock by one and a continuation that
    advances it by six are two different labels of `Sigma`. A projection that
    sends both to the same value is not the label the sink is specified to
    store, and the residual then reports two different continuations as one.

    Both continuations run from the same prefix, so the frozen contract
    observation is identical on both sides and the label is the only thing that
    can distinguish the residuals.
    """
    advance_one = monitor_residual(_halted_prefix(terminal_tick=3), [TickEvent("a", tick=1)])
    advance_six = monitor_residual(_halted_prefix(terminal_tick=3), [TickEvent("a", tick=6)])

    assert advance_one.contract_observation == advance_six.contract_observation, (
        "prefixes must agree, or this test is not about the label"
    )

    assert _sink(advance_one).attempted != _sink(advance_six).attempted
    assert advance_one != advance_six


def test_tick_label_is_invariant_under_the_absolute_clock_of_the_prefix():
    """The other half of the same sentence: "so that concatenation does not
    depend on absolute timestamps" (the totalised transition-system rule), and the traces, prefixes, and continuations rule:
    "so that a continuation can be appended to any prefix without depending on
    absolute timestamps."

    The same advance of six, once from a prefix clocked at 3 and once from a
    prefix clocked at a million. One continuation, one label, one residual.
    """
    from_small_clock = monitor_residual(_halted_prefix(terminal_tick=3), [TickEvent("a", tick=6)])
    from_large_clock = monitor_residual(
        _halted_prefix(terminal_tick=1_000_000), [TickEvent("b", tick=6)]
    )

    assert _sink(from_small_clock).attempted == _sink(from_large_clock).attempted
    assert from_small_clock == from_large_clock


def test_running_tick_advance_is_independent_of_the_prefix_origin():
    """A +1 tick label advances running prefixes at ticks 0 and 100.

    the totalised transition-system rule makes continuation time relative so that concatenation
    does not depend on absolute timestamps. Both prefixes have one obligation
    due after one more tick. Replaying the same +1 label must therefore reach
    ticks 1 and 101, expire the corresponding obligation, and remain Running
    on both sides. The public residual exposes the resulting modes and clause
    lifecycles; it deliberately does not expose the internal absolute clock.
    """
    left_prefix = _running_prefix_at_tick(0, bound=1)
    right_prefix = _running_prefix_at_tick(100, bound=1)
    continuation = (TickEvent("advance-one", tick=1),)

    assert (left_prefix.tick, right_prefix.tick) == (0, 100)

    left = monitor_residual(left_prefix, continuation)
    right = monitor_residual(right_prefix, continuation)

    assert (
        left.contract_observation.mode,
        right.contract_observation.mode,
    ) == (Mode.RUNNING, Mode.RUNNING)
    assert left.contract_observation.occurrences == {"o1": "Breached"}
    assert right.contract_observation.occurrences == {"o1": "Breached"}
    assert left.continuation_status == residual_module.NoDisabledLabel()
    assert right.continuation_status == residual_module.NoDisabledLabel()
    assert left == right


def test_running_six_tick_advance_is_independent_of_prefix_origin():
    """A +6 tick label advances running prefixes at ticks 0 and 1,000,000.

    the typed-trace and contract-adequacy definitions make the label a
    relative advance that can be appended to any prefix. The required next
    absolute ticks are therefore 6 and 1,000,006. Both obligations are due
    after six ticks, so the same +6 label must breach both while both monitors
    remain Running. The six-tick deadline also makes a one-tick substitution
    observably wrong through the public residual.
    """
    left_prefix = _running_prefix_at_tick(0, bound=6)
    right_prefix = _running_prefix_at_tick(1_000_000, bound=6)
    continuation = (TickEvent("advance-six", tick=6),)

    assert (left_prefix.tick, right_prefix.tick) == (0, 1_000_000)

    left = monitor_residual(left_prefix, continuation)
    right = monitor_residual(right_prefix, continuation)

    assert (
        left.contract_observation.mode,
        right.contract_observation.mode,
    ) == (Mode.RUNNING, Mode.RUNNING)
    assert left.contract_observation.occurrences == {"o1": "Breached"}
    assert right.contract_observation.occurrences == {"o1": "Breached"}
    assert left.continuation_status == residual_module.NoDisabledLabel()
    assert right.continuation_status == residual_module.NoDisabledLabel()
    assert left == right


def test_positive_two_tick_continuation_satisfies_shift_identity():
    """Replaying [+2, +37] equals replaying [+37] after the +2 prefix.

    the typed-trace and contract-adequacy definitions make each Tick value a
    relative advance in a continuation word. From origins 0 and 100, the joined
    word must therefore visit ticks 2 then 39, and 102 then 139. Each prefix has
    an obligation due after 39 ticks, so the final tick must breach it.

    The public residual does not expose its final internal clock. The deadline
    makes the required final tick observable: rebasing +37 against the original
    origin reaches only ticks 37 and 137 and leaves the occurrence open.
    """
    outcomes = {}
    for origin in (0, 100):
        joined_prefix = _running_prefix_at_tick(origin, bound=39)
        shifted_prefix = _running_prefix_at_tick(origin, bound=39)
        shifted_prefix.step(TickEvent("source-plus-two", tick=origin + 2))

        joined = monitor_residual(
            joined_prefix,
            (
                TickEvent("advance-two", tick=2),
                TickEvent("advance-thirty-seven", tick=37),
            ),
        )
        split = monitor_residual(
            shifted_prefix,
            (TickEvent("advance-thirty-seven", tick=37),),
        )

        outcomes[origin] = (
            shifted_prefix.tick,
            joined.contract_observation.mode,
            joined.contract_observation.occurrences,
            joined.continuation_status,
            joined == split,
        )

    no_disabled_label = residual_module.NoDisabledLabel()
    assert outcomes == {
        0: (2, Mode.RUNNING, {"o1": "Breached"}, no_disabled_label, True),
        100: (102, Mode.RUNNING, {"o1": "Breached"}, no_disabled_label, True),
    }


def test_running_domain_event_advance_is_relative_to_current_replay_clock():
    """Domain-event continuation labels carry the same relative-time convention.

    the totalised transition-system rule puts domain-event labels and tick labels in the same
    continuation alphabet, then states that continuation time is relative. The
    acceptance example `relative-domain-continuation-independent-of-origin`
    also uses a standalone domain-event continuation record, so this event form
    cannot get its time only from a preceding Tick label.

    The extra +2 step is deliberate. It proves the second event is rebased
    against the current replay clock, not the original prefix clock.
    """
    outcomes = {}
    for origin in (0, 100):
        prefix = _running_prefix_at_tick(origin, bound=10)
        observation = monitor_residual(
            prefix,
            (
                TickEvent("advance-two", tick=2),
                DomainEvent("approval-plus-one", tick=1, tag="approval", fields={"x": "f"}),
            ),
        )
        outcomes[origin] = (
            observation.contract_observation.mode,
            observation.contract_observation.summary,
            observation.contract_observation.diagnostic,
            observation.contract_observation.occurrences,
            observation.continuation_status,
        )

    no_disabled_label = residual_module.NoDisabledLabel()
    assert outcomes == {
        0: (Mode.RUNNING, Summary.UNKNOWN, None, {"o1": "Discharged"}, no_disabled_label),
        100: (Mode.RUNNING, Summary.UNKNOWN, None, {"o1": "Discharged"}, no_disabled_label),
    }


def test_running_terminal_event_advance_is_relative_to_current_replay_clock():
    """Terminal continuation labels are ordinary labels of the same alphabet.

    A Complete label at +1 after a +2 replay step reaches ticks 3 and 103 from
    the two prefixes below. In both cases it closes the open obligation before
    its deadline. Passing the terminal record through with tick 1 makes the
    shifted-origin branch fault instead.
    """
    outcomes = {}
    for origin in (0, 100):
        prefix = _running_prefix_at_tick(origin, bound=10)
        observation = monitor_residual(
            prefix,
            (
                TickEvent("advance-two", tick=2),
                TerminalEvent("complete-plus-one", tick=1, kind=Complete()),
            ),
        )
        outcomes[origin] = (
            observation.contract_observation.mode,
            observation.contract_observation.summary,
            observation.contract_observation.diagnostic,
            observation.contract_observation.occurrences,
            observation.continuation_status,
        )

    no_disabled_label = residual_module.NoDisabledLabel()
    assert outcomes == {
        0: (Mode.COMPLETE, Summary.VIOLATED, None, {"o1": "Breached"}, no_disabled_label),
        100: (Mode.COMPLETE, Summary.VIOLATED, None, {"o1": "Breached"}, no_disabled_label),
    }


def test_tick_label_never_names_an_absolute_tick():
    """the contract-observation rule, on canonicalisation: "absolute times replaced where
    possible by relative times and remaining deadlines".

    Structural version of the invariance above, and the reason it is worth
    having both: equality between two chosen prefixes could be satisfied by
    accident, but a label that names 1000 or 1006 anywhere in its rendering has
    kept the absolute clock whatever else it does. A label that names the
    advance, 6, passes.
    """
    prefix = _halted_prefix(terminal_tick=1000)
    attempted = _sink(monitor_residual(prefix, [TickEvent("tick-id", tick=6)])).attempted
    rendered = repr(attempted)

    assert "1006" not in rendered, f"tick label exposes an absolute tick: {rendered}"
    assert "1000" not in rendered, f"tick label exposes the prefix clock: {rendered}"
    assert "tick-id" not in rendered, f"tick label exposes an event_id: {rendered}"


def test_relative_advance_is_recovered_at_the_after_fault_boundary_too():
    """the totalised transition-system rule gives one alphabet and one label projection. The
    typed reasons `NotEnabled`, `AfterTerminal` and `AfterFault` differ in why
    the transition was disabled, never in what a label of `Sigma` is.

    So the advance must be recovered on the fault branch on the same terms:
    different advances are different labels, and the same advance from
    differently clocked prefixes is one label.
    """
    fault_advance_one = monitor_residual(_faulted_prefix(5), [TickEvent("a", tick=1)])
    fault_advance_six = monitor_residual(_faulted_prefix(5), [TickEvent("a", tick=6)])
    fault_advance_six_late = monitor_residual(
        _faulted_prefix(1_000_000), [TickEvent("b", tick=6)]
    )

    after_fault = DisabledReason.AFTER_FAULT
    assert _sink(fault_advance_one, after_fault).attempted != _sink(
        fault_advance_six, after_fault
    ).attempted
    assert fault_advance_one != fault_advance_six

    assert _sink(fault_advance_six, after_fault).attempted == _sink(
        fault_advance_six_late, after_fault
    ).attempted
    assert fault_advance_six == fault_advance_six_late


# -- a disabled label carries no unapplied observation map --------


def test_disabled_tick_label_drops_the_observation_map_it_never_applies():
    """the totalised transition-system rule: "The disabled label is not passed to the underlying
    contract transition and does not create a trigger occurrence, response
    token, evidence item, clause check, terminal conversion, or contract
    certificate."

    A disabled event is never stepped, so the updates it bundles are never
    applied, and the typed-events rule is what makes that decisive: "Observation
    updates attached to an event are bundled into that event and are applied
    before that event's contract checks." No application, no effect.

    The two maps here are the strongest pair available: applied, they would
    resolve the clause guard in opposite directions. Unapplied, they are two
    renderings of one label, and the residual definition requires "structural equality after
    the canonicalisation required there" of the total observations.

    The label is coarser than the record, and this is the direction of the
    error: keeping the map makes the equivalence finer than the transition
    system it observes, so a witness search can report a distinction that is
    not a semantic one.
    """
    observed_true = monitor_residual(
        _halted_prefix(), [TickEvent("a", tick=1, observations={"Sensitive(f)": K3.T})]
    )
    observed_false = monitor_residual(
        _halted_prefix(), [TickEvent("a", tick=1, observations={"Sensitive(f)": K3.F})]
    )
    observed_nothing = monitor_residual(_halted_prefix(), [TickEvent("a", tick=1)])

    assert _sink(observed_true).attempted == _sink(observed_false).attempted
    assert observed_true == observed_false
    assert observed_true == observed_nothing


def test_domain_and_terminal_labels_already_drop_the_observation_map():
    """The same rule on the other two continuation forms, which carry the same
    `observations` field. Domain and terminal labels also ignore observation
    maps when they are stored as attempted labels.

    the relevant acceptance-test rule renders an attempted domain-event
    label as `export(f)`, argument included, and the specification's worked example
    turns on `export(f)` against a different argument. The argument is label
    content; the observation map is not.
    """

    def domain(argument: str, observations):
        event = DomainEvent(
            "a", tick=1, tag="export", fields={"x": argument}, observations=observations
        )
        return monitor_residual(_halted_prefix(), [event])

    def terminal(observations):
        event = TerminalEvent("a", tick=1, kind=Complete(), observations=observations)
        return monitor_residual(_halted_prefix(), [event])

    assert domain("f", {"Sensitive(f)": K3.T}) == domain("f", {"Sensitive(f)": K3.F})
    assert terminal({"Sensitive(f)": K3.T}) == terminal({"Sensitive(f)": K3.F})

    assert domain("f", {}) != domain("g", {})
    assert _sink(domain("f", {})).attempted != _sink(domain("g", {})).attempted


def test_observations_reach_the_residual_when_they_are_applied():
    """The direction the canonicalisation above must not touch.

    This prefix is still running, so the continuation event is stepped, and
    the typed-events rule applies its updates "before that event's contract
    checks". The guard `Sensitive(f)` is unknown at the trigger, so resolving
    it true opens the obligation and resolving it false makes the occurrence
    inapplicable: the contract-observation rule carries the clause lifecycle inside
    `Obs_C`, so the two residuals differ.

    Observations therefore reach residual equality through the monitor state,
    which is exactly why they do not need to reach it a second time through a
    label that is never applied.
    """
    resolved_true = monitor_residual(
        _running_prefix(), [TickEvent("a", tick=1, observations={"Sensitive(f)": K3.T})]
    )
    resolved_false = monitor_residual(
        _running_prefix(), [TickEvent("a", tick=1, observations={"Sensitive(f)": K3.F})]
    )

    assert resolved_true.continuation_status == residual_module.NoDisabledLabel()
    assert resolved_false.continuation_status == residual_module.NoDisabledLabel()
    assert resolved_true != resolved_false


def test_monitor_residual_rejects_malformed_source_record_before_mode_dispatch():
    monitor = _halted_prefix()
    assert monitor.current_verdict().mode is Mode.COMPLETE

    with pytest.raises(ValueError) as exc_info:
        monitor_residual(monitor, [MalformedEvent("d0", "E_PARSE")])

    assert type(exc_info.value) is ValueError
    assert str(exc_info.value) == "MalformedEvent is not a Closed Core continuation event"
