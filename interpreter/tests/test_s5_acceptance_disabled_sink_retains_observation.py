"""Acceptance tests for disabled sinks that retain full contract observations.

The contract-adequacy acceptance suite includes the acceptance test
`disabled-sink-retains-full-contract-observation`. It runs against the contract
path rather than the abstract transition system, because its expected values are
a contract's mode and summary.

A residual narrowed to `(Summary, Mode)` would erase required fields, including
the continuation status and retained attempted label. These tests require the
full residual codomain.

Every expected value below is read off the specification, never off the implementation.
"""

from __future__ import annotations

from collections.abc import Mapping

from rs_metalang_ref.contracts import AfterClauseSpec, BoundUnaryGuard, Linear
from rs_metalang_ref.events import Complete, DomainEvent, TerminalEvent, TickEvent
from rs_metalang_ref.kleene import K3
from rs_metalang_ref.monitor import SingleClauseMonitor
from rs_metalang_ref.residual import monitor_residual
from rs_metalang_ref.verdict import Mode, Summary

# -- the contract of the acceptance test ------------------------------------


def _contract() -> AfterClauseSpec:
    """the contract-adequacy acceptance tests:

        contract:
          after: export(x)
          when: Sensitive(x)
          require: approval(x)
          within: 2
    """
    return AfterClauseSpec(
        clause_id="c1",
        trigger_tag="export",
        response_tag="approval",
        binding_fields=("x",),
        guard=BoundUnaryGuard("Sensitive", "x"),
        bound=2,
        discharge=Linear(),
    )


def _monitor() -> SingleClauseMonitor:
    """`initial: Sensitive(f): true`."""
    monitor = SingleClauseMonitor(_contract())
    monitor.set_initial("Sensitive(f)", K3.T)
    return monitor


def _left_prefix() -> SingleClauseMonitor:
    """`left_prefix: [0, export, {x: f}], [3, Complete, {}]`."""
    monitor = _monitor()
    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    monitor.step(TerminalEvent("e1", tick=3, kind=Complete()))
    return monitor


def _right_prefix() -> SingleClauseMonitor:
    """`right_prefix: [0, export, {x: f}], [1, approval, {x: f}], [3, Complete, {}]`."""
    monitor = _monitor()
    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    monitor.step(DomainEvent("e1", tick=1, tag="approval", fields={"x": "f"}))
    monitor.step(TerminalEvent("e2", tick=3, kind=Complete()))
    return monitor


def _continuation(tag: str = "export"):
    """`continuation: [1, export, {x: f}]`.

    the traces, prefixes, and continuations rule: "Time inside a continuation is
    represented by relative tick advances, so that a continuation can be appended to any
    prefix without depending on absolute timestamps." The continuation therefore carries
    the relative advance 1, not the prefixes' absolute next tick 4.
    """
    return [DomainEvent("c0", tick=1, tag=tag, fields={"x": "f"})]


# -- reading a residual without prescribing how it is built -----------------


def _field(observation, name):
    """Read a field the specification names, from either a record or a mapping.

    the residual definition fixes the field names but not their Python encoding, so this accepts
    an attribute or a mapping key and fails with the citation when the field is absent
    altogether.
    """
    if isinstance(observation, Mapping) and name in observation:
        return observation[name]
    if hasattr(observation, name):
        return getattr(observation, name)
    raise AssertionError(
        f"the residual definition gives the residual the codomain TotalObservation_C = "
        f"{{contract_observation, continuation_status}}, where contract_observation is the "
        f"range of the complete projection Obs_C of the contract-observation rule. The residual "
        f"{observation!r} of type {type(observation).__name__} has no field {name!r}."
    )


def _reason_name(reason):
    """The specification writes the reason as `AfterTerminal`; the encoding is free."""
    return getattr(reason, "value", reason)


# -- the third listed acceptance fixture ------------------------------------


def test_disabled_sink_retains_full_contract_observation():
    """the contract-adequacy acceptance tests, acceptance test
    `disabled-sink-retains-full-contract-observation`.

        expected:
          left_residual:
            contract_observation:
              mode: Complete
              summary: Violated
            continuation_status:
              Disabled:
                attempted: export(f)
                enabled_set: []
                reason: AfterTerminal
          right_residual:
            contract_observation:
              mode: Complete
              summary: Satisfied
            continuation_status:
              Disabled:
                attempted: export(f)
                enabled_set: []
                reason: AfterTerminal
          residuals_equal: false
          equivalent: false

    The totalised transition-system rule states what the record is for: "Freezing the complete source observation
    additionally ensures that totalisation cannot erase a contract distinction already
    established before the disabled attempt." Here the distinction is `Violated` against
    `Satisfied`, and it must survive an attempt made past the terminal boundary.

    One field is asserted as a property rather than a spelling. The specification's
    continuation alphabet is labels and it writes the attempted one as `export(f)`, while
    this interpreter's continuations are event records, so no literal rendering is fixed by
    the specification. The test asserts that the attempted label is recorded and that it
    distinguishes attempts, which is what the totalised transition-system rule's "The first disabled attempt is
    retained" requires of it; `test_the_attempted_label_is_recorded_and_distinguishes`
    below carries that second half.
    """
    left = monitor_residual(_left_prefix(), _continuation())
    right = monitor_residual(_right_prefix(), _continuation())

    for side, residual, expected_summary in (
        ("left", left, Summary.VIOLATED),
        ("right", right, Summary.SATISFIED),
    ):
        observation = _field(residual, "contract_observation")

        assert _field(observation, "mode") == Mode.COMPLETE, (
            f"{side}_residual.contract_observation.mode: expected Complete, got "
            f"{_field(observation, 'mode')!r}"
        )
        assert _field(observation, "summary") == expected_summary, (
            f"{side}_residual.contract_observation.summary: expected "
            f"{expected_summary.value}, got {_field(observation, 'summary')!r}"
        )

        status = _field(residual, "continuation_status")

        assert _reason_name(_field(status, "reason")) == "AfterTerminal", (
            f"{side}_residual.continuation_status.reason: expected AfterTerminal, got "
            f"{_field(status, 'reason')!r}"
        )
        assert len(_field(status, "enabled_set")) == 0, (
            f"{side}_residual.continuation_status.enabled_set: expected [], got "
            f"{_field(status, 'enabled_set')!r}"
        )
        assert "export" in str(_field(status, "attempted")), (
            f"{side}_residual.continuation_status.attempted: expected the export(f) "
            f"attempt, got {_field(status, 'attempted')!r}"
        )

    assert left != right, "residuals_equal: false"


def test_the_attempted_label_is_recorded_and_distinguishes():
    """the totalised transition-system rule: "The first disabled attempt is retained.
    Every later continuation label leaves the totalised state unchanged", and the residual
    definition puts `attempted = label` in the Disabled record.

    The point of retaining it is stated by the totalised transition-system rule: "Totalisation is what makes the
    enabled-action domain observable. Two prefixes that differ only in which actions are
    available are distinguished by attempting an action that one enables and the other does
    not." A record that stored a constant, or nothing, would satisfy the acceptance test's
    other three expectations and lose that. This pins the field's content without pinning
    its spelling.
    """
    exported = monitor_residual(_left_prefix(), _continuation(tag="export"))
    approved = monitor_residual(_left_prefix(), _continuation(tag="approval"))

    assert exported != approved, (
        "the totalised transition system and residual definitions retain the first disabled attempt in the residual, so two "
        "continuations differing only in the attempted label must not merge"
    )


def test_same_disabled_domain_event_tag_retains_bound_field_values():
    """`export(x=f)` and `export(x=g)` remain distinct after disablement.

    The event ID, tick, and tag are identical. This reads the public `attempted`
    field and compares its semantic payload without choosing a Python encoding.
    """
    attempted_f = monitor_residual(
        _left_prefix(),
        [DomainEvent("c0", tick=1, tag="export", fields={"x": "f"})],
    )
    attempted_g = monitor_residual(
        _left_prefix(),
        [DomainEvent("c0", tick=1, tag="export", fields={"x": "g"})],
    )

    label_f = _field(_field(attempted_f, "continuation_status"), "attempted")
    label_g = _field(_field(attempted_g, "continuation_status"), "attempted")

    assert label_f != label_g, (
        "the totalised transition system and residual definitions retain the bound fields of "
        "the first disabled DomainEvent label"
    )
    assert attempted_f != attempted_g


# -- consequences of retaining complete contract observations --------------


def test_a_contract_that_fired_and_was_discharged_is_not_merged_with_one_that_never_fired():
    """the aggregate summary rule, closing paragraph: "A response
    clause with no trigger occurrence is vacuously satisfied at an actual closing boundary.
    Vacuous satisfaction is reported as such through the monitorability map,
    so that a contract which never fired is never counted as a contract that held."

    Both prefixes below close as `Satisfied` at `Complete`, and the aggregate summary rule
    makes that correct for both: one discharged its obligation, the other
    never had one. The specification's own gloss is that the two must remain distinguishable
    on the verdict surface, and the field that distinguishes them is the clause lifecycle
    that the contract-observation rule puts in `Obs_C`: "Canonical clause lifecycle. For every clause
    and every occurrence: inactive or held-so-far, ... open obligation, discharged,
    inapplicable, breached, waived, unknown-final".

    the residual definition defines the residual as `TotalObs_C(delta_bar*(q_u, w))` over that
    complete projection, and the totalised transition-system rule enumerates what a frozen observation carries: "the
    complete mode, three-valued summary, clause lifecycle, observability map, enabled and
    forbidden actions, evidence summary and delta, certificates, severity, monitorability
    map, and diagnostic". A residual whose codomain is a summary and a mode is not that
    projection, and it merges these two prefixes.
    """
    fired = _monitor()
    fired.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    fired.step(DomainEvent("e1", tick=1, tag="approval", fields={"x": "f"}))
    fired.step(TerminalEvent("e2", tick=3, kind=Complete()))

    never_fired = _monitor()
    never_fired.step(TickEvent("k0", tick=0))
    never_fired.step(TickEvent("k1", tick=1))
    never_fired.step(TerminalEvent("k2", tick=3, kind=Complete()))

    assert fired.current_verdict() != never_fired.current_verdict(), (
        "precondition: the monitor's own verdict object must already distinguish these "
        "two, otherwise this test is about the monitor rather than about the residual"
    )

    assert monitor_residual(fired, []) != monitor_residual(never_fired, []), (
        "the aggregate summary rule: a contract which never fired is never counted as a contract "
        "that held. The residual merged a discharged obligation with an absent one"
    )


def test_two_distinct_monitor_faults_are_not_merged():
    """the totalised transition-system rule names the diagnostic explicitly among what
    the frozen contract observation retains: "It therefore retains the complete mode,
    three-valued summary, clause lifecycle, observability map, enabled and forbidden
    actions, evidence summary and delta, certificates, severity, monitorability map, and
    diagnostic present immediately before the first disabled label."

    the contract-observation rule carries the same field on the verdict object, `diagnostic:
    Optional[TypedDiagnostic]`, and the three-valued epistemic layer fixes one of the two faults below as the
    mechanism "by which the epistemic substrate can report its own failure", producing
    `MonitorFault(InconsistentObservation, proposition_key, prior_value, attempted_value)`.

    The two prefixes below are faulted for different reasons and agree on everything else,
    so the diagnostic is the only field that separates them. A faulted monitor also
    refuses every later event, so if the residual drops the diagnostic no continuation at
    any length recovers the distinction.
    """
    inconsistent = _monitor()
    inconsistent.step(TickEvent("k0", tick=0, observations={"Sensitive(f)": K3.F}))

    non_monotone = _monitor()
    non_monotone.step(TickEvent("k0", tick=5))
    non_monotone.step(TickEvent("k1", tick=2))

    assert inconsistent.current_verdict() != non_monotone.current_verdict(), (
        "precondition: the monitor's own verdict object must already distinguish these two"
    )

    assert monitor_residual(inconsistent, []) != monitor_residual(non_monotone, []), (
        "the totalised transition-system rule retains the diagnostic in the frozen contract observation. The "
        "residual merged an InconsistentObservation fault with a NonMonotoneTick fault"
    )
