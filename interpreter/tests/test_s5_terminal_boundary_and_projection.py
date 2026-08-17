"""Specification-derived tests for totalised residuals and their projections.

Every expected value below is read off the specification, never off the implementation.
Each test names the section it comes from and quotes the sentence that decides it.
"""

from __future__ import annotations

import pytest

import rs_metalang_ref.residual as residual_module
from rs_metalang_ref.residual import (
    DisabledReason,
    TotalizedLTS,
    binary_accept,
    gate_projection,
)
from rs_metalang_ref.verdict import Summary

# -- helpers ----------------------------------------------------------------


def _state_after(lts: TotalizedLTS, state, word):
    """Return the totalised state after applying a continuation prefix."""

    for label in word:
        state = lts.delta_bar(state, label)
    return state


def _terminal_boundary_lts(observations=None, enabled_by_state=None):
    """A model whose only terminal label is `Complete`.

    `enabled` is deliberately terminality-unaware, returning the same labels before and
    after the boundary. That is forced by the specification, not a convenience: a state
    past a terminal boundary and a live state that happens to enable nothing are both
    disabled, and the totalised transition system gives them different reasons (`AfterTerminal` against
    `NotEnabled`). An `enabled` function that already dropped the label after the boundary
    could not produce the first reason. So terminatedness has to reach the transition by
    some route other than the enabled set.
    """
    observations = observations or {}
    enabled_by_state = enabled_by_state or {}
    default_enabled = frozenset({"Complete", "write"})

    return TotalizedLTS(
        transition=lambda state, label: (
            ("closed_" + state) if label == "Complete" else state
        ),
        enabled=lambda state: enabled_by_state.get(state, default_enabled),
        observe=lambda state: observations.get(state, "OBS"),
    )


# -- Shift identity across a terminal boundary ------------------------------


def test_shift_identity_holds_across_the_terminal_boundary():
    """the parameterised-equivalence rule, the shift identity:

        R_{u x}^C(w) = R_u^C(x w)

    quoted from the section as `R_{u x}^C(w) = R_u^C(x w)`, and stated there to hold "for
    every prefix `u` and all continuation words `x` and `w`". The proof given in that
    section goes through `q_{u x} = delta_bar*(q_u, x)`, so splitting a word at any
    position, including at a terminal label, must not change the residual.

    The totalised transition system puts the terminal boundary in the state rather than in the walk: it defines
    `DisabledReason(s, a)` as a function of the state and the label, and fixes that "a
    non-terminal label after a terminal boundary has reason `AfterTerminal`". A prefix that
    has already passed the boundary therefore carries that fact, and a continuation
    evaluated from it sees the same thing a continuation evaluated across the split sees.
    """
    lts = _terminal_boundary_lts()

    whole = lts.residual("live", ("Complete", "write"))
    split = lts.residual(_state_after(lts, "live", ("Complete",)), ("write",))

    assert whole == split, (
        "the parameterised-equivalence rule: R_{u x}(w) = R_u(x w). Splitting the continuation at the "
        "terminal label changed the residual, so the boundary is not carried by the state "
        f"reached after it. R_u(Complete write) = {whole!r}; R_{{u Complete}}(write) = {split!r}"
    )


def test_a_label_after_the_terminal_boundary_is_disabled_from_either_side_of_a_split():
    """the totalised transition-system rule fixes the reason, not merely the fact of
    disablement: "a non-terminal label after a terminal boundary has reason
    `AfterTerminal`".

    This is the companion to the identity test above and rules out one way of satisfying it
    cheaply. Making both sides report `NotEnabled` would equalise the two residuals and
    still contradict the totalised transition system, which assigns `NotEnabled` to "an attempted action outside
    the enabled domain" and reserves `AfterTerminal` for this case.
    """
    lts = _terminal_boundary_lts()

    for label, residual in (
        ("whole word", lts.residual("live", ("Complete", "write"))),
        (
            "after the split",
            lts.residual(_state_after(lts, "live", ("Complete",)), ("write",)),
        ),
    ):
        status = residual.continuation_status
        assert isinstance(status, residual_module.Disabled), (
            f"the totalised transition-system rule: a non-terminal label after a terminal boundary is disabled. "
            f"Evaluated {label}, the residual was {residual!r}"
        )
        assert status.reason is DisabledReason.AFTER_TERMINAL, (
            f"the totalised transition-system rule reserves AfterTerminal for a label after a terminal "
            f"boundary. Evaluated {label}, the reason was {status.reason!r}"
        )


# -- Enabled set recorded in an AfterTerminal sink --------------------------


def test_after_terminal_sink_records_an_empty_enabled_set():
    """the relevant acceptance-test rule, acceptance test
    `disabled-sink-retains-full-contract-observation`, fixes the expected record for both
    prefixes as

        continuation_status:
          Disabled:
            attempted: export(f)
            enabled_set: []
            reason: AfterTerminal

    so `enabled_set` is empty when the reason is `AfterTerminal`. The totalised transition
    system's `enabled_set = Enabled(s)` is consistent with that and explains it: past a terminal
    boundary the state enables nothing, so the set the sink stores is empty. The set the
    state enabled *before* the boundary is not the set the totalised transition system asks for.
    """
    lts = _terminal_boundary_lts()

    observation = lts.residual("live", ("Complete", "write"))
    status = observation.continuation_status

    assert isinstance(status, residual_module.Disabled)
    assert status.enabled_set == frozenset(), (
        "the relevant acceptance-test rule, disabled-sink-retains-full-contract-observation, pins "
        "enabled_set: [] against reason AfterTerminal. The record carries "
        f"{set(status.enabled_set)!r}, which is the set the state enabled before the boundary; "
        "the record therefore asserts both that the continuation is past the terminal "
        "boundary and that these labels are available."
    )


def test_two_prefixes_with_the_same_observation_compare_equal_after_terminal_boundary():
    """The equality consequence of an empty enabled set after the boundary.

    the residual definition: "If two source states have different contract
    observations but the same attempted label, enabled set, and disabled reason, their
    disabled residuals remain different." The converse constrains the quotient: where the
    contract observations agree and the specified record agrees, the residuals must
    agree. Otherwise, the residual representation keeps a distinction that is
    absent from the declared total observation.

    Both prefixes here observe the same contract observation and differ only in what they
    enabled before the terminal boundary, which the monitorability rule says the sink does not record.
    """
    lts = _terminal_boundary_lts(
        enabled_by_state={
            "u": frozenset({"Complete", "write"}),
            "closed_u": frozenset({"Complete", "write"}),
            "v": frozenset({"Complete", "delete"}),
            "closed_v": frozenset({"Complete", "delete"}),
        }
    )

    assert lts.residual("u", ()) == lts.residual("v", ()), (
        "precondition: the two prefixes must agree on the empty continuation, otherwise "
        "this test proves nothing"
    )

    assert lts.residual("u", ("Complete", "x")) == lts.residual(
        "v", ("Complete", "x")
    ), (
        "two prefixes with identical contract observations were distinguished after a "
        "terminal boundary by enabled sets that the relevant acceptance-test rule says the sink does not "
        "carry"
    )


# -- Cross-type equality of frozen contract observations --------------------


def test_frozen_observations_of_different_types_remain_distinct():
    """the residual definition: "Equality of total observations is
    structural equality after the canonicalisation required there", and, on the sink
    specifically, "The frozen contract observation is mandatory. If two source states have
    different contract observations but the same attempted label, enabled set, and disabled
    reason, their disabled residuals remain different."

    `True` and `1` are different contract observations. Structural equality of a typed
    observation distinguishes them; Python's `==` does not, because `bool` is a subclass of
    `int`. A comparison that inherits that coincidence is not the structural equality
    the residual definition requires, and it treats two distinct observations as equal.

    `ContractObservation_C` is the range of the typed projection `Obs_C` from the specification's contract observation. It is a structured record, and two values of different type
    are not the same structure.
    """
    for left, right in ((True, 1), (1, 1.0), (0, False)):
        observations = {"u": left, "v": right}
        lts = TotalizedLTS(
            transition=lambda state, label: state,
            enabled=lambda state: frozenset({"ok"}),
            observe=lambda state, observations=observations: observations[state],
        )

        residual_u = lts.residual("u", ("nope",))
        residual_v = lts.residual("v", ("nope",))

        assert residual_u != residual_v, (
            f"the residual definition: different contract observations keep their disabled "
            f"residuals different. {left!r} ({type(left).__name__}) and {right!r} "
            f"({type(right).__name__}) compared equal"
        )


def test_supplied_continuation_agreement_rejects_cross_type_observations():
    """The supplied finite words expose different residual encodings.

    The result concerns only these three continuations. It makes no claim about
    any continuation that the caller did not supply.
    """
    observations = {"u": True, "v": 1}
    lts = TotalizedLTS(
        transition=lambda state, label: state,
        enabled=lambda state: frozenset({"ok"}),
        observe=lambda state: observations[state],
    )

    assert (
        lts.agrees_on_supplied_continuations("u", "v", [(), ("nope",), ("ok", "nope")])
        is False
    ), "the supplied continuations produced different contract observations"


def test_identical_observations_still_compare_equal():
    """Positive control: identical frozen observations still compare equal.

    the scoped-minimisation rule defines a quotient over observations. Adding a
    state's identity to every comparison would make the quotient collapse to the source
    states, so two equal contract observations must still compare equal.
    """
    observations = {"u": True, "v": True}
    lts = TotalizedLTS(
        transition=lambda state, label: state,
        enabled=lambda state: frozenset({"ok"}),
        observe=lambda state: observations[state],
    )

    assert lts.residual("u", ("nope",)) == lts.residual("v", ("nope",)), (
        "two prefixes the contract does not distinguish must still compare equal, or the "
        "quotient collapses to the identity"
    )


# -- Strict typed projection boundary ---------------------------------------


@pytest.mark.parametrize("projection", [gate_projection, binary_accept])
@pytest.mark.parametrize(
    "summary_string",
    [Summary.SATISFIED.value, Summary.VIOLATED.value, Summary.UNKNOWN.value],
)
def test_summary_projections_reject_serialized_strings(projection, summary_string):
    with pytest.raises(TypeError, match="summary must be a Summary"):
        projection(summary_string)


def test_the_projections_still_separate_satisfied_from_violated():
    """Positive control: projections separate satisfied from violated observations.

    A constant projection would satisfy the two equality tests above but would make
    the profile-relative operational-adequacy rule vacuous. That section's point is that the gate
    projection collapses `Violated` and `Unknown` into one decision, which is only a
    useful distinction if `Satisfied` gets the other one.

    The concrete strings the gate returns are not fixed by the specification, so this
    asserts a separation rather than a value.
    """
    assert gate_projection(Summary.SATISFIED) != gate_projection(Summary.VIOLATED)
    assert gate_projection(Summary.VIOLATED) == gate_projection(Summary.UNKNOWN)
    assert binary_accept(Summary.SATISFIED) != binary_accept(Summary.VIOLATED)
    assert binary_accept(Summary.VIOLATED) == binary_accept(Summary.UNKNOWN)
