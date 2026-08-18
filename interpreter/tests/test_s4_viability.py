"""S4 viability tests.

The S4.8 deterministic fixtures, S4.9 worked example, and S4.10 demonstrated
failure are normative examples, not a supplied complete concrete viability
instance. The finite reachability tests cover a bounded numerical estimate.
Finite updates do not implement the normative infinite-horizon stochastic result
or its `AlmostSure` classification. The public-input tests state Python
reference-interface policy.
"""

from __future__ import annotations

import os
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction

import pytest

from rs_metalang_ref import viability
from rs_metalang_ref.viability import (
    classify_result,
    must_post,
    permit,
    value_iteration_reachability,
    viability_fixpoint,
)

# -- S4.8 acceptance test 1: S4-empty-post-is-not-winning -------------------


def test_s4_empty_post_is_not_winning():
    states = ["s", "g"]
    safe = frozenset({"s", "g"})
    goal = frozenset({"g"})
    # 'a' is not enabled at s (S4.2's requirement excludes it from
    # Enabled^must), so it contributes no actions at all.
    actions_by_state = {"s": frozenset()}
    post = {("s", "a"): frozenset()}
    permission = {("s", "a"): "Allow"}

    w, rank, _ = viability_fixpoint(
        states, safe, goal, actions_by_state, post, permission
    )

    assert "s" not in w
    result, _ = classify_result(
        "s",
        w,
        rank,
        caller_attests_action_and_post_completeness=False,
    )
    assert result == "NoPlanInAbstraction"


# -- S4.8 acceptance test 2: S4-one-bad-successor-defeats-sure-winning ------


def test_s4_one_bad_successor_defeats_sure_winning():
    w0 = frozenset({"g"})
    post = {("s", "a"): frozenset({"g", "bad"})}

    assert must_post(post, "s", "a", w0) is False

    states = ["s", "g", "bad"]
    safe = frozenset({"s", "g", "bad"})
    goal = frozenset({"g"})
    actions_by_state = {"s": frozenset({"a"})}
    permission = {("s", "a"): "Allow"}

    w, _, _ = viability_fixpoint(states, safe, goal, actions_by_state, post, permission)
    assert "s" not in w


# -- S4.8 acceptance test 3: S4-unknown-permission-excluded -----------------


def test_s4_unknown_permission_excluded():
    permission = {("s", "a"): "DenyUnknown"}
    assert permit(permission, "s", "a") is False

    states = ["s", "g"]
    safe = frozenset({"s", "g"})
    goal = frozenset({"g"})
    actions_by_state = {"s": frozenset({"a"})}
    post = {("s", "a"): frozenset({"g"})}

    w, _, _ = viability_fixpoint(states, safe, goal, actions_by_state, post, permission)
    assert "s" not in w  # DenyUnknown never becomes a predecessor action


def test_s4_round_bound_returns_typed_nontermination_abstention():
    states = ["s0", "s1", "g"]
    safe = frozenset(states)
    goal = frozenset({"g"})
    actions_by_state = {
        "s0": frozenset({"advance"}),
        "s1": frozenset({"finish"}),
    }
    post = {
        ("s0", "advance"): frozenset({"s1"}),
        ("s1", "finish"): frozenset({"g"}),
    }
    permission = {key: "Allow" for key in post}

    result = viability_fixpoint(
        states, safe, goal, actions_by_state, post, permission, max_rounds=1
    )

    assert isinstance(result, viability.Abstain)
    assert result == viability.Abstain(
        reason="Unknown(Nontermination)",
        receipt={
            "round_bound": 1,
            "rounds_completed": 1,
            "winning_set": frozenset({"s1", "g"}),
            "rank": {"g": 0, "s1": 1},
            "witness_action": {"s1": "finish"},
        },
    )


def test_s4_winning_sure_reports_the_queried_states_exact_rank_as_horizon():
    states = ["s2", "s1", "g"]
    post = {
        ("s2", "advance"): frozenset({"s1"}),
        ("s1", "finish"): frozenset({"g"}),
    }
    winning, rank, _ = viability_fixpoint(
        states=states,
        safe=frozenset(states),
        goal=frozenset({"g"}),
        actions_by_state={
            "s2": frozenset({"advance"}),
            "s1": frozenset({"finish"}),
        },
        post=post,
        permission={state_action: "Allow" for state_action in post},
    )

    result_kind, horizon = classify_result(
        "s2",
        winning,
        rank,
        caller_attests_action_and_post_completeness=False,
    )

    assert rank["s2"] == 2
    assert result_kind == "WinningSure"
    assert horizon == rank["s2"]


# -- Bounded estimate; not the normative infinite-horizon result ------------


def test_s4_stochastic_self_loop_returns_finite_iteration_estimate():
    states = ["s", "g"]
    goal = frozenset({"g"})
    safe = frozenset({"s", "g"})
    actions_by_state = {"s": frozenset({"a"}), "g": frozenset({"stay"})}
    transition_prob = {
        ("s", "a"): {"s": 0.5, "g": 0.5},
        ("g", "stay"): {"g": 1.0},
    }

    estimate = value_iteration_reachability(
        states, goal, safe, actions_by_state, transition_prob, iterations=200
    )

    assert isinstance(estimate, viability.FiniteReachabilityEstimate)
    assert estimate.iterations == 200
    assert estimate.values["s"] == 1.0


def test_s4_slow_self_loop_remains_a_bounded_numerical_estimate():
    estimate = value_iteration_reachability(
        states=["s", "g"],
        goal=frozenset({"g"}),
        safe=frozenset({"s", "g"}),
        actions_by_state={"s": frozenset({"retry"}), "g": frozenset({"stay"})},
        transition_prob={
            ("s", "retry"): {"s": 0.99, "g": 0.01},
            ("g", "stay"): {"g": 1.0},
        },
        iterations=200,
    )

    assert isinstance(estimate, viability.FiniteReachabilityEstimate)
    assert estimate.iterations == 200
    assert estimate.values["s"] == pytest.approx(1.0 - 0.99**200)
    assert 0.0 < estimate.values["s"] < 1.0


def test_s4_zero_and_near_one_values_remain_finite_estimates():
    estimate = value_iteration_reachability(
        states=["zero", "near", "g"],
        goal=frozenset({"g"}),
        safe=frozenset({"zero", "near", "g"}),
        actions_by_state={"near": frozenset({"retry"})},
        transition_prob={
            ("near", "retry"): {"near": 0.5, "g": 0.5},
        },
        iterations=52,
    )

    assert isinstance(estimate, viability.FiniteReachabilityEstimate)
    assert estimate.iterations == 52
    assert estimate.values["zero"] == 0.0
    assert 0.999 < estimate.values["near"] < 1.0


def test_s4_finite_estimate_api_has_no_float_to_exact_verdict_classifier():
    assert not hasattr(viability, "classify_stochastic")


# -- Non-normative Python reference-interface validation --------------------


@pytest.mark.parametrize(
    "invalid_certificate",
    (1, "yes"),
    ids=("integer", "truthy-string"),
)
def test_s4_classify_result_rejects_a_non_boolean_caller_attestation(
    invalid_certificate,
):
    with pytest.raises(
        ValueError,
        match="^caller_attests_action_and_post_completeness must be a bool$",
    ):
        classify_result(
            "s",
            frozenset(),
            {},
            caller_attests_action_and_post_completeness=invalid_certificate,
        )


def test_s4_classify_result_uses_an_explicit_caller_attestation_without_a_certificate():
    result = classify_result(
        "s",
        frozenset(),
        {},
        caller_attests_action_and_post_completeness=True,
    )

    assert result == ("LosingExact", None)


def test_s4_viability_fixpoint_documents_the_triple_or_abstain_result():
    documentation = " ".join((viability_fixpoint.__doc__ or "").split())

    assert "(W_infinity, rank, witness_action)" in documentation
    assert "Abstain" in documentation
    assert "Returns (W_infinity, rank), where" not in documentation


def _value_iteration_with_row(probabilities, *, iterations=0, goal=frozenset()):
    states = ["s", *probabilities]
    return value_iteration_reachability(
        states=states,
        goal=goal,
        safe=frozenset(states),
        actions_by_state={"s": frozenset({"a"})},
        transition_prob={("s", "a"): probabilities},
        iterations=iterations,
    )


def _value_iteration_with_probability(probability):
    return _value_iteration_with_row(
        {"g": probability},
        iterations=1,
        goal=frozenset({"g"}),
    )


@pytest.mark.parametrize(
    "invalid_probability",
    (True, Decimal("0.5"), "0.5", 0.5 + 0.0j),
    ids=("bool", "decimal", "string", "complex"),
)
def test_s4_value_iteration_rejects_non_real_transition_probability(
    invalid_probability,
):
    with pytest.raises(
        ValueError,
        match=r"^transition_prob\[\('s', 'a'\)\]\['g'\] must be a finite real scalar$",
    ):
        _value_iteration_with_probability(invalid_probability)


@pytest.mark.parametrize(
    "invalid_probability",
    (float("nan"), float("inf")),
    ids=("nan", "infinity"),
)
def test_s4_value_iteration_rejects_non_finite_transition_probability(
    invalid_probability,
):
    with pytest.raises(
        ValueError,
        match=r"^transition_prob\[\('s', 'a'\)\]\['g'\] must be a finite real scalar$",
    ):
        _value_iteration_with_probability(invalid_probability)


@pytest.mark.parametrize(
    "invalid_probability",
    (-0.01, 1.01),
    ids=("below-zero", "above-one"),
)
def test_s4_value_iteration_rejects_transition_probability_outside_unit_interval(
    invalid_probability,
):
    with pytest.raises(
        ValueError,
        match=r"^transition_prob\[\('s', 'a'\)\]\['g'\] must be in \[0, 1\]$",
    ):
        _value_iteration_with_probability(invalid_probability)


def test_s4_value_iteration_accepts_fraction_transition_probability():
    estimate = _value_iteration_with_probability(Fraction(1, 1))

    assert isinstance(estimate, viability.FiniteReachabilityEstimate)
    assert estimate.iterations == 1
    assert estimate.values == {"s": 1.0, "g": 1.0}


def test_s4_value_iteration_accepts_ten_equal_float_probabilities():
    probabilities = {f"successor-{index}": 0.1 for index in range(10)}

    estimate = _value_iteration_with_row(probabilities)

    assert isinstance(estimate, viability.FiniteReachabilityEstimate)


def test_s4_value_iteration_accepts_a_normalized_decimal_float_row():
    probabilities = {"left": 0.01, "middle": 0.29, "right": 0.70}

    estimate = _value_iteration_with_row(probabilities)

    assert isinstance(estimate, viability.FiniteReachabilityEstimate)


def test_s4_value_iteration_accepts_normalized_float_row_in_any_insertion_order():
    entries = [
        *[(f"tiny-{index}", 1e-17) for index in range(10)],
        ("bulk", 1.0 - 1e-16),
    ]
    probabilities_by_order = (dict(entries), dict(reversed(entries)))

    estimates = [
        _value_iteration_with_row(probabilities)
        for probabilities in probabilities_by_order
    ]

    assert all(
        isinstance(estimate, viability.FiniteReachabilityEstimate)
        for estimate in estimates
    )


def test_s4_value_iteration_accepts_exact_fraction_row():
    estimate = _value_iteration_with_row(
        {"left": Fraction(1, 3), "right": Fraction(2, 3)}
    )

    assert isinstance(estimate, viability.FiniteReachabilityEstimate)


@pytest.mark.parametrize(
    "invalid_iterations",
    (True, -1, 1.0),
    ids=("bool", "negative", "float"),
)
def test_s4_value_iteration_rejects_invalid_iteration_count(invalid_iterations):
    with pytest.raises(
        ValueError,
        match="^iterations must be a non-negative integer$",
    ):
        _value_iteration_with_row({"g": 1}, iterations=invalid_iterations)


@pytest.mark.parametrize(
    ("iterations", "expected_values"),
    (
        (0, {"s0": 0.0, "s1": 0.0, "g": 1.0}),
        (1, {"s0": 0.0, "s1": 1.0, "g": 1.0}),
        (3, {"s0": 1.0, "s1": 1.0, "g": 1.0}),
    ),
    ids=("zero", "one", "three"),
)
def test_s4_value_iteration_reports_executed_iteration_count(
    iterations,
    expected_values,
):
    estimate = value_iteration_reachability(
        states=["s0", "s1", "g"],
        goal=frozenset({"g"}),
        safe=frozenset({"s0", "s1", "g"}),
        actions_by_state={
            "s0": frozenset({"advance"}),
            "s1": frozenset({"advance"}),
        },
        transition_prob={
            ("s0", "advance"): {"s1": 1},
            ("s1", "advance"): {"g": 1},
        },
        iterations=iterations,
    )

    assert estimate.iterations == iterations
    assert estimate.values == expected_values


@pytest.mark.parametrize(
    ("actions_by_state", "transition_prob", "expected_error"),
    (
        (
            {"s": frozenset({"a"})},
            {
                ("s", "a"): {"g": 1.0},
                ("outside", "a"): {"g": 1.0},
            },
            "transition_prob[('outside', 'a')] source state is not declared in states",
        ),
        (
            {"s": frozenset({"a"})},
            {
                ("s", "a"): {"g": 1.0},
                ("s", "outside"): {"g": 1.0},
            },
            (
                "transition_prob[('s', 'outside')] action is not declared in "
                "actions_by_state['s']"
            ),
        ),
        (
            {"s": frozenset({"a"})},
            {("s", "a"): {"outside": 1.0}},
            (
                "transition_prob[('s', 'a')]['outside'] has positive mass for a "
                "state not declared in states"
            ),
        ),
        (
            {"s": frozenset({"a"})},
            {},
            "transition_prob[('s', 'a')] is required for enabled action",
        ),
        (
            {"s": frozenset({"a"})},
            {("s", "a"): {"g": 0.4}},
            "transition_prob[('s', 'a')] probabilities must sum to 1",
        ),
    ),
    ids=(
        "undeclared-source-state",
        "undeclared-action",
        "undeclared-positive-mass-successor",
        "missing-enabled-transition-row",
        "non-normalized-transition-row",
    ),
)
def test_s4_value_iteration_rejects_malformed_transition_kernel(
    actions_by_state,
    transition_prob,
    expected_error,
):
    with pytest.raises(ValueError) as raised:
        value_iteration_reachability(
            states=["s", "g"],
            goal=frozenset({"g"}),
            safe=frozenset({"s", "g"}),
            actions_by_state=actions_by_state,
            transition_prob=transition_prob,
            iterations=1,
        )

    assert str(raised.value) == expected_error


def test_s4_value_iteration_accepts_terminal_model_without_transition_rows():
    estimate = value_iteration_reachability(
        states=["terminal", "g"],
        goal=frozenset({"g"}),
        safe=frozenset({"terminal", "g"}),
        actions_by_state={},
        transition_prob={},
        iterations=1,
    )

    assert estimate.values == {"terminal": 0.0, "g": 1.0}


def test_s4_value_iteration_accepts_zero_mass_entry_outside_declared_support():
    estimate = value_iteration_reachability(
        states=["s", "g"],
        goal=frozenset({"g"}),
        safe=frozenset({"s", "g"}),
        actions_by_state={"s": frozenset({"a"})},
        transition_prob={
            ("s", "a"): {"g": 1.0, "outside": 0.0},
        },
        iterations=1,
    )

    assert estimate.values == {"s": 1.0, "g": 1.0}


def test_s4_missing_transition_diagnostic_is_stable_across_hash_seeds():
    script = """
from rs_metalang_ref.viability import value_iteration_reachability

try:
    value_iteration_reachability(
        states=frozenset({"alpha", "beta"}),
        goal=frozenset(),
        safe=frozenset({"alpha", "beta"}),
        actions_by_state={
            "alpha": frozenset({"go"}),
            "beta": frozenset({"go"}),
        },
        transition_prob={},
        iterations=0,
    )
except ValueError as exc:
    print(exc)
else:
    raise AssertionError("expected a missing-transition ValueError")
"""
    messages = []
    for hash_seed in ("1", "2"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            env=environment,
            text=True,
        )
        messages.append(completed.stdout.strip())

    expected = "transition_prob[('alpha', 'go')] is required for enabled action"
    assert messages == [expected, expected]


# -- S4.9: fully worked reproducibility example -----------------------------


def test_s4_worked_reproducibility_example():
    states = ["z0", "z1", "r", "z2", "g", "l"]
    safe = frozenset({"z0", "z1", "r", "z2", "g"})  # l is unsafe
    goal = frozenset({"g"})
    actions_by_state = {
        "z0": frozenset({"edit"}),
        "z1": frozenset({"test", "deleteBackup"}),
        "r": frozenset({"rollback"}),
        "z2": frozenset({"test"}),
    }
    post = {
        ("z0", "edit"): frozenset({"z1"}),
        ("z1", "test"): frozenset({"g", "r"}),
        ("r", "rollback"): frozenset({"g"}),
        ("z1", "deleteBackup"): frozenset({"z2"}),
        ("z2", "test"): frozenset({"g", "l"}),
    }
    permission = {key: "Allow" for key in post}

    w, rank, _ = viability_fixpoint(
        states, safe, goal, actions_by_state, post, permission
    )

    assert rank["r"] == 1
    assert rank["z1"] == 2
    assert rank["z0"] == 3
    assert "z2" not in w  # the only completion from z2 can reach l


# -- S4.10: DEMONSTRATED-FAILURE (omitted recovery action) ------------------


def test_s4_demonstrated_failure_no_plan_in_abstraction():
    # z2's abstraction omits a remote recovery action that concretely exists;
    # without an action-completeness certificate, the engine must not claim
    # LosingExact.
    states = ["z2", "g", "l"]
    safe = frozenset({"z2", "g"})
    goal = frozenset({"g"})
    actions_by_state = {"z2": frozenset({"test"})}
    post = {("z2", "test"): frozenset({"g", "l"})}
    permission = {("z2", "test"): "Allow"}

    w, rank, _ = viability_fixpoint(
        states, safe, goal, actions_by_state, post, permission
    )
    assert "z2" not in w

    result, _ = classify_result(
        "z2",
        w,
        rank,
        caller_attests_action_and_post_completeness=False,
    )
    assert result == "NoPlanInAbstraction"


@pytest.mark.parametrize(
    "unused_result_type",
    (
        "WinningSure",
        "LosingExact",
        "NoPlanInAbstraction",
        "PositiveProbability",
        "AlmostSure",
    ),
)
def test_s4_public_api_omits_unproduced_result_dataclasses(unused_result_type):
    assert not hasattr(viability, unused_result_type)


def test_f_operator_denies_unpermitted_action_like_the_fixpoint():
    """Regression for the permission gap: an action whose permission is
    DenyUnknown must not create a winning predecessor through f_operator,
    matching viability_fixpoint's Allow-only semantics."""
    states = ["s", "g"]
    safe = frozenset({"s", "g"})
    goal = frozenset({"g"})
    actions_by_state = {"s": frozenset({"a"})}
    post = {("s", "a"): frozenset({"g"})}
    denied = {("s", "a"): "DenyUnknown"}
    allowed = {("s", "a"): "Allow"}
    from rs_metalang_ref.viability import f_operator

    x = frozenset({"g"})
    assert f_operator(x, states, goal, safe, actions_by_state, post, denied) == frozenset({"g"})
    assert f_operator(x, states, goal, safe, actions_by_state, post, allowed) == frozenset({"g", "s"})
