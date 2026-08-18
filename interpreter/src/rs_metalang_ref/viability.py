"""Viability fixpoint (S4, "the viability fixed point").

The state space, actions, post-sets, and permission are passed in as plain
dicts by the caller (the abstraction/model-checking front end is out of this
reference's scope); this module implements the fixpoint recurrence, the
finite-termination argument, bounded stochastic reachability estimates, and
bounded result classification over the supplied finite model.
"""

from __future__ import annotations

import dataclasses
import math
from numbers import Rational, Real
from types import MappingProxyType
from typing import Callable, Dict, FrozenSet, Mapping, Optional, Tuple

from rs_metalang_ref.evidence import _PROBABILITY_TOTAL_TOLERANCE


def _require_finite_real(value: object, field: str) -> None:
    has_real_type = not isinstance(value, bool) and isinstance(value, Real)
    try:
        is_finite = has_real_type and math.isfinite(value)
    except OverflowError:
        is_finite = has_real_type
    if not is_finite:
        raise ValueError(f"{field} must be a finite real scalar")


def may_post(post: Dict[Tuple[str, str], FrozenSet[str]], z: str, a: str) -> FrozenSet[str]:
    return post.get((z, a), frozenset())


def must_post(post, z: str, a: str, target: FrozenSet[str]) -> bool:
    """S4.2: MustPost(z,a,X) iff Post(z,a) is nonempty and Post(z,a) subset X."""
    succs = may_post(post, z, a)
    return bool(succs) and succs <= target


def enabled_actions(z: str, actions_by_state: Dict[str, FrozenSet[str]]) -> FrozenSet[str]:
    """Enabled^must(z): the actions this reference is TOLD are enabled in
    every represented concrete state (S4.2 leaves the concrete-to-abstract
    reduction itself to the abstraction front end)."""
    return actions_by_state.get(z, frozenset())


def permit(permission: Dict[Tuple[str, str], str], z: str, a: str) -> bool:
    """Permit_C(z,a): Allow only. DenyUnknown is not permission (S4.2)."""
    return permission.get((z, a)) == "Allow"


def viability_fixpoint(
    states,
    safe: FrozenSet[str],
    goal: FrozenSet[str],
    actions_by_state: Dict[str, FrozenSet[str]],
    post: Dict[Tuple[str, str], FrozenSet[str]],
    permission: Dict[Tuple[str, str], str],
    max_rounds: Optional[int] = None,
):
    """S4.3's W recurrence, iterated to a fixpoint. W_0 = G intersect Safe_C;
    W_{n+1} adds every safe z with an enabled, permitted action whose
    MustPost lands entirely inside W_n. Returns
    (W_infinity, rank, witness_action), where
    rank[z] is the round at which z first entered W (S4.6's certificate
    rank r). On bound exhaustion, returns Abstain("Unknown(Nontermination)",
    receipt) with the bounded-attempt receipt."""
    states = list(states)
    w: FrozenSet[str] = frozenset(goal) & frozenset(safe)
    rank = {z: 0 for z in w}
    witness_action: Dict[str, str] = {}
    n = 0
    bound = max_rounds if max_rounds is not None else len(states) + 1
    while True:
        n += 1
        if n > bound:
            return Abstain(
                reason="Unknown(Nontermination)",
                receipt={
                    "round_bound": bound,
                    "rounds_completed": n - 1,
                    "winning_set": w,
                    "rank": rank,
                    "witness_action": witness_action,
                },
            )
        added = {}
        for z in states:
            if z in w or z not in safe:
                continue
            for a in enabled_actions(z, actions_by_state):
                if not permit(permission, z, a):
                    continue
                if must_post(post, z, a, w):
                    added[z] = a
                    break
        if not added:
            break
        for z, a in added.items():
            rank[z] = n
            witness_action[z] = a
        w = w | frozenset(added)
    return w, rank, witness_action


def classify_result(
    z: str,
    w: FrozenSet[str],
    rank: Dict[str, int],
    caller_attests_action_and_post_completeness: bool,
):
    """Return the S4.6 result triple from a caller attestation.

    This function validates the Boolean attestation but does not validate or
    retain an action/post completeness certificate.
    """
    if type(caller_attests_action_and_post_completeness) is not bool:
        raise ValueError("caller_attests_action_and_post_completeness must be a bool")
    if z in w:
        return ("WinningSure", rank[z])
    if caller_attests_action_and_post_completeness:
        return ("LosingExact", None)
    return ("NoPlanInAbstraction", None)


# -- S4.5: stochastic variant ------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FiniteReachabilityEstimate:
    """Immutable values produced by a fixed number of update rounds."""

    values: Mapping[str, float]
    iterations: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


def value_iteration_reachability(
    states,
    goal: FrozenSet[str],
    safe: FrozenSet[str],
    actions_by_state: Dict[str, FrozenSet[str]],
    transition_prob: Dict[Tuple[str, str], Dict[str, Real]],
    iterations: int = 200,
) -> FiniteReachabilityEstimate:
    """Return finite values after the requested update rounds.

    This numerical estimate is neither an S4.6 result triple nor an Abstain.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a non-negative integer")

    states = sorted(states)
    declared_states = frozenset(states)
    for state_action in sorted(transition_prob):
        probabilities = transition_prob[state_action]
        state, action = state_action
        row_field = f"transition_prob[{state_action!r}]"
        if state not in declared_states:
            raise ValueError(f"{row_field} source state is not declared in states")
        if action not in actions_by_state.get(state, frozenset()):
            raise ValueError(
                f"{row_field} action is not declared in actions_by_state[{state!r}]"
            )

        row_probabilities = []
        for successor in sorted(probabilities):
            probability = probabilities[successor]
            field = f"transition_prob[{state_action!r}][{successor!r}]"
            _require_finite_real(probability, field)
            if not 0 <= probability <= 1:
                raise ValueError(f"{field} must be in [0, 1]")
            if probability > 0 and successor not in declared_states:
                raise ValueError(
                    f"{field} has positive mass for a state not declared in states"
                )
            row_probabilities.append(probability)
        if all(isinstance(probability, Rational) for probability in row_probabilities):
            row_mass = sum(row_probabilities)
            is_normalized = row_mass == 1
        else:
            row_mass = math.fsum(row_probabilities)
            is_normalized = math.isclose(
                row_mass,
                1.0,
                rel_tol=_PROBABILITY_TOTAL_TOLERANCE,
                abs_tol=_PROBABILITY_TOTAL_TOLERANCE,
            )
        if not is_normalized:
            raise ValueError(f"{row_field} probabilities must sum to 1")

    for state in states:
        for action in sorted(actions_by_state.get(state, frozenset())):
            state_action = (state, action)
            if state_action not in transition_prob:
                raise ValueError(
                    f"transition_prob[{state_action!r}] is required for enabled action"
                )

    v = {z: (1.0 if z in goal and z in safe else 0.0) for z in states}
    executed_iterations = 0
    for _ in range(iterations):
        new_v = {}
        for z in states:
            if z in goal and z in safe:
                new_v[z] = 1.0
                continue
            if z not in safe:
                new_v[z] = 0.0
                continue
            acts = actions_by_state.get(z, frozenset())
            if not acts:
                new_v[z] = 0.0
                continue
            new_v[z] = max(
                sum(p * v[z2] for z2, p in transition_prob[(z, a)].items() if p > 0)
                for a in acts
            )
        v = new_v
        executed_iterations += 1
    return FiniteReachabilityEstimate(values=v, iterations=executed_iterations)


# -- S4.4: monotonicity ------------------------------------------------------


def f_operator(
    x: FrozenSet[str],
    states,
    goal: FrozenSet[str],
    safe: FrozenSet[str],
    actions_by_state: Dict[str, FrozenSet[str]],
    post: Dict[Tuple[str, str], FrozenSet[str]],
    permission: Dict[Tuple[str, str], str],
) -> FrozenSet[str]:
    """S4.4's F(X) = G intersect Safe_C union {z in Safe_C : exists a in
    A_C(z) with Permit_C(z,a), empty-set != Post(z,a) subset X}. Uses the
    same permission semantics as `viability_fixpoint`: Allow only;
    DenyUnknown is not permission."""
    result = set(goal) & set(safe)
    for z in states:
        if z in safe:
            for a in actions_by_state.get(z, frozenset()):
                if not permit(permission, z, a):
                    continue
                succs = may_post(post, z, a)
                if succs and succs <= x:
                    result.add(z)
                    break
    return frozenset(result)


def is_monotone(f: Callable[[FrozenSet[str]], FrozenSet[str]], x: FrozenSet[str], y: FrozenSet[str]) -> bool:
    """S4.4: if X subset Y then F(X) subset F(Y)."""
    if not x <= y:
        raise ValueError("is_monotone requires X subset Y")
    return f(x) <= f(y)


@dataclasses.dataclass(frozen=True)
class Abstain:
    reason: str
    receipt: dict
