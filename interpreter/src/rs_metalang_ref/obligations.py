"""Response-obligation lifecycle for the single-`after`-clause fragment.

The `S1.4`-style labels in this module mark the response-obligation rules
(trigger and guard, retroactive activation, response-token discipline); they
are local implementation labels, not specification section identifiers.

Each occurrence has a trace-local canonical alias such as `o1`. The alias is a
monitor-local presentation identifier. The normative structured identity
`(clause_id, (event_id, tick), match_ordinal)` and its canonical mapping to
aliases are outside this fragment.

This module implements the trigger-and-guard rule, retroactive activation, and
linear response-token allocation for the executable fragment.
"""

from __future__ import annotations

import dataclasses
from enum import Enum

from .kleene import K3


class OccurrenceStatus(Enum):
    CONDITIONAL_OPEN = "ConditionalOpen"
    CONDITIONAL_EXPIRED = "ConditionalExpired"
    OPEN = "Open"
    DISCHARGED = "Discharged"
    INAPPLICABLE = "Inapplicable"
    BREACHED = "Breached"
    WAIVED = "Waived"
    UNKNOWN_FINAL = "UnknownFinal"


@dataclasses.dataclass
class ResponseToken:
    """A response event's token r = <rid, p, t_r, explicit_oids,
    broadcast_key, consumed> (S1.4)."""

    rid: str
    response_tick: int
    substitution: dict
    explicit_oids: frozenset = dataclasses.field(default_factory=frozenset)
    broadcast_key: str | None = None
    consumed: bool = False


@dataclasses.dataclass
class Occurrence:
    """One lifecycle record keyed by a monitor-local occurrence alias."""

    oid: str
    clause_id: str
    substitution: dict
    trigger_tick: int
    deadline: int
    guard_key: str
    status: OccurrenceStatus
    candidates: list = dataclasses.field(default_factory=list)
    response: ResponseToken | None = None
    breach_reason: str | None = None
    effective_time: int | None = None
    discovery_time: int | None = None


def trigger_and_guard_rule(
    oid: str,
    clause_id: str,
    substitution: dict,
    trigger_tick: int,
    deadline: int,
    guard_key: str,
    guard_value: K3,
) -> Occurrence:
    """S1.4 "Trigger and guard rule": [[phi_sigma@t0]]_eta = T => Open;
    F => Inapplicable; U => ConditionalOpen(kappa)."""
    status = {
        K3.T: OccurrenceStatus.OPEN,
        K3.F: OccurrenceStatus.INAPPLICABLE,
        K3.U: OccurrenceStatus.CONDITIONAL_OPEN,
    }[guard_value]
    return Occurrence(
        oid, clause_id, substitution, trigger_tick, deadline, guard_key, status
    )


def guard_becomes_false(occ: Occurrence) -> None:
    """S1.4 "Retroactive activation / Guard becomes false": no violation is
    emitted; the occurrence becomes Inapplicable."""
    if occ.status not in (
        OccurrenceStatus.CONDITIONAL_OPEN,
        OccurrenceStatus.CONDITIONAL_EXPIRED,
    ):
        raise ValueError("guard_becomes_false applies only to conditional occurrences")
    occ.status = OccurrenceStatus.INAPPLICABLE


def canonical_order_key(occ: Occurrence):
    """Return this fragment's deterministic local allocation order."""
    return (occ.deadline, occ.trigger_tick, occ.clause_id, occ.oid)


def token_order_key(token: ResponseToken):
    """Order tokens by response tick, then by the canonical event occurrence."""
    return (token.response_tick, token.rid)


def guards_become_true(activated, current_tick: int) -> None:
    """Run one allocation pass over the occurrences whose guards became true.

    Retroactive activation allocates once for the whole configuration, not
    independently for each occurrence. This pass therefore takes every newly
    activated occurrence together. It orders them by deadline, then trigger
    tick, then clause identifier, then occurrence identifier, and orders the
    recorded unconsumed tokens by response tick and then by the canonical event
    occurrence. Each token in turn goes to the first occurrence in the
    occurrence order that holds it as a recorded candidate, is still within its
    deadline, and has not yet received a linear token in this pass. That
    occurrence then leaves the pass, and the token is consumed. A token with no
    such occurrence left stays unconsumed.

    Lifecycle changes happen after the allocation pass. An occurrence that
    received a token becomes `Discharged`; one that received none becomes
    `Open` while the current tick is within its deadline, and `Breached`
    otherwise. The deadline and the binding are never reset, so a delayed
    observation records when the breach happened as well as when it was found.

    Both orderings are computed inside the pass, so the outcome does not depend
    on the order in which the caller supplies the occurrences.
    """
    occurrences = list(activated)
    for occ in occurrences:
        if occ.status not in (
            OccurrenceStatus.CONDITIONAL_OPEN,
            OccurrenceStatus.CONDITIONAL_EXPIRED,
        ):
            raise ValueError(
                "guards_become_true applies only to conditional occurrences"
            )

    occurrence_order = sorted(occurrences, key=canonical_order_key)

    recorded: list[ResponseToken] = []
    for occ in occurrence_order:
        for candidate in occ.candidates:
            if candidate.consumed:
                continue
            if any(candidate is token for token in recorded):
                continue
            recorded.append(candidate)
    recorded.sort(key=token_order_key)

    allocation: dict[int, ResponseToken] = {}
    for token in recorded:
        for occ in occurrence_order:
            if id(occ) in allocation:
                continue
            if token.response_tick > occ.deadline:
                continue
            if not any(candidate is token for candidate in occ.candidates):
                continue
            token.consumed = True
            allocation[id(occ)] = token
            break

    for occ in occurrence_order:
        token = allocation.get(id(occ))
        if token is not None:
            occ.response = token
            occ.status = OccurrenceStatus.DISCHARGED
        elif current_tick <= occ.deadline:
            occ.status = OccurrenceStatus.OPEN
        else:
            occ.status = OccurrenceStatus.BREACHED
            occ.breach_reason = "RetroactiveActivationAfterDeadline"
            occ.effective_time = occ.deadline
            occ.discovery_time = current_tick


def guard_remains_unknown_at_deadline(occ: Occurrence) -> None:
    """S1.4 "Retroactive activation / Guard remains unknown at the
    deadline": the occurrence becomes ConditionalExpired; its verdict is
    unknown, not satisfied and not violated; it remains refinable."""
    if occ.status is not OccurrenceStatus.CONDITIONAL_OPEN:
        raise ValueError(
            "guard_remains_unknown_at_deadline applies only to ConditionalOpen"
        )
    occ.status = OccurrenceStatus.CONDITIONAL_EXPIRED


# -- Implemented linear response-token discipline -------------------------


def _matches_binding(token: ResponseToken, occ: Occurrence) -> bool:
    return dict(token.substitution) == dict(occ.substitution)


def register_response_token(token: ResponseToken, all_occurrences: dict) -> None:
    """Allocate one response token in the executable fragment.

    Explicit alias references form a candidate set. The token discharges the
    first eligible Open occurrence in the deterministic local order. If no
    referenced occurrence is eligible, the token is reserved for each
    referenced conditional occurrence. A generic token uses the same local
    order over all eligible Open occurrences. Otherwise, the token remains a
    candidate for each unconsumed conditional occurrence. The monitor does not
    retract or reallocate a discharge.
    """
    if token.consumed:
        return

    if token.explicit_oids:
        referenced = sorted(
            (
                all_occurrences[oid]
                for oid in token.explicit_oids
                if oid in all_occurrences
            ),
            key=canonical_order_key,
        )
        eligible = [
            occ
            for occ in referenced
            if occ.status is OccurrenceStatus.OPEN
            and token.response_tick <= occ.deadline
            and _matches_binding(token, occ)
        ]
        if eligible:
            target = eligible[0]
            token.consumed = True
            target.response = token
            target.status = OccurrenceStatus.DISCHARGED
            return

        for occ in referenced:
            if (
                occ.status
                in (
                    OccurrenceStatus.CONDITIONAL_OPEN,
                    OccurrenceStatus.CONDITIONAL_EXPIRED,
                )
                and token.response_tick <= occ.deadline
                and _matches_binding(token, occ)
            ):
                occ.candidates.append(token)
        return

    open_occurrences = [
        occ
        for occ in all_occurrences.values()
        if occ.status is OccurrenceStatus.OPEN
        and token.response_tick <= occ.deadline
        and _matches_binding(token, occ)
    ]
    eligible = sorted(open_occurrences, key=canonical_order_key)
    if eligible:
        target = eligible[0]
        token.consumed = True
        target.response = token
        target.status = OccurrenceStatus.DISCHARGED
        return

    for occ in all_occurrences.values():
        if (
            occ.status
            in (OccurrenceStatus.CONDITIONAL_OPEN, OccurrenceStatus.CONDITIONAL_EXPIRED)
            and token.response_tick <= occ.deadline
            and _matches_binding(token, occ)
        ):
            occ.candidates.append(token)
