"""Linear response allocation applies to the explicit-target path as well as
the generic path.

S1.4's response-token discipline
states the default rule without
qualification: "The default rule is linear: one response token discharges at
most one obligation. Eligible open obligations are ordered by deadline, then
trigger tick, then clause identifier, then obligation identifier, and the
earliest in that canonical order receives the token." The licence to discharge
several obligations is granted only to a clause declaring `broadcast(key)`,
whose response carries the matching canonical key, and whose targets are all in
the declared broadcast scope.

The S1.10 acceptance test `linear-response-does-not-discharge-two` exercises only
the generic path (a response with no explicit references). These tests are its
analogue for the path a response takes when it names its targets explicitly,
via the `discharges` field that monitor.py turns into a token's
`explicit_oids`. Nothing in the response-token discipline makes an explicit reference a second
broadcast licence: it reserves a token for an occurrence whose guard is still
unknown, which is a statement about eligibility, not about how many
obligations one token may clear.

Property 5 checks the generic path separately.
"""

from __future__ import annotations

import os
import subprocess
import sys

from rs_metalang_ref.contracts import AfterClauseSpec, BoundUnaryGuard, Linear
from rs_metalang_ref.events import Complete, DomainEvent, TerminalEvent
from rs_metalang_ref.kleene import K3
from rs_metalang_ref.monitor import SingleClauseMonitor
from rs_metalang_ref.obligations import (
    Occurrence,
    OccurrenceStatus,
    ResponseToken,
    canonical_order_key,
    register_response_token,
)
from rs_metalang_ref.verdict import Summary


def _spec(bound: int, clause_id: str = "c1") -> AfterClauseSpec:
    return AfterClauseSpec(
        clause_id=clause_id,
        trigger_tag="export",
        response_tag="approval",
        binding_fields=("x",),
        guard=BoundUnaryGuard("Sensitive", "x"),
        bound=bound,
        discharge=Linear(),
    )


def _statuses(monitor: SingleClauseMonitor) -> list[OccurrenceStatus]:
    return [occ.status for occ in monitor.occurrences.values()]


def _discharged_oids(monitor: SingleClauseMonitor) -> list[str]:
    return sorted(
        oid
        for oid, occ in monitor.occurrences.items()
        if occ.status is OccurrenceStatus.DISCHARGED
    )


# -- Property 1: at most one discharge under Linear, explicit targets -------


def test_p1_linear_explicit_targets_discharge_at_most_one():
    """One response naming two open, in-deadline obligations discharges
    exactly one of them and breaches the other.

    The direct analogue of S1's `linear-response-does-not-discharge-two`,
    with the one difference that the response names its targets.
    """
    monitor = SingleClauseMonitor(_spec(bound=5))
    monitor.set_initial("Sensitive(f)", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    monitor.step(DomainEvent("e1", tick=1, tag="export", fields={"x": "f"}))
    assert sorted(monitor.occurrences) == ["o1", "o2"]
    assert all(occ.status is OccurrenceStatus.OPEN for occ in monitor.occurrences.values())

    monitor.step(
        DomainEvent(
            "e2",
            tick=2,
            tag="approval",
            fields={"x": "f", "discharges": ("o1", "o2")},
        )
    )
    verdict = monitor.step(TerminalEvent("e3", tick=3, kind=Complete()))

    statuses = _statuses(monitor)
    assert statuses.count(OccurrenceStatus.DISCHARGED) == 1
    assert statuses.count(OccurrenceStatus.BREACHED) == 1
    assert verdict.summary is Summary.VIOLATED


# -- Property 2: the survivor is the canonical order, not the iteration or
# -- producer order --------------------------------------------------------


def test_p2a_survivor_is_canonical_order_not_producer_order():
    """The discharged occurrence is the one `canonical_order_key` selects,
    identically to the generic path, even when the producer lists its targets
    in the opposite order.

    Both occurrences are open and in deadline at tick 2. o1 has deadline 5 and
    trigger tick 0; o2 has deadline 6 and trigger tick 1, so o1 is earlier in
    the canonical order (deadline, trigger tick, clause id, oid) and o1 is the
    one that must receive the token. The `discharges` field lists o2 first, so
    an implementation that discharges the first target it is handed picks o2
    and fails here.
    """
    monitor = SingleClauseMonitor(_spec(bound=5))
    monitor.set_initial("Sensitive(f)", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    monitor.step(DomainEvent("e1", tick=1, tag="export", fields={"x": "f"}))

    expected = min(monitor.occurrences.values(), key=canonical_order_key).oid
    assert expected == "o1"

    monitor.step(
        DomainEvent(
            "e2",
            tick=2,
            tag="approval",
            fields={"x": "f", "discharges": ("o2", "o1")},
        )
    )
    verdict = monitor.step(TerminalEvent("e3", tick=3, kind=Complete()))

    assert _discharged_oids(monitor) == [expected]
    assert monitor.occurrences["o2"].status is OccurrenceStatus.BREACHED
    assert verdict.summary is Summary.VIOLATED


def test_p2b_survivor_is_canonical_order_not_oid_order():
    """The same property where canonical order disagrees with oid order.

    Driven at obligations.py's own seam, because monitor.py's single-clause
    engine can only ever produce occurrences whose deadlines increase with
    their oids: there, canonical order and creation order coincide and cannot
    be told apart. Here o2 carries the earlier deadline, so the canonical
    winner is the lexically later oid. The explicit targets form a `frozenset`,
    so this seam has set semantics and carries no producer order. The assertion
    checks that allocation applies canonical order to the unordered target set
    instead of sorting the targets by oid.
    """

    def _occurrence(oid: str, deadline: int) -> Occurrence:
        return Occurrence(
            oid=oid,
            clause_id="c1",
            substitution={"x": "f"},
            trigger_tick=0,
            deadline=deadline,
            guard_key="Sensitive(f)",
            status=OccurrenceStatus.OPEN,
        )

    occurrences = {
        "o1": _occurrence("o1", deadline=9),
        "o2": _occurrence("o2", deadline=5),
    }
    expected = min(occurrences.values(), key=canonical_order_key).oid
    assert expected == "o2"

    token = ResponseToken(
        rid="r1",
        response_tick=3,
        substitution={"x": "f"},
        explicit_oids=frozenset({"o1", "o2"}),
    )
    register_response_token(token, occurrences)

    discharged = sorted(
        oid
        for oid, occ in occurrences.items()
        if occ.status is OccurrenceStatus.DISCHARGED
    )
    assert discharged == [expected]
    assert occurrences["o1"].status is OccurrenceStatus.OPEN


# -- Property 3: determinism across hash seeds -----------------------------


_SEED_DRIVER = """
from rs_metalang_ref.contracts import AfterClauseSpec, BoundUnaryGuard, Linear
from rs_metalang_ref.events import Complete, DomainEvent, TerminalEvent
from rs_metalang_ref.kleene import K3
from rs_metalang_ref.monitor import SingleClauseMonitor
from rs_metalang_ref.obligations import OccurrenceStatus

spec = AfterClauseSpec(
    clause_id="c1",
    trigger_tag="export",
    response_tag="approval",
    binding_fields=("x",),
    guard=BoundUnaryGuard("Sensitive", "x"),
    bound=5,
    discharge=Linear(),
)
monitor = SingleClauseMonitor(spec)
monitor.set_initial("Sensitive(f)", K3.T)
monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
monitor.step(DomainEvent("e1", tick=1, tag="export", fields={"x": "f"}))
monitor.step(DomainEvent("e2", tick=2, tag="export", fields={"x": "f"}))
monitor.step(
    DomainEvent(
        "e3",
        tick=3,
        tag="approval",
        fields={"x": "f", "discharges": tuple(monitor.occurrences)},
    )
)
verdict = monitor.step(TerminalEvent("e4", tick=4, kind=Complete()))
discharged = sorted(
    oid for oid, occ in monitor.occurrences.items()
    if occ.status is OccurrenceStatus.DISCHARGED
)
print(verdict.summary.name + "|" + ",".join(discharged))
"""


def test_p3_determinism_across_hash_seeds():
    """The same trace yields the same verdict and the same discharged
    occurrence under different PYTHONHASHSEED values.

    SingleClauseMonitor._register_response converts the explicit-target
    collection to a `frozenset`. Its iteration order for string members depends
    on hash randomisation, so an implementation that discharges by iterating
    that set makes the discharged occurrence a function of the seed, and the
    occurrences left open to breach follow from it. The monitor's own docstring
    calls it "a total, deterministic reference monitor".

    PYTHONHASHSEED is read once at interpreter start and cannot be varied from
    inside a running process, and the frozenset is built inside production
    code, so this is the one property here that cannot be expressed in-process:
    it uses subprocesses with an explicit environment. Permuting the supplied
    target order in-process (property 2b) is a proxy for the same hazard, and
    it is defeated by any implementation that normalises its input to a set
    before allocating, which is exactly the shape at issue.

    Two clauses, in order. The seeds must agree with each other, and the value
    they agree on must be a single discharge. A deterministic implementation
    that discharges every named target satisfies the first and fails the
    second.
    """
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    results = {}
    for seed in range(6):
        completed = subprocess.run(
            [sys.executable, "-c", _SEED_DRIVER],
            env={**env, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            check=True,
        )
        results[seed] = completed.stdout.strip()

    distinct = sorted(set(results.values()))
    assert len(distinct) == 1, f"outcome varies with PYTHONHASHSEED: {results}"

    summary, discharged = distinct[0].split("|")
    discharged_oids = [oid for oid in discharged.split(",") if oid]
    assert len(discharged_oids) == 1, f"expected one discharge, got {discharged_oids}"
    assert discharged_oids == ["o1"]
    assert summary == Summary.VIOLATED.name


# -- Property 4: three explicit targets, one discharge ---------------------


def test_p4_three_explicit_targets_discharge_exactly_one():
    """Generalises property 1 past two targets."""
    monitor = SingleClauseMonitor(_spec(bound=5))
    monitor.set_initial("Sensitive(f)", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    monitor.step(DomainEvent("e1", tick=1, tag="export", fields={"x": "f"}))
    monitor.step(DomainEvent("e2", tick=2, tag="export", fields={"x": "f"}))
    assert sorted(monitor.occurrences) == ["o1", "o2", "o3"]

    monitor.step(
        DomainEvent(
            "e3",
            tick=3,
            tag="approval",
            fields={"x": "f", "discharges": ("o3", "o2", "o1")},
        )
    )
    verdict = monitor.step(TerminalEvent("e4", tick=4, kind=Complete()))

    statuses = _statuses(monitor)
    assert statuses.count(OccurrenceStatus.DISCHARGED) == 1
    assert statuses.count(OccurrenceStatus.BREACHED) == 2
    assert _discharged_oids(monitor) == ["o1"]
    assert verdict.summary is Summary.VIOLATED


# -- Property 5: the generic path is unchanged ------------------------------


def test_p5_generic_path_discharges_exactly_one():
    """A response carrying no `discharges` field still discharges exactly one
    obligation and breaches the other, and the one it discharges is the
    canonically earliest.

    This checks that the explicit-target path and the generic path have the
    same one-token, one-discharge allocation rule.
    """
    monitor = SingleClauseMonitor(_spec(bound=5))
    monitor.set_initial("Sensitive(f)", K3.T)

    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    monitor.step(DomainEvent("e1", tick=1, tag="export", fields={"x": "f"}))
    monitor.step(DomainEvent("e2", tick=2, tag="approval", fields={"x": "f"}))
    verdict = monitor.step(TerminalEvent("e3", tick=3, kind=Complete()))

    statuses = _statuses(monitor)
    assert statuses.count(OccurrenceStatus.DISCHARGED) == 1
    assert statuses.count(OccurrenceStatus.BREACHED) == 1
    assert _discharged_oids(monitor) == ["o1"]
    assert monitor.occurrences["o1"].response.rid == "e2"
    assert monitor.occurrences["o2"].status is OccurrenceStatus.BREACHED
    assert verdict.summary is Summary.VIOLATED
