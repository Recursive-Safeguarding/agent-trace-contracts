"""The two candidate maps into the capsule's declared state-card syntax."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from rs_metalang_ref.obligations import OccurrenceStatus

from .profile import OccurrenceRecord, SourcePrefix, StateCardTerm


@dataclasses.dataclass(frozen=True)
class CandidateCard:
    card_id: str
    transform: Callable[[SourcePrefix], StateCardTerm]

    def __call__(self, prefix: SourcePrefix) -> StateCardTerm:
        return self.transform(prefix)


def truncation_card(budget: int) -> CandidateCard:
    if type(budget) is not int:
        raise ValueError("budget must be an integer")
    if budget < 0:
        raise ValueError("budget must be non-negative")

    def transform(prefix: SourcePrefix) -> StateCardTerm:
        retained = tuple(prefix.trace[-budget:]) if budget else ()
        return StateCardTerm(retained_events=retained, occurrence_table=())

    return CandidateCard(card_id=f"tail-window-{budget}-v1", transform=transform)


def open_obligation_card() -> CandidateCard:
    def transform(prefix: SourcePrefix) -> StateCardTerm:
        records = tuple(
            OccurrenceRecord(
                oid=occurrence.oid,
                trigger_tick=occurrence.trigger_tick,
                deadline=occurrence.deadline,
            )
            for occurrence in prefix.monitor.occurrences.values()
            if occurrence.status is OccurrenceStatus.OPEN
        )
        return StateCardTerm(retained_events=(), occurrence_table=records)

    return CandidateCard(card_id="obligation-table-v1", transform=transform)
