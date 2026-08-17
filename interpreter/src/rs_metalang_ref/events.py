"""Normalized monitor input events (S1.1, "typed events").

    Event ::=
        DomainEvent(event_id, tick, tag, fields, effect_receipt, observations)
      | Tick(event_id, tick, observations)
      | Terminal(event_id, tick, TerminalKind, observations)
      | Malformed(raw_digest, error_code)

    TerminalKind ::=
        Complete
      | AgentAbort(reason)
      | ExternalCrash(cause)
      | Timeout(horizon_id)
      | ObservationCut(source)

This module models the grammar structurally. Ticks must be strictly
increasing (enforced by monitor.py's transition order, not here, since
monotonicity is a property of a sequence of events, not of one event).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .kleene import K3

# -- TerminalKind -------------------------------------------------------


@dataclass(frozen=True)
class Complete:
    pass


@dataclass(frozen=True)
class AgentAbort:
    reason: str


@dataclass(frozen=True)
class ExternalCrash:
    cause: str


@dataclass(frozen=True)
class Timeout:
    horizon_id: str


@dataclass(frozen=True)
class ObservationCut:
    source: str


TerminalKind = Complete | AgentAbort | ExternalCrash | Timeout | ObservationCut


# -- Event ---------------------------------------------------------------


@dataclass(frozen=True)
class DomainEvent:
    event_id: str
    tick: int
    tag: str
    fields: Mapping[str, Any] = field(default_factory=dict)
    effect_receipt: Any = None
    observations: Mapping[str, K3] = field(default_factory=dict)


@dataclass(frozen=True)
class TickEvent:
    event_id: str
    tick: int
    observations: Mapping[str, K3] = field(default_factory=dict)


@dataclass(frozen=True)
class TerminalEvent:
    event_id: str
    tick: int
    kind: TerminalKind
    observations: Mapping[str, K3] = field(default_factory=dict)


@dataclass(frozen=True)
class MalformedEvent:
    raw_digest: str
    error_code: str


Event = DomainEvent | TickEvent | TerminalEvent | MalformedEvent
