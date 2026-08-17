"""Public verdict object and prospective-permission judgement: the contract
observation, the aggregate summary rule, and prospective permission (S1.3, S1.8).

`VerdictObject` (S1.3) is a typed record with a
`summary: Satisfied | Violated | Unknown` field plus per-obligation lifecycle
detail.

`Unknown` here is a CONTRACT VERDICT (S1.3): "It is not NoWitnessWithinBound,
which is meta-level proof-search evidence defined in S5 and S6." The two
notions live in separate modules (verdict.py vs residual.py).
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Optional

import pydantic


class Summary(str, Enum):
    SATISFIED = "Satisfied"
    VIOLATED = "Violated"
    UNKNOWN = "Unknown"


class Mode(str, Enum):
    RUNNING = "Running"
    COMPLETE = "Complete"
    AGENT_ABORT = "AgentAbort"
    EXTERNAL_CRASH = "ExternalCrash"
    TIMEOUT = "Timeout"
    OBSERVATION_CUT = "ObservationCut"
    FAULTED = "Faulted"


class VerdictObject(pydantic.BaseModel):
    """Public result for the implemented single-`after`-clause lifecycle.

    The `occurrences` map uses each trace-local canonical alias, such as `o1`,
    as a key. Each alias is a monitor-local presentation identifier. The
    normative structured identity and its canonical alias mapping are outside
    this executable fragment.

    This object contains a summary, mode, occurrence states, and an optional
    diagnostic. It does not implement the other normative verdict fields.
    """

    summary: Summary
    mode: Mode
    occurrences: dict[str, str] = pydantic.Field(default_factory=dict)
    diagnostic: Optional[str] = None

    model_config = pydantic.ConfigDict(frozen=True, strict=True)


# -- S1.8: prospective fail-closed permission -----------------------------


@dataclasses.dataclass(frozen=True)
class Allow:
    pass


@dataclasses.dataclass(frozen=True)
class DenyProved:
    reason: str


@dataclasses.dataclass(frozen=True)
class DenyUnknown:
    reason: str


@dataclasses.dataclass(frozen=True)
class Disabled:
    enabled_set: frozenset


@dataclasses.dataclass(frozen=True)
class Abstain:
    reason: str


PermissionResult = Allow | DenyProved | DenyUnknown | Disabled | Abstain
