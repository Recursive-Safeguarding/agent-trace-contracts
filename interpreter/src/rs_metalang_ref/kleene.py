"""Strong-Kleene three-valued epistemic truth domain: the three-valued
epistemic layer used throughout the typed contract semantics (S1.2).

Fully specified by the specification text: negation, conjunction, disjunction,
implication-as-sugar, and the information order U <= T, U <= F with T and F
incomparable.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum


class K3(Enum):
    """The three-valued truth domain {T, F, U} (S1.2)."""

    T = "T"
    F = "F"
    U = "U"

    def __invert__(self) -> K3:
        return _NEGATION[self]

    def __and__(self, other: K3) -> K3:
        if not isinstance(other, K3):
            return NotImplemented
        if self is K3.F or other is K3.F:
            return K3.F
        if self is K3.U or other is K3.U:
            return K3.U
        return K3.T

    def __or__(self, other: K3) -> K3:
        if not isinstance(other, K3):
            return NotImplemented
        if self is K3.T or other is K3.T:
            return K3.T
        if self is K3.U or other is K3.U:
            return K3.U
        return K3.F

    def implies(self, other: K3) -> K3:
        """P => Q := not P or Q (S1.2)."""
        return (~self) | other


_NEGATION = {K3.T: K3.F, K3.F: K3.T, K3.U: K3.U}


def conjoin(values: Iterable[K3]) -> K3:
    """Finite quantification (universal): the fold of conjunction (S1.2)."""
    result = K3.T
    for value in values:
        result = result & value
    return result


def disjoin(values: Iterable[K3]) -> K3:
    """Finite quantification (existential): the fold of disjunction (S1.2)."""
    result = K3.F
    for value in values:
        result = result | value
    return result


def refines(prior: K3, updated: K3) -> bool:
    """The information order (S1.2): a same-key update may only go
    U -> T, U -> F, or stay unchanged. T -> F, F -> T, and any -> U are
    epistemic retractions and are never valid refinements of the same key.
    """
    if type(prior) is not K3 or type(updated) is not K3:
        raise ValueError("prior and updated must be K3 values")
    if prior is updated:
        return True
    return prior is K3.U


class InconsistentObservation(Exception):
    """Raised when an epistemic update on the SAME proposition key would
    retract an already-established truth value (S1.2's contradictory-update
    rule): the monitor 'does not silently choose either truth value' and
    instead surfaces a typed MonitorFault (see monitor.py)."""

    def __init__(self, proposition_key: str, prior_value: K3, attempted_value: K3):
        self.proposition_key = proposition_key
        self.prior_value = prior_value
        self.attempted_value = attempted_value
        super().__init__(
            f"InconsistentObservation({proposition_key}): "
            f"prior={prior_value.value} attempted={attempted_value.value}"
        )
