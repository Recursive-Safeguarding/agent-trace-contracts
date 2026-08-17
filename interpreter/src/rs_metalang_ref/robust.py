"""Finite-discrete robust-risk arithmetic (S3, "the joint robust object and
gate").

`AmbiguitySet` specializes the S3.2 ambiguity set to the
finite discrete case that every test and worked example for the local S3 family
actually uses: a scalar `core_bad_mass` (the core's supremum probability mass
on the bad event) and a contamination fraction gamma ranging over
[gamma_min, gamma_max], with the outside-model component assigning
`outside_model_risk` (the specified risk for an unsupported high-impact
prediction is safety-critical unknown, i.e. risk 1.0 by default) to the
bad event.

Implementation boundary: The general construction is a set of joint measures over an
uncountable space with an explicit rectangularity declaration; representing
that generally (arbitrary priors/likelihoods, arbitrary contamination
measures) is a much larger modelling task the specification does not
reduce to one concrete data structure. The finite-discrete specialization
below implements the S3 acceptance-test and worked-example cases. It reproduces their
expected numeric outputs exactly.
"""

from __future__ import annotations

import dataclasses
import math
from numbers import Real


def _require_real_scalar(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a real scalar")


def _require_unit_interval(name: str, value: float) -> None:
    _require_real_scalar(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}")


@dataclasses.dataclass(frozen=True)
class ThreeWayReport:
    """The reporting triple (S3.3, "three-way reporting") for a declared
    nominal posterior. Its values must sum to 1."""

    p_sat: float
    p_viol: float
    p_unk: float

    def __post_init__(self) -> None:
        _require_unit_interval("p_sat", self.p_sat)
        _require_unit_interval("p_viol", self.p_viol)
        _require_unit_interval("p_unk", self.p_unk)
        total = self.p_sat + self.p_viol + self.p_unk
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"S3.3 requires p_sat + p_viol + p_unk == 1, got {total}")

    @property
    def current_risk_interval(self) -> tuple:
        """Return the current-trace risk interval from the S3.3 three-way
        reporting rule."""
        return (self.p_viol, self.p_viol + self.p_unk)


@dataclasses.dataclass(frozen=True)
class AmbiguitySet:
    core_bad_mass: float
    gamma_min: float
    gamma_max: float
    outside_model_risk: float = 1.0
    nonempty: bool = True

    def __post_init__(self) -> None:
        if type(self.nonempty) is not bool:
            raise ValueError("nonempty must be a bool")
        _require_unit_interval("core_bad_mass", self.core_bad_mass)
        _require_unit_interval("outside_model_risk", self.outside_model_risk)
        _require_unit_interval("gamma_min", self.gamma_min)
        _require_unit_interval("gamma_max", self.gamma_max)
        if self.gamma_min > self.gamma_max:
            raise ValueError("gamma_min must not exceed gamma_max")

    def upper_risk(self) -> float:
        """Return the S3.4 interventional-action-risk bound for this finite
        case:
        R-bar(a) = sup over gamma in [gamma_min, gamma_max] of
        (1-gamma)*core_bad_mass + gamma*outside_model_risk. This is linear in
        gamma, so the supremum is attained at one of the two endpoints."""

        if not self.nonempty:
            raise ValueError("upper_risk requires a non-empty ambiguity set")

        def at(gamma: float) -> float:
            return (1 - gamma) * self.core_bad_mass + gamma * self.outside_model_risk

        return max(at(self.gamma_min), at(self.gamma_max))
