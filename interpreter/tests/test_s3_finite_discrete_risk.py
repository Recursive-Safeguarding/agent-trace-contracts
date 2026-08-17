"""S3 finite-discrete risk arithmetic."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rs_metalang_ref import robust
from rs_metalang_ref.robust import AmbiguitySet, ThreeWayReport


def test_robust_module_has_no_policy_gate_or_action_selector_surface():
    removed_names = {
        "AmbiguityIncoherent",
        "GateResult",
        "NoAdmittedAction",
        "RobustGateOutcome",
        "eliminate_by_proof",
        "robust_gate",
        "select_admitted_action",
    }

    present_names = removed_names.intersection(vars(robust))
    assert present_names == set()


def test_s3_nominal_bad_mass_is_upper_risk_without_contamination():
    ambiguity_set = AmbiguitySet(
        core_bad_mass=0.5,
        gamma_min=0.0,
        gamma_max=0.0,
    )

    assert ambiguity_set.upper_risk() == pytest.approx(0.5)


def test_s3_three_way_report():
    report = ThreeWayReport(p_sat=0.60, p_viol=0.10, p_unk=0.30)

    assert report.p_sat + report.p_viol + report.p_unk == pytest.approx(1.0)
    assert report.current_risk_interval == (pytest.approx(0.10), pytest.approx(0.40))


def test_s3_contamination_contributes_to_finite_discrete_upper_risk():
    ambiguity_set = AmbiguitySet(
        core_bad_mass=0.0,
        gamma_min=0.0,
        gamma_max=0.02,
        outside_model_risk=1.0,
    )

    assert ambiguity_set.upper_risk() == pytest.approx(0.02)


def test_s3_upper_risk_rejects_empty_ambiguity_set():
    ambiguity_set = AmbiguitySet(
        core_bad_mass=0.0,
        gamma_min=0.0,
        gamma_max=0.02,
        outside_model_risk=1.0,
        nonempty=False,
    )

    expected_message = "^upper_risk requires a non-empty ambiguity set$"
    with pytest.raises(ValueError, match=expected_message):
        ambiguity_set.upper_risk()


def test_s3_worked_backup_deletion_risk_arithmetic():
    before_clarification = AmbiguitySet(
        core_bad_mass=0.2,
        gamma_min=0.0,
        gamma_max=0.05,
    )
    after_clarification = AmbiguitySet(
        core_bad_mass=0.0,
        gamma_min=0.0,
        gamma_max=0.05,
    )
    reversible_action = AmbiguitySet(
        core_bad_mass=0.0,
        gamma_min=0.0,
        gamma_max=0.05,
        outside_model_risk=0.0,
    )

    assert before_clarification.upper_risk() == pytest.approx(0.24)
    assert after_clarification.upper_risk() == pytest.approx(0.05)
    assert reversible_action.upper_risk() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("p_sat", "p_viol", "p_unk"),
    [
        (-0.1, 0.2, 0.9),
        (0.2, -0.1, 0.9),
        (0.2, 0.9, -0.1),
    ],
)
def test_s3_three_way_report_rejects_negative_probability_mass(p_sat, p_viol, p_unk):
    with pytest.raises(ValueError):
        ThreeWayReport(p_sat=p_sat, p_viol=p_viol, p_unk=p_unk)


@pytest.mark.parametrize(
    ("p_sat", "p_viol", "p_unk"),
    [
        (0.3, 0.2, 0.4),
        (0.4, 0.3, 0.4),
    ],
)
def test_s3_three_way_report_rejects_total_mass_other_than_one(p_sat, p_viol, p_unk):
    with pytest.raises(ValueError):
        ThreeWayReport(p_sat=p_sat, p_viol=p_viol, p_unk=p_unk)


@pytest.mark.parametrize(
    ("core_bad_mass", "outside_model_risk"),
    [
        (-0.01, 0.5),
        (1.01, 0.5),
        (0.5, -0.01),
        (0.5, 1.01),
    ],
)
def test_s3_ambiguity_set_rejects_probability_mass_outside_unit_interval(
    core_bad_mass,
    outside_model_risk,
):
    with pytest.raises(ValueError):
        AmbiguitySet(
            core_bad_mass=core_bad_mass,
            gamma_min=0.0,
            gamma_max=0.05,
            outside_model_risk=outside_model_risk,
        )


@pytest.mark.parametrize(
    ("gamma_min", "gamma_max"),
    [
        (-0.01, 0.5),
        (0.5, 1.01),
    ],
)
def test_s3_ambiguity_set_rejects_contamination_bound_outside_unit_interval(
    gamma_min,
    gamma_max,
):
    with pytest.raises(ValueError):
        AmbiguitySet(core_bad_mass=0.5, gamma_min=gamma_min, gamma_max=gamma_max)


def test_s3_ambiguity_set_rejects_reversed_contamination_interval():
    with pytest.raises(ValueError):
        AmbiguitySet(core_bad_mass=0.5, gamma_min=0.6, gamma_max=0.2)


@pytest.mark.parametrize("field", ("p_sat", "p_viol", "p_unk"))
@pytest.mark.parametrize(
    "invalid_value",
    (True, "0.5", 0.5 + 0.0j, None, object(), Decimal("0.5")),
    ids=("bool", "str", "complex", "none", "object", "decimal"),
)
def test_s3_three_way_report_rejects_non_real_probability_scalar(field, invalid_value):
    values = {"p_sat": 1.0, "p_viol": 0.0, "p_unk": 0.0}
    values[field] = invalid_value

    with pytest.raises(ValueError, match=rf"^{field} must be a real scalar$"):
        ThreeWayReport(**values)


@pytest.mark.parametrize(
    "field",
    ("core_bad_mass", "outside_model_risk", "gamma_min", "gamma_max"),
)
@pytest.mark.parametrize(
    "invalid_value",
    (True, "0.5", 0.5 + 0.0j, None, object(), Decimal("0.5")),
    ids=("bool", "str", "complex", "none", "object", "decimal"),
)
def test_s3_ambiguity_set_rejects_non_real_probability_scalar(field, invalid_value):
    values = {
        "core_bad_mass": 0.5,
        "gamma_min": 0.0,
        "gamma_max": 0.5,
        "outside_model_risk": 0.5,
    }
    values[field] = invalid_value

    with pytest.raises(ValueError, match=rf"^{field} must be a real scalar$"):
        AmbiguitySet(**values)


def test_s3_scalar_probability_valid_controls_accept_fraction_three_way_report():
    report = ThreeWayReport(
        p_sat=Fraction(1, 2),
        p_viol=Fraction(1, 4),
        p_unk=Fraction(1, 4),
    )

    assert report.current_risk_interval == (Fraction(1, 4), Fraction(1, 2))


def test_s3_scalar_probability_valid_controls_accept_fraction_ambiguity_set():
    ambiguity_set = AmbiguitySet(
        core_bad_mass=Fraction(1, 2),
        gamma_min=Fraction(0, 1),
        gamma_max=Fraction(1, 4),
        outside_model_risk=Fraction(0, 1),
    )

    assert ambiguity_set.upper_risk() == Fraction(1, 2)


@given(
    core_bad_mass=st.floats(min_value=0.0, max_value=1.0),
    outside_model_risk=st.floats(min_value=0.0, max_value=1.0),
    gamma_a=st.floats(min_value=0.0, max_value=1.0),
    gamma_b=st.floats(min_value=0.0, max_value=1.0, exclude_min=True),
)
def test_s3_upper_risk_stays_in_unit_interval_for_valid_scalar_inputs(
    core_bad_mass,
    outside_model_risk,
    gamma_a,
    gamma_b,
):
    gamma_min, gamma_max = sorted((gamma_a, gamma_b))
    ambiguity_set = AmbiguitySet(
        core_bad_mass=core_bad_mass,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        outside_model_risk=outside_model_risk,
    )

    assert 0.0 <= ambiguity_set.upper_risk() <= 1.0
