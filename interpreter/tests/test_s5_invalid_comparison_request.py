"""The public unit advance bound rejects invalid continuation lengths."""

from __future__ import annotations

import pytest

from rs_metalang_ref.residual import (
    Distinguished,
    InvalidComparisonRequest,
    NoWitnessWithinBound,
    UnitDataUnitAdvanceBound,
    bounded_compare,
)


@pytest.mark.parametrize(
    "continuation_length",
    [
        pytest.param(-1, id="negative"),
        pytest.param(2.5, id="float"),
        pytest.param("3", id="string"),
        pytest.param(True, id="bool"),
    ],
)
def test_s5_unit_advance_bound_refuses_an_invalid_continuation_length(
    continuation_length,
):
    with pytest.raises(InvalidComparisonRequest):
        UnitDataUnitAdvanceBound(continuation_length=continuation_length)


def test_s5_zero_continuation_length_searches_the_empty_word_only():
    def residual(word):
        return "same"

    bound = UnitDataUnitAdvanceBound(continuation_length=0)
    result = bounded_compare(
        residual,
        residual,
        alphabet=("approval",),
        bound=bound,
        contract_environment="C",
        observation_projection="identity[str]",
        continuation_family="all-words-through-declared-bound",
    )

    assert isinstance(result, NoWitnessWithinBound)
    assert result.scope.bound is bound
    assert result.exhaustive_search_receipt.words_enumerated == 1


def test_s5_witness_at_length_one_still_distinguishes():
    def residual_u(word):
        return "flagged" if len(word) >= 1 else "clear"

    def residual_v(word):
        return "clear"

    result = bounded_compare(
        residual_u,
        residual_v,
        alphabet=("approval",),
        bound=UnitDataUnitAdvanceBound(continuation_length=1),
        contract_environment="C",
        observation_projection="identity[str]",
        continuation_family="all-words-through-declared-bound",
    )

    assert isinstance(result, Distinguished)
    assert result.witness == ("approval",)
