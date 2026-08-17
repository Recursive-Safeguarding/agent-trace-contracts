"""The five-verdict result algebra with ComparisonScope.

Transcribes the normative text at the result-algebra constructors, discharge scope, and closed vocabulary: every result
constructor carries an explicit `ComparisonScope` as its first field,
`SearchIncomplete` reports a search that ran but did not complete (with a
structured `PartialSearchReceipt` stating what was and was not searched),
`NoWitnessWithinBound` carries a structured `ExhaustiveSearchReceipt` rather
than a bare string. The reference proof-firewall subset rejects every
proof-premise use of `SearchIncomplete`; its permitted meta-reporting is not
implemented.
"""

from __future__ import annotations

import dataclasses

import pytest

from rs_metalang_ref.firewall import (
    DISCHARGEABLE_BY_NO_WITNESS,
    NEVER_DISCHARGEABLE_BY_NO_WITNESS,
    FirewallViolation,
    discharge_search_incomplete,
    require_typecheck,
    typecheck,
)
from rs_metalang_ref.residual import (
    ComparisonScope,
    Distinguished,
    ExhaustiveSearchReceipt,
    InvalidComparisonRequest,
    NoWitnessWithinBound,
    PartialCoverage,
    PartialSearchReceipt,
    ProvedEquivalent,
    SearchIncomplete,
    UnitDataUnitAdvanceBound,
    Untested,
    bounded_compare,
)


def _scope(**overrides):
    fields = {
        "contract_environment": "C",
        "observation_projection": "TotalObservation_C",
        "continuation_family": "all-words-through-declared-bound",
        "bound": UnitDataUnitAdvanceBound(continuation_length=2),
        "exclusions": (),
    }
    fields.update(overrides)
    return ComparisonScope(**fields)


def _unreachable_residual(word):
    raise AssertionError(
        "bounded_compare must not run any search for an invalid request"
    )


# -- ComparisonScope: frozen dataclass, exactly the five specification fields -----


def test_s5_comparison_scope_is_a_frozen_dataclass_with_five_fields():
    assert dataclasses.is_dataclass(ComparisonScope)
    assert [field.name for field in dataclasses.fields(ComparisonScope)] == [
        "contract_environment",
        "observation_projection",
        "continuation_family",
        "bound",
        "exclusions",
    ]

    scope = _scope()
    with pytest.raises(dataclasses.FrozenInstanceError):
        scope.exclusions = ("something",)


def test_s5_comparison_scope_exclusions_states_content_when_present():
    scope_no_exclusions = _scope(exclusions=())
    assert scope_no_exclusions.exclusions == ()

    scope_with_exclusions = _scope(exclusions=("excluded-action",))
    assert scope_with_exclusions.exclusions == ("excluded-action",)


# -- Receipts: structured dataclasses, exact fields --------------------------


def test_s5_exhaustive_search_receipt_has_exactly_three_fields():
    assert [field.name for field in dataclasses.fields(ExhaustiveSearchReceipt)] == [
        "words_enumerated",
        "max_length",
        "alphabet",
    ]


def test_s5_partial_search_receipt_has_exactly_one_field():
    field_names = {field.name for field in dataclasses.fields(PartialSearchReceipt)}
    assert field_names == {"partial_coverage"}


def test_s5_partial_coverage_has_exactly_two_fields():
    field_names = {field.name for field in dataclasses.fields(PartialCoverage)}
    assert field_names == {"searched", "not_searched"}


# -- Every constructor carries scope as its FIRST field ----------------------


def test_s5_distinguished_carries_scope_as_first_field():
    scope = _scope()
    result = Distinguished(scope, ("a",), "left-obs", "right-obs")
    assert result.scope is scope
    assert result.witness == ("a",)
    assert result.left_observation == "left-obs"
    assert result.right_observation == "right-obs"


def test_s5_proved_equivalent_carries_scope_as_first_field():
    scope = _scope()
    result = ProvedEquivalent(scope, "some-certificate")
    assert result.scope is scope
    assert result.certificate == "some-certificate"


def test_s5_proved_equivalent_has_no_top_level_fragment_field():
    # `fragment` is not a top-level field. The continuation family is recorded
    # in scope.continuation_family.
    with pytest.raises(TypeError):
        ProvedEquivalent(_scope(), "some-certificate", fragment="F")


def test_s5_no_witness_within_bound_carries_scope_as_first_field():
    scope = _scope()
    receipt = ExhaustiveSearchReceipt(
        words_enumerated=7, max_length=2, alphabet=("a", "b")
    )
    result = NoWitnessWithinBound(scope, receipt)
    assert result.scope is scope
    assert result.exhaustive_search_receipt is receipt


def test_s5_no_witness_within_bound_has_no_top_level_bound_field():
    # `bound` is not a top-level field. It is recorded in scope.bound.
    receipt = ExhaustiveSearchReceipt(
        words_enumerated=7, max_length=2, alphabet=("a", "b")
    )
    with pytest.raises(TypeError):
        NoWitnessWithinBound(_scope(), receipt, bound={"continuation_length": 2})


def test_s5_search_incomplete_carries_scope_as_first_field():
    scope = _scope()
    receipt = PartialSearchReceipt(
        partial_coverage=PartialCoverage(searched=3, not_searched=4)
    )
    result = SearchIncomplete(scope, "WitnessSearchInterrupted", receipt)
    assert result.scope is scope
    assert result.reason == "WitnessSearchInterrupted"
    assert result.incomplete_comparison_receipt is receipt
    assert not hasattr(result, "partial_search_receipt")


def test_s5_untested_carries_scope_as_first_field():
    scope = _scope()
    result = Untested(scope, "NoComparisonProcedureBegan")
    assert result.scope is scope
    assert result.reason == "NoComparisonProcedureBegan"


# -- bounded_compare (max_words=None): unchanged complete-enumeration path --
# -- returns a structured receipt and a scope, never a bare string ----------


def test_s5_bounded_compare_no_witness_returns_structured_exhaustive_receipt_with_correct_word_count():
    def residual(word):
        return "same"

    result = bounded_compare(
        residual,
        residual,
        alphabet=("approval", "Complete"),
        bound=UnitDataUnitAdvanceBound(continuation_length=2),
        contract_environment="C",
        observation_projection="identity[str]",
        continuation_family="all-words-through-declared-bound",
    )

    assert isinstance(result, NoWitnessWithinBound)
    assert isinstance(result.scope, ComparisonScope)
    assert result.scope.bound == UnitDataUnitAdvanceBound(continuation_length=2)
    assert result.scope.exclusions == ()
    assert isinstance(result.scope.continuation_family, str)
    assert isinstance(result.scope.observation_projection, str)

    receipt = result.exhaustive_search_receipt
    assert not isinstance(receipt, str)
    assert isinstance(receipt, ExhaustiveSearchReceipt)
    # alphabet size 2, max_length 2: 2**0 + 2**1 + 2**2 = 7 words declared.
    assert receipt.words_enumerated == 7
    assert receipt.max_length == 2
    assert receipt.alphabet == ("approval", "Complete")


def test_s5_bounded_compare_distinguished_carries_a_scope():
    def residual_u(word):
        return "flagged" if len(word) >= 1 else "clear"

    def residual_v(word):
        return "clear"

    result = bounded_compare(
        residual_u,
        residual_v,
        alphabet=("a",),
        bound=UnitDataUnitAdvanceBound(continuation_length=1),
        contract_environment="C",
        observation_projection="identity[str]",
        continuation_family="all-words-through-declared-bound",
    )

    assert isinstance(result, Distinguished)
    assert isinstance(result.scope, ComparisonScope)
    assert result.scope.bound == UnitDataUnitAdvanceBound(continuation_length=1)
    assert result.scope.exclusions == ()
    assert result.witness == ("a",)


# -- bounded_compare(..., max_words=N): SearchIncomplete on budget exhaustion


def test_s5_bounded_compare_word_budget_exhausted_returns_search_incomplete_with_hand_computed_partial_coverage():
    def residual(word):
        return "same"

    result = bounded_compare(
        residual,
        residual,
        alphabet=("a", "b"),
        bound=UnitDataUnitAdvanceBound(continuation_length=2),
        max_words=3,
        contract_environment="C",
        observation_projection="identity[str]",
        continuation_family="all-words-through-declared-bound",
    )

    assert isinstance(result, SearchIncomplete)
    assert isinstance(result.scope, ComparisonScope)
    assert result.reason == "WordBudgetExhausted"

    receipt = result.incomplete_comparison_receipt
    assert not isinstance(receipt, str)
    assert isinstance(receipt, PartialSearchReceipt)

    coverage = receipt.partial_coverage
    assert isinstance(coverage, PartialCoverage)
    # Hand-computed: alphabet=("a","b"), max_length=2 declares 2**0 + 2**1 +
    # 2**2 = 7 words (one length-0, two length-1, four length-2). residual is
    # constant, so no witness is ever found; a budget of 3 is exhausted after
    # the first 3 enumerated words, leaving 7 - 3 = 4 unsearched.
    assert coverage.searched == 3
    assert coverage.not_searched == 4


def test_s5_bounded_compare_max_words_covering_the_full_domain_returns_no_witness_within_bound():
    def residual(word):
        return "same"

    # alphabet=("a",), max_length=2 declares 1 + 1 + 1 = 3 words (one word per
    # length, since the alphabet has a single symbol). A budget of 3 exactly
    # covers the whole declared domain, so enumeration does not EXCEED the
    # budget and the search completes -- this must stay NoWitnessWithinBound,
    # not SearchIncomplete.
    result = bounded_compare(
        residual,
        residual,
        alphabet=("a",),
        bound=UnitDataUnitAdvanceBound(continuation_length=2),
        max_words=3,
        contract_environment="C",
        observation_projection="identity[str]",
        continuation_family="all-words-through-declared-bound",
    )

    assert isinstance(result, NoWitnessWithinBound)


def test_s5_bounded_compare_max_words_none_is_still_valid_and_unbudgeted():
    def residual(word):
        return "same"

    result = bounded_compare(
        residual,
        residual,
        alphabet=("a",),
        bound=UnitDataUnitAdvanceBound(continuation_length=2),
        max_words=None,
        contract_environment="C",
        observation_projection="identity[str]",
        continuation_family="all-words-through-declared-bound",
    )
    assert isinstance(result, NoWitnessWithinBound)


# -- bounded_compare(..., max_words=...): refuses a non-positive-int budget -


def test_s5_bounded_compare_zero_max_words_raises_invalid_comparison_request():
    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            _unreachable_residual,
            _unreachable_residual,
            alphabet=("a",),
            bound=UnitDataUnitAdvanceBound(continuation_length=2),
            max_words=0,
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )


def test_s5_bounded_compare_negative_max_words_raises_invalid_comparison_request():
    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            _unreachable_residual,
            _unreachable_residual,
            alphabet=("a",),
            bound=UnitDataUnitAdvanceBound(continuation_length=2),
            max_words=-1,
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )


def test_s5_bounded_compare_float_max_words_raises_invalid_comparison_request():
    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            _unreachable_residual,
            _unreachable_residual,
            alphabet=("a",),
            bound=UnitDataUnitAdvanceBound(continuation_length=2),
            max_words=2.5,
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )


def test_s5_bounded_compare_string_max_words_raises_invalid_comparison_request():
    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            _unreachable_residual,
            _unreachable_residual,
            alphabet=("a",),
            bound=UnitDataUnitAdvanceBound(continuation_length=2),
            max_words="3",
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )


def test_s5_bounded_compare_bool_max_words_raises_invalid_comparison_request():
    # bool subclasses int in Python; True/False must not silently pass as 1/0.
    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            _unreachable_residual,
            _unreachable_residual,
            alphabet=("a",),
            bound=UnitDataUnitAdvanceBound(continuation_length=2),
            max_words=True,
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )


# -- UnitDataUnitAdvanceBound: refuses a negative or bool length ------------


def test_s5_unit_advance_bound_negative_length_raises_invalid_comparison_request():
    with pytest.raises(InvalidComparisonRequest):
        UnitDataUnitAdvanceBound(continuation_length=-1)


def test_s5_unit_advance_bound_bool_length_raises_invalid_comparison_request():
    # bool subclasses int in Python; True/False must not silently pass as 1/0.
    with pytest.raises(InvalidComparisonRequest):
        UnitDataUnitAdvanceBound(continuation_length=True)


# -- firewall: SearchIncomplete can never fill a simulation certificate -----


def test_s5_typecheck_search_incomplete_cannot_fill_simulation_certificate():
    assert typecheck("SearchIncomplete", "SimulationAndLabellingCertificate") is False


def test_s5_require_typecheck_search_incomplete_raises_firewall_violation():
    with pytest.raises(FirewallViolation) as excinfo:
        require_typecheck("SearchIncomplete", "SimulationAndLabellingCertificate")
    assert excinfo.value.code == "E-PROOF-FIREWALL"


# -- firewall subset: SearchIncomplete meta-reporting is not implemented ----


def test_s5_reference_firewall_does_not_implement_search_incomplete_meta_reporting():
    propositions = (
        DISCHARGEABLE_BY_NO_WITNESS
        | NEVER_DISCHARGEABLE_BY_NO_WITNESS
        | {"TestStatus", "SomeUnnamedProposition"}
    )
    assert propositions  # sanity: the union is non-empty

    for proposition in sorted(propositions):
        with pytest.raises(FirewallViolation):
            discharge_search_incomplete(proposition)
