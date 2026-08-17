"""Contracts for observation digests and comparison alphabets.

Two requirements are checked here:

* ``canonical_observation_digest`` must accept only a closed domain of
  admissible values and refuse everything else with ``ValueError`` naming the
  offending type. It does not use a generic object serialiser or repr-based key,
  so a value that ``_structurally_equal`` would compare through permissive
  same-type ``__eq__`` is refused when the digest cannot canonicalise it.
  Floats are admissible only when finite.
* ``bounded_compare`` must validate its ``alphabet`` before running any
  search: a finite non-string ``Sequence`` snapshotted to a tuple, every element
  a ``str``, elements pairwise distinct. Violations raise
  ``InvalidComparisonRequest`` naming the invalid alphabet, before ``residual_u`` or
  ``residual_v`` is ever called. The empty alphabet stays valid.

This file also checks that ``InvalidComparisonRequest`` is a ``ValueError``
subclass, and that ``alphabet=()`` stays valid.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import re
from collections.abc import Mapping

import pytest

import rs_metalang_ref
import rs_metalang_ref.residual as residual_module
from rs_metalang_ref.contracts import AfterClauseSpec, BoundUnaryGuard, Linear
from rs_metalang_ref.events import Complete, DomainEvent, TerminalEvent
from rs_metalang_ref.kleene import K3
from rs_metalang_ref.monitor import SingleClauseMonitor
from rs_metalang_ref.residual import (
    ComparisonScope,
    DisabledReason,
    Distinguished,
    InvalidComparisonRequest,
    NoWitnessWithinBound,
    TotalizedLTS,
    UnitDataDomain,
    UnitDataUnitAdvanceBound,
    bounded_compare,
    canonical_observation_digest,
    monitor_residual,
)


def test_s5_public_module_exposes_the_unit_advance_bound_type():
    """The public comparison seam provides its declared bound type."""
    residual = importlib.import_module("rs_metalang_ref.residual")

    assert hasattr(residual, "UnitDataUnitAdvanceBound")


def test_s5_public_package_module_map_limits_residual_to_bounded_comparison():
    """The public module map states the executable comparison boundary."""
    package_doc = rs_metalang_ref.__doc__

    assert package_doc is not None
    module_map = package_doc.splitlines()
    assert (
        "    residual    -- S5 total residual definition and bounded comparison"
        in module_map
    )
    assert (
        "    residual    -- S5 total residual definition and congruence"
        not in module_map
    )


def test_s5_unit_data_domain_has_frozen_declared_fields():
    assert [field.name for field in dataclasses.fields(UnitDataDomain)] == [
        "kind",
        "note",
    ]

    domain = UnitDataDomain()
    assert domain.kind == "unit"
    assert domain.note == "contract declares no binding fields"
    with pytest.raises(dataclasses.FrozenInstanceError):
        domain.kind = "other"


def test_s5_unit_advance_bound_derives_the_unit_data_and_timing_bound():
    assert [field.name for field in dataclasses.fields(UnitDataUnitAdvanceBound)] == [
        "continuation_length",
        "data_domain",
        "timing_bound",
    ]

    bound = UnitDataUnitAdvanceBound(continuation_length=2)
    assert bound.continuation_length == 2
    assert bound.data_domain == UnitDataDomain()
    assert bound.timing_bound == 2
    assert tuple(inspect.signature(UnitDataUnitAdvanceBound).parameters) == (
        "continuation_length",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        bound.continuation_length = 3


def test_s5_bounded_compare_has_only_the_narrowed_public_parameters():
    signature = inspect.signature(bounded_compare)

    assert tuple(signature.parameters) == (
        "residual_u",
        "residual_v",
        "alphabet",
        "bound",
        "max_words",
        "contract_environment",
        "observation_projection",
        "continuation_family",
    )
    assert signature.parameters["max_words"].default is None
    assert signature.parameters["bound"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["contract_environment"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "max_length" not in signature.parameters
    assert "data_domain" not in signature.parameters
    assert "timing_bound" not in signature.parameters


def test_s5_bounded_compare_rejects_a_bound_subclass_before_callbacks():
    class DerivedBound(UnitDataUnitAdvanceBound):
        pass

    calls = []

    def residual(word):
        calls.append(word)
        return "same"

    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            residual,
            residual,
            alphabet=("approval",),
            bound=DerivedBound(continuation_length=1),
            contract_environment="C",
            observation_projection="identity[str]",
            continuation_family="all-words-through-declared-bound",
        )

    assert calls == []


def _unreachable_residual(word):
    """A residual function that fails the test if the search loop ever runs.

    ``bounded_compare`` must validate ``alphabet`` before any search; wiring
    both sides of a comparison to this function lets the invalid-alphabet
    tests below confirm that directly, matching the pattern already
    established for ``bound``/``max_words`` validation in
    ``tests/test_s5_invalid_comparison_request.py`` and
    ``tests/test_s5_result_scope.py``.
    """
    raise AssertionError(
        "bounded_compare must not run any search for an invalid request"
    )


class _PermissiveEquality:
    """A same-type object outside the canonical digest domain.

    The failure mode this guards against: ``_structurally_equal`` falls back
    to same-type ``__eq__`` for a value like this one, and this ``__eq__`` is
    permissive (it ignores ``tag`` entirely), so two differently-tagged
    instances count as structurally equal -- yet nothing in the canonical
    digest domain can represent this type at all. The digest is required to
    refuse it outright with ``ValueError`` rather than leak whatever
    exception a generic object path would raise, or worse, silently serialise it
    by some equality-blind key.
    """

    def __init__(self, tag):
        self.tag = tag

    def __eq__(self, other):
        return type(other) is type(self)

    def __hash__(self):
        return 0


# == canonical_observation_digest: equality-closed domain =====================


def test_s5_canonical_observation_digest_refuses_a_value_outside_the_closed_domain():
    left = _PermissiveEquality("a")
    right = _PermissiveEquality("b")
    assert left == right  # precondition: permissive same-type __eq__

    with pytest.raises(ValueError):
        canonical_observation_digest(left)


def test_s5_canonical_observation_digest_refuses_nan_float():
    with pytest.raises(ValueError):
        canonical_observation_digest(float("nan"))


def test_s5_canonical_observation_digest_refuses_infinite_float():
    with pytest.raises(ValueError):
        canonical_observation_digest(float("inf"))


def test_s5_canonical_observation_digest_still_accepts_the_admissible_domain():
    admissible_values = (
        None,
        True,
        False,
        0,
        42,
        -7,
        "hello",
        3.14,
        ("a", "b"),
        frozenset({"a", "b"}),
        {"k": "v"},
        DisabledReason.NOT_ENABLED,
    )
    for value in admissible_values:
        digest = canonical_observation_digest(value)
        assert isinstance(digest, str)
        assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_s5_canonical_observation_digest_still_agrees_for_structurally_equal_admissible_observations():
    left = {"reason": "AfterFault", "offset": 3, "labels": ("a", "b")}
    right = {"labels": ("a", "b"), "offset": 3, "reason": "AfterFault"}
    assert left == right

    assert canonical_observation_digest(left) == canonical_observation_digest(right)


# == bounded_compare: alphabet contract ========================================


def test_s5_alphabet_none_raises_invalid_comparison_request_before_any_search():
    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            _unreachable_residual,
            _unreachable_residual,
            alphabet=None,
            bound=UnitDataUnitAdvanceBound(continuation_length=1),
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )


def test_s5_alphabet_int_element_raises_invalid_comparison_request_before_any_search():
    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            _unreachable_residual,
            _unreachable_residual,
            alphabet=("a", 1),
            bound=UnitDataUnitAdvanceBound(continuation_length=1),
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )


def test_s5_alphabet_bytes_element_raises_invalid_comparison_request_before_any_search():
    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            _unreachable_residual,
            _unreachable_residual,
            alphabet=("a", b"b"),
            bound=UnitDataUnitAdvanceBound(continuation_length=1),
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )


def test_s5_alphabet_duplicate_raises_invalid_comparison_request_before_any_search():
    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            _unreachable_residual,
            _unreachable_residual,
            alphabet=("a", "a"),
            bound=UnitDataUnitAdvanceBound(continuation_length=1),
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )


def test_s5_alphabet_empty_tuple_is_valid_and_searches_only_the_empty_word():
    def residual(word):
        return "same"

    result = bounded_compare(
        residual,
        residual,
        alphabet=(),
        bound=UnitDataUnitAdvanceBound(continuation_length=1),
        contract_environment="C",
        observation_projection="identity[str]",
        continuation_family="all-words-through-declared-bound",
    )

    assert isinstance(result, NoWitnessWithinBound)
    assert result.exhaustive_search_receipt.words_enumerated == 1
    assert result.exhaustive_search_receipt.alphabet == ()


def test_s5_alphabet_duplicate_free_case_still_searches_correctly():
    def residual_u(word):
        return "flagged" if len(word) >= 1 else "clear"

    def residual_v(word):
        return "clear"

    result = bounded_compare(
        residual_u,
        residual_v,
        alphabet=("a", "b"),
        bound=UnitDataUnitAdvanceBound(continuation_length=1),
        contract_environment="C",
        observation_projection="identity[str]",
        continuation_family="all-words-through-declared-bound",
    )

    assert isinstance(result, Distinguished)
    assert result.witness == ("a",)


# == InvalidComparisonRequest: exception hierarchy ============================


def test_s5_invalid_comparison_request_is_a_value_error_subclass():
    assert issubclass(InvalidComparisonRequest, ValueError)


# ==============================================================================
# == Further contract boundaries against the normative text at
# == the residual definition: comparison-domain closure for NaN
# == observations, Sequence-only alphabet admission,
# == and the exact closed observation grammar the digest and bounded_compare
# == both admit.
# ==============================================================================


def _real_monitor() -> SingleClauseMonitor:
    """A monitor stepped to a real, terminal VerdictObject: a REAL
    VerdictObject-based observation, built via the monitor exactly as
    tests/test_s5_acceptance_disabled_sink_retains_observation.py does (same
    contract and prefix as that file's ``_left_prefix``), not a synthetic one.
    """
    contract = AfterClauseSpec(
        clause_id="c1",
        trigger_tag="export",
        response_tag="approval",
        binding_fields=("x",),
        guard=BoundUnaryGuard("Sensitive", "x"),
        bound=2,
        discharge=Linear(),
    )
    monitor = SingleClauseMonitor(contract)
    monitor.set_initial("Sensitive(f)", K3.T)
    monitor.step(DomainEvent("e0", tick=0, tag="export", fields={"x": "f"}))
    monitor.step(TerminalEvent("e1", tick=3, kind=Complete()))
    return monitor


# -- Comparison-domain closure for NaN observations ---------------------------


def test_s5_bounded_compare_refuses_same_callback_nan_observation_not_distinguished():
    # Both sides are the same callback and always return NaN. With an empty
    # alphabet and a zero-length bound, the callback runs once on the empty
    # word; this isolates observation validation.
    def residual(word):
        return float("nan")

    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            residual,
            residual,
            alphabet=(),
            bound=UnitDataUnitAdvanceBound(continuation_length=0),
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )


def test_s5_supplied_continuation_agreement_over_a_nan_observation_raises():
    # The finite agreement predicate must validate both residuals before it
    # compares their encodings. It raises ValueError because it receives a
    # supplied continuation list rather than a bounded comparison request. The
    # empty continuation ``()`` means delta_bar is never invoked, so
    # transition/enabled fail loudly if an implementation calls them.
    def transition(state, label):
        raise AssertionError("transition must not run for an empty continuation")

    def enabled(state):
        raise AssertionError("enabled must not run for an empty continuation")

    def observe(state):
        return float("nan")

    lts = TotalizedLTS(transition, enabled, observe)

    with pytest.raises(ValueError):
        lts.agrees_on_supplied_continuations("s0", "s0", [()])


# -- Alphabet admission is Sequence-only ----------------------------------------


def test_s5_alphabet_generator_rejected_without_consuming():
    # A small finite generator demonstrates that the Sequence-only alphabet
    # rule rejects generators before iteration. Generators lack
    # __len__/__getitem__, so
    # the Sequence-only rule requires outright rejection without ever
    # iterating it: an implementation that silently accepted a finite
    # generator (tuple(alphabet) would consume it) would let the search run,
    # so _unreachable_residual would fire instead of InvalidComparisonRequest
    # -- itself evidence the generator was consumed.
    def small_generator():
        yield "a"
        yield "b"

    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            _unreachable_residual,
            _unreachable_residual,
            alphabet=small_generator(),
            bound=UnitDataUnitAdvanceBound(continuation_length=1),
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )


def test_s5_alphabet_dict_rejected_without_consuming():
    # The Sequence-only rule rejects mappings outright even though a dict has
    # __len__: tuple(alphabet) would otherwise reduce a dict to its
    # (distinct, str) keys and silently accept it.
    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            _unreachable_residual,
            _unreachable_residual,
            alphabet={"a": 1, "b": 2},
            bound=UnitDataUnitAdvanceBound(continuation_length=1),
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )


def test_s5_alphabet_set_rejected_without_consuming():
    # The Sequence-only rule rejects sets outright even though a set has
    # __len__: tuple(alphabet) would otherwise be silently accepted.
    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            _unreachable_residual,
            _unreachable_residual,
            alphabet={"a", "b"},
            bound=UnitDataUnitAdvanceBound(continuation_length=1),
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )


def test_s5_alphabet_str_as_container_rejected():
    # Validation rule: a string is not an alphabet container.
    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            _unreachable_residual,
            _unreachable_residual,
            alphabet="ab",
            bound=UnitDataUnitAdvanceBound(continuation_length=1),
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )


def test_s5_alphabet_str_subclass_element_rejected():
    # Validation rule: labels are exact strings, not string subclasses.
    class LabelStr(str):
        pass

    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            _unreachable_residual,
            _unreachable_residual,
            alphabet=("a", LabelStr("b")),
            bound=UnitDataUnitAdvanceBound(continuation_length=1),
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )


# -- Exact closed observation grammar --------------------------------------


def test_s5_canonical_observation_digest_accepts_a_real_verdict_object_from_the_monitor():
    # verdict.py's VerdictObject must be admissible, even though it is
    # defined in a different module than residual.py -- the asymmetry the
    # spec names: comparisons pass a real VerdictObject through
    # _structurally_equal without complaint, so the digest must accept the
    # identical value rather than refuse it on module identity alone.
    monitor = _real_monitor()
    verdict = monitor.current_verdict()

    digest = canonical_observation_digest(verdict)

    assert isinstance(digest, str)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


# ==============================================================================
# == The admitted OBSERVATION domain is an explicit allowlist, stated against
# == the typed-events rule's TerminalKind grammar
# == and the totalised transition system and residual definitions's frozen-observation and
# == disabled-result rules. Three requirements the allowlist states, each
# == pinned below: the proof-search evidence/result types (ComparisonScope
# == and friends) are NOT semantic observations; only tuple/list/dict/frozenset
# == are admitted as containers, never an ABC-interface type (Mapping, Set, a
# == tuple subclass); and a TotalObservation whose Disabled status records one
# == of events.py's five terminal-kind dataclasses IS admissible -- the value
# == the monitor produces once a continuation arrives after a terminal
# == boundary.
# ==============================================================================


# -- TotalObservation with a terminal-kind attempted label, end to end --------


def _total_observation_with_terminal_kind_attempted():
    """the residual definition: a continuation label attempted
    after the totalised state is already terminal returns a TotalObservation
    whose Disabled status records that label in `attempted`. When the label is
    itself a further terminal event, the typed-events rule's TerminalKind grammar (`Complete | AgentAbort(reason) |
    ExternalCrash(cause) | Timeout(horizon_id) | ObservationCut(source)`)
    makes the recorded `attempted` value one of those five dataclasses -- here,
    `Complete`. The admitted-observation allowlist names the five terminal-kind
    dataclasses for exactly this reason.
    """
    monitor = _real_monitor()
    observation = monitor_residual(
        monitor, [TerminalEvent("e2", tick=1, kind=Complete())]
    )
    status = observation.continuation_status
    assert isinstance(status, residual_module.Disabled), (
        "precondition: the continuation must disable"
    )
    assert isinstance(status.attempted, Complete), (
        "precondition: the attempted label must be the terminal-kind dataclass itself"
    )
    return observation


def test_s5_canonical_observation_digest_accepts_a_total_observation_whose_disabled_attempted_is_a_terminal_kind_dataclass():
    observation = _total_observation_with_terminal_kind_attempted()

    digest = canonical_observation_digest(observation)

    assert isinstance(digest, str)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_s5_bounded_compare_accepts_identical_total_observations_with_terminal_kind_attempted_labels():
    # Identical sides over this observation must reach an exhausted search,
    # not a refusal.
    observation = _total_observation_with_terminal_kind_attempted()

    def residual(word):
        return observation

    result = bounded_compare(
        residual,
        residual,
        alphabet=(),
        bound=UnitDataUnitAdvanceBound(continuation_length=0),
        contract_environment="C",
        observation_projection="TotalObservation_C",
        continuation_family="all-words-through-declared-bound",
    )

    assert isinstance(result, NoWitnessWithinBound)


# -- Proof-search evidence and result types are not observations --------------


def _no_witness_within_bound_result() -> NoWitnessWithinBound:
    """A real `NoWitnessWithinBound` (S5.7's exhausted-search result) and, via
    its `scope` field, a real `ComparisonScope`: both are proof-search
    evidence about a comparison, not a semantic observation of contract
    state, and the admitted-observation allowlist excludes both by name.
    """

    def residual(word):
        return "same"

    result = bounded_compare(
        residual,
        residual,
        alphabet=(),
        bound=UnitDataUnitAdvanceBound(continuation_length=0),
        contract_environment="C",
        observation_projection="TotalObservation_C",
        continuation_family="all-words-through-declared-bound",
    )
    assert isinstance(result, NoWitnessWithinBound), (
        "precondition: identical sides do not distinguish"
    )
    assert isinstance(result.scope, ComparisonScope), (
        "precondition: every result carries its scope"
    )
    return result


def test_s5_canonical_observation_digest_refuses_a_comparison_scope_offered_as_an_observation():
    scope = _no_witness_within_bound_result().scope

    with pytest.raises(ValueError):
        canonical_observation_digest(scope)


def test_s5_bounded_compare_refuses_a_comparison_scope_offered_as_an_observation():
    scope = _no_witness_within_bound_result().scope

    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            lambda word: scope,
            lambda word: scope,
            alphabet=(),
            bound=UnitDataUnitAdvanceBound(continuation_length=0),
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )


def test_s5_canonical_observation_digest_refuses_a_no_witness_within_bound_offered_as_an_observation():
    result = _no_witness_within_bound_result()

    with pytest.raises(ValueError):
        canonical_observation_digest(result)


def test_s5_bounded_compare_refuses_a_no_witness_within_bound_offered_as_an_observation():
    result = _no_witness_within_bound_result()

    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            lambda word: result,
            lambda word: result,
            alphabet=(),
            bound=UnitDataUnitAdvanceBound(continuation_length=0),
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )


# -- An external Mapping is refused before any of its methods run -------------


class _ExternalMappingObservation(Mapping):
    """A `collections.abc.Mapping` subclass that is not `dict`, instrumented to
    record every method invocation. The admitted-observation allowlist: 'NO
    ABC interfaces: a Mapping that is not exactly dict is rejected.' An
    external Mapping can run arbitrary code and need not even be internally
    consistent, so the type check must reject it before any of its methods
    run.
    """

    def __init__(self, data):
        self._data = dict(data)
        self.invoked_methods: list[str] = []

    def __getitem__(self, key):
        self.invoked_methods.append("__getitem__")
        return self._data[key]

    def __iter__(self):
        self.invoked_methods.append("__iter__")
        return iter(self._data)

    def __len__(self):
        self.invoked_methods.append("__len__")
        return len(self._data)

    def items(self):
        self.invoked_methods.append("items")
        return super().items()


def test_s5_canonical_observation_digest_refuses_external_mapping_without_invoking_its_methods():
    external = _ExternalMappingObservation({"a": 1})

    with pytest.raises(ValueError):
        canonical_observation_digest(external)

    assert external.invoked_methods == []


def test_s5_bounded_compare_refuses_external_mapping_observation_without_invoking_its_methods():
    external = _ExternalMappingObservation({"a": 1})

    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            lambda word: external,
            lambda word: external,
            alphabet=(),
            bound=UnitDataUnitAdvanceBound(continuation_length=0),
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )

    assert external.invoked_methods == []


# -- A tuple subclass is refused, not silently treated as a sequence ----------


class _LabelTuple(tuple):
    """A `tuple` subclass. The admitted-observation allowlist's container
    rule is 'by exact type only ... a tuple subclass is rejected.'
    """


def test_s5_canonical_observation_digest_refuses_a_tuple_subclass_observation():
    with pytest.raises(ValueError):
        canonical_observation_digest(_LabelTuple(("a", "b")))


def test_s5_bounded_compare_refuses_a_tuple_subclass_observation():
    with pytest.raises(InvalidComparisonRequest):
        bounded_compare(
            lambda word: _LabelTuple(("a", "b")),
            lambda word: _LabelTuple(("a", "b")),
            alphabet=(),
            bound=UnitDataUnitAdvanceBound(continuation_length=0),
            contract_environment="C",
            observation_projection="TotalObservation_C",
            continuation_family="all-words-through-declared-bound",
        )


# -- Generic utility admissible values --------------------------------------
# Plain dict, plain string, and VerdictObject stay admitted by the generic
# utilities. Conformance-fragment comparisons still compare exact
# TotalObservation values.


def test_s5_canonical_observation_digest_still_accepts_a_plain_dict_observation():
    digest = canonical_observation_digest({"mode": "Complete", "summary": "Violated"})

    assert isinstance(digest, str)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_s5_canonical_observation_digest_still_accepts_a_plain_string_observation():
    digest = canonical_observation_digest("delete")

    assert isinstance(digest, str)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_s5_canonical_observation_digest_still_accepts_a_real_verdict_object_observation():
    monitor = _real_monitor()
    verdict = monitor.current_verdict()

    digest = canonical_observation_digest(verdict)

    assert isinstance(digest, str)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
