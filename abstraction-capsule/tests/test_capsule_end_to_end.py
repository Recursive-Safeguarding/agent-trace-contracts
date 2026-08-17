"""End-to-end tests for one source trace and two abstraction cards.

The tests exercise the profile-relative operational-adequacy rule. A candidate
can be rejected by a witness or survive within the declared bound, but both
outcomes must be produced by running the continuation from the transformed
configuration.
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import shutil
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import pytest
from rs_metalang_ref.obligations import OccurrenceStatus
from rs_metalang_ref.residual import (
    Disabled,
    DisabledReason,
    Distinguished,
    ExhaustiveSearchReceipt,
    NoWitnessWithinBound,
    TotalObservation,
)
from rs_metalang_ref.verdict import Mode, Summary

from rs_capsule.capsule import (
    SOURCE_TRACE,
    build_source_prefix,
    card_observation,
    run_capsule,
    source_observation,
)
from rs_capsule.cards import open_obligation_card, truncation_card
from rs_capsule.profile import build_profile

CAPSULE_RESULT_FIELDS = frozenset(
    {
        "profile_id",
        "card_id",
        "comparison",
        "run_metadata",
        "summary",
    }
)

CAPSULE_RUN_METADATA_KEYS = frozenset(
    {
        "source_subjects",
        "trace_ontology_version",
        "contract_set_version",
        "observation_projection",
        "continuation_family",
        "search_domain",
        "implementation_version",
    }
)

# The capsule run metadata's search_domain sub-record names the declared
# continuation bound, alphabet, terminal labels, and enumeration order.
CAPSULE_RUN_SEARCH_DOMAIN_KEYS = frozenset(
    {
        "continuation_length_bound",
        "alphabet",
        "terminal_labels",
        "canonical_enumeration_order",
    }
)

COMPARISON_SCOPE_FIELDS = frozenset(
    {
        "contract_environment",
        "observation_projection",
        "continuation_family",
        "bound",
        "exclusions",
    }
)

EXPECTED_RUN_METADATA = {
    "source_subjects": ("declared-source-prefix",),
    "trace_ontology_version": "rs-metalang-ref-events-v1",
    "contract_set_version": "single-export-approval-v1",
    "observation_projection": "project_summary_mode",
    "continuation_family": "all-words-through-declared-bound",
    "search_domain": {
        "continuation_length_bound": 2,
        "alphabet": ("approval", "Complete"),
        "terminal_labels": ("Complete",),
        "canonical_enumeration_order": "length-then-alphabet-product",
    },
    "implementation_version": "abstraction-capsule-v1",
}

EXPECTED_SERIALIZED_RUN_METADATA = {
    "source_subjects": ["declared-source-prefix"],
    "trace_ontology_version": "rs-metalang-ref-events-v1",
    "contract_set_version": "single-export-approval-v1",
    "observation_projection": "project_summary_mode",
    "continuation_family": "all-words-through-declared-bound",
    "search_domain": {
        "continuation_length_bound": 2,
        "alphabet": ["approval", "Complete"],
        "terminal_labels": ["Complete"],
        "canonical_enumeration_order": "length-then-alphabet-product",
    },
    "implementation_version": "abstraction-capsule-v1",
}


@pytest.fixture
def profile():
    return build_profile()


@pytest.fixture
def prefix(profile):
    return build_source_prefix(profile, SOURCE_TRACE)


# -- the source trace ------------------------------------------------------


def test_the_source_trace_is_nontrivial(prefix):
    """The source trace fires, partially discharges, and leaves one obligation open."""
    assert len(SOURCE_TRACE) >= 3
    assert any(getattr(event, "tag", None) == "export" for event in SOURCE_TRACE)

    statuses = [occ.status for occ in prefix.monitor.occurrences.values()]
    assert OccurrenceStatus.DISCHARGED in statuses, "nothing was discharged"
    assert statuses.count(OccurrenceStatus.OPEN) == 1, (
        "expected exactly one live obligation"
    )


# -- the continuation interface -------------------------------------------


def test_step_is_total_over_every_continuation_label(profile, prefix):
    """the profile-relative operational-adequacy rule requires step_P to be
    totalised over disabled, terminal, and fault cases: a total transition
    for every continuation label, feeding a transformed observation
    function.
    """
    interface = profile.continuation_interface
    card = truncation_card(budget=1)
    term = card(prefix)
    environment = profile.retained_environment_manifest.extract(prefix)

    configuration = interface.initialise(term, environment)
    assert isinstance(configuration, interface.configuration_type)

    for label in profile.continuation_alphabet:
        assert isinstance(
            interface.step(configuration, label), interface.configuration_type
        )

    # Totalized past a terminal label: stepping again must not raise.
    terminal = next(
        label
        for label in profile.continuation_alphabet
        if label
        in {"Complete", "AgentAbort", "ExternalCrash", "Timeout", "ObservationCut"}
    )
    after_terminal = interface.step(configuration, terminal)
    for label in profile.continuation_alphabet:
        assert isinstance(
            interface.step(after_terminal, label), interface.configuration_type
        )


def test_step_records_and_freezes_the_first_label_after_terminal(profile, prefix):
    """A label after ``Complete`` enters the first-disabled sink.

    the totalised transition system and profile-relative operational-adequacy rules require ``step_P`` to be
    totalised after a terminal boundary and ``observe_P`` to expose the resulting
    ``TotalObservation_C``. Later labels must leave that first-disabled state
    unchanged.
    """
    interface = profile.continuation_interface
    term = open_obligation_card()(prefix)
    environment = profile.retained_environment_manifest.extract(prefix)
    configuration = interface.initialise(term, environment)

    after_terminal = interface.step(configuration, "Complete")
    after_first_disabled = interface.step(after_terminal, "approval")
    first_observation = interface.observe(after_first_disabled)

    assert isinstance(first_observation, TotalObservation)
    assert first_observation.continuation_status == Disabled(
        attempted=("approval", {}),
        enabled_set=frozenset(),
        reason=DisabledReason.AFTER_TERMINAL,
    )

    after_later_label = interface.step(after_first_disabled, "Complete")
    assert interface.observe(after_later_label) == first_observation


def test_observe_does_not_expose_the_disabled_state_to_caller_mutation(profile, prefix):
    """A returned observation cannot mutate the unchanged disabled state."""
    interface = profile.continuation_interface
    term = open_obligation_card()(prefix)
    environment = profile.retained_environment_manifest.extract(prefix)
    configuration = interface.initialise(term, environment)
    configuration = interface.step(configuration, "Complete")
    configuration = interface.step(configuration, "approval")
    expected_disabled = Disabled(
        attempted=("approval", {}),
        enabled_set=frozenset(),
        reason=DisabledReason.AFTER_TERMINAL,
    )

    caller_observation = interface.observe(configuration)
    assert caller_observation.continuation_status == expected_disabled
    caller_observation.continuation_status.attempted[1]["mutated-by-caller"] = True

    assert interface.observe(configuration).continuation_status == expected_disabled


def test_configuration_state_snapshot_cannot_change_a_disabled_observation(
    profile, prefix
):
    """A caller cannot change a disabled observation through public configuration state."""
    interface = profile.continuation_interface
    term = open_obligation_card()(prefix)
    environment = profile.retained_environment_manifest.extract(prefix)
    configuration = interface.initialise(term, environment)
    configuration = interface.step(configuration, "Complete")
    configuration = interface.step(configuration, "approval")
    expected_observation = copy.deepcopy(interface.observe(configuration))

    try:
        state_snapshot = configuration.state
    except (AttributeError, TypeError):
        pass
    else:
        try:
            state_snapshot.attempted[1]["mutated-by-caller"] = True
        except (AttributeError, TypeError):
            pass

        try:
            state_snapshot.frozen_contract_observation.occurrences["extra"] = "Injected"
        except (AttributeError, TypeError):
            pass

    assert interface.observe(configuration) == expected_observation


def test_observe_returns_a_total_observation(profile, prefix):
    interface = profile.continuation_interface
    term = truncation_card(budget=1)(prefix)
    environment = profile.retained_environment_manifest.extract(prefix)

    configuration = interface.initialise(term, environment)

    assert isinstance(interface.observe(configuration), TotalObservation)


@pytest.mark.parametrize(
    "card_factory", [lambda: truncation_card(budget=1), open_obligation_card]
)
def test_transformed_residual_is_iterated_step_then_observe_then_project(
    profile, prefix, card_factory
):
    """The definitional test: `card_observation` is
    O_P(observe_P(step_P*(initialise_P(h(u), E_P(u)), w))), word by word.

    This is what forbids the capsule from quietly computing the card side
    from the source monitor instead of from the transformed state.
    """
    interface = profile.continuation_interface
    card = card_factory()
    term = card(prefix)
    environment = profile.retained_environment_manifest.extract(prefix)

    words = (
        word
        for length in range(profile.comparison_bound.continuation_length + 1)
        for word in itertools.product(profile.continuation_alphabet, repeat=length)
    )
    for word in words:
        configuration = interface.initialise(term, environment)
        for label in word:
            configuration = interface.step(configuration, label)
        observed = interface.observe(configuration).contract_observation
        expected = (observed.summary, observed.mode)

        assert card_observation(profile, card, prefix, word) == expected, word


def test_default_profile_projects_observations_to_summary_and_mode(profile, prefix):
    card = truncation_card(budget=1)
    source_total_observation = profile.contract_environment.source_residual(prefix, ())

    interface = profile.continuation_interface
    term = card(prefix)
    environment = profile.retained_environment_manifest.extract(prefix)
    configuration = interface.initialise(term, environment)
    card_total_observation = interface.observe(configuration)

    source_projected = source_observation(profile, prefix, ())
    card_projected = card_observation(profile, card, prefix, ())
    source_expected_projection = (
        source_total_observation.contract_observation.summary,
        source_total_observation.contract_observation.mode,
    )
    card_expected_projection = (
        card_total_observation.contract_observation.summary,
        card_total_observation.contract_observation.mode,
    )

    assert isinstance(source_total_observation, TotalObservation)
    assert isinstance(card_total_observation, TotalObservation)
    assert source_projected == source_expected_projection
    assert card_projected == card_expected_projection
    for projected in (source_projected, card_projected):
        assert isinstance(projected, tuple)
        assert len(projected) == 2
        assert isinstance(projected[0], Summary)
        assert isinstance(projected[1], Mode)


def test_declared_projection_agrees_on_the_empty_continuation(profile, prefix):
    """A theorem relating source and transformed residuals needs "initial
    observations agree" as its first hypothesis. If a candidate already
    disagrees at the empty word, the capsule demonstrates nothing about
    continuation semantics: it compares two states and stops before the
    transformed step.

    So the declared profile must make the empty continuation agree for both
    candidates, and every witness below is therefore a real continuation.
    """
    for card in (truncation_card(budget=1), open_obligation_card()):
        assert source_observation(profile, prefix, ()) == card_observation(
            profile, card, prefix, ()
        ), f"{card.card_id} disagrees before any continuation step"


# -- candidate (a): rejected by a witness ----------------------------------


def test_truncation_card_is_rejected_by_a_witness(profile):
    """The candidate can be rejected by a non-empty witness.

    The comparison is the interpreter's `Distinguished` constructor, reached
    through `bounded_compare`.
    """
    result = run_capsule(profile, truncation_card(budget=1), SOURCE_TRACE)

    assert isinstance(result.comparison, Distinguished)

    witness = result.comparison.witness
    assert isinstance(witness, tuple)
    assert len(witness) >= 1, "the empty word is not a distinguishing continuation"
    assert len(witness) <= profile.comparison_bound.continuation_length
    assert set(witness) <= set(profile.continuation_alphabet)
    assert result.comparison.left_observation != result.comparison.right_observation


def test_the_witness_is_replayable_through_the_profile(profile, prefix):
    """Replaying the witness reproduces the disagreement: the same two
    observations. The end-to-end case must carry one trace, one
    transformation, one continuation, and its replayable result.

    Replayed twice, because a candidate or a residual that mutated the
    source prefix would pass once and diverge afterwards.
    """
    card = truncation_card(budget=1)
    result = run_capsule(profile, card, SOURCE_TRACE)
    witness = result.comparison.witness

    for _ in range(2):
        assert (
            source_observation(profile, prefix, witness)
            == result.comparison.left_observation
        )
        assert (
            card_observation(profile, card, prefix, witness)
            == result.comparison.right_observation
        )


# -- candidate (b): no witness within the declared bound -------------------


def test_open_obligation_card_finds_no_witness_within_the_bound(profile):
    """The interpreter returns `NoWitnessWithinBound` after exhaustive enumeration.

    the result-algebra constructor rule requires the result to include the
    declared comparison scope.
    """
    result = run_capsule(profile, open_obligation_card(), SOURCE_TRACE)

    assert isinstance(result.comparison, NoWitnessWithinBound)
    assert result.comparison.scope.bound is profile.comparison_bound
    assert result.comparison.exhaustive_search_receipt == ExhaustiveSearchReceipt(
        words_enumerated=7,
        max_length=2,
        alphabet=("approval", "Complete"),
    )


def test_run_capsule_uses_the_profiles_single_comparison_bound(profile):
    from rs_metalang_ref import residual as residual_module

    comparison_bound_type = getattr(residual_module, "UnitDataUnitAdvanceBound", None)
    assert comparison_bound_type is not None
    shorter_profile = replace(
        profile,
        comparison_bound=comparison_bound_type(continuation_length=1),
    )

    result = run_capsule(
        shorter_profile,
        open_obligation_card(),
        SOURCE_TRACE,
    )

    assert isinstance(result.comparison, NoWitnessWithinBound)
    assert result.comparison.scope.bound is shorter_profile.comparison_bound
    assert result.comparison.exhaustive_search_receipt == ExhaustiveSearchReceipt(
        words_enumerated=3,
        max_length=1,
        alphabet=("approval", "Complete"),
    )


# -- the result record ------------------------------------------------------


def test_every_result_has_exact_public_fields_and_constructor_status(profile):
    rejected = run_capsule(profile, truncation_card(budget=1), SOURCE_TRACE)
    survived = run_capsule(profile, open_obligation_card(), SOURCE_TRACE)

    for result, comparison_type in (
        (rejected, Distinguished),
        (survived, NoWitnessWithinBound),
    ):
        assert {field.name for field in fields(result)} == CAPSULE_RESULT_FIELDS
        assert not hasattr(result, "scope")
        assert not hasattr(result, "verdict_label")
        assert isinstance(result.comparison, comparison_type)

    assert rejected.profile_id == survived.profile_id == profile.profile_id
    assert rejected.run_metadata == survived.run_metadata
    assert rejected.card_id != survived.card_id


@pytest.mark.parametrize(
    "card_factory", [lambda: truncation_card(budget=1), open_obligation_card]
)
def test_every_result_carries_run_metadata_and_comparison_scope(profile, card_factory):
    """Every result carries capsule run metadata and a normative ComparisonScope.

    The run metadata's search_domain specifies the continuation length bound,
    alphabet, terminal labels, and canonical enumeration order. The comparison
    result carries the normative five-field scope.
    """
    result = run_capsule(profile, card_factory(), SOURCE_TRACE)

    assert set(result.run_metadata) == CAPSULE_RUN_METADATA_KEYS
    assert result.run_metadata == EXPECTED_RUN_METADATA
    assert set(result.run_metadata["search_domain"]) == CAPSULE_RUN_SEARCH_DOMAIN_KEYS
    assert {
        field.name for field in fields(result.comparison.scope)
    } == COMPARISON_SCOPE_FIELDS
    assert result.comparison.scope.observation_projection == "project_summary_mode"
    assert (
        "at the exact project_summary_mode (Summary, Mode) projection"
        in result.summary
    )
    assert (
        result.comparison.scope.continuation_family
        == "all-words-through-declared-bound"
    )


def test_public_cli_refuses_arguments_without_echoing_them():
    sentinel = "SENSITIVE_DISCLOSURE_SENTINEL"
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-m", "rs_capsule", sentinel],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert completed.stderr == "error: rs_capsule does not accept arguments\n"
    assert sentinel not in completed.stdout
    assert sentinel not in completed.stderr


def test_public_cli_is_byte_identical_across_two_fresh_project_environments(
    tmp_path,
):
    uv = shutil.which("uv")
    assert uv, "the uv executable must be on PATH to run the public command"

    capsule_root = Path(__file__).resolve().parents[1]
    outputs = []
    for run_number in range(2):
        run_root = tmp_path / f"run-{run_number}"
        project_environment = run_root / "environment"
        working_directory = run_root / "working-directory"
        home_directory = run_root / "home"
        temporary_directory = run_root / "tmp"
        for directory in (working_directory, home_directory, temporary_directory):
            directory.mkdir(parents=True)

        assert not project_environment.exists()
        command_environment = {
            **os.environ,
            "HOME": str(home_directory),
            "TMPDIR": str(temporary_directory),
            "UV_CACHE_DIR": str(run_root / "uv-cache"),
            "UV_PROJECT_ENVIRONMENT": str(project_environment),
            "UV_PYTHON_DOWNLOADS": "never",
            "XDG_CACHE_HOME": str(run_root / "xdg-cache"),
        }
        for variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
            command_environment.pop(variable, None)

        completed = subprocess.run(
            [
                uv,
                "run",
                "--quiet",
                "--frozen",
                "--python",
                sys.executable,
                "--project",
                str(capsule_root),
                "--directory",
                str(working_directory),
                "python",
                "-I",
                "-B",
                "-m",
                "rs_capsule",
            ],
            check=False,
            capture_output=True,
            env=command_environment,
        )

        assert completed.returncode == 0, completed.stderr.decode(
            "utf-8", errors="replace"
        )
        assert completed.stderr == b""
        assert completed.stdout
        assert project_environment.is_dir()
        outputs.append(completed.stdout)

    assert outputs[0] == outputs[1]


def test_public_json_keeps_the_exact_two_capsule_results():
    completed = subprocess.run(
        [sys.executable, "-m", "rs_capsule"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""

    scope = {
        "contract_environment": {
            "clause_spec": {
                "clause_id": "export-needs-approval",
                "trigger_tag": "export",
                "response_tag": "approval",
                "binding_fields": [],
                "guard": {"proposition_key": "approval-required"},
                "bound": 2,
                "discharge": {"Linear": {}},
                "on_agent_abort": {"Breach": {}},
            }
        },
        "observation_projection": "project_summary_mode",
        "continuation_family": "all-words-through-declared-bound",
        "bound": {
            "continuation_length": 2,
            "data_domain": {
                "kind": "unit",
                "note": "contract declares no binding fields",
            },
            "timing_bound": 2,
        },
        "exclusions": [],
    }
    expected = [
        {
            "profile_id": "profile-p-export-approval-k2-v1",
            "card_id": "tail-window-1-v1",
            "comparison": {
                "Distinguished": {
                    "scope": scope,
                    "witness": ["Complete"],
                    "left_observation": ["Violated", "Complete"],
                    "right_observation": ["Satisfied", "Complete"],
                }
            },
            "run_metadata": EXPECTED_SERIALIZED_RUN_METADATA,
            "summary": (
                "profile-p-export-approval-k2-v1: tail-window-1-v1 is separated "
                "at the exact project_summary_mode (Summary, Mode) projection "
                "by continuation [Complete] within the declared family of words "
                "through length 2."
            ),
        },
        {
            "profile_id": "profile-p-export-approval-k2-v1",
            "card_id": "obligation-table-v1",
            "comparison": {
                "NoWitnessWithinBound": {
                    "scope": scope,
                    "exhaustive_search_receipt": {
                        "words_enumerated": 7,
                        "max_length": 2,
                        "alphabet": ["approval", "Complete"],
                    },
                }
            },
            "run_metadata": EXPECTED_SERIALIZED_RUN_METADATA,
            "summary": (
                "profile-p-export-approval-k2-v1: exhaustive enumeration found "
                "no distinguishing continuation at the exact project_summary_mode "
                "(Summary, Mode) projection for obligation-table-v1 within the "
                "declared family of words through length 2."
            ),
        },
    ]

    assert json.loads(completed.stdout) == expected


def test_the_capsule_is_deterministic(profile):
    """Rerunning the capsule yields the same comparison. A capsule whose
    verdict moved between runs could not be a conformance condition.
    """
    first = run_capsule(profile, truncation_card(budget=1), SOURCE_TRACE)
    second = run_capsule(profile, truncation_card(budget=1), SOURCE_TRACE)

    assert first.comparison == second.comparison
    assert first.run_metadata == second.run_metadata
    assert first.summary == second.summary


def test_readme_names_the_profile_identifier(profile):
    """The README names the profile used by the public command."""
    readme = Path(__file__).resolve().parents[1] / "README.md"
    assert readme.is_file()
    assert profile.profile_id in readme.read_text(encoding="utf-8")


def test_readme_attributes_tail_window_sweep_to_the_test_suite():
    """The two-record CLI must not claim the separate tail-window tests."""
    test_file = Path(__file__).resolve()
    readmes = (
        test_file.parents[1] / "README.md",
        test_file.parents[2] / "README.md",
    )
    expected = (
        "The test suite also checks that two- and three-entry tail windows "
        "return `NoWitnessWithinBound` at the same bounded scope."
    )

    for readme in readmes:
        prose = " ".join(readme.read_text(encoding="utf-8").split())
        assert expected in prose, readme
