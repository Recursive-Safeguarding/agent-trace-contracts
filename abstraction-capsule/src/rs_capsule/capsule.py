"""Run the declared source and transformed residuals through one comparison."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

from rs_metalang_ref.events import DomainEvent, TickEvent
from rs_metalang_ref.residual import (
    Distinguished,
    NoWitnessWithinBound,
    bounded_compare,
)

from .cards import CandidateCard
from .profile import Profile, SourcePrefix, project_summary_mode

SOURCE_TRACE = (
    DomainEvent(event_id="source-export-1", tick=0, tag="export"),
    DomainEvent(event_id="source-approval-1", tick=1, tag="approval"),
    DomainEvent(event_id="source-export-2", tick=2, tag="export"),
    TickEvent(event_id="source-tick-3", tick=3),
)


@dataclasses.dataclass(frozen=True)
class CapsuleResult:
    profile_id: str
    card_id: str
    comparison: Distinguished | NoWitnessWithinBound
    run_metadata: Mapping[str, object]
    summary: str


def build_source_prefix(profile: Profile, trace: Sequence[object]) -> SourcePrefix:
    return profile.contract_environment.build_prefix(trace)


def source_observation(profile: Profile, prefix: SourcePrefix, word: Sequence[str]):
    observation = profile.contract_environment.source_residual(prefix, word)
    return project_summary_mode(observation)


def card_observation(
    profile: Profile,
    card: CandidateCard,
    prefix: SourcePrefix,
    word: Sequence[str],
):
    interface = profile.continuation_interface
    term = card(prefix)
    environment = profile.retained_environment_manifest.extract(prefix)
    configuration = interface.initialise(term, environment)
    for label in word:
        configuration = interface.step(configuration, label)
    return project_summary_mode(interface.observe(configuration))


def _run_metadata(profile: Profile) -> Mapping[str, object]:
    terminal_labels = tuple(
        label
        for label in profile.continuation_alphabet
        if label
        in {"Complete", "AgentAbort", "ExternalCrash", "Timeout", "ObservationCut"}
    )
    return {
        "source_subjects": ("declared-source-prefix",),
        "trace_ontology_version": "rs-metalang-ref-events-v1",
        "contract_set_version": "single-export-approval-v1",
        "observation_projection": project_summary_mode.__name__,
        "continuation_family": "all-words-through-declared-bound",
        "search_domain": {
            "continuation_length_bound": profile.comparison_bound.continuation_length,
            "alphabet": profile.continuation_alphabet,
            "terminal_labels": terminal_labels,
            "canonical_enumeration_order": "length-then-alphabet-product",
        },
        "implementation_version": "abstraction-capsule-v1",
    }


def _summary(
    profile: Profile,
    card: CandidateCard,
    comparison: Distinguished | NoWitnessWithinBound,
) -> str:
    bound = profile.comparison_bound.continuation_length
    if isinstance(comparison, Distinguished):
        witness = ", ".join(comparison.witness)
        text = (
            f"{profile.profile_id}: {card.card_id} is separated at the exact "
            f"project_summary_mode (Summary, Mode) projection by continuation "
            f"[{witness}] within the "
            f"declared family of words through length {bound}."
        )
        return text
    if isinstance(comparison, NoWitnessWithinBound):
        text = (
            f"{profile.profile_id}: exhaustive enumeration found no distinguishing "
            f"continuation at the exact project_summary_mode (Summary, Mode) projection "
            f"for {card.card_id} "
            f"within the declared family of words through length {bound}."
        )
        return text
    raise RuntimeError(
        f"comparison returned an out-of-scope constructor: {comparison!r}"
    )


def _validate_summary_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def run_capsule(
    profile: Profile,
    card: CandidateCard,
    source_trace: Sequence[object],
) -> CapsuleResult:
    _validate_summary_identifier("profile_id", profile.profile_id)
    _validate_summary_identifier("card_id", card.card_id)
    prefix = build_source_prefix(profile, source_trace)
    run_metadata = _run_metadata(profile)
    comparison = bounded_compare(
        residual_u=lambda word: source_observation(profile, prefix, word),
        residual_v=lambda word: card_observation(profile, card, prefix, word),
        alphabet=profile.continuation_alphabet,
        bound=profile.comparison_bound,
        contract_environment=profile.contract_environment,
        observation_projection=run_metadata["observation_projection"],
        continuation_family=run_metadata["continuation_family"],
    )
    summary = _summary(profile, card, comparison)
    return CapsuleResult(
        profile_id=profile.profile_id,
        card_id=card.card_id,
        comparison=comparison,
        run_metadata=run_metadata,
        summary=summary,
    )
