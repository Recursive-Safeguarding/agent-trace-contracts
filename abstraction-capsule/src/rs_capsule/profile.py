"""The fixed profile and state-card syntax used by the capsule."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Mapping, Sequence

from rs_metalang_ref.contracts import AfterClauseSpec, GroundGuard, Linear
from rs_metalang_ref.events import Complete, DomainEvent, TerminalEvent
from rs_metalang_ref.kleene import K3
from rs_metalang_ref.monitor import SingleClauseMonitor
from rs_metalang_ref.residual import (
    Disabled,
    DisabledSink,
    TotalObservation,
    UnitDataUnitAdvanceBound,
    monitor_residual,
)
from rs_metalang_ref.verdict import Mode, Summary, VerdictObject


@dataclasses.dataclass(frozen=True)
class SourcePrefix:
    trace: tuple[object, ...]
    monitor: SingleClauseMonitor


@dataclasses.dataclass(frozen=True)
class OccurrenceRecord:
    oid: str
    trigger_tick: int
    deadline: int


@dataclasses.dataclass(frozen=True)
class StateCardTerm:
    retained_events: tuple[object, ...]
    occurrence_table: tuple[OccurrenceRecord, ...]


@dataclasses.dataclass(frozen=True)
class CapsuleConfiguration:
    _state: SingleClauseMonitor | DisabledSink

    def __post_init__(self) -> None:
        object.__setattr__(self, "_state", copy.deepcopy(self._state))

    def _snapshot(self) -> SingleClauseMonitor | DisabledSink:
        return copy.deepcopy(self._state)


@dataclasses.dataclass(frozen=True)
class ContinuationInterface:
    configuration_type: type
    initialise: Callable[[StateCardTerm, Mapping[str, object]], CapsuleConfiguration]
    step: Callable[[CapsuleConfiguration, str], CapsuleConfiguration]
    observe: Callable[[CapsuleConfiguration], TotalObservation]


@dataclasses.dataclass(frozen=True)
class RetainedEnvironmentManifest:
    fields: tuple[str, ...]

    def extract(self, prefix: SourcePrefix) -> Mapping[str, object]:
        values = {
            "current_tick": prefix.monitor.tick,
            "guard_value": prefix.monitor.guard_value_for({}),
        }
        return {field: values[field] for field in self.fields}


@dataclasses.dataclass(frozen=True)
class ContractEnvironment:
    clause_spec: AfterClauseSpec

    def build_prefix(self, trace: Sequence[object]) -> SourcePrefix:
        monitor = SingleClauseMonitor(self.clause_spec)
        monitor.set_initial(monitor.guard_key_for({}), K3.T)
        for event in trace:
            monitor.step(event)
        return SourcePrefix(trace=tuple(trace), monitor=monitor)

    def source_residual(
        self, prefix: SourcePrefix, word: Sequence[str]
    ) -> TotalObservation:
        events = tuple(_event_for_label(label, 1) for label in word)
        return monitor_residual(prefix.monitor, events)


def project_summary_mode(observation: TotalObservation) -> tuple[Summary, Mode]:
    contract_observation = observation.contract_observation
    if not isinstance(contract_observation, VerdictObject):
        raise TypeError(
            "the fixed profile requires a VerdictObject contract observation"
        )
    return contract_observation.summary, contract_observation.mode


@dataclasses.dataclass(frozen=True)
class Profile:
    profile_id: str
    contract_environment: ContractEnvironment
    continuation_alphabet: tuple[str, ...]
    retained_environment_manifest: RetainedEnvironmentManifest
    continuation_interface: ContinuationInterface
    comparison_bound: UnitDataUnitAdvanceBound


def _event_for_label(label: str, tick: int):
    event_id = f"continuation-{tick}-{label.lower()}"
    if label == "approval":
        return DomainEvent(event_id=event_id, tick=tick, tag="approval")
    if label == "Complete":
        return TerminalEvent(event_id=event_id, tick=tick, kind=Complete())
    raise ValueError(f"label is outside the declared continuation alphabet: {label!r}")


def _build_interface(spec: AfterClauseSpec) -> ContinuationInterface:
    def initialise(
        term: StateCardTerm,
        environment: Mapping[str, object],
    ) -> CapsuleConfiguration:
        return CapsuleConfiguration(
            _state=SingleClauseMonitor.restore_open_state_card(
                spec,
                initial_guard_value=environment["guard_value"],
                retained_events=term.retained_events,
                current_tick=environment["current_tick"],
                open_occurrences=tuple(
                    (record.oid, record.trigger_tick, record.deadline)
                    for record in term.occurrence_table
                ),
            )
        )

    def step(configuration: CapsuleConfiguration, label: str) -> CapsuleConfiguration:
        state = configuration._snapshot()
        if isinstance(state, DisabledSink):
            _event_for_label(label, 1)
            return configuration

        event = _event_for_label(label, state.tick + 1)
        if state.current_verdict().mode is not Mode.RUNNING:
            observation = monitor_residual(state, (event,))
            disabled = observation.continuation_status
            if not isinstance(disabled, Disabled):
                raise RuntimeError(
                    "a post-terminal or post-fault step must produce a disabled observation"
                )
            return CapsuleConfiguration(
                _state=DisabledSink(
                    frozen_contract_observation=copy.deepcopy(
                        observation.contract_observation
                    ),
                    attempted=copy.deepcopy(disabled.attempted),
                    enabled_set=copy.deepcopy(disabled.enabled_set),
                    reason=disabled.reason,
                )
            )

        monitor = copy.deepcopy(state)
        monitor.step(event)
        return CapsuleConfiguration(_state=monitor)

    def observe(configuration: CapsuleConfiguration) -> TotalObservation:
        state = configuration._snapshot()
        if isinstance(state, DisabledSink):
            return TotalObservation(
                contract_observation=copy.deepcopy(state.frozen_contract_observation),
                continuation_status=Disabled(
                    attempted=copy.deepcopy(state.attempted),
                    enabled_set=copy.deepcopy(state.enabled_set),
                    reason=state.reason,
                ),
            )
        return monitor_residual(state, ())

    return ContinuationInterface(
        configuration_type=CapsuleConfiguration,
        initialise=initialise,
        step=step,
        observe=observe,
    )


def build_profile() -> Profile:
    clause_spec = AfterClauseSpec(
        clause_id="export-needs-approval",
        trigger_tag="export",
        response_tag="approval",
        binding_fields=(),
        guard=GroundGuard("approval-required"),
        bound=2,
        discharge=Linear(),
    )
    contract_environment = ContractEnvironment(clause_spec)
    alphabet = ("approval", "Complete")
    length_bound = 2

    return Profile(
        profile_id="profile-p-export-approval-k2-v1",
        contract_environment=contract_environment,
        continuation_alphabet=alphabet,
        retained_environment_manifest=RetainedEnvironmentManifest(
            fields=("current_tick", "guard_value")
        ),
        continuation_interface=_build_interface(clause_spec),
        comparison_bound=UnitDataUnitAdvanceBound(
            continuation_length=length_bound,
        ),
    )
