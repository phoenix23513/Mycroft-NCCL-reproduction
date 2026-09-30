"""Flow timeline recovery contracts for versioned event streams."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from mycroft.schema import Event, EventKind, TimeDomain, validate_event

from .group import OperationKey, group_events, operation_key_for


@dataclass(frozen=True, order=True)
class FlowKey:
    """One comparable clock stream inside an operation."""

    operation: OperationKey
    rank: int
    channel: int
    event_kind: EventKind
    time_domain: TimeDomain


@dataclass(frozen=True, order=True)
class MissingDependency:
    """One dependency ID referenced by an event but absent from the input."""

    event_id: str
    dependency_id: str


@dataclass(frozen=True)
class Timeline:
    """One recovered flow ordered inside a single logical clock domain."""

    key: FlowKey
    events: tuple[Event, ...]


@dataclass(frozen=True)
class RecoveryResult:
    """Recovered flows and non-fatal data-quality findings."""

    timelines: tuple[Timeline, ...]
    duplicate_event_ids: tuple[str, ...]
    missing_dependencies: tuple[MissingDependency, ...]


def flow_key_for(event: Event) -> FlowKey:
    """Return the smallest key whose events share comparable logical time.

    Channel -1 is an internal key for operation-wide completion events; it is
    never serialized as a real NCCL channel.
    """
    validate_event(event)
    return FlowKey(
        operation=operation_key_for(event),
        rank=event.context.rank,
        channel=(
            event.context.channel
            if event.context.channel is not None
            else -1
        ),
        event_kind=event.event_kind,
        time_domain=event.time.domain,
    )


def recover_timelines(events: Iterable[Event]) -> RecoveryResult:
    """Deduplicate, group, order, and report missing dependencies."""
    grouping = group_events(events)
    unique_events = tuple(
        event for group in grouping.groups for event in group.events
    )
    known_event_ids = {event.event_id for event in unique_events}

    events_by_flow: dict[FlowKey, list[Event]] = {}
    missing_dependencies: list[MissingDependency] = []

    for event in unique_events:
        events_by_flow.setdefault(flow_key_for(event), []).append(event)
        for dependency_id in event.dependencies:
            if dependency_id not in known_event_ids:
                missing_dependencies.append(
                    MissingDependency(event.event_id, dependency_id)
                )

    timelines = tuple(
        Timeline(
            key=key,
            events=tuple(
                sorted(
                    events_by_flow[key],
                    key=lambda event: (event.time.value, event.event_id),
                )
            ),
        )
        for key in sorted(events_by_flow)
    )

    return RecoveryResult(
        timelines=timelines,
        duplicate_event_ids=grouping.duplicate_event_ids,
        missing_dependencies=tuple(sorted(missing_dependencies)),
    )
