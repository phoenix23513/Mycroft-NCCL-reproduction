"""Operation-level grouping contracts for versioned event streams."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from mycroft.schema import Event, validate_event


@dataclass(frozen=True, order=True)
class OperationKey:
    """Identity of one collective across ranks, channels, and event kinds."""

    communicator_id: str
    op_seq: int


@dataclass(frozen=True)
class OperationGroup:
    """One operation and its deduplicated events in deterministic order."""

    key: OperationKey
    events: tuple[Event, ...]


@dataclass(frozen=True)
class GroupingResult:
    """Deterministic groups plus IDs that appeared identically more than once."""

    groups: tuple[OperationGroup, ...]
    duplicate_event_ids: tuple[str, ...]


class EventConflictError(ValueError):
    """Raised when one event_id is associated with different event content."""


class OperationMetadataConflictError(ValueError):
    """Raised when one operation identity has inconsistent metadata."""


def operation_key_for(event: Event) -> OperationKey:
    """Return the operation identity shared across ranks and channels."""
    validate_event(event)
    return OperationKey(
        communicator_id=event.context.communicator_id,
        op_seq=event.context.op_seq,
    )


def group_events(events: Iterable[Event]) -> GroupingResult:
    """Group and deduplicate events independently of arrival order."""
    events_by_id: dict[str, Event] = {}
    events_by_operation: dict[OperationKey, list[Event]] = {}
    collective_by_operation: dict[OperationKey, str] = {}
    duplicate_event_ids: set[str] = set()

    for event in events:
        validate_event(event)

        existing = events_by_id.get(event.event_id)
        if existing is not None:
            if existing == event:
                duplicate_event_ids.add(event.event_id)
                continue
            raise EventConflictError(
                f"event_id {event.event_id!r} has conflicting content"
            )
        events_by_id[event.event_id] = event

        operation_key = operation_key_for(event)
        known_collective = collective_by_operation.get(operation_key)
        if known_collective is None:
            collective_by_operation[operation_key] = event.context.collective
        elif known_collective != event.context.collective:
            raise OperationMetadataConflictError(
                f"operation {operation_key!r} has conflicting collective "
                f"values: {known_collective!r} and "
                f"{event.context.collective!r}"
            )

        events_by_operation.setdefault(operation_key, []).append(event)

    groups = tuple(
        OperationGroup(
            key=operation_key,
            events=tuple(
                sorted(
                    events_by_operation[operation_key],
                    key=lambda event: event.event_id,
                )
            ),
        )
        for operation_key in sorted(events_by_operation)
    )
    return GroupingResult(
        groups=groups,
        duplicate_event_ids=tuple(sorted(duplicate_event_ids)),
    )
