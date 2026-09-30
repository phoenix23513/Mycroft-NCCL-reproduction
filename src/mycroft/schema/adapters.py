"""Adapters from existing E01 and E02 records to the current Event schema."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Protocol

from .event import (
    Event,
    EventContext,
    EventKind,
    LogicalTime,
    ProgressSnapshotPayload,
    RingActionPayload,
    TimeDomain,
)
from .validate import validate_event
from .version import SCHEMA_VERSION


class ProgressSnapshotLike(Protocol):
    """Structural input accepted from E02 without importing experiment code."""

    tick: int
    gpu_ready: int
    rdma_transmitted: int
    rdma_done: int


def adapt_e01_record(
    record: Mapping[str, object],
    *,
    communicator_id: str,
    collective: str = "all_reduce",
    event_id: str | None = None,
    dependencies: Iterable[str] = (),
) -> Event:
    """Convert one current E01 JSONL object into a validated current-version Event."""
    op_seq = _required(record, "op_seq")
    rank = _required(record, "rank")
    channel = _required(record, "channel")
    phase = _required(record, "phase")
    step = _required(record, "step")
    chunk = _required(record, "chunk")
    action = _required(record, "action")
    peer = _required(record, "peer")
    timestamp = _required(record, "timestamp")
    resolved_event_id = event_id
    if resolved_event_id is None:
        resolved_event_id = (
            f"e01:{communicator_id}:{op_seq}:{rank}:{channel}:"
            f"{phase}:{step}:{chunk}:{action}"
        )

    event = Event(
        schema_version=SCHEMA_VERSION,
        event_id=resolved_event_id,
        event_kind=EventKind.RING_ACTION,
        source="e01_ring",
        context=EventContext(
            communicator_id=communicator_id,
            op_seq=op_seq,
            collective=collective,
            rank=rank,
            channel=channel,
        ),
        time=LogicalTime(
            value=timestamp,
            domain=TimeDomain.E01_EVENT_ORDER,
        ),
        dependencies=_dependency_tuple(dependencies),
        payload=RingActionPayload(
            phase=phase,
            step=step,
            chunk=chunk,
            action=action,
            peer=peer,
            baseline_timestamp=_required(record, "baseline_timestamp"),
            delay_role=_required(record, "delay_role"),
            value=_required(record, "value"),
            contributor_mask=_required(record, "contributor_mask"),
        ),
    )
    validate_event(event)
    return event


def adapt_e02_snapshot(
    snapshot: ProgressSnapshotLike,
    *,
    total_chunks: int,
    context: EventContext,
    event_id: str | None = None,
    dependencies: Iterable[str] = (),
) -> Event:
    """Convert one E02 snapshot plus explicit context to a current-version Event."""
    tick = _snapshot_field(snapshot, "tick")
    resolved_event_id = event_id
    if resolved_event_id is None:
        resolved_event_id = (
            f"e02:{context.communicator_id}:{context.op_seq}:"
            f"{context.rank}:{context.channel}:{tick}"
        )

    event = Event(
        schema_version=SCHEMA_VERSION,
        event_id=resolved_event_id,
        event_kind=EventKind.PROGRESS_SNAPSHOT,
        source="e02_progress",
        context=context,
        time=LogicalTime(value=tick, domain=TimeDomain.E02_TICK),
        dependencies=_dependency_tuple(dependencies),
        payload=ProgressSnapshotPayload(
            total_chunks=total_chunks,
            gpu_ready=_snapshot_field(snapshot, "gpu_ready"),
            rdma_transmitted=_snapshot_field(snapshot, "rdma_transmitted"),
            rdma_done=_snapshot_field(snapshot, "rdma_done"),
        ),
    )
    validate_event(event)
    return event


def adapt_e01_trace(
    records: Iterable[Mapping[str, object]],
    *,
    communicator_id: str,
    collective: str = "all_reduce",
) -> tuple[Event, ...]:
    """Convert an ordered E01 trace and preserve its model dependencies."""
    converted: list[Event] = []
    event_ids: set[str] = set()
    latest_by_rank: dict[int, str] = {}
    sends: dict[tuple[object, ...], str] = {}

    for record in records:
        event = adapt_e01_record(
            record,
            communicator_id=communicator_id,
            collective=collective,
        )
        if event.event_id in event_ids:
            raise ValueError(f"duplicate E01 event identity: {event.event_id}")

        payload = event.payload
        if not isinstance(payload, RingActionPayload):
            raise ValueError("E01 adapter produced a non-Ring payload")
        dependencies: list[str] = []
        if payload.action == "recv":
            message_predecessor = sends.get(_ring_message_key(event))
            if message_predecessor is not None:
                dependencies.append(message_predecessor)

        rank_predecessor = latest_by_rank.get(event.context.rank)
        if (
            rank_predecessor is not None
            and rank_predecessor not in dependencies
        ):
            dependencies.append(rank_predecessor)

        event = replace(event, dependencies=tuple(dependencies))
        validate_event(event)
        converted.append(event)
        event_ids.add(event.event_id)
        latest_by_rank[event.context.rank] = event.event_id
        if payload.action == "send":
            sends[_ring_message_key(event)] = event.event_id

    return tuple(converted)


def adapt_e02_trace(
    snapshots: Iterable[ProgressSnapshotLike],
    *,
    total_chunks: int,
    context: EventContext,
) -> tuple[Event, ...]:
    """Convert an ordered E02 trace and link each snapshot to its predecessor."""
    converted: list[Event] = []
    previous: Event | None = None

    for snapshot in snapshots:
        event = adapt_e02_snapshot(
            snapshot,
            total_chunks=total_chunks,
            context=context,
        )
        if previous is not None:
            if event.time.value <= previous.time.value:
                raise ValueError("E02 trace ticks must be strictly increasing")
            previous_payload = previous.payload
            current_payload = event.payload
            if not isinstance(
                previous_payload, ProgressSnapshotPayload
            ) or not isinstance(current_payload, ProgressSnapshotPayload):
                raise ValueError("E02 adapter produced a non-progress payload")
            if not (
                previous_payload.gpu_ready <= current_payload.gpu_ready
                and previous_payload.rdma_transmitted
                <= current_payload.rdma_transmitted
                and previous_payload.rdma_done <= current_payload.rdma_done
            ):
                raise ValueError("E02 trace progress counters must be monotonic")
            event = replace(event, dependencies=(previous.event_id,))
            validate_event(event)
        converted.append(event)
        previous = event

    return tuple(converted)


def _ring_message_key(event: Event) -> tuple[object, ...]:
    payload = event.payload
    if not isinstance(payload, RingActionPayload):
        raise ValueError("ring message key requires RingActionPayload")
    if payload.action == "send":
        source_rank = event.context.rank
        destination_rank = payload.peer
    else:
        source_rank = payload.peer
        destination_rank = event.context.rank
    return (
        event.context.communicator_id,
        event.context.op_seq,
        event.context.channel,
        payload.phase,
        payload.step,
        payload.chunk,
        source_rank,
        destination_rank,
    )


def _required(record: Mapping[str, object], name: str) -> object:
    try:
        return record[name]
    except KeyError as error:
        raise ValueError(f"E01 record missing required field: {name}") from error


def _snapshot_field(snapshot: ProgressSnapshotLike, name: str) -> object:
    try:
        return getattr(snapshot, name)
    except AttributeError as error:
        raise ValueError(f"E02 snapshot missing required field: {name}") from error


def _dependency_tuple(dependencies: Iterable[str]) -> tuple[str, ...]:
    if isinstance(dependencies, str):
        raise ValueError("dependencies must be an iterable of event IDs, not a string")
    return tuple(dependencies)
