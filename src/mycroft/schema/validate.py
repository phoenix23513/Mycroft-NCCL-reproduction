"""Semantic validation for Event v1 records."""

from __future__ import annotations

from .event import (
    Event,
    EventContext,
    EventKind,
    LogicalTime,
    ProgressSnapshotPayload,
    RingActionPayload,
    TimeDomain,
)
from .version import SUPPORTED_SCHEMA_VERSIONS


class EventValidationError(ValueError):
    """Raised when an Event does not satisfy the selected schema version."""


def validate_event(event: Event) -> None:
    """Validate the common envelope and the selected typed payload."""
    if not isinstance(event, Event):
        raise EventValidationError("event must be an Event instance")
    _require_int("schema_version", event.schema_version, minimum=1)
    if event.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise EventValidationError(
            f"unsupported schema_version: {event.schema_version}"
        )
    _require_nonempty_string("event_id", event.event_id)
    if not isinstance(event.event_kind, EventKind):
        raise EventValidationError("event_kind must be an EventKind")
    _require_nonempty_string("source", event.source)

    _validate_context(event.context)
    _validate_time(event.time)
    _validate_dependencies(event)

    if event.event_kind is EventKind.RING_ACTION:
        if not isinstance(event.payload, RingActionPayload):
            raise EventValidationError(
                "ring_action event requires RingActionPayload"
            )
        _validate_ring_action(event)
    elif event.event_kind is EventKind.PROGRESS_SNAPSHOT:
        if not isinstance(event.payload, ProgressSnapshotPayload):
            raise EventValidationError(
                "progress_snapshot event requires ProgressSnapshotPayload"
            )
        _validate_progress_snapshot(event)


def _validate_context(context: EventContext) -> None:
    if not isinstance(context, EventContext):
        raise EventValidationError("context must be an EventContext")
    _require_nonempty_string("context.communicator_id", context.communicator_id)
    _require_int("context.op_seq", context.op_seq, minimum=0)
    _require_nonempty_string("context.collective", context.collective)
    _require_int("context.rank", context.rank, minimum=0)
    _require_int("context.channel", context.channel, minimum=0)


def _validate_time(time: LogicalTime) -> None:
    if not isinstance(time, LogicalTime):
        raise EventValidationError("time must be a LogicalTime")
    _require_int("time.value", time.value, minimum=0)
    if not isinstance(time.domain, TimeDomain):
        raise EventValidationError("time.domain must be a TimeDomain")


def _validate_dependencies(event: Event) -> None:
    if not isinstance(event.dependencies, tuple):
        raise EventValidationError("dependencies must be a tuple")
    seen: set[str] = set()
    for dependency in event.dependencies:
        _require_nonempty_string("dependency id", dependency)
        if dependency == event.event_id:
            raise EventValidationError("event cannot depend on itself")
        if dependency in seen:
            raise EventValidationError(f"duplicate dependency id: {dependency}")
        seen.add(dependency)


def _validate_ring_action(event: Event) -> None:
    payload = event.payload
    if not isinstance(payload, RingActionPayload):
        raise EventValidationError("invalid ring_action payload")
    if event.time.domain is not TimeDomain.E01_EVENT_ORDER:
        raise EventValidationError(
            "ring_action requires time domain e01_event_order"
        )
    if payload.phase not in {"reduce_scatter", "all_gather"}:
        raise EventValidationError(f"invalid ring phase: {payload.phase!r}")
    _require_int("payload.step", payload.step, minimum=0)
    _require_int("payload.chunk", payload.chunk, minimum=0)
    if payload.action not in {"send", "recv"}:
        raise EventValidationError(f"invalid ring action: {payload.action!r}")
    _require_int("payload.peer", payload.peer, minimum=0)
    if payload.peer == event.context.rank:
        raise EventValidationError("ring peer must differ from event rank")
    _require_int(
        "payload.baseline_timestamp", payload.baseline_timestamp, minimum=0
    )
    if payload.delay_role not in {"none", "injected_root", "affected"}:
        raise EventValidationError(
            f"invalid delay_role: {payload.delay_role!r}"
        )
    if event.time.value < payload.baseline_timestamp:
        raise EventValidationError(
            "ring logical time cannot precede baseline_timestamp"
        )
    if (
        payload.delay_role == "none"
        and event.time.value != payload.baseline_timestamp
    ):
        raise EventValidationError(
            "delay_role none requires logical time equal to baseline_timestamp"
        )
    if (
        payload.delay_role != "none"
        and event.time.value == payload.baseline_timestamp
    ):
        raise EventValidationError(
            "delayed ring event requires logical time after baseline_timestamp"
        )
    _require_int("payload.value", payload.value)
    _require_int("payload.contributor_mask", payload.contributor_mask, minimum=0)


def _validate_progress_snapshot(event: Event) -> None:
    payload = event.payload
    if not isinstance(payload, ProgressSnapshotPayload):
        raise EventValidationError("invalid progress_snapshot payload")
    if event.time.domain is not TimeDomain.E02_TICK:
        raise EventValidationError(
            "progress_snapshot requires time domain e02_tick"
        )
    _require_int("payload.total_chunks", payload.total_chunks, minimum=1)
    for name, value in (
        ("payload.gpu_ready", payload.gpu_ready),
        ("payload.rdma_transmitted", payload.rdma_transmitted),
        ("payload.rdma_done", payload.rdma_done),
    ):
        _require_int(name, value, minimum=0)
    if not (
        payload.rdma_done
        <= payload.rdma_transmitted
        <= payload.gpu_ready
        <= payload.total_chunks
    ):
        raise EventValidationError(
            "invalid progress state: expected rdma_done <= rdma_transmitted "
            "<= gpu_ready <= total_chunks"
        )


def _require_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EventValidationError(f"{name} must be a non-empty string")


def _require_int(name: str, value: object, minimum: int | None = None) -> None:
    if type(value) is not int:
        raise EventValidationError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise EventValidationError(f"{name} must be >= {minimum}")
