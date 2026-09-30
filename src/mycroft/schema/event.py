"""Typed, versioned events shared by simulation and NCCL adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    """The typed payload variant carried by a versioned event."""

    RING_ACTION = "ring_action"
    PROGRESS_SNAPSHOT = "progress_snapshot"
    OPERATION_COMPLETION = "operation_completion"


class TimeDomain(str, Enum):
    """Clock semantics; values are comparable only inside one recovered flow.

    NCCL monotonic values are process-local and not synchronized across hosts.
    """

    E01_EVENT_ORDER = "e01_event_order"
    E02_TICK = "e02_tick"
    NCCL_MONOTONIC_NS = "nccl_monotonic_ns"


@dataclass(frozen=True)
class EventContext:
    """Identity shared by events belonging to one collective operation.

    Channel is absent only for operation-wide Event v2 records.
    """

    communicator_id: str
    op_seq: int
    collective: str
    rank: int
    channel: int | None


@dataclass(frozen=True)
class LogicalTime:
    """A non-negative value interpreted only inside its named clock domain."""

    value: int
    domain: TimeDomain


@dataclass(frozen=True)
class RingActionPayload:
    """E01-specific Ring send/receive evidence."""

    phase: str
    step: int
    chunk: int
    action: str
    peer: int
    baseline_timestamp: int
    delay_role: str
    value: int
    contributor_mask: int


@dataclass(frozen=True)
class ProgressSnapshotPayload:
    """Cumulative GPU/Proxy/Network progress evidence."""

    total_chunks: int
    gpu_ready: int
    rdma_transmitted: int
    rdma_done: int


@dataclass(frozen=True)
class OperationCompletionPayload:
    """Operation-level completion data measured on a monotonic clock."""

    started_at_ns: int
    message_bytes: int


EventPayload = (
    RingActionPayload | ProgressSnapshotPayload | OperationCompletionPayload
)


@dataclass(frozen=True)
class Event:
    """One versioned event with a common envelope and a typed payload."""

    schema_version: int
    event_id: str
    event_kind: EventKind
    source: str
    context: EventContext
    time: LogicalTime
    dependencies: tuple[str, ...]
    payload: EventPayload

    def to_dict(self) -> dict[str, Any]:
        """Validate and return a JSON-compatible versioned event mapping."""
        from .validate import validate_event

        validate_event(self)
        if isinstance(self.payload, RingActionPayload):
            payload = {
                "phase": self.payload.phase,
                "step": self.payload.step,
                "chunk": self.payload.chunk,
                "action": self.payload.action,
                "peer": self.payload.peer,
                "baseline_timestamp": self.payload.baseline_timestamp,
                "delay_role": self.payload.delay_role,
                "value": self.payload.value,
                "contributor_mask": self.payload.contributor_mask,
            }
        elif isinstance(self.payload, ProgressSnapshotPayload):
            payload = {
                "total_chunks": self.payload.total_chunks,
                "gpu_ready": self.payload.gpu_ready,
                "rdma_transmitted": self.payload.rdma_transmitted,
                "rdma_done": self.payload.rdma_done,
            }
        else:
            payload = {
                "started_at_ns": self.payload.started_at_ns,
                "message_bytes": self.payload.message_bytes,
            }

        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_kind": self.event_kind.value,
            "source": self.source,
            "context": {
                "communicator_id": self.context.communicator_id,
                "op_seq": self.context.op_seq,
                "collective": self.context.collective,
                "rank": self.context.rank,
                "channel": self.context.channel,
            },
            "time": {
                "value": self.time.value,
                "domain": self.time.domain.value,
            },
            "dependencies": list(self.dependencies),
            "payload": payload,
        }

    def to_json(self) -> str:
        """Serialize one validated event as compact deterministic JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Event:
        """Parse and validate one versioned event mapping."""
        _require_exact_keys(
            data,
            {
                "schema_version",
                "event_id",
                "event_kind",
                "source",
                "context",
                "time",
                "dependencies",
                "payload",
            },
            "event",
        )

        context_data = _require_mapping(data["context"], "context")
        _require_exact_keys(
            context_data,
            {"communicator_id", "op_seq", "collective", "rank", "channel"},
            "context",
        )
        time_data = _require_mapping(data["time"], "time")
        _require_exact_keys(time_data, {"value", "domain"}, "time")

        try:
            event_kind = EventKind(data["event_kind"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"unsupported event_kind: {data['event_kind']!r}"
            ) from error
        try:
            time_domain = TimeDomain(time_data["domain"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"unsupported time domain: {time_data['domain']!r}"
            ) from error

        payload_data = _require_mapping(data["payload"], "payload")
        if event_kind is EventKind.RING_ACTION:
            _require_exact_keys(
                payload_data,
                {
                    "phase",
                    "step",
                    "chunk",
                    "action",
                    "peer",
                    "baseline_timestamp",
                    "delay_role",
                    "value",
                    "contributor_mask",
                },
                "ring_action payload",
            )
            payload: EventPayload = RingActionPayload(
                phase=payload_data["phase"],
                step=payload_data["step"],
                chunk=payload_data["chunk"],
                action=payload_data["action"],
                peer=payload_data["peer"],
                baseline_timestamp=payload_data["baseline_timestamp"],
                delay_role=payload_data["delay_role"],
                value=payload_data["value"],
                contributor_mask=payload_data["contributor_mask"],
            )
        elif event_kind is EventKind.PROGRESS_SNAPSHOT:
            _require_exact_keys(
                payload_data,
                {
                    "total_chunks",
                    "gpu_ready",
                    "rdma_transmitted",
                    "rdma_done",
                },
                "progress_snapshot payload",
            )
            payload = ProgressSnapshotPayload(
                total_chunks=payload_data["total_chunks"],
                gpu_ready=payload_data["gpu_ready"],
                rdma_transmitted=payload_data["rdma_transmitted"],
                rdma_done=payload_data["rdma_done"],
            )
        else:
            _require_exact_keys(
                payload_data,
                {"started_at_ns", "message_bytes"},
                "operation_completion payload",
            )
            payload = OperationCompletionPayload(
                started_at_ns=payload_data["started_at_ns"],
                message_bytes=payload_data["message_bytes"],
            )

        dependencies_data = data["dependencies"]
        if not isinstance(dependencies_data, list):
            raise ValueError("dependencies must be a JSON array")

        event = cls(
            schema_version=data["schema_version"],
            event_id=data["event_id"],
            event_kind=event_kind,
            source=data["source"],
            context=EventContext(
                communicator_id=context_data["communicator_id"],
                op_seq=context_data["op_seq"],
                collective=context_data["collective"],
                rank=context_data["rank"],
                channel=context_data["channel"],
            ),
            time=LogicalTime(
                value=time_data["value"],
                domain=time_domain,
            ),
            dependencies=tuple(dependencies_data),
            payload=payload,
        )
        from .validate import validate_event

        validate_event(event)
        return event

    @classmethod
    def from_json(cls, line: str) -> Event:
        """Parse and validate one JSON object encoded as text."""
        try:
            data = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid event JSON: {error.msg}") from error
        if not isinstance(data, dict):
            raise ValueError("event JSON must contain one object")
        return cls.from_dict(data)


def _require_mapping(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    return value


def _require_exact_keys(
    data: Mapping[str, object], expected: set[str], where: str
) -> None:
    actual = set(data)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ValueError(f"{where} missing required fields: {', '.join(missing)}")
    if extra:
        raise ValueError(f"{where} has unknown fields: {', '.join(extra)}")
