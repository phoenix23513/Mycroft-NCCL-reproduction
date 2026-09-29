"""Public API for the versioned Mycroft event schema."""

from .adapters import (
    adapt_e01_record,
    adapt_e01_trace,
    adapt_e02_snapshot,
    adapt_e02_trace,
)
from .event import (
    Event,
    EventContext,
    EventKind,
    LogicalTime,
    ProgressSnapshotPayload,
    RingActionPayload,
    TimeDomain,
)
from .validate import EventValidationError, validate_event
from .version import SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS

__all__ = [
    "Event",
    "EventContext",
    "EventKind",
    "EventValidationError",
    "LogicalTime",
    "ProgressSnapshotPayload",
    "RingActionPayload",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "TimeDomain",
    "adapt_e01_record",
    "adapt_e01_trace",
    "adapt_e02_snapshot",
    "adapt_e02_trace",
    "validate_event",
]
