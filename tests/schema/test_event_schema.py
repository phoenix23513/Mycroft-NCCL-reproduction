from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
E02_DIR = PROJECT_ROOT / "experiments" / "e02_progress_state_machine"
sys.path.insert(0, str(E02_DIR))

from progress_sim import ProgressSnapshot, run_normal_progress  # noqa: E402

from mycroft.schema import (  # noqa: E402
    Event,
    EventContext,
    EventKind,
    EventValidationError,
    LogicalTime,
    OperationCompletionPayload,
    ProgressSnapshotPayload,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    TimeDomain,
    adapt_e01_record,
    adapt_e01_trace,
    adapt_e02_snapshot,
    adapt_e02_trace,
    validate_event,
)


def e01_record() -> dict[str, object]:
    return {
        "op_seq": 0,
        "rank": 1,
        "channel": 0,
        "phase": "reduce_scatter",
        "step": 0,
        "chunk": 0,
        "action": "send",
        "peer": 2,
        "baseline_timestamp": 2,
        "timestamp": 2,
        "delay_role": "none",
        "value": 11,
        "contributor_mask": 2,
    }


def completion_event() -> Event:
    return Event(
        schema_version=SCHEMA_VERSION,
        event_id="completion:comm-7:3:2",
        event_kind=EventKind.OPERATION_COMPLETION,
        source="nccl_trace",
        context=EventContext(
            communicator_id="comm-7",
            op_seq=3,
            collective="all_reduce",
            rank=2,
            channel=None,
        ),
        time=LogicalTime(2_000, TimeDomain.NCCL_MONOTONIC_NS),
        dependencies=(),
        payload=OperationCompletionPayload(
            started_at_ns=1_000,
            message_bytes=4_096,
        ),
    )


class EventSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = EventContext(
            communicator_id="comm-7",
            op_seq=3,
            collective="all_reduce",
            rank=2,
            channel=1,
        )

    def test_e01_adapter_and_json_round_trip(self) -> None:
        event = adapt_e01_record(
            e01_record(),
            communicator_id="comm-7",
            dependencies=("previous-event",),
        )

        validate_event(event)
        self.assertEqual(event.event_kind, EventKind.RING_ACTION)
        self.assertEqual(event.context.rank, 1)
        self.assertEqual(event.time.domain, TimeDomain.E01_EVENT_ORDER)
        self.assertEqual(Event.from_json(event.to_json()), event)
        self.assertNotIn("gpu_ready", event.to_json())

    def test_e02_adapter_and_json_round_trip(self) -> None:
        snapshot = ProgressSnapshot(
            tick=5,
            gpu_ready=4,
            rdma_transmitted=3,
            rdma_done=2,
        )
        event = adapt_e02_snapshot(
            snapshot,
            total_chunks=8,
            context=self.context,
        )

        validate_event(event)
        self.assertEqual(event.event_kind, EventKind.PROGRESS_SNAPSHOT)
        self.assertEqual(event.context, self.context)
        self.assertEqual(event.time.domain, TimeDomain.E02_TICK)
        self.assertEqual(Event.from_dict(event.to_dict()), event)
        self.assertNotIn("phase", event.to_json())

    def test_operation_completion_v2_round_trip(self) -> None:
        event = completion_event()

        validate_event(event)
        self.assertEqual(SCHEMA_VERSION, 2)
        self.assertEqual(SUPPORTED_SCHEMA_VERSIONS, frozenset({1, 2}))
        self.assertIsNone(event.context.channel)
        self.assertEqual(Event.from_json(event.to_json()), event)

    def test_v1_event_remains_readable(self) -> None:
        event = adapt_e02_snapshot(
            ProgressSnapshot(1, 1, 0, 0),
            total_chunks=4,
            context=self.context,
        )
        data = event.to_dict()
        data["schema_version"] = 1

        restored = Event.from_dict(data)

        self.assertEqual(restored.schema_version, 1)
        self.assertEqual(restored.time.domain, TimeDomain.E02_TICK)

    def test_real_progress_v2_accepts_monotonic_time(self) -> None:
        event = Event(
            schema_version=SCHEMA_VERSION,
            event_id="progress:comm-7:3:2:1:1000",
            event_kind=EventKind.PROGRESS_SNAPSHOT,
            source="nccl_trace",
            context=self.context,
            time=LogicalTime(1_000, TimeDomain.NCCL_MONOTONIC_NS),
            dependencies=(),
            payload=ProgressSnapshotPayload(
                total_chunks=8,
                gpu_ready=4,
                rdma_transmitted=3,
                rdma_done=2,
            ),
        )

        validate_event(event)
        with self.assertRaisesRegex(EventValidationError, "time domain"):
            validate_event(replace(event, schema_version=1))

    def test_completion_rejects_invalid_v2_contract(self) -> None:
        event = completion_event()
        invalid_events = (
            replace(event, schema_version=1),
            replace(event, context=replace(event.context, channel=0)),
            replace(
                event,
                time=replace(event.time, domain=TimeDomain.E02_TICK),
            ),
            replace(
                event,
                payload=replace(event.payload, started_at_ns=2_001),
            ),
            replace(
                event,
                payload=replace(event.payload, message_bytes=-1),
            ),
        )
        for invalid in invalid_events:
            with self.subTest(event=invalid):
                with self.assertRaises(EventValidationError):
                    validate_event(invalid)

    def test_e01_trace_adapter_preserves_message_and_rank_dependencies(self) -> None:
        send = e01_record()
        receive = dict(send)
        receive.update(
            rank=2,
            peer=1,
            action="recv",
            baseline_timestamp=3,
            timestamp=3,
        )
        next_on_rank = dict(send)
        next_on_rank.update(
            rank=2,
            peer=3,
            chunk=1,
            baseline_timestamp=4,
            timestamp=4,
        )

        events = adapt_e01_trace(
            (send, receive, next_on_rank), communicator_id="comm-7"
        )

        self.assertEqual(events[0].dependencies, ())
        self.assertEqual(events[1].dependencies, (events[0].event_id,))
        self.assertEqual(events[2].dependencies, (events[1].event_id,))

    def test_e02_trace_adapter_links_monotonic_snapshots(self) -> None:
        events = adapt_e02_trace(
            run_normal_progress(total_chunks=4),
            total_chunks=4,
            context=self.context,
        )

        self.assertEqual(events[0].dependencies, ())
        for previous, current in zip(events, events[1:]):
            self.assertEqual(current.dependencies, (previous.event_id,))

        with self.assertRaisesRegex(ValueError, "monotonic"):
            adapt_e02_trace(
                (ProgressSnapshot(0, 1, 0, 0), ProgressSnapshot(1, 0, 0, 0)),
                total_chunks=4,
                context=self.context,
            )

    def test_adapters_reject_missing_identity_fields(self) -> None:
        for field in ("op_seq", "rank"):
            with self.subTest(field=field):
                record = e01_record()
                del record[field]
                with self.assertRaisesRegex(ValueError, field):
                    adapt_e01_record(record, communicator_id="comm-7")

        invalid_contexts = (
            replace(self.context, communicator_id=""),
            replace(self.context, op_seq=-1),
            replace(self.context, rank=-1),
            replace(self.context, channel=None),
        )
        for context in invalid_contexts:
            with self.subTest(context=context):
                with self.assertRaises(EventValidationError):
                    adapt_e02_snapshot(
                        ProgressSnapshot(1, 1, 0, 0),
                        total_chunks=4,
                        context=context,
                    )

    def test_rejects_invalid_progress_relation(self) -> None:
        invalid_snapshots = (
            ProgressSnapshot(3, 2, 3, 1),
            ProgressSnapshot(3, 5, 3, 1),
            ProgressSnapshot(-1, 0, 0, 0),
        )
        for snapshot in invalid_snapshots:
            with self.subTest(snapshot=snapshot):
                with self.assertRaises(EventValidationError):
                    adapt_e02_snapshot(
                        snapshot,
                        total_chunks=4,
                        context=self.context,
                    )

    def test_rejects_unsupported_version_and_kind_payload_mismatch(self) -> None:
        event = adapt_e01_record(e01_record(), communicator_id="comm-7")

        with self.assertRaisesRegex(EventValidationError, "schema_version"):
            validate_event(replace(event, schema_version=3))
        with self.assertRaisesRegex(EventValidationError, "ProgressSnapshotPayload"):
            validate_event(
                replace(event, event_kind=EventKind.PROGRESS_SNAPSHOT)
            )

    def test_rejects_invalid_dependencies(self) -> None:
        event = adapt_e01_record(e01_record(), communicator_id="comm-7")
        invalid_dependencies = (
            (event.event_id,),
            ("same", "same"),
            ("",),
        )
        for dependencies in invalid_dependencies:
            with self.subTest(dependencies=dependencies):
                with self.assertRaises(EventValidationError):
                    validate_event(replace(event, dependencies=dependencies))

    def test_rejects_wrong_time_domain_and_inconsistent_delay_role(self) -> None:
        ring_event = adapt_e01_record(e01_record(), communicator_id="comm-7")
        with self.assertRaisesRegex(EventValidationError, "time domain"):
            validate_event(
                replace(
                    ring_event,
                    time=replace(ring_event.time, domain=TimeDomain.E02_TICK),
                )
            )

        delayed_record = e01_record()
        delayed_record["delay_role"] = "affected"
        with self.assertRaisesRegex(EventValidationError, "after"):
            adapt_e01_record(delayed_record, communicator_id="comm-7")

    def test_serialized_shape_is_strict_and_versioned(self) -> None:
        event = adapt_e02_snapshot(
            ProgressSnapshot(1, 1, 0, 0),
            total_chunks=4,
            context=self.context,
        )
        data = event.to_dict()
        self.assertEqual(data["schema_version"], SCHEMA_VERSION)
        self.assertEqual(
            set(data),
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
        )

        missing = dict(data)
        del missing["event_id"]
        with self.assertRaisesRegex(ValueError, "event_id"):
            Event.from_dict(missing)

        extra = dict(data)
        extra["fault_target"] = "proxy_transmitter"
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            Event.from_dict(extra)

    def test_json_parser_rejects_non_object_and_unknown_kind(self) -> None:
        with self.assertRaisesRegex(ValueError, "one object"):
            Event.from_json("[]")

        event = adapt_e01_record(e01_record(), communicator_id="comm-7")
        data = event.to_dict()
        data["event_kind"] = "unknown"
        with self.assertRaisesRegex(ValueError, "event_kind"):
            Event.from_dict(data)


if __name__ == "__main__":
    unittest.main()
