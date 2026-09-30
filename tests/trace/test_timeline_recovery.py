from __future__ import annotations

import random
import unittest
from dataclasses import replace

from mycroft.schema import (
    Event,
    EventContext,
    EventKind,
    LogicalTime,
    OperationCompletionPayload,
    ProgressSnapshotPayload,
    RingActionPayload,
    SCHEMA_VERSION,
    TimeDomain,
)
from mycroft.trace import (
    EventConflictError,
    FlowKey,
    MissingDependency,
    OperationKey,
    OperationMetadataConflictError,
    flow_key_for,
    group_events,
    operation_key_for,
    recover_timelines,
)


def progress_event(
    communicator_id: str,
    op_seq: int,
    rank: int,
    channel: int,
    tick: int,
    *,
    dependencies: tuple[str, ...] = (),
) -> Event:
    event_id = (
        f"progress:{communicator_id}:{op_seq}:{rank}:{channel}:{tick}"
    )
    progress = min(tick, 4)
    return Event(
        schema_version=SCHEMA_VERSION,
        event_id=event_id,
        event_kind=EventKind.PROGRESS_SNAPSHOT,
        source="day12_fixture",
        context=EventContext(
            communicator_id=communicator_id,
            op_seq=op_seq,
            collective="all_reduce",
            rank=rank,
            channel=channel,
        ),
        time=LogicalTime(tick, TimeDomain.E02_TICK),
        dependencies=dependencies,
        payload=ProgressSnapshotPayload(
            total_chunks=4,
            gpu_ready=progress,
            rdma_transmitted=progress,
            rdma_done=progress,
        ),
    )


def ring_event(
    communicator_id: str,
    op_seq: int,
    rank: int,
    channel: int,
) -> Event:
    return Event(
        schema_version=SCHEMA_VERSION,
        event_id=f"ring:{communicator_id}:{op_seq}:{rank}:{channel}:0",
        event_kind=EventKind.RING_ACTION,
        source="day12_fixture",
        context=EventContext(
            communicator_id=communicator_id,
            op_seq=op_seq,
            collective="all_reduce",
            rank=rank,
            channel=channel,
        ),
        time=LogicalTime(0, TimeDomain.E01_EVENT_ORDER),
        dependencies=(),
        payload=RingActionPayload(
            phase="reduce_scatter",
            step=0,
            chunk=0,
            action="send",
            peer=(rank + 1) % 4,
            baseline_timestamp=0,
            delay_role="none",
            value=rank + 1,
            contributor_mask=1 << rank,
        ),
    )


def completion_event(
    communicator_id: str,
    op_seq: int,
    rank: int,
    completed_at_ns: int,
) -> Event:
    return Event(
        schema_version=SCHEMA_VERSION,
        event_id=(
            f"completion:{communicator_id}:{op_seq}:{rank}:"
            f"{completed_at_ns}"
        ),
        event_kind=EventKind.OPERATION_COMPLETION,
        source="day13_correction_fixture",
        context=EventContext(
            communicator_id=communicator_id,
            op_seq=op_seq,
            collective="all_reduce",
            rank=rank,
            channel=None,
        ),
        time=LogicalTime(
            completed_at_ns, TimeDomain.NCCL_MONOTONIC_NS
        ),
        dependencies=(),
        payload=OperationCompletionPayload(
            started_at_ns=completed_at_ns - 100,
            message_bytes=4_096,
        ),
    )


def mixed_fixture() -> tuple[Event, ...]:
    comm_a_op0_tick0 = progress_event("comm-a", 0, 0, 0, 0)
    comm_a_op0_tick1 = progress_event(
        "comm-a",
        0,
        0,
        0,
        1,
        dependencies=(comm_a_op0_tick0.event_id,),
    )
    comm_b_op0_tick0 = progress_event("comm-b", 0, 1, 0, 0)
    comm_b_op0_tick1 = progress_event(
        "comm-b",
        0,
        1,
        0,
        1,
        dependencies=(comm_b_op0_tick0.event_id,),
    )
    return (
        comm_a_op0_tick0,
        comm_a_op0_tick1,
        progress_event("comm-a", 0, 0, 1, 0),
        progress_event("comm-a", 1, 0, 0, 0),
        comm_b_op0_tick0,
        comm_b_op0_tick1,
    )


class TimelineRecoveryTests(unittest.TestCase):
    def test_operation_and_flow_keys_keep_only_relevant_identity(self) -> None:
        progress = progress_event("comm-a", 7, 2, 3, 0)
        ring = ring_event("comm-a", 7, 2, 3)

        expected_operation = OperationKey("comm-a", 7)
        self.assertEqual(operation_key_for(progress), expected_operation)
        self.assertEqual(operation_key_for(ring), expected_operation)
        self.assertEqual(
            flow_key_for(progress),
            FlowKey(
                operation=expected_operation,
                rank=2,
                channel=3,
                event_kind=EventKind.PROGRESS_SNAPSHOT,
                time_domain=TimeDomain.E02_TICK,
            ),
        )
        self.assertNotEqual(flow_key_for(progress), flow_key_for(ring))

    def test_operation_completion_uses_operation_wide_timeline(self) -> None:
        progress = progress_event("comm-a", 7, 2, 3, 0)
        completion = completion_event("comm-a", 7, 2, 1_000)

        result = recover_timelines((completion, progress))

        self.assertEqual(len(result.timelines), 2)
        completion_timeline = next(
            timeline
            for timeline in result.timelines
            if timeline.key.event_kind is EventKind.OPERATION_COMPLETION
        )
        self.assertEqual(completion_timeline.key.channel, -1)
        self.assertEqual(completion_timeline.events, (completion,))

    def test_groups_multiple_communicators_operations_and_channels(self) -> None:
        events = mixed_fixture()
        result = group_events((*events, events[0]))
        for seed in range(10):
            with self.subTest(seed=seed):
                shuffled = list(events)
                random.Random(seed).shuffle(shuffled)
                self.assertEqual(
                    group_events((*shuffled, events[0])), result
                )

        self.assertEqual(
            tuple(group.key for group in result.groups),
            (
                OperationKey("comm-a", 0),
                OperationKey("comm-a", 1),
                OperationKey("comm-b", 0),
            ),
        )
        self.assertEqual(result.duplicate_event_ids, (events[0].event_id,))
        self.assertEqual(sum(len(group.events) for group in result.groups), 6)

    def test_same_operation_identity_requires_consistent_collective(self) -> None:
        event = mixed_fixture()[0]
        conflicting_context = replace(event.context, collective="broadcast")
        conflicting = replace(
            event,
            event_id=f"{event.event_id}:broadcast",
            context=conflicting_context,
        )

        with self.assertRaisesRegex(
            OperationMetadataConflictError, "collective"
        ):
            group_events((event, conflicting))

    def test_same_event_id_with_different_content_is_a_conflict(self) -> None:
        event = mixed_fixture()[0]
        conflicting = replace(event, source="different_source")

        with self.assertRaisesRegex(EventConflictError, event.event_id):
            group_events((event, conflicting))

    def test_shuffled_input_recovers_identical_timelines(self) -> None:
        events = list(mixed_fixture())
        expected = recover_timelines(events)

        for seed in range(10):
            with self.subTest(seed=seed):
                shuffled = list(events)
                random.Random(seed).shuffle(shuffled)
                self.assertEqual(recover_timelines(shuffled), expected)

        self.assertEqual(len(expected.timelines), 4)
        for timeline in expected.timelines:
            times = tuple(event.time.value for event in timeline.events)
            self.assertEqual(times, tuple(sorted(times)))

    def test_missing_dependencies_are_reported_not_discarded(self) -> None:
        events = list(mixed_fixture())
        owner = replace(events[1], dependencies=("absent-event",))
        events[1] = owner

        result = recover_timelines(events)

        self.assertEqual(
            result.missing_dependencies,
            (MissingDependency(owner.event_id, "absent-event"),),
        )
        recovered_ids = {
            event.event_id
            for timeline in result.timelines
            for event in timeline.events
        }
        self.assertIn(owner.event_id, recovered_ids)


if __name__ == "__main__":
    unittest.main()
