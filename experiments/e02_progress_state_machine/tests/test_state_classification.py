from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import classify as CLASSIFY  # noqa: E402
from progress_sim import ProgressSnapshot, run_normal_progress  # noqa: E402


def snapshot(
    gpu_ready: int,
    rdma_transmitted: int,
    rdma_done: int,
    tick: int = 7,
) -> ProgressSnapshot:
    return ProgressSnapshot(
        tick=tick,
        gpu_ready=gpu_ready,
        rdma_transmitted=rdma_transmitted,
        rdma_done=rdma_done,
    )


class StateClassificationTests(unittest.TestCase):
    def test_paper_state_expectation_table(self) -> None:
        cases = (
            (
                "not started",
                snapshot(0, 0, 0),
                CLASSIFY.ProgressState.NOT_STARTED,
                ("uninitialized",),
                ("blocked",),
                ("launch_state", "upstream_progress"),
            ),
            (
                "not transmitted",
                snapshot(5, 3, 3),
                CLASSIFY.ProgressState.NOT_TRANSMITTED,
                ("rdma_issue",),
                ("receiver_not_ready",),
                ("receiver_progress",),
            ),
            (
                "not delivered",
                snapshot(5, 5, 2),
                CLASSIFY.ProgressState.NOT_DELIVERED,
                ("rdma_issue",),
                ("receiver_failed",),
                ("receiver_progress",),
            ),
            (
                "gpu not ready",
                snapshot(2, 2, 2),
                CLASSIFY.ProgressState.GPU_NOT_READY,
                ("gpu_issue",),
                (),
                ("gpu_execution",),
            ),
        )

        for name, observed, state, local, remote, gaps in cases:
            with self.subTest(name=name):
                result = CLASSIFY.classify_snapshot(observed, total_chunks=8)
                self.assertEqual(result.states, (state,))
                assessment = result.assessments[0]
                self.assertEqual(assessment.local_possible_causes, local)
                self.assertEqual(assessment.remote_possible_causes, remote)
                self.assertEqual(assessment.evidence_gaps, gaps)

    def test_overlapping_snapshot_conditions_are_preserved_in_order(self) -> None:
        result = CLASSIFY.classify_snapshot(
            snapshot(6, 4, 1),
            total_chunks=8,
        )

        self.assertEqual(
            result.states,
            (
                CLASSIFY.ProgressState.NOT_TRANSMITTED,
                CLASSIFY.ProgressState.NOT_DELIVERED,
            ),
        )

    def test_completed_snapshot_has_no_active_state(self) -> None:
        result = CLASSIFY.classify_snapshot(
            snapshot(8, 8, 8),
            total_chunks=8,
        )

        self.assertEqual(result.assessments, ())

    def test_normal_pipeline_states_are_not_fault_verdicts(self) -> None:
        snapshots = run_normal_progress(total_chunks=4, max_ticks=16)
        expected = (
            (CLASSIFY.ProgressState.NOT_STARTED,),
            (CLASSIFY.ProgressState.NOT_TRANSMITTED,),
            (
                CLASSIFY.ProgressState.NOT_TRANSMITTED,
                CLASSIFY.ProgressState.NOT_DELIVERED,
            ),
            (
                CLASSIFY.ProgressState.NOT_TRANSMITTED,
                CLASSIFY.ProgressState.NOT_DELIVERED,
            ),
            (
                CLASSIFY.ProgressState.NOT_TRANSMITTED,
                CLASSIFY.ProgressState.NOT_DELIVERED,
            ),
            (CLASSIFY.ProgressState.NOT_DELIVERED,),
            (),
        )

        observed = tuple(
            CLASSIFY.classify_snapshot(item, total_chunks=4).states
            for item in snapshots
        )

        self.assertEqual(observed, expected)

    def test_rejects_invalid_snapshot_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "total_chunks"):
            CLASSIFY.classify_snapshot(snapshot(0, 0, 0), total_chunks=0)
        with self.assertRaisesRegex(ValueError, "tick"):
            CLASSIFY.classify_snapshot(
                snapshot(0, 0, 0, tick=-1),
                total_chunks=8,
            )
        with self.assertRaisesRegex(ValueError, "progress state"):
            CLASSIFY.classify_snapshot(snapshot(2, 3, 1), total_chunks=8)


if __name__ == "__main__":
    unittest.main()
