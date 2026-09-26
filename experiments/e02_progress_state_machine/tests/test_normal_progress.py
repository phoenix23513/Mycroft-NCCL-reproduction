from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "progress_sim.py"
SPEC = importlib.util.spec_from_file_location("e02_progress_sim", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load progress simulator from {MODULE_PATH}")
PROGRESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROGRESS
SPEC.loader.exec_module(PROGRESS)


class NormalProgressTests(unittest.TestCase):
    def test_four_chunks_move_through_three_stage_pipeline(self) -> None:
        snapshots = PROGRESS.run_normal_progress(
            total_chunks=4,
            max_ticks=16,
        )

        observed = [
            (
                item.tick,
                item.gpu_ready,
                item.rdma_transmitted,
                item.rdma_done,
            )
            for item in snapshots
        ]
        self.assertEqual(
            observed,
            [
                (0, 0, 0, 0),
                (1, 1, 0, 0),
                (2, 2, 1, 0),
                (3, 3, 2, 1),
                (4, 4, 3, 2),
                (5, 4, 4, 3),
                (6, 4, 4, 4),
            ],
        )

    def test_trace_is_monotonic_and_preserves_stage_order(self) -> None:
        snapshots = PROGRESS.run_normal_progress(
            total_chunks=4,
            max_ticks=16,
        )

        for previous, current in zip(snapshots, snapshots[1:]):
            self.assertLessEqual(previous.gpu_ready, current.gpu_ready)
            self.assertLessEqual(
                previous.rdma_transmitted,
                current.rdma_transmitted,
            )
            self.assertLessEqual(previous.rdma_done, current.rdma_done)
            self.assertLessEqual(current.rdma_done, current.rdma_transmitted)
            self.assertLessEqual(
                current.rdma_transmitted,
                current.gpu_ready,
            )
            self.assertLessEqual(
                current.gpu_ready - previous.gpu_ready,
                1,
            )
            self.assertLessEqual(
                current.rdma_transmitted - previous.rdma_transmitted,
                1,
            )
            self.assertLessEqual(
                current.rdma_done - previous.rdma_done,
                1,
            )

    def test_rejects_invalid_run_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "total_chunks"):
            PROGRESS.run_normal_progress(total_chunks=0)
        with self.assertRaisesRegex(ValueError, "max_ticks"):
            PROGRESS.run_normal_progress(max_ticks=0)


if __name__ == "__main__":
    unittest.main()
