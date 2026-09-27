from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "progress_sim.py"
SPEC = importlib.util.spec_from_file_location("e02_fault_progress_sim", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load progress simulator from {MODULE_PATH}")
PROGRESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROGRESS
SPEC.loader.exec_module(PROGRESS)


class FaultInjectionTests(unittest.TestCase):
    def test_fault_window_is_half_open(self) -> None:
        fault = PROGRESS.FaultSpec(
            target="proxy_transmitter",
            start_tick=3,
            duration_ticks=2,
        )

        self.assertFalse(fault.is_active("proxy_transmitter", 2))
        self.assertTrue(fault.is_active("proxy_transmitter", 3))
        self.assertTrue(fault.is_active("proxy_transmitter", 4))
        self.assertFalse(fault.is_active("proxy_transmitter", 5))
        self.assertFalse(fault.is_active("gpu_producer", 3))

    def test_rejects_invalid_fault_specs(self) -> None:
        with self.assertRaisesRegex(ValueError, "target"):
            PROGRESS.FaultSpec("unknown", 1, 1)
        with self.assertRaisesRegex(ValueError, "start_tick"):
            PROGRESS.FaultSpec("gpu_producer", 0, 1)
        with self.assertRaisesRegex(ValueError, "duration_ticks"):
            PROGRESS.FaultSpec("network_completer", 1, 0)

    def test_each_target_stalls_only_its_own_counter(self) -> None:
        configurations = (
            (4, 3, 2),
            (7, 2, 3),
        )
        counter_fields = {
            "gpu_producer": "gpu_ready",
            "proxy_transmitter": "rdma_transmitted",
            "network_completer": "rdma_done",
        }

        for total_chunks, start_tick, duration_ticks in configurations:
            for target in counter_fields:
                with self.subTest(
                    target=target,
                    total_chunks=total_chunks,
                    start_tick=start_tick,
                    duration_ticks=duration_ticks,
                ):
                    fault = PROGRESS.FaultSpec(
                        target=target,
                        start_tick=start_tick,
                        duration_ticks=duration_ticks,
                    )
                    snapshots = PROGRESS.run_progress(
                        total_chunks=total_chunks,
                        max_ticks=64,
                        fault=fault,
                    )

                    for previous, current in zip(snapshots, snapshots[1:]):
                        normal_next = {
                            "gpu_producer": min(
                                previous.gpu_ready + 1, total_chunks
                            ),
                            "proxy_transmitter": (
                                previous.rdma_transmitted + 1
                                if previous.rdma_transmitted < previous.gpu_ready
                                else previous.rdma_transmitted
                            ),
                            "network_completer": (
                                previous.rdma_done + 1
                                if previous.rdma_done < previous.rdma_transmitted
                                else previous.rdma_done
                            ),
                        }

                        for actor_name, field_name in counter_fields.items():
                            expected = normal_next[actor_name]
                            if fault.is_active(actor_name, current.tick):
                                expected = getattr(previous, field_name)
                            self.assertEqual(
                                getattr(current, field_name),
                                expected,
                            )

                        PROGRESS.validate_snapshot(current, total_chunks)

                    final = snapshots[-1]
                    self.assertEqual(final.gpu_ready, total_chunks)
                    self.assertEqual(final.rdma_transmitted, total_chunks)
                    self.assertEqual(final.rdma_done, total_chunks)


if __name__ == "__main__":
    unittest.main()
