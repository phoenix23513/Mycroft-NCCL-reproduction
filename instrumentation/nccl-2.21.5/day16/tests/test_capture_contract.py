"""M2 格式/单位负例回归；手写数据不构成真实 NCCL 采集证据。"""
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from mycroft.schema import (Event, EventContext, EventKind, LogicalTime,
                            OperationCompletionPayload, ProgressSnapshotPayload, TimeDomain)

SPEC = importlib.util.spec_from_file_location(
    "m2_verify_capture", Path(__file__).resolve().parents[1] / "verify_capture.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class CaptureContractTests(unittest.TestCase):
    def normalize(self, **changes):
        args = dict(nsteps=16, slice_steps=2, gpu_ready_steps=8,
                    transmitted_steps=6, done_steps=4, registered_buffer=False)
        args.update(changes)
        return VERIFY.normalize_send_steps(**args)

    def event(self, kind=EventKind.PROGRESS_SNAPSHOT, domain=TimeDomain.NCCL_MONOTONIC_NS,
              collective="all_reduce", version=2):
        completion = kind is EventKind.OPERATION_COMPLETION
        return Event(
            schema_version=version, event_id="contract-only-1", event_kind=kind,
            source="unit_fixture_not_nccl", context=EventContext(
                "fixture:comm", 0, collective, 0, None if completion else 1),
            time=LogicalTime(20, domain), dependencies=(), payload=(
                OperationCompletionPayload(10, 16) if completion
                else ProgressSnapshotPayload(8, 4, 3, 2)))

    def test_all_counters_use_same_slice_unit(self):
        self.assertEqual(self.normalize(), ProgressSnapshotPayload(8, 4, 3, 2))

    def test_zero_progress_is_valid_and_must_not_be_filtered(self):
        self.assertEqual(self.normalize(gpu_ready_steps=0, transmitted_steps=0, done_steps=0),
                         ProgressSnapshotPayload(8, 0, 0, 0))

    def test_incomplete_slices_cannot_be_rounded_down(self):
        for changes in ({"nsteps": 15}, {"gpu_ready_steps": 7},
                        {"transmitted_steps": 5}, {"done_steps": 3}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "multiples"):
                self.normalize(**changes)

    def test_wrong_progress_order_and_absolute_counters_are_rejected(self):
        for changes in ({"gpu_ready_steps": 4}, {"done_steps": 8}, {"gpu_ready_steps": 18}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "expected"):
                self.normalize(**changes)

    def test_invalid_units_and_types_are_rejected(self):
        for changes in ({"slice_steps": 0}, {"nsteps": 0}, {"done_steps": -2},
                        {"done_steps": True}, {"done_steps": 4.0}, {"nsteps": 2**64}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.normalize(**changes)

    def test_registered_buffer_requires_separate_semantic_verification(self):
        with self.assertRaisesRegex(ValueError, "not verified"):
            self.normalize(registered_buffer=True)

    def test_event_formats_are_separate(self):
        progress = self.event().to_json()
        completion = self.event(EventKind.OPERATION_COMPLETION).to_json()
        self.assertEqual(VERIFY.validate_progress_line(progress), self.event())
        self.assertEqual(VERIFY.validate_completion_line(completion),
                         self.event(EventKind.OPERATION_COMPLETION))
        with self.assertRaises(ValueError):
            VERIFY.validate_completion_line(progress)
        with self.assertRaises(ValueError):
            VERIFY.validate_progress_line(completion)

    def test_simulation_and_other_collectives_are_not_m2_format(self):
        for event in (self.event(domain=TimeDomain.E02_TICK),
                      self.event(version=1, domain=TimeDomain.E02_TICK),
                      self.event(collective="broadcast")):
            with self.subTest(event=event), self.assertRaises(ValueError):
                VERIFY.validate_progress_line(event.to_json())

    def test_valid_format_never_bypasses_real_evidence_gate(self):
        VERIFY.validate_completion_line(self.event(EventKind.OPERATION_COMPLETION).to_json())
        with self.assertRaisesRegex(ValueError, "requires M2"):
            VERIFY.verify_capture(ROOT / "results/samples/e06/m1")


if __name__ == "__main__":
    unittest.main()
