"""CPU 记录器并发/丢失/导出检查；手写轨迹不构成 GPU 或 NCCL 验收证据。"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mycroft.schema import Event, EventKind


class CaptureRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix='m2-cpu-runtime-')
        cls.addClassCleanup(cls.build.cleanup)
        cls.program = Path(cls.build.name) / 'capture-fixture'
        subprocess.run([*shlex.split(os.environ.get('CXX', 'g++')), '-std=c++17', '-pthread',
                        '-Wall', '-Wextra', '-Wpedantic', '-Werror', '-I', str(SOURCE / 'include'),
                        str(SOURCE / 'src/operation_trace.cpp'),
                        str(SOURCE / 'tests/capture_fixture.cpp'), '-o', str(cls.program)],
                       check=True, capture_output=True, text=True)

    def run_case(self, name):
        temp = tempfile.TemporaryDirectory(prefix='m2-cpu-case-')
        self.addCleanup(temp.cleanup)
        output = Path(temp.name) / 'output'
        result = subprocess.run([str(self.program), name, str(output)],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        return output

    def read_lines(self, output, name):
        return [json.loads(line) for line in (output / 'trace' / name).read_text().splitlines()]

    def manifest(self, output):
        return json.loads((output / 'capture-manifest.json').read_text())

    def test_periodic_stall_snapshots_and_event_schema(self):
        output = self.run_case('normal')
        events = [Event.from_dict(row) for row in self.read_lines(output, 'rank0.jsonl')]
        progress = [e for e in events if e.event_kind is EventKind.PROGRESS_SNAPSHOT]
        self.assertEqual([e.time.value for e in progress], [120, 130, 140, 150])
        self.assertEqual(progress[0].payload, progress[1].payload)
        self.assertEqual(progress[0].payload.total_chunks, 8)
        self.assertEqual(progress[0].payload.gpu_ready, 4)
        self.assertEqual(progress[-1].payload.rdma_done, 8)
        completion = [e for e in events if e.event_kind is EventKind.OPERATION_COMPLETION]
        self.assertEqual(len(completion), 1)
        self.assertIsNone(completion[0].context.channel)
        self.assertEqual(completion[0].payload.message_bytes, 4096)
        self.assertEqual(completion[0].payload.started_at_ns, 100)
        self.assertEqual(completion[0].context.communicator_id, 'cpu_fixture:0000000000000abc')
        self.assertEqual(len({e.event_id for e in events}), len(events))
        raw = self.read_lines(output, 'send-rank0.jsonl')
        self.assertEqual([r['gpu_ready_steps'] for r in raw], [8, 8, 16, 16])
        self.assertEqual(raw[0]['step_base'], 64)
        self.assertEqual(len(self.read_lines(output, 'recv-rank0.jsonl')), 2)
        metadata = self.manifest(output)
        self.assertEqual(metadata['recorded'], 12)
        self.assertTrue(metadata['local_contract_valid'])
        self.assertFalse(metadata['diagnostic_eligible'])
        self.assertFalse(metadata['nccl_adapter_verified'])
        self.assertFalse(metadata['completion_observation_verified'])
        self.assertEqual(metadata['evidence_scope'], 'process_local_recorder_only')

    def test_concurrent_capacity_drops_and_partial_evidence(self):
        output = self.run_case('overflow')
        metadata = self.manifest(output)
        self.assertEqual(metadata['recorded'], 6)
        self.assertEqual(metadata['dropped'], 1994)
        rows = self.read_lines(output, 'channel-map.jsonl')
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({r['op_seq_candidate'] for r in rows}), 6)
        self.assertFalse(metadata['local_contract_valid'])
        self.assertEqual(self.read_lines(output, 'rank0.jsonl'), [])

    def test_invalid_and_unsupported_records_never_become_valid_capture(self):
        metadata = self.manifest(self.run_case('invalid'))
        self.assertEqual(metadata['invalid'], 5)
        self.assertEqual(metadata['unsupported'], 4)
        self.assertFalse(metadata['local_contract_valid'])

    def test_concurrent_writers_have_no_holes_or_duplicate_slots(self):
        output = self.run_case('concurrent')
        metadata = self.manifest(output)
        self.assertEqual(metadata['recorded'], 2000)
        self.assertEqual(metadata['dropped'], 0)
        rows = self.read_lines(output, 'channel-map.jsonl')
        self.assertEqual({r['op_seq_candidate'] for r in rows}, set(range(2000)))

    def test_missing_wrong_and_duplicate_completion_proofs_are_preserved(self):
        for scenario in ('missing_plan', 'wrong_stream', 'duplicate_completion', 'duplicate_begin', 'missing_channel'):
            with self.subTest(scenario=scenario):
                output = self.run_case(scenario)
                events = self.read_lines(output, 'rank0.jsonl')
                self.assertFalse(any(e['event_kind'] == 'operation_completion' for e in events))
                plans = self.read_lines(output, 'plan-map.jsonl')
                self.assertTrue(any(r['record_kind'] == 'operation_completion' for r in plans))
                self.assertGreater(self.manifest(output)['association_errors'], 0)
                self.assertFalse(self.manifest(output)['local_contract_valid'])

    def test_regressing_counters_keep_raw_evidence_and_suppress_progress_events(self):
        output = self.run_case('counter_regression')
        self.assertEqual(len(self.read_lines(output, 'send-rank0.jsonl')), 2)
        self.assertFalse(any(e['event_kind'] == 'progress_snapshot'
                             for e in self.read_lines(output, 'rank0.jsonl')))
        self.assertFalse(self.manifest(output)['local_contract_valid'])

    def test_progress_requires_matching_peer_direction(self):
        for scenario in ('orphan_peer', 'wrong_direction', 'ambiguous_send'):
            with self.subTest(scenario=scenario):
                output = self.run_case(scenario)
                self.assertEqual(len(self.read_lines(output, 'send-rank0.jsonl')), 2)
                self.assertFalse(any(e['event_kind'] == 'progress_snapshot'
                                     for e in self.read_lines(output, 'rank0.jsonl')))
                self.assertFalse(self.manifest(output)['local_contract_valid'])

    def test_one_rank_per_process_export_scope(self):
        self.assertFalse(self.run_case('multi_rank').exists())

    def test_empty_capture_does_not_invent_rank_or_operation(self):
        output = self.run_case('empty')
        metadata = self.manifest(output)
        self.assertIsNone(metadata['rank'])
        self.assertEqual(metadata['recorded'], 0)
        self.assertFalse(metadata['local_contract_valid'])
        self.assertEqual(list((output / 'trace').glob('rank*.jsonl')), [])


if __name__ == '__main__':
    unittest.main()
