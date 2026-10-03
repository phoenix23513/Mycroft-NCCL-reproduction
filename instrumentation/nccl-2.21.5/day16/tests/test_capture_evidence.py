"""手写 M2 证据负例/对账回归；混合历史 M1 元数据的 fixture 不是真实 M2 运行。"""
import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location("capture_evidence", Path(__file__).resolve().parents[1] / "verify_capture.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class EvidenceTests(unittest.TestCase):
    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data) + "\n")

    def rows(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in data))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m2-evidence-unit-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.rank = self.folder / "rank0"
        self.capture = self.rank / "capture"
        self.config = {"run_id": "a"*32}
        self.selections = [{"channels": 1} for _ in range(9)]
        self.mapping, self.plans, self.sends, self.recvs, self.events = [], [], [], [], []
        application = ["application_clock=CLOCK_MONOTONIC"]
        for op in range(9):
            start = 1000*(op+1)
            for stage, stamp in (("api_begin", start-10), ("api_return", start+5), ("stream_sync_return", start+100)):
                application.append(f"rank=0 operation={op} count={VERIFY.COUNTS[op//3]} stage={stage} time_ns={stamp}")
            k = {"comm_hash": 42, "op_seq_candidate": op, "rank": 0}
            self.mapping.extend([{**k, "record_kind": "operation_begin", "message_bytes": VERIFY.COUNTS[op//3]*4,
                                  "started_at_ns": start}, {**k, "record_kind": "channel_membership", "channel": 0}])
            self.mapping.extend({**k, "record_kind": "peer_connection", "channel": 0, "connection_id": conn,
                                 "peer_rank": 1, "direction": direction, "transport": "NET/IB"}
                                for direction, conn in (("send", 1), ("recv", 0)))
            self.plans.extend([{**k, "record_kind": "plan_binding", "plan_id": op+1, "stream_id": 1, "collective_count": 1},
                               {**k, "record_kind": "operation_completion", "plan_id": op+1, "stream_id": 1,
                                "observed_at_ns": start+90}])
            context = {"communicator_id": f"{self.config['run_id']}:000000000000002a", "op_seq": op,
                       "collective": "all_reduce", "rank": 0, "channel": 0}
            for offset, ready, submitted, done in ((10, 0, 0, 0), (30, 4, 2, 0), (50, 8, 8, 8)):
                row = {**k, "channel": 0, "connection_id": 1, "observed_at_ns": start+offset, "step_base": 64,
                       "nsteps": 8, "slice_steps": 2, "registered_buffer": False, "gpu_ready_steps": ready,
                       "transmitted_steps": submitted, "done_steps": done}
                self.sends.append(row)
                self.events.append({"schema_version": 2, "event_id": f"unit-progress-{op}-{offset}",
                    "event_kind": "progress_snapshot", "source": "m2_process_local_recorder_unverified",
                    "context": context, "time": {"value": start+offset, "domain": "nccl_monotonic_ns"},
                    "dependencies": [], "payload": {"total_chunks": 4, "gpu_ready": ready//2,
                                                     "rdma_transmitted": submitted//2, "rdma_done": done//2}})
            self.recvs.append({**k, "channel": 0, "connection_id": 0, "observed_at_ns": start+80, "step_base": 64,
                "nsteps": 8, "slice_steps": 2, "registered_buffer": False, "posted_steps": 8, "received_steps": 8,
                "transmitted_steps": 8, "done_steps": 8})
            self.events.append({"schema_version": 2, "event_id": f"unit-completion-{op}",
                "event_kind": "operation_completion", "source": "m2_process_local_recorder_unverified",
                "context": {**context, "channel": None}, "time": {"value": start+90, "domain": "nccl_monotonic_ns"},
                "dependencies": [], "payload": {"started_at_ns": start, "message_bytes": VERIFY.COUNTS[op//3]*4}})
        self.manifest = {"capture_id": self.config["run_id"], "rank": 0, "phase": "stopped", "export_complete": True,
                         "local_contract_valid": True, "recorded": 100, "dropped": 0, "invalid": 0, "unsupported": 0,
                         "association_errors": 0, "sample_period_ns": 10, "event_records": len(self.events)}
        self.adapter = {"adapter_version": 1, "base_commit": VERIFY.COMMIT,
            "scope": "two_node_serial_float32_ring_simple_net_ib", "clock": "CLOCK_MONOTONIC", "adapter_error": 0,
            "pending_operation": False, "completed_operations": 9, "plan_count": 9,
            "completion_observation": "actual_launch_stream_event_query_after_application_stream_sync",
            "single_plan_collectives": 1, "stream_matches_user_stream": True, "producer_join_verified": True,
            "readiness": "observed_eligible_slice_frontier_before_isend", "exact_gpu_timestamp": False,
            "hardware_fault_proof": False, "send_samples": len(self.sends), "recv_samples": len(self.recvs)}
        self.rank.mkdir()
        (self.rank / "run.log").write_text("\n".join(application) + "\n")
        self.save()

    def save(self):
        self.write(self.capture / "capture-manifest.json", self.manifest)
        self.write(self.capture / "adapter-manifest.json", self.adapter)
        for name, rows in (("channel-map.jsonl", self.mapping), ("plan-map.jsonl", self.plans),
                           ("send-rank0.jsonl", self.sends), ("recv-rank0.jsonl", self.recvs), ("rank0.jsonl", self.events)):
            self.rows(self.capture / "trace" / name, rows)

    def verify(self):
        return VERIFY.verify_rank_capture(self.rank, 0, self.config, self.selections)

    def test_consistent_handwritten_contract_has_intermediate_windows(self):
        result = self.verify()
        self.assertEqual(result["periodic_windows"], 9)
        self.assertEqual(result["send_samples"], 27)
        # Rank-only format checks cannot establish source/runtime provenance.
        with self.assertRaises(OSError):
            VERIFY.verify_capture(self.folder)

    def test_loss_and_abort_errors_reject_diagnostic_evidence(self):
        self.manifest["dropped"] = 1
        self.save()
        with self.assertRaisesRegex(ValueError, "loss"):
            self.verify()
        self.manifest["dropped"] = 0
        self.adapter["adapter_error"] = 11
        self.save()
        with self.assertRaisesRegex(ValueError, "adapter"):
            self.verify()

    def test_wrong_raw_peer_and_duplicate_completion_are_rejected(self):
        self.mapping[2]["peer_rank"] = 0
        self.save()
        with self.assertRaisesRegex(ValueError, "peer"):
            self.verify()
        self.mapping[2]["peer_rank"] = 1
        self.plans.append(copy.deepcopy(self.plans[1]))
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.verify()

    def test_valid_event_with_wrong_raw_normalization_is_rejected(self):
        self.events[1]["payload"]["gpu_ready"] = 3
        self.save()
        with self.assertRaisesRegex(ValueError, "normalized Event"):
            self.verify()

    def test_missing_join_and_wrong_application_operation_are_rejected(self):
        self.adapter["producer_join_verified"] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "completion evidence"):
            self.verify()
        self.adapter["producer_join_verified"] = True
        self.mapping[0]["started_at_ns"] = 500
        self.save()
        with self.assertRaisesRegex(ValueError, "application call"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
