"""构造日志用于核对器单元测试；真实本地失败用于打包检查，均非 M1 GPU 证据。"""
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workloads/minimal_allreduce"))
import verify_m1 as VERIFY


class M1VerificationTests(unittest.TestCase):
    def setUp(self):
        self.storage = tempfile.TemporaryDirectory()
        self.addCleanup(self.storage.cleanup)
        self.result = Path(self.storage.name)
        self.json("run-config.json", {"run_id": "unit-fixture", "iterations": 3,
                                      "expected_sha256": VERIFY.TRACE_SHA256, "settings": {}})
        for rank in (0, 1):
            folder = self.result / f"rank{rank}"
            folder.mkdir()
            self.json(f"rank{rank}/status.json", {"rank": rank, "run_id": "unit-fixture", "exit_code": 0})
            self.json(f"rank{rank}/environment.json", {"library_sha256": VERIFY.TRACE_SHA256,
                      "library_path": "<UNIT_LIBRARY>", "settings": {}})
            self.json(f"rank{rank}/probe.json", {"rank": rank, "blocked_reasons": [],
                      "hostname": f"UNIT_HOST_{rank}",
                      "node_identifiers": {"dmi_product_uuid_sha256": str(rank+1)*64,
                                           "kernel_boot_id_sha256": str(rank+3)*64},
                      "gpu_inventory": {"stdout": "index, name, uuid, pci.bus_id, driver_version\n"
                                        f"0, UNIT_GPU, UNIT_UUID_{rank}, 00000000:00:01.0, UNIT_DRIVER\n"}})
            (folder / "build-manifest.txt").write_text(
                f"source_commit={VERIFY.COMMIT}\nsource_tag=v2.21.5-1\nsource_clean=yes\ntrace=1\nstatus=PASS\n")
            lines = ["unit_fixture_only; not real NCCL evidence", "requested_library=<UNIT_LIBRARY>",
                     "loaded_library=<UNIT_LIBRARY>", "nccl_version_code=22105",
                     f"rank={rank} device=0 gpu_name=UNIT_GPU",
                     f"UNIT NCCL INFO comm UNIT rank {rank} nRanks 2 nNodes 2 localRanks 1 localRank 0 MNNVL 0",
                     "UNIT NCCL INFO Using network IB",
                     f"UNIT NCCL INFO ncclCommInitRank comm UNIT rank {rank} nranks 2 cudaDev 0 "
                     "nvmlDev 0 busId 10 commId UNIT_GROUP - Init COMPLETE",
                     f"UNIT NCCL INFO Channel 00/0 : {rank}[0] -> {1-rank}[0] [send] via NET/IB/0",
                     f"UNIT NCCL INFO Channel 00/0 : {1-rank}[0] -> {rank}[0] [receive] via NET/IB/0"]
            for op in range(9):
                count = VERIFY.COUNTS[op//3]
                prefix = f"rank={rank} operation={op} count={count}"
                lines.append(f"{prefix} stage=api_begin time_ns={op*10}")
                lines.append(f"UNIT NCCL INFO AllReduce: opCount {op:x} sendbuff UNIT count {count} "
                             "datatype 7 op 0 root 0 comm UNIT [nranks=2]")
                if rank == 0:
                    lines.append("UNIT NCCL TRACE CBDColl enqueue coll AllReduce(ncclSum, ncclFloat32, "
                                 f"RING, SIMPLE), nChannels 2, count {count} (nbytes {count*4}), UNIT")
                lines.extend([f"{prefix} stage=api_return time_ns={op*10+1}",
                              f"{prefix} stage=stream_sync_return time_ns={op*10+2}",
                              f"{prefix} expected={3+2*op} result=PASS"])
            lines.append(f"rank={rank} operations=9 status=PASS")
            (folder / "run.log").write_text("\n".join(lines)+"\n")

    def json(self, name, value):
        (self.result / name).write_text(json.dumps(value))

    def test_complete_unit_input_parsed(self):
        result = VERIFY.verify(self.result)
        self.assertEqual(result["node_evidence"], "DISTINCT_DMI_IDENTIFIERS")
        self.assertEqual(result["connections"], ["NET/IB/0"])

    def test_complete_proxy_trace_insertions_preserve_all_evidence(self):
        for rank in (0, 1):
            path = self.result / f"rank{rank}/run.log"
            trace = f"UNIT_HOST_{rank}:39:47 [0] 1.234567 ncclIbTest:1871 NCCL TRACE done\n"
            lines = []
            for line in path.read_text().splitlines(keepends=True):
                if re.match(r"rank=[01] operation=\d+ count=", line):
                    if " stage=" in line:
                        position = line.index(" time_ns=")
                    else:
                        position = line.index("expected=") + len("expected=")
                    lines.append(line[:position] + trace + trace + line[position:])
                else:
                    lines.append(line)
            path.write_text("".join(lines))
        result = VERIFY.verify(self.result)
        self.assertEqual(result["recovered_application_records"], {0: 36, 1: 36})

    def test_exact_hostname_and_multiple_fragment_boundaries(self):
        record = "rank=1 operation=4 count=4096 stage=api_begin time_ns=123456"
        trace = "UNIT_HOST_1:39:46 [0] 1.234567 sendProxyProgress:1153 NCCL TRACE done\n"
        for position in range(len("rank="), len(record)+1):
            with self.subTest(position=position):
                mixed = record[:position] + trace + record[position:] + "\n"
                restored, count = VERIFY.restore_application_records(mixed, "UNIT_HOST_1")
                self.assertEqual(restored, record + "\n" + trace)
                self.assertEqual(count, 1)
        mixed = "rank=" + trace + "1 operation=4 count=4096 expected=" + trace + "11 result=PASS\n"
        restored, count = VERIFY.restore_application_records(mixed, "UNIT_HOST_1")
        self.assertEqual(restored.splitlines()[0], "rank=1 operation=4 count=4096 expected=11 result=PASS")
        self.assertEqual(count, 1)

    def test_recovery_rejects_missing_ambiguous_or_unsupported_fragments(self):
        prefix = "rank=0 operation=0 count=4 stage=api_begin time_ns="
        trace = "UNIT_HOST_0:39:47 [0] 1.234567 ncclIbTest:1871 NCCL TRACE done\n"
        for suffix in ("", "\n", "rank=0 operation=1 count=4 stage=api_begin time_ns=100\n"):
            with self.subTest(suffix=suffix):
                with self.assertRaises(ValueError):
                    VERIFY.restore_application_records(prefix + trace + suffix, "UNIT_HOST_0")
        unsupported = trace.replace("ncclIbTest:1871", "addCBDCollToPlan:498")
        with self.assertRaises(ValueError):
            VERIFY.restore_application_records(prefix + trace + unsupported + "100\n", "UNIT_HOST_0")
        mixed = prefix + trace + "100\n"
        self.assertEqual(VERIFY.restore_application_records(mixed, "OTHER_HOST"), (mixed, 0))

    def test_interleaving_does_not_hide_missing_or_bad_results(self):
        path = self.result / "rank0/run.log"
        original = path.read_text()
        trace = "UNIT_HOST_0:39:47 [0] 1.234567 ncclIbTest:1871 NCCL TRACE done\n"
        for fragment in ("4 result=PASS", " result=PASS"):
            with self.subTest(fragment=fragment):
                path.write_text(original.replace("expected=3 result=PASS", "expected=" + trace + fragment))
                with self.assertRaises(ValueError):
                    VERIFY.verify(self.result)

    def test_bad_connection_results_plan_and_identity_rejected(self):
        path = self.result / "rank0/run.log"
        original = path.read_text()
        for before, after in (("NET/IB/0", "NET/Socket/0"), ("NET/IB/0", "P2P/CUMEM"),
                              ("nNodes 2", "nNodes 1"), ("device=0", "device=1"),
                              ("RING, SIMPLE", "RING, LL"), ("expected=3 result=PASS", "expected=4 result=PASS"),
                              ("stage=stream_sync_return", "stage=api_return"),
                              ("commId UNIT_GROUP", "commId DIFFERENT"),
                              ("[receive] via NET/IB/0", "[send] via NET/IB/0")):
            with self.subTest(after=after):
                path.write_text(original.replace(before, after))
                with self.assertRaises(ValueError):
                    VERIFY.verify(self.result)
        path.write_text(original)

    def test_node_confirmation_never_overrides_same_host(self):
        path = self.result / "rank1/probe.json"
        record = json.loads(path.read_text())
        record["node_identifiers"]["dmi_product_uuid_sha256"] = "1"*64
        self.json("rank1/probe.json", record)
        with self.assertRaisesRegex(ValueError, "same host"):
            VERIFY.verify(self.result, confirm_distinct_nodes=True)

    def test_missing_hardware_identity_needs_explicit_gui_confirmation(self):
        for rank in (0, 1):
            record = json.loads((self.result / f"rank{rank}/probe.json").read_text())
            record["node_identifiers"] = {}
            self.json(f"rank{rank}/probe.json", record)
        with self.assertRaisesRegex(ValueError, "physical node evidence missing"):
            VERIFY.verify(self.result)
        self.assertEqual(VERIFY.verify(self.result, confirm_distinct_nodes=True)["node_evidence"],
                         "USER_CONFIRMED")

    def test_stale_run_and_wrong_library_rejected(self):
        self.json("rank1/status.json", {"rank": 1, "run_id": "old", "exit_code": 0})
        with self.assertRaisesRegex(ValueError, "stale"):
            VERIFY.verify(self.result)
        self.json("rank1/status.json", {"rank": 1, "run_id": "unit-fixture", "exit_code": 0})
        with self.assertRaisesRegex(ValueError, "library differs"):
            VERIFY.verify(self.result, "0"*64)


class M1PackagingTests(unittest.TestCase):
    def test_real_local_failure_from_missing_resources_or_library_is_bundled(self):
        with tempfile.TemporaryDirectory() as storage:
            folder = Path(storage)
            run = folder / "new-run"
            processes = []
            try:
                for rank in (1, 0):
                    processes.append(subprocess.Popen(
                        [sys.executable, str(ROOT / "cluster/crater/scripts/run_m1.py"),
                         "--rank", str(rank), "--run-dir", str(run),
                         "--nccl-root", str(folder / "absent-library"), "--peer-timeout", "3"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
                for process in processes:
                    _, error = process.communicate(timeout=45)
                    self.assertNotEqual(process.returncode, 0, error)
                archive = folder / "new-run.tar.gz"
                self.assertTrue(archive.is_file())
                with tarfile.open(archive) as bundle:
                    names = bundle.getnames()
                    self.assertIn("m1-results/run-status.txt", names)
                    self.assertIn("m1-results/rank0/status.json", names)
                    self.assertIn("m1-results/rank1/status.json", names)
                    status = bundle.extractfile("m1-results/run-status.txt").read()
                    self.assertIn(b"job_exit_code=1", status)
                    self.assertNotIn("m1-results/nccl.id", names)
                self.assertEqual(len(list(folder.glob("*.tar.gz"))), 1)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()
