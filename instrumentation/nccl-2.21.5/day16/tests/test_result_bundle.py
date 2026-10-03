"""结果传输工具的单元测试；输入是文件 fixture，不是 NCCL 验收轨迹。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "day16_result_bundle", ROOT / "cluster/crater/scripts/package_day16_results.py")
BUNDLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUNDLE)


class ResultBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="day16-bundle-test-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.results = self.folder / "results"
        self.results.mkdir()
        self.output = self.folder / "download.tar.gz"

    def write(self, name, contents):
        path = self.results / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        return path

    def read_members(self):
        with tarfile.open(self.output) as archive:
            return {item.name: archive.extractfile(item).read() for item in archive.getmembers()}

    def test_single_archive_preserves_paths_and_excludes_binary_products(self):
        self.write("run-status.txt", "job_exit_code=0\n")
        self.write("workload/rank0.log", "packaging unit fixture\n")
        self.write("trace/rank0.jsonl", '{"fixture":"packaging_only"}\n')
        for name in ("send-rank0.jsonl", "send-rank1.jsonl", "recv-rank0.jsonl", "recv-rank1.jsonl",
                     "channel-map.jsonl", "plan-map.jsonl"):
            self.write(f"trace/{name}", '{"fixture":"packaging_only"}\n')
        self.write("lib/libnccl.so", "excluded unit fixture\n")
        self.write("include/nccl.h", "excluded unit fixture\n")
        digest = BUNDLE.package_results(self.results, self.output)
        members = self.read_members()
        self.assertEqual(members["day16-results/workload/rank0.log"], b"packaging unit fixture\n")
        self.assertIn("day16-results/trace/rank0.jsonl", members)
        for name in ("send-rank0.jsonl", "send-rank1.jsonl", "recv-rank0.jsonl", "recv-rank1.jsonl",
                     "channel-map.jsonl", "plan-map.jsonl"):
            self.assertIn(f"day16-results/trace/{name}", members)
        self.assertFalse(any("/lib/" in name or "/include/" in name for name in members))
        self.assertEqual(digest, hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(list(self.folder.glob("*.tar.gz")), [self.output])
        self.assertEqual(list(self.folder.glob(".day16-bundle-*.partial")), [])

    def test_failed_job_preserves_partial_evidence_and_bounded_build_log_tail(self):
        self.write("run-status.txt", "job_exit_code=7\n")
        data = b"prefix\n" + b"x" * BUNDLE.BUILD_LOG_TAIL_BYTES + b"\ncompiler failed\n"
        log = self.results / "nccl-build/build.log"
        log.parent.mkdir()
        log.write_bytes(data)
        BUNDLE.package_results(self.results, self.output)
        members = self.read_members()
        self.assertEqual(members["day16-results/run-status.txt"], b"job_exit_code=7\n")
        self.assertEqual(members["day16-results/nccl-build/build-tail.log"],
                         data[-BUNDLE.BUILD_LOG_TAIL_BYTES:])
        manifest = json.loads(members["day16-results/bundle-manifest.json"])
        self.assertIn("trace/rank0.jsonl", manifest["missing_files"])
        self.assertNotIn("day16-results/nccl-build/build.log", members)

    def test_existing_output_is_not_overwritten(self):
        self.write("run-status.txt", "job_exit_code=2\n")
        self.output.write_bytes(b"keep existing download\n")
        with self.assertRaisesRegex(ValueError, "已存在"):
            BUNDLE.package_results(self.results, self.output)
        self.assertEqual(self.output.read_bytes(), b"keep existing download\n")

    def test_symlinked_subdirectory_is_not_followed(self):
        self.write("run-status.txt", "job_exit_code=2\n")
        external = self.folder / "external"
        external.mkdir()
        (external / "rank0.log").write_text("outside results\n")
        (self.results / "workload").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "符号链接"):
            BUNDLE.package_results(self.results, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_job_status_prevents_publication(self):
        self.write("environment.txt", "packaging unit fixture\n")
        with self.assertRaisesRegex(ValueError, "缺少 run-status"):
            BUNDLE.package_results(self.results, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
