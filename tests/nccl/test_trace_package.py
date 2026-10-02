"""TRACE 实验包和环境错误路径；不伪造成功的 NCCL/CUDA 编译。"""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TracePackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="day15-trace-test-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.folder = Path(cls.directory.name)
        cls.archive = cls.folder / "trace.tar.gz"
        subprocess.run(["python3", str(ROOT / "cluster/crater/scripts/package_day15_trace.py"),
                        "--output", str(cls.archive)], check=True, capture_output=True, text=True)
        subprocess.run(["tar", "-xzf", str(cls.archive), "-C", str(cls.folder)],
                       check=True, capture_output=True, text=True)
        cls.project = cls.folder / "day15-trace"

    def run_experiment(self, result, **environment):
        return subprocess.run(
            ["bash", str(self.project / "cluster/crater/scripts/run_day15_trace.sh")],
            env=dict(os.environ, NCCL_ALGO="", NCCL_PROTO="", RESULT_DIR=str(result), **environment),
            capture_output=True, text=True)

    def test_package_contains_clean_pinned_source_and_required_tools(self):
        source = self.project / "third_party/nccl"
        self.assertEqual(subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
            "ab2b89c4c339bd7f816fbc114a4b05d386b66290")
        self.assertEqual(subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True), "")
        for name in ("workloads/minimal_allreduce/src/main.cpp",
                     "workloads/minimal_allreduce/verify_results.py",
                     "cluster/crater/scripts/build_nccl.sh"):
            self.assertTrue((self.project / name).is_file())
        self.assertNotIn(str(ROOT), (source / ".git/config").read_text())
        self.assertEqual(self.archive.with_name(self.archive.name + ".sha256").read_text(),
                         f"{hashlib.sha256(self.archive.read_bytes()).hexdigest()}  {self.archive.name}\n")

    def test_missing_cuda_preserves_build_failure_and_skips_workload(self):
        with tempfile.TemporaryDirectory(prefix="day15-trace-results-") as folder:
            result = Path(folder) / "results"
            process = self.run_experiment(result, CUDA_HOME=str(Path(folder) / "missing-cuda"))
            self.assertEqual(process.returncode, 2, process.stderr)
            self.assertIn("nvcc not found", process.stderr)
            self.assertEqual((result / "experiment-status.txt").read_text(), "experiment_exit_code=2\n")
            self.assertEqual((result / "nccl-trace/run-status.txt").read_text(), "build_exit_code=2\n")
            self.assertFalse((result / "allreduce").exists())

    def test_existing_result_is_untouched(self):
        result = self.folder / "existing-result"
        result.mkdir()
        marker = result / "marker"
        marker.write_text("keep\n")
        process = self.run_experiment(result)
        self.assertEqual(process.returncode, 2)
        self.assertIn("already exists", process.stderr)
        self.assertEqual(marker.read_text(), "keep\n")
        self.assertFalse((result / "experiment-status.txt").exists())

    def test_forced_algorithm_is_rejected_before_build(self):
        result = self.folder / "forced-result"
        environment = dict(os.environ, NCCL_ALGO="RING", NCCL_PROTO="", RESULT_DIR=str(result))
        process = subprocess.run(
            ["bash", str(self.project / "cluster/crater/scripts/run_day15_trace.sh")],
            env=environment, capture_output=True, text=True)
        self.assertEqual(process.returncode, 2)
        self.assertIn("Unset NCCL_ALGO", process.stderr)
        self.assertFalse(result.exists())

    def test_invalid_trace_flag_rejected_before_environment_check(self):
        process = subprocess.run(
            ["bash", str(self.project / "cluster/crater/scripts/build_nccl.sh"), "--check"],
            env=dict(os.environ, NCCL_TRACE="yes"), capture_output=True, text=True)
        self.assertEqual(process.returncode, 2)
        self.assertIn("NCCL_TRACE must be 0 or 1", process.stderr)


if __name__ == "__main__":
    unittest.main()
