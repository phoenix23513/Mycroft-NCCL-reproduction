"""Day 14 guard checks. These do not constitute real NCCL build acceptance."""
import os
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path("cluster/crater/scripts/build_nccl.sh")


class LibraryProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="day14-probe-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.binary = Path(cls.directory.name) / "nccl_version"
        subprocess.run(
            ["g++", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
             str(ROOT / "cluster/crater/probes/nccl_version.cpp"), "-ldl",
             "-o", str(cls.binary)],
            check=True, capture_output=True, text=True,
        )

    def run_probe(self, *arguments):
        return subprocess.run(
            [str(self.binary), *map(str, arguments)], capture_output=True, text=True,
        )

    def test_library_argument_required(self):
        result = self.run_probe()
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)

    def test_nonexistent_library_rejected(self):
        result = self.run_probe(Path(self.directory.name) / "missing.so")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Cannot resolve", result.stderr)

    def test_non_elf_file_rejected(self):
        result = self.run_probe(ROOT / "README.md")
        self.assertEqual(result.returncode, 1)
        self.assertIn("dlopen failed", result.stderr)

    def test_real_library_without_nccl_api_rejected(self):
        libraries = list(Path("/lib").glob("*-linux-gnu/libc.so.6"))
        libraries += list(Path("/usr/lib").glob("*-linux-gnu/libc.so.6"))
        self.assertTrue(libraries, "This Linux test requires the system libc")
        result = self.run_probe(libraries[0])
        self.assertEqual(result.returncode, 1)
        self.assertIn("ncclGetVersion lookup failed", result.stderr)
        self.assertNotIn("status=PASS", result.stdout)


class BuildSourceGuardTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="day14-source-")
        self.addCleanup(directory.cleanup)
        self.project = Path(directory.name)
        (self.project / SCRIPT).parent.mkdir(parents=True)
        shutil.copy2(ROOT / SCRIPT, self.project / SCRIPT)
        self.source = self.project / "third_party/nccl"
        self.source.parent.mkdir()
        # Local clone only: never modify the user's submodule or use the network.
        subprocess.run(
            ["git", "clone", "--shared", "--quiet",
             str(ROOT / "third_party/nccl"), str(self.source)],
            check=True, capture_output=True, text=True,
        )
        self.environment = dict(os.environ, CUDA_HOME=str(self.project / "no-cuda"),
                                CXX="g++")

    def run_check(self, mode="--check"):
        return subprocess.run(
            ["bash", str(self.project / SCRIPT), mode],
            env=self.environment, capture_output=True, text=True,
        )

    def assert_source_rejected(self):
        result = self.run_check()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("[source rejected]", result.stderr)

    def test_clean_source_without_cuda_is_blocked(self):
        result = self.run_check()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("clean=yes", result.stdout)
        self.assertIn("[environment blocked] nvcc not found", result.stderr)
        self.assertFalse((self.project / ".build").exists())

    def test_modified_tracked_source_rejected(self):
        with (self.source / "src/init.cc").open("a") as stream:
            stream.write("\n// unexpected modification\n")
        self.assert_source_rejected()

    def test_untracked_source_file_rejected(self):
        (self.source / "unexpected-file").write_text("unexpected input\n")
        self.assert_source_rejected()

    def test_missing_pinned_tag_rejected(self):
        subprocess.run(
            ["git", "-C", str(self.source), "tag", "-d", "v2.21.5-1"],
            check=True, capture_output=True,
        )
        self.assert_source_rejected()

    def test_unknown_mode_rejected(self):
        result = self.run_check("--unknown")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)


class ImageProbeTests(unittest.TestCase):
    def test_missing_cuda_reported_without_gpu(self):
        with tempfile.TemporaryDirectory(prefix="day14-no-cuda-") as directory:
            result = subprocess.run(
                ["bash", str(ROOT / "cluster/crater/probes/cuda_devel_probe.sh")],
                env=dict(os.environ, CUDA_HOME=directory, CXX="g++"),
                capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("missing=bin/nvcc", result.stdout)
        self.assertIn("gpu_required=no", result.stdout)
        self.assertIn("status=BLOCKED", result.stdout)
        self.assertNotIn("status=READY_FOR_NCCL_BUILD", result.stdout)

    def test_unexpected_argument_rejected(self):
        result = subprocess.run(
            ["bash", str(ROOT / "cluster/crater/probes/cuda_devel_probe.sh"), "--unknown"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)


class UploadPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="day14-upload-test-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.workdir = Path(cls.directory.name)
        cls.archive = cls.workdir / "day14-nccl-build.tar.gz"
        cls.packager = ROOT / "cluster/crater/scripts/package_day14.py"
        subprocess.run(
            ["python3", str(cls.packager), "--output", str(cls.archive)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["tar", "-xzf", str(cls.archive), "-C", str(cls.workdir)],
            check=True, capture_output=True, text=True,
        )
        cls.project = cls.workdir / "day14-nccl-build"

    def test_fixed_source_portable_and_clean(self):
        source = self.project / "third_party/nccl"
        head = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
        ).strip()
        self.assertEqual(head, "ab2b89c4c339bd7f816fbc114a4b05d386b66290")
        tag = subprocess.check_output(
            ["git", "-C", str(source), "describe", "--tags", "--exact-match"], text=True,
        ).strip()
        self.assertEqual(tag, "v2.21.5-1")
        status = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        )
        self.assertEqual(status, "")
        subprocess.run(
            ["git", "-C", str(source), "fsck", "--no-reflogs"],
            check=True, capture_output=True, text=True,
        )

    def test_only_required_data_and_no_local_git_metadata(self):
        with tarfile.open(self.archive) as archive:
            members = archive.getmembers()
        names = [item.name for item in members]
        self.assertFalse(any(name.startswith("day14-nccl-build/.git") for name in names))
        self.assertFalse(any("/.git/logs" in name or "/.git/hooks" in name for name in names))
        self.assertFalse(any("legacy-nccl" in name or "/.build/" in name for name in names))
        self.assertTrue(all(item.uname == item.gname == "" for item in members))
        config = (self.project / "third_party/nccl/.git/config").read_text()
        self.assertNotIn(str(ROOT), config)
        self.assertNotIn("remote", config)
        self.assertFalse((self.project / "third_party/nccl/.git/objects/info/alternates").exists())

    def test_archive_checksum_matches(self):
        digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        checksum = self.archive.with_name(self.archive.name + ".sha256").read_text()
        self.assertEqual(checksum, f"{digest}  {self.archive.name}\n")

    def test_existing_archive_not_overwritten(self):
        original = self.archive.read_bytes()
        result = subprocess.run(
            ["python3", str(self.packager), "--output", str(self.archive)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("already exists", result.stderr)
        self.assertEqual(self.archive.read_bytes(), original)

    def test_build_failure_preserves_exit_code_and_status(self):
        with tempfile.TemporaryDirectory(prefix="day14-result-test-") as directory:
            results = Path(directory) / "results"
            result = subprocess.run(
                ["bash", str(self.project / "cluster/crater/scripts/run_day14_build.sh")],
                env=dict(os.environ, RESULT_DIR=str(results),
                         CUDA_HOME=str(Path(directory) / "no-cuda"), CXX="g++"),
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual((results / "run-status.txt").read_text(), "build_exit_code=2\n")
            self.assertIn("[environment blocked] nvcc", result.stderr)

    def test_existing_results_preserved(self):
        with tempfile.TemporaryDirectory(prefix="day14-existing-result-") as directory:
            marker = Path(directory) / "marker"
            marker.write_text("keep me\n")
            result = subprocess.run(
                ["bash", str(self.project / "cluster/crater/scripts/run_day14_build.sh")],
                env=dict(os.environ, RESULT_DIR=directory),
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("already exists", result.stderr)
            self.assertEqual(marker.read_text(), "keep me\n")
            self.assertFalse((Path(directory) / "run-status.txt").exists())


if __name__ == "__main__":
    unittest.main()
