"""真实源码打包/补丁准备与失败结果传输检查；不伪装 CUDA 编译或 GPU 验收。"""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "cluster/crater/scripts"))
import package_m2
import prepare_m2
import build_m2


class DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="m2-delivery-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.archive = cls.root / "upload.tar.gz"
        package_m2.package(cls.archive)
        # Only files in our locally generated package, no links or special members.
        with tarfile.open(cls.archive) as archive:
            for member in archive:
                if not member.isfile():
                    raise AssertionError("unexpected upload member")
                path = cls.root / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source:
                    path.write_bytes(source.read())
        cls.bundle = cls.root / "m2-experiment"

    def test_portable_source_patch_and_host_inputs_prepare_without_cuda(self):
        destination = self.root / "prepared"
        proof = prepare_m2.prepare(self.bundle, destination)
        self.assertEqual(proof["base_commit"], prepare_m2.COMMIT)
        self.assertEqual(len(proof["patched_files"]), 9)
        self.assertIn("mycroftM2Identity(comm, plan, collOpCount)", (destination / "src/enqueue.cc").read_text())
        self.assertIn("mycroftM2ProxyStopped(proxyState)", (destination / "src/proxy.cc").read_text())
        for original, target in prepare_m2.COPIES.items():
            self.assertEqual((destination / target).read_bytes(), (self.bundle / original).read_bytes())
        self.assertEqual(subprocess.check_output(["git", "-C", str(ROOT / "third_party/nccl"),
                                                 "status", "--porcelain"], text=True), "")
        with self.assertRaises(FileExistsError):
            prepare_m2.prepare(self.bundle, destination)

    def test_tampered_host_input_is_rejected_before_source_creation(self):
        path = self.bundle / (prepare_m2.ADAPTER + "mycroft_m2.h")
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"// unexpected modification\n")
            destination = self.root / "tampered"
            with self.assertRaisesRegex(ValueError, "differs"):
                prepare_m2.prepare(self.bundle, destination)
            self.assertFalse(destination.exists())
        finally:
            path.write_bytes(original)

    def test_unsafe_source_archive_is_rejected(self):
        source = self.bundle / "nccl-source.tar"
        manifest_path = self.bundle / "upload-manifest.json"
        original, manifest = source.read_bytes(), manifest_path.read_bytes()
        try:
            with tarfile.open(source, "w") as archive:
                info = tarfile.TarInfo("../outside")
                info.size = 3
                archive.addfile(info, io.BytesIO(b"bad"))
            data = json.loads(manifest)
            data["files"]["nccl-source.tar"] = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "unsafe"):
                prepare_m2.prepare(self.bundle, self.root / "unsafe")
            self.assertFalse((self.root / "outside").exists())
        finally:
            source.write_bytes(original)
            manifest_path.write_bytes(manifest)

    def test_missing_cuda_exports_one_failure_bundle_without_binaries(self):
        work = self.root / "failed-build"
        self.assertEqual(build_m2.build(work, 2, self.root / "absent-cuda"), 1)
        with tarfile.open(work.with_name(work.name + ".tar.gz")) as archive:
            names = archive.getnames()
            self.assertIn("m2-build-results/build-status.json", names)
            status = json.loads(archive.extractfile("m2-build-results/build-status.json").read())
            self.assertEqual(status["status"], "FAILED")
            self.assertIn("CUDA devel", status["error"])
            self.assertFalse(any(".so" in name for name in names))


if __name__ == "__main__":
    unittest.main()
