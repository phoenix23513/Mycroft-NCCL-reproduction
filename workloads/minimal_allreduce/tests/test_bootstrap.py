"""验证真实文件 IPC；不调用 CUDA/NCCL，不作为 Day15 GPU 验收。"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


WORKLOAD = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include "bootstrap.h"
#include <array>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  try {
    std::array<unsigned char, 128> bytes{};
    if (std::string(argv[1]) == "publish") {
      for (unsigned i = 0; i < bytes.size(); ++i) bytes[i] = i;
      day15::publish_id(argv[2], bytes.data(), bytes.size());
    } else {
      day15::read_id(argv[2], bytes.data(), bytes.size(), std::chrono::seconds(1));
      for (unsigned i = 0; i < bytes.size(); ++i)
        if (bytes[i] != i) throw std::runtime_error("bytes differ");
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
'''


@unittest.skipUnless(shutil.which("g++"), "bootstrap test requires g++")
class BootstrapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="day15-bootstrap-test-")
        directory = Path(cls.build.name)
        source = directory / "harness.cpp"
        source.write_text(HARNESS)
        cls.binary = directory / "bootstrap_test"
        subprocess.run([
            "g++", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-pthread",
            "-I", str(WORKLOAD / "include"), str(source),
            str(WORKLOAD / "src/bootstrap.cpp"), "-o", str(cls.binary),
        ], check=True, capture_output=True, text=True)
        cls.addClassCleanup(cls.build.cleanup)

    def setUp(self):
        self.storage = tempfile.TemporaryDirectory(prefix="day15-id-test-")
        self.addCleanup(self.storage.cleanup)
        self.path = Path(self.storage.name) / "id"

    def call(self, mode):
        return subprocess.run([str(self.binary), mode, str(self.path)],
                              capture_output=True, text=True, timeout=5)

    def test_reader_and_publisher_are_separate_processes(self):
        reader = subprocess.Popen([str(self.binary), "read", str(self.path)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True)
        try:
            writer = self.call("publish")
            self.assertEqual(writer.returncode, 0, writer.stderr)
            _, error = reader.communicate(timeout=5)
            self.assertEqual(reader.returncode, 0, error)
            self.assertEqual(self.path.read_bytes(), bytes(range(128)))
        finally:
            if reader.poll() is None:
                reader.kill()
                reader.communicate()

    def test_existing_id_is_not_overwritten(self):
        original = b"x" * 128
        self.path.write_bytes(original)
        result = self.call("publish")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.path.read_bytes(), original)

    def test_absent_id_has_bounded_wait(self):
        result = self.call("read")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bootstrap timeout", result.stderr)

    def test_truncated_or_oversized_id_is_rejected(self):
        for size in (127, 129):
            with self.subTest(size=size):
                self.path.write_bytes(b"x" * size)
                result = self.call("read")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("expected size", result.stderr)

    def test_symlink_is_rejected(self):
        target = self.path.parent / "other"
        target.write_bytes(bytes(range(128)))
        self.path.symlink_to(target)
        result = self.call("read")
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
