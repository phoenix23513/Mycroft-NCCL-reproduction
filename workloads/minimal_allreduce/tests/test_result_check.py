"""验证生产代码的 CPU 数值检查；不伪造 CUDA/NCCL，也不替代 GPU 验收。"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


WORKLOAD = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include "result_check.h"
#include <iostream>
#include <limits>
#include <string>
#include <vector>
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string mode(argv[1]);
  std::vector<float> output(4, 3.0f);
  float expected = 3.0f;
  if (mode == "late_error") output.back() = 4.0f;
  else if (mode == "stale") expected = 5.0f;
  else if (mode == "nan") output[2] = std::numeric_limits<float>::quiet_NaN();
  else if (mode == "infinity") output[1] = std::numeric_limits<float>::infinity();
  else if (mode != "valid") return 2;
  try {
    day15::verify_result(output, expected);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
'''


@unittest.skipUnless(shutil.which("g++"), "result check test requires g++")
class ResultCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="day15-result-test-")
        cls.addClassCleanup(cls.build.cleanup)
        directory = Path(cls.build.name)
        source = directory / "harness.cpp"
        source.write_text(HARNESS)
        cls.binary = directory / "result_test"
        subprocess.run([
            "g++", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic",
            "-I", str(WORKLOAD / "include"), str(source), "-o", str(cls.binary),
        ], check=True, capture_output=True, text=True)

    def call(self, mode):
        return subprocess.run([str(self.binary), mode], capture_output=True,
                              text=True, timeout=5)

    def test_correct_output_passes(self):
        result = self.call("valid")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_error_beyond_first_element_reports_index_and_values(self):
        result = self.call("late_error")
        self.assertEqual(result.returncode, 1)
        self.assertIn("index=3 actual=4 expected=3", result.stderr)

    def test_previous_operation_output_is_rejected(self):
        result = self.call("stale")
        self.assertEqual(result.returncode, 1)
        self.assertIn("actual=3 expected=5", result.stderr)

    def test_nan_from_unwritten_output_is_rejected(self):
        result = self.call("nan")
        self.assertEqual(result.returncode, 1)
        self.assertIn("index=2", result.stderr)

    def test_infinity_is_rejected(self):
        result = self.call("infinity")
        self.assertEqual(result.returncode, 1)
        self.assertIn("index=1", result.stderr)


if __name__ == "__main__":
    unittest.main()
