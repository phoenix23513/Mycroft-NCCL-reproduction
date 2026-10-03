"""编译实际设备选择函数；不调用 CUDA，不代表 M1 GPU 运行通过。"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


WORKLOAD = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("g++"), "requires g++")
class DeviceBindingTests(unittest.TestCase):
    def test_legacy_and_single_gpu_node_bindings_and_rejections(self):
        with tempfile.TemporaryDirectory() as storage:
            source = Path(storage) / "binding.cpp"
            source.write_text('''
#include "device_binding.h"
int main() {
  if (day15::select_device(0, 2, false) != 0 || day15::select_device(1, 2, false) != 1 ||
      day15::select_device(0, 1, true) != 0 || day15::select_device(1, 1, true) != 0) return 1;
  for (int mode = 0; mode < 4; ++mode) {
    try {
      if (mode == 0) day15::select_device(1, 1, false);
      if (mode == 1) day15::select_device(1, 0, true);
      if (mode == 2) day15::select_device(1, 2, true);
      if (mode == 3) day15::select_device(2, 1, true);
      return 2;
    } catch (const std::exception&) {}
  }
}
''')
            binary = Path(storage) / "binding"
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
                            "-I", str(WORKLOAD / "include"), str(source), "-o", str(binary)],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(binary)], check=True, timeout=5)
