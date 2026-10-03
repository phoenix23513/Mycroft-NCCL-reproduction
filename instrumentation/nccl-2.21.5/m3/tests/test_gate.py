"""Compile and run the real allocation-free gate with a supplied clock; no CUDA."""
from pathlib import Path
import subprocess
import tempfile
import unittest

M3=Path(__file__).resolve().parents[1]


class GateTests(unittest.TestCase):
    def test_deadline_scope_single_hit_and_invalid_clock(self):
        with tempfile.TemporaryDirectory(prefix="m3-gate-") as temp:
            binary=Path(temp)/"gate"
            subprocess.run(["g++","-std=c++17","-Wall","-Wextra","-Wpedantic","-Werror",
                            "-I",str(M3/"include"),str(M3/"tests/gate_check.cpp"),"-o",str(binary)],check=True)
            subprocess.run([str(binary)],check=True)


if __name__=="__main__":unittest.main()
