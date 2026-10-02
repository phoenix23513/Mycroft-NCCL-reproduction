"""验证无 CMake 的构建参数和失败路径；拒绝编译器始终失败，不伪造成功 NCCL 构建。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


WORKLOAD = Path(__file__).resolve().parents[1]


class BuildGuardsTest(unittest.TestCase):
    def setUp(self):
        self.storage = tempfile.TemporaryDirectory(prefix="day15-build-guard-")
        self.addCleanup(self.storage.cleanup)
        self.root = Path(self.storage.name)
        self.cuda = self.root / "cuda sdk"
        self.nccl = self.root / "nccl root"
        # 空文件只用于走到编译失败路径；不生成、加载或运行任何 CUDA/NCCL 库。
        for directory in (self.cuda, self.nccl):
            (directory / "include").mkdir(parents=True)
            (directory / "lib64" if directory == self.cuda else directory / "lib").mkdir()
        (self.cuda / "include/cuda_runtime.h").touch()
        (self.cuda / "lib64/libcudart.so").touch()
        (self.cuda / "lib64/libcudart_static.a").touch()
        (self.nccl / "include/nccl.h").touch()
        self.library = self.nccl / "lib/libnccl.so.2.21.5"
        self.library.touch()
        (self.nccl / "lib/libnccl.so.2").symlink_to(self.library.name)
        self.arguments = self.root / "compiler-arguments.json"
        self.compiler = self.root / "reject-compiler"
        self.compiler.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['DAY15_TEST_ARGUMENTS'], 'w') as output:\n"
            "    json.dump(sys.argv[1:], output)\n"
            "sys.exit(47)\n"
        )
        self.compiler.chmod(0o755)
        tools = self.root / "tools"
        tools.mkdir()
        for name in ("realpath", "dirname", "mkdir"):
            (tools / name).symlink_to(shutil.which(name))
        self.build = self.root / "build"
        self.env = dict(os.environ, PATH=str(tools), CXX=str(self.compiler),
                        CUDA_HOME=str(self.cuda), NCCL_ROOT=str(self.nccl),
                        BUILDDIR=str(self.build), DAY15_TEST_ARGUMENTS=str(self.arguments))

    def call(self):
        return subprocess.run(["/bin/bash", str(WORKLOAD / "build.sh")], env=self.env,
                              capture_output=True, text=True, timeout=5)

    def test_no_cmake_shared_runtime_arguments_and_compiler_failure(self):
        result = self.call()
        self.assertEqual(result.returncode, 47, result.stderr)
        args = json.loads(self.arguments.read_text())
        self.assertIn(str(self.library), args)
        self.assertIn(str(self.cuda / "lib64/libcudart.so"), args)
        self.assertNotIn(str(self.cuda / "lib64/libcudart_static.a"), args)
        self.assertIn(f'-DDAY15_EXPECTED_NCCL_LIBRARY="{self.library}"', args)
        self.assertIn(f"-Wl,-rpath,{self.nccl}/lib", args)
        self.assertIn(f"-Wl,-rpath,{self.cuda}/lib64", args)
        self.assertIn(str(self.build / "day15_allreduce"), args)
        self.assertFalse((self.build / "day15_allreduce").exists())

    def test_static_runtime_arguments_and_compiler_failure(self):
        (self.cuda / "lib64/libcudart.so").unlink()
        result = self.call()
        self.assertEqual(result.returncode, 47, result.stderr)
        args = json.loads(self.arguments.read_text())
        self.assertIn(str(self.cuda / "lib64/libcudart_static.a"), args)
        for flag in ("-pthread", "-ldl", "-lrt"):
            self.assertIn(flag, args)

    def test_missing_cuda_header_blocks_before_build_directory_creation(self):
        self.env["CUDA_HOME"] = str(self.root / "missing-sdk")
        result = self.call()
        self.assertEqual(result.returncode, 2)
        self.assertIn("Missing CUDA header", result.stderr)
        self.assertFalse(self.build.exists())
        self.assertFalse(self.arguments.exists())

    def test_existing_build_directory_is_preserved(self):
        self.build.mkdir()
        marker = self.build / "keep.txt"
        marker.write_text("keep")
        result = self.call()
        self.assertEqual(result.returncode, 2)
        self.assertIn("Build directory already exists", result.stderr)
        self.assertEqual(marker.read_text(), "keep")
        self.assertFalse(self.arguments.exists())


if __name__ == "__main__":
    unittest.main()
