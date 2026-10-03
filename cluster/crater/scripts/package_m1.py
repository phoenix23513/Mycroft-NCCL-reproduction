#!/usr/bin/env python3
"""打包 M1 运行入口与原生 workload；复用已验证的 NCCL TRACE 库。"""
import argparse
import hashlib
from pathlib import Path
import tarfile

from package_day16_results import portable


ROOT = Path(__file__).resolve().parents[3]
FILES = (
    "LICENSE", "workloads/minimal_allreduce/build.sh",
    "workloads/minimal_allreduce/include/bootstrap.h",
    "workloads/minimal_allreduce/include/device_binding.h",
    "workloads/minimal_allreduce/include/result_check.h",
    "workloads/minimal_allreduce/src/main.cpp", "workloads/minimal_allreduce/src/bootstrap.cpp",
    "cluster/crater/scripts/run_m1.py", "cluster/crater/scripts/package_day16_results.py",
    "cluster/crater/probes/rdma_probe.py", "workloads/minimal_allreduce/verify_results.py",
    "workloads/minimal_allreduce/verify_m1.py", "workloads/minimal_allreduce/M1_README.md")


def package(output):
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise ValueError("上传包已存在；请选择新文件名")
    with tarfile.open(output, "x:gz") as archive:
        for name in FILES:
            archive.add(ROOT / name, f"m1-rdma/{name}", recursive=False, filter=portable)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"archive={output}\narchive_bytes={output.stat().st_size}\narchive_sha256={digest}")
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".build/m1-upload/m1-rdma.tar.gz")
    args = parser.parse_args()
    try:
        package(args.output)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"m1_package=FAILED reason={error}\n")


if __name__ == "__main__":
    main()
