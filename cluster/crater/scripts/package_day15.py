#!/usr/bin/env python3
"""打包 Day15 workload 源码，复用 Crater 上已有的 Day14 NCCL 产物。"""
import argparse
import hashlib
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[3]
FILES = (
    "LICENSE",
    "cluster/crater/scripts/run_day15.sh",
    "workloads/minimal_allreduce/CMakeLists.txt",
    "workloads/minimal_allreduce/build.sh",
    "workloads/minimal_allreduce/run.sh",
    "workloads/minimal_allreduce/README.md",
    "workloads/minimal_allreduce/include/bootstrap.h",
    "workloads/minimal_allreduce/include/device_binding.h",
    "workloads/minimal_allreduce/include/result_check.h",
    "workloads/minimal_allreduce/src/bootstrap.cpp",
    "workloads/minimal_allreduce/src/main.cpp",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / ".build/day15-upload/day15-allreduce.tar.gz")
    output = parser.parse_args().output.resolve()
    if output.exists() or output.with_name(output.name + ".sha256").exists():
        parser.error("输出已存在；请用 --output 选择新文件，避免覆盖旧包")
    output.parent.mkdir(parents=True, exist_ok=True)

    def portable(info):
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        return info

    with tarfile.open(output, "x:gz") as archive:
        for name in FILES:
            archive.add(ROOT / name, f"day15-allreduce/{name}", filter=portable)
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{checksum}  {output.name}\n")
    print(f"archive={output}")
    print(f"archive_bytes={output.stat().st_size}")
    print(f"sha256={checksum}")


if __name__ == "__main__":
    main()
