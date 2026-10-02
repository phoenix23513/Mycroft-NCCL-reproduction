#!/usr/bin/env python3
"""打包固定 NCCL 源码、TRACE 构建入口和双 GPU workload，供离线上传。"""
import argparse
from pathlib import Path
import subprocess

from package_day14 import ROOT, FILES as BUILD_FILES, package
from package_day15 import FILES as WORKLOAD_FILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / ".build/day15-trace-upload/day15-trace.tar.gz")
    args = parser.parse_args()
    files = tuple(dict.fromkeys((*BUILD_FILES, *WORKLOAD_FILES,
                                "cluster/crater/scripts/run_day15_trace.sh",
                                "workloads/minimal_allreduce/verify_results.py")))
    try:
        package(args.output.resolve(), package_root="day15-trace", files=files)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Packaging failed: {error}\n")


if __name__ == "__main__":
    main()
