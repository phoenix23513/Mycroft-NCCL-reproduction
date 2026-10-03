#!/usr/bin/env python3
"""单包交付固定源码、M2+M3 补丁及三组运行/核对入口。"""
import argparse
from pathlib import Path
import subprocess
import tarfile
from package_m2 import package as package_m2
from prepare_m2 import ROOT
from prepare_m3 import M3, PATCH, COPIES

FILES = (PATCH, *COPIES, M3 + "verify_experiment.py", M3 + "README.md",
         "cluster/crater/scripts/prepare_m3.py", "cluster/crater/scripts/build_m3.py",
         "cluster/crater/scripts/run_m3.py", "cluster/crater/scripts/package_m3.py")


def package(output):
    return package_m2(output, extra_files=FILES, package_root="m3-experiment")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".build/m3-upload/m3-experiment.tar.gz")
    args = parser.parse_args()
    try:
        digest = package(args.output.absolute())
    except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError) as error:
        parser.exit(1, f"m3_package=FAILED reason={error}\n")
    print(f"archive={args.output}\narchive_bytes={args.output.stat().st_size}\narchive_sha256={digest}")


if __name__ == "__main__":
    main()
