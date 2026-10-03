#!/usr/bin/env python3
"""复用 M2 构建器，构建供正常/延迟两组共用的 M3 库；不执行 GPU。"""
import argparse
import os
from pathlib import Path
from build_m2 import build
from prepare_m3 import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cuda-home", type=Path, default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")))
    args = parser.parse_args()
    if not 1 <= args.jobs <= 32:
        parser.error("jobs must be between 1 and 32")
    try:
        status = build(args.work_dir.absolute(), args.jobs, args.cuda_home.absolute(),
                       prepare_source=prepare, build_kind="m3_instrumented",
                       extra_symbols=("mycroftM3Configure", "mycroftM3Finish"), package_root="m3-build-results")
    except (OSError, ValueError) as error:
        parser.exit(2, f"m3_build_launch=FAILED reason={error}\n")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
