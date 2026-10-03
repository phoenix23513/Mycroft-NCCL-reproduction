#!/usr/bin/env python3
"""复用 M1 双 rank 运行机制，启用 M2 插桩并收集一个结果包。"""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import sys

from prepare_m2 import ROOT, ADAPTER, COPIES, check_bundle, sha256
from run_m1 import SETTINGS, RESULT_FILES as M1_FILES, execute, run, wait_json, write_json
from package_day16_results import package_results

RESULT_FILES = (*M1_FILES, *(f"rank{rank}/{name}" for rank in (0, 1) for name in (
    "m2-build.json", "m2-source-manifest.json", "capture/capture-manifest.json", "capture/adapter-manifest.json",
    f"capture/trace/rank{rank}.jsonl", f"capture/trace/send-rank{rank}.jsonl", f"capture/trace/recv-rank{rank}.jsonl",
    "capture/trace/channel-map.jsonl", "capture/trace/plan-map.jsonl")))


def collect(directory, config, timeout):
    codes = []
    for rank in (0, 1):
        try:
            status = wait_json(directory / f"rank{rank}/status.json", timeout)
            codes.append(status["exit_code"] if status["run_id"] == config["run_id"] else 1)
        except (OSError, ValueError, KeyError, TimeoutError):
            codes.append(1)
    result = 1
    if codes == [0, 0]:
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        result = run([sys.executable, str(ROOT / "instrumentation/nccl-2.21.5/day16/verify_capture.py"),
                      str(directory)], directory / "analysis.txt", 60, environment)
    else:
        (directory / "analysis.txt").write_text("m2_verification=NOT_RUN reason=rank_failure_or_missing\n")
    (directory / "run-status.txt").write_text(
        f"job_exit_code={result}\nrank0_exit_code={codes[0]}\nrank1_exit_code={codes[1]}\n")
    output = directory.with_name(directory.name + ".tar.gz")
    digest = package_results(directory, output, files=RESULT_FILES, package_root="m2-results")
    print(f"result_bundle={output}\narchive_sha256={digest}\nm2_job_exit_code={result}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--nccl-root", type=Path, required=True, help="build_m2.py 输出的 nccl 目录")
    parser.add_argument("--peer-timeout", type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.peer_timeout <= 900:
        parser.error("peer timeout must be between 1 and 900 seconds")
    os.umask(0o077)
    def interrupt(signum, frame):
        raise InterruptedError(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    directory, product = args.run_dir.absolute(), args.nccl_root.absolute()
    try:
        upload = check_bundle(ROOT)
        inputs = {name: upload["files"][name] for name in [ADAPTER + "nccl-2.21.5-m2.patch", *COPIES]}
        provenance = {"base_commit": upload["base_commit"], "base_tree": upload["base_tree"],
                      "source_archive_sha256": upload["files"]["nccl-source.tar"], "inputs": inputs}
        build = json.loads((product / "m2-build.json").read_text())
        if build["status"] != "PASS" or build["build_kind"] != "m2_instrumented" or any(
                build[key] != value for key, value in provenance.items()):
            raise ValueError("M2 build does not match this upload package")
        digest = sha256(product / "lib/libnccl.so.2.21.5")
        if digest != build["library_sha256"]:
            raise ValueError("M2 library differs from build result")
        if args.rank == 0:
            directory.parent.mkdir(parents=True, exist_ok=True)
            output = directory.with_name(directory.name + ".tar.gz")
            if output.exists() or output.is_symlink():
                raise ValueError("result archive already exists; choose a new run directory")
            directory.mkdir()
            config = {"capture_mode": "m2", "run_id": secrets.token_hex(16), "expected_sha256": digest,
                      "iterations": 3, "settings": SETTINGS, "provenance": provenance}
            write_json(directory / "run-config.json", config)
        else:
            config = wait_json(directory / "run-config.json", args.peer_timeout)
            if config["capture_mode"] != "m2" or config["expected_sha256"] != digest or config[
                    "settings"] != SETTINGS or config["provenance"] != provenance:
                raise ValueError("Master/Worker M2 configuration differs")
        status, interrupted = execute(args.rank, directory, product, config, args.peer_timeout, capture_m2=True)
        if args.rank == 0:
            status = collect(directory, config, 5 if interrupted else args.peer_timeout)
        else:
            print(f"rank=1 exit_code={status}; download only the rank0 result bundle", flush=True)
    except (OSError, ValueError, KeyError, TimeoutError) as error:
        parser.exit(2, f"m2_launch=FAILED reason={error}\n")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
