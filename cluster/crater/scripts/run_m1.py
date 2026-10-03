#!/usr/bin/env python3
"""M1 每 Pod 启动一个原生 rank；rank0 收集两侧证据并自动打包。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from package_day16_results import package_results


ROOT = Path(__file__).resolve().parents[3]
TRACE_SHA256 = "e4437dd0b48e3ab426b17b394043a0d162f1c9040b32bcf34b5b866e6286132a"
SETTINGS = {"NCCL_NET": "IB", "NCCL_NET_PLUGIN": "none", "NCCL_IB_DISABLE": "0",
            "NCCL_ALGO": "RING", "NCCL_PROTO": "SIMPLE", "NCCL_DEBUG": "TRACE",
            "NCCL_DEBUG_SUBSYS": "INIT,ENV,GRAPH,COLL,NET",
            "NCCL_MIN_NCHANNELS": "2", "NCCL_MAX_NCHANNELS": "2"}
RESULT_FILES = ("run-status.txt", "run-config.json", "analysis.txt", *(
    f"rank{rank}/{name}" for rank in (0, 1) for name in
    ("status.json", "probe.json", "environment.json", "build-manifest.txt", "build-tail.log", "run.log")))


def write_json(path, value):
    staging = path.with_name(path.name + ".pending")
    with staging.open("x") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
    staging.replace(path)


def wait_json(path, timeout, failed_status=None):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            if failed_status and failed_status.is_file():
                failure = json.loads(failed_status.read_text())
                if failure.get("exit_code", 1) != 0:
                    raise RuntimeError("peer failed before rendezvous; inspect peer status.json")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"peer did not publish {path.name}; check shared mount and both Roles")
            time.sleep(0.2)


def run(arguments, log, timeout, environment=None):
    with log.open("w") as output:
        child = subprocess.Popen(arguments, stdout=output, stderr=subprocess.STDOUT,
                                 env=environment, start_new_session=True)
        try:
            return child.wait(timeout=timeout)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def execute(rank, directory, nccl_root, config, timeout, *, capture_m2=False):
    folder = directory / f"rank{rank}"
    folder.mkdir()  # 拒绝重试覆盖本次 rank 的已有结果。
    status = {"rank": rank, "run_id": config["run_id"], "exit_code": 1}
    interrupted = False
    try:
        environment = dict(os.environ, **SETTINGS)
        environment.pop("NCCL_DEBUG_FILE", None)
        # An inherited environment must not silently enable instrumentation in M1.
        for name in ("MYCROFT_M2_ENABLE", "MYCROFT_M2_CAPTURE_ID", "MYCROFT_M2_OUTPUT"):
            environment.pop(name, None)
        if capture_m2:
            environment.update(MYCROFT_M2_ENABLE="1", MYCROFT_M2_CAPTURE_ID=config["run_id"],
                               MYCROFT_M2_OUTPUT=str(folder / "capture"))
        status["probe_exit_code"] = run(
            [sys.executable, str(ROOT / "cluster/crater/probes/rdma_probe.py"), "--rank", str(rank)],
            folder / "probe.json", 120, environment)
        if status["probe_exit_code"]:
            raise RuntimeError("RDMA resource probe blocked; inspect probe.json")
        library = (nccl_root / "lib/libnccl.so.2.21.5").resolve(strict=True)
        digest = hashlib.sha256(library.read_bytes()).hexdigest()
        if digest != config["expected_sha256"]:
            raise RuntimeError("library differs from the independently verified expected SHA256")
        shutil.copyfile(nccl_root / "build-manifest.txt", folder / "build-manifest.txt")
        if capture_m2:
            for name in ("m2-build.json", "m2-source-manifest.json"):
                shutil.copyfile(nccl_root / name, folder / name)
        write_json(folder / "environment.json", {"rank": rank, "library_path": str(library),
                   "library_sha256": digest, "settings": SETTINGS,
                   "optional_network_settings": {key: environment[key] for key in
                     ("NCCL_IB_HCA", "NCCL_SOCKET_IFNAME", "NCCL_IB_GID_INDEX", "NCCL_NET_GDR_LEVEL")
                     if key in environment}})
        with tempfile.TemporaryDirectory(prefix="m1-workload-") as task:
            build = Path(task) / "build"
            build_environment = dict(environment, NCCL_ROOT=str(nccl_root), BUILDDIR=str(build))
            build_log = Path(task) / "build.log"
            try:
                status["build_exit_code"] = run(
                    ["bash", str(ROOT / "workloads/minimal_allreduce/build.sh")], build_log, 120, build_environment)
            finally:
                if build_log.exists():
                    with build_log.open("rb") as source:
                        source.seek(max(0, build_log.stat().st_size - 128*1024))
                        (folder / "build-tail.log").write_bytes(source.read(128*1024))
            if status["build_exit_code"]:
                raise RuntimeError("native workload compilation failed; inspect build-tail.log")
            write_json(folder / "ready.json", {"run_id": config["run_id"], "rank": rank})
            peer = wait_json(directory / f"rank{1-rank}/ready.json", timeout,
                             directory / f"rank{1-rank}/status.json")
            if peer["run_id"] != config["run_id"]:
                raise RuntimeError("peer belongs to a different run")
            status["workload_exit_code"] = run(
                [str(build / "day15_allreduce"), str(rank), str(directory / "nccl.id"), "3", "--single-gpu-node"],
                folder / "run.log", 240, environment)
            status["exit_code"] = status["workload_exit_code"]
    except Exception as error:
        interrupted = isinstance(error, InterruptedError)
        status["error"] = str(error)
        print(f"rank={rank} status=FAILED reason={error}", flush=True)
    finally:
        write_json(folder / "status.json", status)
    return status["exit_code"], interrupted


def collect(directory, config, timeout):
    codes = []
    for rank in (0, 1):
        try:
            status = wait_json(directory / f"rank{rank}/status.json", timeout)
            codes.append(status["exit_code"] if status["run_id"] == config["run_id"] else 1)
        except (OSError, ValueError, KeyError, TimeoutError):
            codes.append(1)
    status = 1
    if codes == [0, 0]:
        status = run([sys.executable, str(ROOT / "workloads/minimal_allreduce/verify_m1.py"),
                      str(directory), "--expected-sha256", config["expected_sha256"]],
                     directory / "analysis.txt", 30)
    else:
        (directory / "analysis.txt").write_text("m1_verification=NOT_RUN reason=rank_failure_or_missing\n")
    (directory / "run-status.txt").write_text(
        f"job_exit_code={status}\nrank0_exit_code={codes[0]}\nrank1_exit_code={codes[1]}\n")
    output = directory.with_name(directory.name + ".tar.gz")
    checksum = package_results(directory, output, files=RESULT_FILES, package_root="m1-results")
    print(f"result_bundle={output}\narchive_sha256={checksum}\nm1_job_exit_code={status}", flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="两侧相同的新共享运行目录")
    parser.add_argument("--nccl-root", type=Path, required=True, help="已有固定源码 TRACE=1 构建产物")
    parser.add_argument("--expected-sha256", default=TRACE_SHA256)
    parser.add_argument("--peer-timeout", type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.peer_timeout <= 900 or len(args.expected_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in args.expected_sha256):
        parser.error("invalid peer timeout or expected SHA256")
    os.umask(0o077)
    directory = args.run_dir.absolute()
    def interrupt(signum, frame):
        raise InterruptedError(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    try:
        if args.rank == 0:
            directory.parent.mkdir(parents=True, exist_ok=True)
            directory.mkdir()  # 已有目录不能参与新一轮，不能覆盖旧 ID/日志。
            config = {"run_id": secrets.token_hex(16), "expected_sha256": args.expected_sha256,
                      "iterations": 3, "settings": SETTINGS}
            write_json(directory / "run-config.json", config)
        else:
            config = wait_json(directory / "run-config.json", args.peer_timeout)
            if config["expected_sha256"] != args.expected_sha256 or config["settings"] != SETTINGS:
                raise ValueError("Master/Worker configuration differs")
        status, interrupted = execute(args.rank, directory, args.nccl_root.absolute(), config, args.peer_timeout)
        if args.rank == 0:
            status = collect(directory, config, 5 if interrupted else args.peer_timeout)
        else:
            print(f"rank=1 exit_code={status}; download only the rank0 result bundle", flush=True)
    except (OSError, ValueError, KeyError, TimeoutError) as error:
        parser.exit(2, f"m1_launch=FAILED reason={error}\n")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
