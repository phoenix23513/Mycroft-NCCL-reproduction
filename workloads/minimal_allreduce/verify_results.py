#!/usr/bin/env python3
"""按明确规则核对 Day15 双 rank 日志，不调用大模型或 GPU。"""
import argparse
import csv
import io
from pathlib import Path
import re
from statistics import median


COUNTS = (4, 4096, 1048576)
COMMIT = "ab2b89c4c339bd7f816fbc114a4b05d386b66290"
DAY14_SHA256 = "f3fe9df1bb787e0d8f82d60a185c69299ef3806010dcee35565f6b8f6bad7dc4"
STAGES = ("api_begin", "api_return", "stream_sync_return")
SELECTION = re.compile(
    r"NCCL TRACE (tunedColl|CBDColl|collnetColl) enqueue coll "
    r"AllReduce\(ncclSum, ncclFloat32, (\w+), (\w+)\), "
    r"nChannels (\d+), count (\d+) \(nbytes (\d+)\)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def single(pattern, text, description):
    matches = re.findall(pattern, text, re.MULTILINE)
    require(len(matches) == 1, f"{description}: expected exactly one record")
    return matches[0]


def selection_records(log, stages, iterations):
    """只映射 rank0 串行调用区间中的实际计划日志，不构造 rank1 的选择。"""
    total = len(COUNTS) * iterations
    matches = list(SELECTION.finditer(log))
    require(len(matches) == total and log.count("enqueue coll AllReduce(") == total,
            "rank 0: missing, duplicate or unsupported selection record")
    records = []
    for op, match in enumerate(matches):
        begin, returned = stages[3 * op:3 * op + 2]
        require(begin.end() < match.start() < returned.start(),
                f"rank 0 operation {op}: selection outside its API call")
        path, algo, proto, channels, count, nbytes = match.groups()
        require(algo in {"TREE", "RING", "COLLNET_DIRECT", "COLLNET_CHAIN", "NVLS", "NVLS_TREE"}
                and proto in {"LL", "LL128", "SIMPLE"},
                f"rank 0 operation {op}: unknown algorithm or protocol")
        require(int(count) == COUNTS[op // iterations] and int(nbytes) == int(count) * 4
                and int(channels) > 0, f"rank 0 operation {op}: selection size or channels differ")
        records.append({"algorithm": algo, "protocol": proto,
                        "channels": int(channels), "selection_path": path})
    return records


def verify(directory, iterations=3, expected_sha256=DAY14_SHA256, require_selection=False):
    require(3 <= iterations <= 1000, "iterations must be from 3 to 1000")
    total = len(COUNTS) * iterations
    files = {name: (directory / name).read_text() for name in (
        "run-status.txt", "environment.txt", "nccl-build-manifest.txt",
        "run.log", "rank0.log", "rank1.log")}
    require(single(r"^job_exit_code=(\d+)$", files["run-status.txt"], "job status") == "0",
            "job exit code is nonzero")
    exits = re.findall(r"^rank=([01]) exit_code=(\d+)$", files["run.log"], re.MULTILINE)
    require(sorted(exits) == [("0", "0"), ("1", "0")], "both rank exit codes must be zero")

    manifest = files["nccl-build-manifest.txt"]
    require(f"source_commit={COMMIT}" in manifest and "source_tag=v2.21.5-1" in manifest
            and "source_clean=yes" in manifest, "NCCL source provenance differs")
    statuses = re.findall(r"^status=(.*)$", manifest, re.MULTILINE)
    require(statuses and statuses[-1] == "PASS", "NCCL build did not finish successfully")
    environment = files["environment.txt"]
    recorded_hash = single(r"^([0-9a-f]{64})\s+.*libnccl\.so\.2\.21\.5$",
                           environment, "library SHA256")
    require(recorded_hash == expected_sha256, "library SHA256 differs from expected build")
    nccl_root = single(r"^nccl_root=(.+)$", environment, "NCCL root")
    if require_selection:
        require(single(r"^trace=(\d+)$", manifest, "TRACE build flag") == "1"
                and re.search(r"\bTRACE=1\b", manifest), "selection requires a TRACE=1 build")
        require(single(r"^NCCL_DEBUG=(.+)$", environment, "debug level") == "TRACE",
                "selection requires NCCL_DEBUG=TRACE")
        for variable in ("NCCL_ALGO", "NCCL_PROTO"):
            require(single(rf"^{variable}=(.+)$", environment, variable) == "<unset>",
                    f"{variable} must be unset for automatic selection")
    inventory = environment.split("GPU inventory:\n", 1)[1].split("GPU topology:", 1)[0]
    gpu_rows = list(csv.reader(io.StringIO(inventory.strip())))
    require(gpu_rows and len(gpu_rows[0]) == 5, "invalid GPU inventory header")
    gpus = {}
    for row in gpu_rows[1:]:
        if row:
            require(len(row) == 5, "invalid GPU inventory row")
            index, name, uuid, pci, _ = (item.strip() for item in row)
            require(int(index) not in gpus, "duplicate GPU inventory index")
            gpus[int(index)] = {"name": name, "uuid": uuid, "pci": pci}

    rows, connections, init_records = [], set(), []
    for rank in (0, 1):
        log = files[f"rank{rank}.log"]
        requested = single(r"^requested_library=(.+)$", log, f"rank {rank} requested library")
        loaded = single(r"^loaded_library=(.+)$", log, f"rank {rank} loaded library")
        require(requested == loaded == nccl_root.rstrip("/") + "/lib/libnccl.so.2.21.5",
                f"rank {rank}: library path mismatch")
        require(single(r"^nccl_version_code=(\d+)$", log, f"rank {rank} version") == "22105",
                f"rank {rank}: incorrect NCCL version")
        require(f"rank={rank} device={rank} gpu_name=" in log, f"rank {rank}: GPU binding missing")
        init = single(
            rf"ncclCommInitRank comm \S+ rank {rank} nranks (\d+) cudaDev (\d+) "
            r"nvmlDev (\d+) busId ([0-9a-fA-F]+) commId (\S+) - Init COMPLETE",
            log, f"rank {rank} communicator")
        require(init[0] == "2" and int(init[1]) == rank, f"rank {rank}: group or device differs")
        gpu = gpus.get(int(init[2]))
        require(gpu is not None, f"rank {rank}: selected GPU absent from inventory")
        bus = int(gpu["pci"].replace(":", "").replace(".", ""), 16)
        require(bus == int(init[3], 16), f"rank {rank}: PCI identity mismatch")
        require(re.search(rf"rank {rank} nRanks 2 nNodes 1 localRanks 2 localRank {rank}\b", log),
                f"rank {rank}: single-node group evidence missing")
        init_records.append((gpu["uuid"], bus, init[4], gpu["name"]))
        connections.update(re.findall(r"Channel .*? via (\S+)", log))

        stages = list(re.finditer(
            rf"^rank={rank} operation=(\d+) count=(\d+) stage=(\w+) time_ns=(\d+)$",
            log, re.MULTILINE))
        require([(int(m[1]), m[3]) for m in stages] ==
                [(op, stage) for op in range(total) for stage in STAGES],
                f"rank {rank}: missing, duplicate or reordered stage")
        results = list(re.finditer(
            rf"^rank={rank} operation=(\d+) count=(\d+) expected=(\d+) result=PASS$",
            log, re.MULTILINE))
        require([int(m[1]) for m in results] == list(range(total)),
                f"rank {rank}: missing, duplicate or reordered result")
        calls = re.findall(r"AllReduce: opCount ([0-9a-fA-F]+).*? count (\d+) "
                           r"datatype 7 op 0 root 0 comm \S+ \[nranks=2\]", log)
        require([(int(op, 16), int(count)) for op, count in calls] ==
                [(op, COUNTS[op // iterations]) for op in range(total)],
                f"rank {rank}: NCCL call sequence differs from workload")
        require(single(rf"^rank={rank} operations=(\d+) status=PASS$", log,
                       f"rank {rank} summary") == str(total), f"rank {rank}: incorrect total")
        selections = selection_records(log, stages, iterations) if require_selection and rank == 0 else []
        for op, result in enumerate(results):
            count = COUNTS[op // iterations]
            group = stages[3 * op:3 * op + 3]
            require(all(int(m[2]) == count for m in group) and int(result[2]) == count
                    and int(result[3]) == 3 + 2 * op, f"rank {rank} operation {op}: count or result differs")
            begin, returned, synced = (int(m[4]) for m in group)
            require(begin <= returned <= synced, f"rank {rank} operation {op}: timestamps reversed")
            require(result.start() > group[-1].start() and
                    (op == total - 1 or result.start() < stages[3 * (op + 1)].start()),
                    f"rank {rank} operation {op}: result not between synchronization and next operation")
            rows.append({"rank": rank, "operation": op, "count": count,
                         "bytes": count * 4, "expected": 3 + 2 * op,
                         "api_us": (returned - begin) / 1000,
                         "sync_wait_us": (synced - returned) / 1000,
                         "observed_total_us": (synced - begin) / 1000})
            if require_selection:
                rows[-1].update(selections[op] if rank == 0 else
                                {"algorithm": "", "protocol": "", "channels": "", "selection_path": ""})
    require(init_records[0][0] != init_records[1][0] and init_records[0][1] != init_records[1][1],
            "ranks must use two distinct GPUs")
    require(init_records[0][2] == init_records[1][2], "ranks use different communicator IDs")
    require(bool(connections), "transport connection evidence missing")
    return rows, sorted(connections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--expected-sha256", default=DAY14_SHA256)
    parser.add_argument("--require-selection", action="store_true",
                        help="额外检查 TRACE 构建及 rank0 每次 API 调用中的算法/协议选择")
    parser.add_argument("--csv", type=Path, help="保存新的逐操作阶段耗时 CSV，不覆盖旧文件")
    args = parser.parse_args()
    try:
        rows, connections = verify(args.directory, args.iterations, args.expected_sha256,
                                   args.require_selection)
        if args.csv:
            with args.csv.open("x", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    except (OSError, ValueError, IndexError, KeyError) as error:
        parser.exit(1, f"verification=FAILED reason={error}\n")
    print("functional_baseline=PASS")
    print(f"ranks=2 operations_per_rank={len(rows) // 2} result_checks={len(rows)}")
    print("transport_connections=" + ",".join(connections))
    if args.require_selection:
        print("selection_evidence=PASS scope=rank0_submitted_plan")
        print("operation,bytes,algorithm,protocol,channels,selection_path")
        for row in rows:
            if row["rank"] == 0:
                print(f"{row['operation']},{row['bytes']},{row['algorithm']},{row['protocol']},"
                      f"{row['channels']},{row['selection_path']}")
    print("rank,count,operations,median_api_us,median_sync_wait_us")
    for rank in (0, 1):
        for count in COUNTS:
            group = [row for row in rows if row["rank"] == rank and row["count"] == count]
            print(f"{rank},{count},{len(group)},{median(r['api_us'] for r in group):.3f},"
                  f"{median(r['sync_wait_us'] for r in group):.3f}")
    print("timing_scope=CPU_observed_local_stream_stages; not_GPU_kernel_duration")
    print("full_day15_acceptance=PENDING_USER_REVIEW" if args.require_selection else
          "full_day15_acceptance=NOT_EVALUATED (per-operation algorithm/protocol require separate review)")


if __name__ == "__main__":
    main()
