#!/usr/bin/env python3
"""离线核对 M1 双节点 NET/IB 证据；不会运行 GPU 或修改原始日志。"""
import argparse
import csv
import io
import json
from pathlib import Path
import re

from verify_results import COMMIT, COUNTS, require, selection_records, single


TRACE_SHA256 = "e4437dd0b48e3ab426b17b394043a0d162f1c9040b32bcf34b5b866e6286132a"
CHANNEL = re.compile(r"Channel \d+/\d+ : (\d+)\[\d+\] -> (\d+)\[\d+\] "
                     r"\[(send|receive)\] via (\S+)")
PROXY_TRACE_SUFFIX = (
    r":\d+:\d+ \[\d+\] \d+\.\d+ "
    r"(?:ncclIbTest|sendProxyProgress|recvProxyProgress):\d+ NCCL TRACE [^\n]*$")
APPLICATION_RECORD = re.compile(
    r"rank=[01] operation=-?\d+ count=\d+ "
    r"(?:stage=\w+ time_ns=\d+|expected=\d+ result=PASS)")


def restore_application_records(log, hostname=None):
    """恢复旧 workload 被完整 Proxy TRACE 插入的行，不补造字段或时间。

    仅移动已识别的后台进度 TRACE 到恢复行之后；不移动 API/计划记录。
    非法、缺失或未结束的片段仍失败。原文件不变，返回派生视图及恢复行数。
    """
    if not hostname:
        return log, 0
    # 必须使用探针记录的精确 hostname，否则 stage 与紧接的 hostname 都是
    # 字符串时，通用 hostname 正则可能把 api_begin 等字段一起吞掉。
    proxy_trace = re.compile(re.escape(hostname) + PROXY_TRACE_SUFFIX)
    output, pending, traces = [], None, []
    recovered = 0
    for line in log.splitlines():
        trace = proxy_trace.search(line)
        if pending is None:
            if trace and trace.start() > 0 and line.startswith("rank="):
                pending = line[:trace.start()]
                traces = [trace[0]]
            else:
                output.append(line)
        else:
            if trace:
                pending += line[:trace.start()]
                traces.append(trace[0])
            else:
                pending += line
                require(APPLICATION_RECORD.fullmatch(pending),
                        "interleaved application record cannot be recovered without ambiguity")
                output.extend([pending, *traces])
                pending, traces = None, []
                recovered += 1
    require(pending is None, "unfinished interleaved application record")
    return "\n".join(output) + "\n", recovered


def verify(directory, expected_sha256=TRACE_SHA256, confirm_distinct_nodes=False, *, build_validator=None):
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha256), "invalid expected library SHA256")
    config = json.loads((directory / "run-config.json").read_text())
    require(config["expected_sha256"] == expected_sha256, "run configuration library differs")
    require(config["iterations"] == 3, "M1 requires three operations per message size")
    identities, node_ids, connections, recovered_records = [], [], [], {}
    for rank in (0, 1):
        folder = directory / f"rank{rank}"
        status = json.loads((folder / "status.json").read_text())
        require(status["rank"] == rank and status["exit_code"] == 0, f"rank {rank}: process failed")
        require(status["run_id"] == config["run_id"], f"rank {rank}: stale run identity")
        metadata = json.loads((folder / "environment.json").read_text())
        require(metadata["library_sha256"] == expected_sha256, f"rank {rank}: library hash differs")
        require(metadata["settings"] == config["settings"], f"rank {rank}: run settings differ")
        probe = json.loads((folder / "probe.json").read_text())
        require(probe["rank"] == rank and not probe["blocked_reasons"], f"rank {rank}: resources blocked")
        node_ids.append(probe["node_identifiers"])
        manifest = (folder / "build-manifest.txt").read_text()
        if build_validator is not None:
            build_validator(folder, manifest)
        else:
            require(single(r"^source_commit=(.+)$", manifest, "source commit") == COMMIT and
                single(r"^source_tag=(.+)$", manifest, "source tag") == "v2.21.5-1" and
                single(r"^source_clean=(.+)$", manifest, "source state") == "yes" and
                single(r"^trace=(.+)$", manifest, "TRACE flag") == "1" and
                re.findall(r"^status=(.+)$", manifest, re.MULTILINE)[-1:] == ["PASS"],
                    f"rank {rank}: fixed clean TRACE build evidence missing")
        log, recovered_records[rank] = restore_application_records(
            (folder / "run.log").read_text(), probe.get("hostname"))
        requested = single(r"^requested_library=(.+)$", log, "requested library")
        require(requested == single(r"^loaded_library=(.+)$", log, "loaded library") ==
                metadata["library_path"], f"rank {rank}: loaded library differs")
        require(single(r"^nccl_version_code=(\d+)$", log, "version") == "22105", "wrong NCCL version")
        require(re.search(rf"^rank={rank} device=0 gpu_name=", log, re.MULTILINE), "GPU binding differs")
        init = single(rf"ncclCommInitRank comm \S+ rank {rank} nranks (\d+) cudaDev (\d+) "
                      r"nvmlDev (\d+) busId ([0-9a-fA-F]+) commId (\S+) - Init COMPLETE", log, "init")
        require(init[:2] == ("2", "0"), "group or CUDA device differs")
        require(re.search(rf"rank {rank} nRanks 2 nNodes 2 localRanks 1 localRank 0\b", log),
                "NCCL does not report two nodes with one local rank each")
        inventory = list(csv.DictReader(io.StringIO(probe["gpu_inventory"]["stdout"])))
        gpus = [{key.strip(): value.strip() for key, value in row.items()} for row in inventory]
        gpu = [row for row in gpus if row["index"] == init[2]]
        require(len(gpu) == 1, "selected physical GPU absent from inventory")
        gpu = gpu[0]
        require(int(gpu["pci.bus_id"].replace(":", "").replace(".", ""), 16) == int(init[3], 16),
                "GPU PCI mapping differs")
        identities.append((gpu["uuid"], init[4]))
        require(re.search(r"Using network IB\b", log), "IB network initialization evidence missing")
        transport = re.findall(r"Channel .*? via (\S+)", log)
        require(transport and all(value.startswith("NET/IB/") for value in transport),
                "missing NET/IB connection or transport fallback detected")
        observed = CHANNEL.findall(log)
        require(observed and {item[2] for item in observed} == {"send", "receive"},
                "both send and receive connection evidence required")
        for source, destination, direction, net in observed:
            require((int(source), int(destination)) ==
                    ((rank, 1-rank) if direction == "send" else (1-rank, rank)),
                    "connection rank or direction differs")
            connections.append(net)
        stages = list(re.finditer(rf"^rank={rank} operation=(\d+) count=(\d+) "
                                 r"stage=(\w+) time_ns=(\d+)$", log, re.MULTILINE))
        require([(int(m[1]), m[3]) for m in stages] ==
                [(op, stage) for op in range(9) for stage in
                 ("api_begin", "api_return", "stream_sync_return")], f"rank {rank}: operation stages differ")
        results = list(re.finditer(rf"^rank={rank} operation=(\d+) count=(\d+) "
                                  r"expected=(\d+) result=PASS$", log, re.MULTILINE))
        require(len(results) == 9, "missing or duplicate results")
        for op, result in enumerate(results):
            group = stages[3*op:3*op+3]
            count = COUNTS[op // 3]
            require((int(result[1]), int(result[2]), int(result[3])) == (op, count, 3+2*op) and
                    all(int(m[2]) == count for m in group), "operation size or result differs")
            times = [int(m[4]) for m in group]
            require(times == sorted(times) and result.start() > group[-1].start() and
                    (op == 8 or result.start() < stages[3*(op+1)].start()), "local ordering differs")
        require(single(rf"^rank={rank} operations=(\d+) status=PASS$", log, "summary") == "9",
                "operation total differs")
        calls = re.findall(r"AllReduce: opCount ([0-9a-fA-F]+).*? count (\d+) "
                           r"datatype 7 op 0 root 0 comm \S+ \[nranks=2\]", log)
        require([(int(op, 16), int(count)) for op, count in calls] ==
                [(op, COUNTS[op//3]) for op in range(9)], "NCCL call sequence differs")
        if rank == 0:
            selections = selection_records(log, stages, 3)
            require(all(item["algorithm"] == "RING" and item["protocol"] == "SIMPLE"
                        and 1 <= item["channels"] <= 2 for item in selections),
                    "actual plan differs from RING/SIMPLE or exceeds channel limit")
    require(identities[0][0] != identities[1][0], "both ranks use the same GPU UUID")
    require(identities[0][1] == identities[1][1], "communicator IDs differ")
    node_evidence = "USER_CONFIRMED" if confirm_distinct_nodes else "MISSING"
    for key in ("dmi_product_uuid_sha256", "kernel_boot_id_sha256"):
        first, second = (item.get(key) for item in node_ids)
        if first and second:
            require(first != second, f"same host identity: {key}; distinct Pod hostnames are insufficient")
            if key == "dmi_product_uuid_sha256":
                node_evidence = "DISTINCT_DMI_IDENTIFIERS"
    require(node_evidence != "MISSING", "physical node evidence missing; confirm two distinct GUI Node placements")
    return {"node_evidence": node_evidence, "connections": sorted(set(connections)),
            "selections": selections, "recovered_application_records": recovered_records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-sha256", default=TRACE_SHA256)
    parser.add_argument("--confirm-distinct-nodes", action="store_true",
                        help="仅在用户已核对 GUI 的两个物理 Node 不同时使用；不忽略相同 host 身份")
    args = parser.parse_args()
    try:
        result = verify(args.directory, args.expected_sha256, args.confirm_distinct_nodes)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        parser.exit(1, f"m1_verification=FAILED reason={error}\n")
    print("m1_technical_checks=PASS ranks=2 operations_per_rank=9 result_checks=18")
    print("node_evidence=" + result["node_evidence"])
    print("transport_connections=" + ",".join(result["connections"]))
    for rank, count in result["recovered_application_records"].items():
        print(f"rank={rank} recovered_application_records={count} scope=complete_proxy_trace_insertions_only")
    print("selection_evidence=PASS scope=rank0_submitted_plan algorithm=RING protocol=SIMPLE")
    print("operation,bytes,channels")
    for op, item in enumerate(result["selections"]):
        print(f"{op},{COUNTS[op//3]*4},{item['channels']}")
    print("full_m1_acceptance=PENDING_USER_REVIEW")


if __name__ == "__main__":
    main()
