#!/usr/bin/env python3
"""离线核对 M2 来源、真实运行与逐操作采集；不运行 GPU，不修改原始数据。"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "workloads/minimal_allreduce"))
from verify_m1 import verify as verify_run
from verify_results import COMMIT, COUNTS, require, single

from mycroft.schema import Event, EventKind, ProgressSnapshotPayload, TimeDomain, validate_event


def normalize_send_steps(*, nsteps: int, slice_steps: int, gpu_ready_steps: int,
                         transmitted_steps: int, done_steps: int,
                         registered_buffer: bool) -> ProgressSnapshotPayload:
    """发送侧普通 step→slice 契约检查；不证明 readiness 来源或真实运行。

    不截断非整数 slice，不把 step_base 或 posted 当进度；registered 路径的
    nsteps 可以动态增长，其 GPU/网络计数关系须另行确认，当前显式拒绝转换。
    """
    if type(registered_buffer) is not bool:
        raise ValueError("registered_buffer must be a bool")
    if registered_buffer:
        raise ValueError("registered-buffer normalization is not verified")
    values = {"nsteps": nsteps, "slice_steps": slice_steps,
              "gpu_ready_steps": gpu_ready_steps, "transmitted_steps": transmitted_steps,
              "done_steps": done_steps}
    for name, value in values.items():
        if type(value) is not int or not 0 <= value < 2**64:
            raise ValueError(f"{name} must be an unsigned 64-bit integer")
    if nsteps == 0 or slice_steps == 0:
        raise ValueError("nsteps and slice_steps must be positive")
    if any(value % slice_steps for name, value in values.items() if name != "slice_steps"):
        raise ValueError("step counters must be exact multiples of slice_steps")
    if not 0 <= done_steps <= transmitted_steps <= gpu_ready_steps <= nsteps:
        raise ValueError("expected done <= transmitted <= gpu_ready <= nsteps")
    return ProgressSnapshotPayload(
        total_chunks=nsteps // slice_steps,
        gpu_ready=gpu_ready_steps // slice_steps,
        rdma_transmitted=transmitted_steps // slice_steps,
        rdma_done=done_steps // slice_steps,
    )


def validate_progress_line(line: str) -> Event:
    """发送侧 Event v2 格式检查；接收状态须独立导出，不能伪装成此事件。"""
    event = Event.from_json(line)
    validate_event(event)
    if (event.schema_version != 2 or event.event_kind is not EventKind.PROGRESS_SNAPSHOT
            or event.time.domain is not TimeDomain.NCCL_MONOTONIC_NS
            or event.context.collective != "all_reduce"):
        raise ValueError("M2 requires Event v2 all_reduce progress on NCCL monotonic time")
    return event


def validate_completion_line(line: str) -> Event:
    """复用已有 Event v2 契约；格式有效不代表来源或完成语义有效。"""
    event = Event.from_json(line)
    validate_event(event)
    if (event.schema_version != 2 or event.event_kind is not EventKind.OPERATION_COMPLETION
            or event.time.domain is not TimeDomain.NCCL_MONOTONIC_NS
            or event.context.collective != "all_reduce"):
        raise ValueError("M2 requires Event v2 all_reduce completion on NCCL monotonic time")
    return event


def load_json(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unsigned(value, name):
    require(type(value) is int and 0 <= value < 2**64, f"invalid unsigned counter: {name}")
    return value


def key(row):
    return (unsigned(row["comm_hash"], "comm_hash"),
            unsigned(row["op_seq_candidate"], "operation"), unsigned(row["rank"], "rank"))


def connection(row):
    return (*key(row), unsigned(row["channel"], "channel"), unsigned(row["connection_id"], "connection"))


def verify_rank_capture(folder, rank, config, selections):
    capture = folder / "capture"
    manifest, adapter = (load_json(capture / name) for name in ("capture-manifest.json", "adapter-manifest.json"))
    require(manifest["capture_id"] == config["run_id"] and manifest["rank"] == rank,
            "capture identity differs from launched run")
    require(manifest["phase"] == "stopped" and manifest["export_complete"] is True and
            manifest["local_contract_valid"] is True and manifest["recorded"] > 0,
            "capture incomplete or local contract invalid")
    require(all(manifest[name] == 0 for name in ("dropped", "invalid", "unsupported", "association_errors")),
            "capture has loss, invalid observations or association errors")
    period = unsigned(manifest["sample_period_ns"], "sample period")
    require(period > 0, "zero sampling period")
    require(adapter["adapter_version"] == 1 and adapter["base_commit"] == COMMIT and
            adapter["scope"] == "two_node_serial_float32_ring_simple_net_ib" and
            adapter["clock"] == "CLOCK_MONOTONIC" and adapter["adapter_error"] == 0 and
            adapter["pending_operation"] is False and adapter["completed_operations"] == adapter["plan_count"] == 9,
            "adapter scope, execution or operation count differs")
    require(adapter["completion_observation"] == "actual_launch_stream_event_query_after_application_stream_sync"
            and adapter["single_plan_collectives"] == 1 and adapter["stream_matches_user_stream"] is True
            and adapter["producer_join_verified"] is True,
            "actual stream/single-operation completion evidence missing")
    require(adapter["readiness"] == "observed_eligible_slice_frontier_before_isend" and
            adapter["exact_gpu_timestamp"] is False and adapter["hardware_fault_proof"] is False,
            "unsupported readiness or completion claims")
    mapping = lines(capture / "trace/channel-map.jsonl")
    log = (folder / "run.log").read_text()
    require(single(r"^application_clock=(.+)$", log, "application clock") == "CLOCK_MONOTONIC",
            "application and adapter clock relation unverified")
    application = {int(op): {} for op in range(9)}
    for op, stage, stamp in re.findall(rf"^rank={rank} operation=(\d+) count=\d+ "
                                       r"stage=(\w+) time_ns=(\d+)$", log, re.MULTILINE):
        application[int(op)][stage] = int(stamp)
    begins, channels, peers = {}, defaultdict(set), {}
    for row in mapping:
        k = key(row)
        require(k[2] == rank and k[1] in range(9), "mapping rank or operation differs")
        if row["record_kind"] == "operation_begin":
            require(k not in begins, "duplicate operation begin")
            require(unsigned(row["message_bytes"], "message size") == COUNTS[k[1]//3]*4,
                    "logical message size differs")
            unsigned(row["started_at_ns"], "begin time")
            marks = application[k[1]]
            require(marks["api_begin"] <= row["started_at_ns"] <= marks["api_return"],
                    "NCCL operation not mapped to corresponding application call")
            begins[k] = row
        elif row["record_kind"] == "channel_membership":
            channel = unsigned(row["channel"], "channel")
            require(channel < 2 and channel not in channels[k], "invalid or duplicate channel")
            channels[k].add(channel)
        elif row["record_kind"] == "peer_connection":
            ref = connection(row)
            require(ref not in peers and row["peer_rank"] == 1-rank and row["transport"] == "NET/IB"
                    and row["direction"] in ("send", "recv"), "invalid or duplicate peer connection")
            peers[ref] = row
        else:
            raise ValueError("unknown channel map record")
    require(len(begins) == 9 and {k[1] for k in begins} == set(range(9)) and
            len({k[0] for k in begins}) == 1 and set(channels) == set(begins), "operation/channel mapping incomplete")
    for k in begins:
        require(len(channels[k]) == selections[k[1]]["channels"], "actual plan and captured channel count differ")
        for channel in channels[k]:
            matching = [p for ref, p in peers.items() if ref[:3] == k and ref[3] == channel]
            require(Counter(p["direction"] for p in matching) == {"send": 1, "recv": 1},
                    "each operation/channel needs one send and one receive peer")
    plans, completions = {}, {}
    for row in lines(capture / "trace/plan-map.jsonl"):
        k = key(row)
        require(k in begins and row["stream_id"] == 1, "unknown plan operation or stream")
        if row["record_kind"] == "plan_binding":
            require(k not in plans and row["collective_count"] == 1 and row["plan_id"] == k[1]+1,
                    "plan is duplicate or contains multiple collectives")
            plans[k] = row
        elif row["record_kind"] == "operation_completion":
            require(k not in completions and row["observed_at_ns"] >= begins[k]["started_at_ns"],
                    "duplicate or earlier completion")
            marks = application[k[1]]
            require(marks["api_return"] <= row["observed_at_ns"] <= marks["stream_sync_return"],
                    "completion outside local application wait interval")
            completions[k] = row
        else:
            raise ValueError("unknown plan record")
    require(set(plans) == set(completions) == set(begins), "plan/completion coverage differs")
    require(all(completions[k]["plan_id"] == plans[k]["plan_id"] for k in begins), "completion plan differs")
    raw_send = lines(capture / f"trace/send-rank{rank}.jsonl")
    raw_recv = lines(capture / f"trace/recv-rank{rank}.jsonl")
    require(adapter["send_samples"] == len(raw_send) and adapter["recv_samples"] == len(raw_recv),
            "raw sample count differs")
    periodic_windows, normalized = 0, []
    for direction, rows, counters in (
            ("send", raw_send, ("gpu_ready_steps", "transmitted_steps", "done_steps")),
            ("recv", raw_recv, ("posted_steps", "received_steps", "transmitted_steps", "done_steps"))):
        groups = defaultdict(list)
        for row in rows:
            ref = connection(row)
            require(ref in peers and peers[ref]["direction"] == direction and ref[:3] in begins and
                    ref[3] in channels[ref[:3]], "raw progress connection not mapped")
            require(row["registered_buffer"] is False, "registered-buffer progress is unsupported")
            unit, total = unsigned(row["slice_steps"], "slice"), unsigned(row["nsteps"], "nsteps")
            unsigned(row["step_base"], "step base")
            values = [unsigned(row[name], name) for name in counters]
            require(unit > 0 and total > 0 and all(value % unit == 0 for value in [total, *values]) and
                    total >= values[0] and values == sorted(values, reverse=True), "raw units or ordering differ")
            require(unsigned(row["observed_at_ns"], "sample time") >= begins[ref[:3]]["started_at_ns"],
                    "sample precedes operation")
            groups[ref].append(row)
            if direction == "send":
                normalized.append(normalize_send_steps(nsteps=total, slice_steps=unit, gpu_ready_steps=values[0],
                    transmitted_steps=values[1], done_steps=values[2], registered_buffer=False))
        require(set(groups) == {ref for ref, p in peers.items() if p["direction"] == direction},
                "missing connection progress")
        for ref, records in groups.items():
            times = [r["observed_at_ns"] for r in records]
            require(times == sorted(times), "connection clock regressed")
            for a, b in zip(records, records[1:]):
                require(all(a[name] == b[name] for name in ("step_base", "nsteps", "slice_steps")) and
                        all(a[name] <= b[name] for name in counters), "raw cumulative counters regressed")
                # At least one genuine intermediate window, not just forced final snapshot.
                if direction == "send" and b["observed_at_ns"]-a["observed_at_ns"] >= period and b["done_steps"] < b["nsteps"]:
                    periodic_windows += 1
            require(records[-1]["done_steps"] == records[-1]["nsteps"], "final network progress missing")
    events = [Event.from_dict(row) for row in lines(capture / f"trace/rank{rank}.jsonl")]
    require(len(events) == manifest["event_records"] and len({e.event_id for e in events}) == len(events),
            "event count or duplicate identity differs")
    event_progress, event_completions = [], {}
    comm_hash = next(iter(begins))[0]
    expected_comm = f"{config['run_id']}:{comm_hash:016x}"
    for event in events:
        validate_event(event)
        ctx = event.context
        require(ctx.communicator_id == expected_comm and ctx.rank == rank and ctx.op_seq in range(9) and
                ctx.collective == "all_reduce" and event.time.domain is TimeDomain.NCCL_MONOTONIC_NS and
                event.schema_version == 2 and event.source == "m2_process_local_recorder_unverified" and not event.dependencies,
                "event identity/source/time differs")
        if event.event_kind is EventKind.PROGRESS_SNAPSHOT:
            event_progress.append(event)
        elif event.event_kind is EventKind.OPERATION_COMPLETION:
            k = (comm_hash, ctx.op_seq, rank)
            require(k not in event_completions and ctx.channel is None and
                    event.time.value == completions[k]["observed_at_ns"] and
                    event.payload.started_at_ns == begins[k]["started_at_ns"] and
                    event.payload.message_bytes == begins[k]["message_bytes"], "completion Event differs from raw evidence")
            event_completions[k] = event
        else:
            raise ValueError("unsupported Event kind")
    require(set(event_completions) == set(begins) and len(event_progress) == len(raw_send), "Event coverage differs")
    for event, raw, payload in zip(event_progress, raw_send, normalized):
        require((event.context.op_seq, event.context.channel, event.time.value) ==
                (raw["op_seq_candidate"], raw["channel"], raw["observed_at_ns"]) and event.payload == payload,
                "normalized Event differs from raw send evidence")
    return {"comm_hash": comm_hash, "channels": {k[1]: channels[k] for k in begins},
            "periodic_windows": periodic_windows, "send_samples": len(raw_send), "recv_samples": len(raw_recv)}


def verify_capture(directory: Path):
    config = load_json(directory / "run-config.json")
    require(config.get("capture_mode") == "m2", "requires M2 run provenance, not CPU fixtures or M1 logs")
    require(re.fullmatch(r"[0-9a-f]{32}", config["run_id"]), "invalid capture run identity")
    proof = config["provenance"]
    require(proof["base_commit"] == COMMIT and re.fullmatch(r"[0-9a-f]{40}", proof["base_tree"]) and
            re.fullmatch(r"[0-9a-f]{64}", proof["source_archive_sha256"]) and len(proof["inputs"]) == 5 and
            all(re.fullmatch(r"[0-9a-f]{64}", v) for v in proof["inputs"].values()), "invalid build input provenance")
    def build_validator(folder, text):
        build, source = load_json(folder / "m2-build.json"), load_json(folder / "m2-source-manifest.json")
        require(build["status"] == "PASS" and build["build_kind"] == "m2_instrumented" and build["trace"] == 1 and
                build["runtime_version"] == 22105 and build["library_sha256"] == config["expected_sha256"] and
                all(build[name] == value and source[name] == value for name, value in proof.items()),
                "patched build does not match uploaded inputs or loaded library")
        require(build["patched_files"] == source["patched_files"] and len(source["patched_files"]) == 9,
                "prepared source and built source evidence differs")
        require(single(r"^source_commit=(.+)$", text, "source commit") == COMMIT and
                single(r"^source_tag=(.+)$", text, "source tag") == "v2.21.5-1" and
                single(r"^source_clean=(.+)$", text, "source state") == "no" and
                single(r"^build_kind=(.+)$", text, "build kind") == "m2_instrumented" and
                single(r"^trace=(.+)$", text, "TRACE") == "1" and single(r"^status=(.+)$", text, "status") == "PASS",
                "M2 patched source manifest differs")
        status = load_json(folder / "status.json")
        require(all(status[name] == 0 for name in ("probe_exit_code", "build_exit_code", "workload_exit_code")),
                "M2 runtime stage failed")
    baseline = verify_run(directory, config["expected_sha256"], build_validator=build_validator)
    captures = [verify_rank_capture(directory / f"rank{r}", r, config, baseline["selections"]) for r in (0, 1)]
    require(captures[0]["comm_hash"] == captures[1]["comm_hash"] and captures[0]["channels"] == captures[1]["channels"],
            "cross-rank operation/channel identity differs")
    require(any(len(channels) > 1 for channels in captures[0]["channels"].values()),
            "no actual multi-channel operation to verify identity propagation")
    require(sum(c["periodic_windows"] for c in captures) > 0,
            "no periodic intermediate window: review sampling period/message size before acceptance")
    return {"baseline": baseline, "captures": captures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="包含真实 trace、manifest 和 workload 日志的目录")
    args = parser.parse_args()
    try:
        result = verify_capture(args.directory)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.exit(1, f"capture_verification=FAILED reason={error}\n")
    print("m2_technical_checks=PASS ranks=2 operations_per_rank=9 result_checks=18")
    print("node_evidence=" + result["baseline"]["node_evidence"])
    for rank, capture in enumerate(result["captures"]):
        print(f"rank={rank} completions=9 send_samples={capture['send_samples']} recv_samples={capture['recv_samples']} "
              f"periodic_intermediate_windows={capture['periodic_windows']} loss=0")
    print("completion_scope=local_whole_collective_cpu_observation actual_gpu_timestamp=no")
    print("readiness_scope=observed_eligible_slice_frontier hardware_fault_proof=no")
    print("capture_verification=PASS scope=offline_crosschecked_runtime_evidence")
    print("full_m2_acceptance=PENDING_USER_REVIEW")


if __name__ == "__main__":
    main()
