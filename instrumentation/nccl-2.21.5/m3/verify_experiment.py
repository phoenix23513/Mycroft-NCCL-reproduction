#!/usr/bin/env python3
"""M3 对照与注入核对；注入配置只核对实验，不用作 M4 的诊断输入。"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workloads/minimal_allreduce"))
from verify_m1 import verify as verify_run, restore_application_records
from verify_results import COMMIT, require, single

SPEC = importlib.util.spec_from_file_location("m3_capture_checks", ROOT / "instrumentation/nccl-2.21.5/day16/verify_capture.py")
CAPTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPTURE)
CASES = ("baseline", "capture", "delay")


def read(path):
    return json.loads(path.read_text())


def verify_injection(folder, rank, target):
    item = read(folder / "capture/injection.json")
    require(item["injection_version"] == 1 and item["kind"] == "software_isend_delay" and
            item["clock"] == "CLOCK_MONOTONIC" and item["local_rank"] == rank and
            item["error"] is False, "invalid injection observation")
    for field, config in (("target_rank", "rank"), ("target_operation", "operation"),
                          ("target_channel", "channel"), ("delay_ns", "delay_ns")):
        require(item[field] == target[config], "injection target differs from case configuration")
    for field in ("held_step", "begin_ns", "deadline_ns", "release_ns", "blocked_attempts", "comm_hash", "connection_id"):
        CAPTURE.unsigned(item[field], field)
    if target["delay_ns"] == 0 or rank != target["rank"]:
        require(item["started"] is False and item["released"] is False and
                all(item[key] == 0 for key in ("begin_ns", "deadline_ns", "release_ns", "blocked_attempts")),
                "non-target path actively delayed")
        return None
    require(item["started"] is True and item["released"] is True and item["blocked_attempts"] > 0,
            "configured injection was not hit and released")
    require(item["deadline_ns"] == item["begin_ns"] + target["delay_ns"] and
            item["release_ns"] >= item["deadline_ns"], "injection released before deadline")
    rows = CAPTURE.lines(folder / f"capture/trace/send-rank{rank}.jsonl")
    path = [row for row in rows if row["op_seq_candidate"] == target["operation"] and
            row["channel"] == target["channel"] and row["comm_hash"] == item["comm_hash"] and
            row["connection_id"] == item["connection_id"]]
    held = [row for row in path if item["begin_ns"] <= row["observed_at_ns"] < item["release_ns"]]
    require(len(held) >= 2 and all(row["transmitted_steps"] == item["held_step"] and
            row["gpu_ready_steps"] > row["transmitted_steps"] for row in held),
            "no sampled ready-but-not-submitted interval on target path")
    period = read(folder / "capture/capture-manifest.json")["sample_period_ns"]
    span = held[-1]["observed_at_ns"] - held[0]["observed_at_ns"]
    require(span >= period, "injection interval contains no periodic sampling window")
    require(path[-1]["done_steps"] == path[-1]["nsteps"] and
            path[-1]["observed_at_ns"] >= item["release_ns"], "target did not finish after release")
    completions = CAPTURE.lines(folder / "capture/trace/plan-map.jsonl")
    completion = next(row for row in completions if row["record_kind"] == "operation_completion" and
                      row["op_seq_candidate"] == target["operation"])
    require(completion["observed_at_ns"] >= item["release_ns"], "completion precedes injection release")
    return {"rank": rank, "operation": target["operation"], "channel": target["channel"],
            "requested_delay_ns": target["delay_ns"],
            "observed_hold_ns": item["release_ns"] - item["begin_ns"], "hold_samples": len(held),
            "sampled_hold_span_ns": span,
            "max_sample_gap_ns": max(b["observed_at_ns"]-a["observed_at_ns"] for a,b in zip(held,held[1:]))}


def verify_case(directory):
    config = read(directory / "run-config.json")
    name = config["case"]
    require(name in CASES and config["capture_mode"] == ("m1" if name == "baseline" else "m3"),
            "invalid M3 case identity")
    if name == "baseline":
        result = verify_run(directory, config["expected_sha256"])
        for rank in (0,1):
            folder = directory / f"rank{rank}"
            require(read(folder / "environment.json").get("delay_target") is None and
                    not (folder / "capture").exists(), "baseline unexpectedly enabled capture or injection")
        return {"baseline": result, "captures": [], "injection": None}
    proof = config["provenance"]
    require(proof["base_commit"] == COMMIT and len(proof["inputs"]) == 9 and
            re.fullmatch(r"[0-9a-f]{40}", proof["base_tree"]) and
            re.fullmatch(r"[0-9a-f]{64}", proof["source_archive_sha256"]) and
            all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in proof["inputs"].values()),
            "M3 source provenance missing")
    def validator(folder, text):
        build, source = read(folder / "m2-build.json"), read(folder / "m2-source-manifest.json")
        require(build["status"] == "PASS" and build["build_kind"] == "m3_instrumented" and
                build["trace"] == 1 and build["runtime_version"] == 22105 and
                build["library_sha256"] == config["expected_sha256"] and
                all(build[key] == value and source[key] == value for key,value in proof.items()) and
                build["patched_files"] == source["patched_files"] and len(source["patched_files"]) == 12,
                "M3 source/build/loaded library differs")
        require(single(r"^source_commit=(.+)$",text,"commit") == COMMIT and
                single(r"^source_tag=(.+)$",text,"tag") == "v2.21.5-1" and
                single(r"^source_clean=(.+)$",text,"source") == "no" and
                single(r"^build_kind=(.+)$",text,"kind") == "m3_instrumented" and
                single(r"^trace=(.+)$",text,"trace") == "1" and
                single(r"^status=(.+)$",text,"status") == "PASS", "M3 build manifest differs")
        require(all(read(folder / "status.json")[field] == 0 for field in
                ("probe_exit_code", "build_exit_code", "workload_exit_code")), "runtime stage failed")
        require(read(folder / "environment.json")["delay_target"] == config["delay_target"],
                "actual requested injection environment differs")
    result = verify_run(directory, config["expected_sha256"], build_validator=validator)
    captures = [CAPTURE.verify_rank_capture(directory / f"rank{rank}", rank, config,
                                            result["selections"]) for rank in (0,1)]
    require(captures[0]["comm_hash"] == captures[1]["comm_hash"] and
            captures[0]["channels"] == captures[1]["channels"], "cross-rank identity differs")
    require(any(len(v)>1 for v in captures[0]["channels"].values()) and
            sum(c["periodic_windows"] for c in captures)>0, "multi-channel or periodic evidence missing")
    observations = [verify_injection(directory / f"rank{rank}",rank,config["delay_target"]) for rank in (0,1)]
    return {"baseline": result, "captures": captures,
            "injection": next((v for v in observations if v is not None), None)}


def verify_experiment(directory):
    suite = read(directory / "experiment-config.json")
    require(suite["experiment"] == "m3" and suite["cases"] == list(CASES), "three-case experiment required")
    results, identities, timings = {}, {}, []
    for name in CASES:
        case = directory / name
        config = read(case / "run-config.json")
        require(config["suite_id"] == suite["suite_id"] and config["case"] == name and
                config["settings"] == suite["settings"] and config["iterations"] == 3,
                "case belongs to another suite or uses different settings")
        expected_hash = suite["baseline_sha256"] if name == "baseline" else suite["instrumented_sha256"]
        require(config["expected_sha256"] == expected_hash, "case library differs from suite")
        if name != "baseline":
            expected_target = dict(suite["target"], delay_ns=0 if name == "capture" else suite["target"]["delay_ns"])
            require(config["delay_target"] == expected_target and config["provenance"] == suite["provenance"],
                    "normal and delayed cases differ in more than the injection switch")
        results[name] = verify_case(case)
        for rank in (0,1):
            folder = case / f"rank{rank}"
            probe,env = read(folder / "probe.json"),read(folder / "environment.json")
            identity = (probe["node_identifiers"],probe["gpu_inventory"],probe["kernel"],
                        probe["os_release"],probe["compiler"],probe["cuda_compiler"],
                        probe["memlock_bytes"],env["optional_network_settings"])
            if rank in identities: require(identities[rank] == identity, "case node/GPU/toolchain/network changed")
            identities[rank] = identity
            log,_ = restore_application_records((folder / "run.log").read_text(),probe["hostname"])
            marks = {}
            for op,stage,stamp in re.findall(rf"^rank={rank} operation=(\d+) count=\d+ stage=(\w+) time_ns=(\d+)$",log,re.M):
                marks.setdefault(int(op),{})[stage]=int(stamp)
            for group, size in enumerate((16,16384,4194304)):
                values=[marks[op]["stream_sync_return"]-marks[op]["api_begin"] for op in range(group*3,group*3+3)]
                timings.append({"case":name,"rank":rank,"message_bytes":size,"samples":3,
                                "median_local_call_to_sync_ns":statistics.median(values),"local_call_to_sync_ns":values})
    require(results["capture"]["injection"] is None and results["delay"]["injection"] is not None,
            "injection control mismatch")
    return {"cases": results,"timing_scope":"local_CPU_call_to_stream_sync_not_GPU_duration",
            "timings":timings,"injection":results["delay"]["injection"],
            "remote_propagation":"preserved_for_M4; no_cross_host_timestamp_comparison_or_root_cause_claim"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--case-only",action="store_true")
    args=parser.parse_args()
    try:
        result=verify_case(args.directory) if args.case_only else verify_experiment(args.directory)
    except (OSError,ValueError,KeyError,TypeError,IndexError,StopIteration) as error:
        parser.exit(1,f"m3_verification=FAILED reason={error}\n")
    print("m3_technical_checks=PASS scope="+("single_case" if args.case_only else "three_case_runtime_evidence"))
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,default=sorted))
    print("hardware_fault_proof=no\nfull_m3_acceptance=PENDING_USER_REVIEW")


if __name__ == "__main__":
    main()
