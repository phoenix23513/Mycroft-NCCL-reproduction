#!/usr/bin/env python3
"""每个固定节点启动一个 rank，依次运行三组 M3 对照，rank0 归档一次。"""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys

from prepare_m2 import ROOT, ADAPTER, COPIES as M2_COPIES, check_bundle, sha256
from prepare_m3 import PATCH, COPIES as M3_COPIES
from run_m1 import SETTINGS, TRACE_SHA256, execute, wait_json, write_json, run
from run_m2 import RESULT_FILES as CAPTURE_FILES
from package_day16_results import package_results

CASES = ("baseline", "capture", "delay")
VERIFY = ROOT / "instrumentation/nccl-2.21.5/m3/verify_experiment.py"
CASE_FILES = (*CAPTURE_FILES, "decision.json", "rank0/capture/injection.json", "rank1/capture/injection.json")
RESULT_FILES = ("experiment-config.json", "run-status.txt", "analysis.txt", "rank0-status.json", "rank1-status.json",
                *(f"{case}/{name}" for case in CASES for name in CASE_FILES
                  if case != "baseline" or ("capture/" not in name and "m2-" not in name)))


def product_proof(root, upload):
    names = [ADAPTER+"nccl-2.21.5-m2.patch", *M2_COPIES, PATCH, *M3_COPIES]
    proof = {"base_commit":upload["base_commit"], "base_tree":upload["base_tree"],
             "source_archive_sha256":upload["files"]["nccl-source.tar"],
             "inputs":{name:upload["files"][name] for name in names}}
    build=json.loads((root/"m2-build.json").read_text())
    digest=sha256(root/"lib/libnccl.so.2.21.5")
    if build["status"]!="PASS" or build["build_kind"]!="m3_instrumented" or any(
            build[key]!=value for key,value in proof.items()) or build["library_sha256"]!=digest:
        raise ValueError("M3 library/source provenance differs from this package")
    return proof,digest


def run_suite(rank,directory,baseline,product,baseline_hash,target,timeout):
    upload=check_bundle(ROOT)
    proof,digest=product_proof(product,upload)
    if sha256(baseline/"lib/libnccl.so.2.21.5")!=baseline_hash:
        raise ValueError("baseline library differs from independently verified M1 SHA256")
    archive=directory.with_name(directory.name+".tar.gz")
    claimed=False
    states={}
    status=1
    failure=None
    try:
        if rank==0:
            if archive.exists() or archive.is_symlink(): raise ValueError("result archive exists; choose a new run directory")
            directory.mkdir(parents=True,exist_ok=False);claimed=True
            suite={"experiment":"m3","suite_id":secrets.token_hex(16),"cases":list(CASES),
                   "settings":SETTINGS,"baseline_sha256":baseline_hash,"instrumented_sha256":digest,
                   "target":target,"provenance":proof}
            write_json(directory/"experiment-config.json",suite)
        else:
            suite=wait_json(directory/"experiment-config.json",timeout)
            if any(suite[key]!=value for key,value in {
                    "experiment":"m3","cases":list(CASES),"settings":SETTINGS,"baseline_sha256":baseline_hash,
                    "instrumented_sha256":digest,"target":target,"provenance":proof}.items()):
                raise ValueError("rank0/rank1 experiment configuration differs")
            (directory/"rank1.claim").mkdir()  # Duplicate workers cannot overwrite suite status.
            claimed=True
        for name in CASES:
            case=directory/name
            case_target=None if name=="baseline" else dict(target,delay_ns=0 if name=="capture" else target["delay_ns"])
            library_hash=baseline_hash if name=="baseline" else digest
            expected={"suite_id":suite["suite_id"],"case":name,"capture_mode":"m1" if name=="baseline" else "m3",
                      "expected_sha256":library_hash,"iterations":3,"settings":SETTINGS,"delay_target":case_target}
            if name!="baseline": expected["provenance"]=proof
            if rank==0:
                case.mkdir()
                config=dict(expected,run_id=secrets.token_hex(16))
                write_json(case/"run-config.json",config)
            else:
                config=wait_json(case/"run-config.json",timeout)
                if any(config[key]!=value for key,value in expected.items()):
                    raise ValueError("case configuration differs")
            code,interrupted=execute(rank,case,baseline if name=="baseline" else product,config,timeout,
                                     capture_m2=name!="baseline",delay_target=case_target)
            states[name]=code
            if rank==0:
                codes=[code,1]
                try:
                    peer=wait_json(case/"rank1/status.json",5 if interrupted else timeout)
                    if peer["run_id"]==config["run_id"]:codes[1]=peer["exit_code"]
                except (OSError,ValueError,KeyError,TimeoutError):pass
                checked=1
                if codes==[0,0]:
                    checked=run([sys.executable,str(VERIFY),str(case),"--case-only"],case/"analysis.txt",60)
                else:(case/"analysis.txt").write_text("m3_verification=NOT_RUN reason=rank_failure_or_missing\n")
                (case/"run-status.txt").write_text(f"job_exit_code={checked}\nrank0_exit_code={codes[0]}\nrank1_exit_code={codes[1]}\n")
                decision={"suite_id":suite["suite_id"],"run_id":config["run_id"],"continue":checked==0}
                write_json(case/"decision.json",decision)
            else:
                decision=wait_json(case/"decision.json",5 if interrupted else timeout)
                if decision["suite_id"]!=suite["suite_id"] or decision["run_id"]!=config["run_id"]:
                    raise ValueError("case decision belongs to another run")
            if not decision["continue"]:break
        status=0 if len(states)==3 and all(code==0 for code in states.values()) and decision["continue"] else 1
    except Exception as error:
        failure=str(error)
        raise
    finally:
        if claimed:
            write_json(directory/f"rank{rank}-status.json",{"rank":rank,"suite_id":suite["suite_id"],
                                                        "exit_code":status,"cases":states,"error":failure})
            if rank==0:
                codes=[status,1]
                try:
                    peer=wait_json(directory/"rank1-status.json",timeout)
                    if peer["suite_id"]==suite["suite_id"]:codes[1]=peer["exit_code"]
                except (OSError,ValueError,KeyError,TimeoutError):pass
                if codes==[0,0]:
                    status=run([sys.executable,str(VERIFY),str(directory)],directory/"analysis.txt",60)
                else:(directory/"analysis.txt").write_text("m3_verification=NOT_RUN reason=case_failure_or_missing\n")
                (directory/"run-status.txt").write_text(f"job_exit_code={status}\nrank0_exit_code={codes[0]}\nrank1_exit_code={codes[1]}\n")
                checksum=package_results(directory,archive,files=RESULT_FILES,package_root="m3-results")
                print(f"result_bundle={archive}\narchive_sha256={checksum}\nm3_job_exit_code={status}",flush=True)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank",type=int,choices=(0,1),required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--baseline-nccl-root",type=Path,required=True)
    parser.add_argument("--nccl-root",type=Path,required=True)
    parser.add_argument("--baseline-sha256",default=TRACE_SHA256)
    parser.add_argument("--target-rank",type=int,choices=(0,1),default=0)
    parser.add_argument("--target-operation",type=int,choices=range(9),default=6)
    parser.add_argument("--target-channel",type=int,choices=(0,1),default=0)
    parser.add_argument("--delay-ms",type=int,default=100)
    parser.add_argument("--peer-timeout",type=int,default=900)
    args=parser.parse_args()
    if not 1<=args.delay_ms<=2000 or not 1<=args.peer_timeout<=900:
        parser.error("delay-ms must be 1..2000; peer-timeout must be 1..900")
    os.umask(0o077)
    def interrupt(signum,frame):raise InterruptedError(f"signal {signum}")
    signal.signal(signal.SIGTERM,interrupt)
    target={"rank":args.target_rank,"operation":args.target_operation,"channel":args.target_channel,
            "delay_ns":args.delay_ms*1000000}
    try:
        status=run_suite(args.rank,args.run_dir.absolute(),args.baseline_nccl_root.absolute(),
                         args.nccl_root.absolute(),args.baseline_sha256,target,args.peer_timeout)
    except (OSError,ValueError,KeyError,RuntimeError,TimeoutError,InterruptedError,subprocess.SubprocessError) as error:
        parser.exit(2,f"m3_launch=FAILED reason={error}\n")
    if args.rank==1: print(f"rank=1 exit_code={status}; download only the rank0 result bundle",flush=True)
    raise SystemExit(status)


if __name__=="__main__":main()
