#!/usr/bin/env python3
"""从上传包构建独立的 M2 插桩 NCCL；无需 GPU，失败也打包证据。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from prepare_m2 import ROOT, COMMIT, prepare, sha256
from package_day16_results import package_results


def command(arguments, log, environment=None):
    with log.open("w") as output:
        subprocess.run(arguments, stdout=output, stderr=subprocess.STDOUT, env=environment, check=True)


def build(work, jobs, cuda):
    archive_path = work.with_name(work.name + ".tar.gz")
    if archive_path.exists() or archive_path.is_symlink():
        raise ValueError("build result archive already exists; choose a new work directory")
    work.mkdir(parents=True, exist_ok=False)
    status = 1
    temporary = None
    metadata = {"build_kind": "m2_instrumented", "base_commit": COMMIT, "status": "FAILED"}
    try:
        if any(char.isspace() for char in str(work) + str(cuda)):
            raise ValueError("NCCL make paths must not contain whitespace")
        for tool in ("git", "make", "g++", "python3", "readelf"):
            if not shutil.which(tool):
                raise ValueError(f"required tool missing: {tool}")
        if not (cuda / "bin/nvcc").is_file() or not (cuda / "include/cuda_runtime.h").is_file() or not (
                cuda / "lib64/libcudart_static.a").is_file():
            raise ValueError("CUDA devel nvcc/headers/static runtime missing")
        # Shared storage keeps final products/evidence, not thousands of compiler intermediates.
        temporary = tempfile.TemporaryDirectory(prefix="mycroft-m2-build-")
        scratch = Path(temporary.name)
        source = scratch / "source"
        provenance = prepare(ROOT, source)
        product = scratch / "nccl"
        product.mkdir()
        saved = work / "nccl"
        saved.mkdir()
        shutil.copyfile(source / "m2-source-manifest.json", saved / "m2-source-manifest.json")
        gencode = "-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70"
        arguments = ["make", "-C", str(source / "src"), "-j", str(jobs), "lib",
                     f"BUILDDIR={product}", f"CUDA_HOME={cuda}", f"CUDA_INC={cuda / 'include'}",
                     f"CUDA_LIB={cuda / 'lib64'}", "CXX=g++", f"NVCC_GENCODE={gencode}",
                     "DEBUG=0", "TRACE=1", "ASAN=0", "GCOV=0", "KEEP=0", "VERBOSE=1",
                     "NVTX=1", "PROFAPI=1", "RDMA_CORE=0", "CUDARTLIB=cudart_static", "ONLY_FUNCS="]
        environment = dict(os.environ)
        for name in ("MAKEFLAGS", "MAKEOVERRIDES", "MFLAGS", "CXXFLAGS", "NVCUFLAGS", "LDFLAGS",
                     "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
            environment.pop(name, None)
        with (work / "environment.txt").open("w") as output:
            output.write(Path("/etc/os-release").read_text())
            for args in (["g++", "--version"], [str(cuda / "bin/nvcc"), "--version"], ["make", "--version"],
                         [sys.executable, "--version"]):
                subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, check=True)
        metadata.update(provenance, jobs=jobs, trace=1, nvcc_gencode=gencode, command=arguments)
        print("m2_build=COMPILING gpu_required=no target=V100_sm70", flush=True)
        command(arguments, work / "build.log", environment)
        shutil.copytree(product / "include", saved / "include")
        (saved / "lib").mkdir()
        shutil.copyfile(product / "lib/libnccl.so.2.21.5", saved / "lib/libnccl.so.2.21.5")
        (saved / "lib/libnccl.so.2").symlink_to("libnccl.so.2.21.5")
        (saved / "lib/libnccl.so").symlink_to("libnccl.so.2")
        # Loading this library checks host symbols/version and does not execute a GPU collective.
        command(["g++", "-std=c++17", str(ROOT / "cluster/crater/probes/nccl_version.cpp"), "-ldl",
                 "-o", str(work / "nccl_version")], work / "probe-build.log")
        library = saved / "lib/libnccl.so.2.21.5"
        command([str(work / "nccl_version"), str(library)], work / "verification.log")
        symbols = subprocess.check_output(["readelf", "--dyn-syms", "--wide", str(library)], text=True)
        for name in ("mycroftM2Start", "mycroftM2Complete", "mycroftM2Finish"):
            if not any(line.split()[-1:] == [name] and " UND " not in line for line in symbols.splitlines()):
                raise ValueError(f"M2 exported symbol missing: {name}")
        (work / "m2-symbols.txt").write_text("\n".join(line for line in symbols.splitlines() if "mycroftM2" in line) + "\n")
        # Ensure compilation did not alter the prepared instrumented source inputs.
        if any(sha256(source / name) != digest for name, digest in provenance["patched_files"].items()):
            raise ValueError("instrumented source changed during build")
        metadata.update(status="PASS", library_sha256=sha256(library), runtime_version=22105)
        (saved / "build-manifest.txt").write_text(
            f"source_commit={COMMIT}\nsource_tag=v2.21.5-1\nsource_clean=no\n"
            f"build_kind=m2_instrumented\ntrace=1\nstatus=PASS\nlibrary_sha256={metadata['library_sha256']}\n")
        (saved / "m2-build.json").write_text(json.dumps(metadata, indent=2) + "\n")
        status = 0
        print(f"m2_build=PASS nccl_root={saved}\nlibrary_sha256={metadata['library_sha256']}", flush=True)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        metadata["error"] = str(error)
        print(f"m2_build=FAILED reason={error}", flush=True)
    finally:
        if temporary is not None:
            temporary.cleanup()
        (work / "build-status.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (work / "run-status.txt").write_text(f"build_exit_code={status}\ngpu_execution=NOT_RUN\n")
        # Preserve the full compiler output locally, and a bounded tail in the download bundle.
        if (work / "build.log").exists():
            with (work / "build.log").open("rb") as source:
                source.seek(max(0, (work / "build.log").stat().st_size - 512*1024))
                (work / "build-tail.log").write_bytes(source.read())
        digest = package_results(work, work.with_name(work.name + ".tar.gz"), package_root="m2-build-results",
            files=("run-status.txt", "build-status.json", "environment.txt", "build-tail.log", "probe-build.log",
                   "verification.log", "m2-symbols.txt", "nccl/m2-build.json", "nccl/m2-source-manifest.json",
                   "nccl/build-manifest.txt"))
        print(f"result_bundle={work.with_name(work.name + '.tar.gz')}\narchive_sha256={digest}", flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="全新的持久化构建目录")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cuda-home", type=Path, default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")))
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 32:
        parser.error("jobs must be between 1 and 32")
    try:
        status = build(args.work_dir.absolute(), args.jobs, args.cuda_home.absolute())
    except (OSError, ValueError) as error:
        parser.exit(2, f"m2_build_launch=FAILED reason={error}\n")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
