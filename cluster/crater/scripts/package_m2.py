#!/usr/bin/env python3
"""打包固定 NCCL 原源码、M2 patch/记录器及构建/双 rank 运行入口。"""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

from package_m1 import FILES as M1_FILES
from package_day16_results import portable
from prepare_m2 import ROOT, COMMIT, ADAPTER, COPIES, sha256

FILES = tuple(dict.fromkeys((*M1_FILES,
    "cluster/crater/scripts/package_m1.py", "cluster/crater/scripts/prepare_m2.py",
    "cluster/crater/scripts/build_m2.py", "cluster/crater/scripts/run_m2.py",
    "cluster/crater/probes/nccl_version.cpp", ADAPTER + "nccl-2.21.5-m2.patch", *COPIES,
    "instrumentation/nccl-2.21.5/day16/verify_capture.py",
    "instrumentation/nccl-2.21.5/day16/README.md")))


def add_bytes(archive, name, content, package_root="m2-experiment"):
    info = tarfile.TarInfo(package_root + "/" + name)
    info.size, info.mode = len(content), 0o644
    archive.addfile(info, io.BytesIO(content))


def package(output, *, extra_files=(), package_root="m2-experiment"):
    upstream = ROOT / "third_party/nccl"
    if subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise ValueError("NCCL checkout differs from pinned base")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        raise ValueError("NCCL original checkout must remain clean")
    paths = ["LICENSE.txt", "README.md", "Makefile", "makefiles", "src", "ext-net", "ext-tuner"]
    original = subprocess.check_output(["git", "-C", str(upstream), "archive", "--format=tar", COMMIT, *paths])
    import hashlib
    files = {name: sha256(ROOT / name) for name in (*FILES, *extra_files)}
    for path in sorted((ROOT / "src/mycroft").rglob("*.py")):
        files[str(path.relative_to(ROOT))] = sha256(path)
    files["nccl-source.tar"] = hashlib.sha256(original).hexdigest()
    manifest = {"manifest_version": 1, "base_commit": COMMIT, "base_tag": "v2.21.5-1",
                "base_tree": subprocess.check_output(["git", "-C", str(upstream), "rev-parse", COMMIT + "^{tree}"], text=True).strip(),
                "source_paths": paths, "files": files}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz") as archive:
        for name in files:
            if name != "nccl-source.tar":
                archive.add(ROOT / name, package_root + "/" + name, recursive=False, filter=portable)
        add_bytes(archive, "nccl-source.tar", original, package_root)
        add_bytes(archive, "upload-manifest.json", (json.dumps(manifest, indent=2) + "\n").encode(), package_root)
    return sha256(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".build/m2-upload/m2-experiment.tar.gz")
    args = parser.parse_args()
    try:
        digest = package(args.output.absolute())
    except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError) as error:
        parser.exit(1, f"m2_package=FAILED reason={error}\n")
    print(f"archive={args.output}\narchive_bytes={args.output.stat().st_size}\narchive_sha256={digest}")


if __name__ == "__main__":
    main()
