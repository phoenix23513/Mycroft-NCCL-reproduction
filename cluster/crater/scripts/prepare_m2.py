#!/usr/bin/env python3
"""校验上传包，解出固定 NCCL 原源码，在新副本上应用 M2 补丁。"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
COMMIT = "ab2b89c4c339bd7f816fbc114a4b05d386b66290"
ADAPTER = "instrumentation/nccl-2.21.5/day16/nccl_adapter/"
RECORDER = "instrumentation/nccl-2.21.5/day16/"
COPIES = {ADAPTER + "mycroft_m2.cc": "src/mycroft_m2.cc",
          ADAPTER + "mycroft_m2.h": "src/include/mycroft_m2.h",
          RECORDER + "src/operation_trace.cpp": "src/mycroft_operation_trace.cc",
          RECORDER + "include/operation_trace.h": "src/include/operation_trace.h"}


def sha256(path):
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("unsafe archive or manifest path")
    return path


def check_bundle(root):
    manifest = json.loads((root / "upload-manifest.json").read_text())
    if manifest.get("base_commit") != COMMIT or manifest.get("base_tag") != "v2.21.5-1":
        raise ValueError("unexpected NCCL base version")
    for name, expected in manifest["files"].items():
        safe_name(name)
        path = root / name
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"upload content differs: {name}")
    needed = {"nccl-source.tar", ADAPTER + "nccl-2.21.5-m2.patch", *COPIES}
    if not needed <= manifest["files"].keys():
        raise ValueError("upload source inputs missing")
    return manifest


def prepare(root, destination):
    manifest = check_bundle(root)
    # Existing source/build/result directories must never be reused silently.
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(root / "nccl-source.tar", "r:") as archive:
        members = archive.getmembers()
        for member in members:
            safe_name(member.name)
            if not (member.isdir() or member.isfile()):
                raise ValueError("NCCL source archive contains a link or special file")
        for member in members:
            path = destination / member.name
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, path.open("xb") as target:
                    shutil.copyfileobj(source, target)
                path.chmod(member.mode & 0o777)
    # Scratch Git metadata only makes patch paths unambiguous; it is not upstream provenance.
    subprocess.run(["git", "init", "-q", str(destination)], check=True)
    patch = (root / ADAPTER / "nccl-2.21.5-m2.patch").resolve()
    subprocess.run(["git", "-C", str(destination), "apply", "--check", str(patch)], check=True)
    subprocess.run(["git", "-C", str(destination), "apply", str(patch)], check=True)
    for original, target in COPIES.items():
        shutil.copyfile(root / original, destination / target)
    provenance = {"base_commit": COMMIT, "base_tag": manifest["base_tag"],
                  "base_tree": manifest["base_tree"], "source_archive_sha256": manifest["files"]["nccl-source.tar"],
                  "inputs": {name: manifest["files"][name] for name in [ADAPTER + "nccl-2.21.5-m2.patch", *COPIES]},
                  "patched_files": {name: sha256(destination / name) for name in
                    ("src/Makefile", "src/include/proxy.h", "src/enqueue.cc", "src/proxy.cc", "src/transport/net.cc", *COPIES.values())}}
    (destination / "m2-source-manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="新建的独立 NCCL 副本目录")
    args = parser.parse_args()
    try:
        prepare(ROOT, args.source_dir.absolute())
    except (OSError, ValueError, KeyError, tarfile.TarError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"m2_prepare=FAILED reason={error}\n")
    print(f"m2_prepare=PASS source_dir={args.source_dir}")


if __name__ == "__main__":
    main()
