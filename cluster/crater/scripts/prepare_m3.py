#!/usr/bin/env python3
"""在独立 M2 源码副本上应用 M3 增量补丁；保留固定上游来源。"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from prepare_m2 import ROOT, check_bundle, prepare as prepare_m2, sha256

M3 = "instrumentation/nccl-2.21.5/m3/"
PATCH = M3 + "nccl_adapter/nccl-2.21.5-m3.patch"
COPIES = {M3 + "nccl_adapter/mycroft_m3.cc": "src/mycroft_m3.cc",
          M3 + "nccl_adapter/mycroft_m3.h": "src/include/mycroft_m3.h",
          M3 + "include/send_delay.h": "src/include/mycroft_m3_send_delay.h"}


def prepare(root, destination):
    manifest = check_bundle(root)
    if not {PATCH, *COPIES} <= manifest["files"].keys():
        raise ValueError("M3 source inputs missing")
    proof = prepare_m2(root, destination)
    patch = (root / PATCH).resolve()
    subprocess.run(["git", "-C", str(destination), "apply", "--check", str(patch)], check=True)
    subprocess.run(["git", "-C", str(destination), "apply", str(patch)], check=True)
    for source, target in COPIES.items():
        shutil.copyfile(root / source, destination / target)
    proof["inputs"].update({name: manifest["files"][name] for name in (PATCH, *COPIES)})
    proof["patched_files"].update({name: sha256(destination / name)
                                    for name in ("src/Makefile", "src/transport/net.cc", *COPIES.values())})
    (destination / "m2-source-manifest.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare(ROOT, args.source_dir.absolute())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"m3_prepare=FAILED reason={error}\n")
    print(f"m3_prepare=PASS source_dir={args.source_dir}")


if __name__ == "__main__":
    main()
