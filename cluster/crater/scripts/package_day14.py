#!/usr/bin/env python3
"""Package pinned NCCL source and Day 14 tools for offline Windows upload."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[3]
COMMIT = "ab2b89c4c339bd7f816fbc114a4b05d386b66290"
TAG = "v2.21.5-1"
PACKAGE_ROOT = "day14-nccl-build"
FILES = (
    "LICENSE",
    "cluster/crater/scripts/build_nccl.sh",
    "cluster/crater/scripts/run_day14_build.sh",
    "cluster/crater/probes/nccl_version.cpp",
    "cluster/crater/probes/cuda_devel_probe.sh",
    "instrumentation/nccl-2.21.5/README.md",
)


def git(source, *arguments):
    return subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def package(output, package_root=PACKAGE_ROOT, files=FILES):
    source = ROOT / "third_party/nccl"
    if git(source, "rev-parse", "HEAD") != COMMIT:
        raise RuntimeError("NCCL source is not at the pinned commit")
    if git(source, "rev-parse", f"refs/tags/{TAG}^{{commit}}") != COMMIT:
        raise RuntimeError("NCCL tag does not match the pinned commit")
    if git(source, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("NCCL source has modifications")
    if output.exists() or output.with_name(output.name + ".sha256").exists():
        raise RuntimeError("Upload archive already exists; choose another --output")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="day14-package-") as directory:
        clone = Path(directory) / "nccl"
        # A depth-one local file-protocol clone includes the fixed tree, not history.
        # --no-local avoids hard links and references back to the user's repository.
        subprocess.run(
            ["git", "-c", "protocol.file.allow=always", "clone", "--quiet",
             "--no-local", "--depth", "1", "--branch", TAG,
             source.as_uri(), str(clone)],
            check=True, capture_output=True, text=True,
        )
        if git(clone, "rev-parse", "HEAD") != COMMIT:
            raise RuntimeError("Packaged source does not match the pinned commit")
        (clone / ".git/config").write_text(
            "[core]\n\trepositoryformatversion = 0\n\tfilemode = true\n\tbare = false\n"
        )
        if (clone / ".git/objects/info/alternates").exists():
            raise RuntimeError("Packaged Git objects must be self-contained")
        metadata = ("HEAD", "index", "shallow", "config", "objects", "refs", "packed-refs")
        tracked = git(clone, "ls-files", "-z").split("\0")

        def portable(info):
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            return info

        with tarfile.open(output, "x:gz") as archive:
            for name in files:
                archive.add(ROOT / name, f"{package_root}/{name}", filter=portable)
            for name in tracked:
                if name:
                    archive.add(clone / name, f"{package_root}/third_party/nccl/{name}",
                                recursive=False, filter=portable)
            # Exclude reflogs, hooks, local remotes and all personal project Git data.
            for name in metadata:
                path = clone / ".git" / name
                if path.exists():
                    archive.add(path, f"{package_root}/third_party/nccl/.git/{name}",
                                filter=portable)
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{checksum}  {output.name}\n")
    print(f"archive={output}")
    print(f"archive_bytes={output.stat().st_size}")
    print(f"sha256={checksum}")
    print(f"source_commit={COMMIT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / ".build/day14-upload/day14-nccl-build.tar.gz")
    arguments = parser.parse_args()
    try:
        package(arguments.output.resolve())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Packaging failed: {error}\n")


if __name__ == "__main__":
    main()
