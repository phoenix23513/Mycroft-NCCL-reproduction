#!/usr/bin/env python3
"""将 Day16 必要结果打成一个压缩包；失败作业也可保存已有证据。"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile


FILES = (
    "run-status.txt",
    "environment.txt",
    "capture-manifest.json",
    "analysis.txt",
    "workload/run.log",
    "workload/rank0.log",
    "workload/rank1.log",
    "trace/rank0.jsonl",
    "trace/rank1.jsonl",
    "trace/channel-map.jsonl",
    "nccl-build/build-manifest.txt",
    "nccl-build/run-status.txt",
    "nccl-build/verification.log",
    "nccl-build/saved-verification.log",
)
BUILD_LOG_TAIL_BYTES = 128 * 1024
PACKAGE_ROOT = "day16-results"


def regular_file(root: Path, name: str) -> Path | None:
    path = root
    for component in Path(name).parts:
        path /= component
        if path.is_symlink():
            raise ValueError(f"结果文件或目录不能是符号链接：{name}")
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"结果项不是普通文件：{name}")
    return path


def portable(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(f"{PACKAGE_ROOT}/{name}")
    info.size = len(data)
    info.mode = 0o600
    archive.addfile(portable(info), io.BytesIO(data))


def package_results(directory: Path, output: Path) -> str:
    root = directory.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("结果路径必须是目录")
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("压缩包已存在；请选择新输出路径")
    if regular_file(root, "run-status.txt") is None:
        raise ValueError("缺少 run-status.txt，无法保留作业退出状态")
    files = {name: regular_file(root, name) for name in FILES}
    build_log = regular_file(root, "nccl-build/build.log")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".day16-bundle-",
                                     suffix=".partial", delete=False) as staging:
        staging_path = Path(staging.name)
    try:
        included = []
        with tarfile.open(staging_path, "w:gz") as archive:
            for name, path in files.items():
                if path is not None:
                    archive.add(path, f"{PACKAGE_ROOT}/{name}", recursive=False, filter=portable)
                    included.append(name)
            if build_log is not None:
                with build_log.open("rb") as stream:
                    stream.seek(max(0, build_log.stat().st_size - BUILD_LOG_TAIL_BYTES))
                    tail = stream.read(BUILD_LOG_TAIL_BYTES)
                add_bytes(archive, "nccl-build/build-tail.log", tail)
                included.append("nccl-build/build-tail.log")
            manifest = {"bundle_version": 1, "included_files": included,
                        "missing_files": [name for name, path in files.items() if path is None],
                        "build_log_policy": "last_128_KiB_only"}
            add_bytes(archive, "bundle-manifest.json",
                      (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
        # 原子发布且不覆盖；压缩失败时不会留下貌似完整的目标文件。
        output.hardlink_to(staging_path)
    finally:
        staging_path.unlink()
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        checksum = package_results(args.directory, args.output)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"result_bundle=FAILED reason={error}\n")
    print(f"result_bundle={args.output}")
    print(f"archive_sha256={checksum}")
    print("bundle_status=PACKAGED (job status is recorded inside run-status.txt)")


if __name__ == "__main__":
    main()
