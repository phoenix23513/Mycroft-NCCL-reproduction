#!/usr/bin/env bash
# Run the existing Day14 library on two GPUs; preserve evidence on the mount.
set -euo pipefail
if [[ $# -ne 0 || -z ${NCCL_ROOT:-} || -z ${RESULT_DIR:-} ]]; then
  echo 'Usage: NCCL_ROOT=<Day14产物目录> RESULT_DIR=<新持久化目录> bash run_day15.sh' >&2
  exit 2
fi
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
result_dir=$(realpath -m -- "$RESULT_DIR")
if [[ -e "$result_dir" ]]; then
  echo "Result directory already exists; choose another RESULT_DIR: $result_dir" >&2
  exit 2
fi
mkdir -p -- "$result_dir"
save_status() {
  local status=$?
  printf 'job_exit_code=%s\n' "$status" > "$result_dir/run-status.txt"
}
trap save_status EXIT

export NCCL_ROOT
NCCL_ROOT=$(realpath -e -- "$NCCL_ROOT")
if [[ ! -f "$NCCL_ROOT/include/nccl.h" || ! -f "$NCCL_ROOT/lib/libnccl.so.2.21.5" ]]; then
  echo 'NCCL_ROOT lacks the Day14 headers or shared library' >&2
  exit 2
fi
for tool in g++ python3 timeout sha256sum nvidia-smi; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required tool: $tool" >&2
    exit 2
  fi
done
{
  echo 'verification_scope=real_two_gpu_functional_baseline'
  echo "nccl_root=$NCCL_ROOT"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "NCCL_DEBUG=${NCCL_DEBUG:-INFO}"
  echo "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,ENV,GRAPH,COLL}"
  echo "NCCL_ALGO=${NCCL_ALGO:-<unset>}"
  echo "NCCL_PROTO=${NCCL_PROTO:-<unset>}"
  echo 'os:'
  cat /etc/os-release
  echo 'kernel:'
  uname -sr
  echo 'compiler:'
  g++ --version
  echo 'cmake:'
  if command -v cmake >/dev/null 2>&1; then
    cmake --version || echo 'cmake_version_unavailable (optional)'
  else
    echo 'not installed (optional; workload builds directly with g++)'
  fi
  echo 'GPU inventory:'
  nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version --format=csv
  echo 'GPU topology:'
  nvidia-smi topo -m
  echo 'NCCL library SHA256:'
  sha256sum "$NCCL_ROOT/lib/libnccl.so.2.21.5"
} > "$result_dir/environment.txt" 2>&1
if [[ -f "$NCCL_ROOT/build-manifest.txt" ]]; then
  cp -- "$NCCL_ROOT/build-manifest.txt" "$result_dir/nccl-build-manifest.txt"
fi

status=0
bash "$project_root/workloads/minimal_allreduce/run.sh" 3 \
  2>&1 | tee "$result_dir/run.log" || status=$?

# run.sh 在任何 rank 失败时仍会打印自己的 task_dir；复制已生成的独立日志。
python3 - "$result_dir" <<'PY'
from pathlib import Path
import re
import shutil
import sys

result = Path(sys.argv[1])
match = re.search(r"^task_dir=(/tmp/day15-allreduce\.[A-Za-z0-9]+)$",
                  (result / "run.log").read_text(), re.MULTILINE)
if match:
    task = Path(match.group(1))
    for name in ("rank0.log", "rank1.log"):
        source = task / name
        if source.is_file():
            shutil.copyfile(source, result / name)
PY
exit "$status"
