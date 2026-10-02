#!/usr/bin/env bash
set -euo pipefail
if [[ -z ${NCCL_ROOT:-} || $# -gt 1 ]]; then
  echo 'Usage: NCCL_ROOT=<Day14产物目录> bash run.sh [iterations>=3]' >&2
  exit 2
fi
iterations=${1:-3}
if [[ ! $iterations =~ ^[0-9]+$ || ${#iterations} -gt 4 ]]; then
  echo 'iterations must be an integer from 3 to 1000' >&2
  exit 2
fi
iterations=$((10#$iterations))
if (( iterations < 3 || iterations > 1000 )); then
  echo 'iterations must be an integer from 3 to 1000' >&2
  exit 2
fi
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_dir=$(mktemp -d /tmp/day15-allreduce.XXXXXX)
echo "task_dir=$task_dir"
NCCL_ROOT="$NCCL_ROOT" BUILDDIR="$task_dir/build" bash "$source_dir/build.sh"

# 默认 INFO；单独的 TRACE=1 构建实验显式设置 NCCL_DEBUG=TRACE。
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,ENV,GRAPH,COLL}
unset NCCL_DEBUG_FILE
pids=()
stop_children() {
  if ((${#pids[@]})); then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap stop_children EXIT
for rank in 0 1; do
  timeout --signal=TERM --kill-after=5s 120s \
    "$task_dir/build/day15_allreduce" "$rank" "$task_dir/nccl.id" "$iterations" \
    > "$task_dir/rank$rank.log" 2>&1 &
  pids[$rank]=$!
done
status=0
for rank in 0 1; do
  rank_status=0
  wait "${pids[$rank]}" || rank_status=$?
  # 已经 wait 的 PID 不再由 EXIT trap 处理。
  unset "pids[$rank]"
  echo "rank=$rank exit_code=$rank_status"
  cat "$task_dir/rank$rank.log"
  if (( rank_status != 0 )); then status=1; fi
done
pids=()
exit "$status"
