#!/usr/bin/env bash
# Fresh TRACE library and real workload; compile only in Pod-local storage.
set -euo pipefail
if [[ $# -ne 0 || -z ${RESULT_DIR:-} ]]; then
  echo 'Usage: RESULT_DIR=<新持久化目录> bash run_day15_trace.sh' >&2
  exit 2
fi
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
result_dir=$(realpath -m -- "$RESULT_DIR")
if [[ -e "$result_dir" ]]; then
  echo "Result directory already exists; choose another RESULT_DIR: $result_dir" >&2
  exit 2
fi
# This combined experiment must not compile NCCL on the persistent mount.
if [[ "$project_root" != /tmp/* ]]; then
  echo 'Extract this experiment into /tmp before starting it' >&2
  exit 2
fi
if [[ -n ${NCCL_ALGO:-} || -n ${NCCL_PROTO:-} ]]; then
  echo 'Unset NCCL_ALGO and NCCL_PROTO for the automatic-selection experiment' >&2
  exit 2
fi
mkdir -p -- "$result_dir"
save_status() {
  local status=$?
  printf 'experiment_exit_code=%s\n' "$status" > "$result_dir/experiment-status.txt"
}
trap save_status EXIT

# Keep the verified Day14 library intact; this builds a separate library.
NCCL_TRACE=1 RESULT_DIR="$result_dir/nccl-trace" \
  bash "$project_root/cluster/crater/scripts/run_day14_build.sh"
export NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,COLL
NCCL_ROOT="$result_dir/nccl-trace" RESULT_DIR="$result_dir/allreduce" \
  bash "$project_root/cluster/crater/scripts/run_day15.sh"

library_hash=$(sha256sum "$result_dir/nccl-trace/lib/libnccl.so.2.21.5")
library_hash=${library_hash%% *}
python3 "$project_root/workloads/minimal_allreduce/verify_results.py" \
  "$result_dir/allreduce" --expected-sha256 "$library_hash" --require-selection \
  --csv "$result_dir/allreduce/operations.csv" \
  2>&1 | tee "$result_dir/analysis.txt"
