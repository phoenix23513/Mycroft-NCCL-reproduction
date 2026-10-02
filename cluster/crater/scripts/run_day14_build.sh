#!/usr/bin/env bash
# Run in Pod-local storage; preserve only build products and evidence on the mount.
set -euo pipefail
if [[ $# -ne 0 || -z ${RESULT_DIR:-} ]]; then
  echo "Usage: RESULT_DIR=/path/to/new/persistent/results bash run_day14_build.sh" >&2
  exit 2
fi
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
build_dir="$project_root/.build/nccl"
result_dir=$(realpath -m -- "$RESULT_DIR")
if [[ -e "$result_dir" ]]; then
  echo "Result directory already exists; choose a new RESULT_DIR: $result_dir" >&2
  exit 2
fi
mkdir -p -- "$result_dir"
build_status=0
BUILDDIR="$build_dir" bash "$project_root/cluster/crater/scripts/build_nccl.sh" || build_status=$?
for artifact in include lib nccl_version build-manifest.txt build.log verification.log; do
  if [[ -e "$build_dir/$artifact" ]]; then
    cp -a -- "$build_dir/$artifact" "$result_dir/"
  fi
done
echo "build_exit_code=$build_status" > "$result_dir/run-status.txt"
echo "[Day 14] results_dir=$result_dir"
if [[ "$build_status" -ne 0 ]]; then
  echo "[Day 14] Build failed; available logs preserved, exit_code=$build_status" >&2
  exit "$build_status"
fi
# Confirm that the persistent copy can also load the expected library.
saved_status=0
"$result_dir/nccl_version" "$result_dir/lib/libnccl.so.2" \
  2>&1 | tee "$result_dir/saved-verification.log" || saved_status=$?
echo "saved_verification_exit_code=$saved_status" >> "$result_dir/run-status.txt"
if [[ "$saved_status" -ne 0 ]]; then
  echo "[Day 14] Persistent library loading failed" >&2
  exit "$saved_status"
fi
echo "[Day 14] Persistent library loading verified"
