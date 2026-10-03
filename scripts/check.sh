#!/usr/bin/env sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
build_dir="$project_root/.build"

echo "[check] Python smoke test"
PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m unittest discover -s "$project_root/tests/python" -p "test_*.py"

echo "[check] Versioned Event schema"
PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m unittest discover -s "$project_root/tests/schema" \
  -p "test_*.py"

echo "[check] Event trace recovery"
PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m unittest discover -s "$project_root/tests/trace" \
  -p "test_*.py"

echo "[check] C++ smoke build"
if command -v cmake >/dev/null 2>&1; then
  cmake -S "$project_root" -B "$build_dir"
  cmake --build "$build_dir"
  ctest --test-dir "$build_dir" --output-on-failure
else
  echo "[check] CMake not found; using the C++ compiler fallback"
  mkdir -p "$build_dir"
  "${CXX:-c++}" \
    -std=c++17 \
    -Wall \
    -Wextra \
    -Wpedantic \
    "$project_root/tests/cpp/smoke_test.cpp" \
    -o "$build_dir/mycroft_cpp_smoke"
  "$build_dir/mycroft_cpp_smoke"
fi

echo "[check] E02 progress state machine"
python3 -m unittest discover \
  -s "$project_root/experiments/e02_progress_state_machine/tests" \
  -p "test_*.py"

echo "[check] NCCL baseline guards (no CUDA or GPU required)"
python3 -m unittest discover \
  -s "$project_root/tests/nccl" -p "test_*.py"

echo "[check] E06 workload guards and result checks (no CUDA or GPU required)"
python3 -m unittest discover \
  -s "$project_root/workloads/minimal_allreduce/tests" -p "test_*.py"

echo "[check] Replay sanitized real Day15 evidence (offline; no fresh GPU run)"
python3 "$project_root/workloads/minimal_allreduce/verify_results.py" \
  "$project_root/results/samples/e06/day15/baseline"
python3 "$project_root/workloads/minimal_allreduce/verify_results.py" \
  "$project_root/results/samples/e06/day15/trace" \
  --expected-sha256 e4437dd0b48e3ab426b17b394043a0d162f1c9040b32bcf34b5b866e6286132a \
  --require-selection

echo "[check] Replay sanitized real M1 NET/IB evidence (offline; no fresh GPU run)"
python3 "$project_root/workloads/minimal_allreduce/verify_m1.py" \
  "$project_root/results/samples/e06/m1"

echo "[check] Day16 capture framework (CPU only; collection not implemented)"
bash "$project_root/instrumentation/nccl-2.21.5/day16/check_framework.sh"
python3 -m unittest discover \
  -s "$project_root/instrumentation/nccl-2.21.5/day16/tests" -p "test_*.py"

echo "[check] All repository checks passed"
