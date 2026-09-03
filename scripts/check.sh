#!/usr/bin/env sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
build_dir="$project_root/.build"

echo "[check] Python smoke test"
PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m unittest discover -s "$project_root/tests/python" -p "test_*.py"

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

echo "[check] All Day 01 checks passed"
