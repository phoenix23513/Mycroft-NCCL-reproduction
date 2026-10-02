#!/usr/bin/env bash
# Build the uninstrumented, pinned NCCL baseline; run from any directory.
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
source_dir="$project_root/third_party/nccl"
expected_commit=ab2b89c4c339bd7f816fbc114a4b05d386b66290
expected_tag=v2.21.5-1
mode=${1:---build}
if [[ $# -gt 1 || ( "$mode" != --build && "$mode" != --check ) ]]; then
  echo "Usage: build_nccl.sh [--check|--build]" >&2
  echo "Environment: CUDA_HOME, CXX, JOBS, BUILDDIR, NVCC_GENCODE, NCCL_TRACE=0|1" >&2
  exit 2
fi
trace=${NCCL_TRACE:-0}
[[ "$trace" == 0 || "$trace" == 1 ]] || { echo "NCCL_TRACE must be 0 or 1" >&2; exit 2; }

blocked() { echo "[environment blocked] $*" >&2; exit 2; }
for tool in git make python3 realpath sha256sum readelf tee; do
  command -v "$tool" >/dev/null || blocked "Required tool missing: $tool"
done
[[ -f "$source_dir/src/nccl.h.in" ]] || blocked "Initialize third_party/nccl first"
source_commit=$(git -C "$source_dir" rev-parse HEAD)
if [[ "$source_commit" != "$expected_commit" ]] ||
   [[ $(git -C "$source_dir" rev-parse "refs/tags/$expected_tag^{commit}") != "$expected_commit" ]]; then
  echo "[source rejected] Expected $expected_tag at $expected_commit" >&2
  exit 1
fi
if [[ -n $(git -C "$source_dir" status --porcelain --untracked-files=all) ]]; then
  echo "[source rejected] Day 14 requires a clean, uninstrumented NCCL source tree" >&2
  exit 1
fi
echo "[source] tag=$expected_tag commit=$source_commit clean=yes"

cxx=${CXX:-g++}
command -v "$cxx" >/dev/null || blocked "C++ compiler missing: $cxx"
"$cxx" --version
if [[ -n ${CUDA_HOME:-} ]]; then
  cuda_home=$(realpath -m -- "$CUDA_HOME")
elif command -v nvcc >/dev/null; then
  cuda_home=$(dirname -- "$(dirname -- "$(realpath -- "$(command -v nvcc)")")")
else
  cuda_home=/usr/local/cuda
fi
[[ -x "$cuda_home/bin/nvcc" ]] || blocked "nvcc not found under CUDA_HOME=$cuda_home; use a CUDA devel environment"
[[ -f "$cuda_home/include/cuda_runtime.h" ]] || blocked "Missing CUDA development header cuda_runtime.h"
[[ -f "$cuda_home/lib64/libcudart_static.a" ]] || blocked "Missing CUDA development library libcudart_static.a"
"$cuda_home/bin/nvcc" --version

jobs=${JOBS:-2}
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be a positive integer" >&2; exit 2; }
# V100 machine code + forward-compatible PTX; change explicitly for other targets.
gencode=${NVCC_GENCODE:--gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70}
[[ -n "$gencode" ]] || { echo "NVCC_GENCODE must not be empty" >&2; exit 2; }
build_dir=$(realpath -m -- "${BUILDDIR:-$project_root/.build/nccl}")
if [[ "$build_dir" == "$source_dir" || "$build_dir" == "$source_dir/"* ]]; then
  echo "BUILDDIR must be outside the NCCL source tree" >&2
  exit 2
fi
# NCCL makefiles do not quote paths. Reject whitespace rather than misbuild.
if [[ "$source_dir$build_dir$cuda_home$cxx" =~ [[:space:]] ]]; then
  echo "NCCL make requires source, build, CUDA and compiler paths without whitespace" >&2
  exit 2
fi
echo "[configuration] jobs=$jobs build_dir=$build_dir"
echo "[configuration] NVCC_GENCODE=$gencode"
echo "[configuration] TRACE=$trace"
if [[ "$mode" == --check ]]; then
  echo "[preflight] PASS (no library built or loaded)"
  exit 0
fi
# Fresh output prevents old objects compiled with different settings being reused.
# No automatic deletion, especially when BUILDDIR is on Crater shared storage.
if [[ -e "$build_dir" ]]; then
  echo "Build directory already exists; choose a new BUILDDIR: $build_dir" >&2
  exit 2
fi
mkdir -p -- "$build_dir"
manifest="$build_dir/build-manifest.txt"
trap 'echo "status=FAILED exit_code=$?" >> "$manifest"' ERR
make_command=(make -C "$source_dir/src" -j "$jobs" lib
  "BUILDDIR=$build_dir" "CUDA_HOME=$cuda_home" "CUDA_INC=$cuda_home/include"
  "CUDA_LIB=$cuda_home/lib64" "CXX=$cxx" "NVCC_GENCODE=$gencode"
  DEBUG=0 "TRACE=$trace" ASAN=0 GCOV=0 KEEP=0 VERBOSE=1
  NVTX=1 PROFAPI=1 RDMA_CORE=0 CUDARTLIB=cudart_static ONLY_FUNCS=)
{
  echo "status=BUILDING"
  echo "source_tag=$expected_tag"
  echo "source_commit=$source_commit"
  echo "source_clean=yes"
  echo "cuda_home=$cuda_home"
  echo "build_dir=$build_dir"
  echo "jobs=$jobs"
  echo "trace=$trace"
  echo "NVCC_GENCODE=$gencode"
  echo "os:"
  cat /etc/os-release
  echo "kernel:"
  uname -sr
  echo "compiler:"
  "$cxx" --version
  echo "cuda:"
  "$cuda_home/bin/nvcc" --version
  echo "make:"
  make --version
  echo "python:"
  python3 --version
  printf 'command: '; printf '%q ' "${make_command[@]}"; printf '\n'
  echo "ambient make/compiler flags cleared: MAKEFLAGS MAKEOVERRIDES MFLAGS CXXFLAGS NVCUFLAGS LDFLAGS NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS"
} > "$manifest"
env -u MAKEFLAGS -u MAKEOVERRIDES -u MFLAGS -u CXXFLAGS -u NVCUFLAGS -u LDFLAGS \
  -u NVCC_PREPEND_FLAGS -u NVCC_APPEND_FLAGS \
  "${make_command[@]}" 2>&1 | tee "$build_dir/build.log"

"$cxx" -std=c++17 -Wall -Wextra -Wpedantic -Werror \
  "$project_root/cluster/crater/probes/nccl_version.cpp" -ldl \
  -o "$build_dir/nccl_version"
{
  echo "[library ELF metadata]"
  readelf -d "$build_dir/lib/libnccl.so.2.21.5"
  echo "[library SHA256]"
  sha256sum "$build_dir/lib/libnccl.so.2.21.5"
  echo "[actual loaded symbol and runtime version]"
  "$build_dir/nccl_version" "$build_dir/lib/libnccl.so.2"
} 2>&1 | tee "$build_dir/verification.log"
echo "status=PASS" >> "$manifest"
echo "[Day 14] Build and loading checks passed; evidence: $build_dir"
