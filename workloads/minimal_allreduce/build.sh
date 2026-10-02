#!/usr/bin/env bash
# Native C++ API caller: CUDA kernels are provided by the NCCL shared library.
set -euo pipefail
if [[ $# -ne 0 || -z ${NCCL_ROOT:-} || -z ${BUILDDIR:-} ]]; then
  echo 'Usage: NCCL_ROOT=<Day14产物目录> BUILDDIR=<新目录> bash build.sh' >&2
  exit 2
fi
blocked() { echo "[build blocked] $*" >&2; exit 2; }
cxx=${CXX:-g++}
command -v "$cxx" >/dev/null || blocked "C++ compiler missing: $cxx"
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -n ${CUDA_HOME:-} ]]; then
  cuda_home=$(realpath -m -- "$CUDA_HOME")
elif command -v nvcc >/dev/null; then
  cuda_home=$(dirname -- "$(dirname -- "$(realpath -- "$(command -v nvcc)")")")
else
  cuda_home=/usr/local/cuda
fi
[[ -f "$cuda_home/include/cuda_runtime.h" ]] || blocked "Missing CUDA header under CUDA_HOME=$cuda_home"
if [[ -f "$cuda_home/lib64/libcudart.so" ]]; then
  cudart="$cuda_home/lib64/libcudart.so"
  cudart_link=shared
elif [[ -f "$cuda_home/lib64/libcudart_static.a" ]]; then
  cudart="$cuda_home/lib64/libcudart_static.a"
  cudart_link=static
else
  blocked "Missing CUDA runtime development library under CUDA_HOME=$cuda_home"
fi
nccl_root=$(realpath -e -- "$NCCL_ROOT") || blocked "NCCL_ROOT does not exist"
[[ -f "$nccl_root/include/nccl.h" ]] || blocked "Missing Day14 NCCL header"
[[ -f "$nccl_root/lib/libnccl.so.2.21.5" && -f "$nccl_root/lib/libnccl.so.2" ]] ||
  blocked "Missing Day14 NCCL library or libnccl.so.2 link"
nccl_library=$(realpath -e -- "$nccl_root/lib/libnccl.so.2.21.5")
[[ $(realpath -e -- "$nccl_root/lib/libnccl.so.2") == "$nccl_library" ]] ||
  blocked "libnccl.so.2 must resolve to the specified 2.21.5 library"
build_dir=$(realpath -m -- "$BUILDDIR")
[[ ! -e "$build_dir" ]] || blocked "Build directory already exists; choose another BUILDDIR"
mkdir -p -- "$build_dir"
command=("$cxx" -std=c++17 -O2 -Wall -Wextra -Wpedantic -pthread
  -I "$source_dir/include" -I "$nccl_root/include" -I "$cuda_home/include"
  "-DDAY15_EXPECTED_NCCL_LIBRARY=\"$nccl_library\""
  "$source_dir/src/main.cpp" "$source_dir/src/bootstrap.cpp"
  "$nccl_library" "$cudart" -ldl -lrt
  "-Wl,-rpath,$nccl_root/lib" "-Wl,-rpath,$cuda_home/lib64"
  -o "$build_dir/day15_allreduce")
echo '[build] method=g++ (CMake is optional)'
echo "[build] cuda_home=$cuda_home cudart_link=$cudart_link"
printf 'compile_command: '
printf '%q ' "${command[@]}"
printf '\n'
"${command[@]}"
