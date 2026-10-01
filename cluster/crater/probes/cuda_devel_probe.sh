#!/usr/bin/env bash
# Standalone candidate-image probe: compile/link CUDA, without running on a GPU.
set -euo pipefail
if [[ $# -ne 0 ]]; then
  echo "Usage: cuda_devel_probe.sh (optional environment: CUDA_HOME, CXX, NVCC_GENCODE)" >&2
  exit 2
fi
echo "probe=day14_cuda_devel"
echo "gpu_required=no"
if [[ -r /etc/os-release ]]; then
  . /etc/os-release
  echo "os=${PRETTY_NAME:-unknown}"
fi
echo "kernel=$(uname -sr)"
cxx=${CXX:-g++}
missing=0
for tool in "$cxx" make git python3 realpath sha256sum readelf mktemp; do
  if command -v "$tool" >/dev/null; then
    echo "tool=$tool available=yes"
  else
    echo "missing=$tool"
    missing=1
  fi
done
if command -v "$cxx" >/dev/null; then
  compiler_version=$("$cxx" --version)
  echo "compiler=${compiler_version%%$'\n'*}"
fi
if command -v make >/dev/null; then
  make_version=$(make --version)
  echo "make=${make_version%%$'\n'*}"
fi
if command -v python3 >/dev/null; then python3 --version; fi

cuda_home=${CUDA_HOME:-/usr/local/cuda}
if [[ -z ${CUDA_HOME:-} ]] && command -v nvcc >/dev/null && command -v realpath >/dev/null; then
  cuda_home=$(dirname -- "$(dirname -- "$(realpath -- "$(command -v nvcc)")")")
fi
echo "cuda_home=$cuda_home"
for component in bin/nvcc include/cuda_runtime.h lib64/libcudart_static.a; do
  if [[ ! -f "$cuda_home/$component" ]]; then
    echo "missing=$component"
    missing=1
  fi
done
if [[ ! -x "$cuda_home/bin/nvcc" ]]; then
  echo "missing=executable_nvcc"
  missing=1
fi
if [[ "$missing" -ne 0 ]]; then
  echo "status=BLOCKED"
  exit 2
fi
"$cuda_home/bin/nvcc" --version
gencode=${NVCC_GENCODE:--gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70}
echo "NVCC_GENCODE=$gencode"
# Split flags into argv without eval or shell expansion.
read -r -a gencode_args <<< "$gencode"
probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/day14-cuda-probe.XXXXXX")
cleanup() {
  rm -f -- "$probe_dir/probe.cu" "$probe_dir/probe" "$probe_dir/nvcc.log"
  rmdir -- "$probe_dir" || true
}
trap cleanup EXIT
cat > "$probe_dir/probe.cu" <<'CUDA'
#include <cuda_runtime.h>
__global__ void probe_kernel(float* value) {
  if (threadIdx.x == 0) value[0] = 1.0f;
}
int main() {
  int version = 0;
  return static_cast<int>(cudaRuntimeGetVersion(&version));
}
CUDA
if ! env -u NVCC_PREPEND_FLAGS -u NVCC_APPEND_FLAGS \
  "$cuda_home/bin/nvcc" -ccbin "$cxx" -std=c++11 \
  "${gencode_args[@]}" --cudart static \
  "$probe_dir/probe.cu" -o "$probe_dir/probe" \
  > "$probe_dir/nvcc.log" 2>&1; then
  cat "$probe_dir/nvcc.log"
  echo "status=CUDA_COMPILE_FAILED"
  exit 1
fi
echo "cuda_compile_and_link=PASS"
echo "gpu_execution=NOT_RUN"
echo "nccl_build=NOT_RUN"
echo "status=READY_FOR_NCCL_BUILD"
