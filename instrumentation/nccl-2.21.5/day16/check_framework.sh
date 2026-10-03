#!/usr/bin/env bash
# CPU recorder/schema checks; this is not NCCL build or GPU acceptance.
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$source_dir/../../.." && pwd)
"${CXX:-g++}" -std=c++17 -Wall -Wextra -Wpedantic -Werror \
  -I "$source_dir/include" -fsyntax-only "$source_dir/src/operation_trace.cpp"
PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 - "$source_dir/verify_capture.py" <<'PY'
from pathlib import Path
import importlib.util
import sys

path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("day16_verify_capture", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PY
python3 -m unittest discover -s "$source_dir/tests" -p 'test_*.py'
echo 'day16_framework=PASS scope=CPU_recorder_schema_and_contract_tests_only'
echo 'day16_capture=CPU_RECORDER_ONLY'
echo 'day16_nccl_adapter=PATCH_PRESENT_CUDA_BUILD_NOT_RUN'
echo 'day16_gpu_acceptance=NOT_RUN'
