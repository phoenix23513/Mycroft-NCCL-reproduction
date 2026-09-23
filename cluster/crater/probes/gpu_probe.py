"""Minimal single-GPU runtime probe for Crater Day 04."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any


def collect_gpu_report(
    torch_module: Any, environment: Mapping[str, str]
) -> dict[str, object]:
    """Validate one CUDA device and return a JSON-serializable report.

    Contract:
    - reject a runtime where CUDA is unavailable;
    - reject a runtime where no CUDA device is visible;
    - create [1.0, 2.0, 3.0] and [4.0, 5.0, 6.0] as float32 tensors
      on the CUDA device;
    - add the tensors, sum the result, and convert it to a Python float;
    - synchronize the CUDA device before returning;
    - report status, PyTorch/CUDA versions, visible device count and name,
      CUDA_VISIBLE_DEVICES, and the numeric result.
    """
    cuda_available = torch_module.cuda.is_available()
    if not cuda_available:
        raise RuntimeError("CUDA is not available")

    device_count = torch_module.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("no CUDA device is visible")

    gpu_name = torch_module.cuda.get_device_name(0)
    tensor1 = torch_module.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch_module.float32,
        device="cuda",
    )
    tensor2 = torch_module.tensor(
        [4.0, 5.0, 6.0],
        dtype=torch_module.float32,
        device="cuda",
    )
    total = (tensor1 + tensor2).sum()
    torch_module.cuda.synchronize()
    result = float(total.item())

    report = {
        "status": "ok",
        "torch_version": torch_module.__version__,
        "torch_cuda_version": torch_module.version.cuda,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_name": gpu_name,
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES", ""),
        "result": result,
    }

    return report


def main() -> int:
    import torch

    report = collect_gpu_report(torch, os.environ)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
