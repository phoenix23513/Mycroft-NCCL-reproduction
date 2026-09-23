import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, sentinel


PROBE_PATH = (
    Path(__file__).resolve().parents[2]
    / "cluster"
    / "crater"
    / "probes"
    / "gpu_probe.py"
)
SPEC = importlib.util.spec_from_file_location("gpu_probe", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load probe module from {PROBE_PATH}")
PROBE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)


def make_fake_torch(
    *, cuda_available: bool = True, device_count: int = 1
) -> tuple[SimpleNamespace, MagicMock, MagicMock]:
    left = MagicMock(name="left_tensor")
    right = MagicMock(name="right_tensor")
    added = MagicMock(name="added_tensor")
    total = Mock(name="total_tensor")
    total.item.return_value = 21.5
    left.__add__.return_value = added
    added.sum.return_value = total

    cuda = SimpleNamespace(
        is_available=Mock(return_value=cuda_available),
        device_count=Mock(return_value=device_count),
        get_device_name=Mock(return_value="Fake GPU"),
        synchronize=Mock(),
    )
    torch_module = SimpleNamespace(
        __version__="2.5.1",
        version=SimpleNamespace(cuda="11.8"),
        cuda=cuda,
        float32=sentinel.float32,
        tensor=Mock(side_effect=[left, right]),
    )
    return torch_module, left, added


class GpuProbeContractTests(unittest.TestCase):
    def test_collects_single_gpu_report_after_real_tensor_work(self) -> None:
        torch_module, left, added = make_fake_torch()

        report = PROBE.collect_gpu_report(
            torch_module, {"CUDA_VISIBLE_DEVICES": "GPU-test-uuid"}
        )

        self.assertEqual(
            report,
            {
                "status": "ok",
                "torch_version": "2.5.1",
                "torch_cuda_version": "11.8",
                "cuda_available": True,
                "device_count": 1,
                "device_name": "Fake GPU",
                "cuda_visible_devices": "GPU-test-uuid",
                "result": 21.5,
            },
        )
        self.assertEqual(
            torch_module.tensor.call_args_list,
            [
                call(
                    [1.0, 2.0, 3.0],
                    device="cuda",
                    dtype=torch_module.float32,
                ),
                call(
                    [4.0, 5.0, 6.0],
                    device="cuda",
                    dtype=torch_module.float32,
                ),
            ],
        )
        left.__add__.assert_called_once()
        added.sum.assert_called_once_with()
        torch_module.cuda.get_device_name.assert_called_once_with(0)
        torch_module.cuda.synchronize.assert_called_once_with()

    def test_rejects_runtime_without_cuda(self) -> None:
        torch_module, _, _ = make_fake_torch(cuda_available=False)

        with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
            PROBE.collect_gpu_report(torch_module, {})

        torch_module.tensor.assert_not_called()
        torch_module.cuda.synchronize.assert_not_called()

    def test_rejects_runtime_without_visible_device(self) -> None:
        torch_module, _, _ = make_fake_torch(device_count=0)

        with self.assertRaisesRegex(RuntimeError, "no CUDA device is visible"):
            PROBE.collect_gpu_report(torch_module, {})

        torch_module.tensor.assert_not_called()
        torch_module.cuda.synchronize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
