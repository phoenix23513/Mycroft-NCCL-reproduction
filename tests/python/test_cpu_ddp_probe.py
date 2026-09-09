import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch, sentinel


PROBE_PATH = (
    Path(__file__).resolve().parents[2]
    / "cluster"
    / "crater"
    / "probes"
    / "cpu_ddp_probe.py"
)
SPEC = importlib.util.spec_from_file_location("cpu_ddp_probe", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load probe module from {PROBE_PATH}")
PROBE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)


def make_fake_torch() -> tuple[ModuleType, ModuleType, Mock]:
    tensor = Mock()
    tensor.item.return_value = 3.0

    torch_module = ModuleType("torch")
    torch_module.float32 = sentinel.float32
    torch_module.tensor = Mock(return_value=tensor)

    dist_module = ModuleType("torch.distributed")
    dist_module.ReduceOp = SimpleNamespace(SUM=sentinel.sum)
    dist_module.init_process_group = Mock()
    dist_module.all_reduce = Mock()
    dist_module.destroy_process_group = Mock()
    torch_module.distributed = dist_module
    return torch_module, dist_module, tensor


class DistributedContextTests(unittest.TestCase):
    def test_parses_valid_environment(self) -> None:
        context = PROBE.DistributedContext.from_environment(
            {
                "WORLD_SIZE": "2",
                "RANK": "1",
                "MASTER_ADDR": "master.example",
                "MASTER_PORT": "23456",
            }
        )

        self.assertEqual(context.world_size, 2)
        self.assertEqual(context.rank, 1)
        self.assertEqual(context.master_addr, "master.example")
        self.assertEqual(context.master_port, 23456)

    def test_reports_missing_variable(self) -> None:
        with self.assertRaisesRegex(PROBE.ConfigurationError, "MASTER_ADDR"):
            PROBE.DistributedContext.from_environment(
                {"WORLD_SIZE": "2", "RANK": "0", "MASTER_PORT": "23456"}
            )

    def test_rejects_rank_outside_world(self) -> None:
        with self.assertRaisesRegex(PROBE.ConfigurationError, "0 <= RANK"):
            PROBE.DistributedContext.from_environment(
                {
                    "WORLD_SIZE": "2",
                    "RANK": "2",
                    "MASTER_ADDR": "master.example",
                    "MASTER_PORT": "23456",
                }
            )

    def test_rejects_invalid_port(self) -> None:
        with self.assertRaisesRegex(PROBE.ConfigurationError, "between 1 and 65535"):
            PROBE.DistributedContext.from_environment(
                {
                    "WORLD_SIZE": "2",
                    "RANK": "0",
                    "MASTER_ADDR": "master.example",
                    "MASTER_PORT": "70000",
                }
            )


class AllReduceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = PROBE.DistributedContext(
            world_size=2,
            rank=1,
            master_addr="master.example",
            master_port=23456,
        )

    def test_runs_gloo_sum_and_cleans_up(self) -> None:
        torch_module, dist_module, tensor = make_fake_torch()

        with patch.dict(
            sys.modules,
            {"torch": torch_module, "torch.distributed": dist_module},
        ):
            result = PROBE.run_all_reduce(self.context)

        dist_module.init_process_group.assert_called_once_with(
            backend="gloo",
            init_method="tcp://master.example:23456",
            world_size=2,
            rank=1,
        )
        torch_module.tensor.assert_called_once_with(
            [2], dtype=torch_module.float32
        )
        dist_module.all_reduce.assert_called_once_with(
            tensor, op=dist_module.ReduceOp.SUM
        )
        dist_module.destroy_process_group.assert_called_once_with()
        self.assertEqual(result, 3.0)

    def test_cleans_up_when_collective_raises(self) -> None:
        torch_module, dist_module, _ = make_fake_torch()
        dist_module.all_reduce.side_effect = RuntimeError("collective failed")

        with patch.dict(
            sys.modules,
            {"torch": torch_module, "torch.distributed": dist_module},
        ):
            with self.assertRaisesRegex(RuntimeError, "collective failed"):
                PROBE.run_all_reduce(self.context)

        dist_module.destroy_process_group.assert_called_once_with()

if __name__ == "__main__":
    unittest.main()
