"""计划日志解析单元测试；片段是构造输入，不是 GPU 运行或验收证据。"""
import importlib.util
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_results", ROOT / "verify_results.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class SelectionTests(unittest.TestCase):
    def fixture(self):
        lines = []
        for op in range(9):
            count = VERIFY.COUNTS[op // 3]
            path = "CBDColl" if op % 2 else "tunedColl"
            lines += [f"rank=0 operation={op} count={count} stage=api_begin time_ns=0",
                      f"unit_fixture NCCL TRACE {path} enqueue coll "
                      f"AllReduce(ncclSum, ncclFloat32, RING, LL), "
                      f"nChannels 1, count {count} (nbytes {count * 4}), unused fields",
                      f"rank=0 operation={op} count={count} stage=api_return time_ns=1",
                      f"rank=0 operation={op} count={count} stage=stream_sync_return time_ns=2"]
        return "\n".join(lines) + "\n"

    def parse(self, log):
        stages = list(re.finditer(
            r"^rank=0 operation=\d+ count=\d+ stage=\w+ time_ns=\d+$", log, re.MULTILINE))
        return VERIFY.selection_records(log, stages, 3)

    def test_both_plan_paths_have_operation_mapping(self):
        records = self.parse(self.fixture())
        self.assertEqual(len(records), 9)
        self.assertEqual(records[1]["selection_path"], "CBDColl")
        self.assertEqual(records[8]["algorithm"], "RING")
        self.assertEqual(records[8]["protocol"], "LL")

    def test_missing_selection_rejected(self):
        lines = self.fixture().splitlines(keepends=True)
        del lines[1]
        with self.assertRaisesRegex(ValueError, "missing, duplicate"):
            self.parse("".join(lines))

    def test_duplicate_selection_rejected(self):
        log = self.fixture()
        with self.assertRaisesRegex(ValueError, "missing, duplicate"):
            self.parse(log + log.splitlines()[1] + "\n")

    def test_selection_after_api_return_rejected(self):
        lines = self.fixture().splitlines(keepends=True)
        lines[1], lines[2] = lines[2], lines[1]
        with self.assertRaisesRegex(ValueError, "outside its API call"):
            self.parse("".join(lines))

    def test_wrong_size_rejected(self):
        with self.assertRaisesRegex(ValueError, "size or channels"):
            self.parse(self.fixture().replace("nbytes 16", "nbytes 15", 1))

    def test_unknown_protocol_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown algorithm or protocol"):
            self.parse(self.fixture().replace("RING, LL", "RING, Unknown", 1))

    def test_zero_channels_rejected(self):
        with self.assertRaisesRegex(ValueError, "size or channels"):
            self.parse(self.fixture().replace("nChannels 1", "nChannels 0", 1))


if __name__ == "__main__":
    unittest.main()
