#!/usr/bin/env python3
"""M2 真实采集核对骨架（保留 Day16 路径）；未实现时明确失败。"""
import argparse
from pathlib import Path

from mycroft.schema import Event, EventKind, TimeDomain, validate_event


def validate_completion_line(line: str) -> Event:
    """复用已有 Event v2 契约；格式有效不代表来源或完成语义有效。"""
    event = Event.from_json(line)
    validate_event(event)
    if (event.schema_version != 2 or event.event_kind is not EventKind.OPERATION_COMPLETION
            or event.time.domain is not TimeDomain.NCCL_MONOTONIC_NS):
        raise ValueError("M2 requires Event v2 operation completion on NCCL monotonic time")
    return event


def verify_capture(directory: Path) -> None:
    # TODO(M2) 1：核对构建固定 commit、实际 patch hash、加载库 hash 和真实运行退出码。
    # TODO(M2) 2：核对 manifest 中的观测点、同进程 clock、实际 RING/SIMPLE NET/IB。
    # TODO(M2) 3：逐行调用 validate_completion_line，拒绝缺失、重复、错误大小和时序。
    # TODO(M2) 4：将 channel mapping、workload 数值结果及跨 rank 身份逐项对账。
    # TODO(M2) 5：核对普通串行操作、实际多 channel 与单操作计划；诊断必需记录丢失时拒绝验收。
    # TODO(M2) 6：核对周期 state、中间窗口、step/slice 单位及必要对端/接收证据。
    # Graph/grouped/完整 P2P 不受支持，不作为本轮验收任务。
    raise NotImplementedError("真实采集及 operation/completion 关联尚未实现")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="包含真实 trace、manifest 和 workload 日志的目录")
    args = parser.parse_args()
    try:
        verify_capture(args.directory)
    except NotImplementedError as error:
        parser.exit(2, f"capture_verification=NOT_IMPLEMENTED reason={error}\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"capture_verification=FAILED reason={error}\n")


if __name__ == "__main__":
    main()
