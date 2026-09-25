#!/usr/bin/env python3
"""Black-box checks for the E01 JSONL trace."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from copy import deepcopy


REQUIRED_KEYS = {
    "op_seq",
    "rank",
    "channel",
    "phase",
    "step",
    "chunk",
    "action",
    "peer",
    "timestamp",
    "value",
    "contributor_mask",
}
PHASES = ("reduce_scatter", "all_gather")
EVENTS_PER_STEP = 16
EXPECTED_EVENT_COUNT = 96


def validate_trace(events: list[dict[str, object]]) -> None:
    assert len(events) == EXPECTED_EVENT_COUNT
    assert [event["timestamp"] for event in events] == list(
        range(EXPECTED_EVENT_COUNT)
    )

    counts: Counter[tuple[object, object, object]] = Counter()
    phase_step_order: list[tuple[int, int]] = []

    for event in events:
        assert REQUIRED_KEYS <= event.keys()
        assert event["phase"] in PHASES
        assert event["action"] in ("send", "recv")
        assert event["op_seq"] == 0
        assert event["channel"] == 0
        assert 0 <= int(event["rank"]) < 4
        assert 0 <= int(event["peer"]) < 4
        assert 0 <= int(event["chunk"]) < 8
        assert 0 <= int(event["step"]) < 3

        phase_index = PHASES.index(str(event["phase"]))
        phase_step_order.append((phase_index, int(event["step"])))
        counts[(event["phase"], event["step"], event["action"])] += 1

    assert phase_step_order == sorted(phase_step_order)
    for phase in PHASES:
        for step in range(3):
            assert counts[(phase, step, "send")] == 8
            assert counts[(phase, step, "recv")] == 8

    for send_index, send in enumerate(events):
        if send["action"] != "send":
            continue
        matches = [
            (recv_index, recv)
            for recv_index, recv in enumerate(events)
            if recv["phase"] == send["phase"]
            and recv["step"] == send["step"]
            and recv["chunk"] == send["chunk"]
            and recv["action"] == "recv"
            and recv["rank"] == send["peer"]
            and recv["peer"] == send["rank"]
        ]
        assert len(matches) == 1
        recv_index, recv = matches[0]
        assert send_index < recv_index
        if send["phase"] == "all_gather":
            assert recv["value"] == send["value"]
            assert recv["contributor_mask"] == send["contributor_mask"]


def expect_invalid(events: list[dict[str, object]]) -> None:
    try:
        validate_trace(events)
    except AssertionError:
        return
    raise AssertionError("mutated trace unexpectedly passed validation")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} E01_BINARY", file=sys.stderr)
        return 2

    completed = subprocess.run(
        [sys.argv[1], "--jsonl"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr, end="")
        return 1

    events = [json.loads(line) for line in completed.stdout.splitlines()]
    validate_trace(events)

    missing = deepcopy(events)
    missing.pop(0)
    expect_invalid(missing)

    reordered = deepcopy(events)
    first_send = reordered[0]
    recv_index = next(
        index
        for index, event in enumerate(reordered)
        if event["phase"] == first_send["phase"]
        and event["step"] == first_send["step"]
        and event["chunk"] == first_send["chunk"]
        and event["action"] == "recv"
        and event["rank"] == first_send["peer"]
        and event["peer"] == first_send["rank"]
    )
    reordered[0], reordered[recv_index] = reordered[recv_index], reordered[0]
    expect_invalid(reordered)

    print("[PASS] JSONL schema and dependency consistency")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
