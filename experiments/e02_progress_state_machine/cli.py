"""Command-line entry point for the E02 normal progress simulation."""

from __future__ import annotations

import argparse
import json

from progress_sim import FAULT_TARGETS, FaultSpec, run_normal_progress, run_progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="simulate normal GPU/Proxy/Network progress"
    )
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=32)
    parser.add_argument("--demo-faults", action="store_true")
    parser.add_argument("--fault-start-tick", type=int, default=3)
    parser.add_argument("--fault-duration-ticks", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.demo_faults:
        for target_index, target in enumerate(FAULT_TARGETS):
            fault = FaultSpec(
                target=target,
                start_tick=args.fault_start_tick,
                duration_ticks=args.fault_duration_ticks,
            )
            snapshots = run_progress(
                total_chunks=args.chunks,
                max_ticks=args.max_ticks,
                fault=fault,
            )
            if target_index > 0:
                print()
            print(f"fault_target: {target}")
            print()
            print("tick  gpu_ready  rdma_transmitted  rdma_done")
            for snapshot in snapshots:
                print(
                    f"{snapshot.tick:<4}  "
                    f"{snapshot.gpu_ready:<9}  "
                    f"{snapshot.rdma_transmitted:<16}  "
                    f"{snapshot.rdma_done}"
                )
        return 0

    snapshots = run_normal_progress(
        total_chunks=args.chunks,
        max_ticks=args.max_ticks,
    )
    for snapshot in snapshots:
        print(json.dumps(snapshot.as_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
