"""Command-line entry point for the E02 normal progress simulation."""

from __future__ import annotations

import argparse
import json

from progress_sim import run_normal_progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="simulate normal GPU/Proxy/Network progress"
    )
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshots = run_normal_progress(
        total_chunks=args.chunks,
        max_ticks=args.max_ticks,
    )
    for snapshot in snapshots:
        print(json.dumps(snapshot.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
