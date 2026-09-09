"""Minimal two-process CPU collective probe for Crater Day 03."""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass


class ConfigurationError(ValueError):
    """Raised when Crater's distributed environment is incomplete or invalid."""


@dataclass(frozen=True)
class DistributedContext:
    """Validated process-group settings injected by the platform."""

    world_size: int
    rank: int
    master_addr: str
    master_port: int

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> "DistributedContext":
        required = ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT")
        missing = [name for name in required if not environment.get(name)]
        if missing:
            raise ConfigurationError(
                "missing distributed environment variables: " + ", ".join(missing)
            )

        try:
            world_size = int(environment["WORLD_SIZE"])
            rank = int(environment["RANK"])
            master_port = int(environment["MASTER_PORT"])
        except ValueError as error:
            raise ConfigurationError(
                "WORLD_SIZE, RANK, and MASTER_PORT must be integers"
            ) from error

        if world_size < 1:
            raise ConfigurationError("WORLD_SIZE must be positive")
        if not 0 <= rank < world_size:
            raise ConfigurationError("RANK must satisfy 0 <= RANK < WORLD_SIZE")
        if not 1 <= master_port <= 65535:
            raise ConfigurationError("MASTER_PORT must be between 1 and 65535")

        return cls(
            world_size=world_size,
            rank=rank,
            master_addr=environment["MASTER_ADDR"],
            master_port=master_port,
        )


def run_all_reduce(context: DistributedContext) -> float:
    """Run one Gloo sum and return the scalar visible on this rank.

    Contract:
    - initialize a Gloo process group from ``context``;
    - use ``rank + 1`` as this rank's initial scalar value;
    - sum the values with one all-reduce operation;
    - always destroy the process group before returning or raising;
    - return the reduced scalar as a Python ``float``.
    """

    import torch
    import torch.distributed as dist

    rendezvous_url = f"tcp://{context.master_addr}:{context.master_port}"
    dist.init_process_group(
        backend="gloo",
        init_method=rendezvous_url,
        world_size=context.world_size,
        rank=context.rank,
    )
    try:
        initial_value = context.rank + 1
        tensor = torch.tensor([initial_value], dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        result = float(tensor.item())
        return result
    finally:
        dist.destroy_process_group()


def main() -> int:
    context = DistributedContext.from_environment(os.environ)
    result = run_all_reduce(context)
    print(
        json.dumps(
            {
                "hostname": socket.gethostname(),
                "rank": context.rank,
                "result": result,
                "world_size": context.world_size,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
