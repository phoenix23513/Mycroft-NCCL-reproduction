"""E02 normal progress model for GPU, Proxy, and Network actors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ProgressSnapshot:
    """Accumulated chunk progress observed at the end of one logical tick."""

    tick: int
    gpu_ready: int
    rdma_transmitted: int
    rdma_done: int

    def as_dict(self) -> dict[str, int]:
        return {
            "tick": self.tick,
            "gpu_ready": self.gpu_ready,
            "rdma_transmitted": self.rdma_transmitted,
            "rdma_done": self.rdma_done,
        }


class ProgressActor(Protocol):
    """One logical executor that advances exactly one progress counter."""

    name: str

    def advance(
        self, previous: ProgressSnapshot, total_chunks: int
    ) -> int:
        """Return this actor's accumulated counter for the next tick."""


class GPUProducer:
    name = "gpu_producer"

    def advance(
        self, previous: ProgressSnapshot, total_chunks: int
    ) -> int:
        """Prepare at most one new chunk without exceeding total_chunks."""
        if previous.gpu_ready < total_chunks:
            return previous.gpu_ready + 1
        return previous.gpu_ready


class ProxyTransmitter:
    name = "proxy_transmitter"

    def advance(
        self, previous: ProgressSnapshot, total_chunks: int
    ) -> int:
        """Transmit at most one chunk that was GPU-ready last tick."""
        if previous.rdma_transmitted < previous.gpu_ready:
            return previous.rdma_transmitted + 1
        return previous.rdma_transmitted


class NetworkCompleter:
    name = "network_completer"

    def advance(
        self, previous: ProgressSnapshot, total_chunks: int
    ) -> int:
        """Complete at most one chunk transmitted before this tick."""
        if previous.rdma_done < previous.rdma_transmitted:
            return previous.rdma_done + 1
        return previous.rdma_done


def validate_snapshot(
    snapshot: ProgressSnapshot, total_chunks: int
) -> None:
    """Reject states that violate the physical progress ordering."""
    if not (
        0
        <= snapshot.rdma_done
        <= snapshot.rdma_transmitted
        <= snapshot.gpu_ready
        <= total_chunks
    ):
        raise ValueError(f"invalid progress state: {snapshot}")


def run_normal_progress(
    total_chunks: int = 4, max_ticks: int = 32
) -> list[ProgressSnapshot]:
    """Run the three-stage pipeline until every chunk is complete."""
    if total_chunks <= 0:
        raise ValueError("total_chunks must be positive")
    if max_ticks <= 0:
        raise ValueError("max_ticks must be positive")

    actors: tuple[ProgressActor, ProgressActor, ProgressActor] = (
        GPUProducer(),
        ProxyTransmitter(),
        NetworkCompleter(),
    )
    snapshots = [
        ProgressSnapshot(
            tick=0,
            gpu_ready=0,
            rdma_transmitted=0,
            rdma_done=0,
        )
    ]

    for tick in range(1, max_ticks + 1):
        previous = snapshots[-1]
        current = ProgressSnapshot(
            tick=tick,
            gpu_ready=actors[0].advance(previous, total_chunks),
            rdma_transmitted=actors[1].advance(previous, total_chunks),
            rdma_done=actors[2].advance(previous, total_chunks),
        )
        validate_snapshot(current, total_chunks)
        snapshots.append(current)

        if current.rdma_done == total_chunks:
            return snapshots

    raise RuntimeError("normal progress did not finish within max_ticks")
