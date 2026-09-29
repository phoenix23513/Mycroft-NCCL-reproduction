"""Classify E02 snapshots using Mycroft-style progress states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from progress_sim import ProgressSnapshot, validate_snapshot


class ProgressState(str, Enum):
    """Snapshot states adapted from Mycroft's progress-state table."""

    NOT_STARTED = "not_started"
    NOT_TRANSMITTED = "not_transmitted"
    NOT_DELIVERED = "not_delivered"
    GPU_NOT_READY = "gpu_not_ready"


@dataclass(frozen=True)
class StateAssessment:
    """One snapshot state and the bounded explanations it supports."""

    state: ProgressState
    local_possible_causes: tuple[str, ...]
    remote_possible_causes: tuple[str, ...]
    evidence_gaps: tuple[str, ...]


@dataclass(frozen=True)
class ClassificationResult:
    """All states matching one snapshot; an empty tuple means completion."""

    assessments: tuple[StateAssessment, ...]

    @property
    def states(self) -> tuple[ProgressState, ...]:
        return tuple(item.state for item in self.assessments)


def classify_snapshot(
    snapshot: ProgressSnapshot,
    total_chunks: int,
) -> ClassificationResult:
    """Return every paper state whose condition matches ``snapshot``.

    These states describe where data currently sits in the pipeline. They do
    not prove that the corresponding local component is faulty. Multiple
    conditions may hold at once, and a caller must not treat candidates as a
    unique root cause without temporal and peer-side evidence.
    """
    if total_chunks <= 0:
        raise ValueError("total_chunks must be positive")
    if snapshot.tick < 0:
        raise ValueError("tick must be non-negative")
    validate_snapshot(snapshot, total_chunks)

    assessments: list[StateAssessment] = []

    if (
        snapshot.gpu_ready
        == snapshot.rdma_transmitted
        == snapshot.rdma_done
        == 0
    ):
        assessments.append(
            StateAssessment(
                state=ProgressState.NOT_STARTED,
                local_possible_causes=("uninitialized",),
                remote_possible_causes=("blocked",),
                evidence_gaps=("launch_state", "upstream_progress"),
            )
        )

    if snapshot.gpu_ready > snapshot.rdma_transmitted:
        assessments.append(
            StateAssessment(
                state=ProgressState.NOT_TRANSMITTED,
                local_possible_causes=("rdma_issue",),
                remote_possible_causes=("receiver_not_ready",),
                evidence_gaps=("receiver_progress",),
            )
        )

    if snapshot.rdma_transmitted > snapshot.rdma_done:
        assessments.append(
            StateAssessment(
                state=ProgressState.NOT_DELIVERED,
                local_possible_causes=("rdma_issue",),
                remote_possible_causes=("receiver_failed",),
                evidence_gaps=("receiver_progress",),
            )
        )

    if (
        0
        < snapshot.gpu_ready
        == snapshot.rdma_transmitted
        == snapshot.rdma_done
        < total_chunks
    ):
        assessments.append(
            StateAssessment(
                state=ProgressState.GPU_NOT_READY,
                local_possible_causes=("gpu_issue",),
                remote_possible_causes=(),
                evidence_gaps=("gpu_execution",),
            )
        )

    return ClassificationResult(assessments=tuple(assessments))
