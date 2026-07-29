"""Continual-learning evaluation metrics.

Provides ``summarize_continual_matrix`` which computes standard CL metrics
(average accuracy, forgetting, backward/forward transfer) from a
task-by-time performance matrix.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["ContinualSummary", "summarize_continual_matrix"]


@dataclass(frozen=True)
class ContinualSummary:
    """Summary metrics derived from a task-by-time evaluation matrix."""

    avg_acc: float | None
    forgetting: float | None
    bwt: float | None
    fwt: float | None
    details: dict[str, Any]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _get(matrix: list[list[float | None]], t: int, i: int) -> float | None:
    if t < 0 or i < 0:
        return None
    if t >= len(matrix):
        return None
    row = matrix[t]
    if i >= len(row):
        return None
    value = row[i]
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def summarize_continual_matrix(
    matrix: list[list[float | None]],
    *,
    baseline: Sequence[float | None] | None = None,
) -> ContinualSummary:
    """Compute standard continual-learning metrics from a performance matrix.

    matrix[t][i] = performance on task i after finishing training up to task t.

    Returned metrics follow common CL conventions:
    - avg_acc: average of final performance across tasks
    - forgetting: average over prior tasks of (best past performance - final performance)
    - bwt: average over prior tasks of (final performance - performance right after learning task)
    - fwt: average over future tasks of (performance immediately before learning
      the task - initial-checkpoint baseline). FWT is undefined without a
      baseline vector.
    """

    if not matrix:
        return ContinualSummary(avg_acc=None, forgetting=None, bwt=None, fwt=None, details={"reason": "empty_matrix"})

    t_final = len(matrix) - 1
    n_tasks = max((len(row) for row in matrix), default=0)

    final_vals: list[float] = []
    diag_vals: list[float] = []
    forget_vals: list[float] = []
    bwt_vals: list[float] = []
    fwt_pretrain_vals: list[float] = []
    fwt_vals: list[float] = []

    for i in range(int(n_tasks)):
        v_final = _get(matrix, t_final, i)
        if v_final is not None:
            final_vals.append(v_final)

        v_diag = _get(matrix, i, i)
        if v_diag is not None:
            diag_vals.append(v_diag)

        # The final task has no later task that could cause forgetting.
        if i < t_final and v_final is not None:
            best = None
            for t in range(int(i), int(t_final)):
                v = _get(matrix, t, i)
                if v is None:
                    continue
                best = v if best is None else max(best, v)
            if best is not None:
                forget_vals.append(float(best - v_final))

        # Backward transfer: final - diag.
        if i < t_final and v_final is not None and v_diag is not None:
            bwt_vals.append(float(v_final - v_diag))

        # GEM convention: R[i-1, i] - baseline[i].
        if i > 0:
            pretrain = _get(matrix, int(i) - 1, i)
            if pretrain is not None:
                fwt_pretrain_vals.append(float(pretrain))
                if baseline is not None and i < len(baseline):
                    baseline_value = baseline[i]
                    if baseline_value is not None:
                        fwt_vals.append(float(pretrain - float(baseline_value)))

    details = {
        "t_final": int(t_final),
        "n_tasks": int(n_tasks),
        "final": list(final_vals),
        "diag": list(diag_vals),
        "forget_per_task": list(forget_vals),
        "bwt_per_task": list(bwt_vals),
        "fwt_baseline": list(baseline) if baseline is not None else None,
        "fwt_pretrain": list(fwt_pretrain_vals),
        "fwt_delta_per_task": list(fwt_vals),
        "fwt_defined": baseline is not None and len(fwt_vals) == max(0, int(n_tasks) - 1),
    }

    return ContinualSummary(
        avg_acc=_mean(final_vals),
        forgetting=_mean(forget_vals),
        bwt=_mean(bwt_vals),
        fwt=(
            _mean(fwt_vals)
            if baseline is not None and len(fwt_vals) == max(0, int(n_tasks) - 1)
            else None
        ),
        details=details,
    )
