from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..predictions.predictions_transform import TransformResult, apply_tta


class TTARunner:
    """Base interface for test-time adaptation runners."""

    def reset(self) -> None:
        """Reset internal state (e.g., optimizer/momentum)."""
        return None

    def adapt_step(self, batch: Any) -> dict[str, float]:
        """Run a single adaptation step and return scalar metrics."""
        raise NotImplementedError

    def maybe_log(self) -> dict[str, Any] | None:
        """Optional logging hook for runner-specific diagnostics."""
        return None


@dataclass(frozen=True)
class TTAConfig:
    enabled: bool = False
    seed: int | None = None
    flip_prob: float = 0.5
    norm_only: bool = False
    flip_keypoints: bool = True
    flip_pose_offsets: bool = True


def apply_tta_transform(
    entries: Iterable[dict[str, Any]], *, config: TTAConfig
) -> TransformResult:
    return apply_tta(
        entries,
        enabled=bool(config.enabled),
        seed=config.seed,
        flip_prob=float(config.flip_prob),
        norm_only=bool(config.norm_only),
        flip_keypoints=bool(config.flip_keypoints),
        flip_pose_offsets=bool(config.flip_pose_offsets),
    )
