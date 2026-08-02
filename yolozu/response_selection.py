from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


@dataclass(frozen=True)
class ForegroundSelection:
    mask: Any
    confidence: Any

    @property
    def selected_count(self) -> int:
        return int(self.mask.sum().detach().cpu().item())


def _ensure_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("torch is required for detector response selection")


def select_foreground_queries(
    teacher_logits: Any,
    *,
    confidence_min: float,
    topk: int,
) -> ForegroundSelection:
    """Select teacher queries that beat RT-DETR's final no-object class."""

    _ensure_torch()
    if not isinstance(teacher_logits, torch.Tensor) or teacher_logits.ndim != 3:
        raise ValueError("detector response selection requires logits shaped [B, Q, C]")
    if int(teacher_logits.shape[-1]) < 2:
        raise ValueError("detector response selection requires foreground plus no-object logits")
    if float(confidence_min) < 0.0 or float(confidence_min) > 1.0:
        raise ValueError("foreground confidence_min must be between 0 and 1")
    if int(topk) < 0:
        raise ValueError("foreground topk must be >= 0")

    probs = F.softmax(teacher_logits.detach(), dim=-1)
    foreground_confidence = probs[..., :-1].amax(dim=-1)
    background_confidence = probs[..., -1]
    eligible = (foreground_confidence >= float(confidence_min)) & (
        foreground_confidence > background_confidence
    )

    if int(topk) > 0 and int(topk) < int(eligible.shape[-1]):
        scores = foreground_confidence.masked_fill(~eligible, -1.0)
        indices = torch.topk(scores, k=int(topk), dim=-1).indices
        capped = torch.zeros_like(eligible)
        capped.scatter_(-1, indices, True)
        eligible = eligible & capped

    return ForegroundSelection(mask=eligible, confidence=foreground_confidence)


def detector_response_loss(
    student_outputs: dict[str, Any],
    teacher_outputs: dict[str, Any],
    *,
    confidence_min: float,
    topk: int,
    min_selected: int = 1,
    class_weight: float,
    bbox_weight: float,
    entropy_weight: float,
) -> tuple[Any, dict[str, float]]:
    """Distil selected foreground responses on the same detector queries."""

    _ensure_torch()
    student_logits = student_outputs.get("logits")
    teacher_logits = teacher_outputs.get("logits")
    if not isinstance(student_logits, torch.Tensor) or not isinstance(
        teacher_logits, torch.Tensor
    ):
        raise ValueError("detector response loss requires logits outputs")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher detector logits must have matching shapes")
    for name, value in (
        ("class_weight", class_weight),
        ("bbox_weight", bbox_weight),
        ("entropy_weight", entropy_weight),
    ):
        if float(value) < 0.0:
            raise ValueError(f"{name} must be >= 0")
    if int(min_selected) < 1:
        raise ValueError("min_selected must be >= 1")
    if int(topk) > 0 and int(topk) < int(min_selected):
        raise ValueError("topk must be 0 or >= min_selected")

    selection = select_foreground_queries(
        teacher_logits,
        confidence_min=float(confidence_min),
        topk=int(topk),
    )
    selected = selection.mask
    selected_count = selection.selected_count
    abstained = selected_count < int(min_selected)
    total_queries = int(selected.numel())
    zero = student_logits.sum() * 0.0

    if not abstained:
        teacher_foreground = F.softmax(teacher_logits.detach()[..., :-1], dim=-1)
        student_foreground_log = F.log_softmax(student_logits[..., :-1], dim=-1)
        class_per_query = F.kl_div(
            student_foreground_log,
            teacher_foreground,
            reduction="none",
        ).sum(dim=-1)
        class_loss = class_per_query[selected].mean()

        student_foreground = student_foreground_log.exp()
        entropy_per_query = -(
            student_foreground * student_foreground_log
        ).sum(dim=-1)
        entropy_loss = entropy_per_query[selected].mean()
    else:
        class_loss = zero
        entropy_loss = zero

    bbox_loss = zero
    student_bbox = student_outputs.get("bbox")
    teacher_bbox = teacher_outputs.get("bbox")
    if not abstained and isinstance(student_bbox, torch.Tensor) and isinstance(
        teacher_bbox, torch.Tensor
    ):
        if student_bbox.shape != teacher_bbox.shape or student_bbox.shape[:-1] != selected.shape:
            raise ValueError("student and teacher detector boxes must match selected query shape")
        bbox_per_query = F.smooth_l1_loss(
            student_bbox.sigmoid(),
            teacher_bbox.detach().sigmoid(),
            reduction="none",
        ).mean(dim=-1)
        bbox_loss = bbox_per_query[selected].mean()

    total = (
        float(class_weight) * class_loss
        + float(bbox_weight) * bbox_loss
        + float(entropy_weight) * entropy_loss
    )
    selected_confidence = (
        float(selection.confidence[selected].mean().detach().cpu().item())
        if selected_count > 0
        else 0.0
    )
    metrics = {
        "loss_response_class": float(class_loss.detach().cpu().item()),
        "loss_response_bbox": float(bbox_loss.detach().cpu().item()),
        "loss_response_foreground_entropy": float(entropy_loss.detach().cpu().item()),
        "response_selected_queries": float(selected_count),
        "response_selected_ratio": float(selected_count) / float(max(1, total_queries)),
        "response_selected_confidence_mean": selected_confidence,
        "response_min_selected": float(min_selected),
        "response_abstained": 1.0 if abstained else 0.0,
    }
    return total, metrics
