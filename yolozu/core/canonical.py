"""Canonical data structures shared by YOLOZU's pipeline.

These frozen dataclasses form the *internal* representation.  Most tools
consume / produce plain ``dict`` records (via ``to_record_dict`` /
``to_dict``) so downstream code never needs to depend on the classes
directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["BBox", "Label", "SampleRecord", "TrainConfig"]


@dataclass(frozen=True)
class BBox:
    """Canonical bbox representation (cxcywh_norm).

    - cx, cy, w, h are normalized to [0,1] relative to image width/height.
    """

    cx: float
    cy: float
    w: float
    h: float


@dataclass(frozen=True)
class Label:
    class_id: int
    bbox: BBox
    polygon: list[float] | None = None
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"class_id": int(self.class_id), **asdict(self.bbox)}
        if self.polygon is not None:
            out["polygon"] = list(self.polygon)
        if self.meta is not None:
            out["meta"] = dict(self.meta)
        return out


@dataclass(frozen=True)
class SampleRecord:
    """Canonical per-image record.

    This is YOLOZU's internal "SampleRecord" representation. Most tools use the
    dict form returned by `to_record_dict()`:
      {"image": "...", "labels": [{"class_id": 0, "cx":..., "cy":..., "w":..., "h":...}, ...], ...}
    """

    image_path: str
    width: int | None = None
    height: int | None = None
    labels: list[Label] = field(default_factory=list)

    mask: str | None = None
    depth: str | None = None
    pose: dict[str, Any] | None = None
    intrinsics: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None

    def to_record_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "image": str(self.image_path),
            "labels": [lab.to_dict() for lab in self.labels],
        }
        if self.width is not None and self.height is not None:
            out["image_hw"] = [int(self.height), int(self.width)]
        if self.mask is not None:
            out["mask"] = str(self.mask)
            out["mask_path"] = str(self.mask)
        if self.depth is not None:
            out["depth"] = str(self.depth)
            out["depth_path"] = str(self.depth)
            out["D_obj"] = str(self.depth)
        if self.pose is not None:
            out["pose"] = dict(self.pose)
        if self.intrinsics is not None:
            out["intrinsics"] = dict(self.intrinsics)
        if self.meta is not None:
            out["meta"] = dict(self.meta)
        return out


@dataclass(frozen=True)
class TrainConfig:
    """Canonical training config projection (major keys only)."""

    backend: str | None = None
    task: str | None = None
    model: str | None = None
    imgsz: int | list[int] | None = None
    batch: int | None = None
    epochs: int | None = None
    steps: int | None = None
    optimizer: str | None = None
    lr: float | None = None
    weight_decay: float | None = None
    seed: int | None = None
    device: str | None = None
    precision: str | None = None
    workers: int | None = None
    grad_clip_norm: float | None = None

    dataset: dict[str, Any] | None = None
    preprocess: dict[str, Any] | None = None
    aug: dict[str, Any] | None = None
    loss: dict[str, Any] | None = None
    eval: dict[str, Any] | None = None
    export: dict[str, Any] | None = None
    run_contract: dict[str, Any] | None = None
    backend_options: dict[str, Any] | None = None
    source: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"format": "yolozu_train_config_v1", **asdict(self)}
        # Remove nulls for readability.
        return {k: v for k, v in payload.items() if v is not None}
