"""Prediction export utilities (dummy / labels backends).

These functions produce schema-correct ``predictions.json`` artifacts that
satisfy ``validate_predictions_payload``, useful for smoke-testing the
evaluation pipeline without actual inference.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_PREDICTIONS_PATH",
    "export_dummy_predictions",
    "export_labels_predictions",
    "write_predictions_json",
]


DEFAULT_PREDICTIONS_PATH = "reports/predictions.json"
_NUMERIC_ERRORS = (TypeError, ValueError, OverflowError)


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_json(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _sha256_bytes(data)


def _default_tta_meta() -> dict[str, Any]:
    """Return a disabled-TTA meta block satisfying the interface contract."""
    return {
        "enabled": False,
        "seed": None,
        "flip_prob": 0.0,
        "norm_only": False,
        "warnings": [],
        "summary": None,
    }


def _default_ttt_meta() -> dict[str, Any]:
    """Return a disabled-TTT meta block satisfying the interface contract."""
    return {
        "enabled": False,
        "method": "none",
        "steps": 0,
        "batch_size": 0,
        "lr": 0.0,
        "update_filter": "none",
        "include": None,
        "exclude": None,
        "max_batches": 0,
        "seed": None,
        "mim": {"mask_prob": 0.0, "patch_size": 0, "mask_value": 0.0},
        "report": None,
    }


def _pick_bbox_from_labels(labels: Any) -> dict[str, float]:
    if isinstance(labels, list) and labels:
        lbl = labels[0]
        if isinstance(lbl, dict):
            try:
                return {
                    "cx": float(lbl.get("cx", 0.5)),
                    "cy": float(lbl.get("cy", 0.5)),
                    "w": float(lbl.get("w", 0.2)),
                    "h": float(lbl.get("h", 0.2)),
                }
            except _NUMERIC_ERRORS:
                pass
    return {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}


def export_dummy_predictions(
    *,
    dataset_root: str | Path,
    split: str | None = None,
    max_images: int | None = None,
    score: float = 0.9,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Export schema-correct dummy predictions from a YOLO-format dataset.

    The dummy backend emits 1 detection per image, using the first GT bbox if present.
    """

    from yolozu.dataset import build_manifest

    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    if not (dataset_root / "images").exists():
        raise FileNotFoundError(f"dataset root missing images/: {dataset_root}")
    if not (dataset_root / "labels").exists():
        raise FileNotFoundError(f"dataset root missing labels/: {dataset_root}")
    manifest = build_manifest(dataset_root, split=split)
    records = list(manifest.get("images") or [])
    if max_images is not None:
        records = records[: max(0, int(max_images))]

    preds: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        image = record.get("image")
        if not isinstance(image, str) or not image:
            continue
        labels = record.get("labels", [])
        bbox = _pick_bbox_from_labels(labels)
        class_id = 0
        if isinstance(labels, list) and labels and isinstance(labels[0], dict):
            try:
                class_id = int(labels[0].get("class_id", 0))
            except _NUMERIC_ERRORS:
                class_id = 0

        preds.append(
            {
                "image": image,
                "detections": [
                    {
                        "class_id": int(class_id),
                        "score": float(score),
                        "bbox": bbox,
                    }
                ],
            }
        )

    config_fingerprint = {
        "backend": "dummy",
        "dataset": str(dataset_root),
        "split": split,
        "max_images": max_images,
        "score": float(score),
    }

    meta = {
        "schema_version": 1,
        "timestamp": _now_utc(),
        "generator": "yolozu export --backend dummy",
        "adapter": "dummy",
        "config": str(dataset_root),
        "images": int(len(preds)),
        "tta": _default_tta_meta(),
        "ttt": _default_ttt_meta(),
        "run": {
            "config_fingerprint": config_fingerprint,
            "config_hash": _sha256_json(config_fingerprint),
        },
    }

    payload = {"schema_version": 1, "predictions": preds, "meta": meta}
    return payload, meta["run"]


def export_labels_predictions(
    *,
    dataset_root: str | Path,
    split: str | None = None,
    max_images: int | None = None,
    score: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Export predictions from dataset labels (perfect predictions).

    This is useful for smoke-testing the evaluation pipeline without requiring
    an inference backend.
    """

    from yolozu.dataset import build_manifest

    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    if not (dataset_root / "images").exists():
        raise FileNotFoundError(f"dataset root missing images/: {dataset_root}")
    if not (dataset_root / "labels").exists():
        raise FileNotFoundError(f"dataset root missing labels/: {dataset_root}")
    manifest = build_manifest(dataset_root, split=split)
    records = list(manifest.get("images") or [])
    if max_images is not None:
        records = records[: max(0, int(max_images))]

    preds: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        image = record.get("image")
        if not isinstance(image, str) or not image:
            continue
        labels = record.get("labels", [])
        detections = []
        if isinstance(labels, list):
            for lbl in labels:
                if not isinstance(lbl, dict):
                    continue
                try:
                    class_id = int(lbl.get("class_id", 0))
                except _NUMERIC_ERRORS:
                    class_id = 0
                bbox = lbl.get("bbox")
                if not isinstance(bbox, dict):
                    bbox = {
                        "cx": lbl.get("cx"),
                        "cy": lbl.get("cy"),
                        "w": lbl.get("w"),
                        "h": lbl.get("h"),
                    }
                try:
                    bbox_out = {
                        "cx": float(bbox.get("cx", 0.5)),
                        "cy": float(bbox.get("cy", 0.5)),
                        "w": float(bbox.get("w", 0.0)),
                        "h": float(bbox.get("h", 0.0)),
                    }
                except _NUMERIC_ERRORS:
                    continue
                detections.append({"class_id": int(class_id), "score": float(score), "bbox": bbox_out})

        preds.append({"image": image, "detections": detections})

    config_fingerprint = {
        "backend": "labels",
        "dataset": str(dataset_root),
        "split": split,
        "max_images": max_images,
        "score": float(score),
    }

    meta = {
        "schema_version": 1,
        "timestamp": _now_utc(),
        "generator": "yolozu export --backend labels",
        "adapter": "labels",
        "config": str(dataset_root),
        "images": int(len(preds)),
        "tta": _default_tta_meta(),
        "ttt": _default_ttt_meta(),
        "run": {
            "config_fingerprint": config_fingerprint,
            "config_hash": _sha256_json(config_fingerprint),
        },
    }

    payload = {"schema_version": 1, "predictions": preds, "meta": meta}
    return payload, meta["run"]


def write_predictions_json(*, output: str | Path, payload: dict[str, Any], force: bool = False) -> Path:
    out_path = Path(output)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        raise FileExistsError(f"output exists: {out_path} (use --force to overwrite)")
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path
