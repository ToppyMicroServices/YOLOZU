"""Evaluation protocol loading, validation, and application.

An eval protocol fixes experimental conditions (image size, score threshold,
bbox format, preprocessing) so that results from different systems are
comparable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from yolozu import resources as yz_resources

__all__ = [
    "load_eval_protocol",
    "validate_eval_protocol",
    "apply_eval_protocol_args",
    "eval_protocol_hash",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROTOCOLS = {
    "yolo26": "protocols/yolo26_eval.json",
    "nms_applied": "protocols/nms_applied_eval.json",
    "e2e_nms_free": "protocols/e2e_nms_free_eval.json",
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_eval_protocol(protocol_id: str) -> dict[str, Any]:
    path = _DEFAULT_PROTOCOLS.get(protocol_id)
    if path is None:
        raise ValueError(f"unknown protocol: {protocol_id}")
    if isinstance(path, Path):
        doc = json.loads(path.read_text())
    else:
        doc = json.loads(yz_resources.read_text(str(path)))
    validate_eval_protocol(doc)
    return doc


def validate_eval_protocol(doc: Any) -> None:
    if not isinstance(doc, dict):
        raise ValueError("eval protocol must be an object")

    protocol_id = doc.get("id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("eval protocol.id must be a non-empty string")

    protocol_schema_version = doc.get("protocol_schema_version")
    if not isinstance(protocol_schema_version, int) or isinstance(protocol_schema_version, bool) or protocol_schema_version < 1:
        raise ValueError("eval protocol.protocol_schema_version must be a positive integer")

    split = doc.get("split")
    if not isinstance(split, str) or not split:
        raise ValueError("eval protocol.split must be a non-empty string")

    bbox_format = doc.get("bbox_format")
    if bbox_format not in ("cxcywh_norm", "cxcywh_abs", "xywh_abs", "xyxy_abs"):
        raise ValueError("eval protocol.bbox_format must be a supported bbox format")

    metric_key = doc.get("metric_key")
    if not isinstance(metric_key, str) or not metric_key:
        raise ValueError("eval protocol.metric_key must be a non-empty string")

    imgsz = doc.get("imgsz")
    if imgsz is not None and not (isinstance(imgsz, int) and imgsz > 0):
        raise ValueError("eval protocol.imgsz must be a positive int or null")

    metric_label = doc.get("metric_label")
    if metric_label is not None and not isinstance(metric_label, str):
        raise ValueError("eval protocol.metric_label must be a string or null")

    reports = doc.get("reports", {})
    if reports is not None and not isinstance(reports, dict):
        raise ValueError("eval protocol.reports must be an object or null")

    fixed = doc.get("fixed_conditions")
    if not isinstance(fixed, dict):
        raise ValueError("eval protocol.fixed_conditions must be an object")

    for required in ("imgsz", "score_threshold", "iou_threshold", "max_detections", "bbox_format", "preprocess"):
        if required not in fixed:
            raise ValueError(f"eval protocol.fixed_conditions missing '{required}'")

    fixed_imgsz = fixed.get("imgsz")
    if not isinstance(fixed_imgsz, int) or isinstance(fixed_imgsz, bool) or fixed_imgsz <= 0:
        raise ValueError("eval protocol.fixed_conditions.imgsz must be a positive int")

    fixed_score = fixed.get("score_threshold")
    if not _is_number(fixed_score):
        raise ValueError("eval protocol.fixed_conditions.score_threshold must be numeric")

    fixed_iou = fixed.get("iou_threshold")
    if not _is_number(fixed_iou):
        raise ValueError("eval protocol.fixed_conditions.iou_threshold must be numeric")

    fixed_max_det = fixed.get("max_detections")
    if not isinstance(fixed_max_det, int) or isinstance(fixed_max_det, bool) or fixed_max_det <= 0:
        raise ValueError("eval protocol.fixed_conditions.max_detections must be a positive int")

    fixed_bbox = fixed.get("bbox_format")
    if fixed_bbox not in ("cxcywh_norm", "cxcywh_abs", "xywh_abs", "xyxy_abs"):
        raise ValueError("eval protocol.fixed_conditions.bbox_format must be a supported bbox format")
    if fixed_bbox != bbox_format:
        raise ValueError("eval protocol.fixed_conditions.bbox_format must match eval protocol.bbox_format")

    preprocess = fixed.get("preprocess")
    if not isinstance(preprocess, dict):
        raise ValueError("eval protocol.fixed_conditions.preprocess must be an object")
    for key in ("method", "input_color", "normalize", "resize_interp", "letterbox_fill"):
        if key not in preprocess:
            raise ValueError(f"eval protocol.fixed_conditions.preprocess missing '{key}'")
    if preprocess.get("method") != "letterbox":
        raise ValueError("eval protocol.fixed_conditions.preprocess.method must be 'letterbox'")
    if preprocess.get("input_color") != "RGB":
        raise ValueError("eval protocol.fixed_conditions.preprocess.input_color must be 'RGB'")
    if preprocess.get("normalize") != "0_1":
        raise ValueError("eval protocol.fixed_conditions.preprocess.normalize must be '0_1'")
    if preprocess.get("resize_interp") not in ("linear", "bilinear"):
        raise ValueError("eval protocol.fixed_conditions.preprocess.resize_interp must be 'linear' or 'bilinear'")
    fill = preprocess.get("letterbox_fill")
    if not isinstance(fill, list) or len(fill) != 3:
        raise ValueError("eval protocol.fixed_conditions.preprocess.letterbox_fill must be list[3]")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in fill):
        raise ValueError("eval protocol.fixed_conditions.preprocess.letterbox_fill entries must be integers")


def apply_eval_protocol_args(args: Any, protocol: dict[str, Any]) -> Any:
    """Apply protocol-defined settings to an argparse namespace."""

    args.split = protocol.get("split", args.split)
    args.bbox_format = protocol.get("bbox_format", args.bbox_format)
    return args


def eval_protocol_hash(protocol: dict[str, Any]) -> str:
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
