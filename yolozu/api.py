"""Stable in-process validation and COCO evaluation API.

This module is the supported Python surface for consumers that should not need
to invoke the YOLOZU CLI in a subprocess.  It intentionally accepts either a
path-backed or in-memory predictions payload and returns serializable typed
results.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from yolozu.core.image_keys import add_image_aliases, lookup_image_alias, require_image_key
from yolozu.datasets.dataset import build_manifest
from yolozu.eval.coco_eval import build_coco_ground_truth, evaluate_coco_map, predictions_to_coco_detections
from yolozu.predictions.predictions import (
    canonicalize_predictions,
    normalize_predictions_payload,
    validate_predictions_entries,
    validate_wrapped_meta,
)
from yolozu.predictions.predictions_transform import load_classes_json, normalize_class_ids
from yolozu.predictions.schema_governance import validate_payload_schema_version

__all__ = [
    "APIError",
    "InputError",
    "DatasetError",
    "PredictionsValidationError",
    "EvaluationError",
    "OptionalDependencyError",
    "PredictionsInput",
    "PredictionsValidationResult",
    "CocoMetrics",
    "EvaluationCounts",
    "CocoEvaluationResult",
    "validate_predictions",
    "evaluate_coco",
]

JsonObject = dict[str, Any]
PredictionsPayload = Mapping[str, Any] | Sequence[Mapping[str, Any]]
BBoxFormat = Literal["cxcywh_norm", "cxcywh_abs", "xywh_abs", "xyxy_abs"]


class APIError(Exception):
    """Base class for stable, machine-readable API errors."""

    default_code = "E_API"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code or self.default_code)
        self.message = str(message)
        self.details = dict(details or {})

    def to_dict(self) -> JsonObject:
        payload: JsonObject = {
            "code": self.code,
            "message": self.message,
            "category": type(self).__name__,
        }
        if self.details:
            payload["details"] = copy.deepcopy(self.details)
        return payload


class InputError(APIError):
    """A path or JSON input could not be resolved or decoded."""

    default_code = "E_INPUT"


class DatasetError(APIError):
    """The dataset could not be resolved into an evaluable subset."""

    default_code = "E_DATASET"


class PredictionsValidationError(APIError):
    """The predictions payload violates the predictions interface contract."""

    default_code = "E_PREDICTIONS_INVALID"


class EvaluationError(APIError):
    """COCO evaluation could not be completed."""

    default_code = "E_EVALUATION"


class OptionalDependencyError(EvaluationError):
    """An optional runtime dependency required for the requested mode is absent."""

    default_code = "E_OPTIONAL_DEPENDENCY"


@dataclass(frozen=True)
class PredictionsInput:
    """Typed predictions input with exactly one path or payload source."""

    path: Path | None = None
    payload: PredictionsPayload | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if (self.path is None) == (self.payload is None):
            raise ValueError("PredictionsInput requires exactly one of path or payload")

    @classmethod
    def from_path(cls, path: str | Path, *, label: str | None = None) -> "PredictionsInput":
        return cls(path=Path(path), label=label)

    @classmethod
    def from_payload(
        cls,
        payload: PredictionsPayload,
        *,
        label: str = "<in-memory>",
    ) -> "PredictionsInput":
        return cls(payload=payload, label=label)


@dataclass(frozen=True)
class PredictionsValidationResult:
    """Canonical validated entries plus every compatibility or repair warning."""

    entries: tuple[JsonObject, ...]
    warnings: tuple[str, ...]
    mode: Literal["strict", "repair"]

    @property
    def repair_enabled(self) -> bool:
        return self.mode == "repair"

    def to_dict(self, *, include_entries: bool = True) -> JsonObject:
        payload: JsonObject = {
            "ok": True,
            "mode": self.mode,
            "repair_enabled": self.repair_enabled,
            "counts": {
                "prediction_images": len(self.entries),
                "detections": sum(len(entry.get("detections") or []) for entry in self.entries),
            },
            "warnings": list(self.warnings),
        }
        if include_entries:
            payload["predictions"] = copy.deepcopy(list(self.entries))
        return payload


@dataclass(frozen=True)
class CocoMetrics:
    map50_95: float | None
    map50: float | None
    map75: float | None
    ar100: float | None

    def to_dict(self) -> JsonObject:
        return {
            "map50_95": self.map50_95,
            "map50": self.map50,
            "map75": self.map75,
            "ar100": self.ar100,
        }


@dataclass(frozen=True)
class EvaluationCounts:
    dataset_images_total: int
    images: int
    prediction_images_total: int
    prediction_images_evaluated: int
    prediction_images_excluded: int
    selected_images_without_predictions: int
    detections_input: int
    detections: int
    detections_excluded: int

    def to_dict(self) -> dict[str, int]:
        return {
            "dataset_images_total": self.dataset_images_total,
            "images": self.images,
            "prediction_images_total": self.prediction_images_total,
            "prediction_images_evaluated": self.prediction_images_evaluated,
            "prediction_images_excluded": self.prediction_images_excluded,
            "selected_images_without_predictions": self.selected_images_without_predictions,
            "detections_input": self.detections_input,
            "detections": self.detections,
            "detections_excluded": self.detections_excluded,
        }


@dataclass(frozen=True)
class CocoEvaluationResult:
    """Typed COCO evaluation result with a stable JSON serialization."""

    timestamp: str
    dataset: str
    split: str
    split_requested: str | None
    predictions: str
    bbox_format: BBoxFormat
    max_images: int | None
    dry_run: bool
    repair: bool
    normalization_classes: str | None
    assume_class_id_is_category_id: bool
    metrics: CocoMetrics
    stats: tuple[float, ...]
    counts: EvaluationCounts
    warnings: tuple[str, ...]

    def to_dict(self) -> JsonObject:
        return {
            "report_schema_version": 1,
            "status": "ok",
            "ok": True,
            "timestamp": self.timestamp,
            "dataset": self.dataset,
            "split": self.split,
            "split_requested": self.split_requested,
            "predictions": self.predictions,
            "bbox_format": self.bbox_format,
            "max_images": self.max_images,
            "normalization": {
                "classes": self.normalization_classes,
                "assume_class_id_is_category_id": self.assume_class_id_is_category_id,
            },
            "validation": {
                "mode": "repair" if self.repair else "strict",
                "repair_enabled": self.repair,
            },
            "metrics": self.metrics.to_dict(),
            "stats": list(self.stats),
            "dry_run": self.dry_run,
            "counts": self.counts.to_dict(),
            "warnings": list(self.warnings),
        }


PredictionsSource = PredictionsInput | str | Path | PredictionsPayload


def _unique_warnings(warnings: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(warning) for warning in warnings))


def _resolve_path(
    value: str | Path,
    *,
    base_dir: str | Path | None,
    field: str,
) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if base_dir is None:
        raise InputError(
            f"{field} is relative; provide an absolute path or base_dir",
            code="E_RELATIVE_PATH",
            details={"field": field, "path": str(path)},
        )
    base = Path(base_dir).expanduser()
    if not base.is_absolute():
        raise InputError(
            "base_dir must be absolute",
            code="E_RELATIVE_BASE_DIR",
            details={"base_dir": str(base)},
        )
    return base / path


def _coerce_predictions_input(source: PredictionsSource) -> PredictionsInput:
    if isinstance(source, PredictionsInput):
        return source
    if isinstance(source, (str, Path)):
        return PredictionsInput.from_path(source)
    if isinstance(source, Mapping):
        return PredictionsInput.from_payload(source)
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        return PredictionsInput.from_payload(source)
    raise InputError(
        "predictions must be a PredictionsInput, path, wrapped payload, or entry sequence",
        code="E_PREDICTIONS_SOURCE",
    )


def _load_predictions_source(
    source: PredictionsSource,
    *,
    base_dir: str | Path | None,
) -> tuple[Any, str]:
    resolved = _coerce_predictions_input(source)
    if resolved.path is None:
        return copy.deepcopy(resolved.payload), str(resolved.label or "<in-memory>")

    path = _resolve_path(resolved.path, base_dir=base_dir, field="predictions")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError(
            f"predictions file not found: {path}",
            code="E_PREDICTIONS_NOT_FOUND",
            details={"path": str(path)},
        ) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(
            f"could not read predictions JSON: {path}: {exc}",
            code="E_PREDICTIONS_READ",
            details={"path": str(path)},
        ) from exc
    return payload, str(resolved.label or path)


def _canonicalize_payload(
    payload: Any,
    *,
    repair: bool,
) -> PredictionsValidationResult:
    warnings: list[str] = []
    try:
        warnings.extend(validate_payload_schema_version(payload, artifact="predictions"))
        entries, meta = normalize_predictions_payload(payload)
        if meta is not None:
            validate_wrapped_meta(meta)
        canonical = canonicalize_predictions(
            entries,
            policy="clamp" if repair else "error",
            strict=not repair,
            unknown_keys="warn",
        )
        validation = validate_predictions_entries(canonical.entries, strict=True)
    except (TypeError, ValueError) as exc:
        raise PredictionsValidationError(str(exc)) from exc

    warnings.extend(canonical.warnings)
    warnings.extend(validation.warnings)
    return PredictionsValidationResult(
        entries=tuple(copy.deepcopy(canonical.entries)),
        warnings=_unique_warnings(warnings),
        mode="repair" if repair else "strict",
    )


def validate_predictions(
    predictions: PredictionsSource,
    *,
    repair: bool = False,
    base_dir: str | Path | None = None,
) -> PredictionsValidationResult:
    """Validate predictions without invoking a subprocess.

    Strict fail-closed validation is the default.  ``repair=True`` explicitly
    enables legacy canonicalization and range clamping; every repair is retained
    in ``result.warnings``.
    """

    payload, _ = _load_predictions_source(predictions, base_dir=base_dir)
    return _canonicalize_payload(payload, repair=repair)


def _dataset_aliases(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    aliases: dict[str, int] = {}
    for index, record in enumerate(records):
        image = require_image_key(record.get("image"), where=f"dataset.images[{index}].image")
        add_image_aliases(aliases, image, index)
    return aliases


def _entry_detection_count(entry: Mapping[str, Any]) -> int:
    detections = entry.get("detections")
    return len(detections) if isinstance(detections, list) else 0


def _validate_class_normalization_shape(entries: Sequence[Any]) -> None:
    """Reject structures that normalize_class_ids would otherwise skip."""

    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PredictionsValidationError(f"predictions[{entry_index}]: entry must be an object")
        detections = entry.get("detections")
        if detections is None:
            continue
        if not isinstance(detections, list):
            raise PredictionsValidationError(
                f"predictions[{entry_index}].detections: must be a list"
            )
        for detection_index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                raise PredictionsValidationError(
                    f"predictions[{entry_index}].detections[{detection_index}]: "
                    "detection must be an object"
                )


def evaluate_coco(
    dataset: str | Path,
    predictions: PredictionsSource,
    *,
    split: str | None = None,
    bbox_format: BBoxFormat = "cxcywh_norm",
    max_images: int | None = None,
    dry_run: bool = False,
    repair: bool = False,
    classes: str | Path | None = None,
    assume_class_id_is_category_id: bool = False,
    base_dir: str | Path | None = None,
) -> CocoEvaluationResult:
    """Validate and evaluate predictions against a YOLO-format dataset.

    Relative paths are accepted only when an explicit absolute ``base_dir`` is
    supplied, so library behavior does not silently depend on process-global
    working-directory changes.
    """

    if bbox_format not in ("cxcywh_norm", "cxcywh_abs", "xywh_abs", "xyxy_abs"):
        raise InputError(
            f"unsupported bbox_format: {bbox_format}",
            code="E_BBOX_FORMAT",
        )
    if max_images is not None and (
        not isinstance(max_images, int)
        or isinstance(max_images, bool)
        or max_images <= 0
    ):
        raise InputError("max_images must be a positive integer", code="E_MAX_IMAGES")
    if assume_class_id_is_category_id and classes is None:
        raise InputError(
            "classes is required when assume_class_id_is_category_id is enabled",
            code="E_CLASSES_REQUIRED",
        )

    dataset_path = _resolve_path(dataset, base_dir=base_dir, field="dataset")
    try:
        manifest = build_manifest(dataset_path, split=split)
    except Exception as exc:
        raise DatasetError(
            f"could not load dataset: {dataset_path}: {exc}",
            code="E_DATASET_READ",
            details={"dataset": str(dataset_path), "split": split},
        ) from exc

    full_records = list(manifest.get("images") or [])
    if not full_records:
        raise DatasetError(
            f"no dataset images resolved for split={manifest.get('split')!r} under {dataset_path}",
            code="E_DATASET_EMPTY",
            details={"dataset": str(dataset_path), "split": manifest.get("split")},
        )
    selected_records = full_records if max_images is None else full_records[: int(max_images)]
    if not selected_records:
        raise DatasetError(
            "the selected dataset subset is empty",
            code="E_DATASET_EMPTY",
        )

    raw_payload, predictions_label = _load_predictions_source(predictions, base_dir=base_dir)
    try:
        raw_entries, raw_meta = normalize_predictions_payload(raw_payload)
        wrapper_warnings = validate_payload_schema_version(raw_payload, artifact="predictions")
        if raw_meta is not None:
            validate_wrapped_meta(raw_meta)
    except (TypeError, ValueError) as exc:
        raise PredictionsValidationError(str(exc)) from exc
    if not raw_entries:
        raise PredictionsValidationError(
            "no prediction entries found",
            code="E_PREDICTIONS_EMPTY",
        )

    normalization_warnings: list[str] = []
    if classes is not None:
        _validate_class_normalization_shape(raw_entries)
        classes_path = _resolve_path(classes, base_dir=base_dir, field="classes")
        try:
            classes_payload = load_classes_json(classes_path)
            transformed = normalize_class_ids(
                raw_entries,
                classes_json=classes_payload,
                assume_class_id_is_category_id=assume_class_id_is_category_id,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise InputError(
                f"could not normalize classes from {classes_path}: {exc}",
                code="E_CLASSES_READ",
                details={"classes": str(classes_path)},
            ) from exc
        raw_entries = transformed.entries
        normalization_warnings.extend(transformed.warnings)
        normalization_classes = str(classes_path)
    else:
        normalization_classes = None

    wrapped_for_validation: JsonObject = {"schema_version": 1, "predictions": raw_entries}
    if raw_meta is not None:
        wrapped_for_validation["meta"] = raw_meta
    validated = _canonicalize_payload(wrapped_for_validation, repair=repair)
    validation_warnings = [*wrapper_warnings, *validated.warnings]

    full_aliases = _dataset_aliases(full_records)
    selected_aliases = _dataset_aliases(selected_records)
    selected_entries: list[JsonObject] = []
    selected_dataset_ids_with_predictions: set[int] = set()
    excluded_images = 0
    excluded_detections = 0

    for index, entry in enumerate(validated.entries):
        image = require_image_key(entry.get("image"), where=f"predictions[{index}].image")
        if lookup_image_alias(full_aliases, image) is None:
            raise PredictionsValidationError(
                f"prediction refers to image not present in the full dataset: {image}",
                code="E_PREDICTION_UNKNOWN_IMAGE",
                details={"image": image},
            )
        selected_id = lookup_image_alias(selected_aliases, image)
        if selected_id is None:
            excluded_images += 1
            excluded_detections += _entry_detection_count(entry)
            continue
        selected_dataset_ids_with_predictions.add(selected_id)
        selected_entries.append(copy.deepcopy(entry))

    warnings = [*validation_warnings, *normalization_warnings]
    if excluded_images:
        warnings.append(
            f"max_images subset excluded {excluded_images} prediction image(s) and "
            f"{excluded_detections} detection(s) for known but unselected dataset images"
        )

    missing_selected = len(selected_records) - len(selected_dataset_ids_with_predictions)
    if missing_selected:
        warnings.append(
            f"{missing_selected} selected dataset image(s) have no prediction entry; "
            "they are evaluated as zero detections"
        )

    try:
        ground_truth, coco_index = build_coco_ground_truth(selected_records)
        image_sizes = {
            int(image["id"]): (int(image["width"]), int(image["height"]))
            for image in ground_truth["images"]
        }
        detections = predictions_to_coco_detections(
            selected_entries,
            coco_index=coco_index,
            image_sizes=image_sizes,
            bbox_format=bbox_format,
        )
    except (TypeError, ValueError) as exc:
        raise PredictionsValidationError(str(exc)) from exc
    except Exception as exc:
        raise EvaluationError(str(exc), code="E_EVALUATION_PREPARE") from exc

    if dry_run:
        metric_values: JsonObject = {
            "map50_95": None,
            "map50": None,
            "map75": None,
            "ar100": None,
        }
        stats: Sequence[float] = ()
    else:
        try:
            metric_result = evaluate_coco_map(ground_truth, detections, quiet=True)
        except RuntimeError as exc:
            if "pycocotools" in str(exc).lower():
                raise OptionalDependencyError(
                    str(exc),
                    details={"extra": "coco"},
                ) from exc
            raise EvaluationError(str(exc)) from exc
        except Exception as exc:
            raise EvaluationError(str(exc)) from exc
        metric_values = dict(metric_result.get("metrics") or {})
        stats = tuple(float(value) for value in (metric_result.get("stats") or []))

    detections_input = sum(_entry_detection_count(entry) for entry in validated.entries)
    counts = EvaluationCounts(
        dataset_images_total=len(full_records),
        images=len(selected_records),
        prediction_images_total=len(validated.entries),
        prediction_images_evaluated=len(selected_entries),
        prediction_images_excluded=excluded_images,
        selected_images_without_predictions=missing_selected,
        detections_input=detections_input,
        detections=len(detections),
        detections_excluded=excluded_detections,
    )
    metrics = CocoMetrics(
        map50_95=_optional_float(metric_values.get("map50_95")),
        map50=_optional_float(metric_values.get("map50")),
        map75=_optional_float(metric_values.get("map75")),
        ar100=_optional_float(metric_values.get("ar100")),
    )
    return CocoEvaluationResult(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        dataset=str(dataset_path),
        split=str(manifest.get("split")),
        split_requested=split,
        predictions=predictions_label,
        bbox_format=bbox_format,
        max_images=int(max_images) if max_images is not None else None,
        dry_run=bool(dry_run),
        repair=bool(repair),
        normalization_classes=normalization_classes,
        assume_class_id_is_category_id=bool(assume_class_id_is_category_id),
        metrics=metrics,
        stats=tuple(float(value) for value in stats),
        counts=counts,
        warnings=_unique_warnings(warnings),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _failure_report(
    error: APIError,
    *,
    dataset: str,
    predictions: str,
    split: str | None,
    bbox_format: str,
    max_images: int | None,
    dry_run: bool,
    repair: bool,
) -> JsonObject:
    """Build a non-success report so a stale success artifact is not retained."""

    return {
        "report_schema_version": 1,
        "status": "failed",
        "ok": False,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": dataset,
        "split": split,
        "split_requested": split,
        "predictions": predictions,
        "bbox_format": bbox_format,
        "max_images": max_images,
        "normalization": {
            "classes": None,
            "assume_class_id_is_category_id": False,
        },
        "validation": {
            "mode": "repair" if repair else "strict",
            "repair_enabled": repair,
        },
        "metrics": {
            "map50_95": None,
            "map50": None,
            "map75": None,
            "ar100": None,
        },
        "stats": [],
        "dry_run": dry_run,
        "counts": {
            "dataset_images_total": 0,
            "images": 0,
            "prediction_images_total": 0,
            "prediction_images_evaluated": 0,
            "prediction_images_excluded": 0,
            "selected_images_without_predictions": 0,
            "detections_input": 0,
            "detections": 0,
            "detections_excluded": 0,
        },
        "warnings": [],
        "error": error.to_dict(),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
