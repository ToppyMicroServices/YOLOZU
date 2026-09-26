"""Local-only COCO bounding-box evaluator for adaptive qualification.

This is deliberately not an implementation of official COCOeval. It binds a
bounded, workspace-confined COCO instances file to YOLOZU's existing simple
mAP implementation and evaluates only the pinned input basenames.
"""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from yolozu.eval import simple_map as _simple_map

from ..bundles import AlgorithmBundleSpec
from ..canonical import canonical_sha256_v1
from ..contracts import HANDOFF_MAX_OUTPUT_BYTES, ImageJobSpec
from ..control_stream import (
    read_control_stream_bytes,
    resolve_confined_regular_file,
)
from ..evidence import HANDOFF_ID, HANDOFF_VERSION
from ..inventory import PinnedDecodedInputSet

COCO_BBOX_SIMPLE_MAP_EVALUATOR_ID = "coco_bbox_simple_map_v1"
_EVALUATOR_VERSION = "1"
_METRIC_ID = "simple_bbox_map50_95"
_MAX_GROUND_TRUTH_BYTES = 128 * 1024 * 1024
_MAX_IMAGES = 100_000
_MAX_ANNOTATIONS = 1_000_000
_MAX_CATEGORIES = 10_000
_IOU_THRESHOLDS = tuple(Decimal(value) / Decimal(100) for value in range(50, 100, 5))


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


EVALUATOR_SOURCE_DIGEST = _source_sha256(Path(__file__))
_SIMPLE_MAP_SOURCE_DIGEST = _source_sha256(Path(str(_simple_map.__file__)))


def _semantic_identity() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evaluator_id": COCO_BBOX_SIMPLE_MAP_EVALUATOR_ID,
        "evaluator_version": _EVALUATOR_VERSION,
        "metric_id": _METRIC_ID,
        "ground_truth": {
            "format": "coco_instances_json",
            "selection": "exact_pinned_input_basename",
            "crowd_policy": "exclude_iscrowd_1",
            "box_conversion": "absolute_xywh_to_normalized_cxcywh",
            "category_binding": "exact_bundle_vocabulary_label",
        },
        "predictions": {
            "format": f"{HANDOFF_ID}_v{HANDOFF_VERSION}",
            "box_conversion": "normalized_xyxy_to_normalized_cxcywh",
            "class_binding": "request_index_with_exact_full_bundle_vocabulary",
            "source": "same_qualification_run",
        },
        "matching": {
            "implementation": "yolozu.eval.simple_map.evaluate_map",
            "evaluator_source_sha256": EVALUATOR_SOURCE_DIGEST,
            "simple_map_source_sha256": _SIMPLE_MAP_SOURCE_DIGEST,
            "iou_thresholds": [str(value) for value in _IOU_THRESHOLDS],
            "aggregation": "simple_map_unweighted_class_mean",
        },
    }


EVALUATION_PROTOCOL_SHA256 = canonical_sha256_v1(_semantic_identity())


def _reject_constant(value: str) -> None:
    raise ValueError(f"ground truth contains invalid numeric constant {value!r}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"ground truth contains duplicate object key {key!r}")
        output[key] = value
    return output


def _load_ground_truth(path: Path, workspace: Path) -> tuple[bytes, Mapping[str, Any]]:
    workspace = workspace.resolve(strict=True)
    confined = resolve_confined_regular_file(
        path,
        workspace=workspace,
        label="COCO ground truth",
    )
    payload = read_control_stream_bytes(
        confined,
        maximum_bytes=_MAX_GROUND_TRUTH_BYTES,
        label="COCO ground truth",
    )
    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("COCO ground truth must be strict UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("COCO ground truth root must be an object")
    return payload, decoded


def _bounded_list(value: Any, *, field: str, maximum: int) -> list[Any]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"COCO ground truth {field} exceeds its bounded list contract")
    return value


def _exact_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"COCO ground truth {field} must be an integer >= {minimum}")
    return value


def _finite_decimal(value: Any, *, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"COCO ground truth {field} must be numeric")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"COCO ground truth {field} must be numeric") from exc
    if not number.is_finite():
        raise ValueError(f"COCO ground truth {field} must be finite")
    return number


def _decimal_text(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("simple mAP result must be finite")
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text or "0"


class CocoBBoxSimpleMapEvaluator:
    """Evaluate exact pinned images against a local COCO instances file."""

    evaluator_id = COCO_BBOX_SIMPLE_MAP_EVALUATOR_ID
    evaluator_version = _EVALUATOR_VERSION
    source_digest = EVALUATOR_SOURCE_DIGEST
    metric_id = _METRIC_ID
    direction = "higher_is_better"
    evaluation_protocol_sha256 = EVALUATION_PROTOCOL_SHA256

    def __init__(
        self,
        *,
        ground_truth_path: Path,
        workspace: Path,
        inputs: PinnedDecodedInputSet,
        bundle: AlgorithmBundleSpec,
    ) -> None:
        payload, decoded = _load_ground_truth(ground_truth_path, workspace)
        self.evaluation_dataset_id = "local_coco_instances_v1"
        self.evaluation_dataset_sha256 = hashlib.sha256(payload).hexdigest()

        bundle_record = bundle.to_dict()
        vocabulary = bundle_record.get("class_vocabulary") or {}
        vocabulary_id = vocabulary.get("id")
        labels = vocabulary.get("labels")
        if not isinstance(vocabulary_id, str) or not isinstance(labels, list) or not labels:
            raise ValueError("COCO evaluator requires an inline fixed-class bundle vocabulary")
        self.evaluation_vocabulary_id = vocabulary_id
        self._labels = tuple(str(item) for item in labels)
        self._label_to_index = {label: index for index, label in enumerate(self._labels)}
        if len(self._label_to_index) != len(self._labels):
            raise ValueError("bundle vocabulary contains duplicate labels")

        self._input_basenames = tuple(item.source_basename for item in inputs.inputs)
        if len(set(self._input_basenames)) != len(self._input_basenames):
            raise ValueError("pinned input basenames must be unique")
        self._records = self._prepare_records(decoded, inputs)

    def _prepare_records(
        self,
        decoded: Mapping[str, Any],
        inputs: PinnedDecodedInputSet,
    ) -> list[dict[str, Any]]:
        images = _bounded_list(decoded.get("images"), field="images", maximum=_MAX_IMAGES)
        annotations = _bounded_list(
            decoded.get("annotations"),
            field="annotations",
            maximum=_MAX_ANNOTATIONS,
        )
        categories = _bounded_list(
            decoded.get("categories"),
            field="categories",
            maximum=_MAX_CATEGORIES,
        )

        category_by_id: dict[int, int] = {}
        category_names: set[str] = set()
        for index, raw in enumerate(categories):
            if not isinstance(raw, dict):
                raise ValueError(f"COCO ground truth categories[{index}] must be an object")
            category_id = _exact_int(raw.get("id"), field=f"categories[{index}].id", minimum=1)
            name = raw.get("name")
            if not isinstance(name, str) or name not in self._label_to_index:
                raise ValueError("COCO categories must map exactly into the bundle vocabulary")
            if category_id in category_by_id or name in category_names:
                raise ValueError("COCO categories contain a duplicate ID or label")
            category_by_id[category_id] = self._label_to_index[name]
            category_names.add(name)
        if category_names != set(self._labels):
            raise ValueError("COCO categories do not exactly match the bundle vocabulary")

        image_by_id: dict[int, tuple[str, int, int, int]] = {}
        selected_by_name: dict[str, tuple[int, int, int]] = {}
        requested = set(self._input_basenames)
        for index, raw in enumerate(images):
            if not isinstance(raw, dict):
                raise ValueError(f"COCO ground truth images[{index}] must be an object")
            image_id = _exact_int(raw.get("id"), field=f"images[{index}].id", minimum=1)
            name = raw.get("file_name")
            width = _exact_int(raw.get("width"), field=f"images[{index}].width", minimum=1)
            height = _exact_int(raw.get("height"), field=f"images[{index}].height", minimum=1)
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("COCO image file_name must be one plain basename")
            if image_id in image_by_id:
                raise ValueError("COCO images contain a duplicate image ID")
            image_by_id[image_id] = (name, width, height, index)
            if name in requested:
                if name in selected_by_name:
                    raise ValueError("COCO images contain a duplicate selected basename")
                selected_by_name[name] = (image_id, width, height)
        if set(selected_by_name) != requested:
            raise ValueError("COCO ground truth does not contain every pinned input basename")

        records = [
            {"image": basename, "labels": []}
            for basename in self._input_basenames
        ]
        record_by_name = {record["image"]: record for record in records}
        for index, basename in enumerate(self._input_basenames):
            _image_id, width, height = selected_by_name[basename]
            observed = inputs.inventory.inputs[index]
            if observed.width != width or observed.height != height:
                raise ValueError("COCO image dimensions do not match a pinned input")

        selected_ids = {value[0] for value in selected_by_name.values()}
        for index, raw in enumerate(annotations):
            if not isinstance(raw, dict):
                raise ValueError(f"COCO ground truth annotations[{index}] must be an object")
            image_id = _exact_int(raw.get("image_id"), field=f"annotations[{index}].image_id", minimum=1)
            if image_id not in selected_ids:
                continue
            if _exact_int(raw.get("iscrowd", 0), field=f"annotations[{index}].iscrowd") not in {0, 1}:
                raise ValueError("COCO iscrowd must be 0 or 1")
            if raw.get("iscrowd", 0) == 1:
                continue
            category_id = _exact_int(raw.get("category_id"), field=f"annotations[{index}].category_id", minimum=1)
            if category_id not in category_by_id:
                raise ValueError("COCO annotation references an unknown category")
            bbox = raw.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("COCO annotation bbox must contain four numbers")
            x, y, width, height = (
                _finite_decimal(value, field=f"annotations[{index}].bbox")
                for value in bbox
            )
            image_name, image_width, image_height, _source_index = image_by_id[image_id]
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError("COCO annotation bbox is outside the positive image plane")
            if x + width > image_width or y + height > image_height:
                raise ValueError("COCO annotation bbox exceeds the declared image dimensions")
            record_by_name[image_name]["labels"].append(
                {
                    "class_id": category_by_id[category_id],
                    "cx": float((x + width / 2) / image_width),
                    "cy": float((y + height / 2) / image_height),
                    "w": float(width / image_width),
                    "h": float(height / image_height),
                }
            )
        if not any(record["labels"] for record in records):
            raise ValueError("COCO ground truth has no non-crowd boxes for the pinned inputs")
        return records

    def evaluate(
        self,
        *,
        predictions: tuple[bytes, ...],
        job: ImageJobSpec,
        bundle: AlgorithmBundleSpec,
    ) -> str:
        if len(predictions) != len(self._records):
            raise ValueError("prediction count does not match the pinned evaluation inputs")
        job_record = job.to_dict()
        bundle_record = bundle.to_dict()
        vocabulary = bundle_record.get("class_vocabulary") or {}
        if (
            job_record.get("task") != "object_detection"
            or job_record.get("prompt_mode") != "fixed_classes"
            or tuple(job.prompt_phrases) != self._labels
            or vocabulary.get("id") != self.evaluation_vocabulary_id
            or tuple(vocabulary.get("labels") or ()) != self._labels
        ):
            raise ValueError("COCO evaluator requires the exact full bundle vocabulary")

        entries: list[dict[str, Any]] = []
        for index, payload in enumerate(predictions):
            if len(payload) > HANDOFF_MAX_OUTPUT_BYTES:
                raise ValueError("prediction handoff exceeds its byte limit")
            try:
                handoff = json.loads(
                    payload,
                    object_pairs_hook=_object_without_duplicates,
                    parse_constant=_reject_constant,
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("prediction handoff must be strict UTF-8 JSON") from exc
            if not isinstance(handoff, dict) or set(handoff) != {
                "schema_version",
                "handoff_id",
                "handoff_version",
                "input_index",
                "task",
                "results",
            }:
                raise ValueError("prediction handoff keys do not match the evaluator contract")
            if (
                handoff["schema_version"] != 1
                or handoff["handoff_id"] != HANDOFF_ID
                or handoff["handoff_version"] != HANDOFF_VERSION
                or handoff["input_index"] != index
                or handoff["task"] != "object_detection"
                or not isinstance(handoff["results"], list)
            ):
                raise ValueError("prediction handoff identity does not match the evaluator contract")
            detections: list[dict[str, Any]] = []
            for result_index, result in enumerate(handoff["results"]):
                if not isinstance(result, dict) or set(result) != {
                    "result_index",
                    "score",
                    "bbox",
                    "request_index",
                    "label",
                }:
                    raise ValueError("prediction result keys do not match the evaluator contract")
                request_index = _exact_int(
                    result["request_index"],
                    field=f"predictions[{index}].results[{result_index}].request_index",
                )
                if (
                    result["result_index"] != result_index
                    or request_index >= len(self._labels)
                    or result["label"] != self._labels[request_index]
                ):
                    raise ValueError("prediction result class identity is inconsistent")
                bbox = result["bbox"]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError("prediction bbox must contain four values")
                try:
                    x1, y1, x2, y2 = (Decimal(value) for value in bbox)
                    score = Decimal(result["score"])
                except (InvalidOperation, TypeError) as exc:
                    raise ValueError("prediction values must be canonical decimal strings") from exc
                if (
                    not all(value.is_finite() for value in (x1, y1, x2, y2, score))
                    or not (0 <= x1 < x2 <= 1)
                    or not (0 <= y1 < y2 <= 1)
                    or not (0 <= score <= 1)
                ):
                    raise ValueError("prediction values are outside the normalized contract")
                detections.append(
                    {
                        "class_id": request_index,
                        "score": float(score),
                        "bbox": {
                            "cx": float((x1 + x2) / 2),
                            "cy": float((y1 + y2) / 2),
                            "w": float(x2 - x1),
                            "h": float(y2 - y1),
                        },
                    }
                )
            entries.append(
                {
                    "image": self._input_basenames[index],
                    "detections": detections,
                }
            )
        result = _simple_map.evaluate_map(
            self._records,
            entries,
            iou_thresholds=[float(value) for value in _IOU_THRESHOLDS],
        )
        return _decimal_text(result.map50_95)


def create_coco_bbox_simple_map_evaluator(
    ground_truth_path: Path,
    workspace: Path,
    inputs: PinnedDecodedInputSet,
    bundle: AlgorithmBundleSpec,
) -> CocoBBoxSimpleMapEvaluator:
    return CocoBBoxSimpleMapEvaluator(
        ground_truth_path=ground_truth_path,
        workspace=workspace,
        inputs=inputs,
        bundle=bundle,
    )
