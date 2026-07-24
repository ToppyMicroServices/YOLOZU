# Stable Python API

`yolozu.api` is the supported in-process surface for validating and evaluating
detection predictions. It does not launch a subprocess, change the process
working directory, or write a report unless the caller explicitly serializes
the returned result.

## Shortest CLI path

`eval-coco` now performs strict predictions validation itself, so a separate
`validate predictions` command is optional. The shortest core-install path is:

```bash
yolozu eval-coco \
  --dataset /absolute/path/to/dataset \
  --predictions /absolute/path/to/predictions.json \
  --dry-run \
  --output reports/coco_eval.json
```

This validates, converts, and reports subset counts without installing
`pycocotools`. Invalid input exits nonzero and replaces any report at
`--output` with `status: failed`; it cannot leave a stale success report behind.

For real COCO metrics:

```bash
python3 -m pip install 'yolozu[coco]'
yolozu eval-coco \
  --dataset /absolute/path/to/dataset \
  --predictions /absolute/path/to/predictions.json \
  --output reports/coco_eval.json
```

Evaluation is fail-closed by default. Legacy range coercion is available only
through `--repair`, and every clamp or migration is listed in `warnings`:

```bash
yolozu eval-coco \
  --dataset /absolute/path/to/dataset \
  --predictions /absolute/path/to/legacy_predictions.json \
  --dry-run \
  --repair \
  --output reports/repaired_dry_run.json
```

## In-process evaluation

```python
from pathlib import Path

from yolozu.api import evaluate_coco

result = evaluate_coco(
    dataset=Path("/absolute/path/to/dataset"),
    predictions=Path("/absolute/path/to/predictions.json"),
    split="val",
    max_images=50,
    dry_run=True,
)

report = result.to_dict()
print(report["status"], report["counts"], report["warnings"])
```

For relative paths, pass an explicit absolute `base_dir`. This makes path
resolution visible to the caller:

```python
from pathlib import Path

from yolozu.api import evaluate_coco

workspace = Path("/absolute/path/to/workspace")
result = evaluate_coco(
    "data/smoke",
    "data/smoke/predictions/predictions_dummy.json",
    split="val",
    max_images=2,
    dry_run=True,
    base_dir=workspace,
)
```

## In-memory validation

```python
from yolozu.api import PredictionsInput, validate_predictions

payload = {
    "schema_version": 1,
    "predictions": [
        {
            "schema_version": 2,
            "image": "image.jpg",
            "detections": [
                {
                    "class_id": 0,
                    "score": 0.9,
                    "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                }
            ],
        }
    ],
}

validated = validate_predictions(PredictionsInput.from_payload(payload))
print(validated.to_dict())
```

`evaluate_coco` accepts the same `PredictionsInput`, a wrapped mapping, or an
entry sequence in place of a path.

## `max_images` semantics

The dataset is ordered deterministically and the first `N` records are
evaluated. Predictions for images that exist in the full dataset but fall
outside that subset are excluded and recorded in:

- `counts.prediction_images_excluded`
- `counts.detections_excluded`
- `warnings`

A prediction image that does not exist anywhere in the full dataset remains an
error (`E_PREDICTION_UNKNOWN_IMAGE`). Selected images without a prediction
entry are counted in `counts.selected_images_without_predictions` and are
evaluated as zero detections.

## Stable public symbols

This page is the source of truth for the supported Python surface:

| Symbol | Role |
|---|---|
| `PredictionsInput` | Explicit path-backed or in-memory predictions input |
| `PredictionsValidationResult` | Canonical entries, validation mode, and warnings |
| `CocoMetrics` | Typed COCO metric fields |
| `EvaluationCounts` | Full-dataset, selected-subset, prediction, and detection counts |
| `CocoEvaluationResult` | Typed result with `to_dict()` serialization |
| `validate_predictions` | Strict validation; `repair=True` is explicit opt-in |
| `evaluate_coco` | Strict validation plus dry-run conversion or real COCOeval |
| `APIError` | Base machine-readable exception with `code` and `to_dict()` |
| `InputError` | Path, JSON, or option input error |
| `DatasetError` | Dataset loading or empty-selection error |
| `PredictionsValidationError` | Predictions interface contract or image-key error |
| `EvaluationError` | Evaluation preparation or execution error |
| `OptionalDependencyError` | Missing optional dependency such as `yolozu[coco]` |

`yolozu.api` and its documented symbol names, exception categories/codes, and
serialized result keys are the compatibility surface. Additive fields may be
introduced. Internal helpers and other modules remain implementation details.
The package ships `py.typed` so type checkers can consume these annotations.

The currently emitted exception codes are:

| Code | Category / condition |
|---|---|
| `E_RELATIVE_PATH`, `E_RELATIVE_BASE_DIR` | A path needs an explicit absolute resolution base |
| `E_PREDICTIONS_SOURCE`, `E_PREDICTIONS_NOT_FOUND`, `E_PREDICTIONS_READ` | Predictions source selection or reading failed |
| `E_BBOX_FORMAT`, `E_MAX_IMAGES`, `E_CLASSES_REQUIRED`, `E_CLASSES_READ` | An evaluation option or classes mapping is invalid |
| `E_DATASET_READ`, `E_DATASET_EMPTY` | Dataset discovery failed or selected no images |
| `E_PREDICTIONS_INVALID`, `E_PREDICTIONS_EMPTY`, `E_PREDICTION_UNKNOWN_IMAGE` | Predictions violate the interface contract or dataset identity |
| `E_EVALUATION_PREPARE`, `E_EVALUATION` | COCO conversion or evaluation failed |
| `E_OPTIONAL_DEPENDENCY` | Real evaluation needs an unavailable optional dependency |

`E_API`, `E_INPUT`, and `E_DATASET` are the default codes of the corresponding
base exception categories; callers normally receive one of the more specific
codes above.

Real COCO metrics require:

```bash
python3 -m pip install 'yolozu[coco]'
```
