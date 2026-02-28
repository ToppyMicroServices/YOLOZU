# YOLO26 evaluation protocol (COCO Detect, e2e mAP)

This repository uses a pinned evaluation protocol for comparing against **YOLO26** targets.

Protocol file: `protocols/yolo26_eval.json`

Schema/version metadata:

- `protocol_schema_version`: `1`
- `protocol_hash`: SHA-256 over canonical protocol JSON (emitted in suite report)

## Canonical settings

- Task: `detect`
- Dataset: COCO
- Split: `val2017`
- Image size: `640` (exporter responsibility; evaluator does not resize)
- BBox format: `cxcywh_norm` (normalized `[0,1]` center-x/center-y/width/height)
- Metric key: `map50_95`
- Metric meaning: e2e mAP@[.5:.95] (bbox, no NMS)

### Fixed-condition comparison rules

`tools/eval_suite.py --protocol yolo26` validates each predictions artifact against
the protocol's `fixed_conditions`:

- `imgsz = 640`
- `score_threshold = 0.001`
- `iou_threshold = 0.7`
- `max_detections = 300`
- `bbox_format = cxcywh_norm`
- preprocess must match `letterbox + RGB + 0_1 + linear + fill[114,114,114]`

If any artifact deviates, `eval_suite` exits non-zero.

## e2e mAP (no NMS)

“End-to-end” means **the evaluator scores detections exactly as provided**; it does not apply NMS.
If your exporter runs NMS, you are not measuring e2e performance.
### `--protocol` values and e2e semantics

| `--protocol` value | `e2e.nms` | Meaning |
|---|---|---|
| `yolo26` | `none` | YOLO26 competition: RT-DETR-style NMS-free output. Evaluator receives raw detections. |
| `e2e_nms_free` | `none` | Generic NMS-free protocol (same e2e semantics, different fixed conditions may apply). |
| `nms_applied` | `applied` | Exporter already ran NMS (YOLOv5/v8/v11 style). `score_threshold=0.25`, `iou_threshold=0.45`. |

The evaluator **never applies NMS** regardless of protocol. The `e2e.nms` field records
whether the *exporter* applied NMS, so downstream consumers can distinguish the two modes
programmatically. This field is propagated into evaluation report JSON.
## Reporting schema

Evaluation outputs are JSON and include protocol metadata:

- Single file: `reports/coco_eval.json` (from `tools/eval_coco.py`)
- Multi-file suite: `reports/eval_suite.json` (from `tools/eval_suite.py`)

Both reports include:
- `protocol_id` / `protocol`
- `protocol_schema_version` / `protocol_hash`
- `dataset`, `split` (effective), `split_requested`
- `bbox_format`

## Targets

Targets live in `baselines/yolo26_targets.json` and are validated by:

`python3 tools/validate_map_targets.py --targets baselines/yolo26_targets.json`

## Pre-PR quality checklist (local)

Run this single command before opening a PR:

`bash scripts/pre_pr_quality.sh`

What it runs (CI-aligned local gates):

- Smoke gate: `bash scripts/smoke.sh`
- Lint gate: `python3 -m ruff check tools yolozu tests`
- Focused tests:
	- `tests/test_eval_suite_export_settings.py`
	- `tests/test_eval_suite_determinism.py`
	- `tests/test_check_map_targets_contract.py`
	- `tests/test_yolo26_protocol.py`
	- `tests/test_predictions.py`

Expected artifact:

- `reports/smoke_coco_eval_dry_run.json` exists after the checklist

Pass criteria:

- Command exits with status `0`
- Final line prints `pre-PR quality checklist passed`

