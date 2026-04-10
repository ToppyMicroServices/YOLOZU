# Detectron2 / MMDetection interop

This page is the shortest path for users who keep training/inference in Detectron2 or MMDetection and only use YOLOZU for contract validation and apples-to-apples evaluation.

## Scope (what YOLOZU guarantees)

- Level 1: dataset entry via COCO JSON → YOLO-style wrapper (`dataset.json` + labels).
- Level 2: inference export via Detectron2/MMDetection → `predictions.json`.
- Level 3: fairness metadata via `export_settings` (preprocessing/protocol).

YOLOZU does not require replacing your training framework.

## 1.5) External Detectron2 training lane

YOLOZU can also launch a Detectron2 training lane through an external launcher
while still emitting the shared training summary interface contract.

Use the backend-native config to choose the task family:

- Faster R-CNN style config: `bbox`
- Mask R-CNN style config: `segmentation`
- Keypoint R-CNN style config: `keypoints`

```bash
python3 -m yolozu train \
  --external-backend detectron2 \
  configs/examples/finetune_external/detectron2_finetune_smoke.yaml \
  --dataset data/smoke \
  --split val \
  --task-family bbox \
  --dry-run \
  --output reports/train_external_detectron2_bbox.json
```

To pass dataset registration names or other Detectron2 overrides, repeat
`--train-opt KEY VALUE`. For a real run, also add
`--train-script /path/to/detectron2/tools/train_net.py`.

## 1) Prepare dataset wrapper (COCO JSON to YOLOZU)

```bash
yolozu migrate dataset \
  --from coco \
  --coco-root /path/to/coco \
  --split val2017 \
  --output data/coco_yolo_like \
  --mode manifest
```

## 2) Export predictions from Detectron2

```bash
python3 tools/export_predictions_detectron2.py \
  --dataset data/coco_yolo_like \
  --split val2017 \
  --config /path/to/d2_config.yaml \
  --weights /path/to/model_final.pth \
  --score-thr 0.25 \
  --protocol nms_applied \
  --output reports/pred_detectron2.json
```

## 3) Export predictions from MMDetection

```bash
python3 tools/export_predictions_mmdet.py \
  --dataset data/coco_yolo_like \
  --split val2017 \
  --config /path/to/mmdet_config.py \
  --checkpoint /path/to/epoch_12.pth \
  --score-thr 0.25 \
  --protocol nms_applied \
  --output reports/pred_mmdet.json
```

## 4) Validate and evaluate

```bash
python3 tools/validate_predictions.py reports/pred_detectron2.json --strict
python3 tools/eval_coco.py \
  --dataset data/coco_yolo_like \
  --split val2017 \
  --predictions reports/pred_detectron2.json \
  --protocol nms_applied \
  --classes data/coco_yolo_like/labels/val2017/classes.json \
  --output reports/coco_eval_detectron2.json
```

Use the same pattern for MMDetection predictions.

## Pitfalls to avoid (critical)

- `category_id` vs `class_id`: external COCO outputs often use category IDs; pass `--classes .../classes.json` during eval normalization.
- Preprocessing drift: record resize/pad/normalize/BGR-RGB assumptions in exporter flags so comparisons are reproducible.
- NMS policy mismatch: Detectron2/MMDet exports are usually post-NMS. Use `--protocol nms_applied` unless you explicitly export NMS-free outputs.
- BBox representation: exporters normalize Detectron2/MMDet raw `xyxy_abs` boxes to YOLOZU schema (`cxcywh_norm`) and preserve raw format in `export_settings.raw_output_bbox_format`.

## Finetune smoke matrix

To audit finetune entrypoints and emit a unified interface contract report:

```bash
python3 tools/run_external_finetune_smoke.py \
  --dataset-root data/smoke \
  --split train \
  --output reports/external_finetune_smoke.json
```

For external launcher wiring (`--mmdet-train-script`, `--detectron2-train-script`), see `docs/external_finetune_smoke.md`.
