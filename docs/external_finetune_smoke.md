# External finetune smoke (YOLOv/MMDetection/Detectron2/RT-DETR)

This page defines a practical smoke workflow to check whether finetune entrypoints are usable across four framework paths:

- YOLOv (Ultralytics)
- MMDetection
- Detectron2
- RT-DETR (`rtdetr_pose` in-repo)

The focus is a stable interface contract for reproducible command inputs/outputs and a machine-readable report.

## Config templates

Prepared templates live in:

- `configs/examples/finetune_external/ultralytics_yolov8n_finetune_smoke.yaml`
- `configs/examples/finetune_external/mmdetection_finetune_smoke.py`
- `configs/examples/finetune_external/detectron2_finetune_smoke.yaml`
- `configs/examples/finetune_external/rtdetr_pose_finetune_smoke.yaml`

## 1) Matrix dry-run (safe default)

```bash
python3 tools/run_external_finetune_smoke.py \
  --dataset-root data/smoke \
  --split train \
  --output reports/external_finetune_smoke.json
```

What this does:

- validates template presence,
- synthesizes framework-specific finetune commands,
- writes `reports/external_finetune_smoke.json`.

## 2) Execute selected frameworks (non-dry)

Run real training where runtime dependencies are available.

```bash
python3 tools/run_external_finetune_smoke.py \
  --dataset-root data/smoke \
  --split train \
  --non-dry-framework yolov \
  --non-dry-framework rtdetr \
  --epochs 1 \
  --max-steps 1 \
  --batch-size 2 \
  --image-size 96 \
  --device cpu \
  --require-training-execution \
  --output reports/external_finetune_smoke.exec.json
```

## 3) Add MMDetection / Detectron2 training launchers

For MMDetection and Detectron2, pass your local train script paths to execute real finetune commands in addition to projection checks.

```bash
python3 tools/run_external_finetune_smoke.py \
  --dataset-root /path/to/dataset \
  --split train \
  --non-dry-framework mmdetection \
  --non-dry-framework detectron2 \
  --mmdet-train-script /path/to/mmdetection/tools/train.py \
  --detectron2-train-script /path/to/detectron2/tools/train_net.py \
  --epochs 1 \
  --batch-size 2 \
  --require-training-execution \
  --output reports/external_finetune_smoke.external.json
```

## Notes

- `--require-non-dry` fails when every framework is dry-run.
- `--require-training-execution` fails when no framework executed training.
- The report includes command lines, warnings, and per-framework status to support CI gating.
