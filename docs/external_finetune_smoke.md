# External finetune smoke (YOLOX/Ultralytics/MMDetection/Detectron2/RT-DETR)

This page defines a practical smoke workflow to check whether finetune entrypoints are usable across five framework paths:

- YOLOX (Apache-2.0 primary external lane)
- YOLOv (YOLO-family runtime)
- MMDetection
- Detectron2
- RT-DETR (`rtdetr_pose` in-repo)

Top-level `yolozu train` also supports first-class external lanes for `mmdetection`,
`mmpose`, and `mmseg`. This smoke page stays narrower on purpose: it focuses on
the launcher paths that already have repo-backed smoke coverage.

The focus is a stable interface contract for reproducible command inputs/outputs and a machine-readable report.
YOLOX is the recommended external YOLO-style lane because it preserves YOLOZU's
Apache-2.0 repository policy more cleanly than optional copyleft bridges.

## One-command qualification

To run the repository-local five-stage trainer, attempt all five smoke-matrix
frameworks non-dry, probe the wider advertised runtime surface, and write one
schema-defined decision:

```bash
./.venv/bin/python tools/qualify_finetune_lanes.py \
  --output-dir /tmp/yolozu-finetune-qualification
```

The machine-readable entrypoint is
`<output-dir>/qualification_summary.json`, defined by
[`schemas/finetune_lane_qualification.schema.json`](schemas/finetune_lane_qualification.schema.json).
The command refuses an existing output path and a dataset outside the
repository. It records source/environment/dataset hashes, per-stage checkpoint
handoffs, runtime/memory, dependency probes, actual training execution, metric
scope, and an explicit `hold` decision.

`non-dry` now means a training command must actually run. Config projection
without an external launcher fails with
`E_EXTERNAL_TRAIN_SCRIPT_REQUIRED`; it is not counted as training.

## Config templates

Prepared templates live in:

- `configs/examples/finetune_external/yolox_s_finetune_smoke.py`
- `configs/examples/finetune_external/yolo_runtime_yolov8n_finetune_smoke.yaml`
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
  --non-dry-framework yolox \
  --non-dry-framework yolov \
  --non-dry-framework rtdetr \
  --yolox-train-script /path/to/YOLOX/tools/train.py \
  --epochs 1 \
  --max-steps 1 \
  --batch-size 2 \
  --image-size 96 \
  --device cpu \
  --require-training-execution \
  --output reports/external_finetune_smoke.exec.json
```

If torch is missing for RT-DETR non-dry execution, the report returns:

- `failure_code: E_DEP_TORCH_MISSING`
- `runtime_error`: explicit message with dependency probe detail

This makes the failure explicit instead of a generic train-command failure.

For YOLOX, projection can run even without the YOLOX package installed because the
smoke exp template is clean-room and self-contained. Real training still requires
your local external YOLOX launcher.

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

When projection dependencies (`mmengine`, `detectron2`) are missing, the tool still audits the actual train path if script paths are provided:

- `train_path_audited: true` when external launcher path was checked and executed
- `projection_executed: false` with `projection_error` populated
- `training_executed` reflects the actual external training command result

This keeps train-path auditing usable on minimal environments.

Without a train script, a non-dry MMDetection, Detectron2, or YOLOX selection
fails closed even if projection succeeds. Missing projection dependencies are
reported separately in `dependency_status`.

## 4) machine.dev / GPU validation

For GPU environments (for example `machine.dev`), run non-dry checks with a CUDA device:

```bash
python3 tools/run_external_finetune_smoke.py \
  --dataset-root data/smoke \
  --split train \
  --non-dry-framework rtdetr \
  --device cuda \
  --epochs 1 \
  --max-steps 1 \
  --batch-size 2 \
  --image-size 96 \
  --require-training-execution \
  --output reports/external_finetune_smoke.machine_dev.json
```

## Notes

- `--require-non-dry` fails when every framework is dry-run.
- `--require-training-execution` fails when no framework executed training.
- `--yolox-train-script` lets the smoke tool audit a real Apache-2.0 YOLOX train path without vendoring YOLOX into this repo.
- The report includes command lines, warnings, `failure_code`, and per-framework status fields (`projection_executed`, `projection_error`, `train_path_audited`) to support CI gating.
