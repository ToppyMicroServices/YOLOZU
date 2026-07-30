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

Exit code 0 means `protocol_complete=true`: all required attempts produced
machine-readable evidence. It does not mean that every external backend trained
or that promotion passed. Consumers, including agents, must read
`decision.status`, `decision.training_quality`, and the per-lane
`training_executed` / `failure_code` fields.

The 2026-07-29 clean-source bounded run is recorded in
[`reports/finetune_lane_evidence_2026-07-29.md`](../reports/finetune_lane_evidence_2026-07-29.md).

## Installed-runtime qualification (2026-07-30)

Two independent Python/Torch environments executed real training, export, and
evaluation for the three lanes available on the tested macOS CPU host:
Ultralytics 8.4.112, Transformers 5.14.1 DETR, and Detectron2 0.6. All three
predictions artifacts passed the predictions interface contract. HF DETR
produced byte-identical checkpoints and predictions; Ultralytics and
Detectron2 reproduced the same zero-detection/zero-mAP semantics.

Real non-dry launchers were also invoked for YOLOX, MMDetection, MMPose, MMSeg,
and TAO. They failed with structured environment/runtime errors; config
projection was not counted as training. The matrix now records per-lane wall
time and peak RSS and rejects launchers that print an uncaught traceback but
exit zero.

The machine-readable record and evidence boundary are:

- [`reports/external_runtime_qualification_2026-07-30.json`](../reports/external_runtime_qualification_2026-07-30.json)
- [`schemas/external_runtime_qualification.schema.json`](schemas/external_runtime_qualification.schema.json)
- [`reports/external_runtime_evidence_2026-07-30.md`](../reports/external_runtime_evidence_2026-07-30.md)

All external lanes remain Experimental: successful bounded metrics were zero,
and five runtimes were unavailable on this host.

## Compatible Linux/CUDA qualification

Run the pinned open-source runtime group on a compatible Linux/CUDA host:

```bash
bash scripts/run_external_runtime_gpu_qualification.sh \
  --output-dir reports/compatible_host_external_runtimes \
  --dataset-root data/real_multitask_fewshot
```

The command prepares bounded detection, keypoint, and segmentation layouts,
pins YOLOX and OpenMMLab runtime versions, invokes every launcher non-dry, and
writes `qualification_summary.json` following
[`schemas/compatible_host_external_runtime_qualification.schema.json`](schemas/compatible_host_external_runtime_qualification.schema.json).
It refuses an existing output directory and exits non-zero unless all four
open-source runtime lanes record actual training. NVIDIA TAO runs as a separate
vendor-container workflow step so its runtime and license boundary remain
explicit.

The dated compatible-host result is recorded separately in
[`reports/external_runtime_compatible_host_evidence_2026-07-30.md`](../reports/external_runtime_compatible_host_evidence_2026-07-30.md).
Runtime availability does not establish training quality or promote any lane
beyond Experimental.

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

For TAO and other executable-backed bridges, a missing external command is
reported as a structured runtime failure. TAO uses
`E_EXTERNAL_RUNTIME_MISSING`; the wrapper does not expose an uncaught
`FileNotFoundError`.

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
