# YOLOZU Spec (repo feature summary)

## Purpose
YOLOZU is a framework-agnostic evaluation toolkit for vision models,
designed to support reproducible continual learning and test-time adaptation (TTT) under **domain shift**.

YOLOZU helps mitigate catastrophic forgetting by enabling reproducible workflows
(e.g., self-distillation, replay, parameter-efficient updates) and by making forgetting
measurable and comparable across runs. It does not guarantee elimination of forgetting.

It treats **predictions—not models—as the stable interface contract**, so continual learning
and test-time training are comparable, restartable, and CI-friendly across frameworks and runtimes.

The repo also includes an optional in-repo reference trainer (`rtdetr_pose`) for real-time monocular RGB
object detection + depth + 6DoF pose, while keeping training implementations decoupled from the evaluation surface.

It emphasizes:

- CPU-minimum development/tests (GPU optional)
- precomputed inference JSON ingestion
- reproducible evaluation
- model-family-agnostic interface contracts (predictions/eval/reporting)

## Core capabilities

### 1) Dataset I/O (YOLO format)

- Image layout: `images/<split>/*.{jpg,jpeg,png,bmp,tif,tiff,webp,gif}`
- Labels: `labels/<split>/*.txt` (YOLO: `class cx cy w h`, normalized)
- Optional metadata: `labels/<split>/<image>.json`
  - masks/seg: `mask_path` / `mask` / `M`
  - depth: `depth_path` / `depth` / `D_obj`
  - pose: `R_gt` / `t_gt` (or `pose`)
  - intrinsics: `K_gt` / `intrinsics`

### 2) Mask-only label derivation

If YOLO txt labels are missing and a mask is provided, bbox+class can be derived from masks
(implemented in `yolozu.datasets.dataset`).

- Color mask (RGB): unique colors become classes (optionally `mask_class_map`)
- Instance mask (single-channel IDs): non-zero IDs become instances; class via
  `mask_class_id` (or `mask_class_map`)

### 3) Reference trainer (reference adapter: `rtdetr_pose`)

- Reference training entrypoint: `rtdetr_pose/tools/train_minimal.py`
- Production-style run interface contract (Run Contract; `--run-contract`): fixed artifact paths under
  `runs/<run_id>/...`, full resume, export + parity gate
- Optional Hungarian matcher with staged cost terms
- MIM masking + teacher distillation schedules
- Denoising target augmentation
- Optional LoRA (Linear) for parameter-efficient finetuning (head-only by default)
- Optimizers: AdamW / SGD
- LR warmup + schedules (`none`, `linear`, `cos`)
- Progress bar + per-step loss logging
- Metrics outputs: JSONL/CSV + TensorBoard logging
- Default ONNX export at end of training

### 3.1) Backbone/neck swap boundary (adapter-scoped)

The repository-wide interface contracts are model-family agnostic, but the in-repo training
adapter (`rtdetr_pose`) defines a strict backbone boundary:

- `model.backbone.*` selects the backbone implementation
- backbone must output `P3/P4/P5` with strides `[8,16,32]`
- projector + neck/encoder keep a fixed transformer input interface contract (`d_model`)

Current in-repo `rtdetr_pose` backbone choices:

- `cspresnet`
- `cspdarknet_s`
- `tiny_cnn`
- `resnet50` (torchvision)
- `convnext_tiny` (torchvision)

### 4) Inference + constraints

- Constraint evaluation: depth prior, plane, upright constraints
- Translation recovery from bbox/offsets + corrected intrinsics
- Inference utilities for constraints + template verification

### 5) Template verification + gating

- Symmetry-aware template scoring utilities
- Low-FP gate via `score_tmp_sym`

### 6) Predictions JSON interface contract

- Stable schema for evaluation ingestion
- Supported shapes: list entries, wrapped object, or map (`image -> detections`)
- See `docs/predictions_schema.md` and `schemas/predictions.schema.json`

### 7) Evaluation harness

- COCO eval conversion from YOLO labels and predictions JSON
- NMS-free e2e mAP evaluation
- Scenario suite report (fps/recall/depth/pose/rejection)

### 8) Test-time adaptation (TTA / TTT)

- TTA transforms (flip-based) with logging (prediction-space post-transform)
- TTT (Tent/MIM) integrated into `tools/export_predictions.py` via `--ttt`
  - runs strictly pre-prediction to keep output schema unchanged
  - with `--wrap`, writes `meta.ttt` including config + report (losses, updated params,
    MIM mask ratio)
- Interface notes: `docs/ttt_integration_plan.md`

### 9) CLI convenience

Installed CLI:

- `yolozu doctor`
- `yolozu export`
- `yolozu validate`
- `yolozu eval-instance-seg`
- `yolozu resources`
- `yolozu demo`
- `yolozu test`

These commands are backend-facing and can evaluate predictions produced by external
YOLO/RT-DETR/other model families as long as outputs follow the predictions schema.

Reference training lane:

- `yolozu train ...` (requires `yolozu[train]`; defaults to the RT-DETR pose reference trainer)

Optional extra:

- `yolozu onnxrt export ...` (install `yolozu[onnxrt]`)

Canonical CLI:

- `yolozu ...`
- `python3 -m yolozu ...`

Legacy compatibility wrapper (source checkout only):

- `python3 tools/yolozu.py ...`

## Contracts

- Predictions schema: `docs/predictions_schema.md`
- Adapter interface contract: `docs/adapter_contract.md`

## Non-goals

- Full training framework for every model family (this repo focuses on reference training lanes + artifact layout + run interface contract)
- Heavy dependencies required for local evaluation

## Versioning

- Internal schema versioning for predictions JSON (v1)
- Backward-compatible additions are allowed
- Breaking changes require version bump
- 1.0 interface contract stability boundary: `docs/release_1_0_stability.md`
