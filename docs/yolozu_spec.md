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

This section is an implementation summary, not a production-maturity declaration. Use
[`production_readiness.md`](production_readiness.md) for current maturity boundaries and
[`ssot_capability_coverage_audit.md`](ssot_capability_coverage_audit.md) for evidence coverage.

### Capability maturity boundaries

The maturity label applies to the narrowest capability below. A Stable parent CLI or
manifest entry does not promote opt-in subcommands or flags.

| Spec capability | Maturity boundary |
|---|---|
| Dataset I/O | Deferred as a standalone capability; Stable dataset commands are not evidence that every dataset shape or loader path is production-qualified |
| Mask-only label derivation | Deferred as a standalone capability; implemented and tested, but not independently qualified |
| Reference trainer | Stable reference lane; device and runtime qualification remains environment-specific |
| Backbone/neck swap boundary | Stable only within the reference trainer interface boundary; this is not a repository-wide model-family claim |
| Inference constraints | Deferred as a standalone capability; qualify with the consuming adapter, model, and protocol |
| Template verification and gating | Deferred as a standalone capability; research gate tuning does not promote the runtime utility |
| Predictions JSON interface contract | Stable |
| Evaluation harness | Stable for validation/evaluation of existing wrapped predictions; task-specific tools retain their manifest maturity |
| BOP T-LESS object 6DoF workflow | Research; safe conversion/evaluation wiring is qualified locally, but no release-addressable real multi-seed efficacy result exists |
| TTA | Experimental and opt-in |
| TTT | Research and opt-in |
| CLI convenience | Mixed by capability; entrypoint-level maturity is not transitive to subcommands or flags |
| Searchable web onboarding | Stable generated documentation; linked commands retain their narrower capability maturity |

### 1) Dataset I/O (YOLO format)

- Image layout: `images/<split>/*.{jpg,jpeg,png,bmp,tif,tiff,webp,gif}`
- Labels: `labels/<split>/*.txt` (YOLO: `class cx cy w h`, normalized)
- Optional metadata: `labels/<split>/<image>.json`
  - masks/seg: `mask_path` / `mask` / `M`
  - depth: `depth_path` / `depth` / `D_obj`
  - pose: `R_gt` / `t_gt` (or `pose`)
  - intrinsics: `K_gt` / `intrinsics`

### 1.1) 3D and pose terminology

- 2D keypoints are image-plane `(x, y, visibility)` values.
- `kpts3d_object` means optional object-space `(X, Y, Z)` keypoints.
- Pose `R_gt` / `t_gt` means rigid-object 6DoF object-to-camera pose.
- Human 3D skeleton pose is unsupported.

The BOP T-LESS Research protocol, safe owned-output rules, CAD/ADD/ADDS
boundary, and current evidence gaps are defined in
[`bop_tless_protocol.md`](bop_tless_protocol.md).

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
- Strict-by-default `eval-coco`; explicit `--repair` records every coercion
- Deterministic `max_images` subsetting with excluded/missing prediction counts
- Typed in-process validation/evaluation through `yolozu.api`
- NMS-free e2e mAP evaluation
- Scenario suite report (fps/recall/depth/pose/rejection)

### 8) Test-time adaptation (TTA / TTT)

- Experimental, non-parameter-updating TTA via `--tta`
  - default `postprocess` mode applies flip-based transforms to exported predictions
  - `rtdetr_pose` `model` mode reruns one horizontally flipped inference branch and
    merges it with the baseline predictions
- Research TTT methods (`tent`, `mim`, `cotta`, `eata`, `sar`) integrated into
  `tools/export_predictions.py` via `--ttt`
  - runs strictly pre-prediction to keep output schema unchanged
  - with `--wrap`, writes `meta.ttt` including config + report (losses, updated params,
    MIM mask ratio, memory, update ratio, and forward/backward counters)
  - `tools/run_ttt_evidence_suite.py` runs a fail-closed clean/shift matrix for
    at least three seeds and separates sample-reset from continual-stream results
  - the 2026-07-27 diagnostic bundle is release-addressable, but its zero
    improvement and single-environment origin keep efficacy `not_established`
- Interface notes: `docs/ttt_integration_plan.md`

### 9) CLI convenience

Core installed CLI examples (not an exhaustive command list):

- `yolozu doctor`
- `yolozu export`
- `yolozu validate`
- `yolozu eval-coco`
- `yolozu eval-instance-seg`
- `yolozu resources`
- `yolozu demo`
- `yolozu test`

See [`generated/cli_reference.md`](generated/cli_reference.md) for the complete current command surface.

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

### 10) Searchable web onboarding

- The generated web docs at <https://www.toppymicros.com/yolozu/docs/> provide a
  self-contained strict CLI path and the stable typed `yolozu.api` example.
- `doctor --proof` creates the tutorial dataset and predictions before either
  interface consumes them; the tutorial does not depend on repository fixtures.
- Real COCO metrics are the canonical `yolozu[coco]` path. The explicit
  dependency-free `--dry-run` fallback validates and converts inputs but does
  not produce metric evidence.
- `tools/manifest.json`, `docs/schemas/`, `docs/web_docs_content.json`, and
  `docs/python_api.md` remain the SSOT inputs. Generated drift, links, source
  hashes, and an outside-checkout candidate-wheel journey are tested.

## Interface Contracts

- Predictions schema: `docs/predictions_schema.md`
- Adapter interface contract: `docs/adapter_contract.md`
- Stable Python surface and compatibility policy: `docs/python_api.md`

## Non-goals

- Full training framework for every model family (this repo focuses on reference training lanes + artifact layout + run interface contract)
- Heavy dependencies required for local evaluation

## Versioning

- Predictions wrapper schema version `1`; canonical entry schema version `2`
- Backward-compatible additions are allowed
- Breaking changes require version bump
- 1.0 interface contract stability boundary: `docs/release_1_0_stability.md`
