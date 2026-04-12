# Roadmap notes (PyTorch Model + TensorRT + Metrics/Loss + Datasets)

This document is kept for **historical context**. Active work should be tracked as Beads issues:

```bash
bd list
bd search "<keyword>"
```

Spec reference: `docs/specs/rt_detr_6dof_geom_mim_spec_en_v0_4.md`

## YOLO26 competition tooling

For the Apache-2.0-only toolchain to compete against YOLO26 (COCO detect, e2e mAP, size buckets),
track tasks in `docs/roadmaps/yolo26_competition.md` and Beads issues.

## Implementation status (at a glance)

### Data / validation
- [x] YOLO-format dataset manifest builder
- [x] Sidecar metadata support (mask/depth/pose/intrinsics paths + basic content checks)
- [x] Dataset audit tool + validator
- [x] Dataset loader returns full per-instance GT per spec (`M`, `D_obj`, `R_gt`, `t_gt`, `K_gt`)
- [x] Multi-object samples (multiple instances per image) end-to-end support
- [x] Range/units checks (depth meters, mask binary, bbox normalized) + clear conventions
- [x] Deterministic splits + shuffling + reproducibility hooks (seeded)

### Model (RT-DETR)
- [x] Minimal DETR-style starter model wired to pose heads (shape-correct)
- [x] Historical early-model phase completed (backbone/neck/decoder minimal components were replaced)
- [x] Positional embedding: 2D sin/cos
- [x] Full RT-DETR-style backbone + neck (CSPResNet/CSPDarknet + FPN/PAN path)
- [x] RT-DETR hybrid encoder / multi-scale fusion path
- [x] RT-DETR training-time tricks (e.g., denoising queries) if targeting published behavior

### Heads / geometry
- [x] HeadFast outputs (cls/bbox/log_z/rot6D + optional uncertainty)
- [x] CenterOffsetHead (Δu, Δv) (simple linear head)
- [x] GlobalKHead (δfx, δfy, δcx, δcy) (simple linear head)
- [x] rot6D -> R conversion helper
- [x] Translation recovery from bbox/offsets + corrected intrinsics (spec §2)

### Losses / metrics
- [x] Baseline losses (classification/box/log-depth/rot + basic regularizers)
- [x] Symmetry-aware metrics (geodesic, ADD-S) + unit tests (metrics-level)
- [x] Matching-aware training losses wired to real matcher outputs
- [x] Full detector metrics aggregation (mAP/AR) wired to prediction JSON (COCOeval)

### Training / inference / export
- [x] ONNX export wrapper for the current minimal model (opset 17)
- [x] Training pipeline (matcher, augmentation, MIM teacher, staged schedule)
- [x] Inference path (translation recovery, template verification, constraints gating)
- [x] TensorRT export + engine build + parity tests + benchmarks

Status (2026-01-18)
- Initial `rtdetr_pose/` lane created with dataset loader, validator, and early minimal model components.
- Dataset manifest supports metadata sidecar (`.json`) for mask/depth/pose/intrinsics; validator checks shapes and paths.
- SIM/Blender sidecar schema added (M/D_obj/R_gt/t_gt/K_gt + cad_points) and content checks for mask/depth/bbox/projection.
- Reports for baseline/gates/scenarios are implemented (some model components remained early-phase at that point and were not yet competitive).
- Early `rtdetr_pose` implementation files are tracked in git; generated caches/reports are ignored.
- Training loop starter implementation + ONNX/TRT export/parity tooling are implemented (see `rtdetr_pose/tools/train_minimal.py`, `tools/run_rtdetr_pose_backend_suite.py`).
- RT-DETR-style backbone/neck/encoder/decoder (CSPResNet + FPN/PAN + transformer encoder/decoder) is in place.

Current priorities (auto)
1) Stage 7: CI smoke run on tiny COCO subset.
2) Maintain backend parity and reproducibility contracts.

### Training-first next steps (recommended order)
- [x] Dataset returns per-instance GT: `M`, `D_obj`, `R_gt`, `t_gt`, `K_gt` (+ optional `K_gt'`, `cad_points`)
  - [x] Normalize sidecar keys into canonical fields (`R_gt`, `t_gt`, `K_gt`, `M`, `D_obj`)
  - [x] Preserve paths vs inlined arrays without eager decoding
  - [x] Record per-sample availability stats (pose/intrinsics/mask/depth)
- [x] Batch collation for variable #instances and masks/depth
- [x] Collate keeps per-instance counts (for matcher) and pads/query-aligns labels/bboxes
- [x] Minimal trainer entrypoint (1 epoch over coco128; logs loss scalars)
- [x] Trainer prints GT availability summary for debugging
- [x] Matching (Hungarian) + staged cost terms (start with cls/box, then add z/rot)
- [x] Checkpointing + config-driven runs (resume/repro)
- [x] Run record metadata in metrics/checkpoints (git/versions/argv)
- [x] Loss/metric integration test: one training step + backward + no NaNs

Notes (2026-01-21)
- Training path can now consume full GT availability for mask/depth via `gt_M_mask`/`gt_D_obj_mask` (propagated through Hungarian alignment as `M_mask`/`D_obj_mask`).
- Optional: when `t_gt` is missing, `tools/train_minimal.py` can derive `z` (and `t` if `K_gt` exists) from `D_obj` at bbox center via `--z-from-dobj` (arrays inline by default; paths require `--load-aux`).
- [x] Inference-only utilities (later): decoding + constraints gate + template verify

## Stage 0) Repo + environment alignment
- [x] Decide codebase location (new repo under `/Users/akira/YOLOZU` or existing).
- [x] Lock framework versions (PyTorch, CUDA, TensorRT, onnxruntime) in `docs/versions.md`.
- [x] Define experiment config structure (YAML/JSON) and checkpoints layout (initial version).

## Stage 1) Dataset + validation (per spec §4)
- [x] Implement dataset loader with required GT fields (initially YOLO-bbox-focused):
  - `I`, `class_gt`, `bbox_gt`, `M`, `D_obj`, `R_gt`, `t_gt`, `K_gt`
- [x] Add DataSet type checks (initial image/labels coverage):
  - shapes, ranges, missing depth mask handling
  - bbox/mask consistency; `D_obj` masked by `M`
  - project CAD points with `(R_gt, t_gt, K_gt)` to validate bbox/mask
- [x] Metadata sidecar (`.json`) support for mask/depth/pose/intrinsics paths.
- [x] Validator checks pose/intrinsics shapes; mask/depth path existence and array shape match.
- [x] Optional SIM jitter fields (`K_gt'`) validation (intrinsics_prime).
- [x] Sidecar accepts `M/D_obj/R_gt/t_gt/K_gt/cad_points` and performs content checks when enabled.
- [x] Build a small "dataset audit" CLI and summary report.
- [x] Add range/units checks (depth meters, mask binary) and clarify multi-object behavior.

### SIM/Blender sidecar schema (labels/*.json)
- `M` or `mask_path`: silhouette mask (array or path; 2D).
- `D_obj` or `depth_path`: object-only depth (array or path; 2D).
- `R_gt`: 3x3 rotation matrix.
- `t_gt`: translation vector (len=3).
- `K_gt`: intrinsics (`{fx,fy,cx,cy}` or 3x3 matrix).
- `K_gt'` (optional): corrected intrinsics.
- `cad_points` (optional): list of 3D CAD points for projection checks.

Remaining decisions (整理)
- GT coordinate convention (camera vs object frame) and units (meters vs millimeters).
- Mask/depth file formats and value ranges (png/npy/json; depth scale).
- Multi-object samples: whether `cad_points` is per-class or per-instance.
- Whether `K_gt'` is required in SIM exports or optional with fallback to `K_gt`.

Start next (着手)
- Lock framework versions and export opset.
- Implement RT-DETR backbone/decoder (export-safe) and wire HeadFast outputs.
- Extend losses/metrics to full spec and add symmetry invariance tests.

## Stage 2) Model architecture (per spec §3)
- [x] Implement RT-DETR-like backbone/decoder with HeadFast outputs:
  - class logits, bbox, `log_z`, `rot6D`, optional `log_sigma_z/rot`
- [x] Replace baseline backbone/decoder with full RT-DETR-style backbone/neck/encoder/decoder.
- [x] Implement CenterOffsetHead (`Δu, Δv`) and GlobalKHead (`δf/δc`) (initial simple heads).
- [x] Implement rotation conversion `rot6D -> R ∈ SO(3)`.
- [x] Ensure export-friendly ops for TensorRT (avoid meshgrid; export wrapper in place).

## Stage 3) Losses + metrics (per spec §5, §7, §8)
- [x] Losses (baseline implementations):
  - `L_cls`, `L_box`, `L_z` (log-depth), `L_rot_sym`
  - `L_off`, `L_K`, optional `L_mim`
  - `L_z_prior`, `L_plane`, `L_upright` regularizers
- [x] Metrics (baseline implementations):
  - depth error
  - symmetry-aware geodesic, ADD-S
- [x] Unit tests for symmetry + metric invariance (metrics-level).
- [x] Wire metric aggregation/mAP/AR with prediction JSON (COCOeval).

## Stage 4) Training pipeline (per spec §6, §10, §11)
- [x] Data augmentation + SIM jitter integration.
- [x] Hungarian matching with staged cost terms.
- [x] MIM teacher + masking + loss schedule.
- [x] Staged training: offsets first, then GlobalKHead.

## Stage 5) Inference + constraints (per spec §2, §8, §9)
- [x] Translation recovery using corrected `K'` + offsets (utility-level).
- [x] Symmetry-aware template verification (Top-K only).
- [x] Constraints gating (depth prior, plane, upright) using `configs/runtime/constraints.yaml`.
- [x] Low-FP gate via `score_tmp_sym < τ`.

## Stage 6) TensorRT export + parity
- [x] Export to ONNX and build TensorRT engine.
- [x] Parity tests: PyTorch vs TensorRT outputs within tolerance.
- [x] Benchmark fps and latency vs target (>= 30 fps).

## Stage 7) Evaluation + scenario suite (per spec §6)
- [x] Scenario suite run (symmetry, tabletop, depth extremes, jitter).
- [x] Single report output: fps, mAP/Recall, depth error, pose error, rejection rate.
- [x] CI smoke run on tiny COCO subset.

## Stage 8) Documentation + handoff
- [x] Update spec/checklist with actual implementation references.
  - Spec summary: `docs/yolozu_spec.md`
  - TTT integration notes: `docs/ttt_integration_plan.md`
  - CLI entrypoint: `tools/export_predictions.py` (`--ttt`, `--ttt-method`, `--ttt-log-out`)
- [x] Document training/inference commands and export steps.
