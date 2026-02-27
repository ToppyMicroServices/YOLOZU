# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

No user-facing entries yet.

## [1.0.6] - 2026-02-27

### Added
- `yolozu demo pose --backend aruco`: marker-based 6D pose demo using OpenCV ArUco detection.

### Changed
- `yolozu[demo]` now uses `opencv-contrib-python` to enable ArUco support.

## [1.0.5] - 2026-02-27

### Added
- `yolozu demo pose`: 6D pose demo using chessboard detection + OpenCV solvePnP with overlay output.

## [1.0.4] - 2026-02-26

### Added
- `yolozu demo depth`: added Depth Anything inference via Transformers.
- `yolozu demo depth --compare`: runs Depth Anything + MiDaS + DPT in one run and writes suffixed artifacts.

### Changed
- `yolozu demo depth` now defaults to `depth_anything`.
- `yolozu[demo]` extra now includes Transformers dependencies for the depth demo.

## [1.0.3] - 2026-02-26

### Added
- Added practical CPU-friendly demos backed by real model inference/training:
	- `yolozu demo keypoints` (torchvision Keypoint R-CNN; image overlay + JSON report)
	- `yolozu demo depth` (MiDaS via `torch.hub`; depth images + JSON report; downloads weights on first run)
	- `yolozu demo train` (MNIST fine-tune; checkpoint + JSON report; bounded by `--max-steps`)

### Changed
- `yolozu[demo]` extra now includes `timm` and `opencv-python` to reduce first-run demo friction.
- `tools/yolozu.py` supports `demo` as a passthrough to the package CLI for repo workflows.
- Continual demo: added `--practical` and `--fast` presets to keep CPU runs short while using a realistic vision shift.

## [1.0.2] - 2026-02-25

### Fixed
- Manual DOI publishing: hardened Zenodo `actions/newversion` logic to resolve `conceptrecid` → latest record id before creating a new version, preventing 404 failures.

### Changed
- Manual: added a concise “YOLOZU at a glance” summary to the Overview.

## [1.0.1] - 2026-02-24

### Changed
- Documentation: refocused `README.md` as an entrypoint and moved training/TTT/continual-learning details to `docs/learning_features.md`.
- Manual: updated the “What YOLOZU Is” overview wording for clarity and consistency.
- LLM/MCP: added Ollama (local LLM) setup notes and clarified client routing.

## [1.0.0] - 2026-02-23

### Breaking
- None.

### Added
- FRACAL calibration now supports both bbox and instance-segmentation predictions via `yolozu calibrate --task {bbox,seg,auto}`.
- FRACAL class-frequency stats can now be exported/reused through `--stats-out` and `--stats-in`, enabling stable calibration across runs.
- Trainer now emits FRACAL stats from training records via `--fracal-stats-out`; with `--run-contract`, default output is `runs/<run_id>/reports/fracal_stats_bbox.json`.
- Added alternative calibration methods in `yolozu calibrate`: Logit Adjustment (`--method la --tau`) and NorCal (`--method norcal --gamma`) for side-by-side comparison with FRACAL.
- Added temperature scaling in `yolozu calibrate` (`--method temperature --temperature`, optional `--fit-temperature` with `--temperature-grid`).
- RT-DETR pose scaffold now supports depth integration modes `--depth-mode {none,sidecar,fuse_mid}` with safe default `none`.
- Added sidecar depth ingestion (`depth_path`/`depth`) with per-image `depth_valid` gating and NaN/Inf-safe fallback.
- Added projector-post mid-fusion path (`fuse_mid`) with optional modality dropout via `--depth-dropout`, while preserving the backbone `[P3,P4,P5]` swap boundary.
- Added depth safety controls: `--depth-unit {unspecified,relative,metric}` and `--depth-scale`; absolute-depth matcher terms are disabled outside metric mode.
- Documented depth-mode operation and safety semantics across manifest/readme/docs/manual surfaces.
- Added an explicit 1.0.0 contract stability boundary document (`docs/release_1_0_stability.md`).
- Added generated MCP↔Actions contract reference artifacts and parity drift checks.

### Changed
- Release operation is now explicitly documented as GitHub Release `published` trigger for PyPI Trusted Publishing.
- Added CI golden compatibility gate execution and sdist required-files gate.
- Promoted package classifier from Alpha to Production/Stable.

### Deprecated
- `export_onnx_job` naming is kept as compatibility alias; canonical MCP/Actions job name is `export_predictions_job`.

### Tests
- Added regression coverage for FRACAL stats reuse and instance-segmentation calibration behavior.
- Added run-contract default-path coverage for FRACAL stats artifact output.
- Added depth-mode acceptance tests for no-depth no-op, mixed depth/no-depth batch collation, and `fuse_mid` forward stability.
- Added MCP↔Actions contract parity tests and generated-reference drift tests.

## [0.1.2] - 2026-02-17

### Added
- COCO/Detectron2 keypoint schema ingest on dataset import: `categories[].keypoints` and `categories[].skeleton` are persisted into wrapper metadata (`dataset.json` and `labels/<split>/classes.json`).
- RT-DETR pose trainer auto keypoint setup from dataset metadata: when `--num-keypoints` is not provided, it is inferred from imported keypoint schema.
- Horizontal flip keypoint pairing support based on left/right keypoint names to keep keypoint semantics consistent during augmentation.

### Tests
- Added regression coverage for keypoint schema import persistence and trainer keypoint flip-pair derivation.

## [0.1.1] - 2026-02-15

### Added
- `yolozu validate dataset` to sanity-check YOLO-format datasets (images/labels + normalized bbox ranges).
- `yolozu demo continual --compare/--methods` to run a multi-method continual-learning demo suite and optionally emit a markdown table (`--markdown`).

## [0.1.0] - 2026-02-15

Initial OSS release.

### Added
- `yolozu` pip CLI: `doctor`, `export`, `validate`, `eval-instance-seg`, `resources`, `demo`.
- Predictions JSON schema + validators (backend-agnostic evaluation contract).
- Instance segmentation evaluation (PNG mask contract; mask mAP + optional HTML/overlays).
- Optional extras: `yolozu[demo]` (torch), `yolozu[onnxrt]`, `yolozu[coco]`, `yolozu[full]`.
- TensorRT / ONNXRuntime pipeline helpers (repo checkout; GPU optional).
- RT-DETR pose scaffold (`rtdetr_pose/`) with minimal training + ONNX export hooks.
