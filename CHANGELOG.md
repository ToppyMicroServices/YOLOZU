# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [4.5.1] - 2026-06-21

### Added
- Add YOLO and RT-DETR training-family recipes with task-specific optimizer, augmentation, and stability defaults.

### Changed
- Run CodeQL on pull requests into `main` and give Scorecard read access to workflow metadata.
- Raise optional PyTorch dependency bounds and document temporary no-fixed-version OSV ignores for PyTorch advisories.

## [4.5.0] - 2026-06-19

### Changed
- Sync release changelog automation.
- Improve manual training page break.
- Add train dataset intake checks.
- Cover train dataset preflight paths.
- Add train dataset preflight.
- Sync train manifest help.
- Cover keypoint wrapper training manifest.
- Report reference trainer readiness.
- Normalize training record inputs.
- Clarify training execution status.
- Support training dataset descriptors.
- Update citation file.


## [4.4.1] - 2026-06-18

### Added
- Added `yolozu doctor --proof` as a CPU-only proof path covering toy data, known predictions, schema validation, report generation, and result comparison.
- Added golden artifact evaluation coverage across detection, segmentation, keypoints, depth, and 6D pose lanes.
- Added generated CLI reference documentation and drift coverage for documented commands.
- Added benchmark support matrix generation and clearer real/artifact-backed/skipped benchmark status reporting.

### Changed
- Refocused README and manual onboarding around evaluation-first workflows.
- Improved manual structure, workflow openings, TTT guidance, and representative prose readability.
- Clarified production, external bridge, optional runtime, and research-lane boundaries in documentation.
- Tightened manifest metadata, docs example drift checks, and docs copyability/layout regression coverage.

### Fixed
- Fixed manual PDF layout and workflow diagram issues found during the documentation overhaul.
- Fixed benchmark strict validation and optional runtime reporting so skipped formats are not reported as successful execution.

## [4.4.0] - 2026-04-29

### Added
- Added real TorchScript benchmark orchestration via `tools/export_predictions_torchscript.py`, emitting predictions under the predictions interface contract when PyTorch runtime support is available.
- Added an automated manual-vs-CLI drift audit (`tools/audit_manual_cli_drift.py`) with a documented allowlist for intentional wrapper/manual differences.
- Added a consolidated benchmark support matrix covering real, placeholder, and skipped runtime lanes.

### Changed
- Promoted benchmark reporting to distinguish real execution, placeholder output, and skipped runtime support across ONNX Runtime, TensorRT, TorchScript, and ExecuTorch paths.
- Replaced the ExecuTorch skeleton exporter path with declared runtime decode metadata and explicit skip/fallback reporting.
- Forwarded canonical `yolozu` wrapper commands through the package CLI so documented examples and implementation stay aligned.
- Synchronized tool manifests with new CLI inputs and packaged manifest data.

### Fixed
- Fixed manifest/help drift for `audit_manual_cli_drift --python` and the TorchScript `--input-size` alias.
- Updated manual export and benchmark guidance so CLI examples match the current implementation.

## [4.3.1] - 2026-04-18

### Fixed
- CI smoke bootstrap now chooses a Python interpreter that can actually start the repo-local `yolozu` CLI, preventing deep smoke failures caused by stale or broken local virtualenvs.
- Container bootstrap locks are aligned with current build targets: `requirements-runtime.lock` now carries a Python 3.14-compatible `numpy`, and the RT-DETR pose image lock aligns `cuda-python` with the pinned `torch` CUDA bindings.

### Changed
- Release/publish automation is stricter: `.github/workflows/publish.yml` now validates package version, optional manual-release tag input, and `CHANGELOG.md` release heading before publishing.
- Workflow-only edits are no longer a blind spot in CI. `.github/workflows/build_and_test.yml` now runs release/security workflow regression tests on `.github/workflows/**`-only changes.
- Security/reliability workflows now fail earlier in pull requests: container builds run for container-related PRs, and Scorecard also runs on PRs targeting `main`.

### Contract change
- Reference adapter regression metadata now records provenance/SBOM snapshots in `baseline_meta.provenance` (`pip freeze`, `python -VV`, OS/CPU/torch build hashes).
- Added matrix baseline layout support for reference adapter regression paths (`baselines/<adapter>/<backend>/<device>/<version>/<profile>.json`).
- Added robust behavior metrics for regression gates (`map50`, `map50_95`, worst-k/median class AP, recall@K, IoU quantiles, mismatch counts).

### Added
- **Multi-task SDFT distillation**: `yolozu/sdft.py` now supports task-specific losses for
  6D pose (`rot6d` — geodesic MSE proxy), keypoints (smooth-L1), depth (scale-invariant L1),
  and segmentation (BCE with teacher sigmoid targets). Per-key weights configurable via
  `SdftConfig` fields (`rot6d_weight`, `keypoints_weight`, `depth_weight`, `seg_weight`, etc.).
- **SDFT convenience constructors**: `make_pose_sdft_config()`, `make_keypoints_sdft_config()`,
  `make_depth_sdft_config()`, `make_seg_sdft_config()`, `make_full_sdft_config()`.
- **Multi-task Tent TTT**: `TentRunner` supports auxiliary consistency losses for pose, keypoints,
  depth, and segmentation heads via `aux_pose_weight`, `aux_keypoints_weight`,
  `aux_depth_weight`, `aux_seg_weight` in `TentConfig`.
- **TTT task presets**: Added `pose_safe`, `keypoints_safe`, `depth_safe`, `seg_safe`, `pose_mim`
  presets with task-tuned hyperparameters. `_choose_default_preset_id()` auto-selects
  task-specific presets when `sdft_task` is set.
- **TTTConfig multi-task fields**: `aux_pose_weight`, `aux_keypoints_weight`, `aux_depth_weight`,
  `aux_seg_weight`, `aux_temperature`, `sdft_task`.
- **Tests**: 65 new tests in `test_sdft_multitask.py` and `test_tta_multitask.py` covering all
  task-specific losses, dispatch, convenience constructors, aux consistency, and presets.
- **TTT improvement demo**: `yolozu demo ttt` runs a deterministic domain shift + few-shot training,
  then reports a simple mAP proxy delta (no TTT vs with TTT) with overlay PNG evidence in the
  predictions interface contract.

### Fixed
- **calibration package shadowing**: Standalone `yolozu/calibration.py` was shadowed by
  `yolozu/calibration/` package — merged `apply_temperature()` and
  `calibrate_predictions_entries()` into the package `__init__.py` so imports work correctly.
- **calibration/distillation key preservation**: `calibrate_predictions_entries()` and
  `distill_predictions()` now preserve all original entry keys (e.g. `image_size`, `preprocess`)
  instead of dropping them.

### Changed
- **Image format support**: All image-loading code paths (dataset, CLI, predict, rtdetr_pose, demos,
  pascal_voc, ade20k, make_subset) now accept BMP, TIFF, WebP, and GIF alongside JPEG/PNG.
- **image_size**: Added native header parsers for BMP, GIF, TIFF, WebP; PIL fallback for other formats.
- **letterbox**: `compute_letterbox()` accepts rectangular `(w, h)` tuples in addition to int.
- **geometry**: Zero-division protection in `recover_translation()`; fixed variable shadowing in
  `corrected_intrinsics()`.
- **boxes**: `width`/`height` params accept `int | float`.
- **gates**: `final_score()` uses safe `.get()` with defaults; handles non-dict weights gracefully.
- **constraints**: `apply_constraints()` validates `r_mat` shape (falls back to identity); accepts
  non-dict cfg.
- **cli**: `_detect_config_source_from_path()` raises clear error on unknown `.py` config instead of
  silently defaulting to mmdet.
- **sdft**: Removed duplicate `if total is None:` block.

### Added
- `tests/test_dataset_formats_quality.py`: 26 unit tests covering new image format parsers,
  dataset discovery, letterbox, geometry, calibration, distillation, constraints, and gates.

## [1.0.7] - 2026-02-27

### Added
- `yolozu demo pose --backend densefusion`: optional DenseFusion backend (CUDA + large downloads required).

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
