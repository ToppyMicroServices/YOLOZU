# Reference Adapter Regression Policy

This document defines the test philosophy for `tools/run_reference_adapter_regression.py`.

## Interface contract and behavior split

### Hard invariants (interface contract)

- `predictions interface contract` schema is valid.
- Record preflight succeeds for all inputs:
  - image file exists/readable (`E_IO`)
  - decode succeeds and positive dimensions are available (`E_DECODE`)
- preprocessing interface contract is satisfiable (`E_PREPROC`)
- Image keys are canonicalized and record/prediction mapping is preserved.
- `bbox` uses `cxcywh_norm` with finite values.
- `bbox` constraints: `0<=cx,cy<=1`, `0<w,h<=1`.
- `score` constraints: finite and `0<=score<=1`.
- Duplicate image entries are rejected.
- Detections are stable-sorted by `(-score, class_id, bbox)`.
- `class_id` is required in strict mode.
- Reference adapter entries must include:
  - `image_w/h`
  - `orig_w/h`
  - `model_input_w/h`
  - `preprocess|preproc` metadata (`method`, `resize`, `pad`, `letterbox`, `color_order`, `dtype`, `normalize`)
- Runtime lock + optional expected hash checks (`dataset_hash`, `weights_hash`, `checkpoint_hash`) are consistency checks.
- `weights_hash` is hard when enforced, or when both baseline/current runs are checkpoint-backed.
- A recorded baseline `profile` must match the current invocation.

### Soft invariants (behavior)

- Score drift on aggregate metrics (`total_detections`, `score_sum`, `score_mean`, `bbox_checksum`).
- Robust score drift (`map50`, `map50_95`, `worst_k_map50`, `median_class_map50`, `recall_at_k`, IoU quantiles, count diagnostics).
- Performance drift (`min_fps_ratio`, `absolute_floor_fps`).
- Optional backend parity drift against `--peer-report` (`map50`, `map50_95`).

Behavior gates should start as `warn` and be promoted to `hard` after stabilization.

## ReproPolicy and determinism knobs

`--repro-policy`:

- `strict`: deterministic policy + seeded execution.
- `relaxed`: seeded execution without full deterministic enforcement.
- `off`: speed-first mode.

In `strict`, YOLOZU records and enforces deterministic knobs beyond seed:

- image decode/preprocess library: Pillow
- EXIF orientation normalization: enabled
- color order: `RGB`
- resize algorithm: `bilinear`
- preprocess dtype: `float32`
- input resolution policy: fixed resize
- torch deterministic flags (`torch.use_deterministic_algorithms`, cuDNN deterministic/benchmark)

These knobs are emitted in `run_meta.determinism_knobs`.

## Provenance / SBOM capture

`--capture-provenance`:

- `full`: include SBOM/environment snapshot (`pip freeze`, `python -VV`, OS, CPU flags, torch build metadata).
- `minimal`: include hashes/counters without full lists.
- `off`: disable provenance snapshot payload.

Every run writes provenance into `run_meta.provenance` and `baseline_meta.provenance`.

## Matrix baselines and backend-first thresholds

Two baseline layouts are supported:

- `flat` (legacy): use `--baseline`.
- `matrix`: `baselines/<adapter>/<backend>/<device>/<version>/<profile>.json`

Thresholds support backend-first configuration (`metric_by_backend`, `backend_parity_by_backend`) with fallback to common thresholds.

The fast CI lane keeps the legacy
`baselines/reference_adapter/rtdetr_pose_smoke_val.json` micro/relaxed
baseline. The manual full lane uses the separate
`baselines/reference_adapter/rtdetr_pose/torch/cpu/v1/full.json` full/strict
baseline. Full baseline refreshes must be followed immediately by a check with
the same arguments.

## Two-stage regression policy

- `micro` (PR/fast): prioritize interface contract break detection (`schema_drift`, `consistency_drift`).
- `full` (manual): emphasize score/perf regression (`metric_drift`, `speed_drift`) and backend parity checks.

## Fixed real scenario automation

CI runs one fixed real-image scenario using `data/real_multitask_fewshot` (`split=val`, `max-images=1`) in addition to `data/smoke`.

- Step 1 writes a temporary baseline from the real-image scenario.
- Step 2 immediately re-checks against that baseline with the same deterministic settings.

This keeps the regression gate decomposition exercised on a real dataset path while preserving stable CI execution time.

## Failure codes

- `E_SCHEMA_*` for schema/interface contract violations
- `E_CANON_*` for canonicalization/consistency violations
- `E_IO` / `E_DECODE` / `E_PREPROC` for I/O boundary failures
- `E_SCORE_*` for metric/robust/parity drift
- `E_PERF_*` for performance drift

Reports include `failure_records` (`{gate, mode, code, message}`) and gate-level minimal counterexamples.

## Diff artifacts on failure

When baseline comparison fails (hard failure), the tool emits:

- `diff_summary.json` (gate status, failure counts, first counterexamples, top failure records)
- optional `topk_examples/` overlays (prediction vs GT boxes) for fast diagnosis

Use:

- `--diff-summary-out` to control output file
- `--topk-examples-dir` and `--topk-examples` to control overlays

## Baseline lifecycle and interface-contract-change procedure

Baseline updates must be intentional and reviewed in PR.

Required in a baseline update PR:

1. Explain what changed (`interface contract`, canonicalization, metric, or model/runtime).
2. Include old-vs-new regression summary.
3. Explicitly state whether `dataset_hash`/`weights_hash` changed.
4. If schema-breaking, update `docs/schema_governance.md` and add a `Contract change` entry in `CHANGELOG.md`.
