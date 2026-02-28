# Reference Adapter Regression Policy

This document defines the test philosophy for `tools/run_reference_adapter_regression.py`.

## Interface contract and behavior split

### Hard invariants (interface contract)

- `predictions interface contract` schema is valid.
- Image keys are canonicalized and record/prediction mapping is preserved.
- `bbox` uses `cxcywh_norm` with finite values.
- `bbox` constraints: `0<=cx,cy<=1`, `0<w,h<=1`.
- `score` constraints: finite and `0<=score<=1`.
- Duplicate image entries are rejected.
- Detections are stable-sorted by `(-score, class_id, bbox)`.
- `class_id` is required in strict mode.
- Runtime lock + optional expected hash checks (`dataset_hash`, `weights_hash`, `checkpoint_hash`) are consistency checks.
- `weights_hash` is a hard invariant only when enforced, or when both baseline/current runs are checkpoint-backed.

### Soft invariants (behavior)

- Score drift on aggregate metrics (`total_detections`, `score_sum`, `score_mean`, `bbox_checksum`).
- Robust score drift (`map50`, `map50_95`, `worst_k_map50`, `median_class_map50`, `recall_at_k`, IoU quantiles, count diagnostics).
- Performance drift (`min_fps_ratio`, `absolute_floor_fps`).
- Optional backend parity drift against `--peer-report` (`map50`, `map50_95`).

Behavior gates should be introduced as `warn` and promoted to `hard` when stable.

## ReproPolicy and provenance/SBOM

`--repro-policy`:

- `strict`: deterministic policy + seeded execution.
- `relaxed`: seeded execution without full deterministic enforcement.
- `off`: speed-first mode.

`--capture-provenance`:

- `full`: include SBOM/environment snapshot (`pip freeze`, `python -VV`, OS, CPU flags, torch build metadata).
- `minimal`: include hashes/counters without full lists.
- `off`: disable provenance snapshot payload.

Every run writes provenance into `run_meta.provenance` and `baseline_meta.provenance`, including:

- generator metadata
- git identity (`sha`, `tag`, refs)
- CI metadata (`run_id`, `workflow`, `job`, URL when available)

## Matrix baselines and profiles

Two baseline layouts are supported:

- `flat` (legacy): use `--baseline`.
- `matrix`: derive path as  
  `baselines/<adapter>/<backend>/<device>/<version>/<profile>.json`

Use `--profile micro|full|custom` to label regression runs and artifacts.

## Two-stage regression policy

- `micro` (PR/fast): prioritize interface contract break detection (`schema_drift`, `consistency_drift`).
- `full` (nightly/manual): emphasize score/perf regression (`metric_drift`, `speed_drift`) and backend parity checks.

## Failure codes

Regression failures emit grep-friendly codes:

- `E_SCHEMA_*` for schema/interface contract violations
- `E_CANON_*` for canonicalization/consistency violations
- `E_SCORE_*` for metric/robust/parity drift
- `E_PERF_*` for performance drift

Reports include `failure_records` (`{gate, mode, code, message}`) and gate-level minimal counterexamples.

## Baseline lifecycle and contract-change procedure

Baseline updates must be intentional and reviewed in PR.

Required in a baseline update PR:

1. Explain what changed (`interface contract`, canonicalization, metric, or model/runtime).
2. Include old-vs-new regression summary.
3. Explicitly state whether `dataset_hash`/`weights_hash` changed.
4. If schema-breaking, update `docs/schema_governance.md` and add a `Contract change` entry in `CHANGELOG.md`.
