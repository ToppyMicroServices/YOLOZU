# TTT / TTA support matrix (safe-by-default)

This page summarizes what YOLOZU supports today for **test-time training / test-time adaptation (TTT/TTA)**.

Principles:
- **Safe by default**: clear reset policies and guard rails.
- **Explicit scope**: we prefer “supported / partial / not supported” over vague claims.
- **Interface-contract-first**: evaluation remains comparable via `predictions.json`.

## Maturity boundary

- Non-parameter-updating TTA is **Experimental** and opt-in.
- Parameter-updating TTT is **Research** and opt-in.
- `Supported` below means the implementation path exists; it does not mean Stable
  production maturity.

### TTA execution modes

- `--tta` uses `postprocess` by default: it transforms exported predictions and does
  not rerun the model.
- `--tta --tta-mode model` reruns one horizontally flipped inference branch for
  `rtdetr_pose`, maps it back, and merges it with the baseline predictions.
- Other adapters warn and fall back to `postprocess` when `model` mode is requested.

## Algorithms

Legend:
- **Supported**: implemented end-to-end with documented knobs.
- **Partial**: implemented core logic, but missing a key piece (e.g., only a subset of backbones, or limited hooks).
- **Planned**: tracked / intended but not shipped.

| Method | Status | Notes |
|---|---:|---|
| Tent | Supported | Entropy minimization style updates with guard rails.
| MIM | Supported | Mutual-information based objective; safe presets.
| CoTTA | Supported | Teacher-student style adaptation with reset/EMA policies.
| EATA | Supported | Reliable sample selection + adaptation under constraints.
| SAR | Supported | Sharpness-aware / robustness-style adaptation under safe policies.

## Safety / reproducibility controls

| Control | Status | Notes |
|---|---:|---|
| Reset policy | Supported | Deterministic reset points / schedules.
| Update budget | Supported | Limit steps / samples / time.
| Drift detection hooks | Supported | Integrates with `yolozu doctor` and run metadata.
| Metrics comparability | Supported | Evaluation stays contract-first via schema + validators.

## Where to go next

- Protocol overview: [ttt_protocol.md](ttt_protocol.md)
- How evaluation stays comparable: [predictions_schema.md](predictions_schema.md)
- External inference entry point: [external_inference.md](external_inference.md)
