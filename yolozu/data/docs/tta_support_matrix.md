# TTT / TTA support matrix (safe-by-default)

This page summarizes what YOLOZU supports today for **test-time training / test-time adaptation (TTT/TTA)**.

Principles:
- **Safe by default**: clear reset policies and guard rails.
- **Explicit scope**: we prefer “supported / partial / not supported” over vague claims.
- **Interface-contract-first**: evaluation remains comparable via `predictions.json`.

## Algorithms

Legend:
- **Supported**: implemented end-to-end with documented knobs.
- **Partial**: implemented core logic, but missing a key piece (e.g., only a subset of backbones, or limited hooks).
- **Planned**: tracked / intended but not shipped.

| Method | Status | Notes |
|---|---:|---|
| Tent | Supported | Entropy minimization style updates with guard rails.
| MIM | Supported | Masked image modeling with masked reconstruction; safe presets.
| CoTTA | Supported | Teacher-student style adaptation with reset/EMA policies.
| EATA | Supported | Reliable sample selection + adaptation under constraints.
| SAR | Supported | Sharpness-aware / robustness-style adaptation under safe policies.

## Safety / reproducibility controls

| Control | Status | Notes |
|---|---:|---|
| Reset policy | Supported | Deterministic reset points / schedules.
| Update budget | Supported | Limit steps / samples / time.
| Drift detection hooks | Supported | Integrates with `yolozu doctor` and run metadata.
| Metrics comparability | Supported | Evaluation stays interface-contract-first via schema + validators.

## Where to go next

- Protocol overview: `docs/ttt_protocol.md`
- How evaluation stays comparable: `docs/predictions_schema.md`
- External inference entry point: `docs/external_inference.md`
