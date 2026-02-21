# CI incidents memo

This page records concrete CI failures and the guard rails added afterward.

## 2026-02-21 — Incident 1 (Hessian CLI smoke test)

- Cause: the smoke test passed `--refine-offsets` but omitted `--enable`, while refinement is opt-in.
- Prevention: keep opt-in semantics explicit in smoke tests (`--enable --refine-offsets`).
- Rule: do not assume feature flags implicitly enable refinement behavior.

## 2026-02-21 — Incident 2 (training contract smoke)

- Cause: `--image-size 32` with `--batch-size 1` reached a BatchNorm 1x1 path and failed during training.
- Prevention: keep CI smoke with `--image-size >= 64` for this path, or increase batch size.
- Rule: keep smoke settings in numerically stable ranges first, then optimize runtime.
