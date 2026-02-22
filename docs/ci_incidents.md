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

## 2026-02-22 — Incident 3 (TensorRT runner `invalid resource handle`)

- Cause: CUDA stream/context was initialized after TensorRT execution context in the `cuda-python` path, producing invalid handles at `enqueueV3`.
- Prevention: initialize CUDA backend + stream before TensorRT runtime/context and normalize CUDA handles/pointers to integer values before runtime calls.
- Rule: keep CUDA resource lifecycle ordering consistent across `pycuda` and `cuda-python` paths.

## 2026-02-22 — Incident 4 (ONNXRuntime IR-version mismatch in GPU smoke)

- Cause: `tools/ci/gen_dummy_dets_onnx.py` emitted an ONNX model with IR version 13, while the CI container ORT build supported up to IR 11.
- Prevention: pin dummy model generation to IR version 11 (`--ir-version 11`) for GPU smoke compatibility.
- Rule: for CI-generated ONNX artifacts, pin both opset and IR version explicitly rather than relying on library defaults.

## 2026-02-22 — Incident 5 (CI over-triggering and queue inefficiency)

- Cause: heavy CI jobs and container checks were triggered for commits that only changed docs/metadata.
- Prevention: add change-scope detection in `ci.yml` and skip heavy jobs when no runtime/code paths changed; keep a fast-path job so workflow still reports status.
- Rule: heavy CI should be path-scoped, while still leaving a deterministic lightweight status for docs-only commits.
