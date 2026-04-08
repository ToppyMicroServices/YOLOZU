# Benchmark Gap Audit

This document records the current gap between YOLOZU's `yolozu benchmark`
surface and the broader public benchmark/export surface that users now expect.

Licensing note:
- This audit compares public CLI/interface behavior only.
- YOLOZU keeps its Apache-2.0-only repository operations policy.
- Do not copy GPL-licensed implementation code or docs text into YOLOZU.

## Summary

YOLOZU has caught up on the basic benchmark entrypoint shape:

- `--model`
- `--data`
- `--imgsz`
- `--half`
- `--int8`
- `--device`
- `--verbose`
- `--format`

It is already stronger than a plain benchmark wrapper in a few areas:

- stable `predictions interface contract` artifacts
- explicit `skipped`/`partial` status instead of silent fallback
- `run_meta`, `repro_policy`, `runtime_lock`, and history artifacts
- protocol-pinned eval output intended for CI regression
- explicit runtime/license boundary documentation for benchmark backends

The remaining gap is mainly breadth, not the core interface shape.

## Highest-leverage gap closers

If the goal is to close the practical gap against the current benchmark/export
surface users expect
without giving up YOLOZU's Apache-2.0 and artifact-first strengths, the most
effective next steps are:

1. Turn `segmentation`, `classification`, and `obb` into real
   backend/eval paths for the existing `torch` / `onnx` / `engine` benchmark
   flow.
2. Promote `torchscript` from planning-only semantics to a real execution path,
   then follow with `openvino` as the next conditional runtime target.
3. Keep one support matrix that shows whether each format has real inference,
   real eval, real parity artifacts, or only placeholder/skipped semantics.
4. Expand parity artifacts beyond today's `torch`-anchored backend comparisons.
5. Keep format-specific flag validation strict so inert combinations fail early
   instead of looking supported.

## Current gap by area

### 1. Export / benchmark format breadth

The reference benchmark/export ecosystem exposes a much wider format matrix
publicly. YOLOZU currently
documents or partially wires only:

- `torch`
- `onnx`
- `engine`
- `torchscript`
- `executorch`
- `opencv_dnn`

Missing or only planned relative to that public surface:

- `openvino`
- `coreml`
- `saved_model`
- `pb`
- `tflite`
- `edgetpu`
- `tfjs`
- `paddle`
- `ncnn`
- `rknn`
- `mnn`
- `imx`
- `axelera`

Improvement priority:

1. Promote `torchscript` from accepted synthetic/skip semantics to dedicated real orchestration
2. Promote `openvino` from planned to conditional implementation
3. Promote `ncnn` and `rknn` from planned to explicit adapter targets
4. Keep `implemented`, `conditional`, and `planned` formats aligned with the runtime/license matrix

### 2. Task exposure parity

The common benchmark/export surface prominently documents:

- detection
- segmentation
- classification
- pose
- OBB

YOLOZU already has extra task value beyond that baseline:

- monocular depth
- 6DoF pose

The benchmark entrypoint now records explicit task semantics in the benchmark
report, but real backend/eval coverage still lags for:

- segmentation
- classification
- OBB

Improvement priority:

1. Turn the task matrix into task-specific real backend/eval execution paths
2. Keep task-specific eval metric keys visible in the report examples
3. Keep explicit `classification` and `obb` support-status lines in docs/manual
4. Keep `depth` and `pose6d` as YOLOZU-native extensions, not fake benchmark-surface parity
5. Keep the depth and pose6d lanes clearly documented as artifact-backed real eval/parity rather than inference-backed parity

### 3. Argument-surface gaps

Public benchmark/export docs expose more export-oriented knobs around format-specific
behavior. YOLOZU already supports a useful subset:

- `--batch`
- `--dynamic`
- `--nms`
- `--simplify`
- `--opset`
- `--workspace`
- `--fraction`

Still missing or not yet surfaced in the benchmark entrypoint:

- `--keras`
- `--name`
- `--data` semantics tied to INT8 calibration per backend
- format-specific validation around unsupported knob combinations

Improvement priority:

1. Validate format/flag compatibility instead of accepting inert flags
2. Add `--name` for user-friendly artifact grouping
3. Add `--keras` only when there is a real backend path for it
4. Record backend-specific calibration intent explicitly when `--int8` is set

### 4. Real benchmark semantics vs placeholders

YOLOZU is intentionally honest about placeholder outputs. This is good, but the
docs should make the distinction sharper than they do today.

Current behavior:

- `torch` / `onnx` / `engine` can orchestrate real runs
- `keypoints` can use an artifact-backed real eval/parity lane for `torch` / `onnx` / `engine`
- `depth` can use an artifact-backed real eval/parity lane for `torch` / `onnx` / `engine`
- `pose6d` can use an artifact-backed real eval/parity lane for `torch` / `onnx` / `engine`
- `torchscript` is accepted and recorded honestly, but still uses synthetic / skipped semantics
- `executorch` / `opencv_dnn` remain synthetic or skipped
- parity artifacts are real for successful `torch`-anchored backend comparisons, and remain placeholders for dry-run / skipped / synthetic-only formats

Improvement priority:

1. Mark per-format artifact status as `real`, `placeholder`, or `skipped`
2. Add backend matrix examples with actual artifact expectations
3. Distinguish latency benchmarking from export success more clearly
4. Expand real parity beyond the current `torch`-anchored backend comparisons

### 4.1 Runtime / license boundary

YOLOZU now needs a stable backend matrix that distinguishes:

- Apache-2.0 repository code
- external runtimes/SDKs
- supported via adapter/wrapper
- bundled vs not bundled

The benchmark source of truth for that is:

- [Backend runtime / license boundary matrix](benchmark_backend_runtime_matrix.md)

### 5. Docs/readability gap

Right now the benchmark docs are accurate, but the support matrix is spread
across multiple files and not easy to scan.

Improvement priority:

1. Add a single benchmark parity matrix to README/docs/manual
2. Keep one table with columns:
   - format
   - current support level
   - real inference
   - real eval
   - parity artifact
   - status vs the reference benchmark surface
3. Keep one short "what is still missing" section near the CLI docs

## Recommended next implementation set

The highest-value next steps are:

1. Promote `torchscript` from accepted format support to a real backend path
2. Turn the new task matrix into real benchmark/eval coverage for `segmentation`, `classification`, and `obb`
3. Promote `openvino` to conditional support if the runtime path is available
4. Add per-format flag validation so unsupported knobs fail early
5. Add a single support matrix that distinguishes real parity, placeholder parity, and skipped backends at a glance

## Repository policy reminder

When closing the parity gap:

- compare behavior, not code
- keep Apache-2.0-only operational policy
- treat external backends as adapters and interface boundaries
- keep all produced artifacts aligned with the predictions interface contract
