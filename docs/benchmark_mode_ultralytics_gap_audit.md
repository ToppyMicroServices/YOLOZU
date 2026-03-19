# Benchmark Gap Audit vs Ultralytics Docs

This document records the current gap between YOLOZU's `yolozu benchmark`
surface and the public benchmark/export surface documented by Ultralytics.

Reference baseline:
- [Ultralytics benchmark mode](https://docs.ultralytics.com/modes/benchmark/)
- [Ultralytics tasks overview](https://docs.ultralytics.com/tasks/)
- [Ultralytics export/integrations index](https://docs.ultralytics.com/integrations/)

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

The remaining gap is mainly breadth, not the core interface shape.

## Current gap by area

### 1. Export / benchmark format breadth

Ultralytics exposes a much wider format matrix in public docs. YOLOZU currently
documents or partially wires only:

- `torch`
- `onnx`
- `engine`
- `torchscript`
- `executorch`
- `opencv_dnn`

Missing or only planned relative to the Ultralytics docs surface:

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
4. Separate `implemented`, `conditional`, and `planned` formats in README/docs

### 2. Task exposure parity

Ultralytics prominently documents:

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
- keypoints / pose
- classification
- OBB
- depth
- 6DoF pose

Improvement priority:

1. Turn the task matrix into task-specific real backend/eval execution paths
2. Keep task-specific eval metric keys visible in the report examples
3. Keep explicit `classification` and `obb` support-status lines in docs/manual
4. Keep `depth` and `pose6d` as YOLOZU-native extensions, not fake Ultralytics parity

### 3. Argument-surface gaps

Ultralytics docs expose more export-oriented knobs around format-specific
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
- `torchscript` is accepted and recorded honestly, but still uses synthetic / skipped semantics
- `executorch` / `opencv_dnn` remain synthetic or skipped
- parity artifacts are real for successful `torch`-anchored backend comparisons, and remain placeholders for dry-run / skipped / synthetic-only formats

Improvement priority:

1. Mark per-format artifact status as `real`, `placeholder`, or `skipped`
2. Add backend matrix examples with actual artifact expectations
3. Distinguish latency benchmarking from export success more clearly
4. Expand real parity beyond the current `torch`-anchored backend comparisons

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
   - status vs Ultralytics docs
3. Keep one short "what is still missing" section near the CLI docs

## Recommended next implementation set

The highest-value next steps are:

1. Promote `torchscript` from accepted format support to a real backend path
2. Turn the new task matrix into real benchmark/eval coverage for `segmentation`, `classification`, and `obb`
3. Promote `openvino` to conditional support if the runtime path is available
4. Add per-format flag validation so unsupported knobs fail early
5. Add a single support matrix that distinguishes real parity, placeholder parity, and skipped backends at a glance

## Repository policy reminder

When closing the parity gap against Ultralytics docs:

- compare behavior, not code
- keep Apache-2.0-only operational policy
- treat external backends as adapters and interface boundaries
- keep all produced artifacts aligned with the predictions interface contract
