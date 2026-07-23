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

## Reconciled current semantics

The current CLI behavior, report metadata, support metadata, and maintained
documentation agree on these boundaries:

- For `detect`, `--latency-source auto` resolves to
  `dataset_pass_wall_time` on real backend formats. An explicit
  `dataset_pass_wall_time` request follows the same backend path; missing
  runtimes or model artifacts are reported as `skipped`.
- Explicit `--task detect --latency-source artifact_eval` fails before report,
  artifact, or backend writes because no prepared detection-artifact
  evaluation path is implemented. The error directs users to `auto` or
  `dataset_pass_wall_time` for backend execution, or `synthetic_step` for
  explicitly synthetic timing.
- For `classification`, `obb`, `segmentation`, `keypoints`, `depth`, and
  `pose6d`, `auto` resolves to `artifact_eval` before flag validation.
  `--no-half --batch 1 --no-nms` remains valid, while non-default values fail
  before report or backend writes.
- A non-dry-run artifact-backed task forced to
  `dataset_pass_wall_time` fails early and directs the user to `auto` or
  `artifact_eval`.
- `openvino` detect is conditional real support. Its runtime and IR artifact
  remain external; missing prerequisites produce a skipped result.
  Artifact-backed OpenVINO tasks consume prepared files without checking or
  invoking the OpenVINO runtime. Their report fields therefore record
  `runtime.required=false`, `runtime.checked=false`, and
  `runtime.available=false` instead of claiming a runtime probe.
- `executorch` and `opencv_dnn` are accepted format labels but benchmark
  orchestration is not wired, so their benchmark results are
  `unsupported/skipped`, not synthetic successes.

The earlier explicit detect `artifact_eval` mismatch is corrected under
`YOLOZU-ll2.28`. Regression coverage patches both the backend command runner
and artifact writer and proves that neither can be reached for this rejected
combination; subprocess coverage checks both public CLI surfaces.

## Highest-leverage gap closers

If the goal is to close the practical gap against the current benchmark/export
surface users expect
without giving up YOLOZU's Apache-2.0 and artifact-first strengths, the most
effective next steps are:

1. Keep `openvino` as a conditional real detect runtime target on the same
   canonical CLI surface as the other implemented detect lanes.
2. Keep the canonical support matrix that shows whether each format has real inference,
   real eval, real parity artifacts, or only placeholder/skipped semantics.
3. Add parity artifacts to the artifact-backed classification and OBB lanes;
   other comparable lanes already support an explicit selected reference.
4. Keep format-specific flag validation strict so inert combinations fail early
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
- `openvino`
- `executorch`
- `opencv_dnn`

Missing or only planned relative to that public surface:

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

1. Keep the current `torchscript` and conditional `openvino` detect orchestration synced with docs/manifest
2. Keep canonical and standalone benchmark flags and backend choices identical
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
report, and artifact-backed real eval coverage now exists for `classification`
and `obb`; artifact-backed real eval/parity coverage exists for `segmentation`,
`keypoints`, `depth`, and `pose6d`. The remaining lag is parity attachment for
the classification and OBB artifact lanes. Their input interface contracts now
fail closed on duplicate ids, class/score shape drift, non-finite values, and
out-of-range OBB confidence scores while preserving empty OBB detection lists.

Improvement priority:

1. Expand parity artifacts for artifact-backed classification and OBB reports
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
- additional validation for future backend-specific knob combinations beyond
  the current task/source/format rules

Improvement priority:

1. Add `--name` for user-friendly artifact grouping
2. Add `--keras` only when there is a real backend path for it
3. Record backend-specific calibration intent explicitly when `--int8` is set
4. Extend task/source/format validation alongside each new backend flag

Current validation resolves `--latency-source auto` before checking flags.
Detect rejects explicit `artifact_eval` before flag applicability because no
prepared detection-artifact evaluation path exists. Supported effective
`artifact_eval` lanes reject non-default `--half`, `--batch`, and `--nms`,
while preserving their defaults. The canonical applicability table is
generated in `docs/benchmark_support_matrix.md`.

### 4. Real benchmark semantics vs placeholders

YOLOZU distinguishes execution evidence through the per-format
`support_status`, `execution_semantics`, and `artifact_status` fields. The
canonical support matrix mirrors those report fields.

Current behavior:

- `torch` / `onnx` / `engine` / `torchscript` / `openvino` can orchestrate real detect runs when their external runtimes and artifacts are available
- `classification` and `obb` can use artifact-backed real eval lanes for `torch` / `onnx` / `engine` / `torchscript` / `openvino`; parity remains skipped
- `segmentation` can use an artifact-backed real eval/parity lane for `torch` / `onnx` / `engine` / `torchscript` / `openvino`
- `keypoints` can use an artifact-backed real eval/parity lane for `torch` / `onnx` / `engine` / `torchscript` / `openvino`
- `depth` can use an artifact-backed real eval/parity lane for `torch` / `onnx` / `engine` / `torchscript` / `openvino`
- `pose6d` can use an artifact-backed real eval/parity lane for `torch` / `onnx` / `engine` / `torchscript` / `openvino`
- `executorch` / `opencv_dnn` report unsupported/skipped in benchmark orchestration
- parity artifacts are real for successful comparisons against the selected
  reference backend (with `auto` preferring `torch`); they remain placeholders
  for dry-run or skipped lanes and for classification/OBB artifact evaluation

Maintained boundary:

1. Keep `support_status` limited to `real`, `artifact-backed`, or `skipped`
2. Keep the support matrix aligned with actual artifact expectations
3. Keep `latency_source` separate from export/evaluation success
4. Add real parity to classification and OBB only after their artifact
   validation and metric semantics are evidence-backed

### 4.1 Runtime / license boundary

YOLOZU uses a stable backend matrix that distinguishes:

- Apache-2.0 repository code
- external runtimes/SDKs
- supported via adapter/wrapper
- bundled vs not bundled

The benchmark source of truth for that is:

- [Backend runtime / license boundary matrix](benchmark_backend_runtime_matrix.md)

### 5. Docs/readability invariant

The canonical support-status table now lives in
[Benchmark support matrix](benchmark_support_matrix.md). Keep summaries in this
audit, `docs/benchmark_mode.md`, README, manual, and manifest pointed back to
that page instead of creating competing matrices.

Improvement priority:

1. Keep README/docs/manual linked to the canonical support matrix
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

1. Keep `openvino` detect support honest when its external runtime or IR artifact is unavailable
2. Expand parity artifacts for artifact-backed classification and OBB reports
3. Keep per-format flag validation strict so unsupported knobs fail early
4. Expand the support matrix when benchmark semantics change

## Repository policy reminder

When closing the parity gap:

- compare behavior, not code
- keep Apache-2.0-only operational policy
- treat external backends as adapters and interface boundaries
- keep all produced artifacts aligned with the predictions interface contract
