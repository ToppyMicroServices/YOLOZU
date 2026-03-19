# Benchmark Mode Spec (Ultralytics Parity Target)

This document defines the target specification for a future `yolozu benchmark`
entrypoint that aims to catch up with the user experience of the Ultralytics
benchmark mode while preserving YOLOZU's predictions interface contract,
artifact traceability, and CI-friendly regression workflow.

Reference baseline:
- [Ultralytics benchmark arguments](https://docs.ultralytics.com/modes/benchmark/#arguments)
- [Ultralytics benchmark export formats](https://docs.ultralytics.com/modes/benchmark/#export-formats)

## 1. Goal

YOLOZU should support a short benchmark workflow that feels close to:

```bash
yolozu benchmark --model runs/foo/model.pt --data data/coco8.yaml --imgsz 640
```

but the implementation must remain aligned with YOLOZU's core rule:

- every backend/export path should emit artifacts compatible with the
  predictions interface contract;
- benchmark results should be reproducible, versioned, and suitable for CI
  regression;
- unsupported formats must be reported explicitly rather than silently skipped.

## 2. Scope

This spec covers:

- CLI surface
- format support policy
- artifact outputs
- benchmark metrics
- parity/eval integration

This spec does not require all formats to be implemented immediately.
Implementation is intentionally staged.

## 3. Compatibility Target

### 3.1 Ultralytics-aligned core arguments

The following arguments should be accepted by `yolozu benchmark` in Phase 1:

- `--model`
- `--data`
- `--imgsz`
- `--half`
- `--int8`
- `--device`
- `--verbose`
- `--format`

These are the minimum arguments needed to feel benchmark-mode compatible for
users coming from Ultralytics.

### 3.2 YOLOZU-required extensions

YOLOZU should add the following arguments because benchmark output is expected
to feed downstream validation, eval, parity, and CI workflows:

- `--task`
- `--split`
- `--max-images`
- `--dry-run`
- `--strict`
- `--repro-policy`
- `--runtime-lock`
- `--run-id`
- `--output`
- `--history`
- `--predictions-output`
- `--eval-output`
- `--parity-output`

These are not Ultralytics-compatibility features; they are required to keep
benchmark runs auditable and comparable under the predictions interface
contract.

`--task` should not remain a free-form label. The benchmark surface should
accept explicit canonical tasks and aliases:

- `detect`, `detection`
- `segmentation`, `seg`
- `classification`, `classify`, `cls`
- `obb`
- `keypoints`, `pose`
- `depth`
- `pose6d`, `6dof`, `pose_6d`, `pose-6d`

The report should record both the canonical task and the originally requested
label, alongside task semantics such as metric family, expected metric keys,
support level, and whether the task is an Ultralytics-surface target or a
YOLOZU-native extension.

## 4. Format Policy

### 4.1 Phase 1: first-class formats

Phase 1 should support formats already close to the current repository:

- `torch`
- `onnx`
- `engine`
- `torchscript`
- `executorch`
- `opencv_dnn`

`torchscript` is now part of the first-class benchmark surface because it
extends deployment coverage without depending on vendor-specific GPU runtimes.

### 4.2 Phase 2: conditional formats

These are valid targets but may remain conditional on external runtimes:

- `openvino`
- `coreml`

### 4.3 Phase 3: external / adapter-led formats

These formats should be explicitly tracked as planned, not implied:

- `tflite`
- `edgetpu`
- `saved_model`
- `pb`
- `tfjs`
- `ncnn`
- `mnn`
- `rknn`
- `imx`
- `axelera`
- `paddle`

### 4.4 Unsupported behavior

If a requested format is not implemented on the current platform/runtime,
YOLOZU should emit:

- an explicit `skipped` status in the benchmark report,
- a machine-readable skip reason,
- no fake success.

Examples:

- `unsupported_format`
- `missing_runtime_dependency`
- `gpu_required`
- `platform_not_supported`

## 5. Format-specific Arguments

### 5.1 Common export/benchmark knobs

Phase 1 should normalize the following knobs across supported formats:

- `--batch`
- `--dynamic`
- `--nms`

### 5.2 ONNX / engine-specific knobs

These should be supported where applicable:

- `--simplify`
- `--opset`
- `--workspace`
- `--fraction`

Until a backend really honors them, benchmark mode should fail early on
non-default inert flags instead of silently recording them. In the current
implementation that means:

- `--half`, `--batch`, and `--nms` are meaningful only for the current `torch`
  benchmark path
- `--int8`, `--dynamic`, `--simplify`, `--opset`, `--workspace`, and
  `--fraction` should be rejected when a selected format cannot honor them

### 5.3 Deferred knobs

The following may be accepted later once their backend path exists:

- `--keras`
- `--name`

## 6. Artifact Contract

Each benchmark run should produce stable, versioned artifacts.

Minimum outputs:

- `benchmark_report.json`
- `predictions_<format>.json`
- `eval_<format>.json`
- `export_settings_<format>.json`

Optional outputs:

- `benchmark_history.jsonl`
- `parity_<format>.json`
- `topk_examples/`

### 6.1 Report structure

The top-level benchmark report should contain:

- `schema_version`
- `kind`
- `task`
- `task_requested`
- `task_semantics`
- `execution_semantics`
- `model`
- `data`
- `split`
- `imgsz`
- `format`
- `device`
- `precision`
- `status`
- `latency`
- `throughput`
- `eval_metrics`
- `parity`
- `artifacts`
- `run_meta`

### 6.2 Reproducibility metadata

`run_meta` should record at least:

- `git_sha`
- `python_version`
- `device`
- `backend`
- `seed`
- `repro_policy`
- `runtime_lock`
- `weights_hash`
- `dataset_hash`

## 7. Metrics

Minimum benchmark metrics:

- latency:
  - `mean`
  - `p50`
  - `p90`
  - `p95`
  - `p99`
- throughput:
  - `fps`
- task metric:
  - detection: `bbox_map` family such as `mAP50-95`, `mAP50`, `AR@100`
  - segmentation: `mask_map` family such as `mask_mAP50-95`, `mask_mAP50`
  - classification: `topk_accuracy` family such as `top1`, `top5`
  - OBB: `obb_map` family
  - keypoints / pose: `oks_map` family such as `OKS_mAP`, `PCK`
  - depth: `depth_error` family such as `abs_rel`, `rmse`, `delta1`
  - 6DoF pose: `pose6d_error` family such as `ADD`, `ADDS`, `reprojection_error`

## 8. Parity Rules

Benchmark mode should not be latency-only.

Where a reference backend exists, Phase 2 should compare candidate outputs
against a reference path and record:

- candidate availability
- parity verdict
- metric deltas
- failure reasons

Recommended reference order:

- `torch` as canonical reference when available
- otherwise format-specific reference declared in the run config

## 9. CLI Shape

Recommended Phase 1 CLI:

```bash
yolozu benchmark \
  --model runs/foo/model.pt \
  --data data/coco8.yaml \
  --imgsz 640 \
  --format torch,onnx,engine \
  --device cpu \
  --output reports/benchmark_report.json
```

Recommended behavior:

- if `--format` is omitted, use repo-supported default formats;
- if `--format all` is requested, expand only to formats supported on the
  current machine;
- if `--dry-run` is set, validate argument wiring and planned artifacts without
  pretending to have real benchmark numbers.

Task-specific behavior should also be explicit:

- real backend execution is currently detect-first
- non-detect tasks should remain planning/synthetic-only until dedicated
  backend/eval paths exist
- `depth` and `pose6d` should stay clearly marked as YOLOZU-native extensions

## 10. Mapping to Existing YOLOZU Tools

Current repo pieces that should be reused rather than rewritten:

- `tools/benchmark_latency.py`
- `tools/measure_trt_latency.py`
- `tools/run_trt_pipeline.py`
- `tools/rtdetr_pose_backend_suite.py`
- `tools/export_predictions_onnxrt.py`
- `tools/export_predictions_trt.py`
- `tools/export_predictions_executorch.py`
- `tools/export_predictions_opencv_dnn.py`
- `tools/export_predictions_opencv_dnn_unified.py`

The long-term benchmark entrypoint should orchestrate these existing tools
behind one benchmark-oriented CLI instead of duplicating backend logic.

## 11. Staged Delivery Plan

### P1

- add spec-backed benchmark CLI surface
- support `torch`, `onnx`, `engine`, `executorch`, `opencv_dnn`
- emit stable benchmark report JSON
- support `--dry-run`

### P2

- integrate eval and parity into the benchmark run
- add history/baseline tracking
- add explicit skip reporting for partially supported formats

### P3

- extend toward external/export formats beyond current in-repo backends
- add platform-specific backends such as OpenVINO/CoreML when available
- keep a backend runtime/license matrix so `supported` never implies `bundled`

## 12. Non-goals

The benchmark mode should not:

- silently fabricate backend success when a runtime is missing,
- treat raw latency as sufficient evidence without eval/parity context,
- bypass the predictions interface contract,
- overload the CLI with backend-specific flags before the format exists,
- imply that vendor runtimes or SDKs are bundled with YOLOZU.
