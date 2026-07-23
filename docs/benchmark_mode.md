# Benchmark Mode (`yolozu benchmark`)

`yolozu benchmark` is the benchmark entrypoint aligned with the
benchmark-mode argument surface that users commonly expect while keeping YOLOZU's
predictions interface contract mindset.

Related docs:

- [Benchmark mode spec (parity target)](benchmark_mode_spec_parity_target.md)
- [Benchmark gap audit](benchmark_mode_gap_audit.md)
- [Benchmark support matrix](benchmark_support_matrix.md)
- [Backend runtime / license boundary matrix](benchmark_backend_runtime_matrix.md)
- [Latency benchmark harness](benchmark_latency.md)
- [Docs index](README.md)

## What the current implementation does

Today the command provides:

- a benchmark-style CLI surface,
- explicit format planning for `torch`, `onnx`, `engine`, `torchscript`, `openvino`, `executorch`, and `opencv_dnn`,
- explicit task semantics for `detect`, `segmentation`, `classification`, `obb`, `keypoints` / `pose`, `depth`, and `pose6d`,
- a stable benchmark report JSON,
- explicit `skipped` statuses when a format is unavailable,
- a clearly labeled synthetic latency probe,
- real backend orchestration for `torch`, `onnx`, `engine`, `torchscript`, and conditional `openvino` when artifacts and runtimes are available.
- artifact-backed segmentation evaluation/parity for `torch`, `onnx`, `engine`, `torchscript`, and `openvino` when backend-specific mask-prediction artifacts are available.
- artifact-backed keypoints evaluation/parity for `torch`, `onnx`, `engine`, `torchscript`, and `openvino` when backend-specific predictions artifacts are available.
- artifact-backed depth evaluation/parity for `torch`, `onnx`, `engine`, `torchscript`, and `openvino` when backend-specific depth-map artifacts are available.
- artifact-backed pose6d evaluation/parity for `torch`, `onnx`, `engine`, `torchscript`, and `openvino` when backend-specific predictions artifacts are available.
- artifact-backed classification evaluation for `torch`, `onnx`, `engine`, `torchscript`, and `openvino` when backend-specific score artifacts are available.
- artifact-backed OBB evaluation for `torch`, `onnx`, `engine`, `torchscript`, and `openvino` when backend-specific rotated-box artifacts are available.

It still does **not** claim end-to-end backend inference benchmarking for every
format. `executorch` and `opencv_dnn` are explicit unsupported/skipped benchmark
orchestration lanes for now, and missing runtime/model artifacts are reported honestly.

## Where YOLOZU still trails the broader benchmark/export surface

The broader public benchmark/export surface still exposes more formats and task
paths than YOLOZU does. Today the most important remaining gaps are:

- missing benchmark/export formats such as `coreml`, `saved_model`, `tflite`,
  `ncnn`, `rknn`, and `paddle`; `openvino` is conditional and reports skipped
  when the runtime or IR artifact is unavailable
- incomplete parity attachment for artifact-backed classification and OBB lanes
- incomplete flag validation for format-specific knobs

The detailed audit and recommended implementation order live in:

- [Benchmark gap audit](benchmark_mode_gap_audit.md)

## Highest-leverage next steps

If we want to close the user-visible gap efficiently, the
best next steps are:

- keep the OBB lane artifact-backed and explicit instead of pretending YOLOZU ran the underlying backend inference
- keep the classification lane artifact-backed and explicit instead of pretending YOLOZU ran the underlying backend inference
- keep the keypoints lane artifact-backed and explicit instead of pretending YOLOZU ran the underlying backend inference
- keep the segmentation lane artifact-backed and explicit instead of pretending YOLOZU ran the underlying backend inference
- keep the depth lane artifact-backed and explicit instead of pretending YOLOZU ran the underlying backend inference
- keep the pose6d lane artifact-backed and explicit instead of pretending YOLOZU ran the underlying backend inference
- keep the canonical [Benchmark support matrix](benchmark_support_matrix.md)
  synced whenever format/task semantics change
- keep conditional `openvino` reporting honest through the same canonical CLI
  used for the other real backend lanes
- expand real parity artifacts beyond the current `torch`-anchored comparisons
- keep format-specific flag validation strict so unsupported combinations fail
  early

These items are the fastest route to parity without giving up YOLOZU's
predictions interface contract and Apache-2.0 repository policy.

## Quick start

Dry-run planning:

```bash
yolozu benchmark \
  --model runs/foo/model.pt \
  --data data/coco8.yaml \
  --imgsz 640 \
  --format all \
  --dry-run \
  --output reports/benchmark_report.json
```

Repository wrapper equivalent:

```bash
python3 tools/benchmark_model.py \
  --model runs/foo/model.pt \
  --data data/coco8.yaml \
  --imgsz 640 \
  --format torch,onnx,engine \
  --dry-run \
  --output reports/benchmark_report.json
```

Real backend run with explicit backend artifacts:

```bash
yolozu benchmark \
  --model runs/foo/model.pt \
  --onnx-model exports/foo.onnx \
  --engine-model exports/foo.plan \
  --data data/coco8.yaml \
  --format torch,onnx,engine \
  --latency-source auto \
  --output reports/benchmark_report.json
```

Conditional OpenVINO run through the canonical installed CLI:

```bash
yolozu benchmark \
  --model runs/foo/model.pt \
  --openvino-model exports/foo.xml \
  --data data/coco8.yaml \
  --format torch,openvino \
  --latency-source auto \
  --parity-reference-backend openvino \
  --output reports/benchmark_openvino_report.json
```

OpenVINO is not bundled. Without a compatible external runtime and IR artifact,
the report records a skipped lane rather than claiming execution.

Detect run with an explicit parity reference backend:

```bash
yolozu benchmark \
  --model runs/foo/model.pt \
  --onnx-model exports/foo.onnx \
  --data data/coco8.yaml \
  --format torch,onnx \
  --latency-source auto \
  --parity-reference-backend onnx \
  --output reports/benchmark_report.json
```

Protocol-pinned backend run with history tracking:

```bash
yolozu benchmark \
  --model runs/foo/model.pt \
  --onnx-model exports/foo.onnx \
  --engine-model exports/foo.plan \
  --data data/coco8.yaml \
  --format torch,onnx,engine \
  --protocol nms_applied \
  --latency-source auto \
  --history reports/benchmark_report.jsonl \
  --output reports/benchmark_report.json
```

Synthetic latency probe:

```bash
yolozu benchmark \
  --model runs/foo/model.pt \
  --data data/coco8.yaml \
  --format torch \
  --latency-source synthetic_step \
  --iterations 50 \
  --warmup 5 \
  --output reports/benchmark_report.json
```

Task-oriented dry-run for segmentation semantics:

```bash
yolozu benchmark \
  --model runs/foo/model.pt \
  --data data/coco8.yaml \
  --task segmentation \
  --format torchscript \
  --dry-run \
  --output reports/benchmark_report.json
```

YOLOZU-native depth semantics:

```bash
yolozu benchmark \
  --model runs/foo/model.pt \
  --data data/coco8.yaml \
  --task depth \
  --format torchscript \
  --dry-run \
  --output reports/benchmark_report.json
```

Artifact-backed depth benchmark/parity:

```bash
yolozu benchmark \
  --task depth \
  --model reports/depth_torch.npy \
  --onnx-model reports/depth_onnx.npy \
  --data data/reference/gt_depth.npy \
  --format torch,onnx \
  --latency-source artifact_eval \
  --depth-align median_scale \
  --output reports/benchmark_depth_report.json
```

Artifact-backed keypoints benchmark/parity:

```bash
yolozu benchmark \
  --task keypoints \
  --model reports/keypoints_torch.json \
  --onnx-model reports/keypoints_onnx.json \
  --data data/keypoints_dataset \
  --format torch,onnx \
  --latency-source artifact_eval \
  --keypoints-parity-kp-atol 1e-4 \
  --output reports/benchmark_keypoints_report.json
```

Artifact-backed classification benchmark:

```bash
yolozu benchmark \
  --task classification \
  --model reports/classification_torch.json \
  --onnx-model reports/classification_onnx.json \
  --data data/classification_labels.json \
  --format torch,onnx \
  --latency-source artifact_eval \
  --output reports/benchmark_classification_report.json
```

Artifact-backed OBB benchmark:

```bash
yolozu benchmark \
  --task obb \
  --model reports/obb_torch.json \
  --onnx-model reports/obb_onnx.json \
  --data data/obb_labels.json \
  --format torch,onnx \
  --latency-source artifact_eval \
  --output reports/benchmark_obb_report.json
```

Artifact-backed 6DoF benchmark/parity:

```bash
yolozu benchmark \
  --task pose6d \
  --model reports/pose_torch.json \
  --onnx-model reports/pose_onnx.json \
  --data data/pose_dataset \
  --format torch,onnx \
  --latency-source artifact_eval \
  --pose-parity-trans-atol 1e-4 \
  --output reports/benchmark_pose6d_report.json
```

TorchScript real detect benchmark path:

```bash
yolozu benchmark \
  --model exports/foo.torchscript \
  --data data/coco8.yaml \
  --format torchscript \
  --output reports/benchmark_report.json
```

The TorchScript exporter expects a combined detection tensor shaped `(N,6)` or
`(1,N,6)` with rows `[x1,y1,x2,y2,score,class_id]`.

## Core arguments

Benchmark-aligned core:

- `--model`
- `--data`
- `--imgsz`
- `--half`
- `--int8`
- `--device`
- `--verbose`
- `--format`

YOLOZU additions for reproducibility and CI:

- `--torch-model`
- `--onnx-model`
- `--engine-model`
- `--task`
- `--split`
- `--protocol`
- `--max-images`
- `--depth-mask`
- `--depth-align`
- `--depth-parity-mae-atol`
- `--depth-parity-rmse-atol`
- `--keypoints-parity-iou-thresh`
- `--keypoints-parity-score-atol`
- `--keypoints-parity-bbox-atol`
- `--keypoints-parity-kp-atol`
- `--pose-parity-rot-deg-atol`
- `--pose-parity-trans-atol`
- `--pose-parity-depth-atol`
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

## Early validation rules

`yolozu benchmark` now fails early when a non-default flag would be inert for a
selected format. The goal is to avoid quietly accepting options that the
current benchmark implementation cannot honor.

Current rules:

| Format | Supported non-default flags today | Early-rejected examples |
| --- | --- | --- |
| `torch` | `--half`, `--batch`, `--nms` | `--int8`, `--dynamic`, `--simplify`, `--opset`, `--workspace`, `--fraction` |
| `onnx` | none | `--half`, `--batch`, `--nms`, `--int8`, `--dynamic`, `--simplify`, `--opset`, `--workspace`, `--fraction` |
| `engine` | none | `--half`, `--batch`, `--nms`, `--int8`, `--dynamic`, `--simplify`, `--opset`, `--workspace`, `--fraction` |
| `torchscript` | none | all export-oriented non-default flags |
| `openvino` | none | all export-oriented non-default flags |
| `executorch` | none | all export-oriented non-default flags |
| `opencv_dnn` | none | all export-oriented non-default flags |

In addition, real backend execution is intentionally split between backend orchestration and artifact-backed evaluation:

- `--task detect` can use real `torch` / `onnx` / `engine` / `torchscript` / `openvino` orchestration
- `--task classification` can use `--latency-source artifact_eval` to evaluate backend-specific class-score artifacts and report top-k metrics
- `--task obb` can use `--latency-source artifact_eval` to evaluate backend-specific rotated-box artifacts and report OBB metrics
- `--task segmentation` can use `--latency-source artifact_eval` to evaluate backend-specific mask-prediction artifacts with `tools/eval_segmentation.py` and attach real parity reports
- `--task keypoints` can use `--latency-source artifact_eval` to evaluate backend-specific predictions artifacts with `tools/eval_keypoints.py` and attach real parity reports
- `--task depth` can use `--latency-source artifact_eval` to evaluate backend-specific depth artifacts and attach real parity reports
- `--task pose6d` can use `--latency-source artifact_eval` to evaluate backend-specific predictions artifacts with `tools/eval_pose.py` and attach real parity reports

## Output artifacts

Each run writes:

- `benchmark_report.json`
- `export_settings_<format>.json`
- `predictions_<format>.json`
- `eval_<format>.json`
- `parity_<format>.json`

When `torch`, `onnx`, `engine`, `torchscript`, or conditional `openvino` can run for real, the benchmark writes actual
predictions and eval artifacts and attaches real parity artifacts for candidate
backends against the chosen reference backend (preferring `torch` when
available). When a backend is unavailable or the command is invoked with
`--dry-run`, the command writes placeholders instead of pretending that
inference ran.

For `--task segmentation`, the real lane is artifact-backed rather than inference-backed:

- `--model` / `--torch-model` / `--onnx-model` / `--engine-model` / `--torchscript-model` / `--openvino-model` point to backend-specific segmentation predictions artifacts
- `--data` points to the dataset root or `dataset.json` used by `tools/eval_segmentation.py`
- `predictions_<format>.json` is a normalized benchmark-local copy whose relative mask paths have been rewritten under the benchmark artifact layout
- `eval_<format>.json` is produced by `tools/eval_segmentation.py`
- `parity_<format>.json` compares matched masks directly and records per-sample mismatch summaries

For `--task depth`, the real lane is artifact-backed rather than inference-backed:

- `--model` / `--torch-model` / `--onnx-model` / `--engine-model` / `--torchscript-model` / `--openvino-model` point to backend-specific depth artifacts such as `.npy`, `.npz`, or single-channel image files
- `--data` points to the ground-truth depth artifact
- `predictions_<format>.json` records the source depth artifact metadata rather than fabricating a `predictions.json`
- `eval_<format>.json` is produced by `tools/eval_depth.py`
- `parity_<format>.json` compares candidate depth arrays against the chosen reference backend

For `--task keypoints`, the real lane is also artifact-backed:

- `--model` / `--torch-model` / `--onnx-model` / `--engine-model` / `--torchscript-model` / `--openvino-model` point to backend-specific `predictions.json` artifacts with keypoints
- `--data` points to the dataset root used by `tools/eval_keypoints.py`
- `predictions_<format>.json` is a copied backend predictions artifact kept under the benchmark artifact layout
- `eval_<format>.json` is produced by `tools/eval_keypoints.py`
- `parity_<format>.json` compares matched detections plus normalized keypoints directly

For `--task pose6d`, the real lane is also artifact-backed:

- `--model` / `--torch-model` / `--onnx-model` / `--engine-model` / `--torchscript-model` / `--openvino-model` point to backend-specific `predictions.json` artifacts with pose fields
- `--data` points to the dataset root used by `tools/eval_pose.py`
- `predictions_<format>.json` is a copied backend predictions artifact kept under the benchmark artifact layout
- `eval_<format>.json` is produced by `tools/eval_pose.py`
- `parity_<format>.json` compares matched detections plus pose fields such as rotation, translation, and depth

Typical artifact layout:

```text
reports/
  benchmark_report.json
  benchmark_report.jsonl                # only when --history is set
  export_settings_torch.json
  predictions_torch.json
  eval_torch.json
  parity_torch.json
  export_settings_onnx.json
  predictions_onnx.json
  eval_onnx.json
  parity_onnx.json
```

The top-level benchmark report records, per format:

- `task`
- `task_requested`
- `task_semantics.metric_family` / `task_semantics.expected_metric_keys`
- `status`
- `validation_summary`
- `support_status` (`real`, `artifact-backed`, or `skipped`)
- `support_reason`
- `skip_reason` when skipped
- `latency_source`
- `runtime.available` / `runtime.reason` / `runtime.latency_source`
- `execution_semantics.execution_mode`
- `execution_semantics.artifact_expectation`
- `execution_semantics.eval_expectation`
- `artifact_status.predictions` / `artifact_status.eval` / `artifact_status.parity`
- `parity.reference_backend` / `parity.candidate_backends` or parity summary stats
- top-level `parity_summary` with reference backend, comparison counts, skipped formats, and per-format parity artifacts
- `artifacts.predictions`
- `artifacts.eval`
- `artifacts.parity`
- `artifacts.export_settings`

By default the benchmark chooses `torch` as the parity reference backend when
`torch` succeeded. If `torch` is unavailable, it falls back to the first
eligible real backend. Use `--parity-reference-backend torch|onnx|engine|torchscript|openvino` when
you want a specific backend to act as the reference for detect/parity reports.

## Status model

Per-format result statuses are:

- `ok`
- `partial`
- `failed`
- `dry_run`
- `skipped`

For P1 benchmark DoD checks, read `support_status` instead of guessing from
`status`. It is restricted to:

- `real`: YOLOZU ran a real backend/eval path for that requested format.
- `artifact-backed`: YOLOZU consumed backend-specific artifacts and ran real
  eval/parity without claiming it executed backend inference.
- `skipped`: the path was dry-run-only, unsupported, missing a runtime, missing
  an artifact, or otherwise did not produce real benchmark evidence.

The top-level `support_summary` records requested formats, reported formats,
missing formats, per-format support statuses, and counts. CI and release smoke
checks should fail if a requested format silently disappears from `results`.

The top-level `validation_summary` records the format-specific strict
validation policy that was applied before execution:

- `bad_flag_policy: fail_early` for unsupported non-default export/benchmark flags
- `bad_task_source_policy: fail_early` for task/source combinations such as
  artifact-backed tasks forced to dataset-pass timing
- `missing_runtime_policy: report_skipped`
- `missing_artifact_policy: report_skipped`

`openvino_applicable` is `true` once OpenVINO is present in the benchmark format
surface. OpenVINO remains optional: missing runtime or missing IR artifacts are
reported as skipped instead of becoming an install requirement.

Typical skip reasons:

- `unsupported_format`
- `missing_runtime_dependency`
- `gpu_required`
- `platform_not_supported`
- `model_artifact_required`
- `model_artifact_mismatch`

If `--strict` is set and any requested format is skipped, the command exits
with code `2`.

## Why the latency source is explicit

The CLI now defaults to:

- `--latency-source auto`

`auto` prefers a real dataset-pass wall-clock measurement for `torch`, `onnx`,
`engine`, `torchscript`, and `openvino`, prefers `artifact_eval` for `--task classification`, `--task obb`,
`--task segmentation`, `--task keypoints`, `--task depth`, and `--task pose6d`, and falls back to `synthetic_step` for the remaining
formats.
The report records the per-format `latency_source` so CI and readers do not
confuse placeholder timing with a real backend pass.

## Task semantics

The benchmark surface is no longer detection-only. The CLI accepts the
following canonical tasks and aliases:

| Canonical task | Accepted labels | Metric family | Current benchmark state | Notes |
| --- | --- | --- | --- | --- |
| `detect` | `detect`, `detection` | `bbox_map` | real for `torch` / `onnx` / `engine` / `torchscript` / `openvino` | Default benchmark path; OpenVINO remains conditional on external runtime and IR availability. |
| `segmentation` | `segmentation`, `seg` | `mask_map` | artifact-backed real eval/parity for `torch` / `onnx` / `engine` / `torchscript` / `openvino` | Benchmark mode evaluates backend mask-prediction artifacts with `tools/eval_segmentation.py` and compares matched masks directly. |
| `classification` | `classification`, `classify`, `cls` | `topk_accuracy` | artifact-backed real eval for `torch` / `onnx` / `engine` / `torchscript` / `openvino` | Benchmark mode evaluates backend class-score artifacts directly and reports top1/top5/accuracy without claiming YOLOZU ran backend inference. |
| `obb` | `obb` | `obb_map` | artifact-backed real eval for `torch` / `onnx` / `engine` / `torchscript` / `openvino` | Benchmark mode evaluates backend rotated-box artifacts directly and reports OBB metrics without claiming YOLOZU ran backend inference. |
| `keypoints` | `keypoints`, `pose` | `oks_map` | artifact-backed real eval/parity for `torch` / `onnx` / `engine` / `torchscript` / `openvino` | `pose` is accepted as an alias and normalized to `keypoints`; benchmark mode evaluates backend predictions artifacts with `tools/eval_keypoints.py` and compares keypoints directly. |
| `depth` | `depth` | `depth_error` | artifact-backed real eval/parity for `torch` / `onnx` / `engine` / `torchscript` / `openvino` | YOLOZU-native extension; compares backend depth artifacts honestly instead of claiming end-to-end benchmark-surface parity. |
| `pose6d` | `pose6d`, `6dof`, `pose_6d`, `pose-6d` | `pose6d_error` | artifact-backed real eval/parity for `torch` / `onnx` / `engine` / `torchscript` / `openvino` | YOLOZU-native extension; compares backend predictions artifacts honestly instead of claiming end-to-end benchmark-surface parity. |

The top-level `task_semantics` block and each per-format result include:

- canonical task label
- originally requested task label
- accepted aliases
- metric family
- expected metric keys
- support level
- whether the task is part of the mainstream benchmark surface or a YOLOZU-native extension

The per-format `execution_semantics` block now complements that task matrix:

- `execution_mode`: `real_backend_eval`, `real_artifact_eval`, `unsupported_skipped`, `synthetic_planning_only`, or `dry_run_planning`
- `artifact_expectation`: whether predictions/eval/parity are expected to be real, skipped, or dry-run placeholders
- `eval_expectation`: metric family + expected metric keys for that task/backend combination

For `classification`, `obb`, `segmentation`, `keypoints`, `depth`, and `pose6d`, this is especially important: the benchmark report now
records explicit metric expectations and artifact-backed real eval/parity semantics instead of leaving those tasks as vague
future work.

## Current format coverage

| Format | Current state | Notes |
| --- | --- | --- |
| `torch` | real orchestration when runtime + model are available | Delegates to the current torch exporter path and suite eval. |
| `onnx` | real orchestration when runtime + model are available | Requires an explicit ONNX artifact when the primary model is not `.onnx`. |
| `engine` | real orchestration when runtime + model are available | Requires TensorRT-capable runtime and an engine/plan artifact. |
| `torchscript` | real detect orchestration when runtime + model are available | Depends on local PyTorch and a compatible combined-output decode path. |
| `openvino` | conditional real detect orchestration when runtime + IR are available | Depends on an external OpenVINO runtime and compatible `.xml` IR artifact; missing prerequisites are reported as skipped. |
| `executorch` | unsupported/skipped in benchmark orchestration | Standalone exporter supports declared runtime-output decode, but `yolozu benchmark` does not claim this lane as real yet. |
| `opencv_dnn` | unsupported/skipped in benchmark orchestration | Standalone OpenCV-DNN exporters exist, but `yolozu benchmark` does not claim this lane as real yet. |

## Runtime / license boundary

Benchmark support does not imply that a runtime is bundled or that redistribution
is covered by YOLOZU's Apache-2.0 license. Use the backend matrix below as the
runtime boundary reference:

- [Backend runtime / license boundary matrix](benchmark_backend_runtime_matrix.md)
