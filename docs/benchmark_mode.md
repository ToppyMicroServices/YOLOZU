# Benchmark Mode (`yolozu benchmark`)

`yolozu benchmark` is the benchmark entrypoint aligned with the
Ultralytics benchmark-mode argument surface while keeping YOLOZU's
predictions interface contract mindset.

Related docs:

- [Benchmark mode spec (Ultralytics parity target)](benchmark_mode_spec_ultralytics_parity.md)
- [Benchmark gap audit vs Ultralytics docs](benchmark_mode_ultralytics_gap_audit.md)
- [Latency benchmark harness](benchmark_latency.md)
- [Docs index](README.md)

## What the current implementation does

Today the command provides:

- an Ultralytics-like CLI surface,
- explicit format planning for `torch`, `onnx`, `engine`, `torchscript`, `executorch`, and `opencv_dnn`,
- explicit task semantics for `detect`, `segmentation`, `classification`, `obb`, `keypoints` / `pose`, `depth`, and `pose6d`,
- a stable benchmark report JSON,
- explicit `skipped` statuses when a format is unavailable,
- a clearly labeled synthetic latency probe,
- real backend orchestration for `torch`, `onnx`, and `engine` when artifacts and runtimes are available.

It still does **not** claim end-to-end backend inference benchmarking for every
format. `executorch` and `opencv_dnn` remain explicit synthetic/skip territory
for now, and missing runtime/model artifacts are reported honestly.

## Where YOLOZU still trails the Ultralytics docs surface

The official Ultralytics docs currently expose a broader public benchmark/export
matrix than YOLOZU does. Today the most important remaining gaps are:

- missing benchmark/export formats such as `openvino`, `coreml`,
  `saved_model`, `tflite`, `ncnn`, `rknn`, and `paddle`
- partial backend/eval execution coverage for tasks such as `segmentation`,
  `classification`, `obb`, `keypoints`, `depth`, and `pose6d`
- incomplete flag validation for format-specific knobs

The detailed audit and recommended implementation order live in:

- [Benchmark gap audit vs Ultralytics docs](benchmark_mode_ultralytics_gap_audit.md)

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

TorchScript acceptance with the current honest support semantics:

```bash
yolozu benchmark \
  --model exports/foo.torchscript \
  --data data/coco8.yaml \
  --format torchscript \
  --dry-run \
  --output reports/benchmark_report.json
```

## Core arguments

Ultralytics-aligned core:

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

## Output artifacts

Each run writes:

- `benchmark_report.json`
- `export_settings_<format>.json`
- `predictions_<format>.json`
- `eval_<format>.json`
- `parity_<format>.json`

When `torch`, `onnx`, or `engine` can run for real, the benchmark writes actual
predictions and eval artifacts and attaches real parity artifacts for candidate
backends against the chosen reference backend (preferring `torch` when
available). When a backend is unavailable or the command is invoked with
`--dry-run`, the command writes placeholders instead of pretending that
inference ran.

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
- `skip_reason` when skipped
- `latency_source`
- `parity.reference_backend` / `parity.candidate_backends` or parity summary stats
- `artifacts.predictions`
- `artifacts.eval`
- `artifacts.parity`
- `artifacts.export_settings`

## Status model

Per-format result statuses are:

- `ok`
- `partial`
- `failed`
- `dry_run`
- `skipped`

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
and `engine`, and falls back to `synthetic_step` for the remaining formats.
The report records the per-format `latency_source` so CI and readers do not
confuse placeholder timing with a real backend pass.

## Task semantics

The benchmark surface is no longer detection-only. The CLI accepts the
following canonical tasks and aliases:

| Canonical task | Accepted labels | Metric family | Current benchmark state | Notes |
| --- | --- | --- | --- | --- |
| `detect` | `detect`, `detection` | `bbox_map` | real for `torch` / `onnx` / `engine` | Default benchmark path. |
| `segmentation` | `segmentation`, `seg` | `mask_map` | documented partial | Mask metric expectations are explicit in the report; backend/eval breadth is still partial. |
| `classification` | `classification`, `classify`, `cls` | `topk_accuracy` | documented planned | Visible in the benchmark interface contract and report schema, but dedicated real eval wiring is still pending. |
| `obb` | `obb` | `obb_map` | documented planned | Explicitly benchmarkable at the interface level; backend/eval implementation remains a follow-up. |
| `keypoints` | `keypoints`, `pose` | `oks_map` | documented partial | `pose` is accepted as an alias and normalized to `keypoints` in the report. |
| `depth` | `depth` | `depth_error` | documented partial | YOLOZU-native extension, not an Ultralytics parity claim. |
| `pose6d` | `pose6d`, `6dof`, `pose_6d`, `pose-6d` | `pose6d_error` | documented partial | YOLOZU-native extension with explicit metric expectations in the report schema. |

The top-level `task_semantics` block and each per-format result include:

- canonical task label
- originally requested task label
- accepted aliases
- metric family
- expected metric keys
- support level
- whether the task is an Ultralytics-surface parity target or a YOLOZU-native extension

## Current format coverage

| Format | Current state | Notes |
| --- | --- | --- |
| `torch` | real orchestration when runtime + model are available | Delegates to the Ultralytics exporter path and suite eval. |
| `onnx` | real orchestration when runtime + model are available | Requires an explicit ONNX artifact when the primary model is not `.onnx`. |
| `engine` | real orchestration when runtime + model are available | Requires TensorRT-capable runtime and an engine/plan artifact. |
| `torchscript` | accepted now; synthetic / skipped semantics for the current phase | Depends on a local PyTorch runtime only; the benchmark writes honest placeholder artifacts until a dedicated real-orchestration path lands. |
| `executorch` | synthetic / skipped | Artifact-first placeholder behavior only for now. |
| `opencv_dnn` | synthetic / skipped | Artifact-first placeholder behavior only for now. |
