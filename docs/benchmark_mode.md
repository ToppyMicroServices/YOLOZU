# Benchmark Mode (`yolozu benchmark`)

`yolozu benchmark` is the benchmark entrypoint aligned with the
Ultralytics benchmark-mode argument surface while keeping YOLOZU's
predictions interface contract mindset.

Related docs:

- [Benchmark mode spec (Ultralytics parity target)](benchmark_mode_spec_ultralytics_parity.md)
- [Latency benchmark harness](benchmark_latency.md)
- [Docs index](README.md)

## What the current implementation does

Today the command provides:

- an Ultralytics-like CLI surface,
- explicit format planning for `torch`, `onnx`, `engine`, `executorch`, and `opencv_dnn`,
- a stable benchmark report JSON,
- explicit `skipped` statuses when a format is unavailable,
- a clearly labeled synthetic latency probe,
- real backend orchestration for `torch`, `onnx`, and `engine` when artifacts and runtimes are available.

It still does **not** claim end-to-end backend inference benchmarking for every
format. `executorch` and `opencv_dnn` remain explicit synthetic/skip territory
for now, and missing runtime/model artifacts are reported honestly.

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
predictions and eval artifacts, and keeps `parity_<format>.json` as a
placeholder until parity-gate integration lands. When a backend is unavailable
or the command is invoked with `--dry-run`, the command writes placeholders
instead of pretending that inference ran.

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

If `--strict` is set and any requested format is skipped, the command exits
with code `2`.

## Why the latency source is explicit

The CLI now defaults to:

- `--latency-source auto`

`auto` prefers a real dataset-pass wall-clock measurement for `torch`, `onnx`,
and `engine`, and falls back to `synthetic_step` for the remaining formats.
The report records the per-format `latency_source` so CI and readers do not
confuse placeholder timing with a real backend pass.
