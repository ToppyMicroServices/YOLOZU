# Benchmark Mode (`yolozu benchmark`)

`yolozu benchmark` is the Phase-1 benchmark entrypoint aligned with the
Ultralytics benchmark-mode argument surface while keeping YOLOZU's
predictions interface contract mindset.

Related docs:

- [Benchmark mode spec (Ultralytics parity target)](benchmark_mode_spec_ultralytics_parity.md)
- [Latency benchmark harness](benchmark_latency.md)
- [Docs index](README.md)

## What Phase 1 does

Phase 1 is intentionally conservative.

It provides:

- an Ultralytics-like CLI surface,
- explicit format planning for `torch`, `onnx`, `engine`, `executorch`, and `opencv_dnn`,
- a stable benchmark report JSON,
- explicit `skipped` statuses when a format is unavailable,
- a clearly labeled synthetic latency probe.

It does **not** yet claim end-to-end backend inference benchmarking for every
format. If a runtime is missing, the report says so explicitly.

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

## Output artifacts

Each run writes:

- `benchmark_report.json`
- `export_settings_<format>.json`
- `predictions_<format>.json`
- `eval_<format>.json`
- `parity_<format>.json`

In Phase 1, the predictions/eval/parity files are placeholder artifacts when
the backend is unavailable or when the command is invoked with `--dry-run`.
This is deliberate: the command records the planned artifact layout without
pretending that inference already ran.

## Status model

Per-format result statuses are:

- `ok`
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

Phase 1 defaults to:

- `--latency-source synthetic_step`

This means the timing result is an honest synthetic probe, not a claim that the
full backend inference path already ran. The report records that source under
`latency_source` so CI and readers do not confuse control-plane timing with a
real deployment benchmark.
