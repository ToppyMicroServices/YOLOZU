# Latency/FPS benchmark harness (YOLO26)

This repo includes a lightweight latency/FPS benchmark runner that produces a stable JSON report and can append to a JSONL history file for comparisons over time.

Tool: `tools/benchmark_latency.py`

For the higher-level `yolozu benchmark` interface and the
parity target, see
[benchmark_mode.md](benchmark_mode.md) and
[benchmark_mode_spec_parity_target.md](benchmark_mode_spec_parity_target.md).

## Quick start (synthetic step)

```bash
python3 tools/benchmark_latency.py \
  --iterations 200 \
  --warmup 20 \
  --output reports/benchmark_latency.json \
  --history reports/benchmark_latency.jsonl \
  --notes "baseline on target HW"
```

This uses the built-in timing step (no inference). It is mainly for validating the harness.

## Config-driven runs (per-bucket)

Create a JSON config that declares buckets and links engine/model paths. You can also point to a per-bucket metrics JSON if you measure latency externally (e.g., from TensorRT once the pipeline is ready).

Example config: [configs/benchmark_latency_example.json](configs/benchmark_latency_example.json)

```json
{
  "output": "reports/benchmark_latency.json",
  "history": "reports/benchmark_latency.jsonl",
  "notes": "Jetson Orin NX - TRT FP16",
  "engine_template": "/abs/path/engines/{bucket}.plan",
  "model_template": "/abs/path/models/{bucket}.onnx",
  "buckets": [
    {"name": "yolo26n", "metrics_path": "reports/latency_yolo26n.json"},
    {"name": "yolo26s", "metrics_path": "reports/latency_yolo26s.json"},
    {"name": "yolo26m", "metrics_path": "reports/latency_yolo26m.json"},
    {"name": "yolo26l", "metrics_path": "reports/latency_yolo26l.json"},
    {"name": "yolo26x", "metrics_path": "reports/latency_yolo26x.json"}
  ]
}
```

Run with:

```bash
python3 tools/benchmark_latency.py --config configs/benchmark_latency.json
```

### External metrics JSON schema

The `metrics_path` file can be either:

- A full report with `{ "metrics": { ... } }`, or
- A metrics dict directly, such as:

```json
{
  "iterations": 200,
  "warmup": 20,
  "total_sec": 5.32,
  "fps": 37.6,
  "latency_ms": {"mean": 26.6, "p50": 26.4, "p90": 27.9, "p95": 28.5, "p99": 30.1, "min": 25.9, "max": 31.4}
}
```

### Generating TensorRT metrics JSON

On Linux+NVIDIA with TensorRT Python bindings, you can generate a compatible per-bucket file using:

```bash
python3 tools/measure_trt_latency.py \
  --engine engines/yolo26n_fp16.plan \
  --shape 1x3x640x640 \
  --iterations 200 \
  --warmup 20 \
  --output reports/latency_yolo26n.json
```

This writes a metrics report JSON (with a top-level `metrics` dict) that the benchmark harness can consume via
`metrics_path`.

## Comparing runs over time

- Each run writes a JSON report to `--output` and appends the full report to `--history` (JSONL).
- You can diff/plot the JSONL history to track regressions across hardware, engines, or model buckets.

## Notes

- The harness is agnostic to the inference backend (ONNXRuntime/TensorRT/etc.).
- Once the TensorRT pipeline is ready, point `metrics_path` to the real measurements for each bucket.
