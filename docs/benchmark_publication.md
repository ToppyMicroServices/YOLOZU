# Interface-contract-first benchmark publication loop

This document defines the official benchmark publication loop under fixed protocol.

## Benchmark table format

Published artifacts:

- JSON: `reports/benchmark_table.json`
- Markdown: `reports/benchmark_table.md`

The JSON table includes:

- `protocol.id` and `protocol.hash`
- `cadence` (for refresh policy)
- `source_reports` (input report files + `run_id`)
- `rows` with per-bucket metrics (`fps`, `latency_ms_mean`) and `run_id`

This makes every published row traceable to originating run ids.

## Source commands and update cadence

Recommended cadence: weekly (or per release cut).

Typical source commands:

```bash
python3 tools/benchmark_latency.py --config configs/benchmark_latency_example.json --output reports/benchmark_latency.json
python3 tools/publish_benchmark_table.py \
  --report reports/benchmark_latency.json \
  --output-json reports/benchmark_table.json \
  --output-md reports/benchmark_table.md \
  --source-command "python3 tools/benchmark_latency.py --config configs/benchmark_latency_example.json"
```

For multi-run publication, pass multiple `--report` values.

## CI gate

Use the publication smoke in CI to ensure benchmark table artifacts remain generatable and traceable.
