# Training Orchestration

YOLOZU now includes a lightweight orchestration layer for multi-backend training batches.

The idea is simple:

1. declare a small list of experiments
2. keep the backend id explicit
3. plan or execute them from one spec
4. write one orchestration report

This is intentionally lightweight. It is not a cluster scheduler.
It is a reproducible repo-side orchestration entrypoint.

## Spec shape

Minimal example:

```json
{
  "schema_version": 1,
  "experiments": [
    {
      "name": "yolox-smoke",
      "backend": "yolox",
      "config": "configs/examples/finetune_external/yolox_s_finetune_smoke.py",
      "dataset": "data/smoke",
      "split": "val",
      "extra_args": [
        "--dry-run",
        "--output",
        "reports/train_external_yolox.json"
      ]
    }
  ]
}
```

## Commands

Plan only:

```bash
python3 tools/orchestrate_train.py \
  --spec reports/train_orchestration_spec.json \
  --output reports/training_orchestration_report.json
```

Execute:

```bash
python3 tools/orchestrate_train.py \
  --spec reports/train_orchestration_spec.json \
  --output reports/training_orchestration_report.json \
  --execute
```

Execute and append a JSONL run registry:

```bash
python3 tools/orchestrate_train.py \
  --spec reports/train_orchestration_spec.json \
  --output reports/training_orchestration_report.json \
  --registry-out reports/training_registry.jsonl \
  --execute
```

Top-level alias:

```bash
python3 -m yolozu train-orchestrate \
  --spec reports/train_orchestration_spec.json \
  --output reports/training_orchestration_report.json
```

## Output

The report uses:

- `format = yolozu_training_orchestration_report_v1`
- one row per experiment
- the exact command that was planned or executed
- execution status and output tails when `--execute` is used
- when the experiment writes a training summary JSON, the orchestration row also records `summary_json`, `work_dir`, and `next_steps`
- when `--registry-out` is set, executed runs are also appended to one JSONL registry file using `yolozu_training_registry_entry_v1`

Schema reference:

- `docs/schemas/training_orchestration_report.schema.json`

## Relationship to the training run summary

Each experiment still owns its own backend-level training summary interface contract.
The orchestration report is the outer batch-level record.

Think of it this way:

- `training_summary.json` = one backend run
- `training_orchestration_report.json` = one batch of runs
- `training_registry.jsonl` = append-only index across many runs/batches

For external backends, that means you can run one batch, then immediately open the
captured `next_steps` commands to continue with export, evaluation, and parity.

## Related docs

- [Training Backend Interface](training_backend_interface.md)
- [Training capability matrix](training_capability_matrix.md)
- [Training, inference, and export](training_inference_export.md)
