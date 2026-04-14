# Run Contract (production-style training artifacts)

The RT-DETR pose trainer (`rtdetr_pose.train_minimal`, invoked via `yolozu train`)
supports a **run contract** mode that pins:
- reproducibility inputs (config + seed + environment metadata)
- fixed output paths (checkpoints, metrics JSONL, exports)
- safety gates (full resume, non-finite guards, parity checks)

The goal is: **the same YAML can always do** `train → resume → export → eval`.
Artifacts are always in the same place.

## Enable

Enable the contract by passing either:
- `--run-contract`, or
- `--run-id <id>` (implicitly enables the contract)

Recommended (repo checkout):

```bash
yolozu train configs/examples/train_contract.yaml --run-id exp01
```

## Required inputs

When the run contract is enabled:
- `--config <train_setting.yaml>` is required (this is the trainer settings file)
- `config_version: 1` is required (set in the YAML; enforced at startup)

Also required as explicit YAML/config keys (no implicit defaults):
- `dataset_root`
- `seed`
- `device` (`cpu` / `cuda:0`, ...)
- `amp` (or legacy `use_amp`) / precision (`none` / `fp16` / `bf16`)
- `ddp` (`false` for single-process; `true` when using `torchrun`)

## Outputs (fixed paths)

All outputs live under:

```
runs/<run_id>/
```

Required artifacts:
- `runs/<run_id>/checkpoints/last.pt`
- `runs/<run_id>/checkpoints/best.pt`
- `runs/<run_id>/reports/train_metrics.jsonl` (1 line per optimizer step)
- `runs/<run_id>/reports/val_metrics.jsonl`
- `runs/<run_id>/reports/config_resolved.yaml` (fully-resolved args)
- `runs/<run_id>/reports/run_meta.json` (git SHA, torch/cuda, host, cmdline, etc.)

`run_meta.json` is now contract-validated and must include:
- `schema_version`
- `git.sha`
- `dependency_lock` (package set + requirements file hashes)
- `preprocess` (image size + multiscale params)
- `hardware` / `runtime`
- `command` (argv + command string)

Export + verification artifacts (enabled by default):
- `runs/<run_id>/exports/model.onnx`
- `runs/<run_id>/exports/model.onnx.meta.json`
- `runs/<run_id>/reports/onnx_parity.json` (Torch vs ONNXRuntime diff stats)
- `runs/<run_id>/reports/training_summary.json` (backend-neutral top-level training summary interface contract)

Validation command:

```bash
python3 tools/validate_run_meta.py runs/<run_id>/reports/run_meta.json
```

### Best checkpoint definition

`best.pt` is updated when validation `map50_95` improves (maximization).

## Safety / operability knobs (implemented)

The contracted trainer is meant to be “hard to break” in production-style runs:

- Resume (full state): `--resume` (contract) or `--resume-from <path>`
- NaN/Inf guard (loss/grad): `--stop-on-non-finite-loss`, `--non-finite-max-skips`, `--non-finite-lr-decay`
- Grad clip: `--clip-grad-norm` (recommended >0 for pose/TTT/MIM stability)
- AMP: `--amp {none,fp16,bf16}` (`bf16` preferred when supported)
- EMA: `--use-ema`, `--ema-decay`, `--ema-eval`
- Validation cadence: `--val-every` (epoch) and `--val-every-steps` (optimizer steps)
- Early stop: `--early-stop-patience`, `--early-stop-min-delta`

## Resume (full state)

Contract resume is:

```bash
yolozu train configs/examples/train_contract.yaml --run-id exp01 --resume
```

This resumes from:
- `runs/<run_id>/checkpoints/last.pt`

The contracted checkpoint bundle restores (when present):
- model weights
- optimizer + scheduler state
- AMP scaler state
- EMA state
- progress counters (epoch/global_step)
- RNG state (torch/cuda + python + optional numpy)

## ONNX parity gate

When the contract is enabled, parity runs by default and:
- writes stats to `runs/<run_id>/reports/onnx_parity.json`
- fails the run if drift exceeds thresholds (`--parity-policy fail`)

You can relax this behavior with:
- `--parity-policy warn`

Dependencies:
- ONNX export requires `onnx`
- `model.onnx.meta.json` records whether post-training export succeeded, was skipped, or failed with a structured error.
- parity requires `onnxruntime`
  - easiest: `python3 -m pip install 'yolozu[onnxrt]'`

The richer reference-trainer run contract now also writes a backend-neutral
summary file:

- `reports/training_summary.json`

This is the common top-level training summary interface contract used across
the reference trainer and the external training lanes.

## DDP (1-node multi-GPU)

Use `torchrun` and `--ddp`:

```bash
torchrun --nproc_per_node=2 rtdetr_pose/tools/train_minimal.py \
  --config configs/examples/train_contract.yaml \
  --run-id exp01 \
  --ddp
```

Only rank 0 writes contracted artifacts; non-zero ranks run training and synchronize at barriers.
