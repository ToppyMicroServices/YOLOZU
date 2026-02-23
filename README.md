# YOLOZU (萬)

日本語: [`Readme_jp.md`](Readme_jp.md)

[![PyPI](https://img.shields.io/pypi/v/yolozu?logo=pypi&logoColor=white)](https://pypi.org/project/yolozu/)
[![Zenodo (software DOI)](https://zenodo.org/badge/DOI/10.5281/zenodo.18744756.svg)](https://doi.org/10.5281/zenodo.18744756)
[![Zenodo (manual DOI)](https://zenodo.org/badge/DOI/10.5281/zenodo.18744926.svg)](https://doi.org/10.5281/zenodo.18744926)
[![Python >=3.10](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://pypi.org/project/yolozu/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI (required)](https://github.com/ToppyMicroServices/YOLOZU/actions/workflows/ci.yml/badge.svg)](https://github.com/ToppyMicroServices/YOLOZU/actions/workflows/ci.yml)
[![Container (optional)](https://github.com/ToppyMicroServices/YOLOZU/actions/workflows/container.yml/badge.svg)](https://github.com/ToppyMicroServices/YOLOZU/actions/workflows/container.yml)
[![PR Gate](https://img.shields.io/badge/PR%20gate-ci%20(required)-0A7A0A)](https://github.com/ToppyMicroServices/YOLOZU/actions/workflows/ci.yml)
[![Publish](https://img.shields.io/badge/container-optional-9E9E9E)](https://github.com/ToppyMicroServices/YOLOZU/actions/workflows/container.yml)

Interface-contract-first evaluation + learning research harness for detection / segmentation / pose (keypoints / 6DoF).

YOLOZU also supports **learning + operations research**, not just evaluation: artifact-first training outputs via a **run interface contract (Run Contract)**, continual learning (anti-forgetting), and protocol-pinned test-time training (TTT). See: [Learning features](#learning-features).

Learning / research highlights:
- **Run interface contract training (Run Contract)** — reproducible `runs/<run_id>/...` artifacts for detection + keypoints + 6DoF pose.
- **Continual learning (anti-forgetting)** — self-distillation + optional replay/LoRA + forgetting evaluation.
- **Test-time training (TTT)** — Tent / MIM / CoTTA / EATA / SAR under a protocol-pinned, bounded-cost setup.
- **Prediction distillation** — offline blending of teacher/student `predictions.json` artifacts for faster ablations.
- **Hessian-based refinement** — optional engine-external postprocess over `predictions.json` (experimental).

Run inference in any backend, export a stable `predictions.json` **predictions interface contract**, and evaluate apples-to-apples with the same validators and metrics.

This pattern makes backend comparisons fair (same dataset + same evaluator), and keeps results reproducible over time by pinning preprocessing/protocol settings in `export_settings`.

## Quickstart (run this first)

```bash
bash scripts/smoke.sh
```

Output artifact: `reports/smoke_coco_eval_dry_run.json`.

Docs index (start here): [`docs/README.md`](docs/README.md).

## Learning features

### 1) Run interface contract training (Run Contract: reproducible artifacts)

Value: Reproducible training operations for detection + keypoints + 6DoF pose: pin artifacts (checkpoints / metrics / exports / parity) under `runs/<run_id>/` so runs are easy to compare, regression-check, and fully resume.

Representative command:

```bash
yolozu train configs/examples/train_contract.yaml --run-id exp01
```

Artifacts (fixed paths):
- `runs/<run_id>/checkpoints/{last,best}.pt`
- `runs/<run_id>/reports/{train_metrics,val_metrics}.jsonl`
- `runs/<run_id>/reports/{config_resolved.yaml,run_meta.json,onnx_parity.json}`
- `runs/<run_id>/exports/model.onnx` (+ meta)

Model variants: swap backbones (ResNet/ConvNeXt/CSP/...) while keeping the same artifact layout: [`docs/backbones.md`](docs/backbones.md).

Details: [`docs/run_contract.md`](docs/run_contract.md), [`docs/training_inference_export.md`](docs/training_inference_export.md).

### 2) Continual learning (anti-forgetting across task/domain sequences)

Value: Fine-tune across a task/domain sequence while measuring and mitigating catastrophic forgetting via (a) memoryless self-distillation, (b) optional replay buffer, and (c) optional parameter-efficient updates (LoRA) + regularizers (EWC/SI/DER++).

Representative commands:

```bash
python3 rtdetr_pose/tools/train_continual.py \
  --config configs/continual/rtdetr_pose_domain_inc_example.yaml

python3 tools/eval_continual.py \
  --run-json runs/continual/<run>/continual_run.json \
  --device cpu \
  --max-images 50
```

Artifacts:
- `runs/continual/<run>/continual_run.json` (single source of truth)
- `runs/continual/<run>/replay_buffer.json` (+ per-task `replay_records.json`)
- `runs/continual/<run>/continual_eval.{json,html}` (from `eval_continual.py`)

Details: [`docs/continual_learning.md`](docs/continual_learning.md).

### 3) Test-time training (TTT) under domain shift (Tent / MIM / CoTTA / EATA / SAR)

Value: Reproducible test-time adaptation with bounded cost caps, reset policies (`stream` vs `sample`), and fixed eval subsets for fair comparisons.

Representative command (export predictions with TTT enabled):

```bash
python3 tools/yolozu.py export \
  --backend torch \
  --dataset data/coco128 \
  --split train2017 \
  --checkpoint runs/exp01/checkpoints/best.pt \
  --device cuda \
  --max-images 50 \
  --ttt --ttt-preset safe --ttt-reset sample \
  --ttt-log-out reports/ttt_log_safe.json \
  --output reports/pred_ttt_safe.json
```

Artifacts:
- `reports/pred_ttt_safe.json` (predictions interface contract)
- `reports/ttt_log_safe.json` (TTT step log)
- Optional: fixed subset artifacts via `tools/make_subset_dataset.py` (`subset.json`, `subset_images.txt`)

Details: [`docs/ttt_protocol.md`](docs/ttt_protocol.md).

### 4) (Research helper) Prediction distillation (offline)

Value: Blend teacher/student `predictions.json` artifacts to accelerate ablations without retraining.

Representative command:

```bash
python3 tools/distill_predictions.py \
  --student reports/predictions_student.json \
  --teacher reports/predictions_teacher.json \
  --dataset data/coco128 \
  --output reports/predictions_distilled.json \
  --output-report reports/distill_report.json \
  --add-missing
```

Artifacts:
- `reports/predictions_distilled.json` (+ `reports/distill_report.json`)

Details: [`docs/distillation.md`](docs/distillation.md).

### 5) Hessian-based refinement (post-inference, per-detection; experimental)

Value: A safe Newton / finite-diff Hessian stepper to refine pose-related prediction fields as an engine-external postprocess over `predictions.json`.

Representative command:

```bash
python3 tools/refine_predictions_hessian.py \
  --predictions reports/predictions.json \
  --output reports/predictions_hessian.json \
  --enable \
  --device cpu \
  --log-output reports/hessian_log.json
```

Artifacts:
- `reports/predictions_hessian.json` (predictions interface contract)
- `reports/hessian_log.json` (optional)

Details: [`docs/hessian_solver.md`](docs/hessian_solver.md).

## Start here (choose 1 of 4 entry points)

- **A: Evaluate from precomputed predictions (no inference deps)** — `predictions.json` → validate → eval.
- **B: Train → Export → Eval (RT-DETR scaffold + run interface contract / Run Contract)** — run artifacts → ONNX → parity/eval.
- **C: Interface contracts (predictions / adapter / TTT protocol)** — schemas + adapter interface contract boundary + safe adaptation protocol.
- **D: Bench/Parity (TensorRT / latency benchmark)** — parity checks + pinned-protocol benchmarks.

All four entry points are documented (with copy-paste commands) in [`docs/README.md`](docs/README.md).

CLI note:
- `yolozu ...` is the pip/package CLI.
- `python3 tools/yolozu.py ...` is the repo wrapper CLI.
- For equivalent commands, swap only the executable (`yolozu` ↔ `python3 tools/yolozu.py`).

## Key points

- Bring-your-own inference → stable `predictions.json` predictions interface contract.
- Validators catch schema drift early.
- Protocol-pinned `export_settings` makes comparisons reproducible.
- Parity/bench quantify backend drift and performance.
- Tooling stays CPU-friendly by default (GPU optional).
- Apache-2.0-only ops policy is enforced in repo tooling.

## Why YOLOZU?

- Run inference in any environment you prefer (PyTorch / ONNXRuntime / TensorRT / C++ / etc.) and save only the results to the common `predictions.json` predictions interface contract.
- YOLOZU validates and scores that JSON with the same evaluator, so you can compare backend differences under identical conditions and reproduce results more easily.
- `export_settings` records preprocessing/protocol settings, making comparisons reproducible over time.
- Details: [`docs/yolozu_spec.md`](docs/yolozu_spec.md).

## Install (pip users)

```bash
python3 -m pip install yolozu
yolozu --help
yolozu doctor --output -
```

Optional extras and CPU demos: [`docs/install.md`](docs/install.md).

## Source checkout (repo users)

```bash
python3 -m pip install -r requirements-test.txt
python3 -m pip install -e .
python3 tools/yolozu.py --help
python3 -m unittest -q
```

## Manual (PDF)

Printable manual source: [`manual/`](manual/README.md).

## Support / legal

- Contact: develop@toppymicros.com
- © 2026 ToppyMicroServices OÜ
Full support/legal: [`docs/support.md`](docs/support.md).

## License

Code in this repository is licensed under the Apache License, Version 2.0. See `LICENSE`.
