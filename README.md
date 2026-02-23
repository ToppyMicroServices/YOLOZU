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

Interface-contract-first evaluation harness for detection / segmentation / pose.

**YOLOZU supports learning and operational research by providing a robust system for evaluation, training, and deployment workflows.** See the new "Learning Features" section below for more details.

Run inference in any backend, export a stable `predictions.json` **predictions interface contract**, and evaluate apples-to-apples with the same validators and metrics.

This pattern makes backend comparisons fair (same dataset + same evaluator), and keeps results reproducible over time by pinning preprocessing/protocol settings in `export_settings`.

## Quickstart (run this first)

```bash
bash scripts/smoke.sh
```

Output artifact: `reports/smoke_coco_eval_dry_run.json`.

Docs index (start here): [`docs/README.md`](docs/README.md).

## Learning Features

### 1. Run Contract Training (再現可能な学習運用)
- **Value:** Consolidates training artifacts (checkpoints, metrics, exports, parity reports) under `runs/<run_id>/`, ensuring easy comparison, regression detection, and reproducibility.
- **Example Command:** `yolozu train ... --run-id exp01` (contract example)
- **Artifacts:** Checkpoints, metrics JSONL, ONNX exports, parity reports.

### 2. Continual Learning (反忘却：タスク列の継続微調整)
- **Value:** Enables fine-tuning across task/domain sequences while providing mechanisms to evaluate and mitigate forgetting.
  - **Core Ideas:** Memoryless self-distillation, optional replay buffer, and parameter-efficient updates like LoRA.
- **Entry Points:**
  - Training: `python3 rtdetr_pose/tools/train_continual.py --config ...`
  - Evaluation: `python3 tools/eval_continual.py --run-json ...` (can run on CPU)
- **Artifacts:** `continual_run.json` (single source of truth), replay_buffer.json.

### 3. Test-Time Training (TTT: Tent/MIM/CoTTA/EATA/SAR)
- **Value:** Provides reproducible test-time adaptation under domain shifts with constraints on cost, reset protocols, and evaluation conditions.
- **Details:** Includes methods like Tent, MIM, CoTTA, EATA, SAR and ensures fixed datasets, batch limits, and seed/reset controls.

### 4. (Quasi-training feature) Prediction Distillation (教師pred → 生徒predのブレンド)
- **Value:** Fast-tracks research loops and ablation by generating distilled predictions from teacher/student predictions.
- **Example Command:** `python3 tools/distill_predictions.py --student ... --teacher ... --dataset ...`

## Start here (choose 1 of 4 entry points)

- **A: Evaluate from precomputed predictions (no inference deps)** — `predictions.json` → validate → eval.
- **B: Train → Export → Eval (RT-DETR scaffold)** — run artifacts → ONNX → parity/eval.
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