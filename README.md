# YOLOZU (萬)

日本語: [`Readme_jp.md`](Readme_jp.md)

[![PyPI](https://img.shields.io/pypi/v/yolozu?logo=pypi&logoColor=white)](https://pypi.org/project/yolozu/)
[![Python >=3.10](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://pypi.org/project/yolozu/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI (required)](https://github.com/ToppyMicroServices/YOLOZU/actions/workflows/ci.yml/badge.svg)](https://github.com/ToppyMicroServices/YOLOZU/actions/workflows/ci.yml)
[![Container (optional)](https://github.com/ToppyMicroServices/YOLOZU/actions/workflows/container.yml/badge.svg)](https://github.com/ToppyMicroServices/YOLOZU/actions/workflows/container.yml)
[![PR Gate](https://img.shields.io/badge/PR%20gate-ci%20(required)-0A7A0A)](https://github.com/ToppyMicroServices/YOLOZU/actions/workflows/ci.yml)
[![Publish](https://img.shields.io/badge/container-optional-9E9E9E)](https://github.com/ToppyMicroServices/YOLOZU/actions/workflows/container.yml)

Contract-first evaluation harness for detection / segmentation / pose.

YOLOZU supports different models and datasets through unified contracts and adapters.
Run inference in any backend, export a common `predictions.json`,
and evaluate apples-to-apples with the same validators and metrics.

## Quickstart (run this first)

```bash
bash scripts/smoke.sh
```

Output artifact: `reports/smoke_coco_eval_dry_run.json`.

This is the one-line copy-paste path. Detailed command breakdown is in
the `Quickstart details` section below.

pip users: go to [Install (pip users)](README.md#install-pip-users). Repo users: go to [Source checkout (repo users)](README.md#source-checkout-repo-users).

## Start here (choose 1 of 4 entry points)

- **A: Evaluate from precomputed predictions (no inference deps)**
  — `predictions.json` → validate → eval.
  Start: [`docs/README.md`](docs/README.md)
- **B: Train → Export → Eval (RT-DETR scaffold)**
  — reproducible run artifacts → ONNX → parity/eval.
  Start: [`docs/README.md`](docs/README.md)
- **C: Contracts (predictions / adapter / TTT protocol)**
  — stable schema + adapter boundary + safe adaptation protocol.
  Start: [`docs/README.md`](docs/README.md)
- **D: Bench/Parity (TensorRT pipeline / latency benchmark)**
  — backend parity checks + fixed-protocol latency benchmarking.
  Start: [`docs/README.md`](docs/README.md)

## Key points

- Bring-your-own inference → stable `predictions.json`.
- Validators catch schema drift early.
- Metrics stay comparable across backends/environments.
- Protocol-pinned parity gate is available for backend consistency checks.
- Eval suite carries fixed-condition `export_settings` for reproducible comparisons.
- Tooling stays CPU-friendly by default (GPU optional).
- RT-DETR pose scaffold is available for train→export→eval.
- Apache-2.0-only ops policy is enforced in repo tooling.
- TTT is torch-backend-only and opt-in, with guard-railed presets.
- Depth mode keeps backbone boundary stable and includes unit/scale safety controls.

## Quickstart details

With this repo checkout, run:

```bash
bash scripts/smoke.sh
```

This runs `doctor` → `validate dataset` → `validate predictions` →
`eval-coco --dry-run` using bundled smoke assets in `data/smoke`.

Manual equivalent (same fixed inputs):

```bash
yolozu doctor --output -
yolozu validate dataset data/smoke
yolozu validate predictions data/smoke/predictions/predictions_dummy.json --strict
yolozu eval-coco \
  --dataset data/smoke \
  --split val \
  --predictions data/smoke/predictions/predictions_dummy.json \
  --dry-run \
  --output reports/smoke_coco_eval_dry_run.json
```

Detailed option patterns are in [`docs/README.md`](docs/README.md).

## Workflow policy (required vs optional)

- Required PR gate: `.github/workflows/ci.yml` (`ci` workflow)
- Optional publish workflow: `.github/workflows/container.yml` (`container` workflow)

Interpretation:

- PR quality is judged by `ci`.
- `ci` uses a change-scope fast path: docs/metadata-only updates skip heavy test jobs.
- `container` now runs build checks on `main` pushes, and publishes images on tag/manual runs.
- `container` `main` runs are limited to container-related paths (Dockerfiles/deploy/packaging inputs).
- Container failures may still be treated independently from required PR quality decisions (`ci`).

Optional extras:

See [Install (pip users)](README.md#install-pip-users).

Docs index (start here): [`docs/README.md`](docs/README.md)

One-page proof (shortest path + report shape): [`docs/proof_onepager.md`](docs/proof_onepager.md)

## Keypoints onboarding (one command)

Moved to docs: [docs/tools_index.md](docs/tools_index.md).

## Why YOLOZU (what's unique)

In one glance:

- **BYO inference + contract-first eval**: export the same `predictions.json` and compare apples-to-apples.
- **Safe TTT**: guard rails + reset policies for online adaptation.
- **Apache-2.0-only ops**: license policy + checks to keep the toolchain clean.
- **Parity/bench**: diff stats + fixed-protocol benchmarks across backends.
- **Unified CLI**: `yolozu` (pip) + `python3 tools/yolozu.py` (repo)
  wrap backends with consistent args and caching (`--cache`),
  and always write run metadata (git SHA / env / GPU / config hash).
- **AI-friendly repo surface**: stable schemas + [tools/manifest.json](tools/manifest.json) + [docs/tools_index.md](docs/tools_index.md) for tool discovery / automation.

## Feature highlights and advanced workflows

Moved to docs (entry-focused README policy):

- Feature highlights and capability map: [docs/yolozu_spec.md](docs/yolozu_spec.md)
- Instance segmentation examples: [examples/instance_seg_demo/README.md](examples/instance_seg_demo/README.md)
- Tool/CLI catalog: [docs/tools_index.md](docs/tools_index.md)
- Training/inference/export guide: [docs/training_inference_export.md](docs/training_inference_export.md)
- YOLO26 protocol/gates: [docs/yolo26_eval_protocol.md](docs/yolo26_eval_protocol.md)

## Install (pip users)

```bash
python3 -m pip install yolozu
yolozu --help
yolozu doctor --output -
```

Support / legal:
- Contact: develop@toppymicros.com
- © 2026 ToppyMicroServices OÜ
- Legal address: Karamelli tn 2, 11317 Tallinn, Harju County, Estonia
- Registry code: 16551297

Optional extras (recommended as needed):

```bash
python3 -m pip install 'yolozu[demo]'    # torch demos (CPU OK)
python3 -m pip install 'yolozu[onnxrt]'  # ONNXRuntime CPU exporter
python3 -m pip install 'yolozu[coco]'    # pycocotools COCOeval
python3 -m pip install 'yolozu[full]'
```

CPU demos:

```bash
yolozu demo instance-seg
yolozu demo continual --method ewc_replay     # requires yolozu[demo]
yolozu demo continual --compare --markdown    # suite: naive/ewc/replay/ewc_replay
```

## Source checkout (repo users)

This path unlocks the full repo tooling (`tools/`, `rtdetr_pose/`, scenarios, etc.).

```bash
python3 -m pip install -r requirements-test.txt
python3 -m pip install -e .
python3 tools/yolozu.py --help

# Tiny smoke dataset (optional but useful for scenario runs)
bash tools/fetch_coco128.sh

python3 -m unittest -q
```

For command matrix and detailed CLI behavior, use [docs/tools_index.md](docs/tools_index.md).

## Training pipeline (RT-DETR pose)

For entry-level use, keep this README as a shortest-path guide.
Detailed training/eval/export recipes are maintained in docs:

- Training/inference/export quick guide: [docs/training_inference_export.md](docs/training_inference_export.md)
- Run contract and artifact policy: [docs/run_contract.md](docs/run_contract.md)
- Dataset and schema contracts: [docs/predictions_schema.md](docs/predictions_schema.md)
- Adapter and backend integration: [docs/adapter_contract.md](docs/adapter_contract.md)
- COCO/eval-suite protocol and gates: [docs/yolo26_eval_protocol.md](docs/yolo26_eval_protocol.md)
- External baseline import flow: [docs/yolo26_baseline_repro.md](docs/yolo26_baseline_repro.md)
- TensorRT/ONNX parity and deployment: [docs/tensorrt_pipeline.md](docs/tensorrt_pipeline.md)
- Performance/benchmark workflows: [docs/benchmark_latency.md](docs/benchmark_latency.md)
- TTT/continual-learning safety guide: [docs/ttt_protocol.md](docs/ttt_protocol.md)

## License

Code in this repository is licensed under the Apache License, Version 2.0. See `LICENSE`.
