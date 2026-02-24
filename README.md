# YOLOZU

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

Contract-first evaluation toolkit for vision models (detection / segmentation / pose).
Run inference anywhere, export outputs as a stable `predictions.json` (+ `export_settings`), then evaluate with a fixed protocol for reproducible, apples-to-apples comparisons.

## Quickstart (run this first)

```bash
bash scripts/smoke.sh
```

Output artifact: `reports/smoke_coco_eval_dry_run.json`.

Docs index (start here): [`docs/README.md`](docs/README.md).

AI-friendly tool registry (source of truth): [`tools/manifest.json`](tools/manifest.json).

Tool list + args examples: [`docs/tools_index.md`](docs/tools_index.md).

Learning features (training / continual learning / TTT / distillation): [`docs/learning_features.md`](docs/learning_features.md).

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

- Bring-your-own inference → stable `predictions.json` interface contract.
- Validators catch schema drift early.
- Protocol-pinned `export_settings` makes comparisons reproducible.
- Parity/bench quantify backend drift and performance.
- Tooling stays CPU-friendly by default (GPU optional).
- Apache-2.0-only ops policy is enforced in repo tooling.

## Why YOLOZU?

YOLOZU standardizes evaluation around a predictions-first interface contract: run inference anywhere, export `predictions.json` (+ `export_settings`), then validate and evaluate with fixed protocols for reproducible comparisons.

Details: [`docs/yolozu_spec.md`](docs/yolozu_spec.md).

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
