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

## YOLO users (v5/v8/11/26) quick path

```bash
python3 tools/import_ultralytics_data_yaml.py --data-yaml /path/to/data.yaml --split val --output data/ultra_wrapper --force
python3 tools/export_predictions_ultralytics.py --model yolo11n.pt --dataset data/ultra_wrapper --split val --protocol nms_applied --wrap --output reports/pred_ultra.json
python3 -m yolozu.cli eval-coco --dataset data/ultra_wrapper --split val --predictions reports/pred_ultra.json --protocol nms_applied --output reports/coco_eval_ultra.json
```

If your predictions include COCO `category_id`, add `--classes data/ultra_wrapper/labels/<split>/classes.json` to `eval-coco` for automatic class-id normalization.

## Detectron2/MMDetection users quick path

```bash
yolozu migrate dataset --from coco --coco-root /path/to/coco --split val2017 --output data/coco_yolo_like --mode manifest
python3 tools/export_predictions_detectron2.py --dataset data/coco_yolo_like --split val2017 --config /path/to/d2_config.yaml --weights /path/to/model_final.pth --protocol nms_applied --output reports/pred_detectron2.json
python3 tools/export_predictions_mmdet.py --dataset data/coco_yolo_like --split val2017 --config /path/to/mmdet_config.py --checkpoint /path/to/epoch_12.pth --protocol nms_applied --output reports/pred_mmdet.json
```

Then validate/evaluate with `--classes data/coco_yolo_like/labels/<split>/classes.json` to normalize COCO `category_id` safely.

## OpenCV-DNN users quick path

```bash
python3 tools/yolozu.py export --backend opencv-dnn --onnx /path/to/model.onnx --dataset /path/to/coco-yolo --split val2017 --imgsz 640 --preprocess rtdetr_resize_640 --decode rtdetr --dump-io reports/opencv_dump_io.json --output reports/pred_opencv.json
python3 tools/validate_predictions.py reports/pred_opencv.json --strict
python3 tools/eval_coco.py --dataset /path/to/coco-yolo --split val2017 --predictions reports/pred_opencv.json --output reports/eval_opencv.json
```

Use `--dnn-backend opencv|cuda|openvino` and `--dnn-target cpu|cuda|cuda_fp16|opencl|opencl_fp16` to compare runtime backends.

## YOLOX users quick path

```bash
python3 tools/yolozu.py export --backend yolox --dataset /path/to/coco-yolo --split val2017 --exp /path/to/yolox_exp.py --weights /path/to/yolox_ckpt.pth --imgsz 640 --score-thr 0.01 --nms-iou 0.65 --output reports/pred_yolox.json
python3 tools/validate_predictions.py reports/pred_yolox.json --strict
python3 tools/eval_coco.py --dataset /path/to/coco-yolo --split val2017 --predictions reports/pred_yolox.json --protocol nms_applied --classes /path/to/coco-yolo/labels/val2017/classes.json --output reports/eval_yolox.json
```

`predictions.json` includes `weights_sha256` and projected exp parameters in `export_settings` for reproducibility.

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
- Optional GPU self-hosted smoke: `.github/workflows/ngc_test.yml` (`gpu-ngc` workflow)

Interpretation:

- PR quality is judged by `ci`.
- `ci` uses a change-scope fast path: docs/metadata-only updates skip heavy test jobs.
- `container` now runs build checks on `main` pushes, and publishes images on tag/manual runs.
- `container` `main` runs are limited to container-related paths (Dockerfiles/deploy/packaging inputs).
- `gpu-ngc` first checks for an idle `self-hosted + gpu` runner and skips cleanly when unavailable.
- If `gpu-ngc` runner probing returns 403, set `RUNNER_DISCOVERY_TOKEN` secret (repo-scoped PAT with Actions read/admin visibility on self-hosted runners).
- Pre-release reliability gate is documented in `docs/release_reliability_checklist.md`.
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
