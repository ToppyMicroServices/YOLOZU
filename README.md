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
- **C: Contracts (predictions / adapter / ttt protocol)**
  — stable schema + adapter boundary + safe adaptation protocol.
  Start: [`docs/README.md`](docs/README.md)
- **D: Bench/Parity (TensorRT pipeline / latency benchmark)**
  — backend parity checks + fixed-protocol latency benchmarking.
  Start: [`docs/README.md`](docs/README.md)

## Key points

- Bring-your-own inference → stable `predictions.json`.
- Validators catch schema drift early.
- Metrics stay comparable across backends/environments.
- Tooling stays CPU-friendly by default (GPU optional).
- RT-DETR pose scaffold is available for train→export→eval.
- Safe TTT presets exist (Tent/MIM/CoTTA/EATA/SAR).

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
- `container` is for image publishing flows (tag/release/manual) and may fail independently
  without blocking normal PR quality decisions.

Optional extras:

```bash
python3 -m pip install 'yolozu[demo]'    # torch demos (CPU OK)
python3 -m pip install 'yolozu[onnxrt]'  # ONNXRuntime CPU exporter
python3 -m pip install 'yolozu[coco]'    # pycocotools COCOeval
python3 -m pip install 'yolozu[full]'
```

Docs index (start here): [`docs/README.md`](docs/README.md)

One-page proof (shortest path + report shape): [`docs/proof_onepager.md`](docs/proof_onepager.md)

## Keypoints onboarding (one command)

Prepare keypoints data into YOLOZU-ready layout:

```bash
python3 tools/yolozu.py prepare-keypoints-dataset \
  --source data/keypoints_src \
  --format auto \
  --out data/keypoints_dataset
```

Supported direct keypoints inputs:

- `auto`
- `yolo_pose`
- `coco`
- `cvat_xml`

Not direct (convert first):

- `detectron2_dataset_dict`
- `labelme_keypoints`

Format matrix/help:

```bash
python3 tools/yolozu.py prepare-keypoints-dataset \
  --list-formats \
  --source . \
  --out .
```

Minimal CVAT XML smoke test:

```bash
python3 -m pytest -q tests/test_prepare_keypoints_dataset_cvat_xml.py
```

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

## Feature highlights (what you can do)

- Dataset I/O: YOLO-format images/labels + optional per-image JSON metadata.
- Stable evaluation contract: versioned predictions-JSON schema + adapter contract.
- Unified CLI:
  - pip: `yolozu` (install-safe commands + CPU demos)
  - repo: `python3 tools/yolozu.py` (power-user research/eval workflows)
- Inference/export: `python3 tools/yolozu.py export --backend {torch,onnxrt,trt}` (wrapper) or the low-level scripts
  (`tools/export_predictions*.py`).
- Test-time adaptation options:
  - TTA: lightweight prediction-space post-transform (`--tta`).
  - TTT: pre-prediction test-time training (Tent or MIM) via `--ttt` on the **torch backend**
    (see `docs/ttt_protocol.md`).
- Hessian refinement (post-processing):
  per-detection iterative refinement on exported predictions JSON;
  default is disabled and must be opt-in.
- TensorRT note:
  TRT conversion targets the inference graph only;
  Hessian refinement runs outside the engine as a separate post-processing step.
- Evaluation: COCO mAP conversion/eval and scenario suite reporting.
- Keypoints:
  YOLO pose-style keypoints in labels/predictions + PCK evaluation
  + optional COCO OKS mAP (`tools/eval_keypoints.py --oks`) and parity/benchmark helpers.
  COCO/Detectron2 keypoint schema (`categories[].keypoints` / `skeleton`)
  is auto-ingested into wrapper metadata so training can auto-set
  `num_keypoints` and left/right flip pairs.
- Semantic seg:
  dataset prep helpers + `tools/eval_segmentation.py`
  (mIoU/per-class IoU/ignore_index + optional HTML overlays).
- Instance seg:
  `tools/eval_instance_segmentation.py`
  (mask mAP from per-instance binary PNG masks + optional HTML overlays).
- Training pipeline:
  RT-DETR pose trainer with run contract, metrics output, ONNX export,
  and optional SDFT-style self-distillation.
- Depth-aware training path (optional):
  `--depth-mode {none,sidecar,fuse_mid}` with sidecar depth validity gating
  and safe default `none`.

## Instance segmentation (PNG masks)

YOLOZU evaluates instance segmentation with **per-instance binary PNG masks** (no RLE/polygons required).

Shortest path:

```bash
python3 -m yolozu.cli demo instance-seg
python3 tools/eval_instance_segmentation.py \
  --dataset examples/instance_seg_demo/dataset \
  --split val2017 \
  --predictions examples/instance_seg_demo/predictions/instance_seg_predictions.json
```

Full examples and conversion workflows moved to:

- [examples/instance_seg_demo/README.md](examples/instance_seg_demo/README.md)
- [docs/tools_index.md](docs/tools_index.md)

## Documentation

Start here: [docs/README.md](docs/README.md)

- Repo feature summary: [docs/yolozu_spec.md](docs/yolozu_spec.md)
- Model/spec note: [docs/specs/rt_detr_6dof_geom_mim_spec_en_v0_4.md](docs/specs/rt_detr_6dof_geom_mim_spec_en_v0_4.md)
- Training / inference / export quick steps: [docs/training_inference_export.md](docs/training_inference_export.md)
- Hessian solver for regression refinement: [docs/hessian_solver.md](docs/hessian_solver.md)
- Predictions schema (stable): [docs/predictions_schema.md](docs/predictions_schema.md)
- Adapter contract (stable): [docs/adapter_contract.md](docs/adapter_contract.md)
- Migration helpers: [docs/migrate.md](docs/migrate.md)
- License policy: [docs/license_policy.md](docs/license_policy.md)
- Tools index (AI-friendly): [docs/tools_index.md](docs/tools_index.md) / [tools/manifest.json](tools/manifest.json)
- AI-first usage guide: [docs/ai_first.md](docs/ai_first.md)
- PyInstaller/PyArmor packaging notes: [deploy/pyinstaller/README.md](deploy/pyinstaller/README.md)

## Roadmap (priorities)

- P0: Unified CLI (`torch` / `onnxruntime` / `tensorrt`) with consistent args
  + same output schema; always write meta (git SHA / env / GPU / seed / config hash);
  keep `tools/manifest.json` updated.
- P1: `doctor` (deps/GPU/driver/onnxrt/TRT diagnostics)
  + `predict-images` (folder input → predictions JSON + overlays)
  + HTML report.
- P2: cache/re-run (fingerprinted runs) + sweeps (wrapper exists;
  expand sweeps for TTT/threshold/gate weights)
  + production inference cores (C++/Rust) as needed.
- Long-form notes: `docs/roadmap.md`

### Status snapshot (2026-02-17)

- P0: implemented in unified wrapper CLI
  (`python3 tools/yolozu.py export --backend {dummy,torch,onnxrt,trt}`)
  with wrapped predictions JSON and `meta.run` (`git/env/gpu/seed/config_hash`).
- P1: implemented (`doctor`, `predict-images`, HTML overlays/report path).
- P2: implemented baseline (`--cache`, sweep wrapper) and ongoing expansion for broader production cores/tuning presets.

Recent compatibility additions:
- import/doctor auto-detection:
  `yolozu import ... --from auto`,
  `yolozu doctor import --config-from auto|--dataset-from auto`.
- train shorthand preview:
  `yolozu train --import auto --cfg configs/examples/train_setting.yaml`
  writes resolved canonical `TrainConfig`.

### Depth mode (RT-DETR pose scaffold)

`rtdetr_pose/tools/train_minimal.py` supports optional depth integration
without breaking the backbone swap boundary (`[P3,P4,P5]`):

- `--depth-mode none` (default): no depth path, baseline behavior.
- `--depth-mode sidecar`: read per-image sidecar depth (`depth_path`/`depth`) and propagate `depth_valid`.
- `--depth-mode fuse_mid`: sidecar + lightweight mid-fusion
  after projector (outside backbone boundary), with `--depth-dropout`
  for modality dropout.

Safety defaults:

- `--depth-unit` controls absolute-depth safety (`unspecified|relative|metric`, default `unspecified`).
- Absolute depth matcher costs are only active in metric mode.
  Non-metric modes disable `cost_z`/`cost_t` safety-sensitively.
- `--depth-scale` applies unit scaling to sidecar depth values before use.

## Pros / Operational Notes (project-level)

### Pros
- Apache-2.0-only utilities and evaluation harnesses (no vendored GPL/AGPL inference code).
- CPU-first development workflow: dataset tooling, validators, scenario suite, and unit tests run without a GPU.
- Adapter interface decouples inference backend from evaluation (PyTorch/ONNXRuntime/TensorRT/custom), so you can
  run inference elsewhere and still score/compare locally.
- Reproducible artifacts: stable JSON reports + optional JSONL history for regressions.
- Symmetry + commonsense constraints are treated as first-class, test-covered utilities (not ad-hoc postprocess).

### Operational notes and mitigations
- Training in `rtdetr_pose/` is run-contract based
  (data/loss/export wiring, resume, parity gate).
  Continual-learning behavior is testable from pip with
  `yolozu demo continual --compare --markdown`, and source training stays
  available via `yolozu train configs/examples/train_setting.yaml`
  (`docs/training_inference_export.md`, requires `yolozu[train]`).
- A one-command folder inference path is available from pip:
  `yolozu predict-images --backend onnxrt --input-dir data/smoke/images/val --onnx runs/smoke/model.onnx`,
  which writes predictions JSON + overlays + HTML in one run.
- TensorRT remains NVIDIA/Linux-centric, while macOS can run CPU validation and ONNXRuntime export:
  `yolozu onnxrt export ...`; GPU/TRT build/eval is pinned to Runpod/container workflows (`docs/tensorrt_pipeline.md`).
- Backend parity drift is handled by a dedicated checker:
  `yolozu parity --reference reports/pred_torch.json --candidate reports/pred_onnxrt.json`
  plus protocol-pinned eval settings (`docs/yolo26_eval_protocol.md`).
- Lightweight metrics stay available for fast loops, and full COCOeval is directly exposed from pip:
  install extras and run:

  ```bash
  python3 -m pip install 'yolozu[coco]'
  yolozu eval-coco --dataset data/smoke --predictions data/smoke/predictions/predictions_dummy.json
  ```
- Long-tail focused post-hoc path is available without retraining:
  ```bash
  yolozu calibrate --method fracal --task bbox --dataset data/smoke \
    --predictions data/smoke/predictions/predictions_dummy.json \
    --output runs/smoke/predictions_calibrated.json \
    --stats-out runs/smoke/fracal_stats_bbox.json
  yolozu eval-long-tail --dataset data/smoke --predictions runs/smoke/predictions_calibrated.json
  ```
  Reuse training-time stats with `--stats-in reports/fracal_stats_bbox.json` (also supported for `--task seg`).
  Alternative methods are also available for comparison:
  `--method la --tau <value>` and `--method norcal --gamma <value>`.
- Model weights/datasets stay outside git by design; reproducibility is maintained through stable JSON artifacts and
  pinned path conventions documented in `docs/external_inference.md` and `docs/yolo26_inference_adapters.md`.

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

# Tiny smoke dataset (optional but useful for scenario runs)
bash tools/fetch_coco128.sh

python3 -m unittest -q
```

## CLI: pip vs repo

- Environment report:
  `yolozu doctor --output -`
  / `python3 tools/yolozu.py doctor --output reports/doctor.json`
- Export smoke (no inference):
  `yolozu export --backend labels --dataset data/smoke --output runs/smoke/predictions_labels.json`
  / same in repo wrapper.
- Folder inference + overlays/HTML:
  `yolozu predict-images --backend onnxrt --input-dir data/smoke/images/val --model runs/smoke/model.onnx`
  / `python3 tools/yolozu.py predict-images ...`
- Backend parity check:
  `yolozu parity --reference reports/pred_ref.json --candidate reports/pred_cand.json`
  / `python3 tools/check_predictions_parity.py ...`
- Validate dataset layout:
  `yolozu validate dataset data/smoke --strict`
  / `python3 tools/validate_dataset.py ... --strict`
- Validate predictions JSON:
  `yolozu validate predictions reports/predictions.json --strict`
  / `python3 tools/validate_predictions.py ... --strict`
- COCOeval mAP:
  `yolozu eval-coco --dataset data/smoke --predictions data/smoke/predictions/predictions_dummy.json`
  (`yolozu[coco]`) / `python3 tools/eval_coco.py ...`
- Long-tail post-hoc + report:
  `yolozu calibrate --method fracal ... && yolozu eval-long-tail ...`
  / same via `python3 tools/yolozu.py ...`
- Long-tail train recipe:
  `yolozu long-tail-recipe --dataset data/smoke ...`
  / same via `python3 tools/yolozu.py ...`
- Instance-seg eval (PNG masks):
  `yolozu eval-instance-seg --dataset /path --predictions preds.json ...`
  / `python3 tools/eval_instance_segmentation.py ...`
- ONNXRuntime CPU export:
  `yolozu onnxrt export ...` (`yolozu[onnxrt]`)
  / `python3 tools/export_predictions_onnxrt.py ...`
- Training pipeline:
  `yolozu train configs/examples/train_contract.yaml --run-id exp01` (`yolozu[train]`)
  / `python3 rtdetr_pose/tools/train_minimal.py ...`
- Scenario suite:
  `yolozu test configs/examples/test_setting.yaml`
  / `python3 tools/run_scenarios.py ...`

The “power-user” unified CLI lives in-repo: `python3 tools/yolozu.py --help`.

Path behavior in tool CLIs:
- Relative input paths are resolved from the current working directory (with repo-root fallback for compatibility).
- Relative output paths are written under the current working directory.
- For config-driven tools such as `tools/tune_gate_weights.py`,
  relative paths in the config are resolved from the config file directory.

## Container images (GHCR)

YOLOZU can publish Docker images to GitHub Container Registry (GHCR) on tags `vX.Y.Z`.

- Minimal (no torch): `ghcr.io/<owner>/yolozu:<tag>`
- Demo (includes torch): `ghcr.io/<owner>/yolozu-demo:<tag>`

Examples:

```bash
docker run --rm ghcr.io/<owner>/yolozu:0.1.0 doctor --output -
docker run --rm ghcr.io/<owner>/yolozu-demo:0.1.0 demo continual --method ewc_replay
```

Publish trigger:
- Push a tag `vX.Y.Z` to run `.github/workflows/container.yml`.
- If the tag existed before the workflow was added,
  run it manually via GitHub Actions (workflow_dispatch) or cut a new tag.
- This workflow is optional publish automation and not the primary PR gate.

Details: [deploy/docker/README.md](deploy/docker/README.md)

### GPU notes
- GPU is supported (training/inference): install CUDA-enabled PyTorch in your environment and use `--device cuda:0`.
- CI/dev does not require GPU; many checks are CPU-friendly.

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
