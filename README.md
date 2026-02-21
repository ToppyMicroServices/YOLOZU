# YOLOZU (萬)

日本語: [`Readme_jp.md`](Readme_jp.md)

[![PyPI](https://img.shields.io/pypi/v/yolozu?logo=pypi&logoColor=white)](https://pypi.org/project/yolozu/)
[![Python >=3.10](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://pypi.org/project/yolozu/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

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

Key points:

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

- **Bring-your-own inference + contract-first evaluation**:
  run inference in PyTorch / ONNXRuntime / TensorRT / C++ / Rust,
  export the same `predictions.json`, and compare apples-to-apples.
- **Safe TTT (test-time training)**: presets + guard rails + reset policies (see `docs/ttt_protocol.md`).
- **Apache-2.0-only ops**: license policy + checks to keep the toolchain clean (see `docs/license_policy.md`).
- **Unified CLI**: `yolozu` (pip) + `python3 tools/yolozu.py` (repo)
  wrap backends with consistent args and caching (`--cache`),
  and always write run metadata (git SHA / env / GPU / config hash).
- **Parity + benchmarks**: backend diff stats (torch vs onnxrt vs trt) and fixed-protocol latency/FPS reports.
- **AI-friendly repo surface**: stable schemas + `tools/manifest.json` for tool discovery / automation.

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

YOLOZU evaluates instance segmentation using **per-instance binary PNG masks** (no RLE/polygons required).

Predictions JSON (minimal):
```json
[
  {
    "image": "000001.png",
    "instances": [
      { "class_id": 0, "score": 0.9, "mask": "masks/000001_inst0.png" }
    ]
  }
]
```

Validate an artifact:
```bash
python3 tools/validate_instance_segmentation_predictions.py reports/instance_seg_predictions.json
```

Eval outputs:
- mask mAP (`map50`, `map50_95`)
- per-class AP table
- per-image diagnostics (TP/FP/FN, mean IoU) and overlay selection
  (`--overlay-sort {worst,best,first}`; default: `worst`)

Run the synthetic demo and render overlays/HTML:
```bash
python3 tools/eval_instance_segmentation.py \
  --dataset examples/instance_seg_demo/dataset \
  --split val2017 \
  --predictions examples/instance_seg_demo/predictions/instance_seg_predictions.json \
  --pred-root examples/instance_seg_demo/predictions \
  --classes examples/instance_seg_demo/classes.txt \
  --html reports/instance_seg_demo_eval.html \
  --overlays-dir reports/instance_seg_demo_overlays \
  --max-overlays 10
```

Same via the unified CLI:
```bash
python3 tools/yolozu.py eval-instance-seg \
  --dataset examples/instance_seg_demo/dataset \
  --split val2017 \
  --predictions examples/instance_seg_demo/predictions/instance_seg_predictions.json \
  --pred-root examples/instance_seg_demo/predictions \
  --classes examples/instance_seg_demo/classes.txt \
  --html reports/instance_seg_demo_eval.html \
  --overlays-dir reports/instance_seg_demo_overlays \
  --max-overlays 10
```

Optional: prepare COCO instance-seg dataset with per-instance PNG masks (requires `pycocotools`):
```bash
python3 tools/prepare_coco_instance_seg.py \
  --coco-root data/coco128 \
  --split train2017 \
  --out data/smoke_instance_seg
```

Optional: convert COCO instance-seg predictions (RLE/polygons) into YOLOZU PNG masks (requires `pycocotools`):
```bash
python3 tools/convert_coco_instance_seg_predictions.py \
  --predictions reports/smoke_coco_instance_seg_preds.json \
  --instances-json data/coco/annotations/instances_val2017.json \
  --output reports/instance_seg_predictions.json \
  --masks-dir reports/instance_seg_masks
```

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

The trainer implementation lives in `rtdetr_pose/rtdetr_pose/train_minimal.py`
(source-checkout wrapper: `rtdetr_pose/tools/train_minimal.py`).

Quickest path (source checkout):

```bash
python3 -m pip install -r requirements-test.txt
bash tools/fetch_coco128.sh
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --config rtdetr_pose/configs/base.json \
  --max-steps 50 \
  --run-dir runs/train_minimal_smoke
```

Config-driven path (recommended for repeatability):

```bash
yolozu train configs/examples/train_setting.yaml
```

### Custom data training (recommended flow)

1) Prepare YOLO-format dataset root:
- `images/train/*.{jpg,png}` (and `images/val/*.{jpg,png}`)
- `labels/train/*.txt` (and `labels/val/*.txt`, YOLO txt: `class cx cy w h`)

2) Validate dataset before training:

```bash
yolozu validate dataset data/smoke --split val --strict
yolozu validate dataset data/smoke --split val --strict --no-check-images
```

3) Copy and edit contract config (set dataset root/splits and training knobs):

```bash
cp configs/examples/train_contract.yaml configs/runtime/train_contract_custom.yaml
```

4) Launch run-contract training:

```bash
yolozu train configs/runtime/train_contract_custom.yaml --run-id custom_exp01
```

5) Resume / dry-run when needed:

```bash
yolozu train configs/runtime/train_contract_custom.yaml --run-id custom_exp01 --resume
yolozu train configs/runtime/train_contract_custom.yaml --run-id custom_exp01 --dry-run
```

6) Export predictions from your trained checkpoint and evaluate:

```bash
python3 tools/export_predictions.py \
  --adapter rtdetr_pose \
  --config rtdetr_pose/configs/base.json \
  --checkpoint runs/custom_exp01/checkpoints/best.pt \
  --dataset data/smoke \
  --split val \
  --wrap \
  --output reports/custom_exp01_predictions_val.json
yolozu eval-coco \
  --dataset data/smoke \
  --split val \
  --predictions reports/custom_exp01_predictions_val.json \
  --bbox-format cxcywh_norm
```

For quick experiments, set `--run-dir`, which writes a standard artifact set:
- `metrics.jsonl` (+ final `metrics.json` / `metrics.csv`)
- `checkpoint.pt` (+ optional `checkpoint_bundle.pt`)
- `model.onnx` (+ `model.onnx.meta.json`)
- `run_record.json` (git SHA / platform / args)

For production-style repeatability (fixed paths, resume, best/last, parity gate), use the **run contract**:

```bash
yolozu train configs/examples/train_contract.yaml --run-id exp01

# Resume (full state: model/optim/sched/amp/ema/step + RNG)
yolozu train configs/examples/train_contract.yaml --run-id exp01 --resume

# Smoke wiring check (1 step then exit 0)
yolozu train configs/examples/train_contract.yaml --run-id exp01 --dry-run
```

Contracted artifacts (fixed paths):
- `runs/exp01/checkpoints/{last,best}.pt`
- `runs/exp01/reports/train_metrics.jsonl` (1 line per optimizer step)
- `runs/exp01/reports/val_metrics.jsonl`
- `runs/exp01/reports/config_resolved.yaml`
- `runs/exp01/reports/run_meta.json`
- `runs/exp01/reports/onnx_parity.json` (Torch vs ONNXRuntime; fails run on drift by default)
- `runs/exp01/exports/model.onnx` (+ `model.onnx.meta.json`)

Best definition: max `map50_95` on validation.

Run contract spec: [`docs/run_contract.md`](docs/run_contract.md)

Trainer core features (implemented):
- Full resume (model/optim/sched/AMP scaler/EMA/progress + RNG) via `--resume-from` or contracted `--resume`.
- NaN/Inf guard with skip + LR decay knobs
  (`--stop-on-non-finite-loss`, `--non-finite-max-skips`, `--non-finite-lr-decay`).
- Grad clipping (`--clip-grad-norm`, recommended >0 for pose/TTT/MIM stability).
- AMP (`--amp {none,fp16,bf16}`), EMA (`--use-ema` + `--ema-eval`), DDP (`--ddp` via `torchrun`).
- (Optional) `torch.compile`: `--torch-compile`
  (+ `--torch-compile-*`; experimental; falls back by default if compile fails).
- (Optional) torchao quantization / QLoRA:
  `--torchao-quant {int8wo,int4wo}` / `--qlora`
  (experimental; requires torchao).
- Lightweight aug: multiscale (`--multiscale`), hflip (`--hflip-prob`),
  photometric HSV/grayscale/noise/blur (`--hsv-*`, `--gray-prob`, `--gaussian-noise-*`, `--blur-*`;
  effective when `--real-images` is used).
- Validation cadence: epoch (`--val-every`) and step-based (`--val-every-steps`).
- Early stop: `--early-stop-patience` (+ `--early-stop-min-delta`).
- Activation swapping (config):
  set `model.backbone_activation` / `model.head_activation`
  to `silu|gelu|swish|hardswish|hard-swish|leakyrelu`,
  or use `model.activation_preset` (recommended default: `default` = SiLU/SiLU).
- Backbone swap (config):
  prefer `model.backbone.name|norm|args` plus `model.projector.d_model`
  (legacy `model.backbone_name` is still accepted).
- Backbone contract:
  model backbones return `[P3,P4,P5]` at strides `[8,16,32]`;
  channels are aligned by 1x1 projection before encoder input.

Common next checks:

```bash
python3 tools/plot_metrics.py --jsonl runs/train_minimal_smoke/metrics.jsonl --out reports/train_loss.png
python3 tools/export_predictions.py \
  --adapter rtdetr_pose \
  --config rtdetr_pose/configs/base.json \
  --checkpoint runs/train_minimal_smoke/checkpoint.pt \
  --max-images 20 \
  --wrap \
  --output reports/predictions.json
python3 tools/eval_coco.py \
  --dataset data/smoke \
  --predictions reports/predictions.json \
  --bbox-format cxcywh_norm \
  --max-images 20 \
  --dry-run
```

If you explicitly want the downloaded subset path, use:

```bash
python3 tools/eval_coco.py \
  --dataset data/coco128 \
  --predictions reports/predictions.json \
  --bbox-format cxcywh_norm \
  --max-images 20 \
  --dry-run
```

Backbone details and extension guide: [docs/backbones.md](docs/backbones.md)

Plot a loss curve (requires matplotlib):

```bash
python3 tools/plot_metrics.py \
  --jsonl runs/train_minimal_smoke/metrics.jsonl \
  --out reports/train_loss.png
```

### ONNX export

ONNX export runs when `--run-dir` is set (defaulting to `runs/smoke/model.onnx`) or when `--onnx-out` is provided.

Useful flags:
- `--run-dir runs/smoke`
- `--onnx-out runs/smoke/model.onnx`
- `--onnx-meta-out runs/smoke/model.onnx.meta.json`
- `--onnx-opset <int>`
- `--onnx-dynamic-hw` (dynamic H/W axes)

## Dataset format (YOLO + optional metadata)

Base dataset format:
- Images: `images/train/*.{jpg,png}` (and `images/val/*.{jpg,png}`)
- Labels: `labels/train/*.txt` (and `labels/val/*.txt`, YOLO: `class cx cy w h` normalized)

### Compatibility (YOLOv8 / YOLO11 / YOLOX)

- Ultralytics YOLOv8 / YOLO11:
  if your dataset root contains `images/train` + `labels/train`
  (and `images/val` + `labels/val`),
  it is already compatible. Use `yolozu validate dataset data/smoke --strict`.
  - You can also pass an Ultralytics `data.yaml` as `--dataset`
    (expects `path:` + `train:`/`val:` pointing to `images/train` and `images/val`).
- YOLOX: common setups use COCO JSON (`instances_*.json`). Convert once with `tools/prepare_coco_yolo.py`
  to generate YOLO-format labels (and an optional `dataset.json` descriptor) under a YOLOZU-compatible dataset root.
  - If you want a **read-only** wrapper (no label txt generation),
    use import adapters: [`docs/import_adapters.md`](docs/import_adapters.md).

Optional per-image metadata (JSON): `labels/train/000001.json` (same pattern for `labels/val/*.json`)
- Masks/seg: `mask_path` / `mask` / `M`
- Depth: `depth_path` / `depth` / `D_obj`
- Pose: `R_gt` / `t_gt` (or `pose`)
- Intrinsics: `K_gt` / `intrinsics` (also supports OpenCV FileStorage-style `camera_matrix: {rows, cols, data:[...]}`)

Notes on units (pixels vs mm/m) and intrinsics coordinate frames:
- [docs/predictions_schema.md](docs/predictions_schema.md)

### Mask-only labels (seg -> bbox/class)

If YOLO txt labels are missing and a mask is provided, bbox+class can be derived from masks.
Details (including color/instance modes and multi-PNG-per-class options) are documented in:
- [rtdetr_pose/README.md](rtdetr_pose/README.md)

## Symmetry-aware utilities (object symmetry checks)

- Symmetry specs live in `configs/runtime/symmetry.json` (validated loader: `yolozu.config.load_symmetry_map`).
- Core ops: `yolozu/symmetry.py` (types: `none`, `Cn`/`C2`/`C4`, `Cinf`).
- Symmetry-aware template verification: `yolozu/template_verification.py`.
- Tests: `python3 -m pytest -q tests/test_symmetry.py tests/test_template_verification.py`.

## Evaluation / contracts (stable)

This repo evaluates models through a stable predictions JSON format:
- Schema doc: [docs/predictions_schema.md](docs/predictions_schema.md)
- Machine-readable schema: [schemas/predictions.schema.json](schemas/predictions.schema.json)

Adapters power `tools/export_predictions.py --adapter <name>` and follow:
- [docs/adapter_contract.md](docs/adapter_contract.md)

## Export + evaluate predictions (TTT supported)

There are two common workflows:

### A) Evaluate precomputed `predictions.json` (no torch required)

If you run real inference elsewhere (PyTorch/ONNXRuntime/TensorRT/etc.),
you can evaluate this repo without installing heavy deps locally.

- Validate the JSON:
  ```bash
  python3 tools/validate_predictions.py reports/predictions.json
  ```
- Consume predictions locally:
  ```bash
  yolozu test configs/examples/test_setting.yaml \
    --adapter precomputed \
    --predictions reports/predictions.json \
    --max-images 50
  python3 tools/run_scenarios.py --adapter precomputed --predictions reports/predictions.json --max-images 50
  ```

### B) Export predictions in this repo (Torch/ONNXRuntime/TensorRT)

TTT updates weights in-memory at inference time and is OFF by default (opt-in via `--ttt`).

- Torch backend (`rtdetr_pose`, supports **TTA + TTT**):
  - Baseline:

    ```bash
    python3 tools/yolozu.py export \
      --backend torch \
      --checkpoint runs/smoke/checkpoints/best.pt \
      --device cuda \
      --max-images 50 \
      --output reports/predictions.json
    ```
  - TTT (Tent, safe preset):

    ```bash
    python3 tools/yolozu.py export \
      --backend torch \
      --checkpoint runs/smoke/checkpoints/best.pt \
      --device cuda \
      --max-images 50 \
      --ttt \
      --ttt-preset safe \
      --ttt-reset sample \
      --ttt-log-out reports/ttt_log_safe.json \
      --output reports/predictions_ttt_safe.json
    ```
  - TTT batch/chunk knobs: add `--ttt-batch-size <N>` and `--ttt-max-batches <K>`
    to cap adaptation cost (example: `--ttt-batch-size 4 --ttt-max-batches 8`).
  - TTT reset behavior: use `--ttt-reset stream` for one adaptation phase then fast prediction,
    or `--ttt-reset sample` for per-image/per-batch reset-ablation runs.
  - Note: `tools/yolozu.py export` always writes the wrapped
    `{ "predictions": [...] }` form (so `--wrap` is not needed).
  - Note: TTT is supported in repo tooling (`python3 tools/yolozu.py ...`)
    on the torch backend; pip CLI `yolozu export` is intentionally
    smoke-only (dummy/labels).
  - Recommended protocol + rationale (domain shift, presets, guards): [docs/ttt_protocol.md](docs/ttt_protocol.md)
- ONNXRuntime/TensorRT backends:
  use `python3 tools/yolozu.py export --backend onnxrt|trt ...`
  (TTT is torch-only; use TTA or export precomputed predictions for other backends).

Supported predictions JSON shapes:
- `[{"image": "...", "detections": [...]}, ...]`
- `{ "predictions": [ ... ] }`
- `{ "000000000009.jpg": [...], "/abs/path.jpg": [...] }` (image -> detections)

Schema details:
- [docs/predictions_schema.md](docs/predictions_schema.md)

## COCO mAP (end-to-end, no NMS)

To compete on **e2e mAP** (NMS-free), evaluate detections as-is (no NMS postprocess applied).

This repo includes a COCO-style evaluator that:
- Builds COCO ground truth from YOLO-format labels
- Converts YOLOZU predictions JSON into COCO detections
- Runs COCO mAP via `pycocotools` (optional dependency)

Example (smoke-first, works with bundled assets):

```bash
python3 tools/export_predictions.py \
  --adapter dummy \
  --dataset data/smoke \
  --split val \
  --max-images 10 \
  --wrap \
  --output reports/predictions_smoke.json
python3 tools/eval_coco.py \
  --dataset data/smoke \
  --split val \
  --predictions reports/predictions_smoke.json \
  --bbox-format cxcywh_norm \
  --max-images 10 \
  --dry-run
```

Alternative (coco128 quick run):

```bash
python3 tools/export_predictions.py --adapter dummy --max-images 50 --wrap --output reports/predictions.json
python3 tools/eval_coco.py \
  --dataset data/coco128 \
  --predictions reports/predictions.json \
  --bbox-format cxcywh_norm \
  --max-images 50
```

Note:
- `--bbox-format cxcywh_norm` expects bbox dict `{cx,cy,w,h}`
  normalized to `[0,1]` (matching the RTDETR pose adapter bbox head).
- For run-contract variable naming and artifact conventions, see
  `docs/run_contract.md`; README examples intentionally use fixed
  concrete paths such as `runs/smoke`.

## Training recipe (v1)

Reference recipe for external training runs (augment, multiscale, schedule, EMA):
- `docs/training_recipe_v1.md`

## Training, inference, export (quick steps)

- `docs/training_inference_export.md`

## Hyperparameter sweep harness

Run a configurable sweep and emit CSV/MD tables:
- `docs/hpo_sweep.md`

## Latency/FPS benchmark harness

Report latency/FPS per YOLO26 bucket and archive runs over time:
- `docs/benchmark_latency.md`

## Inference-time gating / score fusion

Fuse detection/template/uncertainty signals into a single score and tune weights offline (CPU-only):
- `docs/gate_weight_tuning.md`

## TensorRT FP16/INT8 pipeline

Reproducible engine build + parity validation steps:
- `docs/tensorrt_pipeline.md`

## External baselines (Apache-2.0-friendly)

This repo does **not** require (or vendor) any GPL/AGPL inference code.

To compare against external baselines (including YOLO26) while keeping this repo Apache-2.0-only:
- Run baseline inference in your own environment/implementation (ONNX Runtime / TensorRT / custom code).
- Export detections to YOLOZU predictions JSON (see schema below).
- (Optional) Normalize class ids using COCO `classes.json` mapping.
- Validate + evaluate mAP in this repo:

  ```bash
  python3 tools/validate_predictions.py reports/predictions.json
  python3 tools/eval_coco.py \
    --dataset data/smoke \
    --split val \
    --predictions reports/predictions.json \
    --bbox-format cxcywh_norm
  ```

Minimal predictions entry schema:
- `{"image": "/abs/or/rel/path.jpg", "detections": [...]}`
  where each detection includes `class_id`, `score`, and `bbox {cx,cy,w,h}`.

Optional class-id normalization (when your exporter produces COCO `category_id`):

```bash
python3 tools/normalize_predictions.py \
  --input reports/predictions.json \
  --output reports/predictions_norm.json \
  --classes data/coco-yolo/labels/val2017/classes.json \
  --wrap
```

## COCO dataset prep (official JSON -> YOLO-format)

If you have the official COCO layout (images + `annotations/instances_*.json`), you can generate YOLO-format labels:

- `python3 tools/prepare_coco_yolo.py --coco-root data/coco --split val2017 --out data/coco-yolo`

This creates:
- `data/coco-yolo/labels/val2017/*.txt` (YOLO normalized `class cx cy w h`)
- `data/coco-yolo/labels/val2017/classes.json` (category_id <-> class_id mapping)

### Dataset layout under `data/`

For local development, keep datasets under `data/`:
- Debug/smoke: `data/coco128` (already included)
- Full COCO (official): `data/coco` (your download)
- YOLO-format labels generated from official JSON: `data/coco-yolo` (your output from `tools/prepare_coco_yolo.py`)

### Size-bucket competition (yolo26n/s/m/l/x)

If you export `yolo26n/s/m/l/x` predictions as separate JSON files (e.g. `reports/pred_yolo26n.json`, ...),
you can score them together:

- Protocol details: `docs/yolo26_eval_protocol.md`
- Evaluate suite:

  ```bash
  python3 tools/eval_suite.py \
    --protocol yolo26 \
    --dataset data/coco-yolo \
    --predictions-glob 'reports/pred_yolo26*.json' \
    --output reports/eval_suite.json
  ```
- Fill in targets: `baselines/yolo26_targets.json`
- Validate targets: `python3 tools/validate_map_targets.py --targets baselines/yolo26_targets.json`
- Check pass/fail:

  ```bash
  python3 tools/check_map_targets.py \
    --suite reports/eval_suite.json \
    --targets baselines/yolo26_targets.json \
    --key map50_95
  ```
- Print a table:

  ```bash
  python3 tools/print_leaderboard.py \
    --suite reports/eval_suite.json \
    --targets baselines/yolo26_targets.json \
    --key map50_95
  ```
- Archive the run (commands + hardware + suite output):

  ```bash
  python3 tools/import_yolo26_baseline.py --dataset data/coco-yolo --predictions-glob 'reports/pred_yolo26*.json'
  ```

### Debug without `pycocotools`

If you don't have `pycocotools` installed yet, you can still validate/convert predictions on `data/coco128`:

```bash
python3 tools/export_predictions.py --adapter dummy --max-images 10 --wrap --output reports/predictions_dummy.json
python3 tools/eval_coco.py --predictions reports/predictions_dummy.json --dry-run
```

## Deployment notes
- Keep symmetry/commonsense logic in lightweight postprocess utilities, outside any inference graph export.

## License

Code in this repository is licensed under the Apache License, Version 2.0. See `LICENSE`.
