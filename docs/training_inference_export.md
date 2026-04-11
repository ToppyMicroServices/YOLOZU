# Training, inference, and export

This note provides a minimal, end-to-end path for training, inference, and exporting predictions.

Note: the in-repo trainer under `rtdetr_pose/` is the repo's **reference trainer**. In other words,
training is supported in YOLOZU for this RT-DETR pose lane. What YOLOZU does not claim is a
general-purpose training framework for every model family. The reference trainer still supports a **production-style run contract**
(fixed artifact paths, full resume, safety guards, export + parity checks).

For YOLO-style training outside the reference trainer, use the **external training lane**.
YOLOX is the primary Apache-2.0-friendly path; the Ultralytics bridge remains optional and
is documented as a separate runtime/license boundary.

The platform-level docs for training are:

- [`training_backend_interface.md`](training_backend_interface.md)
- [`training_capability_matrix.md`](training_capability_matrix.md)
- [`training_orchestration.md`](training_orchestration.md)

## Current training support

Use this scope boundary when deciding whether YOLOZU should own the training step or just the artifact/evaluation boundary:

| Lane | Status | What it is for | Notes |
|---|---|---|---|
| RT-DETR pose reference trainer | Stable reference lane | In-repo training, resume, export, parity, and run artifacts | This is the default `yolozu train` path. |
| YOLOX external lane | Supported external lane | Apache-2.0-friendly YOLO-style training launched from the top-level CLI | Prefer this when you want a YOLO-style trainer without pulling copyleft code into YOLOZU. |
| MMDetection external lane | Experimental external lane | Bbox / instance-seg training launched from the top-level CLI | Best fit when the backend-native detection stack already lives in OpenMMLab. |
| MMPose external lane | Experimental external lane | Keypoints / pose training launched from the top-level CLI | Export stays backend-specific today; evaluation/parity still converge on YOLOZU once predictions are exported. |
| MMSeg external lane | Experimental external lane | Semantic segmentation training launched from the top-level CLI | Export stays backend-specific today; YOLOZU takes over at evaluation once masks are exported. |
| Ultralytics bridge | Optional external bridge | User-installed external runtime bridge | Keep the license boundary explicit; see `docs/license_policy.md`. |
| HF DETR bridge | Optional external bridge | User-installed DETR-family bridge | Useful when a DETR-family training stack already exists outside this repo. |
| Generic training platform for every model family | Not claimed | Universal training framework | YOLOZU does not claim this scope. It standardizes the run artifacts and the predictions interface contract around the supported lanes above. |

## Training platform layer

The training platform layer now has five explicit pieces:

1. one canonical `TrainConfig` projection
2. one backend interface registry
3. one shared training run summary interface contract
4. one training capability matrix
5. one lightweight orchestration entrypoint

This does not mean every backend is identical. It means the repo can describe
different training lanes with one shared top-level shape.

For external backends, YOLOZU now standardizes one wrapper-level run bundle under
`work_dir/`:

- `dataset/`
- `configs/train_config_projection.json`
- `reports/training_summary.json`
- `reports/external_run_meta.json`
- `reports/launcher_plan.json`
- `reports/execution.json`

This keeps external lanes auditable even when the backend-native trainer owns the
actual checkpoint layout.

External lanes now also write `next_steps` into `training_summary.json`, so the
report itself tells you which export / evaluation / parity command to run next.
For MM-family lanes, these steps make the boundary explicit:

- `mmdetection`: bbox is wrapper-ready; mask export can still be backend-specific.
- `mmpose`: training is first-class, while prediction export remains backend-specific.
- `mmseg`: training is first-class, while prediction export remains backend-specific.

OpenCV DNN and ONNX Runtime are not training lanes. They remain inference/export
targets after a model has already been trained.

## TL;DR (copy-paste)

```bash
python3 -m pip install -r requirements-test.txt
bash tools/fetch_coco128.sh
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --config rtdetr_pose/configs/base.json \
  --max-steps 50 \
  --run-dir runs/train_minimal_smoke
python3 tools/export_predictions.py \
  --adapter rtdetr_pose \
  --config rtdetr_pose/configs/base.json \
  --checkpoint runs/train_minimal_smoke/checkpoint.pt \
  --max-images 20 \
  --wrap \
  --output reports/predictions.json
python3 tools/eval_coco.py \
  --dataset data/coco128 \
  --predictions reports/predictions.json \
  --bbox-format cxcywh_norm \
  --max-images 20 \
  --dry-run
```

## Canonical COCO data placement

All data lives under `data/` (gitignored). The following table standardizes
the four accepted dataset paths and how to obtain each:

| Path | Format | How to obtain | Use case |
|---|---|---|---|
| `data/smoke/` | YOLO (committed) | Already in repo | Offline CI, unit tests |
| `data/coco128/` | YOLO (128 images) | `bash tools/fetch_coco128.sh` | Quick smoke training / eval |
| `data/coco/` | Standard COCO (`images/val2017/` + `annotations/instances_val2017.json`) | Manual download or `python3 scripts/download_coco_instances_tiny.py` | Native COCO evaluation, instance-seg |
| `data/coco-yolo/` | YOLO-converted from full COCO | `python3 tools/prepare_coco_yolo.py --coco-root data/coco --split val2017 --out data/coco-yolo` | Full YOLO-format training / eval |

### Copy-paste setup commands

```bash
# 1. Smoke dataset (already committed — nothing to do)
ls data/smoke/images/val/

# 2. COCO-128 (quick download, ~6 MB)
bash tools/fetch_coco128.sh

# 3. Full COCO val2017 (standard format)
#    Download from https://cocodataset.org/#download and arrange as:
#      data/coco/annotations/instances_val2017.json
#      data/coco/images/val2017/*.jpg

# 4. Convert COCO → YOLO format
python3 tools/prepare_coco_yolo.py \
  --coco-root data/coco \
  --split val2017 \
  --out data/coco-yolo
# Writes data/coco-yolo/dataset.json with absolute paths.
# Add --copy-images to copy images into data/coco-yolo/images/<split>/.
```

### Path resolution in code

CLI tools accept `--dataset` / `--dataset-root` as relative paths (resolved
from `$PWD`). No environment variables are required — paths are resolved via:

```python
dataset_root = Path(str(args.dataset)).expanduser()
if not dataset_root.is_absolute():
    dataset_root = Path.cwd() / dataset_root
```

The adapter registry (`yolozu.datasets.registry.probe_format`) auto-detects
whether a directory is COCO-native or YOLO-format, so both `data/coco/` and
`data/coco-yolo/` work transparently.

## Training (RT-DETR pose reference trainer)

1) Install dependencies (CPU PyTorch for local dev):
- python3 -m pip install -r requirements-test.txt

2) Fetch the sample dataset (coco128):
- bash tools/fetch_coco128.sh

3) Run the minimal trainer:
- python3 rtdetr_pose/tools/train_minimal.py --dataset-root data/coco128 --config rtdetr_pose/configs/base.json --max-steps 50 --use-matcher

### Backbone swap (P3/P4/P5 contract)

Backbone is now configurable via `model.backbone.*` and projected to transformer `d_model` via `model.projector.d_model`.

Example config fragment:

```yaml
model:
  backbone:
    name: cspdarknet_s
    norm: bn
    args:
      width_mult: 0.5
      depth_mult: 0.5
  projector:
    d_model: 256
```

Other supported names: `resnet50`, `convnext_tiny`, `cspresnet`, `tiny_cnn`.

Contract details and extension guide: [backbones.md](backbones.md)

Common options:
- --device auto
- --device mps
- --batch-size 4
- --num-queries 10
- --stage-off-steps 1000 --stage-k-steps 1000

macOS / Apple Silicon beta notes:
- `--device auto` resolves in `cuda -> mps -> cpu` order.
- `--device mps` is supported for the reference trainer.
- `--amp fp16|bf16` on MPS is best-effort; unsupported autocast modes warn and fall back to fp32.
- If an op is still missing on MPS, retry with `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Post-train ONNX export is attempted on CPU even when training ran on CUDA/MPS. This avoids backend-specific exporter failures and keeps the exported artifact path portable.
- --cost-z 1.0 --cost-rot 1.0 --cost-t 1.0
- --cost-z-start-step 500 --cost-rot-start-step 1000 --cost-t-start-step 1500
- --checkpoint-out reports/rtdetr_pose_ckpt.pt
- --metrics-jsonl reports/train_metrics.jsonl
- --metrics-csv reports/train_metrics.csv

### Depth mode (none / sidecar / fuse_mid)

`rtdetr_pose/tools/train_minimal.py` supports optional depth integration while preserving the backbone swap boundary (`[P3,P4,P5]`):

- `--depth-mode none` (default): no depth path; baseline-compatible behavior.
- `--depth-mode sidecar`: ingest sidecar depth (`depth_path` / `depth`) and propagate per-image `depth_valid`.
- `--depth-mode fuse_mid`: apply lightweight depth fusion after projector (outside backbone), with optional `--depth-dropout` modality dropout.

Safety and unit handling:

- `--depth-unit` controls whether absolute-depth constraints are allowed (`unspecified|relative|metric`, default `unspecified`).
- In non-metric modes, absolute-depth matcher costs are disabled for safety (`cost_z`, `cost_t`).
- `--depth-scale` applies scaling to sidecar depth values before fusion/consumption.

Example (mixed sidecar depth ingestion):

```bash
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --config rtdetr_pose/configs/base.json \
  --use-matcher \
  --depth-mode sidecar \
  --depth-unit metric \
  --depth-scale 1.0
```

### Imbalance handling + explicit backbone override + strict task-data checks

`rtdetr_pose/tools/train_minimal.py` now supports:

- `--imbalance-strategy class_balanced` (+ `--imbalance-gamma`, `--imbalance-min-weight`, `--imbalance-max-weight`, `--imbalance-aggregate`)
- `--backbone-name`, `--backbone-norm`, `--backbone-args '{"...": ...}'` (explicit runtime override without editing model JSON)
- `--strict-task-data` (fails fast when required real-data supervision is missing for bbox/keypoints/depth/pose)

Example:

```bash
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/real_multitask_fewshot \
  --split train --val-split val \
  --real-images --strict-task-data \
  --use-matcher --model-config rtdetr_pose/configs/base.json \
  --imbalance-strategy class_balanced --imbalance-gamma 1.0 \
  --backbone-name cspdarknet_s --backbone-norm bn \
  --backbone-args '{"width_mult": 0.5, "depth_mult": 0.34}' \
  --epochs 1 --max-steps 10
```

### Real-image few-shot multitask finetune demo (bbox/seg/keypoints/depth/pose6d)

Use these tools for an end-to-end staged demo on real source images:

```bash
# 0) Download tiny COCO subset manually (review license terms)
python3 scripts/download_coco_instances_tiny.py \
  --out-root data/coco --split val2017 --num-images 8 --seed 0 --force

# 1) Prepare compact real-image dataset (COCO images + annotation-derived sidecars)
python3 tools/prepare_real_multitask_fewshot.py \
  --instances-json data/coco/annotations/instances_val2017.json \
  --images-dir data/coco/images/val2017 \
  --out data/real_multitask_fewshot \
  --train-images 6 --val-images 2 --strict-provenance --force

# 2) Run staged finetuning:
# bbox -> segmentation -> keypoints -> depth -> pose6d
python3 tools/run_real_multitask_finetune_demo.py \
  --dataset-root data/real_multitask_fewshot \
  --out reports/real_multitask_finetune_demo \
  --device cpu \
  --epochs 1 --max-steps 1 --batch-size 2 --image-size 96 \
  --strict-provenance --force
```

The report is written to:
`reports/real_multitask_finetune_demo/multitask_finetune_demo_report.json`.
`prepare_summary.json` には各タスク教師信号の provenance（COCO GT / annotation-derived heuristic）も記録されます。

### External YOLO-style training lane (YOLOX primary, optional bridges second)

Use this path when you want YOLOZU to standardize dataset resolution, reports, and the
predictions interface contract while the actual YOLO training loop stays in an external repo/runtime.

Top-level `yolozu train` route:

```bash
python3 -m yolozu train \
  --external-backend yolox \
  configs/examples/finetune_external/yolox_s_finetune_smoke.py \
  --dataset data/smoke \
  --split val \
  --dry-run \
  --output reports/train_external_yolox.json
```

Equivalent repo helper:

```bash
python3 tools/support_external_training.py train-yolox \
  --dataset data/smoke \
  --split val \
  --exp configs/examples/finetune_external/yolox_s_finetune_smoke.py \
  --dry-run \
  --output reports/support_external_training.train_yolox.json
```

Optional Ultralytics bridge:

```bash
python3 -m yolozu train \
  --external-backend ultralytics \
  yolo11n.pt \
  --dataset data/smoke \
  --split val \
  --dry-run \
  --output reports/train_external_ultralytics.json

python3 tools/support_external_training.py train-ultralytics \
  --dataset data/smoke \
  --split val \
  --preset smoke \
  --dry-run \
  --output reports/support_external_training.train_ultralytics.json
```

Optional HF DETR bridge:

```bash
python3 -m yolozu train \
  --external-backend hf-detr \
  facebook/detr-resnet-50 \
  --dataset data/smoke \
  --split val \
  --dry-run \
  --output reports/train_external_hf_detr.json
```

Detectron2 external lane (`bbox`, instance `segmentation`, or `keypoints`
selected by your Detectron2 config):

```bash
python3 -m yolozu train \
  --external-backend detectron2 \
  configs/examples/finetune_external/detectron2_finetune_smoke.yaml \
  --dataset data/smoke \
  --split val \
  --task-family bbox \
  --dry-run \
  --output reports/train_external_detectron2_bbox.json
```

The same lane can describe Mask R-CNN and Keypoint R-CNN style runs:

```bash
python3 -m yolozu train \
  --external-backend detectron2 \
  /path/to/mask_rcnn_config.yaml \
  --dataset /path/to/dataset \
  --split train \
  --task-family segmentation \
  --train-opt DATASETS.TRAIN "(\"my_seg_train\",)" \
  --dry-run \
  --output reports/train_external_detectron2_seg.json

python3 -m yolozu train \
  --external-backend detectron2 \
  /path/to/keypoint_rcnn_config.yaml \
  --dataset /path/to/dataset \
  --split train \
  --task-family keypoints \
  --train-opt DATASETS.TRAIN "(\"my_kpts_train\",)" \
  --dry-run \
  --output reports/train_external_detectron2_keypoints.json
```

For non-dry execution, add `--train-script /path/to/detectron2/tools/train_net.py`.
Use repeated `--train-opt KEY VALUE` pairs to pass Detectron2 config overrides
such as dataset registration names.

### External finetune smoke matrix (YOLOX/Ultralytics/MMDetection/Detectron2/RT-DETR)

Use a single command to audit external finetune entrypoints and emit a stable interface contract report:

```bash
python3 tools/run_external_finetune_smoke.py \
  --dataset-root data/smoke \
  --split train \
  --output reports/external_finetune_smoke.json
```

Run real training for selected frameworks:

```bash
python3 tools/run_external_finetune_smoke.py \
  --dataset-root data/smoke \
  --split train \
  --non-dry-framework yolox \
  --non-dry-framework yolov \
  --non-dry-framework rtdetr \
  --yolox-train-script /path/to/YOLOX/tools/train.py \
  --epochs 1 --max-steps 1 --batch-size 2 --image-size 96 \
  --device cpu \
  --require-training-execution \
  --output reports/external_finetune_smoke.exec.json
```

Prepared per-framework templates:

- `configs/examples/finetune_external/yolox_s_finetune_smoke.py`
- `configs/examples/finetune_external/yolo_runtime_yolov8n_finetune_smoke.yaml`
- `configs/examples/finetune_external/mmdetection_finetune_smoke.py`
- `configs/examples/finetune_external/detectron2_finetune_smoke.yaml`
- `configs/examples/finetune_external/rtdetr_pose_finetune_smoke.yaml`

For MMDetection/Detectron2 external launchers, see:
`docs/external_finetune_smoke.md`.

Report behavior notes:

- YOLOX is the preferred Apache-2.0-friendly YOLO-style lane for external training.
- RT-DETR non-dry torch-missing failures are explicit (`failure_code=E_DEP_TORCH_MISSING`).
- With `--mmdet-train-script` / `--detectron2-train-script`, train-path audit can continue even when projection deps are unavailable; `projection_error` is recorded while `training_executed` reflects external launcher execution.

machine.dev / GPU example:

```bash
python3 tools/run_external_finetune_smoke.py \
  --dataset-root data/smoke \
  --split train \
  --non-dry-framework rtdetr \
  --device cuda \
  --epochs 1 --max-steps 1 --batch-size 2 --image-size 96 \
  --require-training-execution \
  --output reports/external_finetune_smoke.machine_dev.json
```

### Config source-of-truth and key mapping

`rtdetr_pose/tools/train_minimal.py` reads YAML/JSON via `--config`, then applies explicit CLI flags on top.

- Priority: **CLI flags > config file > built-in defaults**
- Config keys use argparse destination names (`--weight-decay` -> `weight_decay`)
- Alias: `grad_accum` in config is accepted and mapped to `gradient_accumulation_steps`
- In strict run-contract mode (`run_contract` or `run_id` or `config_version` present), unknown config keys fail fast

### Optimizer / solver options (code-accurate)

Supported optimizer choices (`--optimizer`):
- `adamw` (default)
- `sgd`

Relevant parameters:

| Key (CLI / config) | Type | Default | Choices / Notes |
|---|---:|---:|---|
| `--optimizer` / `optimizer` | str | `adamw` | `adamw`, `sgd` |
| `--lr` / `lr` | float | `1e-4` | base LR |
| `--weight-decay` / `weight_decay` | float | `0.01` | base WD |
| `--momentum` / `momentum` | float | `0.9` | used by SGD |
| `--nesterov` / `nesterov` | bool | `false` | SGD only |
| `--use-param-groups` / `use_param_groups` | bool | `false` | split backbone/head groups |
| `--backbone-lr-mult` / `backbone_lr_mult` | float | `1.0` | group LR multiplier |
| `--head-lr-mult` / `head_lr_mult` | float | `1.0` | group LR multiplier |
| `--backbone-wd-mult` / `backbone_wd_mult` | float | `1.0` | group WD multiplier |
| `--head-wd-mult` / `head_wd_mult` | float | `1.0` | group WD multiplier |
| `--wd-exclude-bias` / `wd_exclude_bias` | bool | `true` | set bias WD=0 |
| `--wd-exclude-norm` / `wd_exclude_norm` | bool | `true` | set norm WD=0 |

### LR scheduler options (code-accurate)

Supported scheduler choices (`--scheduler`):
- `none` (default)
- `cosine`
- `onecycle`
- `multistep`

Relevant parameters:

| Key (CLI / config) | Type | Default | Choices / Notes |
|---|---:|---:|---|
| `--scheduler` / `scheduler` | str | `none` | `none`, `cosine`, `onecycle`, `multistep` |
| `--min-lr` / `min_lr` | float | `0.0` | cosine `eta_min` |
| `--scheduler-milestones` / `scheduler_milestones` | str/list | `""` | comma list for multistep |
| `--scheduler-gamma` / `scheduler_gamma` | float | `0.1` | multistep decay |
| `--lr-warmup-steps` / `lr_warmup_steps` | int | `0` | linear warmup steps |
| `--lr-warmup-init` / `lr_warmup_init` | float | `0.0` | LR at warmup step 0 |

Note: `linear` scheduler is **not** a supported value in current code.

### Production run contract (recommended)

For reproducible runs with fixed artifact paths, full resume, best/last checkpoints, and an ONNX parity gate:

```bash
yolozu train configs/examples/train_contract.yaml --run-id exp01

# Resume (from runs/exp01/checkpoints/last.pt)
yolozu train configs/examples/train_contract.yaml --run-id exp01 --resume
```

Contracted artifacts live under `runs/<run_id>/...`:
- `checkpoints/{last,best}.pt`
- `reports/{train_metrics,val_metrics}.jsonl`
- `reports/config_resolved.yaml`
- `reports/run_meta.json`
- `reports/onnx_parity.json`
- `exports/model.onnx` (+ meta JSON)

Full spec: [run_contract.md](run_contract.md)

### Optimizer options

Choose between SGD and AdamW optimizers with configurable learning rates and weight decay:

```bash
# AdamW (default)
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --optimizer adamw \
  --lr 1e-4 \
  --weight-decay 0.01

# SGD with momentum and Nesterov
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --optimizer sgd \
  --lr 0.1 \
  --momentum 0.9 \
  --nesterov \
  --weight-decay 1e-4

# Use parameter groups with different lr/wd for backbone vs head
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --optimizer adamw \
  --lr 1e-4 \
  --use-param-groups \
  --backbone-lr-mult 0.1 \
  --head-lr-mult 1.0 \
  --backbone-wd-mult 0.5 \
  --head-wd-mult 1.0
```

### Learning rate scheduler options

Multiple scheduler types are supported with optional warmup:

```bash
# Cosine annealing with warmup
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --scheduler cosine \
  --min-lr 1e-6 \
  --lr-warmup-steps 500 \
  --lr-warmup-init 1e-6

# OneCycleLR for super-convergence
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --scheduler onecycle

# MultiStepLR with decay at specific steps
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --scheduler multistep \
  --scheduler-milestones 1000,2000,3000 \
  --scheduler-gamma 0.1
```

### LoRA / QLoRA options (code-accurate)

LoRA is enabled when `--lora-r > 0`.

| Key (CLI / config) | Type | Default | Choices / Notes |
|---|---:|---:|---|
| `--lora-r` / `lora_r` | int | `0` | `>0` enables LoRA |
| `--lora-alpha` / `lora_alpha` | float/null | `null` | null means `alpha=r` |
| `--lora-dropout` / `lora_dropout` | float | `0.0` | LoRA input dropout |
| `--lora-target` / `lora_target` | str | `head` | `head`, `all_linear`, `all_conv1x1`, `all_linear_conv1x1` |
| `--lora-freeze-base` / `lora_freeze_base` | bool | `true` | train LoRA-only when true |
| `--lora-train-bias` / `lora_train_bias` | str | `none` | `none`, `all` |

TorchAO / QLoRA integration:

| Key (CLI / config) | Type | Default | Choices / Notes |
|---|---:|---:|---|
| `--torchao-quant` / `torchao_quant` | str | `none` | `none`, `int8wo`, `int4wo` |
| `--torchao-required` / `torchao_required` | bool | `false` | fail run if quant unavailable |
| `--qlora` / `qlora` | bool | `false` | requires `lora_r>0`; forces `torchao_quant=int4wo` (if none) and `lora_freeze_base=true` |

### Additional fine-grained training knobs (selected)

| Key (CLI / config) | Type | Default | Notes |
|---|---:|---:|---|
| `--gradient-accumulation-steps` / `gradient_accumulation_steps` | int | `1` | alias in config: `grad_accum` |
| `--clip-grad-norm` / `clip_grad_norm` | float | `0.0` | `>0` enables clipping |
| `--use-ema` / `use_ema` | bool | `false` | EMA on train weights |
| `--ema-decay` / `ema_decay` | float | `0.999` | EMA decay |
| `--ema-eval` / `ema_eval` | bool | `false` | use EMA weights at eval/export |
| `--amp` / `amp` | str | `none` | `none`, `fp16`, `bf16` (CUDA only) |
| `--use-amp` / `use_amp` | bool | `false` | back-compat alias for `amp=fp16` |
| `--task-aligner` / `task_aligner` | str | `none` | `none`, `uncertainty` |
| `--cost-z` / `cost_z` | float | `0.0` | matcher depth cost |
| `--cost-rot` / `cost_rot` | float | `0.0` | matcher rotation cost |
| `--cost-t` / `cost_t` | float | `0.0` | matcher translation cost |
| `--cost-z-start-step` / `cost_z_start_step` | int | `0` | staged matcher cost gate |
| `--cost-rot-start-step` / `cost_rot_start_step` | int | `0` | staged matcher cost gate |
| `--cost-t-start-step` / `cost_t_start_step` | int | `0` | staged matcher cost gate |
| `--stage-off-steps` / `stage_off_steps` | int | `0` | offsets-only stage |
| `--stage-k-steps` / `stage_k_steps` | int | `0` | k-head-only stage |

Continual-learning regularizers in the same trainer:

| Key (CLI / config) | Type | Default | Notes |
|---|---:|---:|---|
| `--self-distill-from` / `self_distill_from` | str/null | `null` | enable teacher distillation |
| `--self-distill-weight` / `self_distill_weight` | float | `1.0` | distill global weight |
| `--self-distill-temperature` / `self_distill_temperature` | float | `1.0` | logits temperature |
| `--self-distill-kl` / `self_distill_kl` | str | `reverse` | `forward`, `reverse`, `sym` |
| `--self-distill-keys` / `self_distill_keys` | str | `logits,bbox` | comma list |
| `--derpp` / `derpp` | bool | `false` | DER++ replay distillation |
| `--derpp-teacher-key` / `derpp_teacher_key` | str | `derpp_teacher_npz` | record key/path |
| `--derpp-weight` / `derpp_weight` | float | `1.0` | DER++ global weight |
| `--ewc` / `ewc` | bool | `false` | EWC regularizer |
| `--ewc-lambda` / `ewc_lambda` | float | `1.0` | EWC penalty weight |
| `--si` / `si` | bool | `false` | SI regularizer |
| `--si-c` / `si_c` | float | `1.0` | SI penalty weight |
| `--si-epsilon` / `si_epsilon` | float | `1e-3` | SI stabilization |

### Advanced training options

```bash
# Gradient clipping
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --clip-grad-norm 1.0

# Gradient accumulation (effective batch size = batch_size * gradient_accumulation_steps)
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --batch-size 2 \
  --gradient-accumulation-steps 4

# Automatic Mixed Precision on CUDA
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --device cuda:0 \
  --use-amp

# macOS / MPS beta smoke
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --device mps \
  --amp fp16 \
  --max-steps 2

# Exponential Moving Average (EMA) of model weights
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --use-ema \
  --ema-decay 0.999 \
  --ema-eval  # Use EMA weights for evaluation/export

# Combined example: SGD + cosine scheduler + param groups + EMA
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --optimizer sgd \
  --momentum 0.9 \
  --scheduler cosine \
  --min-lr 1e-6 \
  --lr-warmup-steps 500 \
  --use-param-groups \
  --backbone-lr-mult 0.1 \
  --use-ema \
  --ema-decay 0.999 \
  --clip-grad-norm 1.0
```

Plot loss curve (requires matplotlib):
- python3 tools/plot_metrics.py --jsonl reports/train_metrics.jsonl --out reports/train_loss.png

## Inference (adapter run)

Use the adapter tools to run inference and produce predictions JSON.

- python3 tools/export_predictions.py --adapter rtdetr_pose --config rtdetr_pose/configs/base.json --checkpoint /path/to.ckpt --max-images 50 --wrap --output reports/predictions.json

Torch推論の軽量拡張（PyTorch 2.x）:
- `--infer-batch-size N`: 推論バッチサイズ（既定 `1`）
- `--torch-compile`: `torch.compile` 有効化
- `--torch-compile-backend` / `--torch-compile-mode`: compile backend/mode 指定
- `--torch-amp {off,fp16,bf16}`: autocast dtype
- `--torch-channels-last`: channels-last memory format
- `--torch-inference-mode` / `--no-torch-inference-mode`: forward context 切替

- python3 tools/export_predictions.py --adapter rtdetr_pose --config rtdetr_pose/configs/base.json --device cuda --infer-batch-size 8 --torch-compile --torch-compile-backend inductor --torch-compile-mode reduce-overhead --torch-amp bf16 --torch-channels-last --torch-inference-mode --max-images 50 --wrap --output reports/predictions_torch_compiled.json

Optional TTA:
- python3 tools/export_predictions.py --adapter rtdetr_pose --tta --tta-seed 0 --tta-flip-prob 0.5 --tta-flip-keypoints --tta-flip-pose-offsets --wrap --output reports/predictions_tta.json

Note: TTA here is a lightweight **prediction-space transform** (post-transform on exported outputs). It does not rerun the model on augmented inputs.

Optional TTT (test-time training, pre-prediction):
- Tent (recommended safe preset + guard rails):
	- python3 tools/export_predictions.py --adapter rtdetr_pose --ttt --ttt-preset safe --ttt-reset sample --wrap --output reports/predictions_ttt_safe.json
- Pose-targeted Tent (task-aware defaults + auxiliary consistency):
	- python3 tools/export_predictions.py --adapter rtdetr_pose --ttt --ttt-sdft-task pose --ttt-aux-pose-weight 0.5 --ttt-aux-temperature 1.0 --ttt-reset sample --wrap --output reports/predictions_ttt_pose.json
- MIM (recommended safe preset + guard rails):
	- python3 tools/export_predictions.py --adapter rtdetr_pose --ttt --ttt-preset mim_safe --ttt-reset sample --wrap --output reports/predictions_ttt_mim_safe.json
- Bounded adaptation-cost run (stream + batch/chunk knobs):
  - python3 tools/export_predictions.py --adapter rtdetr_pose --ttt --ttt-preset safe --ttt-reset stream --ttt-batch-size 4 --ttt-max-batches 8 --wrap --output reports/predictions_ttt_stream_b4_k8.json

Notes:
- TTT requires an adapter that supports `get_model()` + `build_loader()` and requires torch.
- TTT updates model parameters in-memory before calling `adapter.predict(records)`.
- `--ttt-batch-size` controls images per adaptation step; `--ttt-max-batches` caps adaptation batches for predictable runtime.
- Recommended comparison protocol and more examples: `docs/ttt_protocol.md`.

## Export predictions for evaluation

If you run inference externally (PyTorch/TensorRT/ONNX), export to the YOLOZU predictions schema.
Then validate and evaluate in this repo.

- python3 tools/validate_predictions.py reports/predictions.json
- python3 tools/eval_coco.py --dataset data/coco128 --predictions reports/predictions.json --bbox-format cxcywh_norm --max-images 50

## Scenario suite (local evaluation)

- `yolozu test configs/examples/test_setting.yaml --adapter precomputed --predictions reports/predictions.json --max-images 50`
- (source checkout) `python3 tools/run_scenarios.py --adapter precomputed --predictions reports/predictions.json --max-images 50`

## Notes
- When using GPU, install CUDA-enabled PyTorch and use --device cuda:0.
- Keep the predictions schema consistent with the adapter output: image path + detections list.

## YOLO26n smoke (RT-DETR reference trainer)

This repo includes a tiny “it runs end-to-end” smoke command that:
- fetches `data/coco128` if missing
- runs a few steps of `rtdetr_pose/tools/train_minimal.py`
- exports `model.onnx`
- exports wrapped predictions JSON
- runs `tools/eval_suite.py --dry-run` to validate the full I/O chain

```bash
python3 tools/run_yolo26n_smoke_rtdetr_pose.py
```

Multi-bucket variant (n/s/m/l/x) + bucket configs:
- `python3 tools/run_yolo26_smoke_rtdetr_pose.py --buckets n,s,m,l,x`
- `docs/yolo26_rtdetr_pose_recipes.md`
