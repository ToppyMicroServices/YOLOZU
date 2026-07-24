# Training recipe v1 (COCO detect)

This is a **reference recipe** for training a detector in an external environment.

This repo ships an in-repo reference trainer (`rtdetr_pose/`) to wire data/loss/export/metrics,
but it is **not** a production-grade training stack. The stable integration point is the
exported `predictions.json`.

Goal: a reproducible recipe that can generate predictions JSON which improves
over the current baseline on coco128 (quick check) and COCO val2017.

## Scope
- Task: COCO detect
- Input size: 640 (exporter responsibility)
- Evaluation: e2e mAP (no NMS in evaluator)
- BBox format: `cxcywh_norm`

## Determinism checklist
- Fix seeds: Python, NumPy, and framework RNGs.
- Set `PYTHONHASHSEED` and log it.
- Set deterministic flags (e.g., disable nondeterministic kernels, disable benchmark mode).
- Dataloader worker seeds: `base_seed + worker_id`.
- Log: git commit, dataset hash/path, training command line, framework versions, GPU/CPU.
- For PyTorch (example):
  - `torch.use_deterministic_algorithms(True)`
  - `torch.backends.cudnn.benchmark = False`
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cuda.matmul.allow_tf32 = False`
  - `torch.backends.cudnn.allow_tf32 = False`
  - Set `CUBLAS_WORKSPACE_CONFIG=:4096:8`

## Data + preprocessing
- Use official COCO, convert to YOLO-format labels:
  - `python3 tools/prepare_coco_yolo.py --coco-root /path/to/coco --split train2017 --out /path/to/coco-yolo`
  - `python3 tools/prepare_coco_yolo.py --coco-root /path/to/coco --split val2017 --out /path/to/coco-yolo`
- Preserve aspect ratio (letterbox) when resizing to 640.
- Normalize images to [0,1] (or your framework default); log normalization.

## Augmentation (baseline)
Always on:
- Random horizontal flip: `p=0.5`
- HSV / color jitter: small deltas (avoid extreme hue shifts)
- Random affine: translate `0.1`, scale `0.5-1.5`, rotate `+-10 deg`, mild shear

Heavy aug (first 90% of epochs, then **off** for last 10%):
- Mosaic: `p=0.5`
- Mixup: `p=0.1`
- Copy-paste: `p=0.1`

Rationale: reduce train/test gap and stabilize final epochs by disabling heavy aug.

Suggested schedule (by epoch %):
- 0% → 90%: heavy aug **on** (mosaic/mixup/copy-paste enabled)
- 90% → 100%: heavy aug **off**; keep only Always-on aug

## Multi-scale
- Random resize **per batch** in `[512, 768]` with step `32`
- Last 10% epochs: fixed `640`

## Optimizer + schedule (starting point)
- Optimizer: SGD
  - momentum `0.937`
  - weight decay `5e-4`
- Base LR: `0.01` at **global batch 64**
  - Scale linearly with batch size
- Warmup: `3-5` epochs (lr and momentum ramp)
- Schedule: cosine decay to `0.01 * base_lr`
- Gradient clip: `1.0`
- EMA: decay `0.9999`, use EMA weights for eval/export

Implementation notes:
- Exclude weight decay for bias and norm parameters.
- If using gradient accumulation, compute warmup and decay in **steps**, not epochs.

## Training duration
- COCO full: start at `300` epochs, adjust after first eval sweep
- coco128 quick check: `10-30` epochs, batch as large as possible

## Evaluation + export
Export **pre-NMS** detections and validate schema:

```bash
python3 tools/validate_predictions.py /path/to/predictions.json
python3 tools/eval_coco.py --dataset /path/to/coco-yolo --split val2017 \
  --predictions /path/to/predictions.json --bbox-format cxcywh_norm
```

If you export size buckets (yolo26n/s/m/l/x), run the suite:

```bash
python3 tools/eval_suite.py --protocol yolo26 --dataset /path/to/coco-yolo \
  --predictions-glob '/path/to/pred_yolo26*.json' --output reports/eval_suite.json
```

Export checklist:
- Export **pre-NMS** outputs (disable NMS or set IoU very high).
- Keep a consistent cap on detections per image (e.g., top-300 by score).
- Record the score threshold and top-k settings in the run record.

## Quick sanity check (coco128)
Use coco128 to validate the pipeline before full COCO:
- Train briefly (10-30 epochs).
- Export predictions JSON.
- Evaluate via `tools/eval_coco.py` on `data/coco128`.
- Compare mAP to your previous baseline and record delta.

If you want a **repo-local** sanity check that uses real JPEG pixels (instead of the synthetic-image fallback), `rtdetr_pose/tools/train_minimal.py` supports an optional flag:

```bash
python3 rtdetr_pose/tools/train_minimal.py --dataset-root data/coco128 --split train2017 \
  --real-images --image-size 160 --epochs 1 --max-steps 30
```

Note: this is still a minimal training loop; for competitive results, train externally with a full pipeline and export predictions JSON.

Minimal external run checklist (coco128):
- (Optional) normalize class ids:
  - `python3 tools/normalize_predictions.py --input /path/to/predictions.json --output /path/to/predictions_norm.json --classes /path/to/coco-yolo/labels/val2017/classes.json --wrap`
- Validate schema:
  - `python3 tools/validate_predictions.py /path/to/predictions_norm.json`
- Evaluate mAP:
  - `python3 tools/eval_coco.py --dataset /path/to/coco-yolo --split val2017 --predictions /path/to/predictions_norm.json --bbox-format cxcywh_norm`

Target: $\Delta$ mAP50-95 $> 0$ vs the current baseline on coco128 (even a small gain is acceptable).

## Run record (required)
Keep a short run note in your training env:
- Dataset path + hash
- Command line + config file
- Augment schedule + multiscale range
- Optimizer, LR, warmup, EMA
- Seed + determinism flags
- Framework versions + hardware

Suggested run record template:

```text
run_id: yolo26_recipe_v1_<date>
code_commit: <git sha>
dataset: /path/to/coco-yolo (hash=...)
train_cmd: ...
epochs: 300 (final 10% heavy aug off, fixed 640)
batch: 64 (global) | accum: ...
opt: SGD lr=0.01 wd=5e-4 momentum=0.937 warmup=5
ema: decay=0.9999 (used for eval/export)
multiscale: [512,768] step 32
aug: mosaic=0.5 mixup=0.1 copy-paste=0.1 flip=0.5 hsv=on
export: pre-nms topk=300 score_thresh=0.001
metrics: coco128 mAP50-95=... (baseline=..., delta=...)
predictions_json: /path/to/predictions.json
hardware: <gpu> | framework: <version>
```

## Historical Coco128 GPU run (2026-02-02)

This section preserves a historical run recorded in source commit
`72f0862f2487c7a23267820cc2dfc4818e46118b`. The recorded command named
`reports/rtdetr_pose_minimal_config.json`, but that config was not committed.
The command is therefore incomplete and must not be presented as a
reproducible current recipe.

Historical command record (not a current copy-paste recipe):

```bash
python3 rtdetr_pose/tools/train_minimal.py \
  --config reports/rtdetr_pose_minimal_config.json \
  --dataset-root data/coco128 --split train2017 --real-images \
  --image-size 320 --batch-size 8 --epochs 20 --max-steps 0 \
  --lr 1e-4 --device cuda --use-matcher \
  --checkpoint-out reports/rtdetr_pose_ckpt_coco128_gpu_matcher.pt

python3 tools/export_predictions.py \
  --adapter rtdetr_pose --dataset data/coco128 --split train2017 \
  --device cuda --config reports/rtdetr_pose_minimal_config.json \
  --image-size 320 --score-threshold 0.001 --max-detections 10 \
  --checkpoint reports/rtdetr_pose_ckpt_coco128_gpu_matcher.pt \
  --output reports/pred_coco128_rtdetr_pose_gpu_matcher.json

python3 tools/eval_coco.py \
  --dataset data/coco128 --split train2017 \
  --predictions reports/pred_coco128_rtdetr_pose_gpu_matcher.json \
  --bbox-format cxcywh_norm \
  --output reports/coco_eval_rtdetr_pose_gpu_matcher.json
```

Results:

- Recorded comparison baseline mAP50-95: 1.829e-06. The named baseline file
  `reports/coco_eval_rtdetr_pose.json` is not tracked, so this value and the
  recorded delta cannot be independently recomputed from the repository.
- Historical mAP50-95
  (`reports/coco_eval_rtdetr_pose_gpu_matcher.json`): 0.000622
- Recorded historical delta: +0.000620
- Predictions JSON: reports/pred_coco128_rtdetr_pose_gpu_matcher.json
- Checkpoint: reports/rtdetr_pose_ckpt_coco128_gpu_matcher.pt
- Hardware: NVIDIA RTX A4500

The checkpoint is not fully compatible with the current
`rtdetr_pose/configs/base.json`: only 20 of 308 current model-state tensors
match by name and shape, and current model-parameter numel coverage is about
3.61%. It now fails closed in public inference/export/parity paths. Treat the
checkpoint, predictions, and evaluation as a historical provenance set, not
current full-checkpoint evidence. Pinned hashes, source commit, missing-config
status, and the compatibility summary are recorded in
[`../reports/rtdetr_pose_coco128_gpu_matcher_historical.json`](../reports/rtdetr_pose_coco128_gpu_matcher_historical.json).
Use a checkpoint produced from a checked-in current config for new runs.
