# TTT (Tent / MIM) protocol: safe + reproducible comparisons

This repo supports **test-time training (TTT)** for the `rtdetr_pose` adapter via `tools/export_predictions.py`
(or the unified wrapper `tools/yolozu.py export --backend torch ...`).

TTT updates model weights **in-memory** using unlabeled test data before (or per-sample during) inference.

TTT is **OFF by default** and only enabled when you pass `--ttt` (opt-in).

## When COCO is a good choice

For **detection metrics** that other people recognize, COCO (`val2017`) is a good baseline.

However, to demonstrate **TTT/Tent improvements**, you typically need a **domain shift** (e.g., corruptions, style change,
different camera/weather). On *clean COCO*, TTT can be neutral or even harmful.

Recommended:
- Baseline: clean COCO `val2017`
- Target domain: either (a) a shifted dataset (BDD100K/Cityscapes-style domain), or (b) a **corrupted copy** of COCO images

## Presets (recommended starting points)

Both CLIs expose `--ttt-preset`:
- `safe`: Tent + BN-affine only (`update_filter=norm_only`)
- `adapter_only`: Tent + adapter/head only
- `mim_safe`: MIM + adapter/head only
- `cotta_safe`: CoTTA with conservative update/restore guard rails
- `eata_safe`: EATA selective adaptation with anti-forgetting defaults
- `sar_safe`: SAR LoRA-first sharpness-aware adaptation defaults
- `pose_safe`: Tent defaults tuned for pose-heavy adaptation
- `keypoints_safe`: Tent defaults tuned for keypoint-heavy adaptation
- `depth_safe`: Tent defaults tuned for depth-heavy adaptation
- `seg_safe`: Tent defaults tuned for segmentation-heavy adaptation
- `pose_mim`: MIM defaults tuned for pose-heavy adaptation

Presets:
- override core knobs (`method/steps/lr/update_filter/max_batches`)
- fill conservative safety guards unless you explicitly set them (`--ttt-max-...`)

Guard defaults (when unset):
- `safe`: `max_grad_norm=1.0`, `max_update_norm=1.0`, `max_total_update_norm=1.0`, `max_loss_ratio=3.0`
- `adapter_only` / `mim_safe`: `max_grad_norm=5.0`, `max_update_norm=5.0`, `max_total_update_norm=5.0`, `max_loss_ratio=3.0`

If you pass `--ttt` without `--ttt-preset` and leave the core knobs at defaults, the CLI auto-applies a conservative preset:
- Tent → `safe`
- MIM (`--ttt-method mim`) → `mim_safe`
- CoTTA/EATA/SAR methods auto-map to `cotta_safe`/`eata_safe`/`sar_safe`
- `--ttt-sdft-task pose|keypoints|depth|seg|full` steers default selection to task-specific presets where available

Task-aware auxiliary knobs:
- `--ttt-aux-pose-weight`
- `--ttt-aux-keypoints-weight`
- `--ttt-aux-depth-weight`
- `--ttt-aux-seg-weight`
- `--ttt-aux-temperature`

## Reset policy (stream vs sample)

`--ttt-reset stream` (default):
- adapt once using up to `--ttt-max-batches` batches
- keep adapted weights for all subsequent images

`--ttt-reset sample`:
- restore base state per image (selected parameters + normalization running stats)
- run TTT on that single image (or its batch) then predict
- slower, but comparisons are cleaner (no cross-image state)

For ablations/plots, start with `--ttt-reset sample`.

## Batch/chunk knobs (`--ttt-batch-size`, `--ttt-max-batches`)

- `--ttt-batch-size N`: number of images per adaptation step.
- `--ttt-max-batches K`: hard cap on how many adaptation batches are consumed.

Practical guidance:
- Start with `--ttt-batch-size 1 --ttt-max-batches 1` for safest smoke checks.
- Increase `--ttt-batch-size` first when GPU memory allows; this often improves throughput.
- Increase `--ttt-max-batches` only when you explicitly want stronger adaptation (and higher latency).

Example (stream reset, bounded adaptation cost):

```bash
python3 tools/yolozu.py export \
  --backend torch \
  --dataset reports/coco128_50 \
  --split train2017 \
  --checkpoint /path/to.ckpt \
  --device cuda \
  --ttt \
  --ttt-preset safe \
  --ttt-reset stream \
  --ttt-batch-size 4 \
  --ttt-max-batches 8 \
  --output reports/pred_ttt_stream_b4_k8.json
```

## Fixed eval subset (for plots)

To make comparisons fair and reproducible, evaluate on the **same image subset** every time.

`tools/make_subset_dataset.py` creates a tiny YOLO dataset root containing only a deterministic subset
(symlinks by default, or `--copy`):

```bash
python3 tools/make_subset_dataset.py \
  --dataset data/coco128 \
  --split train2017 \
  --n 50 \
  --seed 0 \
  --out reports/coco128_50
```

Outputs:
- `reports/coco128_50/` (YOLO dataset root)
- `reports/coco128_50/subset.json` (includes `images_sha256`)
- `reports/coco128_50/subset_images.txt` (frozen image list)

## Example: baseline vs TTT (coco128 smoke)

Baseline (no TTT):

```bash
python3 tools/yolozu.py export \
  --backend torch \
  --dataset reports/coco128_50 \
  --split train2017 \
  --checkpoint /path/to.ckpt \
  --device cuda \
  --max-images 50 \
  --output reports/pred_baseline.json
```

TTT (safe preset, per-sample reset, with a log):

```bash
python3 tools/yolozu.py export \
  --backend torch \
  --dataset reports/coco128_50 \
  --split train2017 \
  --checkpoint /path/to.ckpt \
  --device cuda \
  --max-images 50 \
  --ttt \
  --ttt-preset safe \
  --ttt-reset sample \
  --ttt-log-out reports/ttt_log_safe.json \
  --output reports/pred_ttt_safe.json
```

Then score with COCO mAP (requires `pycocotools`):

```bash
python3 tools/eval_coco.py \
  --dataset reports/coco128_50 \
  --predictions reports/pred_baseline.json \
  --bbox-format cxcywh_norm \
  --max-images 50

python3 tools/eval_coco.py \
  --dataset reports/coco128_50 \
  --predictions reports/pred_ttt_safe.json \
  --bbox-format cxcywh_norm \
  --max-images 50
```

## SDFT / prediction distillation (quick)

If you already have teacher+student predictions on the same dataset subset:

```bash
python3 tools/distill_predictions.py \
  --student reports/pred_student.json \
  --teacher reports/pred_teacher.json \
  --dataset reports/coco128_50 \
  --split train2017 \
  --output reports/pred_distilled.json \
  --output-report reports/distill_report.json \
  --add-missing
```

For credible plots:
- use the same subset (`subset.json` hash pinned)
- run multiple seeds (for TTT stochasticity and training variance)
- report mean±std and runtime cost (TTT adds latency)
