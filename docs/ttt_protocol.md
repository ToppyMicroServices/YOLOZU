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

## Deterministic shift-target recipe (recommended)

Use the built-in deterministic recipe generator to create a reproducible corrupted target split:

```bash
python3 scripts/prepare_ttt_domain_shift_target.py \
  --dataset-root data/smoke \
  --split val \
  --out reports/domain_shift/smoke_gaussian_blur_s2 \
  --corruption gaussian_blur \
  --severity 2 \
  --seed 2026 \
  --force
```

Generated artifacts:
- `reports/domain_shift/smoke_gaussian_blur_s2/domain_shift_recipe.json`
- `reports/domain_shift/smoke_gaussian_blur_s2/images/<split>/*`
- `reports/domain_shift/smoke_gaussian_blur_s2/labels/<split>/*` (copied)

To bind this target into inference outputs, pass the recipe during export:

```bash
python3 tools/export_predictions.py \
  --adapter dummy \
  --dataset reports/domain_shift/smoke_gaussian_blur_s2 \
  --split val \
  --wrap \
  --domain-shift-recipe reports/domain_shift/smoke_gaussian_blur_s2/domain_shift_recipe.json \
  --output reports/pred_shift_target.json
```

When `--wrap` is enabled, the output contains:
- `meta.export_settings.domain_shift_target`
- `meta.export_settings.domain_shift_recipe` (`path`, `sha256`)

This keeps TTT evidence explicit and deterministic across runs/CI.

## TTT improvement micro-demo (show a delta)

If you want a self-contained micro-demo that shows a small metric delta under a fixed domain shift, run:

```bash
python3 -m pip install -U 'yolozu[demo]'
yolozu demo ttt

# repo checkout equivalent:
python3 tools/yolozu.py demo ttt
```

This writes `ttt_improvement_report.json` containing `metrics.no_ttt`, `metrics.with_ttt`, and `metrics.delta`,
plus two overlay PNGs (`overlay_no_ttt.png`, `overlay_ttt.png`) rendered from predictions in the predictions interface contract.

Example stdout (CPU, deterministic seeds; values are intentionally tiny, the point is the reproducible delta):
- `map50 0.00326797 → 0.00392157`
- `map50_95 0.000326797 → 0.000392157`

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

## Implemented algorithms and concrete repo examples

The currently implemented parameter-updating methods are:
- `tent`
- `mim`
- `cotta`
- `eata`
- `sar`

The practical pattern is always the same:
1. freeze a deterministic dataset subset,
2. export wrapped predictions with one method,
3. export wrapped predictions with another method,
4. compare the resulting logs and method-specific reports.

### Tent

Use Tent when you want the smallest and safest step away from baseline inference.

```bash
python3 tools/yolozu.py export \
  --backend torch \
  --dataset reports/smoke_50 \
  --split val \
  --checkpoint /path/to.ckpt \
  --device cuda \
  --ttt \
  --ttt-method tent \
  --ttt-preset safe \
  --ttt-reset sample \
  --ttt-log-out reports/ttt_tent.json \
  --output reports/pred_tent.json
```

### MIM

Use MIM when the geometry-aware masked reconstruction signal is available and you want a stronger adaptation objective than plain entropy minimization.

```bash
python3 tools/yolozu.py export \
  --backend torch \
  --dataset reports/smoke_50 \
  --split val \
  --checkpoint /path/to.ckpt \
  --device cuda \
  --ttt \
  --ttt-method mim \
  --ttt-preset mim_safe \
  --ttt-reset sample \
  --ttt-log-out reports/ttt_mim.json \
  --output reports/pred_mim.json
```

For pose-heavy runs, switch the preset to `pose_mim`.

### CoTTA

Use CoTTA when stream-mode adaptation matters and you want EMA-teacher smoothing with restoration against long-run drift.

```bash
python3 tools/yolozu.py export \
  --backend torch \
  --dataset reports/smoke_50 \
  --split val \
  --checkpoint /path/to.ckpt \
  --device cuda \
  --ttt \
  --ttt-method cotta \
  --ttt-preset cotta_safe \
  --ttt-reset stream \
  --ttt-cotta-augmentations identity \
  --ttt-cotta-augmentations hflip \
  --ttt-log-out reports/ttt_cotta.json \
  --output reports/pred_cotta.json
```

Method-specific evidence:

```bash
python3 tools/eval_cotta_drift.py \
  --baseline reports/pred_tent.json \
  --cotta reports/pred_cotta.json \
  --output-json reports/cotta_drift.json \
  --output-md reports/cotta_drift.md
```

### EATA

Use EATA when you want selective adaptation and anchor regularization so low-quality batches can be skipped safely.

```bash
python3 tools/yolozu.py export \
  --backend torch \
  --dataset reports/smoke_50 \
  --split val \
  --checkpoint /path/to.ckpt \
  --device cuda \
  --ttt \
  --ttt-method eata \
  --ttt-preset eata_safe \
  --ttt-reset sample \
  --ttt-log-out reports/ttt_eata.json \
  --output reports/pred_eata.json
```

Method-specific evidence:

```bash
python3 tools/benchmark_eata_stability.py \
  --baseline reports/pred_tent.json \
  --eata reports/pred_eata.json \
  --output-json reports/eata_benchmark.json \
  --output-md reports/eata_benchmark.md
```

### SAR

Use SAR when you want sharpness-aware entropy minimization and can afford the extra adaptation cost.

```bash
python3 tools/yolozu.py export \
  --backend torch \
  --dataset reports/smoke_50 \
  --split val \
  --checkpoint /path/to.ckpt \
  --device cuda \
  --ttt \
  --ttt-method sar \
  --ttt-preset sar_safe \
  --ttt-reset sample \
  --ttt-log-out reports/ttt_sar.json \
  --output reports/pred_sar.json
```

Method-specific evidence:

```bash
python3 tools/benchmark_sar_robustness.py \
  --cotta reports/pred_cotta.json \
  --eata reports/pred_eata.json \
  --sar reports/pred_sar.json \
  --output-json reports/sar_robustness.json \
  --output-md reports/sar_robustness.md
```

### Task-aware examples

The method is still usually Tent first; what changes is the preset and auxiliary weighting for the task emphasis:

```bash
# pose-heavy Tent
python3 tools/yolozu.py export \
  --backend torch --dataset reports/smoke_50 --split val \
  --checkpoint /path/to.ckpt --device cuda \
  --ttt --ttt-method tent --ttt-preset pose_safe \
  --ttt-sdft-task pose --ttt-aux-pose-weight 0.5 \
  --ttt-log-out reports/ttt_pose_safe.json \
  --output reports/pred_pose_safe.json

# keypoints-heavy Tent
python3 tools/yolozu.py export \
  --backend torch --dataset reports/smoke_50 --split val \
  --checkpoint /path/to.ckpt --device cuda \
  --ttt --ttt-method tent --ttt-preset keypoints_safe \
  --ttt-sdft-task keypoints --ttt-aux-keypoints-weight 0.5 \
  --ttt-log-out reports/ttt_keypoints_safe.json \
  --output reports/pred_keypoints_safe.json

# depth-heavy Tent
python3 tools/yolozu.py export \
  --backend torch --dataset reports/smoke_50 --split val \
  --checkpoint /path/to.ckpt --device cuda \
  --ttt --ttt-method tent --ttt-preset depth_safe \
  --ttt-sdft-task depth --ttt-aux-depth-weight 0.5 \
  --ttt-log-out reports/ttt_depth_safe.json \
  --output reports/pred_depth_safe.json

# segmentation-heavy Tent
python3 tools/yolozu.py export \
  --backend torch --dataset reports/smoke_50 --split val \
  --checkpoint /path/to.ckpt --device cuda \
  --ttt --ttt-method tent --ttt-preset seg_safe \
  --ttt-sdft-task seg --ttt-aux-seg-weight 0.5 \
  --ttt-log-out reports/ttt_seg_safe.json \
  --output reports/pred_seg_safe.json
```

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
