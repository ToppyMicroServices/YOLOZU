# TTT (Tent / MIM) protocol: safe + reproducible comparisons

This repo supports **test-time training (TTT)** for the `rtdetr_pose` adapter via `tools/export_predictions.py`
(or the canonical CLI `yolozu export --backend torch ...` / `python3 -m yolozu export --backend torch ...`).

TTT updates model weights **in-memory** using unlabeled test data before (or per-sample during) inference.

TTT is an **opt-in research lane**, not a stable production default. It starts from a baseline evaluated artifact and writes separate adapted predictions, logs, and before/after reports.

TTT is **OFF by default** and only enabled when you pass `--ttt` (opt-in).
Default validation, evaluation, demo, and export commands do not enable TTT implicitly.

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
python3 -m yolozu demo ttt
```

This writes `ttt_improvement_report.json` containing `metrics.no_ttt`, `metrics.with_ttt`, and `metrics.delta`,
plus two overlay PNGs (`overlay_no_ttt.png`, `overlay_ttt.png`) rendered from predictions in the predictions interface contract.

Example stdout (CPU, deterministic seeds; values are intentionally tiny, the point is the reproducible delta):
- `map50 0.00326797 → 0.00392157`
- `map50_95 0.000326797 → 0.000392157`

For the manual, we convert these fixed artifacts into three beginner-facing PNGs:
- `docs/assets/ttt_method_results_summary.png`
- `docs/assets/ttt_compare_pipeline.png`
- `docs/assets/ttt_probe_example_panel.png`

Use them in this order:
1. `ttt_method_results_summary.png` for the actual effect
2. `ttt_compare_pipeline.png` for the processing steps
3. `ttt_probe_example_panel.png` for the per-image shifted probe view: one actual tile per image, with the highest-score baseline box and highest-score TTT box overlaid on the same input

To regenerate those figures from the fixed result source JSON:

```bash
python3 tools/render_ttt_manual_figures.py
```

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

The recommended operator workflow is now the boilerplate compare wrapper:

```bash
bash scripts/ttt_compare.sh \
  --boilerplate tent \
  --dataset data/smoke \
  --split val \
  --checkpoint /path/to.ckpt \
  --run-dir reports/ttt_compare/tent \
  --device cuda
```

This produces a baseline export, an adapted export, and a before-after report without requiring you to hand-write the long `--ttt-*` flag set.

Reference:
- [TTT before-after compare boilerplates](ttt_compare_boilerplates.md)

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

## How to read the compare outputs first

Before diving into the individual methods, it helps to read the generated
artifacts in a fixed order:

1. `plan.json`
2. `<method>_before_after_compare.md`
3. `<method>_ttt_log.json`
4. optional follow-up method-specific tools such as `eval_cotta_drift.py`

This order is intentional. The compare markdown is the beginner-friendly entry
point. It answers:

- did adaptation actually run?
- how many update steps were applied?
- did exported predictions change?
- what warning or guard rail fired?

Do not interpret the smoke compare table as a leaderboard. It is a workflow
validation summary.

- `real compare status=completed` means the pipeline ran end-to-end
- `steps run` tells you whether adaptation actually updated parameters
- `mean final loss` is the adaptation objective, not a universal quality score
- warning notes should be read from the caption or prose, not from a very wide table column

Example:
- EATA can report `steps_run=0` with `eata_empty_selected_set`
- that means EATA intentionally skipped adaptation on that tiny subset
- it does not mean the implementation is broken

## Shared step-by-step workflow

For every method, use the same operational recipe:

1. fix the dataset subset or deterministic domain-shift target
2. run `scripts/ttt_compare.sh` with one boilerplate
3. open `<method>_before_after_compare.md`
4. inspect `<method>_ttt_log.json` only if you need detailed reasoning
5. use the method-specific follow-up tool only when the compact compare report is not enough

This matters because many users jump directly into the raw TTT log and get
lost. In practice, the compare markdown should be treated as the primary
operator report.

### Tent

Meaning of the name:
- Tent is short for **fully Test-time adaptation by ENTropy minimization**
- the name already tells you the core idea: adapt at test time, and use entropy as the signal

Principle:
- Tent minimizes prediction entropy online
- in plain language, it pushes the model toward sharper predictions on shifted inputs
- in YOLOZU, the default safe rollout keeps updates constrained to norm-affine parameters

What entropy means here:
- if the model spreads probability mass across many classes, entropy is high
- if the model becomes confident and concentrates probability on one class, entropy is low
- for one prediction vector `p`, the standard entropy term is:

$$
H(p) = - \sum_c p_c \log p_c
$$

- Tent reduces the average entropy over selected test-time predictions:

$$
\min_{\theta_{\text{adapt}}} \; \mathbb{E}_{x \sim \text{test stream}}[H(p_\theta(y \mid x))]
$$

- in YOLOZU, `theta_adapt` is intentionally small in the safe presets, typically norm-affine or adapter/head subsets

Why this can work:
- under mild domain shift, the model often still looks at roughly the right object
- the failure is that the output distribution becomes softer or less decisive
- entropy minimization nudges the model toward a cleaner local decision boundary without asking for labels

What it is not:
- it is not supervised fine-tuning
- it does not tell the model which class is correct
- it assumes ``become more confident'' is a useful direction on the current shifted sample

When to use it:
- first ablation
- first production-style smoke compare
- safest method to explain to someone new to TTT

```bash
bash scripts/ttt_compare.sh \
  --boilerplate tent \
  --dataset data/smoke \
  --split val \
  --checkpoint /path/to.ckpt \
  --run-dir reports/ttt_compare/tent \
  --device cuda
```

Read it like this:
- open `tent_before_after_compare.md`
- check whether `steps_run > 0`
- if `changed_images=0`, adaptation still ran; it just did not change exported predictions on that subset
- open `tent_ttt_log.json` if you want the loss and runtime detail

Concrete repo result:
- on the fixed 10-image shifted probe, Tent improves `map50` from `0.00326797` to `0.00392157`
- on the same probe, it improves `map50_95` from `0.000326797` to `0.000392157`

Pros:
- simplest method
- lowest extra compute
- easiest to debug

Cons:
- weakest self-supervised signal
- can sharpen wrong predictions under severe shift
- usually less expressive than reconstruction-based methods

### MIM

Meaning of the name:
- MIM stands for **Masked Image Modeling**
- in plain language, we deliberately hide part of the visual signal and ask the model to reconstruct or predict what is missing

Principle:
- MIM uses masked reconstruction and optional entropy terms
- instead of only asking the model to be more confident, it asks the model to reconstruct hidden structure from partially masked features
- this creates a stronger self-supervised signal than Tent
- the practical references to know are denoising autoencoders (Vincent et al., ICML 2008), Context Encoders (Pathak et al., CVPR 2016), MAE (He et al., CVPR 2022), SimMIM (Xie et al., CVPR 2022), Test-Time Training with Masked Autoencoders (Gandelsman et al., 2022), and TTT-MIM (Mansour et al., ECCV 2024)

Minimal objective view:
- a mask `M` hides part of the representation or image-like state
- the model reconstructs the hidden content from the visible part
- a simplified reconstruction loss looks like:

$$
L_{\text{MIM}} = \lVert R_\theta((1-M)\odot x) - M \odot x \rVert
$$

- some practical variants combine this with entropy minimization:

$$
L_{\text{adapt}} = \lambda_{\text{mim}} L_{\text{MIM}} + \lambda_{\text{ent}} H(p_\theta(y \mid x))
$$

- the exact reconstruction target differs across implementations, but the operational idea is the same: force the model to explain the hidden structure instead of only sharpening the class probabilities

Why this can work:
- entropy-only adaptation can be too weak when the shift is geometric or structural
- MIM gives the model a denser self-supervised signal
- that is why it is often more attractive for pose-heavy or geometry-sensitive adaptation than plain Tent

What it is not:
- it is not only MAE pretraining reused verbatim
- it is not a guarantee that every test-time run will improve exported detections
- it usually requires more model-specific wiring than Tent because the masked-reconstruction branch must exist and be stable

Research lineage and IP note:
- the broad idea behind MIM-style adaptation predates current test-time wording: corrupt or mask part of the representation, reconstruct hidden structure, and use that self-supervised loss as a robustness signal
- this section cites representative prior-art waypoints for technical lineage and prior-art positioning
- this is not legal advice and not a non-infringement opinion; deployment-specific patent review belongs to qualified counsel

When to use it:
- pose-heavy or geometry-sensitive adaptation
- cases where confidence-only adaptation is too weak

```bash
bash scripts/ttt_compare.sh \
  --boilerplate mim \
  --dataset data/smoke \
  --split val \
  --checkpoint /path/to.ckpt \
  --run-dir reports/ttt_compare/mim \
  --device cuda
```

For pose-heavy runs, switch the preset to `pose_mim`.
YOLOZU now ships two practical MIM entrypoints:

- `mim`: the repo-backed smoke fixture for verifying that the masked-reconstruction
  path is wired correctly
- `mim_probe`: a fixed real probe that demonstrates an actual before/after metric
  delta on the bundled shifted ten-image probe

```bash
bash scripts/ttt_compare.sh \
  --boilerplate mim \
  --dataset data/smoke \
  --split val \
  --checkpoint /path/to/current-compatible.ckpt \
  --run-dir reports/ttt_compare/mim_smoke_cpu \
  --device cpu \
  --max-images 2 \
  --skip-eval \
  --force
```

Repo-backed smoke snapshot:
- `reports/ttt_compare/mim_smoke_cpu/mim_before_after_compare.md`
- `steps_run=2`
- `mean_final_loss=0.461853`
- `changed_images=0 / 2`

Fixed real-probe snapshot:

```bash
bash scripts/ttt_compare.sh \
  --boilerplate mim_probe \
  --dataset reports/ttt_improvement_probe/domain_shift_dataset \
  --split val \
  --checkpoint reports/ttt_improvement_probe/checkpoint.pt \
  --run-dir reports/ttt_compare/mim_probe_cpu \
  --device cpu \
  --max-images 10
```

- `reports/ttt_compare/mim_probe_cpu/mim_before_after_compare.md`
- `steps_run=10`
- `mean_final_loss=0.0791513`
- `changed_images=10 / 10`
- `map50: 0.00326797 -> 0.00392157`
- `map50_95: 0.000326797 -> 0.000392157`
- metric backend: `simple_map_proxy` when `pycocotools` is unavailable in the current runtime

Read it like this:
- open `mim_before_after_compare.md`
- then inspect `mim_ttt_log.json`
- in MIM, the adaptation loss is usually more meaningful than in Tent because it contains the reconstruction term
- the smoke fixture proves execution; the fixed real probe proves that MIM can also move final detections in a visible, repeatable way
- when `pycocotools` is missing, YOLOZU falls back to `simple_map_proxy` so the before/after compare still yields a deterministic quality delta instead of silently dropping eval

Concrete repo results:
- historical smoke compare: `steps_run=2`, `mean_final_loss=0.461853`,
  `changed_images=0 / 2` (preserved as a source-commit record, not current
  full-checkpoint evidence)
- fixed real probe compare: `steps_run=10`, `mean_final_loss=0.0791513`, `changed_images=10 / 10`
- fixed real probe metrics: `map50 0.00326797 -> 0.00392157`, `map50_95 0.000326797 -> 0.000392157`

Pros:
- stronger adaptation signal than Tent
- useful for geometry-aware adaptation
- a good next step after Tent

Cons:
- more model-specific
- harder to explain and inspect
- loss values are not directly comparable with Tent

### CoTTA

Meaning of the name:
- CoTTA is short for **Continual Test-Time Adaptation**
- the name signals that this method is designed for a stream, not only for one isolated image

Principle:
- CoTTA uses augmentation-averaged predictions, an EMA teacher, and partial restoration
- the goal is to adapt in a stream while reducing long-run drift

Minimal objective view:
- a student model is updated online
- an EMA teacher provides a smoother target
- multiple augmentations reduce the chance that one noisy view dominates the update
- a simplified consistency view is:

$$
L_{\text{CoTTA}} = \sum_{a \in \mathcal{A}} d\!\left(p_{\theta}(y \mid a(x)),\; p_{\theta^{\text{EMA}}}(y \mid x)\right)
$$

- partial restoration then pulls part of the student back toward older weights so the stream does not drift too far

Why this can work:
- in a long stream, a pure online update can slowly walk away from a good solution
- CoTTA tries to smooth that process with three safeguards:
  - augmentation averaging
  - EMA teacher targets
  - stochastic restoration

What it is not:
- it is not the simplest first TTT baseline
- it is not stateless; the order of images matters more than in per-sample adaptation
- it usually makes the most sense when stream behavior itself is part of the problem

When to use it:
- continual or streaming adaptation
- when you care about long-run behavior more than one-image adaptation

```bash
bash scripts/ttt_compare.sh \
  --boilerplate cotta \
  --dataset data/smoke \
  --split val \
  --checkpoint /path/to.ckpt \
  --run-dir reports/ttt_compare/cotta \
  --device cuda
```

Method-specific evidence:

```bash
python3 tools/eval_cotta_drift.py \
  --baseline reports/ttt_compare/tent/tent_predictions.json \
  --cotta reports/ttt_compare/cotta/cotta_predictions.json \
  --output-json reports/ttt_compare/cotta/cotta_drift.json \
  --output-md reports/ttt_compare/cotta/cotta_drift.md
```

Read it like this:
- open `cotta_before_after_compare.md`
- inspect `cotta_ttt_log.json` for stream update behavior
- open `cotta_drift.md` when you want to reason about long-run drift

Concrete repo result:
- on the same fixed 10-image shifted probe as Tent, CoTTA reaches `map50=0.00392157` and `map50_95=0.000392157`

Pros:
- well suited to continual adaptation
- EMA teacher smooths noisy updates
- restoration helps prevent collapse

Cons:
- more moving parts than Tent or MIM
- more stateful and harder to debug
- higher runtime cost

### EATA

Meaning of the name:
- EATA is commonly read as **Efficient Test-Time Adaptation**
- the practical emphasis is not just speed, but selective adaptation: only adapt when the evidence looks trustworthy

Principle:
- EATA filters the batch first, then adapts only on trusted samples
- it also uses anchor regularization to reduce forgetting

Minimal objective view:
- first, select samples whose entropy or reliability statistics pass a gate
- then minimize entropy only on that selected set
- add a regularizer that keeps the updated model close to an anchor state

$$
L_{\text{EATA}} = \sum_{x \in S_{\text{selected}}} H(p_{\theta}(y \mid x)) + \lambda_{\text{anchor}} R(\theta, \theta_0)
$$

- the exact selection rule varies, but the operational idea is stable: no trusted set, no update

Why this can work:
- one bad batch can make online adaptation worse instead of better
- EATA reduces that risk by refusing to adapt when the current batch does not look informative enough
- that is why it often appears conservative on tiny or noisy subsets

What it is not:
- it is not broken when it skips updates
- it is not trying to adapt on every sample
- it usually trades aggressiveness for safety

When to use it:
- when low-quality or misleading test samples are common
- when you prefer the method to skip adaptation rather than force updates

```bash
bash scripts/ttt_compare.sh \
  --boilerplate eata \
  --dataset data/smoke \
  --split val \
  --checkpoint /path/to.ckpt \
  --run-dir reports/ttt_compare/eata \
  --device cuda
```

Method-specific evidence:

```bash
python3 tools/benchmark_eata_stability.py \
  --baseline reports/ttt_compare/tent/tent_predictions.json \
  --eata reports/ttt_compare/eata/eata_predictions.json \
  --output-json reports/ttt_compare/eata/eata_benchmark.json \
  --output-md reports/ttt_compare/eata/eata_benchmark.md
```

Read it like this:
- open `eata_before_after_compare.md`
- if `steps_run=0`, check the warning field before assuming there was an error
- on small smoke subsets, `eata_empty_selected_set` is expected conservative behavior

Concrete repo result:
- smoke fixture: `steps_run=0` with `eata_empty_selected_set`
- real shifted probe: same small metric gain as Tent / CoTTA / SAR

Pros:
- safest method when adaptation mistakes are expensive
- explicit skip behavior is operationally easy to explain
- regularization helps protect important weights

Cons:
- may appear inactive on tiny subsets
- more thresholds and diagnostics to understand
- can be harder for new users to trust at first glance

### SAR

Meaning of the name:
- SAR refers to a **Sharpness-Aware** adaptation rule
- the key idea is to prefer updates that remain good even after a small perturbation

Principle:
- SAR performs sharpness-aware entropy minimization
- instead of trusting only the immediate entropy gradient, it prefers updates that remain stable after a small perturbation

Minimal objective view:
- ordinary entropy minimization asks for a parameter update that lowers the entropy now
- SAR asks for an update that still looks good in a small neighborhood around the new point

$$
\min_{\theta} \max_{\lVert \epsilon \rVert \le \rho} L_{\text{entropy}}(\theta + \epsilon)
$$

- the inner perturbation approximates ``what if this update lands in a sharp, fragile region?''
- the outer step then prefers flatter, more stable regions

Why this can work:
- some domain shifts are noisy enough that the first entropy gradient is too eager
- SAR slows the optimizer down and asks whether the local improvement is robust, not just immediate
- that often makes it easier to justify in unstable mixed-shift scenarios

What it is not:
- it is not the cheapest method
- it is not necessary when Tent already behaves well
- it is mainly about update geometry, not a stronger self-supervised signal like MIM

When to use it:
- noisy shifts
- cases where plain Tent is too eager but CoTTA feels too heavy

```bash
bash scripts/ttt_compare.sh \
  --boilerplate sar \
  --dataset data/smoke \
  --split val \
  --checkpoint /path/to.ckpt \
  --run-dir reports/ttt_compare/sar \
  --device cuda
```

Method-specific evidence:

```bash
python3 tools/benchmark_sar_robustness.py \
  --cotta reports/ttt_compare/cotta/cotta_predictions.json \
  --eata reports/ttt_compare/eata/eata_predictions.json \
  --sar reports/ttt_compare/sar/sar_predictions.json \
  --output-json reports/ttt_compare/sar/sar_robustness.json \
  --output-md reports/ttt_compare/sar/sar_robustness.md
```

Read it like this:
- open `sar_before_after_compare.md`
- compare `sar_ttt_log.json` runtime against Tent to understand the cost increase
- use `sar_robustness.md` when comparing SAR against CoTTA and EATA

Concrete repo result:
- on the fixed 10-image shifted probe, SAR reaches `map50=0.00392157` and `map50_95=0.000392157`, matching Tent / CoTTA / EATA on that probe

Pros:
- more conservative update geometry than Tent
- often useful for unstable or noisy shifts
- good middle ground between Tent and CoTTA

Cons:
- slower than Tent
- harder for beginners to understand
- can be unnecessary on mild shifts

## Choosing a method quickly

| Method | Best first use | Main trade-off |
| --- | --- | --- |
| Tent | first ablation, safest first comparison | weakest adaptation signal |
| MIM | geometry- or pose-sensitive shift | more model-specific and harder to interpret |
| CoTTA | streaming / continual adaptation | stateful and more expensive |
| EATA | conservative adaptation with explicit skipping | may do nothing on small subsets |
| SAR | noisy shift with stability concerns | slower than Tent |

### Task-aware examples

The method is still usually Tent first; what changes is the preset and auxiliary weighting for the task emphasis:

```bash
# pose-heavy Tent
python3 -m yolozu export \
  --backend torch --dataset reports/smoke_50 --split val \
  --checkpoint /path/to.ckpt --device cuda \
  --ttt --ttt-method tent --ttt-preset pose_safe \
  --ttt-sdft-task pose --ttt-aux-pose-weight 0.5 \
  --ttt-log-out reports/ttt_pose_safe.json \
  --output reports/pred_pose_safe.json

# keypoints-heavy Tent
python3 -m yolozu export \
  --backend torch --dataset reports/smoke_50 --split val \
  --checkpoint /path/to.ckpt --device cuda \
  --ttt --ttt-method tent --ttt-preset keypoints_safe \
  --ttt-sdft-task keypoints --ttt-aux-keypoints-weight 0.5 \
  --ttt-log-out reports/ttt_keypoints_safe.json \
  --output reports/pred_keypoints_safe.json

# depth-heavy Tent
python3 -m yolozu export \
  --backend torch --dataset reports/smoke_50 --split val \
  --checkpoint /path/to.ckpt --device cuda \
  --ttt --ttt-method tent --ttt-preset depth_safe \
  --ttt-sdft-task depth --ttt-aux-depth-weight 0.5 \
  --ttt-log-out reports/ttt_depth_safe.json \
  --output reports/pred_depth_safe.json

# segmentation-heavy Tent
python3 -m yolozu export \
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
python3 -m yolozu export \
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

Recommended short entrypoint:

```bash
bash scripts/ttt_compare.sh \
  --boilerplate tent \
  --dataset reports/coco128_50 \
  --split train2017 \
  --checkpoint /path/to.ckpt \
  --run-dir reports/ttt_compare/coco128_tent \
  --device cuda
```

If you need the raw export commands for debugging or custom orchestration, the wrapper writes them into `plan.json`.

Baseline (no TTT):

```bash
python3 -m yolozu export \
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
python3 -m yolozu export \
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
