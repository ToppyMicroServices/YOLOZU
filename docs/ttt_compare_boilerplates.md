# TTT before-after compare boilerplates

The recommended operator workflow for TTT/CTTA comparisons is:

1. freeze the dataset subset,
2. run one baseline export without TTT,
3. run one adapted export with the selected method boilerplate,
4. inspect the generated before-after report.

Use the short shell entrypoint instead of hand-writing long `--ttt-*` flag sequences:

```bash
bash scripts/ttt_compare.sh \
  --boilerplate tent \
  --dataset data/smoke \
  --split val \
  --checkpoint /path/to.ckpt \
  --run-dir reports/ttt_compare/tent \
  --device cuda
```

The shell wrapper delegates to `python3 tools/run_ttt_compare.py` and loads one of the boilerplates in `configs/examples/ttt_compare/`.

Available boilerplates:
- `tent`
- `mim`
- `mim_probe`
- `cotta`
- `eata`
- `sar`

## Method comparison map

The wrapper always writes one baseline prediction file and one adapted prediction
file for the selected method. The main thing to compare is the
`<method>_before_after_compare.{json,md}` pair.

| Method | Boilerplate | Baseline artifact | Adapted artifact | Primary before/after artifact | Optional follow-up |
| --- | --- | --- | --- | --- | --- |
| Tent | `tent` | `reports/ttt_compare/tent/baseline_predictions.json` | `reports/ttt_compare/tent/tent_predictions.json` | `reports/ttt_compare/tent/tent_before_after_compare.md` | Inspect `tent_ttt_log.json` for loss/runtime guards |
| MIM (generic) | `mim` | `reports/ttt_compare/mim/baseline_predictions.json` | `reports/ttt_compare/mim/mim_predictions.json` | `reports/ttt_compare/mim/mim_before_after_compare.md` | Inspect `mim_ttt_log.json`; use `pose_mim` for pose-heavy runs |
| MIM (fixed real probe) | `mim_probe` | `reports/ttt_compare/mim_probe_cpu/baseline_predictions.json` | `reports/ttt_compare/mim_probe_cpu/mim_predictions.json` | `reports/ttt_compare/mim_probe_cpu/mim_before_after_compare.md` | Uses the fixed 10-image shifted probe and the built-in `simple_map_proxy` fallback when `pycocotools` is unavailable |
| CoTTA | `cotta` | `reports/ttt_compare/cotta/baseline_predictions.json` | `reports/ttt_compare/cotta/cotta_predictions.json` | `reports/ttt_compare/cotta/cotta_before_after_compare.md` | `tools/eval_cotta_drift.py` for augmentation/EMA drift review |
| EATA | `eata` | `reports/ttt_compare/eata/baseline_predictions.json` | `reports/ttt_compare/eata/eata_predictions.json` | `reports/ttt_compare/eata/eata_before_after_compare.md` | `tools/benchmark_eata_stability.py` for selective-update stability |
| SAR | `sar` | `reports/ttt_compare/sar/baseline_predictions.json` | `reports/ttt_compare/sar/sar_predictions.json` | `reports/ttt_compare/sar/sar_before_after_compare.md` | `tools/benchmark_sar_robustness.py` for CoTTA/EATA/SAR side-by-side review |

Recommended reading order for every method:
1. open `plan.json` to confirm the boilerplate and dataset subset,
2. open `<method>_before_after_compare.md` for the compact summary,
3. open `<method>_ttt_log.json` if you need the adaptation loss/runtime detail,
4. only then reach for method-specific follow-up tools.

Notes:
- The shipped `mim` and `sar` boilerplates expand the safe defaults directly
  and force `--ttt-update-filter norm_only`; provide a current-compatible
  checkpoint unless the selected workflow documents a different fixture.
- The `mim` boilerplate also injects the repo-backed config `configs/examples/ttt_compare/rtdetr_pose_mim_compare.json` into both the baseline and adapted export so the MIM branch is enabled without a long operator command.
- The `mim_probe` boilerplate injects `configs/examples/ttt_compare/yolo26n_mim_real_probe.json` so the fixed yolo26n probe checkpoint can demonstrate a real before/after metric change with MIM enabled.
- If you have a checkpoint with dedicated adapter/LoRA parameters, copy the JSON boilerplate and switch the update filter there instead of expanding the raw CLI.

## Beginner reading guide

If you are new to TTT, do not start from the raw JSON logs.

Read the generated files in this order:
1. `plan.json`
2. `<method>_before_after_compare.md`
3. `<method>_ttt_log.json`
4. optional follow-up method-specific reports

Why this order works:
- `plan.json` tells you what actually ran
- `before_after_compare.md` tells you whether anything changed
- `ttt_log.json` tells you why

Common interpretations:
- `steps_run > 0`: the method really adapted
- `changed_images = 0`: the workflow ran, but exported predictions did not change on that subset
- `mean_final_loss`: adaptation objective value, not a universal leaderboard metric
- warning notes belong in prose, not in a wide table column

Example:
- EATA can finish successfully with `steps_run=0`
- that usually means its sample-selection guard decided adaptation was unsafe on that subset
- that is conservative behavior, not an implementation failure

## Fixed compare snapshots

### Real fixed probe for operator-visible effect

The clearest built-in MIM example is the fixed ten-image shifted probe used by
the TTT improvement demo. This is the compare to use when you want to confirm
that MIM changes predictions and improves the repo's built-in quality proxy on a
stable subset.

```bash
bash scripts/ttt_compare.sh \
  --boilerplate mim_probe \
  --dataset reports/ttt_improvement_probe/domain_shift_dataset \
  --split val \
  --checkpoint reports/ttt_improvement_probe/checkpoint.pt \
  --run-dir reports/ttt_compare/mim_probe_cpu \
  --device cpu \
  --max-images 10 \
  --force
```

Fixed real-probe result:
- `reports/ttt_compare/mim_probe_cpu/mim_before_after_compare.md`
- `steps_run=10`
- `changed_images=10 / 10`
- `map50 0.00326797 -> 0.00392157`
- `map50_95 0.000326797 -> 0.000392157`
- metric backend: `simple_map_proxy` (used automatically here because `pycocotools` is absent in the local runtime)

### Historical smoke compare snapshot

The historical source commit
`72f0862f2487c7a23267820cc2dfc4818e46118b` ran the boilerplates against
`reports/rtdetr_pose_ckpt_coco128_gpu_matcher.pt` on `data/smoke`, `split=val`,
`max-images=2`, `device=cpu`, with `--skip-eval`.

The checkpoint's claimed config was not committed, and it is not fully
compatible with the current model. These rows are preserved only as historical
workflow records; they are neither current execution evidence nor performance
claims. Pinned hashes and the current compatibility audit are in
`reports/rtdetr_pose_coco128_gpu_matcher_historical.json`.

| Method | Run dir | Real compare status | Steps run | Mean final loss |
| --- | --- | --- | ---: | ---: |
| Tent | `reports/ttt_compare/tent_smoke_cpu` | completed | `2` | `4.214431` |
| MIM | `reports/ttt_compare/mim_smoke_cpu` | completed | `2` | `0.461853` |
| CoTTA | `reports/ttt_compare/cotta_smoke_cpu` | completed | `1` | `4.243579` |
| EATA | `reports/ttt_compare/eata_smoke_cpu` | completed | `0` | `null` |
| SAR | `reports/ttt_compare/sar_smoke_cpu` | completed | `2` | `4.211009` |

Warning notes:
- MIM uses the repo-backed compare config `configs/examples/ttt_compare/rtdetr_pose_mim_compare.json`
- EATA reports `eata_empty_selected_set` on both smoke images
- Tent, CoTTA, and SAR complete without warnings on this smoke subset

For the completed runs, the compact before/after summaries live here:
- `reports/ttt_compare/mim_probe_cpu/mim_before_after_compare.md`
- `reports/ttt_compare/tent_smoke_cpu/tent_before_after_compare.md`
- `reports/ttt_compare/mim_smoke_cpu/mim_before_after_compare.md`
- `reports/ttt_compare/cotta_smoke_cpu/cotta_before_after_compare.md`
- `reports/ttt_compare/eata_smoke_cpu/eata_before_after_compare.md`
- `reports/ttt_compare/sar_smoke_cpu/sar_before_after_compare.md`

## Method selection cheat sheet

| Method | Start here when | Why it helps | Main cost |
| --- | --- | --- | --- |
| Tent | you want the safest first compare | simplest entropy-based adaptation | weakest signal |
| MIM | you need geometry-aware adaptation | stronger self-supervised signal | more model-specific |
| CoTTA | you care about streaming behavior | EMA teacher + restoration | more state and more compute |
| EATA | you prefer conservative adaptation | selective updates + regularization | may skip adaptation often |
| SAR | the shift is noisy or unstable | sharpness-aware updates | slower per step |

## Method-specific entry examples

Tent:

```bash
bash scripts/ttt_compare.sh \
  --boilerplate tent \
  --dataset data/smoke \
  --split val \
  --checkpoint /path/to.ckpt \
  --run-dir reports/ttt_compare/tent \
  --device cuda
```

MIM:

```bash
bash scripts/ttt_compare.sh \
  --boilerplate mim \
  --dataset data/smoke \
  --split val \
  --checkpoint /path/to.ckpt \
  --run-dir reports/ttt_compare/mim \
  --device cuda
```

Fixed real-probe example:

```bash
bash scripts/ttt_compare.sh \
  --boilerplate mim_probe \
  --dataset reports/ttt_improvement_probe/domain_shift_dataset \
  --split val \
  --checkpoint reports/ttt_improvement_probe/checkpoint.pt \
  --run-dir reports/ttt_compare/mim_probe_cpu \
  --device cpu \
  --max-images 10 \
  --force
```

Current-compatible smoke example:

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

CoTTA:

```bash
bash scripts/ttt_compare.sh \
  --boilerplate cotta \
  --dataset data/smoke \
  --split val \
  --checkpoint /path/to.ckpt \
  --run-dir reports/ttt_compare/cotta \
  --device cuda
```

EATA:

```bash
bash scripts/ttt_compare.sh \
  --boilerplate eata \
  --dataset data/smoke \
  --split val \
  --checkpoint /path/to.ckpt \
  --run-dir reports/ttt_compare/eata \
  --device cuda
```

SAR:

```bash
bash scripts/ttt_compare.sh \
  --boilerplate sar \
  --dataset data/smoke \
  --split val \
  --checkpoint /path/to.ckpt \
  --run-dir reports/ttt_compare/sar \
  --device cuda
```

## Generated artifacts

Each compare run writes:
- `plan.json`
- `baseline_predictions.json`
- `<method>_predictions.json`
- `<method>_ttt_log.json`
- `baseline_eval.json` (best effort; omitted when `--skip-eval` is used)
- `<method>_eval.json` (best effort)
- `<method>_before_after_compare.json`
- `<method>_before_after_compare.md`

The compare report summarizes:
- baseline vs adapted prediction counts,
- adapted TTT summary (`method`, `preset`, `mean_final_loss`, `mean_seconds`, guard warnings),
- before-after prediction drift summary (`changed_images`, `missing_match_failures`, `value_mismatch_failures`),
- optional eval metrics when `eval_suite.py` succeeds.

## Dry-run for planning

Use `--dry-run` to write the plan without running export/eval:

```bash
bash scripts/ttt_compare.sh \
  --boilerplate tent \
  --dataset data/smoke \
  --split val \
  --checkpoint /path/to.ckpt \
  --run-dir reports/ttt_compare/tent_plan \
  --dry-run \
  --force
```

This is the easiest way to confirm the exact baseline/adapted commands before launching a real compare job.
