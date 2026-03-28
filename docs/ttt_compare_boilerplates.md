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
| MIM | `mim` | `reports/ttt_compare/mim/baseline_predictions.json` | `reports/ttt_compare/mim/mim_predictions.json` | `reports/ttt_compare/mim/mim_before_after_compare.md` | Inspect `mim_ttt_log.json`; use `pose_mim` for pose-heavy runs |
| CoTTA | `cotta` | `reports/ttt_compare/cotta/baseline_predictions.json` | `reports/ttt_compare/cotta/cotta_predictions.json` | `reports/ttt_compare/cotta/cotta_before_after_compare.md` | `tools/eval_cotta_drift.py` for augmentation/EMA drift review |
| EATA | `eata` | `reports/ttt_compare/eata/baseline_predictions.json` | `reports/ttt_compare/eata/eata_predictions.json` | `reports/ttt_compare/eata/eata_before_after_compare.md` | `tools/benchmark_eata_stability.py` for selective-update stability |
| SAR | `sar` | `reports/ttt_compare/sar/baseline_predictions.json` | `reports/ttt_compare/sar/sar_predictions.json` | `reports/ttt_compare/sar/sar_before_after_compare.md` | `tools/benchmark_sar_robustness.py` for CoTTA/EATA/SAR side-by-side review |

Recommended reading order for every method:
1. open `plan.json` to confirm the boilerplate and dataset subset,
2. open `<method>_before_after_compare.md` for the compact summary,
3. open `<method>_ttt_log.json` if you need the adaptation loss/runtime detail,
4. only then reach for method-specific follow-up tools.

Notes:
- The shipped `mim` and `sar` boilerplates expand the safe defaults directly and force `--ttt-update-filter norm_only` so they work with the repo-shipped checkpoint without requiring LoRA-specific weights.
- The `mim` boilerplate also injects the repo-backed config `configs/examples/ttt_compare/rtdetr_pose_mim_compare.json` into both the baseline and adapted export so the MIM branch is enabled without a long operator command.
- If you have a checkpoint with dedicated adapter/LoRA parameters, copy the JSON boilerplate and switch the update filter there instead of expanding the raw CLI.

## Smoke compare snapshot (repo-shipped checkpoint)

We also ran the boilerplates against the repo-shipped checkpoint
`reports/rtdetr_pose_ckpt_coco128_gpu_matcher.pt` on `data/smoke`, `split=val`,
`max-images=2`, `device=cpu`, with `--skip-eval`.

These are workflow-validation results, not performance claims. The tiny smoke
subset produced zero detections in both baseline and adapted runs, so the useful
signal here is whether the method completed and what the TTT log reported.

| Method | Run dir | Real compare status | Steps run | Mean final loss | Warning summary |
| --- | --- | --- | ---: | ---: | --- |
| Tent | `reports/ttt_compare/tent_smoke_cpu` | completed | `2` | `4.214431` | none |
| MIM | `reports/ttt_compare/mim_smoke_cpu` | completed | `2` | `0.461853` | repo-backed config `configs/examples/ttt_compare/rtdetr_pose_mim_compare.json` |
| CoTTA | `reports/ttt_compare/cotta_smoke_cpu` | completed | `1` | `4.243579` | none |
| EATA | `reports/ttt_compare/eata_smoke_cpu` | completed | `0` | `null` | `eata_empty_selected_set` on both smoke images |
| SAR | `reports/ttt_compare/sar_smoke_cpu` | completed | `2` | `4.211009` | none |

For the completed runs, the compact before/after summaries live here:
- `reports/ttt_compare/tent_smoke_cpu/tent_before_after_compare.md`
- `reports/ttt_compare/mim_smoke_cpu/mim_before_after_compare.md`
- `reports/ttt_compare/cotta_smoke_cpu/cotta_before_after_compare.md`
- `reports/ttt_compare/eata_smoke_cpu/eata_before_after_compare.md`
- `reports/ttt_compare/sar_smoke_cpu/sar_before_after_compare.md`

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

Repo-backed smoke example:

```bash
bash scripts/ttt_compare.sh \
  --boilerplate mim \
  --dataset data/smoke \
  --split val \
  --checkpoint reports/rtdetr_pose_ckpt_coco128_gpu_matcher.pt \
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
