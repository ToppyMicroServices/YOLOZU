# Score calibration (no NMS)

This repo provides post-hoc score calibration paths for long-tail robustness and stable comparison across runs.

## `yolozu calibrate` methods

The unified CLI supports these methods:

- `fracal`: class-frequency-aware non-linear score transform.
- `la`: Logit Adjustment (LA).
- `norcal`: normalized frequency reweighting (NorCal-style).
- `temperature`: temperature scaling.

### Temperature scaling

Basic softmax form:

$$
p(y\mid x)=\mathrm{softmax}(z)_y,
\quad
p_T(y\mid x)=\mathrm{softmax}(z/T)_y
$$

Intuition:

- $T>1$: softer/flatter confidence distribution (typically reduces over-confidence).
- $T<1$: sharper confidence distribution.

Fitting objective on validation data:

$$
\min_{T>0}\sum_i -\log p_T(y_i\mid x_i)
$$

In this repo, `yolozu calibrate --method temperature` applies global $T$ directly; optional `--fit-temperature` performs a lightweight 1D grid search on the validation subset (bbox/pose path) using matched prediction targets.

Detector placement:

- For multi-class softmax classifiers, apply $z/T$ to classification logits.
- For sigmoid (independent class heads), apply $\sigma(z/T)$ per class.
- Start with one global $T$; class-wise/bin-wise temperatures are possible later.

### Logit Adjustment (LA)

Formula:

$$
z'_y = z_y - \tau \log p_s(y)
$$

- Required stats: source-domain class prior $p_s(y)$ (derived from class counts).
- Main knob: `--tau` (typically sweep around $0\sim1$).
- Practical property: relatively insensitive to explicit/implicit background handling, and easy to apply when logits can be recovered from scores.

### NorCal

Formula:

$$
p'_y \propto \frac{p_y}{n_y^{\gamma}}
$$

(with re-normalization implied)

- Required stats: class frequency $n_y$.
- Main knob: `--gamma`.
- Practical property: very short implementation and easy interpretation as probability re-distribution.
- Caveat: with sigmoid + implicit background, the exact multi-class renormalization form is less direct; current implementation uses a one-vs-bg logit-shift approximation.

## CLI examples

```bash
# FRACAL baseline
yolozu calibrate \
  --method fracal \
  --task bbox \
  --dataset data/coco128 \
  --predictions reports/predictions_smoke.json \
  --stats-out reports/fracal_stats_bbox.json \
  --output reports/predictions_fracal.json \
  --output-report reports/calibration_fracal_report.json

# LA
yolozu calibrate \
  --method la \
  --tau 0.6 \
  --task bbox \
  --dataset data/coco128 \
  --predictions reports/predictions_smoke.json \
  --stats-in reports/fracal_stats_bbox.json \
  --output reports/predictions_la.json \
  --output-report reports/calibration_la_report.json

# NorCal
yolozu calibrate \
  --method norcal \
  --gamma 0.8 \
  --task bbox \
  --dataset data/coco128 \
  --predictions reports/predictions_smoke.json \
  --stats-in reports/fracal_stats_bbox.json \
  --output reports/predictions_norcal.json \
  --output-report reports/calibration_norcal_report.json

# Temperature scaling
yolozu calibrate \
  --method temperature \
  --temperature 1.5 \
  --task bbox \
  --dataset data/coco128 \
  --predictions reports/predictions_smoke.json \
  --output reports/predictions_temperature.json \
  --output-report reports/calibration_temperature_report.json

# Temperature fitting (bbox/pose)
yolozu calibrate \
  --method temperature \
  --fit-temperature \
  --temperature-grid 0.5,0.75,1.0,1.25,1.5,2.0 \
  --task bbox \
  --dataset data/coco128 \
  --predictions reports/predictions_smoke.json \
  --output reports/predictions_temperature_fitted.json \
  --output-report reports/calibration_temperature_fitted_report.json
```

The same methods can be compared across `--task bbox|seg|pose`.

## Legacy helper

This repo also keeps a lightweight temperature-scaling helper script for quick experiments.

## CLI

```bash
python3 tools/calibrate_scores.py \
  --dataset data/coco128 \
  --split val2017 \
  --predictions reports/predictions_dummy.json \
  --output reports/predictions_calibrated.json \
  --output-report reports/calibration_report.json \
  --output-artifact reports/calibration_artifact.json \
  --temperatures 0.5,1.0,1.5,2.0 \
  --min-score 0.0 \
  --max-score 1.0
```

## Outputs

- `reports/predictions_calibrated.json`: calibrated predictions JSON.
- `reports/calibration_report.json`: before/after metrics + selected temperature.
- `reports/calibration_artifact.json`: grid search results.

## Notes

This uses a lightweight mAP proxy (`yolozu.simple_map`) to avoid external dependencies. The metrics are suitable for relative comparisons on a fixed subset.

Optional legacy-helper knobs:

- `--split`: override dataset split auto-detection.
- `--temperatures`: comma-separated grid for the temperature sweep.
- `--min-score` / `--max-score`: clamp calibrated scores before writing the artifact.
