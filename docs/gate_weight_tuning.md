# Inference-time gate weight tuning (CPU-only)

YOLOZU’s spec includes a lightweight **inference-time rescoring** step that can be tuned *without retraining*.

Final score (spec §8.4):
- `S = w_det*score_det + w_tmp*score_tmp_sym - w_unc*(sigma_z + sigma_rot)`

This repo provides:
- the scoring primitive in `yolozu/gates.py`
- a score-fusion transform in `yolozu/predictions_transform.py`
- an offline tuner in `tools/tune_gate_weights.py` (uses the dependency-light `yolozu.simple_map` proxy)

## Expected detection fields

Per detection (inside `detections[]`), the tuner looks for:
- detection score: `score` (configurable via `--det-score-key`)
- template score: `score_tmp_sym` (optional; defaults to `0.0` if missing)
- uncertainty proxies: `sigma_z`, `sigma_rot` (optional; default to `0.0` if missing)

Notes:
- `RTDETRPoseAdapter` can emit `log_sigma_*` and `sigma_*` when `use_uncertainty=true` in the model config.
- You can plug your own template verification and write `score_tmp_sym` per detection.

## Quick start (coco128)

1) Produce predictions JSON (any backend is fine). For the in-repo reference trainer:

```bash
python3 tools/export_predictions.py \
  --adapter rtdetr_pose \
  --dataset data/coco128 \
  --max-images 50 \
  --wrap \
  --output reports/predictions_rtdetr_pose.json
```

2) Tune weights on CPU:

```bash
python3 tools/tune_gate_weights.py \
  --dataset data/coco128 \
  --predictions reports/predictions_rtdetr_pose.json \
  --metric map50_95 \
  --grid-det 1.0 \
  --grid-tmp 0.0,0.5,1.0 \
  --grid-unc 0.0,0.5,1.0 \
  --output-report reports/gate_tuning_report.json
```

3) (Optional) Write tuned predictions for the best weights:

```bash
python3 tools/tune_gate_weights.py \
  --dataset data/coco128 \
  --predictions reports/predictions_rtdetr_pose.json \
  --output-report reports/gate_tuning_report.json \
  --output-predictions reports/predictions_tuned.json \
  --wrap-output
```

## Config file

`tools/tune_gate_weights.py` supports a JSON config. Convenience keys:
- `keys`: `{det_score, template_score, sigma_z, sigma_rot, preserve_det_score}`
- `grid`: `{det, tmp, unc, tau}`
- `template_gate`: `{enabled, tau}`

Minimal example:

```json
{
  "dataset": "data/coco128",
  "predictions": "reports/predictions.json",
  "metric": "map50_95",
  "grid": { "det": [1.0], "tmp": [0.0, 0.5, 1.0], "unc": [0.0, 0.5, 1.0] },
  "output_report": "reports/gate_tuning_report.json"
}
```

### Precedence and path resolution

- Priority: **CLI flags > config file > built-in defaults**.
- Relative paths passed via CLI are resolved from your current working directory.
- Relative paths inside `--config` are resolved from that config file’s directory.
- For backward compatibility, repo-root fallback is still applied for input paths.

## Why this is useful on macOS

You can do:
- schema validation + transforms
- offline score/weight tuning on CPU
- regression tracking (JSONL history)

…and keep GPU/TensorRT build + real FPS measurement for Linux/Runpod.
