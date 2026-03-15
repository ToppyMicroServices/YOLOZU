# Prediction distillation helper

This repo includes a lightweight distillation helper that blends teacher predictions into student predictions and records a distillation loss term.

This is an **offline, prediction-artifact** utility: it rewrites a predictions JSON by blending another model's predictions.
It is **not** the continual-learning / anti-forgetting mechanism (no training loop, no replay buffer, no checkpoint-based regularization).

The current approach is intentionally **score-level / label-free**. It is suitable for quick offline ablations, but production use should enable guard parameters to avoid over-injection and duplicate propagation.

If your goal is catastrophic-forgetting mitigation across tasks/domains, use the continual-learning workflow:
- `rtdetr_pose/tools/train_continual.py` (runner)
- `rtdetr_pose/tools/train_minimal.py` with `--self-distill-from` (+ optional replay/EWC/SI)
- Docs: `docs/continual_learning.md`

## CLI

```bash
python3 tools/distill_predictions.py \
  --student reports/predictions_student.json \
  --teacher reports/predictions_teacher.json \
  --dataset data/coco128 \
  --split val2017 \
  --alpha 0.5 \
  --iou-threshold 0.7 \
  --output reports/predictions_distilled.json \
  --output-report reports/distill_report.json \
  --add-missing \
  --teacher-min-score 0.25 \
  --max-added-per-image 20 \
  --add-duplicate-iou-threshold 0.9
```

## Config

Optional JSON config allows enabling/disabling distillation and tuning parameters:

```json
{
  "enabled": true,
  "iou_threshold": 0.7,
  "alpha": 0.5,
  "add_missing": true,
  "add_score_scale": 0.5,
  "teacher_min_score": 0.25,
  "max_added_per_image": 20,
  "add_duplicate_iou_threshold": 0.9
}
```

## Notes

Evaluation uses the lightweight `yolozu.simple_map` proxy to compare student vs distilled predictions on a fixed subset.

Recommended safety defaults for practical use:
- `teacher_min_score >= 0.2` to suppress low-confidence teacher noise
- finite `max_added_per_image` to cap unmatched-teacher injection
- high `add_duplicate_iou_threshold` (e.g., `0.85`-`0.95`) to prevent duplicate growth
- use `--split` when the dataset root contains multiple YOLO splits and auto-detection would be ambiguous

For continual-learning evaluation (including forgetting summaries), see `tools/eval_continual.py` and `docs/continual_learning.md`.
