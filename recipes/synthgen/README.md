# SynthGen intake recipes

This recipe shows how to consume external `YOLOZU-synthgen` shards in YOLOZU.
Generation stays in the external repo; YOLOZU handles contract validation, visualization, and evaluation.

## 1) Validate contract records

```bash
python3 tools/validate_synthgen_contract.py \
  --input /path/to/synthgen_dataset/shards/train_000.jsonl \
  --max-samples 200
```

## 2) Schema-specific task templates

- `configs/examples/synthgen/synthgen_animal_kpt.yaml`
- `configs/examples/synthgen/synthgen_mechanical_kpt.yaml`

Use these as run configs to pin `schema_id`, required fields, and default metric families.

## 3) Render debug overlays

```bash
python3 tools/render_synthgen_overlay.py \
  --dataset-root /path/to/synthgen_dataset \
  --schema-id animal_v1 \
  --sample-index 0 \
  --output reports/synthgen_overlay_animal.png
```

Overlay includes:
- semantic overlay (`sem_id`)
- instance boundaries (`inst_id`)
- keypoints + schema-specific skeleton (`kpts2d`)

## 4) Evaluate prediction outputs

Expected prediction record format (JSON list):

```json
[
  {
    "sample_id": "train_000:12",
    "sem_id": "pred/sem_12.npy",
    "inst_id": "pred/inst_12.npy",
    "depth_ndc": "pred/depth_12.npy",
    "kpts2d": "pred/kpts_12.npy"
  }
]
```

Run evaluator:

```bash
python3 tools/eval_synthgen.py \
  --dataset-root /path/to/synthgen_dataset \
  --predictions reports/synthgen_predictions.json \
  --schema-id animal_v1 \
  --output reports/synthgen_eval_animal.json
```

Report includes:
- keypoint visible-point mean pixel error
- semantic mIoU summary
- depth MAE/MSE
- instance pixel agreement
