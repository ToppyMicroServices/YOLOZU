# SynthGen intake (YOLOZU side)

YOLOZU does **not** generate synthetic data.  
`YOLOZU-synthgen` (external repo) produces shards; YOLOZU consumes them through a stable contract.

## Contract boundary

- Contract spec: `docs/contracts/synthgen.md`
- Runtime validator/coercion: `yolozu/contracts/synthgen.py`
- Required fields:
  - `image`, `depth_ndc`, `inst_id`, `sem_id`, `kpts2d`
  - `prompt`, `scene_spec`, `schema_id`, `schema_version`, `asset_ids`, `inst_map`

## Dataset adapters

- Shard adapter: `yolozu/data/synthgen_shard_dataset.py`
- Stream adapter: `yolozu/data/synthgen_stream_dataset.py`

Schema filter is first-class (`schema_id`), so `animal_v1` and `mechanical_v1` can be trained/evaluated independently.

## Task templates

- `configs/examples/synthgen/synthgen_animal_kpt.yaml`
- `configs/examples/synthgen/synthgen_mechanical_kpt.yaml`

These templates pin:
- required fields
- schema filter
- keypoint visibility policy (`vis==2` for regression)
- evaluation families (keypoints/seg/depth)

## Commands

Validate contract:

```bash
python3 tools/validate_synthgen_contract.py \
  --input /path/to/synthgen_dataset/shards/train_000.jsonl \
  --max-samples 200
```

Render overlay:

```bash
python3 tools/render_synthgen_overlay.py \
  --dataset-root /path/to/synthgen_dataset \
  --schema-id animal_v1 \
  --sample-index 0 \
  --output reports/synthgen_overlay.png
```

Evaluate predictions:

```bash
python3 tools/eval_synthgen.py \
  --dataset-root /path/to/synthgen_dataset \
  --predictions reports/synthgen_predictions.json \
  --schema-id animal_v1 \
  --output reports/synthgen_eval.json
```
