# YOLOZU-synthgen integration checklist

Use this page when an external generator repo such as `YOLOZU-synthgen` is preparing to hand off synthetic shards to YOLOZU.

## Goal

Keep the integration boundary small and stable:

- generator repo owns prompts, rendering, asset selection, and shard creation
- YOLOZU owns intake validation, loading, overlay inspection, and evaluation
- the shared boundary is the SynthGen predictions interface contract

Primary references:

- [`docs/synthgen_contract.md`](synthgen_contract.md)
- [`docs/synthgen_intake.md`](synthgen_intake.md)
- [`schemas/synthgen_sample.schema.json`](../schemas/synthgen_sample.schema.json)

## What YOLOZU-synthgen must emit

Each record in `shards/*.jsonl` must satisfy `schema_version = "1"` of the SynthGen sample interface contract.

Required fields:

- `image`
- `depth_ndc`
- `inst_id`
- `sem_id`
- `kpts2d`
- `prompt`
- `scene_spec`
- `schema_id`
- `schema_version`
- `asset_ids`
- `inst_map`

Required semantics:

- `image` is RGB `uint8[H,W,3]`
- `depth_ndc` is `float32[H,W]` in `[0,1]`
- `inst_id` is `uint32[H,W]`
- `sem_id` is `uint16[H,W]`
- `kpts2d` is `float32[N_inst,K,3]` with `(u,v,vis)`
- `kpts2d[...,2]` uses `0/1/2` visibility semantics

## Recommended handoff layout

```text
<dataset-root>/
  shards/
    train_000.jsonl
  <sample>_image.png
  <sample>_depth.npy
  <sample>_inst.npy
  <sample>_sem.npy
  <sample>_kpts.npy
  predictions_synthgen.json
```

Notes:

- relative asset paths in shard rows should resolve from the dataset root
- `predictions_synthgen.json` should follow the YOLOZU predictions interface contract
- generator-side metadata can grow, but breaking contract changes require a new schema version

## Handoff checks to run in YOLOZU

1. Validate contract rows:

```bash
python3 tools/validate_synthgen_contract.py \
  --input /path/to/synthgen_dataset/shards/train_000.jsonl \
  --max-samples 200
```

2. Render one overlay:

```bash
python3 tools/render_synthgen_overlay.py \
  --dataset-root /path/to/synthgen_dataset \
  --schema-id animal_v1 \
  --sample-index 0 \
  --output reports/synthgen_overlay.png
```

3. Evaluate one predictions artifact:

```bash
python3 tools/eval_synthgen.py \
  --dataset-root /path/to/synthgen_dataset \
  --predictions /path/to/synthgen_dataset/predictions_synthgen.json \
  --schema-id animal_v1 \
  --output reports/synthgen_eval.json
```

4. Run the bundled end-to-end smoke:

```bash
python3 tools/smoke_synthgen.py \
  --dataset-root /path/to/synthgen_dataset \
  --predictions /path/to/synthgen_dataset/predictions_synthgen.json \
  --output-dir reports
```

## CI-ready acceptance criteria

Integration is ready when all of the following are true:

- `validate_synthgen_contract.py` returns `OK`
- `render_synthgen_overlay.py` writes a non-empty overlay image
- `eval_synthgen.py` writes a report JSON without schema or shape errors
- `smoke_synthgen.py` writes:
  - `reports/smoke_synthgen_overlay.png`
  - `reports/smoke_synthgen_eval.json`
  - `reports/smoke_synthgen_summary.json`

## Versioning rule

- add optional fields freely
- add new `schema_id` values freely
- do not change required dtype / shape / range semantics inside `schema_version = "1"`
- if the generator needs breaking changes, version the contract first in YOLOZU, then enable the new version in adapters and docs
