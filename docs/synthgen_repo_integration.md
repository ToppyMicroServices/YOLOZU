# YOLOZU-synthgen integration checklist

Use this page when an external generator repo such as `YOLOZU-synthgen` is preparing to hand off synthetic shards to YOLOZU.

This handoff was verified end to end with a local `YOLOZU-synthgen` checkout on April 2, 2026 using the CPU path. No GPU or MPS runtime is required for the intake, overlay, or evaluation steps below.

## Goal

Keep the integration boundary small and stable:

- generator repo owns prompts, rendering, asset selection, and shard creation
- YOLOZU owns intake validation, loading, overlay inspection, and evaluation
- the shared boundary is the SynthGen predictions interface contract

Primary references:

- [`docs/synthgen_contract.md`](synthgen_contract.md)
- [`docs/synthgen_intake.md`](synthgen_intake.md)
- [`schemas/synthgen_sample.schema.json`](../schemas/synthgen_sample.schema.json)

## How YOLOZU-synthgen generates samples

`YOLOZU-synthgen` is split-responsibility by design.

- renderer owns geometry, object count, pose, camera, base lighting, and all authoritative ground truth
- gen AI owns recipe drafting assistance, appearance recipe drafting, appearance editing, and QA assistance
- gen AI must not add/remove labeled objects or become the source of instance/semantic/keypoint ground truth

In practice, the generator-side pipeline is:

1. create `scene_recipe`
2. materialize `scene_spec`
3. render `image_render` plus GT passes
4. create `appearance_recipe`
5. materialize `appearance_spec`
6. apply appearance editing if the selected mode allows it
7. run QA gate
8. emit accepted sample directories and shard rows

Current geometry sources are renderer-side `primitive_box` placeholders and `mesh_file` assets. That means CAD-derived assets fit naturally once they are exported as meshes and referenced in the generator asset catalogs.

Supported generation modes:

- `render_only`
- `bg_only_inpaint`
- `appearance_only_conditioned`

Intentionally unsupported as the main labeled-data path:

- full-image regeneration (`full_regen`)
- gen-AI-predicted masks or keypoints as ground truth
- text-to-3D as the main geometry source

## How backgrounds are generated and selected

The first background choice is renderer-side, not free-form image generation:

- `scene_recipe.lighting_policy.envmap_ids` defines the allowed environment-map pool
- the compiler samples one `envmap_id` from that pool
- `lighting_policy.env_intensity_range` controls the sampled ambient/background intensity

That produces the base render and keeps labels stable because depth, instance ids, semantic ids, and keypoints are all computed from the same rendered scene state.

Optional post-render background generation happens only through label-safe appearance modes:

- `render_only`: no appearance editing
- `bg_only_inpaint`: edit only background pixels/masks outside labeled objects
- `appearance_only_conditioned`: object-interior appearance edits while preserving object count and silhouette

For `bg_only_inpaint`, the contract is explicit:

- `mask_scope = background_only`
- controls may include renderer-derived guides such as `depth_ndc`, `inst_boundary`, and `foreground_union_mask`
- object boundaries and labels must not change

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

## How images and labels are handled

The generator keeps two image concepts:

- `image_render`: direct renderer output before optional appearance editing
- `image`: final sample image after the selected mode is applied

For `render_only`, `image == image_render`. For appearance-edited modes, `image` may differ visually, but labels remain tied to the renderer-owned scene geometry.

Ground-truth handling is renderer-derived throughout:

- `depth_ndc` comes from the renderer depth pass
- `inst_id` comes from the instance-id pass
- `sem_id` comes from the semantic-id pass
- `kpts2d` is projected from object-space keypoints through the scene camera and then refined against depth / instance-id where available

Typical per-sample files in the generator repo are:

- `image.png`
- `depth_ndc.npy`
- `inst_id.npy`
- `sem_id.npy`
- `kpts2d.npy`
- `meta.json`
- `appearance.json`
- `qa.json`
- `image_render.png` when the final image is not `render_only`

## Recommended handoff layout

```text
<dataset-root>/
  shards/
    train_000.jsonl
    predictions_synthgen.json
  <sample>_image.png
  <sample>_depth.npy
  <sample>_inst.npy
  <sample>_sem.npy
  <sample>_kpts.npy
```

Notes:

- relative asset paths in shard rows should resolve from the dataset root
- `predictions_synthgen.json` should follow the YOLOZU predictions interface contract
- if prediction records reuse shard-relative asset paths such as `../samples/...`, write the predictions artifact under `shards/` so those paths keep resolving from the same base
- generator-side metadata can grow, but breaking contract changes require a new schema version

## Verified repo-to-repo probe

The following probe was run successfully across `YOLOZU-synthgen` and YOLOZU:

1. Generate demo shards in `YOLOZU-synthgen`:

```bash
PYTHONPATH=src ./.venv/bin/python -m yolozu_synthgen generate-demo-dataset \
  --output-dir /tmp/yolozu_synthgen_demo \
  --num-train 2 \
  --num-val 1
```

2. Export a YOLOZU-compatible dataset root:

```bash
PYTHONPATH=src ./.venv/bin/python -m yolozu_synthgen export-yolozu-synthgen \
  --shards-root /tmp/yolozu_synthgen_demo/shards \
  --output-dir /tmp/yolozu_synthgen_export
```

3. Run the YOLOZU handoff checks against `/tmp/yolozu_synthgen_export`.

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
  --predictions /path/to/synthgen_dataset/shards/predictions_synthgen.json \
  --schema-id animal_v1 \
  --output reports/synthgen_eval.json
```

4. Run the bundled end-to-end smoke:

```bash
python3 tools/smoke_synthgen.py \
  --dataset-root /path/to/synthgen_dataset \
  --predictions /path/to/synthgen_dataset/shards/predictions_synthgen.json \
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
- path resolution is explicit: either keep predictions under `shards/` or rewrite any shard-relative asset paths before evaluation

## Versioning rule

- add optional fields freely
- add new `schema_id` values freely
- do not change required dtype / shape / range semantics inside `schema_version = "1"`
- if the generator needs breaking changes, version the contract first in YOLOZU, then enable the new version in adapters and docs
