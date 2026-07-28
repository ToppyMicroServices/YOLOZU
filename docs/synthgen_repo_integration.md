# YOLOZU-synthgen integration checklist

Use this page when an external generator repo such as `YOLOZU-synthgen` is preparing to hand off synthetic shards to YOLOZU.

This handoff was re-qualified end to end with the Open3D renderer on July 28,
2026. The intake, loading, overlay, and evaluation steps are CPU-capable.
Treat this as an experimental handoff path: `render_only` and the deterministic
`internal_stub` for `bg_only_inpaint` are qualified locally, but no external
generative provider or downstream model quality is qualified.
See [`production_readiness.md`](production_readiness.md).

## One-page handoff order

1. Export a YOLOZU-compatible dataset root from the generator repo.
2. Keep shard rows and asset paths explicit.
3. Place `predictions_synthgen.json` where its relative paths still resolve.
4. Run `validate_synthgen_contract.py`.
5. Run `render_synthgen_overlay.py`.
6. Run `eval_synthgen.py`.
7. Run `smoke_synthgen.py`; pass `--synthgen-repo` for a fresh one-command
   generator-to-YOLOZU qualification.

## Goal

Keep the integration boundary small and stable:

- generator repo owns prompts, rendering, asset selection, and shard creation
- YOLOZU owns intake validation, loading, overlay inspection, and evaluation
- the shared boundary is the SynthGen sample interface contract

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

- `render_only`: locally qualified with Open3D and YOLOZU intake
- `bg_only_inpaint`: only the deterministic `internal_stub` is locally
  qualified; a real external provider needs separate evidence
- `appearance_only_conditioned`: internal experimental only

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

For `bg_only_inpaint`, the interface contract is explicit:

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
- if you rewrite asset paths to dataset-root-relative paths during export, the predictions artifact may live outside `shards/`
- generator-side metadata can grow, but breaking contract changes require a new schema version

Practical placement rule:

- keep `predictions_synthgen.json` under `shards/` when you want the safest default with shard-relative paths
- move it elsewhere only after rewriting paths and verifying one overlay + one eval run

## Verified repo-to-repo probe

The current probe is one command from the YOLOZU checkout:

```bash
./.venv/bin/python tools/smoke_synthgen.py \
  --synthgen-repo ../YOLOZU-synthgen \
  --output-dir /tmp/yolozu-synthgen-qualification
```

It creates five Open3D samples, applies `bg_only_inpaint` to a selected sample,
fails closed on QA rejection, compares renderer-owned truth byte-for-byte,
exports and validates the shard, executes both generator and YOLOZU loaders,
renders an overlay, and runs an oracle interface self-check. The output root
must be fresh; the command never deletes an existing generated handoff.

Use `--mode render_only` for the other public mode. Use `--synthgen-python` when
the generator environment is not under `.venv312` or `.venv`.

Python consumers can use the shipped adapters directly:

```python
from torch.utils.data import DataLoader
from yolozu.data.synthgen_shard_dataset import (
    SynthGenShardDataset,
    collate_synthgen_batch,
)

dataset = SynthGenShardDataset("/path/to/export", schema_id="animal_v1")
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_synthgen_batch)
batch = next(iter(loader))
```

For agents, inspect `tools/manifest.json` entry `smoke_synthgen`, use `--help`,
read `smoke_synthgen_summary.json`, and stop if `status != "ok"`,
`qa_report.accepted != true`, or any `truth_equal` field is false.

The pinned run, hashes, provider/privacy boundary, rejected path, and measured
results are in
[`reports/synthgen_handoff_2026-07-28.md`](../reports/synthgen_handoff_2026-07-28.md).

## Handoff checks to run in YOLOZU

1. Validate interface contract rows:

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

4. Run the bundled end-to-end smoke on an existing export:

```bash
python3 tools/smoke_synthgen.py \
  --dataset-root /path/to/synthgen_dataset \
  --predictions /path/to/synthgen_dataset/shards/predictions_synthgen.json \
  --output-dir reports
```

Recommended reading order for operators:

1. `validate_synthgen_contract.py`: fail fast on schema/path issues
2. `render_synthgen_overlay.py`: visually confirm one sample before batch eval
3. `eval_synthgen.py`: generate the metrics report
4. `smoke_synthgen.py`: bundle the three steps into one reproducible probe

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
- a fresh cross-repo run additionally records generator/YOLOZU commits, strict
  QA, truth equality, loader summaries, and artifact hashes

## Versioning rule

- add optional fields freely
- add new `schema_id` values freely
- do not change required dtype / shape / range semantics inside `schema_version = "1"`
- if the generator needs breaking changes, version the interface contract first in YOLOZU, then enable the new version in adapters and docs
