# SynthGen sample interface contract (v1)

This document defines the **YOLOZU intake interface contract** for synthetic samples generated in external repos (for example `YOLOZU-synthgen`).

YOLOZU scope is intentionally limited to:
- interface-contract validation
- dataset loading/adaptation
- visualization/evaluation

Generation logic is out of scope.

## Versioning policy

- Contract id: `synthgen_sample`
- Current version: `schema_version = "1"` (string in sample payload)
- Backward-compatible changes:
  - adding optional fields
  - adding new `schema_id` values
- Breaking changes:
  - changing required fields
  - changing required dtype/shape/range semantics
  - changing visibility semantics (`vis`) or coordinate interpretation

Breaking changes require a new schema version and a migration note in docs.

## Required fields

All fields below are required in v1.

| Field | Type | Shape / Range | Notes |
|---|---|---|---|
| `image` | `uint8` array | `[H,W,3]` | RGB |
| `depth_ndc` | `float32` array | `[H,W]`, values in `[0,1]` | Normalized depth |
| `inst_id` | `uint32` array | `[H,W]` | Instance id map |
| `sem_id` | `uint16` array | `[H,W]` | Semantic id map |
| `kpts2d` | `float32` array | `[N_inst,K,3]` | `(u,v,vis)` |
| `prompt` | `string` | non-empty | Generation prompt |
| `scene_spec` | JSON object or object-string | object | Scene metadata |
| `schema_id` | `string` | non-empty | e.g. `animal_v1`, `mechanical_v1` |
| `schema_version` | `string` | non-empty | Contract version |
| `asset_ids` | `list[string]` | list | Asset references |
| `inst_map` | JSON object or object-string | object | Instance index ↔ `inst_id` |

## Optional fields (v1+)

- `bbox2d_visible`: `float32[N_inst,4]`, half-open absolute-pixel `xyxy`
- `kpts3d_object`: `float32[N_inst,K,3]`
- `pose_obj2cam`: `float32[4,4]` or `float32[N_inst,4,4]`
- `bbox2d_visible` is derived from the renderer-owned `inst_id` mask.
- `kpts3d_object` uses object coordinates.
- `pose_obj2cam = world_to_camera x object_to_world` using row-major matrices.
- Use `pad_instance_labels=True` to pad all instance-aligned labels during collation; `pad_keypoints` remains a backward-compatible alias.

## Visibility (`kpts2d[...,2]`) semantics

- `0`: not labeled / absent
- `1`: labeled but not visible (occluded/truncated)
- `2`: labeled and visible

MVP training/eval policy in YOLOZU uses `vis == 2` for keypoint regression metrics.

## Missing-data policy

- Required fields must always be present.
- Optional fields may be omitted.
- For no-instance samples:
  - `kpts2d` should be `shape=[0,K,3]`
  - `inst_map` should still be a valid (possibly empty) JSON object

## Coordinate / unit rules

- `kpts2d` coordinates are pixel coordinates in the same image frame as `image`.
- `depth_ndc` is normalized depth in `[0,1]`, not metric depth.
- `inst_id` / `sem_id` share the exact same `H,W` as `image`.

## Machine-readable schema and runtime validator

- JSON Schema: `schemas/synthgen_sample.schema.json`
- Python TypedDict + runtime validation: `yolozu/contracts/synthgen.py`

## Canonical ingestion path

For a fresh sibling-repository qualification, use the one-command wrapper:

```bash
./.venv/bin/python tools/smoke_synthgen.py \
  --synthgen-repo ../YOLOZU-synthgen \
  --output-dir /tmp/yolozu-synthgen-qualification
```

1. Validate interface contract:

```bash
python3 tools/validate_synthgen_contract.py \
  --input /path/to/synthgen_dataset/shards/train_000.jsonl \
  --max-samples 200
```

2. Load via shard adapter:

```bash
python3 tools/render_synthgen_overlay.py \
  --dataset-root /path/to/synthgen_dataset \
  --schema-id animal_v1 \
  --sample-index 0 \
  --output reports/synthgen_overlay.png
```

3. Evaluate predictions:

```bash
python3 tools/eval_synthgen.py \
  --dataset-root /path/to/synthgen_dataset \
  --predictions /path/to/synthgen_dataset/shards/predictions_synthgen.json \
  --schema-id animal_v1 \
  --output reports/synthgen_eval.json
```

If the predictions artifact reuses shard-relative paths such as `../samples/...`, keep it under `shards/` so the relative base matches the exported shard rows.
