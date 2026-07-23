# Predictions JSON schema (entry v2)

This document defines the stable Predictions JSON interface contract used by YOLOZU evaluation tools.
The goal is simple: run inference anywhere, then compare results fairly in one place.

## Versioning

- Wrapped payload `schema_version`: integer (current wrapper policy: `1`)
- Entry-level `schema_version`: integer (current entry policy: `2`)
- Backward-compatible additions are allowed (new optional fields)
- Breaking changes require bumping `schema_version`
- Lifecycle/migration policy: `docs/schema_governance.md`
- Hard/Soft boundary: `docs/predictions_interface_contract_policy.md`

## Allowed top-level shapes

### Shape A: array of entries

```json
[
  {
    "image": "path/or/name.jpg",
    "detections": [
      {
        "class_id": 0,
        "score": 0.9,
        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
      }
    ]
  }
]
```

### Shape B: wrapped object

```json
{
  "schema_version": 1,
  "predictions": [
    {
      "schema_version": 2,
      "image": "path/or/name.jpg",
      "detections": [
        {
          "class_id": 0,
          "score": 0.9,
          "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
        }
      ]
    }
  ],
  "meta": {
    "adapter": "rtdetr_pose",
    "images": 1
  }
}
```

### Optional runtime-execution evidence in Shape B

Runtime-execution evidence is optional wrapper metadata, not a hard invariant
for every predictions artifact. The bundled YOLOX, YOLO-runtime, Detectron2,
and MMDetection exporters place the following fields in `meta.extra`, and
`tools/audit_backend_support.py` verifies them:

- Dry run: `dry_run=true`, `execution_status="dry_run"`,
  `runtime_executed=false`, and `inference_calls=0`.
- Successful non-dry run: `dry_run=false`,
  `execution_status="completed"`, `runtime_executed=true`,
  `inference_calls>0`, and no `runtime_error`.
- `model_provenance`: a nonempty object identifying the model, exp, config,
  weights, or checkpoint; local files include SHA-256 values.

The YOLO-runtime exporter additionally records `source_mode`,
`selected_inputs`, `selected_input_count`, and `result_count`. A completed
non-dry artifact requires
`selected_input_count == result_count == inference_calls == len(predictions)`.
For a dry run, `result_count` and `inference_calls` are zero while
`selected_input_count == len(predictions)` describes the planned inputs.

These fields distinguish schema-valid dry-run placeholders from completed
runtime inference. A completed artifact may still contain an empty
`detections` list when the model found no detections.

### Shape C: map (`image -> detections`)

```json
{
  "path/or/name.jpg": [
    {
      "class_id": 0,
      "score": 0.9,
      "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
    }
  ]
}
```

## Image keys (join behavior)

Each entry must include `image` (string). In YOLOZU this is primarily a join key
for matching predictions to dataset records.

Current evaluator behavior:

1. Try exact string match against dataset record `image`
2. If not found, try basename match (e.g. `000000123.jpg`)

Recommendations:

- Prefer basename-style keys for portability across machines/OS path separators
- Keep basenames unique per evaluated split
- Do not treat prediction file location as an implicit base path for `image`

## Entry schema migration

- Legacy entries without `schema_version` are treated as v1 and migrated to v2 during canonicalization.
- Supported migration window is one generation: `v1 -> v2`.
- Entries declaring unsupported future versions are rejected.

## Detection fields

Required:

- `class_id` (int)
- `score` (float)
- `bbox` (`{cx, cy, w, h}`)

Optional examples:

- `bbox_abs`
- `mask` / `mask_path`
- `depth` / `depth_path`
- `rot6d`, `R`, `t_xyz`
- `k_delta`, `k_prime`
- `keypoints`

### Keypoints formats

Recommended (YOLO pose-style):

- flat list in normalized coords: `[x1,y1,v1,x2,y2,v2,...]`
- `v` visibility: `0/1/2`

Also accepted by some tools:

- object list: `[{x,y,v?}, ...]`

## BBox format responsibility split

Canonical protocol outputs use `bbox` with keys `{cx, cy, w, h}`.
Evaluation and parity tools can also consume explicit raw forms when the
format is provided through `--bbox-format` or `bbox.format`.

Supported formats:

- `cxcywh_norm` (canonical for protocol-based eval)
- `cxcywh_abs`
- `xywh_abs`
- `xyxy_abs`

Tool expectations:

- `tools/eval_coco.py` / `tools/eval_suite.py` default to `cxcywh_norm`
- `tools/check_predictions_parity.py` defaults to `auto` and accepts
  `bbox.format` when present
- YOLO26 protocol requires `cxcywh_norm` (`docs/yolo26_eval_protocol.md`)
- `tools/export_predictions.py --adapter rtdetr_pose` emits `cxcywh_norm`

## Masks (PNG) and path resolution

There are two related artifacts:

1. detection-style predictions (`{image, detections}`)
2. instance-seg predictions (`{image, instances}`), validated by
   `docs/schemas/instance_segmentation_predictions.schema.json`

### `mask` vs `mask_path`

When both are accepted by a tool:

- `mask` is primary
- `mask_path` is compatibility alias

### PNG requirements (instance-seg)

For `eval-instance-seg`, each predicted mask should:

- be 2D (PNG recommended)
- follow standard image coordinates (origin top-left; `y` down, `x` right)
- match GT mask size exactly (`H, W`)
- use non-zero as foreground

### Relative path resolution

When loading mask files:

1. If `--pred-root` is provided, resolve relative paths under it
2. Otherwise resolve under predictions JSON directory

For reproducibility, prefer explicit absolute `--pred-root`.

## Units and coordinate-system notes

YOLOZU does not convert geometry units automatically.

### Intrinsics (`intrinsics` / `K` / `K_gt`)

- interpreted as `(fx, fy, cx, cy)` in pixels
- must correspond to the same image coordinate system used for bbox/offsets
- after resize/letterbox, provide post-transform intrinsics

### `bbox` + `offsets`

- translation recovery uses bbox center in pixels
- if `--bbox-format cxcywh_norm`, bbox is converted using `image_size` / `image_hw`
- `offsets` are treated as pixel offsets `(du, dv)`

### `log_z` / `z`

- used as depth / translation `z`
- unit is whatever your dataset uses (meters, millimeters, ...)
- recommendation: meters for interoperability

### `k_delta` / `k_prime`

`k_delta = [dfx, dfy, dcx, dcy]` corrects baseline intrinsics:

- `fx' = fx * (1 + dfx)`
- `fy' = fy * (1 + dfy)`
- `cx' = cx + dcx`
- `cy' = cy + dcy`

This is correction, not full intrinsics estimation.

### Resize/letterbox intrinsics rule of thumb

If resize scale is `(s_x, s_y)` and pad is `(p_x, p_y)`:

$$
fx' = fx\,s_x,\quad fy' = fy\,s_y,\quad cx' = cx\,s_x + p_x,\quad cy' = cy\,s_y + p_y
$$

## Schema files

- Detection schema: `schemas/predictions.schema.json`
- Instance-seg schema: `docs/schemas/instance_segmentation_predictions.schema.json`
