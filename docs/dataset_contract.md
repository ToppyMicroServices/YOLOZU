# Dataset Contract v1

YOLOZU dataset records use a backend-neutral bbox interface contract.

Preferred stored bbox representation:

```json
{
  "bbox": { "format": "xyxy_abs", "x1": 10, "y1": 20, "x2": 50, "y2": 80 },
  "bbox_xyxy_abs": { "format": "xyxy_abs", "x1": 10, "y1": 20, "x2": 50, "y2": 80 }
}
```

Adapter views may be present on the same label:

```json
{
  "bbox_xywh_abs": { "format": "xywh_abs", "x": 10, "y": 20, "w": 40, "h": 60 },
  "bbox_cxcywh_norm": { "format": "cxcywh_norm", "cx": 0.15, "cy": 0.5, "w": 0.2, "h": 0.6 }
}
```

Rules:
- `xyxy_abs` is the preferred dataset storage form.
- `xywh_abs` is the COCO/export view.
- `cxcywh_norm` is the YOLO-family training/export view.
- DETR/torchvision-style adapters may consume `xyxy_abs` directly.
- Existing legacy records with top-level `cx`, `cy`, `w`, `h` or `bbox: {cx, cy, w, h}` remain accepted.
- `image_hw: [height, width]` or `image_size: {width, height}` is required when converting between absolute and normalized coordinates.

The implementation reference is `yolozu/datasets/dataset_contract.py`.

## Training data flow

Training data uses the same route for every detector-family backend:

```text
raw dataset
  -> DatasetAdapter
  -> YOLOZU Dataset Contract
  -> TrainingBackend
      -> YOLO family
      -> DETR family
```

The DatasetAdapter owns source-format differences such as YOLO `data.yaml`,
`dataset.json` wrappers, COCO JSON, and SynthGen shards. The TrainingBackend
receives Dataset Contract records and selects the backend view it needs:

- YOLO-family backends consume the `cxcywh_norm` adapter view and keep
  letterbox/NMS assumptions explicit in run metadata.
- DETR-family backends consume `xyxy_abs` and keep DETR preprocessing,
  AdamW parameter groups, and NMS-free assumptions explicit in run metadata.

Training summaries expose this route as `training_data_flow` so downstream
automation can verify that a backend did not bypass the Dataset Contract.
