# Dataset Contract v1

YOLOZU dataset records use a backend-neutral bbox contract.

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
