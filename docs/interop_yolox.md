# YOLOX interop

Use this path if you already train/infer with YOLOX and want contract-first evaluation in YOLOZU.

## 5-minute flow

```bash
yolozu migrate dataset \
  --from coco \
  --coco-root /path/to/coco \
  --split val2017 \
  --output data/coco_yolo_like \
  --mode manifest

python3 tools/yolozu.py export \
  --backend yolox \
  --dataset data/coco_yolo_like \
  --split val2017 \
  --exp /path/to/yolox_exp.py \
  --weights /path/to/yolox_ckpt.pth \
  --imgsz 640 \
  --score-thr 0.01 \
  --nms-iou 0.65 \
  --output reports/pred_yolox.json

python3 tools/validate_predictions.py reports/pred_yolox.json --strict
python3 tools/eval_coco.py \
  --dataset data/coco_yolo_like \
  --split val2017 \
  --predictions reports/pred_yolox.json \
  --protocol nms_applied \
  --classes data/coco_yolo_like/labels/val2017/classes.json \
  --output reports/eval_yolox.json
```

## Important compatibility points

- Exp parameters are captured in `meta.extra.export_settings.exp_params` when exp projection succeeds.
- YOLOX decode is treated as anchor-free grid decode with NMS; use `protocol=nms_applied` by default.
- Preprocess assumptions are stored in `export_settings.preprocessing` (letterbox/normalize/input color).
- `weights_sha256` is stored for reproducibility.
