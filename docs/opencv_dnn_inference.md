# OpenCV-DNN inference exporter (ONNX → predictions.json)

YOLOZU supports generating `predictions.json` using OpenCV’s DNN module (`cv2.dnn`) as an alternative ONNX execution path.

This is useful when:
- you want a lightweight inference dependency surface (OpenCV-only),
- you are targeting environments where ONNXRuntime is undesirable, or
- you want to compare multiple ONNX backends (`onnxrt` vs `opencv_dnn`) using the same postprocessing + validation.

## Tool

- Script: `tools/export_predictions_opencv_dnn.py`
- Manifest id: `export_predictions_opencv_dnn`

### Dry-run (contract wiring)

Dry-run does **not** import OpenCV, and writes schema-valid output with empty detections:

```bash
python3 tools/export_predictions_opencv_dnn.py \
  --dataset data/coco128 \
  --dry-run \
  --output reports/pred_opencv_dnn.json
python3 tools/validate_predictions.py reports/pred_opencv_dnn.json --strict
```

### Real inference (requires OpenCV)

Install OpenCV first:

```bash
python3 -m pip install opencv-python
```

Then run:

```bash
python3 tools/export_predictions_opencv_dnn.py \
  --dataset /path/to/coco-yolo \
  --onnx /abs/path/model.onnx \
  --output reports/pred_opencv_dnn.json
```

## Model output expectations

Current implementation targets YOLOv8-style raw heads:

- `--raw-format yolo_84` (default)
- Output shape compatible with one of:
  - `(1, 84, N)`
  - `(1, N, 84)`
  - `(84, N)`
  - `(N, 84)`
- Interprets:
  - `[:4]` as `cx, cy, w, h`
  - `[4:]` as class scores

If your model exports a different head layout (e.g. YOLOv5-style `85` with objectness), you will need to adapt decoding.

### YOLOv5-style (85 with objectness)

YOLOv5 ONNX exports commonly produce a tensor like `(1, N, 85)`:

- `[0:4]`: `cx, cy, w, h`
- `[4]`: `objectness`
- `[5:]`: class probabilities

Use:

```bash
python3 tools/export_predictions_opencv_dnn.py ... --raw-format yolo_85_obj
```

Scoring uses `score = objectness * class_prob` before thresholding and NMS.

## Backend knobs (optional)

If your OpenCV build supports it, you can try selecting a DNN backend/target:

```bash
python3 tools/export_predictions_opencv_dnn.py ... --dnn-backend cuda --dnn-target cuda_fp16
```

These flags are best-effort; unsupported configurations are ignored by OpenCV.
