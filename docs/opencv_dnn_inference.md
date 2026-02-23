# OpenCV-DNN inference exporter (ONNX → predictions.json)

YOLOZU supports generating `predictions.json` using OpenCV’s DNN module (`cv2.dnn`) as an alternative ONNX execution path.

This is useful when:
- you want a lightweight inference dependency surface (OpenCV-only),
- you are targeting environments where ONNXRuntime is undesirable, or
- you want to compare multiple ONNX backends (`onnxrt` vs `opencv_dnn`) using the same postprocessing + validation.

## 5-minute unified path

```bash
python3 tools/yolozu.py export \
  --backend opencv-dnn \
  --onnx /abs/path/model.onnx \
  --dataset /path/to/coco-yolo \
  --split val2017 \
  --imgsz 640 \
  --preprocess rtdetr_resize_640 \
  --decode rtdetr \
  --dump-io reports/opencv_dump_io.json \
  --output reports/pred_opencv.json

python3 tools/validate_predictions.py reports/pred_opencv.json --strict
python3 tools/eval_coco.py --dataset /path/to/coco-yolo --split val2017 --predictions reports/pred_opencv.json --output reports/eval_opencv.json
```

The `*.meta.json` sidecar stores reproducibility metadata (`export_settings`, backend/target/version, decode and preprocessing knobs), and `--dump-io` stores model IO tensor signatures for debugging.

## Tool

- Script: `tools/export_predictions_opencv_dnn.py`
- Manifest id: `export_predictions_opencv_dnn`
- Unified wrapper: `tools/export_predictions_opencv_dnn_unified.py`

## RT-DETR (OpenCV-DNN + decode)

RT-DETR-style models often emit a fixed number of **object queries** (no NMS expected), with outputs like:

- `boxes`: `[B, N, 4]` (often `cxcywh` normalized to 0..1)
- `logits`: `[B, N, C]` (often softmax with a background class)

YOLOZU provides a dedicated exporter that runs OpenCV-DNN and decodes these outputs into `predictions.json` while recording
preprocessing/export settings for parity:

- Script: `tools/export_predictions_opencv_dnn_rtdetr.py`
- Manifest id: `export_predictions_opencv_dnn_rtdetr`

Dry-run (no OpenCV required):

```bash
python3 tools/export_predictions_opencv_dnn_rtdetr.py \
  --dataset data/coco128 \
  --dry-run \
  --output reports/pred_rtdetr_opencv_dnn.json
python3 tools/validate_predictions.py reports/pred_rtdetr_opencv_dnn.json --strict
```

Real inference (requires OpenCV):

```bash
python3 tools/export_predictions_opencv_dnn_rtdetr.py \
  --dataset /path/to/coco-yolo \
  --onnx /abs/path/rtdetr.onnx \
  --imgsz 640 \
  --keep-aspect \
  --boxes-format cxcywh \
  --boxes-scale norm \
  --scores-activation softmax \
  --background-class last \
  --score-thr 0.01 \
  --output reports/pred_rtdetr_opencv_dnn.json
```

Notes:
- The exporter writes a separate `*.meta.json` file containing `export_settings.preprocess` and decode settings.
- `predictions.json` remains schema-valid under `--strict` (no wrapped `meta` contract required).

### Dry-run (contract wiring)

Dry-run does **not** import OpenCV, and writes schema-valid output with empty detections:

```bash
python3 tools/export_predictions_opencv_dnn.py \
  --dataset data/coco128 \
  --dry-run \
  --dump-io reports/pred_opencv_dnn.io.json \
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

Supported selectors in YOLOZU CLI:

- `--dnn-backend opencv|cuda|openvino`
- `--dnn-target cpu|cuda|cuda_fp16|opencl|opencl_fp16`
