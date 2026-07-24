# YOLO26 inference adapters (per bucket): ONNXRuntime + TensorRT

This note documents the **per-bucket** inference/export entrypoints used by the YOLO26 tooling (`yolo26n/s/m/l/x`).

Goal: produce **five** YOLOZU-compatible predictions JSON artifacts (one per bucket) with **pinned preprocessing/output**
so evaluation is apples-to-apples.

Protocol reference: `docs/yolo26_eval_protocol.md`

## Interface contract (what you must export)

Each bucket produces a YOLOZU predictions JSON file:

- Wrapper shape (recommended):
  - `{ "predictions": [...], "meta": { ... } }`
- Entry shape:
  - `{ "image": "/abs/or/rel/path.jpg", "detections": [ ... ] }`
- Detection shape (minimum):
  - `class_id: int`
  - `score: number`
  - `bbox: { cx, cy, w, h }` (normalized `[0,1]`, `cxcywh`)

Validate anytime:

```bash
python3 tools/validate_predictions.py reports/pred_trt_yolo26n.json --strict
```

## Recommended artifact layout

Keep paths bucket-friendly:

```text
models/
  yolo26n.onnx
  yolo26s.onnx
  yolo26m.onnx
  yolo26l.onnx
  yolo26x.onnx
engines/
  yolo26n_fp16.plan
  ...
reports/
  pred_onnxrt_yolo26n.json
  pred_trt_yolo26n.json
  ...
```

## ONNXRuntime adapter (CPU)

Single bucket (repo users):

```bash
python3 tools/export_predictions_onnxrt.py \
  --dataset /path/to/coco-yolo \
  --onnx models/yolo26n.onnx \
  --combined-output output0 \
  --combined-format xyxy_score_class \
  --boxes-scale abs \
  --min-score 0.0 \
  --topk 300 \
  --wrap \
  --output reports/pred_onnxrt_yolo26n.json
```

Multi-bucket shortcut: reuse the TRT pipeline runner and disable TRT steps:

```bash
python3 tools/run_trt_pipeline.py \
  --dataset /path/to/coco-yolo \
  --onnx-template 'models/{bucket}.onnx' \
  --skip-build --skip-trt --skip-parity --skip-latency --skip-benchmark
```

pip users (optional): `yolozu onnxrt export ...` (requires `pip install 'yolozu[onnxrt]'`).

## TensorRT adapter (GPU)

Single bucket (export predictions only; engine must already exist):

```bash
python3 tools/export_predictions_trt.py \
  --dataset /path/to/coco-yolo \
  --engine engines/yolo26n_fp16.plan \
  --combined-output output0 \
  --combined-format xyxy_score_class \
  --boxes-scale abs \
  --min-score 0.0 \
  --topk 300 \
  --wrap \
  --output reports/pred_trt_yolo26n.json
```

End-to-end multi-bucket (build engine → export → parity → eval → latency → benchmark):

```bash
python3 tools/run_trt_pipeline.py \
  --dataset /path/to/coco-yolo \
  --onnx-template 'models/{bucket}.onnx' \
  --engine-template 'engines/{bucket}_{precision}.plan' \
  --precision fp16
```

Details: `docs/tensorrt_pipeline.md`

## Evaluation (suite)

Once you have 5 predictions files (any names are OK as long as the glob matches):

```bash
python3 tools/eval_suite.py \
  --protocol yolo26 \
  --dataset /path/to/coco-yolo \
  --predictions-glob 'reports/pred_*_yolo26*.json' \
  --output reports/eval_suite.json
```
