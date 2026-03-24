# ONNX export parity check (predictions JSON)

Goal: ensure an ONNX export produces **numerically equivalent** predictions to a PyTorch reference (within tolerances), using the same protocol/bbox format.

## Produce two predictions JSONs

1) **Reference** (PyTorch): run your reference inference path and export YOLOZU predictions JSON.
2) **Candidate** (ONNXRuntime): run your ONNX model and export YOLOZU predictions JSON (e.g. `tools/export_predictions_onnxrt.py`).

Important:
- Use the same preprocessing (imgsz=640 + letterbox) and output format (`cxcywh_norm`).
- Do not apply NMS (e2e/no-NMS protocol).
 

## YOLO26 concrete flow

This is a concrete, end2end/no-NMS parity flow for YOLO26 models exported from a YOLO-family runtime.

### 1) Export ONNX (end2end/no-NMS)

Export with NMS disabled and fixed input size 640:

```bash
yolo export model=yolo26n.pt format=onnx opset=17 imgsz=640 nms=False
```

If your environment prefers Python, you can also call the runtime export API. Ensure
`nms=False` (or `end2end=True`) so the export keeps the raw/combined output without NMS.

### 2) Reference predictions (PyTorch)

Run the PyTorch reference using the YOLO runtime and export YOLOZU JSON (bbox format is `cxcywh_norm`):

```bash
python3 tools/export_predictions_yolo_runtime.py \
  --model yolo26n.pt \
  --dataset /path/to/coco-yolo \
  --image-size 640 \
  --conf 0.001 \
  --iou 0.7 \
  --max-det 300 \
  --end2end \
  --output /path/to/pred_ref.json
```

### 3) ONNXRuntime predictions

Inspect ONNX output names (pick one of the flows below):

```bash
python3 -c "import onnx; m=onnx.load('yolo26n.onnx'); print([o.name for o in m.graph.output])"
```

**Combined output** (preferred for parity):

```bash
python3 tools/export_predictions_onnxrt.py \
  --dataset /path/to/coco-yolo \
  --onnx yolo26n.onnx \
  --combined-output output0 \
  --combined-format xyxy_score_class \
  --boxes-scale abs \
  --min-score 0.0 \
  --output /path/to/pred_onnxrt.json
```

**Raw head output + NMS** (use only if you cannot export end2end):

```bash
python3 tools/export_predictions_onnxrt.py \
  --dataset /path/to/coco-yolo \
  --onnx yolo26n.onnx \
  --raw-output output0 \
  --raw-format yolo_84 \
  --raw-postprocess yolo_runtime \
  --boxes-format xyxy \
  --boxes-scale abs \
  --min-score 0.001 \
  --nms-iou 0.7 \
  --topk 300 \
  --output /path/to/pred_onnxrt.json
```

### 4) Parity check

```bash
python3 tools/check_predictions_parity.py \
  --reference /path/to/pred_ref.json \
  --candidate /path/to/pred_onnxrt.json \
  --image-size 640 \
  --iou-thresh 0.99 \
  --score-atol 1e-4 \
  --bbox-atol 1e-4
```

### End2end ONNX models (combined output)

Some end2end exports emit a single output tensor shaped $(1,N,6)$ (or $(N,6)$) with
`[x1, y1, x2, y2, score, class_id]` in input-image coordinates. For these models:

- Use `--combined-output` and `--combined-format xyxy_score_class`.
- Set `--boxes-scale abs` (coordinates are already in pixels).
- Use `--min-score 0.0` to avoid dropping low-confidence entries that may still
  exist in the PyTorch reference output.

## Run parity checker

```bash
python3 tools/check_predictions_parity.py \
  --reference /path/to/pred_ref.json \
  --candidate /path/to/pred_onnxrt.json \
  --iou-thresh 0.99 \
  --score-atol 1e-4 \
  --bbox-atol 1e-4

# Optional: if image files are not available, use a fixed size
python3 tools/check_predictions_parity.py \
  --reference /path/to/pred_ref.json \
  --candidate /path/to/pred_onnxrt.json \
  --image-size 640 \
  --iou-thresh 0.99 \
  --score-atol 1e-4 \
  --bbox-atol 1e-4
```

If end2end parity is still slightly off, relax tolerances (e.g. $\text{IoU} \ge 0.96$,
`score_atol=5e-3`, `bbox_atol=1e-2`) and document the residual mismatch count in your
report.

### Raw head output + NMS (unified postprocess)

If your export emits raw head output (e.g. shape $(1,84,8400)$), decode and run NMS in
`export_predictions_onnxrt.py`:

```bash
python3 tools/export_predictions_onnxrt.py \
  --dataset /path/to/dataset \
  --onnx /path/to/model.onnx \
  --input-name images \
  --raw-output output0 \
  --raw-format yolo_84 \
  --raw-postprocess yolo_runtime \
  --boxes-format xyxy \
  --boxes-scale abs \
  --min-score 0.001 \
  --nms-iou 0.7 \
  --topk 300 \
  --output /path/to/pred_onnxrt.json
```

This path applies class-aware NMS (matching the YOLO-runtime postprocess) and produces the same
postprocess as the PyTorch reference.

**Note:** even with the YOLO-runtime postprocess, raw-output parity can still diverge
because tiny numerical differences in raw logits ($\le 2.5\times 10^{-3}$ observed)
change NMS decisions. If parity fails with large `missing_match` counts, treat it
as NMS sensitivity rather than a decode bug and consider using the end2end
combined output path for parity checks.

The checker exits non-zero and prints a JSON report if any image/detection mismatches.

## TensorRT parity (PyTorch vs TRT)

If you have a TensorRT engine and a PyTorch reference JSON, you can run an end-to-end
parity check that exports TRT predictions and compares them to the PyTorch reference:

```bash
python3 tools/check_predictions_parity_trt.py \
  --reference /path/to/pred_ref.json \
  --engine /path/to/model.plan \
  --dataset /path/to/coco-yolo \
  --input-name images \
  --boxes-output boxes \
  --scores-output scores \
  --output /path/to/pred_trt.json \
  --image-size 640 \
  --iou-thresh 0.99 \
  --score-atol 1e-4 \
  --bbox-atol 1e-4
```

This tool reuses `export_predictions_trt.py` and then runs `check_predictions_parity.py`.

## End-to-end TRT workflow helper

You can chain build + parity using the helper script:

```bash
python3 tools/run_trt_workflow.py \
  --onnx /path/to/model.onnx \
  --engine /path/to/model.plan \
  --dataset /path/to/coco-yolo \
  --reference /path/to/pred_ref.json \
  --input-name images \
  --input-shape 1x3x640x640 \
  --fp16 \
  --image-size 640
```

Use `--skip-build` or `--skip-parity` to run only one step, and `--dry-run`
to print commands without executing them.

