# TensorRT FP16/INT8 pipeline (YOLO26)

This repo stays Apache-2.0-only, so the TensorRT build and inference steps are scripted but rely on your local TensorRT installation. The goal is a reproducible engine build, plus parity validation against ONNX outputs.

## Canonical export route (PyTorch → ONNX → TensorRT)

For in-repo PyTorch models (notably `rtdetr_pose/`), use the wrapper tool that pins the intermediate ONNX artifact and drives `trtexec` via `tools/build_trt_engine.py`:

```bash
python3 tools/export_trt.py \
  --config rtdetr_pose/configs/base.json \
  --checkpoint /path/to/checkpoint.pt \
  --image-size 320 \
  --onnx models/rtdetr_pose.onnx \
  --opset 17 \
  --dynamic-hw \
  --engine engines/rtdetr_pose_fp16.plan \
  --precision fp16 \
  --min-shape 1x3x320x320 \
  --opt-shape 1x3x640x640 \
  --max-shape 1x3x960x960
```

Artifacts:
- ONNX: `models/rtdetr_pose.onnx` + `models/rtdetr_pose.onnx.meta.json`
- Engine: `engines/rtdetr_pose_fp16.plan` + `engines/rtdetr_pose_fp16.plan.meta.json`

Engine metadata includes best-effort `nvidia-smi` (GPU/driver/CUDA) and `trtexec --version` (TensorRT) for reproducibility.

## RTDETRPose parity + benchmark (torch/onnxrt/trt)

Once you have a checkpoint + ONNX (and optionally a TensorRT engine), you can run a single report that:
- compares derived `score` and `bbox` stats across backends
- benchmarks latency/FPS per backend (best-effort VRAM snapshots via `nvidia-smi`)

```bash
python3 tools/rtdetr_pose_backend_suite.py \
  --config rtdetr_pose/configs/base.json \
  --checkpoint /path/to/checkpoint.pt \
  --onnx models/rtdetr_pose.onnx \
  --engine engines/rtdetr_pose_fp16.plan \
  --backends torch,onnxrt,trt \
  --device cuda \
  --image-size 320 \
  --samples 2 \
  --warmup 20 \
  --iterations 200 \
  --output reports/rtdetr_pose_backend_suite.json
```

## Runpod shortcut (recommended)

If you're developing on macOS, keep GPU/TensorRT work on Runpod (or any Linux+NVIDIA machine):
- Docs + Docker skeleton: `deploy/runpod/README.md`
- Local/GPU split preflight: `python3 tools/gpu_validation_preflight.py --output reports/gpu_validation_preflight.json`
- One-command runner: `python3 tools/run_trt_pipeline.py ...`
- End-to-end `rtdetr_pose` export + parity + benchmark: `python3 tools/run_rtdetr_pose_backend_suite.py ...`

For YOLO26 per-bucket adapter entrypoints (ONNXRuntime + TensorRT), see:
- `docs/yolo26_inference_adapters.md`

Example (writes a self-contained run folder under `runs/`):

```bash
python3 tools/run_rtdetr_pose_backend_suite.py \
  --config rtdetr_pose/configs/base.json \
  --checkpoint /path/to/checkpoint.pt \
  --device cuda \
  --precision fp16 \
  --dynamic-hw \
  --export-image-size 320 \
  --suite-image-size 640
```

## 1) Export ONNX (end2end/no-NMS)

Use your preferred exporter (YOLO-family runtime or custom) and keep NMS disabled so parity is meaningful:

```bash
yolo export model=yolo26n.pt format=onnx opset=17 imgsz=640 nms=False
```

## 2) Build TensorRT engine (FP16)

Use the build wrapper to generate the engine and a metadata JSON:

```bash
python3 tools/build_trt_engine.py \
  --onnx yolo26n.onnx \
  --engine engines/yolo26n_fp16.plan \
  --precision fp16 \
  --input-name images \
  --min-shape 1x3x640x640 \
  --opt-shape 1x3x640x640 \
  --max-shape 1x3x640x640 \
  --timing-cache engines/timing.cache \
  --meta-output reports/trt_engine_yolo26n_fp16.json
```

The metadata JSON captures the full `trtexec` command, git head, and engine path to make builds reproducible.

If `trtexec` is not available, the builder can fall back to the TensorRT Python API:

```bash
python3 tools/build_trt_engine.py --builder python ...
```

This path requires the TensorRT Python package (e.g. `pip install nvidia-tensorrt`) and CUDA bindings (`pycuda` or `cuda-python`).

If neither `trtexec` nor the Python TensorRT package is available, `--builder auto` will fail fast with a clear error and write the
failure reason into the `--meta-output` JSON for debugging/repro capture.

## 3) Build TensorRT engine (INT8, optional)

INT8 requires a calibration cache. If your TRT workflow uses a custom calibrator, you can still use the wrapper to generate a calibration image list and pass the cache path to `trtexec`:

```bash
python3 tools/build_trt_engine.py \
  --onnx yolo26n.onnx \
  --engine engines/yolo26n_int8.plan \
  --precision int8 \
  --calib-cache engines/yolo26n_int8.cache \
  --calib-dataset /path/to/coco-yolo \
  --calib-images 500 \
  --calib-list-output reports/calib_images.txt \
  --meta-output reports/trt_engine_yolo26n_int8.json
```

## 4) Export predictions (TensorRT)

Use your TensorRT inference pipeline to export YOLOZU predictions JSON. If you implement the Apache-2.0-friendly skeleton in [tools/export_predictions_trt.py](tools/export_predictions_trt.py), point it at the engine:

```bash
python3 tools/export_predictions_trt.py \
  --dataset /path/to/coco-yolo \
  --engine engines/yolo26n_fp16.plan \
  --wrap \
  --output reports/pred_trt_yolo26n.json
```

Note on input size: ONNX/TRT execution is usually shape-pinned unless you export with dynamic shapes / profiles. `export_predictions_trt.py` will try to infer `imgsz` from the engine input shape when possible; pass `--imgsz <N>` to override (especially for dynamic engines).

## 5) Parity vs ONNX

Use the ONNXRuntime exporter to generate the reference JSON, then compare with the TRT JSON:

```bash
python3 tools/export_predictions_onnxrt.py \
  --dataset /path/to/coco-yolo \
  --onnx yolo26n.onnx \
  --combined-output output0 \
  --combined-format xyxy_score_class \
  --boxes-scale abs \
  --min-score 0.0 \
  --output reports/pred_onnxrt_yolo26n.json

python3 tools/check_predictions_parity.py \
  --reference reports/pred_onnxrt_yolo26n.json \
  --candidate reports/pred_trt_yolo26n.json \
  --image-size 640 \
  --iou-thresh 0.99 \
  --score-atol 1e-4 \
  --bbox-atol 1e-4
```

If parity fails with large mismatches, confirm that the same preprocessing (letterbox, RGB ordering) and output format (`cxcywh_norm`) are used in both paths.

## 6) Measure latency/FPS

Once engines are built, record latency/FPS in the benchmark harness:

```bash
python3 tools/benchmark_latency.py --config configs/benchmark_latency_example.json
```
