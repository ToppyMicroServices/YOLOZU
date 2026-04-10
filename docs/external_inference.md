# External inference backends (C++ / Rust / anything) and YOLOZU integration

YOLOZU is an evaluation + tooling harness. You can run inference **inside this repo** (PyTorch adapter, ONNXRuntime, TensorRT),
or you can run inference **elsewhere** (C++/Rust/mobile/edge) and bring results back as a `predictions.json`.

The key design is: **bring your own inference**, but keep a stable **contract** for outputs.

## Contract-first workflow (recommended)

1) Produce a YOLOZU predictions JSON artifact:
   - Canonical shape:
     - `{ "predictions": [ { "image": "...", "detections": [ ... ] }, ... ], "meta": { ... } }`
   - Minimal detection schema:
     - `score: number`
     - `bbox: { cx, cy, w, h }` (normalized 0..1, cxcywh)
     - `class_id: int` (recommended; some flows can work without it)

2) Validate locally:

```bash
python3 tools/validate_predictions.py /path/to/predictions.json --strict
```

3) Run evaluation / reports:
- COCO-style: `python3 tools/eval_coco.py ...`
- Suite: `python3 tools/eval_suite.py ...`
- Parity checks (ONNX vs TRT, etc.): `python3 tools/check_predictions_parity.py ...`

## Fast paths already in this repo

- PyTorch adapter (research reference adapter): `python3 tools/export_predictions.py --adapter rtdetr_pose ...`
- YOLOX backend wrapper: `python3 tools/yolozu.py export --backend yolox --dataset data/coco-yolo --exp /path/to/yolox_exp.py --weights /path/to/yolox_ckpt.pth --imgsz 640 --score-thr 0.01 --nms-iou 0.65 --output reports/pred_yolox.json --force`
- ONNXRuntime (exported `.onnx`): `python3 tools/export_predictions_onnxrt.py ...`
- ExecuTorch (exported `.pte`): `python3 tools/export_predictions_executorch.py --dataset data/smoke --split val --model /abs/path/model.pte --output reports/pred_executorch.json --wrap`
- OpenCV DNN (single-backend UX): `python3 tools/yolozu.py export --backend opencv-dnn --onnx path/to/model.onnx --dataset data/coco-yolo --imgsz 640 --decode auto --preprocess yolo_letterbox_640 --dump-io reports/opencv_dump_io.json --output reports/pred_opencv_dnn.json --force`
- OpenCV DNN (single-backend script): `python3 tools/export_predictions_opencv_dnn_unified.py --dataset data/coco-yolo --split val2017 --max-images 8 --onnx path/to/model.onnx --imgsz 640 --decode auto --preprocess yolo_letterbox_640 --dump-io reports/opencv_dump_io.json --output reports/pred_opencv_dnn.json`
- OpenCV DNN (YOLO-style heads): `python3 tools/export_predictions_opencv_dnn.py ...`
- OpenCV DNN (YOLO-style heads via unified CLI): `python3 tools/yolozu.py export --backend opencv-dnn-yolo --onnx path/to/model.onnx --dataset data/coco-yolo --imgsz 640 --score-thr 0.25 --output reports/pred_opencv_dnn_yolo.json --force`
- OpenCV DNN (RT-DETR decode, no NMS):
  - Direct script: `python3 tools/export_predictions_opencv_dnn_rtdetr.py --dataset data/coco-yolo --onnx path/to/rtdetr.onnx --imgsz 640 --score-thr 0.01 --output reports/pred_rtdetr_opencv_dnn.json`
  - Unified wrapper: `python3 tools/yolozu.py export --backend opencv-dnn-rtdetr --onnx path/to/rtdetr.onnx --dataset data/coco-yolo --imgsz 640 --score-thr 0.01 --output reports/pred_rtdetr_opencv_backend.json --force`
- TensorRT (exported `.plan`): `python3 tools/export_predictions_trt.py ...`
- Full TRT pipeline (engine build → export → parity → eval → latency): `python3 tools/run_trt_pipeline.py ...`

**Static input shapes (ORT/TRT) 注意**: 多くのエクスポート済み ONNX は入力が固定（例: 1×3×64×64）。ONNXRuntime/TensorRT ではモデルが宣言する入力サイズに合わせたテンソルのみ受け付けるため、異なる解像度（例: 640×640）を与えると次元エラーになります。解像度を変えたい場合は動的軸付き ONNX を再エクスポートするか、モデル側を再エクスポートしてください。

YOLO26 per-bucket entrypoints (n/s/m/l/x): `docs/yolo26_inference_adapters.md`

These are the fastest way to iterate in **research/eval**. For production, you might prefer C++/Rust inference.

## Production path: C++ / Rust inference

The main benefit of YOLOZU is you can incrementally migrate:

- Research/Eval: Python (pip) + Docker on GPU (Runpod)
- Production: C++ (TensorRT official path) and/or Rust (ONNXRuntime)
- Verification: parity + contract validation stays the same

### C++ template (submodule-ready)

See `examples/infer_cpp/` for a minimal, self-contained CMake project that is intended to be **extractable into a separate repo**
and added back as a git submodule later.

It focuses on:
- a small CLI surface
- producing YOLOZU-compatible `predictions.json`
- being easy to build inside Docker images that already contain TensorRT / ONNXRuntime headers + libs

### Rust template (submodule-ready)

See `examples/infer_rust/` for a minimal Rust starter. It keeps a no-deps **stub** binary as the default contract path, and also provides an
optional ONNXRuntime mode behind a Cargo feature:

```bash
# no-deps stub (always available)
cargo build --release --manifest-path examples/infer_rust/Cargo.toml
examples/infer_rust/target/release/yolozu_infer_rust --out reports/pred_rust_stub.json

# optional ONNXRuntime runner
cargo build --release --features onnxruntime --manifest-path examples/infer_rust/Cargo.toml
examples/infer_rust/target/release/yolozu_infer_rust \
  --mode onnxrt \
  --onnx /abs/path/model.onnx \
  --input-shape 1,3,64,64 \
  --image images/val/000001.jpg \
  --out reports/pred_rust_onnxrt.json
```

Expected production environment for `--features onnxruntime`:
- Linux `x86_64` base image with Rust toolchain.
- Python 3 with `onnxruntime` and `numpy` installed (the optional Rust mode shells out to Python ORT for the forward pass).
- Keep the stub mode as CI baseline, and run ONNXRuntime mode in dedicated runtime images where ORT dependencies are intentionally provisioned.

## Notes

- Keep model weights / datasets out of git.
- Keep inference repos/containers separate if license constraints differ; YOLOZU can still consume the output JSON.
