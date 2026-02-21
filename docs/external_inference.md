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

- PyTorch adapter (research scaffold): `python3 tools/export_predictions.py --adapter rtdetr_pose ...`
- ONNXRuntime (exported `.onnx`): `python3 tools/export_predictions_onnxrt.py ...`
- OpenCV DNN (exported `.onnx`): `python3 tools/export_predictions_opencv_dnn.py ...`
- OpenCV DNN (RT-DETR decode): `python3 tools/export_predictions_opencv_dnn_rtdetr.py ...`
- TensorRT (exported `.plan`): `python3 tools/export_predictions_trt.py ...`
- Full TRT pipeline (engine build → export → parity → eval → latency): `python3 tools/run_trt_pipeline.py ...`

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

See `examples/infer_rust/` for a minimal Rust starter. It intentionally builds a no-deps **stub** binary by default (contract wiring),
so you can layer in ONNXRuntime/TensorRT/FFI later without changing the output JSON contract.

## Notes

- Keep model weights / datasets out of git.
- Keep inference repos/containers separate if license constraints differ; YOLOZU can still consume the output JSON.
