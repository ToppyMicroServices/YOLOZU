# External inference backends (C++ / Rust / anything) and YOLOZU integration

YOLOZU is an evaluation + tooling harness. You can run inference **inside this repo** (PyTorch adapter, ONNXRuntime, TensorRT),
or you can run inference **elsewhere** (C++/Rust/mobile/edge) and bring results back as a `predictions.json`.

The key design is: **bring your own inference**, but keep a stable **predictions interface contract** for outputs.

For copy-paste repo-checkout paths from Ultralytics, Detectron2, MMDetection,
and YOLOX through strict validation and a common evaluation report, use
[`byop_quickstarts.md`](byop_quickstarts.md). Its schema-only smoke commands
are CI-checked and explicitly separated from real third-party runtime evidence.

## Predictions-interface-contract-first workflow (recommended)

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
- YOLO runtime (manifest-bounded inputs): `python3 tools/export_predictions_yolo_runtime.py --model yolo11n.pt --dataset data/coco-yolo --split val2017 --max-images 8 --protocol nms_applied --wrap --output reports/pred_yolo_runtime.json`
- YOLOX backend wrapper: `python3 -m yolozu export --backend yolox --dataset data/coco-yolo --exp /path/to/yolox_exp.py --weights /path/to/yolox_ckpt.pth --imgsz 640 --score-thr 0.01 --nms-iou 0.65 --output reports/pred_yolox.json --force`
- ONNXRuntime (exported `.onnx`): `python3 tools/export_predictions_onnxrt.py ...`
- TorchScript (exported `.torchscript` / `.ts`): `python3 tools/export_predictions_torchscript.py --dataset data/smoke --split val --model /abs/path/model.torchscript --output reports/pred_torchscript.json --wrap`
- ExecuTorch runtime output decode (exported `.pte` plus runtime JSON): `python3 tools/export_predictions_executorch.py --dataset data/smoke --split val --model /abs/path/model.pte --runtime-output-json reports/executorch_runtime_outputs.json --output reports/pred_executorch.json --wrap`
- OpenCV DNN (single-backend UX): `python3 -m yolozu export --backend opencv-dnn --onnx path/to/model.onnx --dataset data/coco-yolo --imgsz 640 --decode auto --preprocess yolo_letterbox_640 --dump-io reports/opencv_dump_io.json --output reports/pred_opencv_dnn.json --force`
- OpenCV DNN (single-backend script): `python3 tools/export_predictions_opencv_dnn_unified.py --dataset data/coco-yolo --split val2017 --max-images 8 --onnx path/to/model.onnx --imgsz 640 --decode auto --preprocess yolo_letterbox_640 --dump-io reports/opencv_dump_io.json --output reports/pred_opencv_dnn.json`
- OpenCV DNN (YOLO-style heads): `python3 tools/export_predictions_opencv_dnn.py ...`
- OpenCV DNN (YOLO-style heads via canonical CLI): `python3 -m yolozu export --backend opencv-dnn-yolo --onnx path/to/model.onnx --dataset data/coco-yolo --imgsz 640 --score-thr 0.25 --output reports/pred_opencv_dnn_yolo.json --force`
- OpenCV DNN (RT-DETR decode, no NMS):
  - Direct script: `python3 tools/export_predictions_opencv_dnn_rtdetr.py --dataset data/coco-yolo --onnx path/to/rtdetr.onnx --imgsz 640 --score-thr 0.01 --output reports/pred_rtdetr_opencv_dnn.json`
  - Canonical CLI: `python3 -m yolozu export --backend opencv-dnn-rtdetr --onnx path/to/rtdetr.onnx --dataset data/coco-yolo --imgsz 640 --score-thr 0.01 --output reports/pred_rtdetr_opencv_backend.json --force`
- TensorRT (exported `.plan`): `python3 tools/export_predictions_trt.py ...`
- Full TRT pipeline (engine build → export → parity → eval → latency): `python3 tools/run_trt_pipeline.py ...`

**Static input shapes (ORT/TRT) 注意**: 多くのエクスポート済み ONNX は入力が固定（例: 1×3×64×64）。ONNXRuntime/TensorRT ではモデルが宣言する入力サイズに合わせたテンソルのみ受け付けるため、異なる解像度（例: 640×640）を与えると次元エラーになります。解像度を変えたい場合は動的軸付き ONNX を再エクスポートするか、モデル側を再エクスポートしてください。

YOLO26 per-bucket entrypoints (n/s/m/l/x): `docs/yolo26_inference_adapters.md`

These are the fastest way to iterate in **research/eval**. For production, you might prefer C++/Rust inference.

## Production path: C++ / Rust inference

The main benefit of YOLOZU is you can incrementally migrate:

- Research/Eval: Python (pip) + Docker on GPU (Runpod)
- Production: C++ (TensorRT official path) and/or Rust (ONNXRuntime)
- Verification: parity + predictions interface contract validation stays the same

### Production lane interface contract

Treat the C++/Rust runner as the owner of model execution, and YOLOZU as the owner of validation, evaluation, and report handoff.
The stable handoff is `predictions.json`; any runtime is acceptable if it emits the same predictions interface contract.

Inputs:
- model/runtime inputs owned by your inference project (`.onnx`, TensorRT engine, TorchScript, camera frame, batch file, or device stream)
- image identity preserved in each prediction item as `image`
- optional dataset root or manifest only when YOLOZU evaluation needs labels

Outputs:
- `predictions.json` as the only required artifact
- optional runtime metadata in a wrapper-owned report, not as a substitute for schema-valid predictions
- YOLOZU evaluation/parity reports written under `reports/`

Error behavior:
- return nonzero for missing arguments, runtime initialization failures, unreadable inputs, invalid decode settings, or write failures
- do not report success after writing a partial or schema-invalid `predictions.json`
- report optional runtime gaps as skipped/missing-runtime in the wrapper or benchmark report instead of silently passing

### Bundled exporter execution evidence

The bundled YOLOX, YOLO-runtime, Detectron2, and MMDetection exporters distinguish
schema-only dry runs from completed runtime inference in wrapped
`meta.extra` metadata:

- Both real and dry-run artifacts declare entry `schema_version: 2`; wrapped
  output also declares wrapper `schema_version: 1`. Strict validation should
  not emit legacy-version migration warnings for newly generated output.
- `dry_run`: whether runtime execution was intentionally skipped
- `execution_status`: `dry_run` or `completed`
- `runtime_executed`: `false` for a dry run and `true` after successful non-dry inference
- `inference_calls`: `0` for a dry run and greater than zero after successful non-dry inference
- `runtime_error`: absent or empty for a successful non-dry run
- `model_provenance`: model/config/checkpoint paths or names, with SHA-256 values for local files

For the YOLO runtime exporter, the default source is the exact ordered list of
images selected from the dataset manifest. `--max-images N` truncates that list
before it is passed to the runtime, so it bounds actual inference rather than
only the output metadata. Zero selected inputs fail before model
initialization.

An explicit `--source` accepts one local image file or a directory. Directory
images are expanded recursively in sorted order. `--source` cannot be combined
with `--max-images`; use one selection policy per invocation. In wrapped output,
`source_mode`, `selected_inputs`, `selected_input_count`, and `result_count`
record the policy and cardinality. A non-dry run succeeds only when the runtime
returns exactly one ordered result for every selected input; mismatches return
nonzero without writing a new predictions artifact.

Non-dry YOLOX requires both an existing `--exp` file and an existing `--weights`
file. Detectron2 requires existing `--config` and `--weights` files, while
MMDetection requires existing `--config` and `--checkpoint` files. If a
prerequisite is missing, the runtime cannot initialize, an input image cannot
be read, or inference fails, the exporter exits nonzero without writing a new
predictions artifact.

An empty `detections` list is not by itself evidence of a skipped runtime. It is
a valid result when inference ran and found no detections; use
`runtime_executed`, `inference_calls`, and `execution_status` to distinguish that
case from a dry-run placeholder.

`tools/audit_backend_support.py --require-non-dry` succeeds only when at least
one backend selected with `--non-dry-backend` produces verified execution
evidence. The report records per-backend `execution_evidence_error` and the
accepted backend names in `verified_non_dry_backends`. For the YOLO runtime,
the audit also rejects missing or inconsistent selected-input/result
cardinality evidence.

Schema validation:

```bash
python3 tools/validate_predictions.py /path/to/predictions.json --strict
```

Report handoff:

```bash
yolozu eval-coco --dataset /path/to/coco-yolo --predictions /path/to/predictions.json --output reports/external_eval.json
python3 tools/check_predictions_parity.py --reference reports/pred_torch.json --candidate /path/to/predictions.json > reports/external_parity.json
```

`eval-coco` validates predictions strictly before scoring. Use `--dry-run` for
validation/conversion without `pycocotools`. Use `--repair` only when you
intentionally accept legacy coercion; every repair is recorded in report
`warnings`. With `--max-images N`, known predictions outside the deterministic
dataset subset are counted and excluded, while images unknown to the full
dataset still fail. The typed in-process equivalent is documented in
[`python_api.md`](python_api.md).

### C++ template (submodule-ready)

See `examples/infer_cpp/` for a minimal, self-contained CMake project that is intended to be **extractable into a separate repo**
and added back as a git submodule later.

It focuses on:
- a small CLI surface
- producing YOLOZU-compatible `predictions.json`
- being easy to build inside Docker images that already contain TensorRT / ONNXRuntime headers + libs

### Rust template (submodule-ready)

See `examples/infer_rust/` for a minimal Rust starter. It keeps a no-deps **stub** binary as the default predictions interface contract path, and also provides an
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
