# YOLOZU Rust inference template (stub + optional ONNXRuntime)

This folder is a **submodule-ready** starting point for production inference pipelines in Rust.

- By default it builds a tiny **no-deps** binary that writes a schema-valid `predictions.json` stub.
- Optional `onnxruntime` cargo feature enables a minimal ONNXRuntime-backed runner for production wiring checks.
- Replace/extend these paths with your real backend (TensorRT via FFI, custom decode, etc.), keeping the predictions interface contract stable.

## Production lane interface contract

Use this template as the production runner boundary: Rust owns inference, decode, and file writes; YOLOZU owns schema validation, evaluation, parity, and reports.
The only required handoff artifact is `predictions.json` in the YOLOZU predictions interface contract.

Inputs:
- model/runtime inputs owned by the Rust project (`.onnx`, TensorRT FFI handle, custom accelerator output, or batch source)
- image identity forwarded into each `predictions[*].image`
- optional dataset root only when running YOLOZU evaluation after export

Outputs:
- `predictions.json` with normalized `cx, cy, w, h` boxes and numeric scores
- optional backend metadata in a separate report or wrapper log
- YOLOZU reports such as `reports/external_eval.json` and `reports/external_parity.json`

Error behavior:
- exit nonzero for missing arguments, disabled optional features, runtime load failures, invalid decode settings, or write failures
- never treat a partial or schema-invalid `predictions.json` as success
- keep optional ONNXRuntime dependency failures explicit instead of silently producing empty detections

Schema validation and report handoff:

```bash
python3 tools/validate_predictions.py /path/to/predictions.json --strict
yolozu eval-coco --dataset /path/to/coco-yolo --predictions /path/to/predictions.json --output reports/external_eval.json
python3 tools/check_predictions_parity.py --reference reports/pred_torch.json --candidate /path/to/predictions.json > reports/external_parity.json
```

## Build

```bash
cargo build --release
```

## Run (stub)

```bash
./target/release/yolozu_infer_rust --out reports/pred_custom_cpp.json
python3 tools/validate_predictions.py reports/pred_custom_cpp.json --strict
```

## Build (ONNXRuntime feature)

```bash
cargo build --release --features onnxruntime
```

## Run (ONNXRuntime smoke path)

```bash
./target/release/yolozu_infer_rust \
  --mode onnxrt \
  --onnx /abs/path/model.onnx \
  --input-shape 1,3,64,64 \
  --combined-format xyxy_score_class \
  --boxes-scale norm \
  --input-size 64x64 \
  --image images/val/000001.jpg \
  --out reports/pred_rust_onnxrt.json

python3 tools/validate_predictions.py reports/pred_rust_onnxrt.json --strict
```

The ONNXRuntime mode performs a minimal forward pass and decodes the first output with the declared
`xyxy_score_class` decode interface (`x1,y1,x2,y2,score,class_id`). If your model uses a different output layout,
add another declared decoder rather than silently emitting empty detections.

## Expected build/runtime environment (production)

- Linux `x86_64` is the most common production target.
- Rust toolchain (`cargo`) for the template binary.
- Python 3 runtime with `onnxruntime` + `numpy` installed (the optional mode shells out to Python ORT).
- ONNX model input shape known/pinned (`--input-shape` must match the ONNX graph expectations).
- CI/container recommendation: keep `cargo build --release` (stub) as baseline, and run `--features onnxruntime` in a dedicated image where ONNXRuntime dependencies are intentionally provisioned.

Notes:
- Stub mode outputs empty predictions and intentionally omits `meta` so `--strict` validation passes. It is intended as a wiring/predictions interface contract template, not a model runner.
- In `onnxruntime` mode the template includes a `meta` block for reproducibility/debugging.
- For integrating into YOLOZU parity checks, use `custom_cpp` (external backend) routes in `docs/external_inference.md`.
