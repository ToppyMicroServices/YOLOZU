# YOLOZU Rust inference template (stub + optional ONNXRuntime)

This folder is a **submodule-ready** starting point for production inference pipelines in Rust.

- By default it builds a tiny **no-deps** binary that writes a schema-valid `predictions.json` stub.
- Optional `onnxruntime` cargo feature enables a minimal ONNXRuntime-backed runner for production wiring checks.
- Replace/extend these paths with your real backend (TensorRT via FFI, custom decode, etc.), keeping the JSON contract stable.

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
  --image images/val/000001.jpg \
  --out reports/pred_rust_onnxrt.json

python3 tools/validate_predictions.py reports/pred_rust_onnxrt.json --strict
```

The ONNXRuntime mode currently performs a minimal forward pass and emits empty detections with backend metadata (`meta.extra.input_shape`, `meta.extra.output_shapes`).
You can then plug in model-specific decode logic without changing the YOLOZU contract surface.

## Expected build/runtime environment (production)

- Linux `x86_64` is the most common production target.
- Rust toolchain (`cargo`) for the template binary.
- Python 3 runtime with `onnxruntime` + `numpy` installed (the optional mode shells out to Python ORT).
- ONNX model input shape known/pinned (`--input-shape` must match the ONNX graph expectations).
- CI/container recommendation: keep `cargo build --release` (stub) as baseline, and run `--features onnxruntime` in a dedicated image where ONNXRuntime dependencies are intentionally provisioned.

Notes:
- The output is empty predictions and intentionally omits `meta` so `--strict` validation passes. It is intended as a wiring/contract template, not a model runner.
- In `onnxruntime` mode the template includes a `meta` block for reproducibility/debugging.
- For integrating into YOLOZU parity checks, use `custom_cpp` (external backend) routes in `docs/external_inference.md`.
