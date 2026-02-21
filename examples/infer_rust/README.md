# YOLOZU Rust inference template (stub)

This folder is a **submodule-ready** starting point for production inference pipelines in Rust.

- By default it builds a tiny **no-deps** binary that writes a schema-valid `predictions.json` stub.
- Replace the stub with a real backend (ONNX Runtime, TensorRT via FFI, etc.), keeping the JSON contract stable.

## Build

```bash
cargo build --release
```

## Run (stub)

```bash
./target/release/yolozu_infer_rust --out reports/pred_custom_cpp.json
python3 tools/validate_predictions.py reports/pred_custom_cpp.json --strict
```

Notes:
- The output is empty predictions and intentionally omits `meta` so `--strict` validation passes. It is intended as a wiring/contract template, not a model runner.
- For integrating into YOLOZU parity checks, use `custom_cpp` (external backend) routes in `docs/external_inference.md`.
