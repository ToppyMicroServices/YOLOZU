# Version Compatibility

Target envelope for training/export/inference qualification.

| Component | Target | Used for | Notes |
|---|---|---|---|
| PyTorch | 2.1.2 | training/export reference envelope | Update before publishing backend parity claims. |
| CUDA | 12.1 | GPU backend qualification | CPU-only validation/evaluation does not require CUDA. |
| TensorRT | 8.6.1 | TensorRT engine/parity lane | Treat as environment-qualified. |
| ONNX Runtime | 1.17.1 | ONNX parity lane | Regenerate artifacts when runtime support changes. |
| ONNX opset | 17 | export target | Protocols may override this when needed. |

If target GPU/driver constraints differ, update this matrix before export.

See [`production_readiness.md`](production_readiness.md) for the production
readiness matrix and [`evaluation_protocol_template.md`](evaluation_protocol_template.md)
for the reusable evaluation protocol template.
