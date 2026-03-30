# Quantization (ONNXRuntime / torchao)

YOLOZU keeps quantization **opt-in**. The default evaluation contract (`predictions.json`) does not assume a specific backend.

## 1) ONNXRuntime dynamic quantization (recommended for CPU)

If you have an ONNX model, you can produce an INT8-weight model using ONNXRuntime (CPU-friendly):

```bash
yolozu onnxrt quantize --onnx runs/<run_id>/exports/model.onnx --output runs/<run_id>/exports/model_int8.onnx
```

This path does **not** require torchao.

## 2) torchao quantization (experimental)

PyTorch quantization is increasingly moving toward `torchao`. YOLOZU exposes a lightweight, best-effort integration
in the RT-DETR pose trainer:

- `--torchao-quant {int8wo,int4wo}` (weight-only recipes)
- `--torchao-required` (fail fast if torchao is missing or the API call fails)
- `--qlora` (convenience flag: sets `--torchao-quant=int4wo` and forces `--lora-freeze-base`; requires `--lora-r>0`)

Notes:
- torchao is an optional dependency; install it separately (example: `pip install torchao`).
- The integration is defensive because torchao APIs can evolve; treat this as experimental.
- For the beginner-facing intuition of how QLoRA differs from plain LoRA in this repo, read [`docs/continual_learning.md`](continual_learning.md) first and come back here for the concrete knobs.
