# Doctor diagnostics for environment drift

`yolozu doctor` now reports runtime capability differences that often explain parity drift across backends.

## What is reported

- `runtime_capabilities.cuda`: CUDA visibility and GPU presence from `nvidia-smi`
- `runtime_capabilities.torch`: Torch CUDA + MPS availability/version/cudnn/device count
- `runtime_capabilities.onnxruntime`: provider list (`CUDAExecutionProvider`, `TensorrtExecutionProvider`, `CoreMLExecutionProvider`)
- `runtime_capabilities.tensorrt`: Python package availability + `trtexec` availability/version
- `runtime_capabilities.opencv`: OpenCV module/version and CUDA-enabled device count
- `env.PYTORCH_ENABLE_MPS_FALLBACK`: whether MPS CPU fallback is enabled in the current shell
- `drift_hints`: human-readable likely causes and remediation links
- `guidance_links`: canonical docs for parity, TensorRT, and baseline reproducibility

## Typical command

```bash
yolozu doctor --output -
```

## Example drift hints

- Torch uses CUDA but ONNXRuntime has no CUDA provider
- TensorRT provider appears in ORT but `trtexec` is missing
- OpenCV CUDA path disabled while Torch CUDA is enabled
- `CUDA_VISIBLE_DEVICES` masks devices and forces CPU fallback

## macOS / Apple Silicon note

For macOS hosts, `yolozu doctor` now shows:

- `runtime_capabilities.torch.mps_built`
- `runtime_capabilities.torch.mps_available`
- `runtime_capabilities.onnxruntime.coreml_provider`

That makes it easier to distinguish a plain CPU install from a real MPS/CoreML-capable local setup.

Use reported `guidance_links` to jump to remediation docs.
