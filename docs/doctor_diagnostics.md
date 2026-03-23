# Doctor diagnostics for environment drift

`yolozu doctor` now reports runtime capability differences that often explain parity drift across backends.

## What is reported

- `runtime_capabilities.cuda`: CUDA visibility and GPU presence from `nvidia-smi`
- `runtime_capabilities.torch`: Torch CUDA availability/version/cudnn/device count
- `runtime_capabilities.onnxruntime`: provider list (`CUDAExecutionProvider`, `TensorrtExecutionProvider`)
- `runtime_capabilities.tensorrt`: Python package availability + `trtexec` availability/version
- `runtime_capabilities.opencv`: OpenCV module/version and CUDA-enabled device count
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

Use reported `guidance_links` to jump to remediation docs.

## Best-effort behavior

- Optional runtime probes are best-effort: missing extras such as TensorRT, ONNX Runtime, or
	OpenCV CUDA bindings now degrade the reported capability block instead of aborting the whole
	`doctor` run.
- Version/package metadata collection is also best-effort, so malformed local metadata or missing
	package distributions are reported as unavailable rather than crashing diagnostics.
