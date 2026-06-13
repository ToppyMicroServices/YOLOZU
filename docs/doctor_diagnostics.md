# Doctor diagnostics for environment drift

`yolozu doctor` now reports runtime capability differences that often explain parity drift across backends.
It is a capability report, not a blanket production-readiness verdict.

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

## CPU proof

Use `--proof` when you want more than an environment inventory:

```bash
yolozu doctor --proof
```

This writes a tiny YOLO-style dataset, known predictions, an eval report, and a proof report under `reports/doctor_proof/`. The proof validates the toy dataset and predictions interface contract, runs the detection mAP path, and compares the observed metrics against pinned expected values (`map50=1.0`, `map50_95=1.0`). It is CPU-only and does not download models or datasets.

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

Interpretation rule:

- MPS is supported when `torch.backends.mps.is_available()` is `true`
- a tool marked `macos_ok: true` in the manifest only means the CLI can run on macOS
- `macos_ok: true` does not imply that MPS is available on that machine

Practical macOS triage:

- if `mps_built=false`, your Torch build has no MPS backend
- if `mps_built=true` but `mps_available=false`, the wheel/runtime combo is the likely blocker
- Python itself can create the environment with `venv`/`pip`; on some Apple Silicon hosts, Miniforge/conda PyTorch may expose MPS correctly even when `pip` wheels do not
- after switching environments, rerun `yolozu doctor --output -` and compare `runtime_capabilities.torch.*`

For a tested Miniforge setup path, see [`install.md`](install.md#macos--apple-silicon-miniforgemps-workflow).

Use reported `guidance_links` to jump to remediation docs.
