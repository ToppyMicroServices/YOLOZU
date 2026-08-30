# Version Compatibility

This page separates install floors, repository-pinned validation environments,
protocol choices, and GPU environment evidence. They are not interchangeable:
an install floor is not a claim that every newer version was qualified, and a
configured GPU container is not proof that a particular host completed a run.

## Package and CPU validation envelope

| Component | Declared install floor | Current repository-pinned evidence | Scope |
|---|---|---|---|
| PyTorch | `torch>=2.10.0` in the Torch-backed extras | `torch==2.10.0+cpu` in `requirements-locks/requirements-ci.lock`; `torch==2.10.0` in the demo and RT-DETR locks | The CI pin qualifies repository CPU tests. Device, accelerator, and custom-wheel behavior remain environment-specific. |
| Torchvision | `torchvision>=0.25.0` in the Torch-backed extras | `torchvision==0.25.0` in `requirements-locks/requirements-demo-extra.lock` | Applies to the demo/runtime bundle, not the dependency-free validation/evaluation core. |
| ONNX | `onnx>=1.21.0` in ONNX-backed extras | `onnx==1.21.0` in the CI, TensorRT-tool, and RT-DETR locks | Regenerate and recheck exported artifacts when the runtime or exporter changes. |
| ONNX Runtime | `onnxruntime>=1.17` in ONNX-backed extras | CPU `onnxruntime==1.24.2` in the CI and TensorRT-tool locks; task-specific RT-DETR locks use `onnxruntime==1.24.3` or `onnxruntime-gpu==1.24.4` | The floor is packaging compatibility. Backend evidence must record the runtime actually used. |

The exact lock files, rather than this summary, remain the machine-consumed
source for each test environment.

## ONNX opset boundary

`tools/export_trt.py` defaults to ONNX opset `18`. The current TensorRT/YOLO
examples and GPU smoke workflow explicitly pin opset `17`; that is a recorded
protocol choice, not the tool default. Keep the selected opset in export
metadata and compare artifacts only under the same preprocessing, decode, and
opset protocol.

## TensorRT and CUDA qualification

YOLOZU does not declare one universally qualified TensorRT/CUDA pair. The
self-hosted GPU workflow is configured with
`nvcr.io/nvidia/tensorrt:24.08-py3`, but it runs only when the required runner
and credentials are available. A successful run must retain:

- the exact container image or installed package identity;
- `nvidia-smi` GPU, driver, and reported CUDA context;
- `trtexec --version` or the TensorRT Python package version;
- the generated engine metadata and parity/latency reports.

Without those run artifacts, describe TensorRT/CUDA support as
environment-qualified rather than attaching a static version claim.

See [`production_readiness.md`](production_readiness.md) for the production
readiness matrix and [`evaluation_protocol_template.md`](evaluation_protocol_template.md)
for the reusable evaluation protocol template.
