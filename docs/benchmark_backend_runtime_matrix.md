# Benchmark Backend Runtime / License Boundary Matrix

This page is the benchmark-side source of truth for runtime and redistribution
boundaries.

YOLOZU itself remains an **Apache-2.0** repository. Several benchmark/export
backends rely on **external runtimes or vendor SDKs** with their own license
terms. That means:

- `supported` does **not** mean `bundled`
- `adapter available` does **not** mean `vendor runtime redistributed`
- GPU/runtime redistribution terms must be verified separately by the user

This document is **not legal advice**. It is an operational boundary guide for
benchmark users and maintainers.

## Matrix

| Backend / format | Current benchmark state | Runtime requirement | License / redistribution note | Bundled with YOLOZU |
| --- | --- | --- | --- | --- |
| `torch` | real orchestration for `detect`; other tasks planning-only | Local PyTorch + Ultralytics runtime | Python packages have their own terms; verify your chosen model/runtime stack separately | No |
| `onnx` | real orchestration for `detect`; other tasks planning-only | Local ONNX Runtime install | ONNX Runtime is external to this repo; keep binary/runtime terms separate from YOLOZU | No |
| `engine` / TensorRT | real orchestration for `detect`; other tasks planning-only | Linux + NVIDIA GPU + TensorRT runtime/engine | Requires external NVIDIA runtime/SDK; verify redistribution terms for CUDA/TensorRT/NGC artifacts | No |
| `torchscript` | accepted; synthetic / planning-only | Local PyTorch runtime | No vendor GPU runtime is implied; still external to this repo | No |
| `executorch` | synthetic / planning-only | External ExecuTorch runtime if used | ExecuTorch runtime is external; verify platform packaging constraints separately | No |
| `opencv_dnn` | synthetic / planning-only | Local OpenCV runtime | OpenCV is external to this repo; optional contrib/nonfree modules are not bundled here | No |
| `openvino` | planned / conditional | External Intel OpenVINO runtime | External vendor/runtime terms apply; do not assume redistribution from YOLOZU | No |
| `coreml` | planned / conditional | Apple platform runtime/tooling | Apple platform/runtime terms apply; not bundled here | No |
| `tflite` | planned | TensorFlow Lite runtime | External runtime terms apply; not bundled here | No |
| `ncnn` | planned | External NCNN runtime | External runtime/build terms apply; not bundled here | No |
| `rknn` | planned | External Rockchip RKNN SDK/runtime | Vendor SDK/runtime terms apply; verify redistribution separately | No |
| `paddle` | planned | External Paddle runtime | External runtime terms apply; not bundled here | No |

## Interpretation rules

- `real orchestration` means the benchmark can run repository-side prediction
  and evaluation paths when the external runtime and artifacts are present.
- `planning-only` means the benchmark report is still useful for artifact
  planning, task semantics, and CI/report stability, but it does not imply a
  real backend pass happened.
- `bundled with YOLOZU = No` is intentional. This repo provides adapter/wrapper
  logic and interface-contract-safe orchestration, not vendor runtime
  redistribution.

## Apache-2.0 positioning

The practical value of YOLOZU versus GPL alternatives is that:

- the repository code stays Apache-2.0
- adapter/wrapper logic can remain Apache-2.0-friendly
- organizations can evaluate external runtime usage separately from repo code

That is useful, but it should never be overstated into a legal guarantee about
third-party runtimes, datasets, or model weights.
