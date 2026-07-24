# Benchmark Backend Runtime / License Boundary Matrix

This page is the benchmark-side source of truth for runtime and redistribution
boundaries. For per-task artifact status, use the canonical
[Benchmark support matrix](benchmark_support_matrix.md).
That matrix also defines backend-flag applicability by task, requested/effective
latency source, and format. Supported `artifact_eval` tasks consume prepared
artifacts and therefore require `--no-half --batch 1 --no-nms`. Detect has no
prepared detection-artifact evaluation path: explicit detect `artifact_eval`
fails before report, artifact, or backend writes, while `auto` and
`dataset_pass_wall_time` retain the conditional real backend path.

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
| `torch` | real orchestration for `detect`; artifact-backed real lanes for classification/OBB/segmentation/keypoints/depth/pose6d | Local PyTorch + external YOLO-family runtime for detect; artifact-backed tasks consume supplied files | Python packages have their own terms; verify your chosen model/runtime stack separately | No |
| `onnx` | real orchestration for `detect`; artifact-backed real lanes for classification/OBB/segmentation/keypoints/depth/pose6d | Local ONNX Runtime install for detect; artifact-backed tasks consume supplied files | ONNX Runtime is external to this repo; keep binary/runtime terms separate from YOLOZU | No |
| `engine` / TensorRT | real orchestration for `detect`; artifact-backed real lanes for classification/OBB/segmentation/keypoints/depth/pose6d | Linux + NVIDIA GPU + TensorRT runtime/engine for detect; artifact-backed tasks consume supplied files | Requires external NVIDIA runtime/SDK; verify redistribution terms for CUDA/TensorRT/NGC artifacts | No |
| `torchscript` | real orchestration for `detect`; artifact-backed real lanes for classification/OBB/segmentation/keypoints/depth/pose6d | Local PyTorch runtime for detect; artifact-backed tasks consume supplied files | No vendor GPU runtime is implied; still external to this repo | No |
| `openvino` | conditional real orchestration for `detect`; artifact-backed real lanes for classification/OBB/segmentation/keypoints/depth/pose6d | External Intel OpenVINO runtime and compatible IR artifact for detect; artifact-backed tasks consume supplied files | OpenVINO is external to this repo; do not assume runtime or model redistribution from YOLOZU | No |
| `executorch` | unsupported/skipped | Benchmark runtime is not invoked; standalone exporter utilities may use an external ExecuTorch runtime | ExecuTorch runtime is external; verify platform packaging constraints separately | No |
| `opencv_dnn` | unsupported/skipped | Benchmark runtime is not invoked; standalone exporter utilities may use local OpenCV | OpenCV is external to this repo; optional contrib/nonfree modules are not bundled here | No |
| `coreml` | planned / conditional | Apple platform runtime/tooling | Apple platform/runtime terms apply; not bundled here | No |
| `tflite` | planned | TensorFlow Lite runtime | External runtime terms apply; not bundled here | No |
| `ncnn` | planned | External NCNN runtime | External runtime/build terms apply; not bundled here | No |
| `rknn` | planned | External Rockchip RKNN SDK/runtime | Vendor SDK/runtime terms apply; verify redistribution separately | No |
| `paddle` | planned | External Paddle runtime | External runtime terms apply; not bundled here | No |

## Interpretation rules

- `real orchestration` means the benchmark can run repository-side prediction
  and evaluation paths when the external runtime and artifacts are present.
- `unsupported/skipped` means the format label is visible in the benchmark
  interface contract, but benchmark orchestration does not invoke that runtime
  and writes explicit skipped records.
- `planned` means the format is not part of the current benchmark CLI surface.
- `runtime.available` is evidence of a runtime probe only when
  `runtime.checked` is `true`. Artifact-backed lanes set `runtime.required` and
  `runtime.checked` to `false` because they consume prepared files.
- classification and OBB artifact-backed lanes validate their input interface
interface contracts independently of the selected backend: duplicate ids, non-finite
  task values, class/score shape drift, and OBB scores outside `[0,1]` fail
  before metric computation.
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
