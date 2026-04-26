# Benchmark Support Matrix

This is the canonical benchmark support matrix for `yolozu benchmark` and
`tools/benchmark_model.py`. It describes benchmark artifacts, not every
standalone exporter utility.

Legend:

- `real`: benchmark runs the backend/export/eval path and writes real predictions/eval artifacts when the required model artifact and runtime are present; otherwise the report is `skipped`.
- `artifact-real`: benchmark consumes a backend-specific artifact and writes real eval/parity artifacts without claiming YOLOZU ran backend inference.
- `placeholder`: benchmark writes an explicit placeholder artifact, usually for dry-run or synthetic planning.
- `skipped`: benchmark records a missing runtime, missing model artifact, platform, or GPU requirement.
- `planned`: task/format is visible in the interface contract but real benchmark wiring is intentionally not implemented yet.

| Format | Task | Inference artifact | Eval artifact | Parity artifact | Notes |
| --- | --- | --- | --- | --- | --- |
| `torch` | `detect` | real or skipped | real or skipped | real when comparable | Uses `export_predictions_ultralytics.py` plus `eval_suite.py`. |
| `torch` | `segmentation` | artifact-real | real | real when comparable | Consumes backend mask-prediction artifacts. |
| `torch` | `classification` | planned | planned | planned | Dedicated classification eval wiring is pending. |
| `torch` | `obb` | planned | planned | planned | Dedicated OBB eval wiring is pending. |
| `torch` | `keypoints` | artifact-real | real | real when comparable | Consumes backend predictions artifacts. |
| `torch` | `depth` | artifact-real | real | real when comparable | Consumes backend depth artifacts. |
| `torch` | `pose6d` | artifact-real | real | real when comparable | Consumes backend pose predictions artifacts. |
| `onnx` | `detect` | real or skipped | real or skipped | real when comparable | Requires ONNX Runtime and an `.onnx` artifact or `--onnx-model`. |
| `onnx` | `segmentation` | artifact-real | real | real when comparable | Consumes backend mask-prediction artifacts. |
| `onnx` | `classification` | planned | planned | planned | Dedicated classification eval wiring is pending. |
| `onnx` | `obb` | planned | planned | planned | Dedicated OBB eval wiring is pending. |
| `onnx` | `keypoints` | artifact-real | real | real when comparable | Consumes backend predictions artifacts. |
| `onnx` | `depth` | artifact-real | real | real when comparable | Consumes backend depth artifacts. |
| `onnx` | `pose6d` | artifact-real | real | real when comparable | Consumes backend pose predictions artifacts. |
| `engine` | `detect` | real or skipped | real or skipped | real when comparable | Requires Linux, GPU, TensorRT/CUDA bindings, and `.engine`/`.plan`. |
| `engine` | `segmentation` | artifact-real | real | real when comparable | Consumes backend mask-prediction artifacts. |
| `engine` | `classification` | planned | planned | planned | Dedicated classification eval wiring is pending. |
| `engine` | `obb` | planned | planned | planned | Dedicated OBB eval wiring is pending. |
| `engine` | `keypoints` | artifact-real | real | real when comparable | Consumes backend predictions artifacts. |
| `engine` | `depth` | artifact-real | real | real when comparable | Consumes backend depth artifacts. |
| `engine` | `pose6d` | artifact-real | real | real when comparable | Consumes backend pose predictions artifacts. |
| `torchscript` | `detect` | real or skipped | real or skipped | real when comparable | Uses local PyTorch and the declared combined-output decode path in `export_predictions_torchscript.py`. |
| `torchscript` | `segmentation` | artifact-real | real | real when comparable | Consumes backend mask-prediction artifacts. |
| `torchscript` | `classification` | planned | planned | planned | Dedicated classification eval wiring is pending. |
| `torchscript` | `obb` | planned | planned | planned | Dedicated OBB eval wiring is pending. |
| `torchscript` | `keypoints` | artifact-real | real | real when comparable | Consumes backend predictions artifacts. |
| `torchscript` | `depth` | artifact-real | real | real when comparable | Consumes backend depth artifacts. |
| `torchscript` | `pose6d` | artifact-real | real | real when comparable | Consumes backend pose predictions artifacts. |
| `executorch` | `detect` | placeholder or skipped | placeholder | placeholder | Standalone `export_predictions_executorch.py` has a declared runtime-output decode path; benchmark orchestration remains planned. |
| `executorch` | `segmentation` | placeholder or skipped | placeholder | placeholder | Benchmark artifact lane remains planned. |
| `executorch` | `classification` | planned | planned | planned | Benchmark wiring is pending. |
| `executorch` | `obb` | planned | planned | planned | Benchmark wiring is pending. |
| `executorch` | `keypoints` | placeholder or skipped | placeholder | placeholder | Benchmark artifact lane remains planned. |
| `executorch` | `depth` | placeholder or skipped | placeholder | placeholder | Benchmark artifact lane remains planned. |
| `executorch` | `pose6d` | placeholder or skipped | placeholder | placeholder | Benchmark artifact lane remains planned. |
| `opencv_dnn` | `detect` | placeholder or skipped | placeholder | placeholder | Benchmark orchestration remains planned. |
| `opencv_dnn` | `segmentation` | placeholder or skipped | placeholder | placeholder | Benchmark artifact lane remains planned. |
| `opencv_dnn` | `classification` | planned | planned | planned | Benchmark wiring is pending. |
| `opencv_dnn` | `obb` | planned | planned | planned | Benchmark wiring is pending. |
| `opencv_dnn` | `keypoints` | placeholder or skipped | placeholder | placeholder | Benchmark artifact lane remains planned. |
| `opencv_dnn` | `depth` | placeholder or skipped | placeholder | placeholder | Benchmark artifact lane remains planned. |
| `opencv_dnn` | `pose6d` | placeholder or skipped | placeholder | placeholder | Benchmark artifact lane remains planned. |

When CLI behavior changes, update this page, `docs/benchmark_mode.md`,
`manual/chapters/09_parity_bench_protocols.tex`, `tools/manifest.json`, and the
packaged manifest copy in the same PR.
