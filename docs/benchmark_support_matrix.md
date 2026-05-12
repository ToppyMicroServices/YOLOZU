# Benchmark Support Matrix

This is the canonical benchmark support matrix for `yolozu benchmark` and
`tools/benchmark_model.py`. It describes benchmark artifacts, not every
standalone exporter utility.

Legend:

- `real`: benchmark runs the backend/export/eval path and writes real predictions/eval artifacts when the required model artifact and runtime are present; otherwise the report is `skipped`.
- `artifact-real`: benchmark consumes a backend-specific artifact and writes real eval/parity artifacts without claiming YOLOZU ran backend inference.
- `placeholder`: benchmark writes an explicit placeholder artifact for dry-run planning only.
- `skipped`: benchmark records a missing runtime, missing model artifact, platform, or GPU requirement.
- `unsupported/skipped`: task or format is visible in the interface contract but real benchmark wiring is not shipped; benchmark runs report `skipped`.

| Format | Task | Inference artifact | Eval artifact | Parity artifact | Notes |
| --- | --- | --- | --- | --- | --- |
| `torch` | `detect` | real or skipped | real or skipped | real when comparable | Uses `export_predictions_ultralytics.py` plus `eval_suite.py`. |
| `torch` | `segmentation` | artifact-real | real | real when comparable | Consumes backend mask-prediction artifacts. |
| `torch` | `classification` | skipped | skipped | skipped | Dedicated classification eval wiring is not shipped; benchmark reports `benchmark_task_not_wired`. |
| `torch` | `obb` | skipped | skipped | skipped | Dedicated OBB eval wiring is not shipped; benchmark reports `benchmark_task_not_wired`. |
| `torch` | `keypoints` | artifact-real | real | real when comparable | Consumes backend predictions artifacts. |
| `torch` | `depth` | artifact-real | real | real when comparable | Consumes backend depth artifacts. |
| `torch` | `pose6d` | artifact-real | real | real when comparable | Consumes backend pose predictions artifacts. |
| `onnx` | `detect` | real or skipped | real or skipped | real when comparable | Requires ONNX Runtime and an `.onnx` artifact or `--onnx-model`. |
| `onnx` | `segmentation` | artifact-real | real | real when comparable | Consumes backend mask-prediction artifacts. |
| `onnx` | `classification` | skipped | skipped | skipped | Dedicated classification eval wiring is not shipped; benchmark reports `benchmark_task_not_wired`. |
| `onnx` | `obb` | skipped | skipped | skipped | Dedicated OBB eval wiring is not shipped; benchmark reports `benchmark_task_not_wired`. |
| `onnx` | `keypoints` | artifact-real | real | real when comparable | Consumes backend predictions artifacts. |
| `onnx` | `depth` | artifact-real | real | real when comparable | Consumes backend depth artifacts. |
| `onnx` | `pose6d` | artifact-real | real | real when comparable | Consumes backend pose predictions artifacts. |
| `engine` | `detect` | real or skipped | real or skipped | real when comparable | Requires Linux, GPU, TensorRT/CUDA bindings, and `.engine`/`.plan`. |
| `engine` | `segmentation` | artifact-real | real | real when comparable | Consumes backend mask-prediction artifacts. |
| `engine` | `classification` | skipped | skipped | skipped | Dedicated classification eval wiring is not shipped; benchmark reports `benchmark_task_not_wired`. |
| `engine` | `obb` | skipped | skipped | skipped | Dedicated OBB eval wiring is not shipped; benchmark reports `benchmark_task_not_wired`. |
| `engine` | `keypoints` | artifact-real | real | real when comparable | Consumes backend predictions artifacts. |
| `engine` | `depth` | artifact-real | real | real when comparable | Consumes backend depth artifacts. |
| `engine` | `pose6d` | artifact-real | real | real when comparable | Consumes backend pose predictions artifacts. |
| `torchscript` | `detect` | real or skipped | real or skipped | real when comparable | Uses local PyTorch and the declared combined-output decode path in `export_predictions_torchscript.py`. |
| `torchscript` | `segmentation` | artifact-real | real | real when comparable | Consumes backend mask-prediction artifacts. |
| `torchscript` | `classification` | skipped | skipped | skipped | Dedicated classification eval wiring is not shipped; benchmark reports `benchmark_task_not_wired`. |
| `torchscript` | `obb` | skipped | skipped | skipped | Dedicated OBB eval wiring is not shipped; benchmark reports `benchmark_task_not_wired`. |
| `torchscript` | `keypoints` | artifact-real | real | real when comparable | Consumes backend predictions artifacts. |
| `torchscript` | `depth` | artifact-real | real | real when comparable | Consumes backend depth artifacts. |
| `torchscript` | `pose6d` | artifact-real | real | real when comparable | Consumes backend pose predictions artifacts. |
| `executorch` | `detect` | skipped | skipped | skipped | Standalone `export_predictions_executorch.py` has a declared runtime-output decode path; benchmark orchestration reports `benchmark_format_not_wired`. |
| `executorch` | `segmentation` | skipped | skipped | skipped | Benchmark artifact lane is not shipped; benchmark reports `benchmark_format_not_wired`. |
| `executorch` | `classification` | skipped | skipped | skipped | Benchmark wiring is not shipped; benchmark reports `benchmark_task_not_wired`. |
| `executorch` | `obb` | skipped | skipped | skipped | Benchmark wiring is not shipped; benchmark reports `benchmark_task_not_wired`. |
| `executorch` | `keypoints` | skipped | skipped | skipped | Benchmark artifact lane is not shipped; benchmark reports `benchmark_format_not_wired`. |
| `executorch` | `depth` | skipped | skipped | skipped | Benchmark artifact lane is not shipped; benchmark reports `benchmark_format_not_wired`. |
| `executorch` | `pose6d` | skipped | skipped | skipped | Benchmark artifact lane is not shipped; benchmark reports `benchmark_format_not_wired`. |
| `opencv_dnn` | `detect` | skipped | skipped | skipped | Standalone OpenCV-DNN exporters exist; benchmark orchestration reports `benchmark_format_not_wired`. |
| `opencv_dnn` | `segmentation` | skipped | skipped | skipped | Benchmark artifact lane is not shipped; benchmark reports `benchmark_format_not_wired`. |
| `opencv_dnn` | `classification` | skipped | skipped | skipped | Benchmark wiring is not shipped; benchmark reports `benchmark_task_not_wired`. |
| `opencv_dnn` | `obb` | skipped | skipped | skipped | Benchmark wiring is not shipped; benchmark reports `benchmark_task_not_wired`. |
| `opencv_dnn` | `keypoints` | skipped | skipped | skipped | Benchmark artifact lane is not shipped; benchmark reports `benchmark_format_not_wired`. |
| `opencv_dnn` | `depth` | skipped | skipped | skipped | Benchmark artifact lane is not shipped; benchmark reports `benchmark_format_not_wired`. |
| `opencv_dnn` | `pose6d` | skipped | skipped | skipped | Benchmark artifact lane is not shipped; benchmark reports `benchmark_format_not_wired`. |

When CLI behavior changes, update this page, `docs/benchmark_mode.md`,
`manual/chapters/09_parity_bench_protocols.tex`, `tools/manifest.json`, and the
packaged manifest copy in the same PR.
