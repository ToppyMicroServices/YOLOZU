# C++ inference template (submodule-ready)

This folder is a **minimal CMake-based C++ template** meant to help you start a production-grade inference path while keeping
YOLOZU's evaluation pipeline unchanged.

It is intentionally structured so you can:

1) develop it *in-tree* under `examples/infer_cpp/`, then
2) **extract it into a separate repo**, and
3) add it back to YOLOZU as a git submodule (once the repo URL exists).

## What it provides

- `yolozu_infer_stub`: builds everywhere, emits schema-correct `predictions.json` with empty detections
  - Optional (when deps exist):
  - `yolozu_infer_onnxrt`: ONNXRuntime C++ runner (CPU)
  - `yolozu_infer_trt`: TensorRT C++ runner (Linux+NVIDIA)

All runners aim to produce YOLOZU predictions JSON so you can:

```bash
python3 tools/validate_predictions.py /path/to/predictions.json --strict
python3 tools/eval_coco.py --predictions /path/to/predictions.json --dataset /path/to/coco-yolo --output reports/external_eval.json
```

## Production lane interface contract

Use this template as the production runner boundary: C++ owns inference, decode, and file writes; YOLOZU owns schema validation, evaluation, parity, and reports.
The only required handoff artifact is `predictions.json` in the YOLOZU predictions interface contract.

Inputs:
- model/runtime inputs owned by the C++ project (`.onnx`, TensorRT engine, calibration files, or camera/batch source)
- image identity forwarded into each `predictions[*].image`
- optional dataset root only when running YOLOZU evaluation after export

Outputs:
- `predictions.json` with normalized `cx, cy, w, h` boxes and numeric scores
- optional backend metadata in a separate report or wrapper log
- YOLOZU reports such as `reports/external_eval.json` and `reports/external_parity.json`

Error behavior:
- exit nonzero for missing arguments, runtime load failures, unreadable input images, decode mismatches, or write failures
- never treat a partial or schema-invalid `predictions.json` as success
- keep optional ONNXRuntime/TensorRT dependency failures explicit in the runner output or wrapper report

Schema validation and report handoff:

```bash
python3 tools/validate_predictions.py /path/to/predictions.json --strict
python3 tools/eval_coco.py --dataset /path/to/coco-yolo --predictions /path/to/predictions.json --output reports/external_eval.json
python3 tools/check_predictions_parity.py --reference reports/pred_torch.json --candidate /path/to/predictions.json > reports/external_parity.json
```

## Recommended workflow

1) Start with the stub to validate your end-to-end predictions interface contract (C++ → `predictions.json` → Python validators).
2) Implement real inference in ONNXRuntime or TensorRT while keeping the JSON output stable.
3) Use YOLOZU’s evaluator/suite to compare backends apples-to-apples.

Note: the stub intentionally emits **empty detections**, so COCO mAP will be ~0. It is only a predictions interface contract check.
The stub intentionally omits `meta` to keep `python3 tools/validate_predictions.py ... --strict` passing; you can add backend metadata later.
For a “sanity mAP” run without any inference backend, use:

```bash
yolozu export --backend labels --dataset /path/to/coco-yolo --output /tmp/predictions_labels.json --force
python3 tools/eval_coco.py --predictions /tmp/predictions_labels.json --dataset /path/to/coco-yolo --output reports/labels_eval.json
```

## Build (stub)

```bash
cmake -S . -B build
cmake --build build -j
./build/yolozu_infer_stub --image /abs/path.jpg --output /tmp/predictions.json
python3 tools/validate_predictions.py /tmp/predictions.json --strict
```

## Build (ONNXRuntime)

This template expects ONNXRuntime headers/libs to be available and passed via `ONNXRUNTIME_ROOT`.

Example (after downloading an ONNXRuntime release tarball and extracting to `/opt/onnxruntime`):

```bash
cmake -S . -B build -DONNXRUNTIME_ROOT=/opt/onnxruntime
cmake --build build -j
```

## Build (TensorRT)

TensorRT C++ is easiest inside an NVIDIA/TensorRT base image (Runpod/Linux). The CMake checks for `NvInfer.h` + `libnvinfer`.

## Docker build recipes

These are meant as **repeatable build environments** (Linux) for the optional runners.

### ONNXRuntime (CPU) container

Build:

```bash
docker build -f examples/infer_cpp/docker/Dockerfile.onnxrt -t yolozu-infer-cpp:onnxrt .
```

Run (prints help by default):

```bash
docker run --rm -it yolozu-infer-cpp:onnxrt
```

### TensorRT container

Build:

```bash
docker build -f examples/infer_cpp/docker/Dockerfile.trt -t yolozu-infer-cpp:trt .
```

Run (prints help by default):

```bash
docker run --rm -it --gpus all yolozu-infer-cpp:trt
```

## Extracting into a submodule (recommended later)

Once you create a separate repo (e.g. `yolozu-infer-cpp`) under the same GitHub org:

1) Move this folder into its own repo (keep Apache-2.0 compatible code).
2) In YOLOZU:

```bash
git submodule add <YOUR_REPO_URL> external/yolozu-infer-cpp
```

Then update docs to point users to that submodule for production inference.
