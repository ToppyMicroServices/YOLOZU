# Repo Map

This workspace contains a **contract-first evaluation harness** plus an **in-repo reference trainer** (RT-DETR pose).
It also provides an **external training lane** for Apache-2.0-friendly YOLOX-style training, with optional copyleft-sensitive bridges kept explicit.

Two usage modes:
- **pip users**: `pip install yolozu` → stable, CPU-friendly CLI (`yolozu doctor|export|validate|eval-instance-seg|resources|demo`)
- **repo users**: source checkout unlocks additional tools (`tools/`, `rtdetr_pose/`, scenario suite runners, TensorRT pipeline helpers)

## Key paths
- `yolozu/`: pip-installable package (CLI + schemas + demos)
  - `yolozu/core/`: foundational utilities (config, boxes, keypoints, image_keys, letterbox, resources)
  - `yolozu/datasets/`: dataset adapters, registry, fetch, validation, migration (COCO, VOC, ADE20K, Cityscapes)
  - `yolozu/eval/`: evaluation metrics (COCO AP, keypoints, segmentation, long-tail, continual, pose)
  - `yolozu/predictions/`: predictions I/O, schema governance, transforms, parity checks
  - `yolozu/inference/`: model adapters, inference engine, pipeline, ONNX export, model fetch
  - `yolozu/geometry/`: 3-D geometry, camera intrinsics, constraints, template verification
  - `yolozu/training/`: training helpers (distillation, gates, replay buffer, SDFT, continual regularizers)
- `docs/`: user-facing docs (protocols, pipelines, recipes)
- `tests/`: unit/integration tests (CPU-friendly by default; GPU optional)
- `tools/`: repo-only scripts (exporters, suites, benchmarks, smoke runs)
  - `tools/support_external_training.py`: external training bridge (YOLOX primary, optional Ultralytics/HF bridges)
- `rtdetr_pose/`: RT-DETR pose reference trainer (training/inference/export helpers)
- `data/smoke/`: committed offline smoke assets (10 images + labels + fixed predictions)
- `data/coco128/`: tiny COCO dataset for extended local checks (downloaded via `tools/fetch_coco128.sh`)

## Module path policy
- Canonical modules are the categorized package paths above.
- Top-level shim files under `yolozu/*.py` were removed; legacy imports are resolved via `yolozu.__init__` module aliasing.
- New code should import canonical paths directly (example: `from yolozu.datasets.dataset import build_manifest`).
