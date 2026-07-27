# Import adapters (canonical schema projection)

YOLOZU’s fastest path to “use other platform datasets/configs as-is” is:

1) Fix **one internal canonical schema** (YOLOZU meaning is stable)
2) Add platform-specific **Import adapters** (read-only projection into the canonical schema)
3) Keep evaluation apples-to-apples via the `predictions.json` interface contract

This doc describes the **canonical schema** and the current import entry points.

## Canonical schema

### SampleRecord (per image)

Canonical representation is a list of records:

```json
{
  "image": "/abs/or/rel/path.jpg",
  "labels": [
    {
      "class_id": 0,
      "bbox": { "format": "xyxy_abs", "x1": 256, "y1": 192, "x2": 384, "y2": 288 },
      "bbox_cxcywh_norm": { "format": "cxcywh_norm", "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2 },
      "bbox_xywh_abs": { "format": "xywh_abs", "x": 256, "y": 192, "w": 128, "h": 96 }
    }
  ],
  "image_hw": [480, 640]
}
```

Notes:
- Dataset Contract v1 prefers **`xyxy_abs`** for stored boxes.
- `cxcywh_norm` is the YOLO-family adapter view.
- `xywh_abs` is the COCO/export adapter view.
- Optional fields are allowed and may be absent:
  - `mask` / `mask_path`, `depth` / `depth_path` / `D_obj`
  - `pose`, `intrinsics`, `meta`

Implementation references: `yolozu/core/canonical.py`, `yolozu/datasets/dataset_contract.py`.

### TrainConfig (major keys only)

Canonical training config projection focuses on “same meaning” keys:
- `imgsz`, `batch`, `epochs/steps`, `optimizer`, `lr`, `weight_decay`, `seed`, `device`
- plus optional buckets: `preprocess`, `aug`, `loss`, `eval`, `export`

Implementation reference: `yolozu/core/canonical.py` (`TrainConfig`).

## Dataset import (read-only)

Import adapters **do not rewrite the original dataset**. They generate a small descriptor artifact that points to
the original dataset and explains how to interpret it.

### COCO instances JSON (Detectron2 / MMDetection / YOLOX style)

Create a wrapper directory:

```bash
yolozu import dataset \
  --from coco-instances \
  --instances /path/to/instances_val2017.json \
  --images-dir /path/to/images/val2017 \
  --split val2017 \
  --output data/coco_as_is_wrapper \
  --force
```

Then validate or evaluate by pointing at the wrapper:

```bash
yolozu validate dataset data/coco_as_is_wrapper --split val2017
```

## Config import (Mode A / static files)

### YOLO-family args.yaml (YOLOv8 / YOLO11)

```bash
yolozu import config \
  --from auto \
  --args /path/to/runs/.../args.yaml \
  --output reports/train_config_import.json \
  --force
```

This produces a canonical `TrainConfig` JSON (major keys only).

## Config import (Mode B / optional deps; higher compatibility)

These modes may require importing framework-specific config loaders (optional dependencies).

### MMDetection / MMEngine config (python)

```bash
yolozu import config \
  --from mmdet \
  --config /path/to/mmdet_config.py \
  --output reports/train_config_import.json \
  --force
```

Dependency: `mmengine` (example: `pip install mmengine`)

### YOLOX exp (python)

```bash
yolozu import config \
  --from yolox \
  --config /path/to/exps/default/yolox_s.py \
  --output reports/train_config_import.json \
  --force
```

Notes:
- This executes the exp python file (read-only projection, but code is executed).
- Many exp files import `yolox.*`, so you may need YOLOX installed in the environment.

### Detectron2 config

```bash
yolozu import config \
  --from detectron2 \
  --config /path/to/config.yaml \
  --output reports/train_config_import.json \
  --force
```

Dependency: `detectron2` (installation varies by CUDA/PyTorch)

## “It worked as-is” visibility (doctor import)

You can print a resolved summary without writing artifacts:

```bash
# Dataset summary
yolozu doctor import \
  --dataset-from coco-instances \
  --instances /path/to/instances_val2017.json \
  --images-dir /path/to/images/val2017 \
  --split val2017

# Config summary
yolozu doctor import \
  --config-from auto \
  --args /path/to/runs/.../args.yaml
```

Auto-detect mode is also available:

```bash
yolozu doctor import --dataset-from auto --dataset /path/to/dataset_root --split val2017 --output -
yolozu doctor import --dataset-from auto --instances /path/to/instances.json --images-dir /path/to/images --output -
yolozu doctor import --config-from auto --args /path/to/args.yaml --output -
```

When COCO categories include `category_id=0`, doctor/import reports a warning and expects normalized mapping via `classes.json` for fair cross-platform evaluation.

For dataset roots, auto-detect covers the common cases:
- plain COCO root: `images/<split>/` + `annotations/instances_<split>.json`
- COCO keypoints root: `images/<split>/` + `annotations/person_keypoints_<split>.json`
- YOLO/Ultralytics `data.yaml`
- direct YOLO layout: `images/<split>/` + `labels/<split>/`
- YOLOZU wrapper: `dataset.json`
- semantic-segmentation descriptor: `dataset.json` with `task=semantic_segmentation`
- Pascal VOC semantic-segmentation root
- Cityscapes semantic-segmentation root
- ADE20K semantic-segmentation root

`doctor import --dataset-from auto` reports `task_family` and
`reference_trainer` readiness. Use `reference_trainer.direct_train_ready` to
check whether the source can be passed directly to the RT-DETR reference
trainer, and `reference_trainer.train_ready_after_migration` to check whether
`yolozu migrate dataset` / `yolozu import dataset` can produce a train-ready
YOLO label wrapper or record JSON. Native COCO bbox/keypoints roots normally
need that migration step; semantic-segmentation descriptors remain
import/evaluation inputs for the segmentation lane rather than direct
reference-trainer inputs.

For training-specific checks, use `doctor train-dataset`. It accepts the same
source families plus explicit `--from coco-keypoints`, `--from classification`,
`--from obb`, `--from depth`, and `--from pose6d`, validates direct
reference-trainer readiness or records JSON shape, and prints the next command
without launching training. Classification and OBB are recognized intake
interface contracts for external training lanes. Depth and pose6d are direct for the
reference trainer only when the normalized bbox records include the required
sidecars.

```bash
yolozu doctor train-dataset --from auto --dataset /path/to/dataset_root --split val2017 --output -
yolozu doctor train-dataset --from coco-keypoints --dataset /path/to/coco_keypoints_root --split val2017 --output -
yolozu doctor train-dataset --from coco-instances --instances /path/to/instances_val2017.json --images-dir /path/to/images/val2017 --split val2017 --output -
yolozu doctor train-dataset --from pose6d --dataset /path/to/yolozu_pose_wrapper --split train --output -
```

The explicit COCO form requires both `--instances` and `--images-dir`, but not
`--dataset`. Direct YOLO/YOLOZU and records inputs pass the same strict
validation used by `validate dataset`; zero-record splits fail, and
`records.validation` reports whether the full dataset or only the configured
`--max-images` prefix was checked.

## Train shorthand (`train --import`)

For quick demos, you can use a shorthand that resolves external config into canonical `TrainConfig`
and prints doctor-import summary first.

```bash
# YOLO-family runtime
yolozu train --import auto --data /path/to/data.yaml --cfg /path/to/args.yaml

# Auto-detect (from --cfg)
yolozu train --import auto --cfg /path/to/args_or_config.{yaml,py}

# MMDetection
yolozu train --import mmdet --cfg /path/to/config.py
```

Notes:
- This writes resolved config to `reports/train_config_resolved_import.json` by default.
- Add `--resolved-config-out <path>` to change output location.
- If you omit positional `config`, command runs in preview-only mode (no RT-DETR training is launched).

For the repo-side Apache-2.0-friendly external YOLOX lane, use:

```bash
yolozu train \
  --external-backend yolox \
  configs/examples/finetune_external/yolox_s_finetune_smoke.py \
  --dataset data/smoke \
  --split val \
  --dry-run
```

## Goals / non-goals

The goal is compatibility for:
1) dataset reference (paths/splits/classes/label interpretation)
2) major hyperparameters meaning (`imgsz/batch/lr/epochs`)
3) resolved artifacts to make “it worked as-is” visible

Non-goal (initially): perfectly reproducing framework-specific training behavior (aug hooks, schedulers, etc.).

## Adapter starters and parity onboarding

For inference adapters and parity-ready output templates, use:

- `docs/adapter_templates.md`
- `examples/adapter_starters/`

Run adapter parity suite across framework outputs:

```bash
python3 tools/adapter_parity_suite.py \
  --adapter-predictions rtdetr=/path/to/reference.json \
  --adapter-predictions mmdet=/path/to/mmdet.json \
  --adapter-predictions detectron2=/path/to/detectron2.json \
  --adapter-predictions yolo_runtime=/path/to/yolo_runtime.json \
  --adapter-predictions opencv_dnn=/path/to/opencv_dnn.json \
  --adapter-predictions custom_cpp=/path/to/custom_cpp.json \
  --reference-adapter rtdetr \
  --output reports/adapter_parity_suite.json
```
