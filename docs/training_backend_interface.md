# Training Backend Interface

This page defines the backend-neutral surface that YOLOZU uses for training lanes.

The goal is not to force every backend into the same internal trainer loop.
The goal is to make different training lanes look comparable at the orchestration layer.

## Shared concepts

Every training backend should expose the same high-level ideas:

- one canonical `TrainConfig`
- one backend id
- one machine-readable training run summary
- one clear statement of whether export / eval / parity are supported

YOLOZU keeps the in-repo RT-DETR pose trainer as the reference lane, then treats
YOLOX, Detectron2, Ultralytics, and HF DETR as external lanes that still publish the same
top-level summary interface contract plus one standardized external run bundle.

## Backend ids

Current backend ids:

- `reference-rtdetr-pose`
- `yolox`
- `detectron2`
- `ultralytics`
- `hf-detr`

Implementation reference:

- `yolozu/training/platform.py`

## Canonical TrainConfig

The canonical training config is backend-independent. It is a projection layer,
not a claim that every backend supports every field in the same way.

Important fields:

- `backend`
- `task`
- `model`
- `imgsz`
- `batch`
- `epochs`
- `steps`
- `optimizer`
- `lr`
- `weight_decay`
- `seed`
- `device`
- `precision`
- `workers`
- `grad_clip_norm`
- `dataset`
- `preprocess`
- `eval`
- `export`
- `run_contract`
- `source`

Implementation reference:

- `yolozu/core/canonical.py`

## Shared training run summary

Every backend-level training lane should be able to emit:

- `format = yolozu_training_run_summary_v1`
- `backend`
- `canonical_train_config`
- `run_output_contract`
- `steps.train`
- `steps.export`
- `steps.eval`
- `steps.parity`
- `next_steps`

This is the common top-level summary interface contract for training.

`next_steps` is the standardized hand-off list of copy-paste commands that move a
completed run into export, evaluation, or parity.

The reference trainer also emits richer artifacts such as:

- `reports/run_meta.json`
- `reports/training_summary.json`
- `exports/model.onnx`
- `reports/onnx_parity.json`

External lanes may not emit the same backend-native checkpoint layout, but they do emit
the same top-level training summary shape and the same fixed wrapper artifacts under
`work_dir/reports/` and `work_dir/configs/`.

The standardized external bundle is:

- `work_dir/configs/train_config_projection.json`
- `work_dir/reports/training_summary.json`
- `work_dir/reports/external_run_meta.json`
- `work_dir/reports/launcher_plan.json`
- `work_dir/reports/execution.json`

## Why this matters

This separation makes it possible to:

- compare training lanes without conflating them with one trainer implementation
- orchestrate multi-backend experiments from one spec
- expose a capability matrix without pretending every backend is equally complete

## Related docs

- [Training, inference, and export](training_inference_export.md)
- [Run Contract](run_contract.md)
- [Training capability matrix](training_capability_matrix.md)
- [Training orchestration](training_orchestration.md)
