# Training Backend Interface

This page defines the backend-neutral surface that YOLOZU uses for training lanes.

The goal is not to force every backend into the same internal trainer loop.
The goal is to make different training lanes look comparable at the orchestration layer.

## Shared concepts

Every training backend should expose the same high-level ideas:

- one canonical `TrainConfig`
- one backend id
- one machine-readable training run summary
- one machine-readable resume handoff
- one standardized export / eval / parity handoff bundle
- one optional training registry entry
- one clear statement of whether export / eval / parity are supported

YOLOZU keeps the in-repo RT-DETR pose trainer as the reference lane, then treats
YOLOX, Detectron2, MMDetection, MMPose, MMSeg, TAO, Ultralytics, and HF DETR as external lanes that still publish the same
top-level summary interface contract plus one standardized external run bundle.

## Training data flow

All detector-family training lanes use the same data route:

```text
raw dataset
  -> DatasetAdapter
  -> YOLOZU Dataset Contract
  -> TrainingBackend
      -> YOLO family
      -> DETR family
```

The DatasetAdapter is the only layer that handles source-format differences
such as YOLO `data.yaml`, `dataset.json` wrappers, COCO JSON, and SynthGen
shards. The YOLOZU Dataset Contract is the training boundary: dataset records
prefer `xyxy_abs` storage and expose `xywh_abs` and `cxcywh_norm` adapter
views when image size is available. Backends select their family-specific view
after this boundary.

Training summaries expose this as `training_data_flow` with
`format = yolozu_training_data_flow_v1`.

## Supported model-family lanes

YOLOZU supports two first-class detector-family routes at the training surface:

- `reference-rtdetr-pose`: the in-repo RT-DETR-family reference lane.
- `yolox` / optional YOLO bridges: YOLO-family external lanes.

They intentionally do not share optimizer, preprocessing, or postprocess defaults:

| Family | Primary lane | Optimizer policy | Preprocess policy | Postprocess policy |
|---|---|---|---|---|
| RT-DETR / DETR-family | `reference-rtdetr-pose` | AdamW with separate backbone/head parameter groups; backbone LR is normally lower than head LR. | Reference trainer transforms; do not assume YOLO letterbox inside the DETR trainer loop. | NMS-free by default; use `e2e_nms_free` when recording the eval protocol. |
| YOLO-family | `yolox` and optional YOLO bridges | SGD with momentum/Nesterov-style YOLO defaults unless the backend config overrides them. | Letterbox resize/pad is part of the model/export assumption and must be recorded in metadata. | NMS-applied predictions by default; use `nms_applied` when recording the eval protocol. |

RT-DETR stability knobs are part of the reference trainer surface: gradient clipping,
LR warmup, EMA, optional AMP, and strict run-contract mode. The example recipe is
[`../configs/examples/train_rtdetr_stable.yaml`](../configs/examples/train_rtdetr_stable.yaml).
The YOLOX smoke exp records the YOLO-family counterpart:
[`../configs/examples/finetune_external/yolox_s_finetune_smoke.py`](../configs/examples/finetune_external/yolox_s_finetune_smoke.py).

## Backend ids

Current backend ids:

- `reference-rtdetr-pose`
- `yolox`
- `detectron2`
- `mmdetection`
- `mmpose`
- `mmseg`
- `tao`
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

For RT-DETR-family runs, populate `optimizer=adamw`, `use_param_groups=true`,
`backbone_lr_mult < head_lr_mult`, and `clip_grad_norm > 0` for stable fine-tuning.
For YOLO-family runs, keep `preprocess.method=letterbox` and `eval.protocol=nms_applied`
when exporting/evaluating backend predictions.

Implementation reference:

- `yolozu/core/canonical.py`

## Shared training run summary

Every backend-level training lane should be able to emit:

- `format = yolozu_training_run_summary_v1`
- `backend`
- `canonical_train_config`
- `training_data_flow`
- `run_output_contract`
- `steps.train`
- `steps.resume`
- `steps.export`
- `steps.eval`
- `steps.parity`
- `next_steps`

This is the common top-level summary interface contract for training.
Schema: [`schemas/training_run_summary.schema.json`](schemas/training_run_summary.schema.json).

`next_steps` is the standardized hand-off list of copy-paste commands that move a
completed run into resume, export, evaluation, or parity.
Each step records `stage`, `command`, `input_contract`, and `output_contract` so
agents and downstream automation can route the handoff without scraping prose.

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
- `work_dir/reports/resume_handoff.json`
- `work_dir/reports/export_handoff.json`
- `work_dir/reports/eval_handoff.json`
- `work_dir/reports/parity_handoff.json`
- `work_dir/reports/training_registry_entry.json`

Each `*_handoff.json` follows
[`schemas/training_handoff.schema.json`](schemas/training_handoff.schema.json).

The handoff JSON files make resume / export / eval / parity machine-readable even when the
backend runtime itself stays external. This is how `mmpose`, `mmseg`, and `tao` stop being
ad-hoc handoff lanes: YOLOZU now fixes the accepted handoff format even when the
final exporter is still launched from the backend side.

OpenCV DNN and ONNX Runtime do not appear in this list because YOLOZU treats them as
export / inference / parity runtimes, not as training backends.

## Why this matters

This separation makes it possible to:

- compare training lanes without conflating them with one trainer implementation
- orchestrate multi-backend experiments from one spec
- append executed runs into one JSONL registry for later audit/replay
- expose a capability matrix without pretending every backend is equally complete

## Related docs

- [Training, inference, and export](training_inference_export.md)
- [Run Contract](run_contract.md)
- [Training capability matrix](training_capability_matrix.md)
- [Training orchestration](training_orchestration.md)
