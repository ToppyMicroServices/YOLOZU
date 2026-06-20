# Training Capability Matrix

This page is the source of truth for what each training lane claims today.

It complements:

- [Production Readiness](production_readiness.md)
- [Training Backend Interface](training_backend_interface.md)

## Matrix

| Backend | Maturity | Lane | Run contract | Export | Eval | Parity | Resume | MPS |
|---|---|---|---|---|---|---|---|---|
| `reference-rtdetr-pose` | Stable | Reference | Yes | Yes | Yes | Yes | Yes | Qualification path |
| `yolox` | Stable | External | External run contract | Wrapper-ready | Wrapper-ready | Wrapper-ready | Shared resume handoff | No blanket claim |
| `detectron2` | Experimental | External | External run contract | Wrapper-ready | Wrapper-ready | Wrapper-ready | Shared resume handoff | No blanket claim |
| `mmdetection` | Experimental | External | External run contract | Wrapper-ready (bbox / instance-seg handoff) | Wrapper-ready | Wrapper-ready | Shared resume handoff | No blanket claim |
| `mmpose` | Experimental | External | External run contract | Standardized COCO-keypoints-to-predictions bridge | Wrapper-ready | Wrapper-ready | Shared resume handoff | No blanket claim |
| `mmseg` | Experimental | External | External run contract | Standardized mask-packaging bridge | Wrapper-ready | Wrapper-ready | Shared resume handoff | No blanket claim |
| `tao` | Experimental | Qualified external bridge | External run contract | Wrapper-ready | Wrapper-ready | Wrapper-ready | Shared resume handoff | No blanket claim |
| `ultralytics` | Experimental | Optional external bridge | External run contract | Wrapper-ready | Wrapper-ready | Wrapper-ready | Shared resume handoff | No blanket claim |
| `hf-detr` | Experimental | Optional external bridge | External run contract | Wrapper-ready | Wrapper-ready | Wrapper-ready | Shared resume handoff | No blanket claim |

## Detector-family policy

| Backend | Family | Optimizer policy | Preprocess policy | Postprocess policy |
|---|---|---|---|---|
| `reference-rtdetr-pose` | RT-DETR / DETR-family | AdamW with separate backbone/head parameter groups; use a lower backbone LR for stable fine-tuning. | Reference trainer transforms; do not assume YOLO letterbox in the DETR trainer loop. | NMS-free by default; record `e2e_nms_free` when applicable. |
| `yolox` | YOLO-family | SGD with momentum/Nesterov-style YOLO defaults unless the exp overrides them. | Letterbox resize/pad is part of the export/eval assumption. | NMS-applied predictions by default; record `nms_applied`. |
| `ultralytics` | YOLO-family | Backend-native YOLO optimizer settings from the user-provided runtime/config. | Preserve YOLO-family letterbox assumptions in export/eval metadata. | NMS-applied predictions by default; keep the runtime/license boundary explicit. |
| `hf-detr` | DETR-family | External trainer should supply DETR-family AdamW-style settings. | Record backend-native DETR preprocessing through the external run handoff. | NMS-free outputs when supported by the external stack. |

## How to read this

- `Run contract = Yes` means the backend owns the richer fixed artifact layout under `runs/<run_id>/`.
- `Run contract = External run contract` means YOLOZU standardizes `work_dir/dataset/`, `work_dir/configs/train_config_projection.json`, `work_dir/reports/{training_summary,external_run_meta,launcher_plan,execution}.json`, `reports/resume_handoff.json`, the export/eval/parity handoff JSON files, and one registry entry even when the backend-native trainer remains external.
- External training reports include `execution_status.state` so callers can distinguish `dry_run_handoff`, `requires_external_train_script`, `runtime_failed`, and `executed` without treating every wrapper report as real backend training.
- `MPS` is only claimed when the local runtime actually reports availability. Do not read this table as a blanket macOS guarantee.

## Production posture

- Start with `reference-rtdetr-pose` if you want the richest in-repo training path.
- Prefer `yolox` if you want an Apache-2.0-friendly external YOLO-style lane.
- Use `detectron2` when bbox, instance segmentation, or keypoints training already lives in a Detectron2 stack.
- Use `mmdetection` when bbox or instance-seg training already lives in an OpenMMLab detection stack.
- Use `mmpose` for keypoints/pose training when the backend-native pipeline already lives in MMPose; the recommended export handoff is COCO keypoints results JSON normalized into the predictions interface contract.
- Use `mmseg` for semantic segmentation training when the backend-native pipeline already lives in MMSeg; the recommended export handoff is class-id mask packaging into the segmentation predictions interface contract.
- Use `tao` when an NVIDIA TAO stack already owns the trainer/runtime environment and you want YOLOZU to normalize resume/export/eval/parity handoff around that external lane.
- Use `reference-rtdetr-pose` when the task extends into depth or pose6d training.
- Treat `ultralytics` and `hf-detr` as environment-qualified optional bridges. Their dry-run reports include `runtime_license_boundary` so the runtime/license boundary is preserved without implying the third-party runtime is bundled or installed by default.
- OpenCV DNN and ONNX Runtime are not training backends in YOLOZU. They stay in export / inference / parity lanes.

## Machine-readable source

Implementation reference:

- `yolozu/training/platform.py`

Tooling surface:

- `tools/orchestrate_train.py`
- `tools/support_external_training.py`
- `tools/export_predictions_coco_keypoints.py`
- `tools/package_segmentation_predictions.py`
- `tools/check_segmentation_parity.py`
- `python3 -m yolozu train`
