# Training Capability Matrix

This page is the source of truth for what each training lane claims today.

It complements:

- [Production Readiness](production_readiness.md)
- [Training Backend Interface](training_backend_interface.md)

## Matrix

| Backend | Maturity | Lane | Run contract | Export | Eval | Parity | Resume | MPS |
|---|---|---|---|---|---|---|---|---|
| `reference-rtdetr-pose` | Stable | Reference | Yes | Yes | Yes | Yes | Yes | Qualification path |
| `yolox` | Stable | External | External run contract | Wrapper-ready | Wrapper-ready | Wrapper-ready | Backend-specific | No blanket claim |
| `detectron2` | Experimental | External | External run contract | Wrapper-ready | Wrapper-ready | Wrapper-ready | Backend-specific | No blanket claim |
| `mmdetection` | Experimental | External | External run contract | Wrapper-ready (bbox primary) | Wrapper-ready (bbox primary) | Wrapper-ready (bbox primary) | Backend-specific | No blanket claim |
| `mmpose` | Experimental | External | External run contract | Backend-specific exporter | Wrapper-ready | Wrapper-ready | Backend-specific | No blanket claim |
| `mmseg` | Experimental | External | External run contract | Backend-specific exporter | Wrapper-ready | Backend-specific | Backend-specific | No blanket claim |
| `ultralytics` | Experimental | Optional external bridge | External run contract | Wrapper-ready | Wrapper-ready | Wrapper-ready | Backend-specific | No blanket claim |
| `hf-detr` | Experimental | Optional external bridge | External run contract | Wrapper-ready | Wrapper-ready | Wrapper-ready | Backend-specific | No blanket claim |

## How to read this

- `Run contract = Yes` means the backend owns the richer fixed artifact layout under `runs/<run_id>/`.
- `Run contract = External run contract` means YOLOZU standardizes `work_dir/dataset/`, `work_dir/configs/train_config_projection.json`, and `work_dir/reports/{training_summary,external_run_meta,launcher_plan,execution}.json` even when the backend-native trainer remains external.
- `MPS` is only claimed when the local runtime actually reports availability. Do not read this table as a blanket macOS guarantee.

## Production posture

- Start with `reference-rtdetr-pose` if you want the richest in-repo training path.
- Prefer `yolox` if you want an Apache-2.0-friendly external YOLO-style lane.
- Use `detectron2` when bbox, instance segmentation, or keypoints training already lives in a Detectron2 stack.
- Use `mmdetection` when bbox or instance-seg training already lives in an OpenMMLab detection stack.
- Use `mmpose` for keypoints/pose training when the backend-native pipeline already lives in MMPose.
- Use `mmseg` for semantic segmentation training when the backend-native pipeline already lives in MMSeg.
- Use `reference-rtdetr-pose` when the task extends into depth or pose6d training.
- Treat `ultralytics` and `hf-detr` as environment-qualified bridges.
- OpenCV DNN and ONNX Runtime are not training backends in YOLOZU. They stay in export / inference / parity lanes.

## Machine-readable source

Implementation reference:

- `yolozu/training/platform.py`

Tooling surface:

- `tools/orchestrate_train.py`
- `tools/support_external_training.py`
- `python3 -m yolozu train`
