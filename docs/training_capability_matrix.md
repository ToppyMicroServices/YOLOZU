# Training Capability Matrix

This page is the source of truth for what each training lane claims today.

It complements:

- [Production Readiness](production_readiness.md)
- [Training Backend Interface](training_backend_interface.md)

## Matrix

| Backend | Maturity | Lane | Run contract | Export | Eval | Parity | Resume | MPS |
|---|---|---|---|---|---|---|---|---|
| `reference-rtdetr-pose` | Stable | Reference | Yes | Yes | Yes | Yes | Yes | Qualification path |
| `yolox` | Stable | External | Summary-level | Planned through wrapper lane | Planned through wrapper lane | Planned through wrapper lane | Backend-specific | No blanket claim |
| `detectron2` | Experimental | External | Summary-level | Planned through wrapper lane | Planned through wrapper lane | Planned through wrapper lane | Backend-specific | No blanket claim |
| `ultralytics` | Experimental | Optional external bridge | Summary-level | Planned through wrapper lane | Planned through wrapper lane | Planned through wrapper lane | Backend-specific | No blanket claim |
| `hf-detr` | Experimental | Optional external bridge | Summary-level | Planned through wrapper lane | Planned through wrapper lane | Planned through wrapper lane | Backend-specific | No blanket claim |

## How to read this

- `Run contract = Yes` means the backend owns the richer fixed artifact layout under `runs/<run_id>/`.
- `Run contract = Summary-level` means YOLOZU still emits a common training summary interface contract, but the backend-native artifact layout is not standardized by YOLOZU.
- `MPS` is only claimed when the local runtime actually reports availability. Do not read this table as a blanket macOS guarantee.

## Production posture

- Start with `reference-rtdetr-pose` if you want the richest in-repo training path.
- Prefer `yolox` if you want an Apache-2.0-friendly external YOLO-style lane.
- Use `detectron2` when bbox, instance segmentation, or keypoints training already lives in a Detectron2 stack.
- Treat `ultralytics` and `hf-detr` as environment-qualified bridges.

## Machine-readable source

Implementation reference:

- `yolozu/training/platform.py`

Tooling surface:

- `tools/orchestrate_train.py`
- `tools/support_external_training.py`
- `python3 -m yolozu train`
