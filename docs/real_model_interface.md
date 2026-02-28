# Real model repo + interface (current selection)

Current choice for a "real" training/inference path is the **in-repo** `rtdetr_pose`
scaffold. It is Apache-2.0-compatible and already wired into the adapter layer.

## Entry points

Training (CPU scaffold, metrics output):
- `python3 rtdetr_pose/tools/train_minimal.py --dataset-root data/coco128 --max-steps 50 --metrics-jsonl reports/train_metrics.jsonl --metrics-csv reports/train_metrics.csv`
- Optional checkpoint save: `--checkpoint-out /path/to/checkpoint.pt`
- GPU (AMP + accum + standard artifacts): `python3 rtdetr_pose/tools/train_minimal.py --dataset-root data/coco128 --device cuda --amp fp16 --grad-accum 2 --run-dir runs/train_minimal_demo --epochs 1 --max-steps 30`

Inference / predictions export:
- `python3 tools/export_predictions.py --adapter rtdetr_pose --dataset data/coco128 --checkpoint /path/to/checkpoint.pt --max-images 50 --wrap`
- Torch acceleration knobs: add `--infer-batch-size 8 --torch-compile --torch-compile-backend inductor --torch-compile-mode reduce-overhead --torch-amp bf16 --torch-channels-last --torch-inference-mode`

Scenario runner (metrics pipeline):
- `yolozu test configs/examples/test_setting.yaml --adapter rtdetr_pose --dataset data/coco128 --checkpoint /path/to/checkpoint.pt --max-images 50`
- (source checkout) `python3 tools/run_scenarios.py --adapter rtdetr_pose --dataset data/coco128 --checkpoint /path/to/checkpoint.pt --max-images 50`

Baseline report (real outputs + fps):
- `python3 tools/run_baseline.py --adapter rtdetr_pose --dataset data/coco128 --max-images 50 --output reports/baseline.json`

Reference adapter regression (real-image pinned baseline + CI gates):
- `python3 tools/run_reference_adapter_regression.py --dataset data/real_multitask_fewshot --split val --max-images 2 --device cpu --image-size 160 --score-threshold 0.05 --max-detections 20 --init-seed 2026 --baseline baselines/reference_adapter/rtdetr_pose_real_multitask_fewshot.json --output reports/reference_adapter_regression.json`
- Baseline refresh after intentional behavior changes:
  - `python3 tools/run_reference_adapter_regression.py --dataset data/real_multitask_fewshot --split val --max-images 2 --baseline baselines/reference_adapter/rtdetr_pose_real_multitask_fewshot.json --write-baseline --output reports/reference_adapter_regression_baseline_write.json`

Export (ONNX):
- Prefer `train_minimal.py --run-dir ...` (writes `model.onnx` + `model.onnx.meta.json`).
- Or call `python3 -c "from rtdetr_pose.export import export_onnx; ..."` (see `rtdetr_pose/rtdetr_pose/export.py`).
- Canonical PyTorch → ONNX → TensorRT (engine build): `python3 tools/export_trt.py ...` (see `docs/tensorrt_pipeline.md`).

Backend parity + benchmark (torch vs ONNXRuntime vs TensorRT):
- Export ONNX (and optional engine): `python3 tools/export_trt.py --skip-engine ...`
- Run the suite: `python3 tools/rtdetr_pose_backend_suite.py --config ... --checkpoint ... --onnx ... [--engine ...] --backends torch,onnxrt,trt --output reports/rtdetr_pose_backend_suite.json`
- Notes:
  - ONNXRuntime path needs `onnxruntime` (CPU is fine for CI).
  - TensorRT path needs `tensorrt` + CUDA bindings (`pycuda` or `cuda-python`) and a built engine plan.

## No-torch path (precomputed predictions)

If PyTorch is unavailable locally (e.g., macOS), you can still run the pipeline
using **precomputed** predictions JSON:

- Generate predictions JSON from your external inference code.
- Ensure it matches the YOLOZU schema (see README predictions schema).
- Run scenario runner with the precomputed adapter:

```bash
yolozu test configs/examples/test_setting.yaml \
  --adapter precomputed \
  --predictions /path/to/predictions.json \
  --dataset /path/to/coco-yolo \
  --max-images 50

# (source checkout)
python3 tools/run_scenarios.py \
  --adapter precomputed \
  --predictions /path/to/predictions.json \
  --dataset /path/to/coco-yolo \
  --max-images 50
```

## Adapter interface (expected inputs/outputs)

Input records (from `yolozu.datasets.dataset.build_manifest`):
- list of dicts with `image` path and optional labels

Output per image (from `RTDETRPoseAdapter`):
- `image`: original path
- `detections`: list of dicts with `class_id`, `score`, `bbox` (cxcywh_norm), and optional pose fields

### Units & intrinsics (common pitfall)

- `intrinsics` / `K` / `K_gt` are expected in **pixel units** $(fx, fy, cx, cy)$.
- They must match the **image coordinate system used by the model outputs** (i.e., after any resize/letterbox preprocessing).
- `log_z`/`z` and the derived `t_xyz` are in the **same length unit as your dataset** (YOLOZU does not convert mm↔m).
- `k_delta` is a small correction on top of a provided baseline intrinsics; it is not a per-image Newton/Hessian optimizer.

## Deterministic predictions (for tests)

- Use fixed seeds (`--seed`) and `--deterministic` in `train_minimal.py`
- For `RTDETRPoseAdapter` inference without a checkpoint, use `--init-seed` in `tools/run_reference_adapter_regression.py`
- Limit to `data/coco128` and set `--max-images` for quick runs
- Keep `--image-size`, `--score-threshold`, and `--max-detections` fixed
- Record the config + checkpoint hash in run notes

## Gaps / follow-ups

- `train_minimal.py` is a scaffold and not intended to reach strong mAP.
- For competitive results, plug in a full training repo and keep the adapter contract identical.
