# RT-DETR Pose Reference Trainer

This folder contains the in-repo reference trainer for the RT-DETR 6DoF pose spec.
Training is supported here as the repo's official RT-DETR pose training lane.

## Quick checks
- Dataset audit (default coco128 path): `python3 tools/dataset_audit.py`
- Dataset audit (with deeper checks): `python3 tools/dataset_audit.py --check-content --check-ranges --fail-on-issues`
- Dataset test: `python3 -m unittest tests/test_dataset.py`

## Reference trainer (CPU/GPU)
- Install deps: `python3 -m pip install -r requirements-test.txt`
- Fetch coco128 (once, from repo root): `bash tools/fetch_coco128.sh`
- Run: `python3 tools/train_minimal.py --epochs 1 --batch-size 2 --max-steps 30`
- GPU example (AMP + accum + standard artifacts):
  - `python3 tools/train_minimal.py --device cuda --amp fp16 --grad-accum 2 --run-dir runs/train_minimal_demo --epochs 1 --max-steps 30`
  - Outputs include `runs/train_minimal_demo/model.onnx` (+ `model.onnx.meta.json`) and checkpoints/metrics.
- Production-style run contract (fixed artifact paths + full resume + export/parity gate; from repo root):
  - `yolozu train configs/examples/train_contract.yaml --run-id exp01` (see `docs/run_contract.md`, requires `yolozu[train]`)
- Continual FT (SDFT-inspired): add a frozen teacher checkpoint via `--self-distill-from <ckpt>` (defaults: keys=logits,bbox; logits uses reverse-KL).
- Continual learning runner (multi-task sequential FT + replay buffer):
  - From this folder (`rtdetr_pose/`): `python3 tools/train_continual.py --config ../configs/continual/rtdetr_pose_domain_inc_example.yaml`
  - From repo root: `python3 rtdetr_pose/tools/train_continual.py --config configs/continual/rtdetr_pose_domain_inc_example.yaml`
- Optional LoRA (parameter-efficient FT / CL):
  - `python3 tools/train_minimal.py --lora-r 8 --lora-freeze-base --lora-target head ...`

Optional full-GT consumption (mask/depth)
- Derive `z` (and `t` if `K_gt` exists) from per-instance `D_obj` at bbox center: `python3 tools/train_minimal.py --use-matcher --z-from-dobj`
- If `M`/`D_obj` are stored as file paths instead of inlined arrays, allow loading: `python3 tools/train_minimal.py --use-matcher --z-from-dobj --load-aux`

Mask-only labels (simple mode)
- If YOLO labels are missing and `mask`/`mask_path` is provided, `--load-aux` will derive bbox+class from the mask.
- `mask_format: color` uses RGB colors as classes (optional `mask_class_map`).
- `mask_format: instance` treats non-zero IDs as instances (single `mask_class_id`, default 0).

Mask-only labels (optional)
- If YOLO txt labels are missing and `labels/<split>/<image>.json` has `mask_path`, the loader will derive bbox labels from the mask.
- Single PNG with multiple classes: use a single-channel mask (values are class ids) or RGB mask (unique colors become classes).
- Multiple PNGs (one per class): set `mask_path` to a list and `mask_classes` to class ids; `mask_instances: true` splits connected components into instances.

## Config
- Example config: `configs/base.json`

## Notes
- The trainer is intended as the repo's reference implementation for training/export/run artifacts; it is not a model-zoo-wide one-click training framework.
- Backbone swapping supports the new nested keys:
  - `model.backbone.name` (e.g. `cspresnet`, `cspdarknet_s`, `resnet50`, `convnext_tiny`)
  - `model.backbone.norm` (`bn|syncbn|frozenbn|gn`)
  - `model.backbone.args` (optional backbone-specific args)
  - `model.projector.d_model` (channel projection size into transformer)
- Legacy `model.backbone_name` / `model.backbone_channels` / `model.stage_blocks` fields remain supported for backward compatibility.
