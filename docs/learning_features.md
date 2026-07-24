# Learning features

This page collects training and adaptation workflows that are intentionally *optional* in YOLOZU.
The README stays focused on evaluation entrypoints; use this doc when you want opt-in research workflows such as continual learning, test-time adaptation, distillation, or Hessian refinement after the stable evaluation path has produced an artifact.

## 1) Run interface contract training (Run Contract: reproducible artifacts)

Value: Reproducible training operations for detection + keypoints + 6DoF pose: pin artifacts (checkpoints / metrics / exports / parity) under `runs/<run_id>/` so runs are easy to compare, regression-check, and fully resume.

Representative command:

```bash
yolozu train configs/examples/train_contract.yaml --run-id exp01
```

Artifacts (fixed paths):
- `runs/<run_id>/checkpoints/{last,best}.pt`
- `runs/<run_id>/reports/{train_metrics,val_metrics}.jsonl`
- `runs/<run_id>/reports/{config_resolved.yaml,run_meta.json,onnx_parity.json}`
- `runs/<run_id>/exports/model.onnx` (+ meta)

Model variants: swap backbones (ResNet/ConvNeXt/CSP/...) while keeping the same artifact layout: [`docs/backbones.md`](backbones.md).

Details: [`docs/run_contract.md`](run_contract.md), [`docs/training_inference_export.md`](training_inference_export.md).

## 2) Continual learning (anti-forgetting across task/domain sequences)

Value: Fine-tune across a task/domain sequence while measuring and mitigating catastrophic forgetting via (a) memoryless self-distillation, (b) optional replay buffer, and (c) optional parameter-efficient updates (LoRA) + regularizers (EWC/SI/DER++).
This is an opt-in research workflow and should be promoted only through explicit evaluation and decision reports.

If you are new to LoRA / QLoRA, start with the plain-language diagrams in [`docs/continual_learning.md`](continual_learning.md) before reading the full config tables.

Representative commands:

```bash
python3 rtdetr_pose/tools/train_continual.py \
  --config configs/continual/rtdetr_pose_domain_inc_example.yaml

python3 tools/eval_continual.py \
  --run-json runs/continual/<run>/continual_run.json \
  --device cpu \
  --max-images 50
```

Artifacts:
- `runs/continual/<run>/continual_run.json` (single source of truth)
- `runs/continual/<run>/replay_buffer.json` (+ per-task `replay_records.json`)
- `runs/continual/<run>/continual_eval.{json,html}` (from `eval_continual.py`)

Details: [`docs/continual_learning.md`](continual_learning.md).

## 3) Test-time training (TTT) under domain shift (Tent / MIM / CoTTA / EATA / SAR)

Value: Reproducible test-time adaptation with bounded cost caps, reset policies (`stream` vs `sample`), and fixed eval subsets for fair comparisons.
This is an opt-in research workflow; default export/eval commands do not enable `--ttt`.

Representative command (export predictions with TTT enabled):

```bash
python3 -m yolozu export \
  --backend torch \
  --dataset data/coco128 \
  --split train2017 \
  --checkpoint runs/exp01/checkpoints/best.pt \
  --device cuda \
  --max-images 50 \
  --ttt --ttt-preset safe --ttt-reset sample \
  --ttt-log-out reports/ttt_log_safe.json \
  --output reports/pred_ttt_safe.json
```

Artifacts:
- `reports/pred_ttt_safe.json` (predictions interface contract)
- `reports/ttt_log_safe.json` (TTT step log)
- Optional: fixed subset artifacts via `tools/make_subset_dataset.py` (`subset.json`, `subset_images.txt`)

Task-aware knobs: `--ttt-sdft-task {pose,keypoints,depth,seg,full}` and `--ttt-aux-*` consistency weights for multi-task heads.

Details: [`docs/ttt_protocol.md`](ttt_protocol.md).

## 4) (Research helper) Prediction distillation (offline)

Value: Blend teacher/student `predictions.json` artifacts to accelerate ablations **without retraining**. This is useful when you want to test whether a stronger teacher contains helpful signal before investing in training-time distillation.

The distillation guide now explains the workflow in human terms first: what counts as a matched detection, what a teacher-only injection means, and how to read the report without starting from raw metrics.

Representative command:

```bash
python3 tools/distill_predictions.py \
  --student reports/predictions_student.json \
  --teacher reports/predictions_teacher.json \
  --dataset data/coco128 \
  --split val2017 \
  --config configs/examples/distill_predictions.yaml \
  --output reports/predictions_distilled.json \
  --output-report reports/distill_report.json
```

Artifacts:
- `reports/predictions_distilled.json`
- `reports/distill_report.json`

Read the outputs in this order:
- `distill_report.json` first
- distilled predictions second
- then the full evaluator if the quick signal looks promising

Important distinction:
- this helper is **offline prediction distillation**
- it is **not** training-time self-distillation for continual learning
- it is **not** TTT / CTTA

Details: [`docs/distillation.md`](distillation.md).

## 5) Hessian-based refinement (post-inference, per-detection; experimental)

Value: A safe Newton / finite-diff Hessian stepper to refine pose-related prediction fields as an engine-external postprocess over `predictions.json`.
This is an opt-in offline analysis or controlled-study workflow; default evaluation does not run Hessian refinement.

Read [`docs/hessian_solver.md`](hessian_solver.md) if you want the plain-language intuition for why pose and geometry-heavy outputs benefit from this kind of local second-order correction.

Representative command:

```bash
python3 tools/refine_predictions_hessian.py \
  --predictions reports/predictions.json \
  --output reports/predictions_hessian.json \
  --enable \
  --device cpu \
  --log-output reports/hessian_log.json
```

Artifacts:
- `reports/predictions_hessian.json` (predictions interface contract)
- `reports/hessian_log.json` (optional)

Details: [`docs/hessian_solver.md`](hessian_solver.md).

## 6) Long-tail recipe with PyTorch-selectable plugins

Value: Generate a reproducible long-tail training plan with explicit PyTorch-ready choices for loss, validation metric, and lr scheduler.

Representative command:

```bash
python3 -m yolozu long-tail-recipe \
  --dataset data/smoke \
  --split val \
  --loss-plugin torch_cross_entropy \
  --metric-plugin torch_top1_accuracy \
  --lr-scheduler torch_step_lr \
  --output reports/long_tail_recipe_torch.json \
  --force
```

PyTorch-oriented options:
- `--loss-plugin`: `torch_cross_entropy`, `torch_nll_loss`, `torch_bce_with_logits`
- `--metric-plugin`: `torch_top1_accuracy`, `torch_top5_accuracy`, `torch_cross_entropy`
- `--lr-scheduler`: `torch_step_lr`, `torch_cosine_annealing_lr`, `torch_reduce_on_plateau`, `torch_one_cycle_lr`

Output (`recipe.plugins`) now includes `loss`, `metric`, and `lr_scheduler` selections and monitor metadata for downstream training wiring.

## 7) PyTorch utility wrappers (`yolozu.training.torch_utils`)

Value: Thin, composable helpers wrapping core PyTorch APIs (`torch.amp`, `torch.compile`, `torch.profiler`, `torch.nn`) for common training/inference tasks.

| Helper | PyTorch API | Purpose |
|--------|------------|---------|
| `amp_inference_context()` | `torch.amp.autocast` | Mixed-precision inference context manager |
| `compile_model()` | `torch.compile` | JIT-compile a model with configurable backend/mode |
| `profile_callable()` | `torch.profiler` | Profile any callable and return structured summary |
| `model_info()` | `torch.nn.Module` | Parameter count, dtype breakdown, device, buffer stats |
| `auto_device()` | `torch.cuda` / `torch.backends.mps` | Auto-detect best device (CUDA → MPS → CPU) |
| `seed_everything()` | `torch.manual_seed` | Set all random seeds for reproducibility |

Example usage:

```python
from yolozu.training.torch_utils import (
    amp_inference_context, compile_model, model_info, auto_device, seed_everything,
)

seed_everything(42)
device = auto_device()
model = model.to(device)
compiled = compile_model(model, mode="reduce-overhead")
print(model_info(compiled))

with amp_inference_context(device.type):
    output = compiled(images)
```

## 8) PyTorch inference acceleration (`yolozu.inference.torch_export`, `yolozu.inference.profiler`)

Value: Export and compile PyTorch models for production inference, with kernel-level profiling.

| Module | PyTorch API | Purpose |
|--------|------------|---------|
| `torch_export.compile_for_inference()` | `torch.compile` | Track requested versus actual compilation and fail closed through the first execution unless eager fallback is explicit |
| `torch_export.export_model_onnx()` | `torch.onnx.export` | Export model to ONNX with dynamic axes |
| `profiler.profile_inference()` | `torch.profiler` | Profile adapter inference with Chrome trace output |

`compile_for_inference()` attaches JSON-safe evidence retrievable with
`get_compile_evidence()`. A successful setup remains
`pending_first_execution` until the returned model completes a call. API,
setup, or lazy execution failure raises `TorchCompileError` by default.
`allow_fallback=True` permits eager execution and records `status=fallback`;
it does not count as compiled evidence.

## 9) Training AMP + transforms bridge (`yolozu.training.amp_utils`, `yolozu.training.transforms_bridge`)

Value: Standardize AMP across all training loops (SDFT, TTA, custom), and provide composable detection transforms via torchvision v2.

| Module | PyTorch API | Purpose |
|--------|------------|---------|
| `amp_utils.make_amp_context()` | `torch.amp` + `torch.GradScaler` | Unified AMP context + scaler for training |
| `transforms_bridge.build_detection_transforms()` | `torchvision.transforms.v2` | Joint image + bbox + keypoint augmentation |
| `transforms_bridge.build_eval_transforms()` | `torchvision.transforms.v2` | Eval-time resize + normalize |
