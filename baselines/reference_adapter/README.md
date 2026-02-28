# Reference Adapter Baseline

This directory pins the in-repo reference adapter baseline used by CI:

- adapter: `RTDETRPoseAdapter`
- dataset: `data/real_multitask_fewshot` (`split=val`, `max_images=2`)
- gate runner: `tools/run_reference_adapter_regression.py`

Check against baseline:

```bash
python3 tools/run_reference_adapter_regression.py \
  --dataset data/real_multitask_fewshot \
  --split val \
  --max-images 2 \
  --baseline baselines/reference_adapter/rtdetr_pose_real_multitask_fewshot.json \
  --output reports/reference_adapter_regression.json
```

Refresh baseline only for intentional interface/behavior changes:

```bash
python3 tools/run_reference_adapter_regression.py \
  --dataset data/real_multitask_fewshot \
  --split val \
  --max-images 2 \
  --baseline baselines/reference_adapter/rtdetr_pose_real_multitask_fewshot.json \
  --write-baseline \
  --output reports/reference_adapter_regression_baseline_write.json
```
