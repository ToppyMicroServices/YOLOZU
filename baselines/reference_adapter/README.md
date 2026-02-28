# Reference Adapter Baseline

This directory pins the in-repo reference adapter baseline used by CI:

- adapter: `RTDETRPoseAdapter`
- dataset: `data/smoke` (`split=val`, `max_images=2`)
- gate runner: `tools/run_reference_adapter_regression.py`
- baseline metadata: `baseline_meta` (weights/config/dataset/env hashes)
- protocol metadata: `protocol` + `gate_policy`

Check against baseline:

```bash
python3 tools/run_reference_adapter_regression.py \
  --dataset data/smoke \
  --split val \
  --max-images 2 \
  --repro-policy relaxed \
  --baseline baselines/reference_adapter/rtdetr_pose_smoke_val.json \
  --output reports/reference_adapter_regression.json
```

Contract-only hard gate:

```bash
python3 tools/run_reference_adapter_regression.py \
  --dataset data/smoke \
  --split val \
  --max-images 2 \
  --score-gate-mode off \
  --perf-gate-mode off \
  --baseline baselines/reference_adapter/rtdetr_pose_smoke_val.json \
  --output reports/reference_adapter_regression_contract.json
```

Behavior-only warn gate:

```bash
python3 tools/run_reference_adapter_regression.py \
  --dataset data/smoke \
  --split val \
  --max-images 2 \
  --schema-gate-mode off \
  --consistency-gate-mode off \
  --score-gate-mode warn \
  --perf-gate-mode warn \
  --baseline baselines/reference_adapter/rtdetr_pose_smoke_val.json \
  --output reports/reference_adapter_regression_behavior.json
```

Refresh baseline only for intentional interface/behavior changes:

```bash
python3 tools/run_reference_adapter_regression.py \
  --dataset data/smoke \
  --split val \
  --max-images 2 \
  --repro-policy relaxed \
  --baseline baselines/reference_adapter/rtdetr_pose_smoke_val.json \
  --write-baseline \
  --output reports/reference_adapter_regression_baseline_write.json
```
