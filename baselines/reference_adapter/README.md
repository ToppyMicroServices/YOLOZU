# Reference Adapter Baselines

This directory stores reference-adapter baselines used by regression tooling.

## Legacy flat baseline (current CI default)

- adapter: `RTDETRPoseAdapter`
- dataset: `data/smoke` (`split=val`, `max_images=2`)
- baseline: `baselines/reference_adapter/rtdetr_pose_smoke_val.json`
- runner: `tools/run_reference_adapter_regression.py`

Check against flat baseline:

```bash
python3 tools/run_reference_adapter_regression.py \
  --dataset data/smoke \
  --split val \
  --max-images 2 \
  --profile micro \
  --repro-policy relaxed \
  --runtime-lock requirements-ci.lock \
  --baseline baselines/reference_adapter/rtdetr_pose_smoke_val.json \
  --output reports/reference_adapter_regression.json
```

## Matrix baseline layout

Enable matrix layout with:

```bash
--baseline-layout matrix \
--baseline-root baselines/reference_adapter \
--adapter-id rtdetr_pose \
--backend-id torch \
--baseline-version v1 \
--profile micro
```

This resolves to:

`baselines/reference_adapter/rtdetr_pose/torch/<device>/v1/<profile>.json`

Example full profile:

```bash
python3 tools/run_reference_adapter_regression.py \
  --dataset data/smoke \
  --split val \
  --max-images 2 \
  --profile full \
  --baseline-layout matrix \
  --baseline-root baselines/reference_adapter \
  --adapter-id rtdetr_pose \
  --backend-id torch \
  --device cpu \
  --baseline-version v1 \
  --repro-policy strict \
  --capture-provenance full \
  --runtime-lock requirements-ci.lock \
  --output reports/reference_adapter_regression_full.json
```

## Baseline refresh

Refresh baseline only for intentional interface contract or behavior changes:

```bash
python3 tools/run_reference_adapter_regression.py \
  --dataset data/smoke \
  --split val \
  --max-images 2 \
  --profile micro \
  --repro-policy relaxed \
  --runtime-lock requirements-ci.lock \
  --write-baseline \
  --baseline baselines/reference_adapter/rtdetr_pose_smoke_val.json \
  --output reports/reference_adapter_regression_baseline_write.json
```
