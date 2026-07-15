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
  --runtime-lock requirements-locks/requirements-ci.lock \
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
  --runtime-lock requirements-locks/requirements-ci.lock \
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
  --runtime-lock requirements-locks/requirements-ci.lock \
  --write-baseline \
  --baseline baselines/reference_adapter/rtdetr_pose_smoke_val.json \
  --output reports/reference_adapter_regression_baseline_write.json
```

## Config fingerprint-only updates

`baseline_meta.config_hash` is the raw SHA256 of the configured JSON file. A config edit can therefore change this fingerprint even when it explicitly selects the existing disabled/default behavior.

Before changing only this field:

1. Identify the exact config diff and verify that old and new values normalize to the same runtime behavior.
2. Confirm that the interface contract gate reports only `config_hash` drift; weights, dataset, runtime lock, schema, and record mapping must still match.
3. Update only `baseline_meta.config_hash`. Do not regenerate behavior metrics or timestamps unless the executed behavior changed.
4. Run the hard interface contract command and the tracked fingerprint test.

On 2026-07-15, the fingerprint was updated from `369aa76e...` to `4a02c59a...` after commit `9481882` explicitly added `model.graph_refine.mode=none` to `rtdetr_pose/configs/base.json`. Both the omitted field and explicit `none` normalize to `{"enabled": false}`; the existing behavior, weights, dataset, and runtime-lock baseline fields were preserved.
