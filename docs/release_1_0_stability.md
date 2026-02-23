# 1.0.0 stability boundary (contract-first)

This document defines what YOLOZU 1.0.0 treats as a stable compatibility surface.

## Scope

YOLOZU 1.0.0 stabilizes **contracts**, not every internal implementation detail.

Stable for 1.x:
- Predictions schema v1 (`schemas/predictions.schema.json`)
- Adapter contract v1 (`docs/adapter_contract.md`)
- Run contract artifact layout (`docs/run_contract.md`)
- Core CLI compatibility surface listed below

Not guaranteed stable:
- Internal model architectures/backbone internals
- Experimental scripts outside documented CLI contract
- GPU environment-specific runtime behavior (driver/container constraints)

## Contract guarantees

### 1) Predictions schema v1

Reference: `docs/predictions_schema.md`

Guarantee:
- `schema_version: 1` payloads remain valid across 1.x.
- Backward-compatible additions are allowed (new optional fields).
- Breaking changes require schema version bump and migration notes.

### 2) Adapter contract v1

Reference: `docs/adapter_contract.md`

Guarantee:
- Adapter `predict(records)` signature and required output keys remain stable:
  - `image`
  - `detections[]`
  - detection required keys: `class_id`, `score`, `bbox`
- `bbox` semantic interpretation remains explicit via evaluator/export settings (`bbox_format`).
- `export_settings` provenance in generated reports is preserved as contract metadata.

### 3) Run contract

Reference: `docs/run_contract.md`

Guarantee:
- Contracted run artifacts remain at fixed paths under `runs/<run_id>/...`.
- Required artifacts (`checkpoints`, metrics JSONL, resolved config, run meta) remain stable.
- ONNX export + parity report paths remain stable when enabled by run contract settings.

## CLI compatibility surface (1.x)

Primary stable commands:
- `yolozu doctor`
- `yolozu validate` (`predictions`, `dataset`, `seg`, `instance-seg`)
- `yolozu export`
- `yolozu eval-coco`
- `yolozu eval-instance-seg`
- `yolozu eval-long-tail`
- `yolozu calibrate`
- `yolozu parity`
- `yolozu migrate` / `yolozu import`

Policy:
- Existing flags/behavior are kept backward-compatible across 1.x where practical.
- Deprecated aliases remain for at least one minor release window before removal.
- MCP/Actions aliases are explicitly documented in generated tool reference docs.

## CI gates that enforce this boundary

1. Schema compatibility gate (`python3 tools/check_schema_compatibility.py`)
2. Manifest declarative validation gate (`tools/validate_tool_manifest.py --require-declarative`)
3. Golden compatibility gate (`python3 tools/check_golden_compatibility.py`)
4. Wheel/sdist contents gates (required packaged artifacts)
5. Contract-oriented unit tests:
   - `tests/test_backend_shape_format_contracts.py`
   - `tests/test_manifest_docs_references.py`
   - `tests/test_run_record.py`
   - `tests/test_integrations_mcp_actions_parity.py`

## Compatibility change policy

When contract behavior must change:
1. Open RFC (`docs/rfc_workflow.md`)
2. Update golden assets (`baselines/golden/v1/*`) or versioned successor set
3. Update schema/contract docs + changelog migration note
4. Keep backward-compatible path where feasible (aliases/adapters) within 1.x
