# RFC workflow and golden compatibility assets

This workflow is required for schema/protocol changes that can affect comparability.

## Approval flow

1. Open an RFC issue (title prefix: `RFC:`) with:
   - affected interface contract(s)/protocol(s)
   - before/after examples
   - migration and rollback plan
   - risk assessment for parity/eval drift
2. Link the RFC in the PR description.
3. Require at least one maintainer approval for the RFC and the implementation PR.
4. Keep the RFC status updated: `proposed` → `approved` → `implemented`.

## Golden compatibility assets

Versioned golden assets live under `baselines/golden/<version>/`.

For `v1`:

- `manifest.json` (hash-pinned protocol + assets)
- `predictions_legacy_wrapped_no_schema.json`
- `predictions_v1.json`
- `segmentation_v1.json`
- `instance_segmentation_v1.json`

These assets represent old/current accepted inputs and are used by compatibility gates.

## Compatibility gates

- Unit test gate: `tests/test_golden_compatibility_tool.py`
- Tool gate: `python3 tools/check_golden_compatibility.py`
- CI must pass both before merge.

If you intentionally change schema/protocol behavior, update golden assets + hashes in the same PR and link the RFC.
