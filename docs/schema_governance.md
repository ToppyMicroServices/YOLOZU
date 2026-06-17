# Schema governance

This document defines how YOLOZU evolves JSON artifact schemas without breaking comparability.

## Canonical schema location

The authoritative predictions JSON Schema lives at `docs/schemas/predictions.schema.json`
(JSON Schema draft 2020-12). The copies at `schemas/predictions.schema.json` and
`yolozu/data/schemas/predictions.schema.json` (packaged) must stay in sync - they are
overwritten from the canonical copy during release preparation.

## Scope

This governance applies to wrapped prediction-style payloads that include:

- `schema_version`
- `predictions`
- optional `meta`

Current version:

- `current_schema_version = 1`
- `minimum_supported_schema_version = 1`
- `current_entry_schema_version = 2`
- `minimum_supported_entry_schema_version = 1`

## Lifecycle rules

1. Backward-compatible changes do **not** bump `schema_version`.
   - Examples: adding optional fields, adding optional metadata keys, expanding accepted optional aliases.
2. Breaking changes **must** bump `schema_version`.
   - Examples: removing required fields, changing required field type/meaning, changing required coordinate conventions.
3. Validators reject unknown future versions.
   - If a payload declares `schema_version > current_schema_version`, validation fails until YOLOZU is upgraded.
4. Legacy wrapped payloads without `schema_version` are accepted with warning in compatibility mode.

## Compatibility policy

- `schema_version` present:
  - must be integer
  - must satisfy `minimum_supported <= schema_version <= current`
- `schema_version` missing in wrapped payload:
  - accepted for backward compatibility
  - validator emits warning and treats payload as legacy mode
- entry-level `schema_version` missing:
  - treated as legacy entry v1
  - canonicalization migrates to entry v2

## Breaking-change process (checklist)

When proposing a schema-breaking change:

1. Open an RFC issue describing:
   - old vs new interface contract
   - expected migration cost
   - affected tools/adapters/protocols
2. Add/update migration utility for old artifacts.
3. Add golden test vectors for old and new versions.
4. Add CI gate coverage:
   - current version must pass
   - future/unsupported versions must fail
5. Update docs:
   - this governance doc
   - schema-specific docs (e.g., predictions schema pages)
   - release notes with migration steps
6. Update release/process metadata:
   - add a `Contract change` note in `CHANGELOG.md`
   - include baseline lifecycle impact (`dataset_hash`, `weights_hash`, baseline version path)
   - update PR checklist entries for baseline updates

See also: [RFC workflow + golden compatibility assets](rfc_workflow.md).

## Migration steps template

Use this template in release notes when introducing schema `N+1`:

1. Identify old artifacts (`schema_version == N` or missing).
2. Run migration tool to emit `schema_version == N+1` payloads (predictions entry v1->v2: `yolozu predictions migrate --from v1 --to v2 ...`).
3. Validate migrated artifacts with `yolozu validate ...` in CI.
4. Re-run evaluation protocol on migrated artifacts and compare metrics.
5. Remove compatibility mode only after deprecation window ends.

## CI enforcement

CI includes a schema compatibility gate that asserts:

- v1 wrapped payloads pass validation
- v2 wrapped payloads fail while current is v1
- entry v1 payloads are migrated to entry v2 during canonicalization/load paths

This prevents silent schema drift and guarantees interface-contract-first behavior.

In addition, golden compatibility assets are versioned under `baselines/golden/` and validated by:

- `python3 tools/check_golden_compatibility.py`

This gate pins protocol + golden artifact hashes and fails when schema/protocol behavior changes without coordinated golden updates.

## Schema Browser Coverage

Use this table as the maintained schema browser until a generated web page is
introduced. Each row points to the canonical schema or the closest current
schema surface for that artifact family.

| Artifact family | Canonical schema | Main docs | Notes |
|---|---|---|---|
| Predictions | `docs/schemas/predictions.schema.json` | [`predictions_schema.md`](predictions_schema.md) | Packaged copies live in `schemas/` and `yolozu/data/schemas/`. |
| Detection / COCO eval reports | `docs/schemas/coco_eval_report.schema.json`, `docs/schemas/eval_suite_report.schema.json` | [`yolo26_eval_protocol.md`](yolo26_eval_protocol.md), [`evaluation_protocol_template.md`](evaluation_protocol_template.md) | Protocol hash must be recorded before fair comparison. |
| Segmentation dataset/eval | `docs/schemas/seg_dataset.schema.json`, `docs/schemas/seg_eval_report.schema.json` | [`predictions_schema.md`](predictions_schema.md) | Dataset and eval schemas are separate from predictions payloads. |
| Training handoff | `docs/schemas/training_run_summary.schema.json`, `docs/schemas/training_handoff.schema.json` | [`training_orchestration.md`](training_orchestration.md) | Handoff JSON carries next steps for resume/export/eval/parity. |
| SynthGen sample | `schemas/synthgen_sample.schema.json` | [`synthgen_contract.md`](synthgen_contract.md), [`synthgen_repo_integration.md`](synthgen_repo_integration.md) | SynthGen intake remains an external generator handoff. |
| Research reports | `docs/schemas/research_lane_report.schema.json`, `docs/schemas/research_note.schema.json` | [`research_lanes.md`](research_lanes.md), [`research_note_template.md`](research_note_template.md) | Research results stay separate from stable evaluation reports. |
