# Manifest coverage audit

- Source of truth: `tools/manifest.json`
- Triage policy: `docs/manifest_unmanifested_tools_policy.json`
- Scope: declarative fields (`platform`, `inputs`, `effects`, `outputs`, `examples`), structural consistency, and direct `tools/` entrypoint coverage
- Status: `PASS` (no known gaps)

## Coverage policy

`tools/manifest.json` declares public, user-facing, and automation-facing tool
entrypoints. Some direct files under `tools/` are intentionally repo-local:
compatibility wrappers, release/operator scripts, fixture generators, one-off
migration helpers, or maintenance audits.

Every direct `tools/*.py` or `tools/*.sh` file that is not declared in the
manifest must be listed in `docs/manifest_unmanifested_tools_policy.json` with a
disposition and rationale. New files must either become manifest entries or be
triaged in that policy file.

## Validation checks

```bash
python3 tools/validate_tool_manifest.py --manifest tools/manifest.json
python3 tools/validate_tool_manifest.py --manifest tools/manifest.json --require-declarative
python3 -m unittest tests.test_manifest_tool_coverage
```

These checks pass on current `main`.

## Notes

- Historical gaps identified in `YOLOZU-1m2.2` were normalized in `YOLOZU-1m2.4`.
- Future manifest updates should keep strict declarative mode green.
- `tools/manifest.json` is not a blanket inventory of every maintainer helper;
  it is the stable registry for declared tool behavior.
