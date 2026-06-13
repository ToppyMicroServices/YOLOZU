# Manifest authoring workflow

This page documents the contributor workflow for adding or updating `tools/manifest.json` entries.
Use this together with `docs/manifest_declarative_spec.md`.

## Goal

- keep manifest entries complete and machine-readable,
- keep docs and CLI behavior aligned,
- prevent regressions via strict validation.

## Required fields (quick checklist)

Every tool entry should include:

- `id`, `entrypoint`, `runner`, `summary`
- `platform` with `cpu_ok`, `gpu_required`, `macos_ok`, `linux_ok`
- `inputs` (empty list allowed)
- `effects` with `writes` and `fixed_writes`
- `outputs` (empty list allowed)
- `examples` with at least one runnable `command`

Optional but recommended when applicable:

- `contracts.{consumes,produces}`
- `contract_outputs`
- `docs`
- `requires`

## Contributor update flow

1. Implement or modify the tool behavior first.
2. Update the corresponding entry in `tools/manifest.json`:
   - add/adjust `inputs` for CLI flags,
   - declare write paths in `effects`,
   - declare produced artifacts in `outputs`,
   - keep at least one real `examples[].command`.
3. Validate manifest structure:

```bash
python3 tools/validate_tool_manifest.py --manifest tools/manifest.json
python3 tools/validate_tool_manifest.py --manifest tools/manifest.json --require-declarative
```

4. Sync packaged manifest copy (single source of truth sync):

```bash
python3 tools/sync_tools_manifest.py
```

5. Run regression tests:

```bash
python -m pytest -q tests/test_tool_manifest.py tests/test_packaged_tools_manifest.py tests/test_manifest_docs_references.py
```

6. Update related docs if command surface changed:
   - `docs/tools_index.md`
   - workflow/protocol docs that reference the tool

## Concrete example: simple validator tool

```json
{
  "id": "check_license_policy",
  "entrypoint": "tools/check_license_policy.py",
  "runner": "python3",
  "summary": "Enforce Apache-2.0-only constraints.",
  "platform": { "cpu_ok": true, "gpu_required": false, "macos_ok": true, "linux_ok": true },
  "inputs": [],
  "effects": { "writes": [], "fixed_writes": [] },
  "outputs": [
    { "name": "stdout", "kind": "stdout", "description": "OK or failure reason." }
  ],
  "examples": [
    { "description": "Run policy checks.", "command": "python3 tools/check_license_policy.py" }
  ]
}
```

## Concrete example: file-producing benchmark tool

```json
{
  "id": "benchmark_latency",
  "entrypoint": "tools/benchmark_latency.py",
  "runner": "python3",
  "summary": "Latency/FPS benchmark harness.",
  "platform": { "cpu_ok": true, "gpu_required": false, "macos_ok": true, "linux_ok": true },
  "inputs": [
    { "name": "output", "kind": "file", "required": false, "flag": "--output", "default": "reports/benchmark_latency.json" }
  ],
  "effects": {
    "writes": [
      { "flag": "--output", "kind": "file", "scope": "path", "description": "Writes benchmark JSON report." }
    ],
    "fixed_writes": []
  },
  "outputs": [
    {
      "name": "report_json",
      "kind": "file",
      "default": "reports/benchmark_latency.json",
      "description": "Benchmark latency report JSON."
    }
  ],
  "examples": [
    { "description": "Run synthetic benchmark.", "command": "python3 tools/benchmark_latency.py --output reports/benchmark_latency.json" }
  ]
}
```

## Common mistakes

- missing `inputs`/`outputs` field when there are no items (use `[]`)
- declaring `effects.writes[].flag` not present in `inputs[].flag`
- missing `effects.fixed_writes` key
- missing `outputs[].description`
- missing `outputs[].default` for file/dir outputs; use `null` when no built-in default path exists
- no runnable `examples[].command`
- using non-repo-relative paths in `entrypoint`/`docs`/schema references

## PR checklist snippet

- [ ] `tools/manifest.json` updated for changed tool behavior
- [ ] `python3 tools/sync_tools_manifest.py` run (`yolozu/data/manifest/tools_manifest.json` synced)
- [ ] strict declarative validation passes
- [ ] manifest regression tests pass (`test_packaged_tools_manifest`, `test_manifest_docs_references`)
- [ ] docs links/examples updated
