# AI-first usage guide

This page defines the current stable AI/agent surface for YOLOZU.

## 1) Purpose

Use YOLOZU as an interface-contract-first execution layer where agents:

- discover tools from the manifest packaged with `yolozu`
- generate deterministic run configs
- review configs against safety constraints
- execute only allowlisted operations

## 2) Safety principles

- **Determinism first**: prefer `--dry-run`, fixed `--max-images`, stable output paths.
- **Reproducibility**: always emit JSON artifacts under `reports/` with explicit config fields.
- **Allowlist execution**: use `python3 -m yolozu registry run ...` for side-effect checks.
- **No-network by default**: network/GPU operations are opt-in and not part of the default AI-safe set.

## 3) Official MCP support boundary

Guaranteed (deterministic/lightweight MCP tool ids):

- `doctor`
- `generate_config`
- `review_config`
- `validate_predictions`

AI-safe tool scripts (CLI; not exposed as guaranteed MCP tool ids):

- `validate_synthgen_contract`
- `render_synthgen_overlay`
- `smoke_synthgen`

Best-effort (environment dependent, not in stable AI-safe guarantee):

- tool-runner operations beyond the guaranteed MCP tool ids (for example `eval_coco`, `validate_dataset`, `parity_check`)
- training jobs (`train`, `ttt`, `ctta`)
- TensorRT build/export
- OpenCV CUDA/OpenVINO backend execution

The live MCP server registers 25 canonical tool ids. The Actions API shares 21
canonical operations. `generate_config` and `review_config` are in-process
config-review tools and are not Actions endpoints. These sets are intentionally
different and are machine-readable in
`docs/generated/mcp_actions_tool_reference.json` under `surfaces`.
Discovery responses expose the same boundaries as `guaranteed_mcp_tools` and
`live_mcp_tools`; the older `supported_mcp_tools` field remains a compatibility
alias for the guaranteed set.
With `--ids-only`, `manifest_tools` and `selected_tool_ids` contain only the
filtered selection, while `surface_counts` keeps the response compact. Omit
`--ids-only` when full records and the expanded `surfaces` object are needed.
Full live discovery includes a nonempty summary and the exact MCP JSON
`input_schema` for every id from the generated reference packaged in the
wheel. Surface membership and maturity are separate: no `stable` maturity is
inferred from `guaranteed_ai_safe`. A maturity/tag filter excludes
unclassified records and reports those exclusions in `filter_diagnostics`.

## 4) Fast path (3 commands)

```bash
yolozu-mcp --print-tools --guaranteed --ids-only
yolozu-mcp --sample-generate-config > reports/ai_generate_config.json
yolozu-mcp --sample-review-config reports/ai_generate_config.json
```

`--sample-review-config` exits `1` when the parsed config is rejected and `2`
when the config cannot be read safely, so agents can use the process status
without parsing human text.

Inspect all 25 registered MCP operations, or filter the broader manifest
registry without returning full records:

```bash
yolozu-mcp --print-tools --supported --ids-only
yolozu-mcp --print-tools --ids-only --maturity stable --tag validation
```

`mcp_live` means the operation is registered and discoverable. It is not an
execution guarantee: only `guaranteed_ai_safe` carries the deterministic,
lightweight guarantee, and every other operation remains subject to its
declared dependencies and runtime inputs.

SynthGen-safe fast path (interface-contract-only, CPU):

```bash
python3 tools/validate_synthgen_contract.py --input data/smoke/synthgen_minishard/shards/train_000.jsonl --max-samples 2
python3 tools/render_synthgen_overlay.py --dataset-root data/smoke/synthgen_minishard --schema-id animal_v1 --sample-index 0 --output reports/smoke_synthgen_overlay.png
python3 tools/smoke_synthgen.py --dataset-root data/smoke/synthgen_minishard --output-dir reports
```

Start MCP stdio server:

```bash
yolozu-mcp
```

For queued operations, poll `jobs_status`. A command result with `ok=false` or
a nonzero `exit_code` produces `job.status: "failed"` and a top-level
`jobs_status.ok: false`; the nested result remains available for diagnostics.

Installed-wheel Python fast path (run from the consumer workspace):

```python
from yolozu.integrations.ai_surface import list_manifest_tools
from yolozu.integrations.tool_runner import validate_predictions

tool_ids = list_manifest_tools(guaranteed=True, ids_only=True)
result = validate_predictions("reports/predictions.json", strict=True)
if not result["ok"]:
    raise RuntimeError(result.get("validation") or result["summary"])
```

`result["validation"]` follows the validation-result schema described below.
The MCP/tool-runner default is fail-closed (`strict=True`). Passing
`strict=False` is the explicit compatibility-repair mode; the result then
contains `mode: "repair"`, `repair_enabled: true`, up to 100 repair warnings,
and the omitted count in `limits.warnings_truncated`.

## 5) JSON interface contracts for AI surface

### 5.0 Predictions validation result

Human output remains the default. Add `--json` for a bounded result with
`schema_version`, `ok`, `mode`, `repair_enabled`, `warnings`, and `errors`:

```bash
yolozu validate predictions reports/predictions.json --strict --json
```

Schema file: `docs/schemas/predictions_validation_result.schema.json`

### 5.1 `doctor` response

`doctor` writes JSON with environment/runtime diagnostics. Required top-level keys for AI consumption:

- `timestamp`
- `gpu`
- `env`
- `runtime_capabilities`
- `drift_hints`

### 5.2 `generate_config` response schema

Schema file: `docs/schemas/ai_generate_config.schema.json`

Required top-level keys:

- `schema_version`
- `goal`
- `tool`
- `arguments`
- `safety`
- `recommended_sequence`

### 5.3 `review_config` response schema

Schema file: `docs/schemas/ai_review_config.schema.json`

Required top-level keys:

- `schema_version`
- `ok`
- `issues`
- `warnings`
- `summary`

## 6) Manifest requirements for AI use

Agent-facing tools in `tools/manifest.json` should provide:

- `id`
- `summary`
- `inputs` (args schema)
- `examples`
- `effects` (side-effects / write locations)
- `requires` (network/GPU constraints)

For SynthGen intake tools, also include:
- explicit `schema_id` controls
- deterministic fixture examples under `data/smoke/synthgen_minishard`
- interface contract reference `docs/synthgen_contract.md`

Safe defaults for AI execution:

- `dry_run=true` when supported
- bounded `max_images` (e.g. 50)
- outputs under `reports/`
- no network unless explicitly needed

Installed Python helpers resolve relative input and output paths against the
caller's current working directory. Absolute paths are accepted only when they
remain inside that workspace; `..`, home-directory shortcuts, and paths that
resolve outside it are rejected, including values passed as `--flag=path`.
Scenario `extra_args` accept only declared long-form, one-value flags; short or
unknown flags are rejected before execution. Manifest discovery uses the
packaged resource by default, so it does not require a repository checkout. An
explicit `--manifest` override is resolved against the same caller workspace.

## 7) CI gate for AI/MCP surface

The CI gate should verify:

- `yolozu-mcp --help`
- manifest validation (`tools/validate_tool_manifest.py --require-declarative`)
- deterministic sample interface contracts (`generate_config` / `review_config`) via tests
- candidate artifacts through git archive, sdist, wheel, a clean virtual
  environment, and an outside-checkout strict validation with `PYTHONPATH`
  cleared
