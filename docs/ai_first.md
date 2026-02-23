# AI-first usage guide

This page defines the stable AI/agent surface for YOLOZU 1.0.x.

## 1) Purpose

Use YOLOZU as an interface-contract-first execution layer where agents:

- discover tools from `tools/manifest.json`
- generate deterministic run configs
- review configs against safety constraints
- execute only allowlisted operations

## 2) Safety principles

- **Determinism first**: prefer `--dry-run`, fixed `--max-images`, stable output paths.
- **Reproducibility**: always emit JSON artifacts under `reports/` with explicit config fields.
- **Allowlist execution**: use `python3 tools/yolozu.py registry run ...` for side-effect checks.
- **No-network by default**: network/GPU operations are opt-in and not part of the default AI-safe set.

## 3) Official MCP support boundary

Guaranteed (1.0.x, deterministic/lightweight MCP tool ids):

- `doctor`
- `generate_config`
- `review_config`
- `validate_predictions`

AI-safe tool scripts (CLI; not exposed as MCP tool ids in 1.0.x):

- `validate_synthgen_contract`
- `render_synthgen_overlay`
- `smoke_synthgen`

Best-effort (environment dependent, not in stable AI-safe guarantee):

- tool-runner operations beyond the guaranteed MCP tool ids (for example `eval_coco`, `validate_dataset`, `parity_check`)
- training jobs (`train`, `ttt`, `ctta`)
- TensorRT build/export
- OpenCV CUDA/OpenVINO backend execution

## 4) Fast path (3 commands)

```bash
python3 tools/run_mcp_server.py --print-tools
python3 tools/run_mcp_server.py --sample-generate-config > reports/ai_generate_config.json
python3 tools/run_mcp_server.py --sample-review-config reports/ai_generate_config.json
```

SynthGen-safe fast path (interface-contract-only, CPU):

```bash
python3 tools/validate_synthgen_contract.py --input data/smoke/synthgen_minishard/shards/train_000.jsonl --max-samples 2
python3 tools/render_synthgen_overlay.py --dataset-root data/smoke/synthgen_minishard --schema-id animal_v1 --sample-index 0 --output reports/smoke_synthgen_overlay.png
python3 tools/smoke_synthgen.py --dataset-root data/smoke/synthgen_minishard --output-dir reports
```

Start MCP stdio server:

```bash
python3 tools/run_mcp_server.py
```

## 5) JSON interface contracts for AI surface

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

## 7) CI gate for AI/MCP surface

The CI gate should verify:

- `python3 tools/run_mcp_server.py --help`
- manifest validation (`tools/validate_tool_manifest.py --require-declarative`)
- deterministic sample contracts (`generate_config` / `review_config`) via tests
