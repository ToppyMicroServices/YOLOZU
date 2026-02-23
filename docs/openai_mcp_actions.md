# OpenAI integration (MCP + GPT Actions)

This document provides a practical setup path for OpenAI clients.

## Route A: MCP (recommended)

Run the shared MCP backend:

```bash
python3 tools/run_mcp_server.py
```

Use these tools:
- `doctor`
- `validate_predictions`
- `validate_dataset`
- `eval_coco`
- `run_scenarios`
- `convert_dataset`

Why MCP first:
- one implementation reused across clients
- same JSON response shape as other integrations
- minimal glue code

## Route B: GPT Actions (OpenAPI)

Run API server:

```bash
python3 tools/run_actions_api.py
```

OpenAPI schema URL:
- `http://<host>:8080/openapi.json`

Optional static export for registration workflows:

```bash
python3 tools/export_actions_openapi.py --output reports/actions_openapi.json
```

Main endpoints:
- `POST /doctor`
- `POST /validate/predictions`
- `POST /validate/dataset`
- `POST /eval/coco`
- `POST /run/scenarios`
- `POST /convert/dataset`
- `POST /predict/images`
- `POST /parity/check`
- `POST /calibrate/predictions`
- `POST /eval/instance-seg`
- `POST /eval/long-tail`
- `POST /jobs/export-predictions` (canonical)
- `POST /jobs/export-onnx` (compatibility alias)
- `POST /jobs/*` and `GET /runs/*` style equivalents for async control/reporting

## Request example

```bash
curl -sS -X POST http://127.0.0.1:8080/eval/coco \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset": "data/smoke",
    "split": "val",
    "predictions": "data/smoke/predictions/predictions_dummy.json",
    "dry_run": true,
    "output": "reports/actions_eval_coco_dry_run.json"
  }'
```

## Operational notes

- Prefer MCP for day-to-day automation; add Actions only when OpenAPI registration is mandatory.
- Keep payload handling contract-first: check `ok/tool/summary/exit_code` first, then parse optional artifact JSON fields.
- For heavy work, submit async jobs and poll status instead of relying on long request timeouts.
- Keep MCP/Actions signatures in sync using the generated reference at `docs/generated/mcp_actions_tool_reference.md`.
