# OpenAI integration (MCP + GPT Actions)

This document provides a practical setup path for OpenAI clients.

## Route A: MCP (recommended)

Run the shared MCP backend:

```bash
yolozu-mcp
```

Inspect the compact registry and exact public surface sets:

```bash
yolozu-mcp --print-tools --guaranteed --ids-only
```

Use `yolozu-mcp --print-tools --supported --ids-only` to inspect all 27
registered MCP operations.

Use these tools:
- `doctor`
- `generate_config`
- `review_config`
- `validate_predictions`
- `validate_dataset`
- `eval_coco`
- `run_scenarios`
- `convert_dataset`

Why MCP first:
- one implementation reused across clients
- same JSON response shape as other integrations
- minimal glue code

The generated machine-readable reference separates the 27 live MCP tool ids,
the four guaranteed AI-safe ids, the two config-review ids, and the 21
canonical Actions operations. A live registration is not by itself a
deterministic or dependency-free guarantee.

The MCP-only `recommend_image_pipeline` operation is Experimental and read-only.
It can return a selected or abstained SelectionDecision for a structured local
image job, but it does not execute a model or download or write assets. The
packaged registry and public evidence stream are empty, so the default installed
call currently abstains. It is not exposed through GPT Actions.

The paired MCP-only `process_images` operation is also Experimental and not
exposed through Actions. It requires a complete selected decision, defaults to a
no-write dry-run, rejects local-state drift, and uses only registered code-owned
network-free execution. No bundle or runner is registered in the public baseline,
so it does not make a current runnable-model claim.

## Route B: GPT Actions (OpenAPI)

Run API server:

```bash
python3 tools/run_actions_api.py
```

OpenAPI schema URL:
- `http://127.0.0.1:8080/openapi.json`

Custom bind settings:

```bash
python3 tools/run_actions_api.py --host 127.0.0.1 --port 8080 --workers 1
```

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
- `POST /jobs/ttt`
- `POST /jobs/ctta`
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
    "repair": false,
    "output": "reports/actions_eval_coco_dry_run.json"
  }'
```

TTT job request (the checkpoint must be fully compatible with the selected
config):

```bash
curl -sS -X POST http://127.0.0.1:8080/jobs/ttt \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset": "data/smoke",
    "checkpoint": "checkpoints/rtdetr_pose.pt",
    "config": "builtin:base",
    "split": "val",
    "method": "tent",
    "reset": "sample",
    "steps": 1,
    "max_images": 1
  }'
```

An accepted request returns a `job_id`; poll `GET /jobs/{job_id}`. Missing or
non-full checkpoints fail before queueing with `stage=preflight`. The resulting
predictions and TTT report are local diagnostics, not efficacy evidence.

## Operational notes

- Prefer MCP for day-to-day automation; add Actions only when OpenAPI registration is mandatory.
- Keep payload handling interface-contract-first: check `ok/tool/summary/exit_code` first, then parse optional artifact JSON fields.
- For heavy work, submit async jobs and poll status instead of relying on long request timeouts.
- Keep MCP/Actions signatures in sync using the generated reference at `docs/generated/mcp_actions_tool_reference.md`.
- Relative paths are resolved from the server process's current working
  directory. Absolute paths must remain within that workspace; `..` and path
  escapes fail closed.
- Use `yolozu validate predictions <path> --strict --json` when an agent needs
  a bounded success/failure payload instead of human-readable output.
- The validation endpoint defaults to `strict: true`; `strict: false` explicitly
  selects compatibility repair. COCO evaluation defaults to `repair: false`.
