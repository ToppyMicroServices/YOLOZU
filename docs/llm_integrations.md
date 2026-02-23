# YOLOZU LLM integrations (MCP-first)

This project standardizes LLM integrations around one backend implementation.

Architecture reference: [MCP extension architecture](mcp_extension_architecture.md)

## 1) Common base (highest priority): YOLOZU MCP server

Start server:

```bash
python3 tools/run_mcp_server.py
```

Exposed tools (minimum):
- `doctor`
- `generate_config`
- `review_config`
- `validate_predictions`
- `validate_dataset`
- `eval_coco`
- `run_scenarios`
- `convert_dataset` (optional but available)

Also available in the same backend surface:
- inference/calibration: `predict_images`, `parity_check`, `calibrate_predictions`
- evaluation: `eval_instance_seg`, `eval_long_tail`
- async jobs: `train_job`, `export_predictions_job`, `test_job`, `ttt_job`, `ctta_job`
- compatibility alias: `export_onnx_job` (same behavior as `export_predictions_job`)
- job/run control: `jobs_list`, `jobs_status`, `jobs_cancel`, `runs_list`, `runs_describe`

Official AI-safe support (1.0.x guarantee):
- `doctor`, `generate_config`, `review_config`, `validate_predictions`

Best-effort only (environment-dependent):
- training jobs, TensorRT pipelines, OpenCV CUDA/OpenVINO paths

Return format policy:
- Always machine-readable JSON with stable top-level keys:
  - `ok` (bool)
  - `tool` (string)
  - `summary` (short sentence)
  - `exit_code` (int)
  - `stdout` / `stderr` (string, MCP route; capped + truncation metadata)
  - `stdout` / `stderr` (omitted on Actions API route by default; see `limits.stdio_redacted`)
  - optional parsed JSON artifacts (e.g. `report_json`)

This format is designed so Claude/Copilot/other MCP-capable clients can summarize consistently.

## 1.1) AI-facing guardrails (important)

- Path policy: `..` is rejected; absolute paths outside workspace are rejected by integration layer guards.
- Use workspace-relative paths whenever possible.
- For long-running tasks, use `job_id` + `jobs_status` instead of waiting on one synchronous call.
- Treat `ok/tool/summary/exit_code` as canonical status and `meta` as optional provenance.
- Set `dry_run`, `strict`, and `force` explicitly to avoid client-specific default drift.

## 2) OpenAI (ChatGPT) routes

Detailed setup: [OpenAI MCP / Actions](openai_mcp_actions.md)

### A. MCP route (recommended)

Use the same YOLOZU MCP server as remote MCP endpoint.

- Reuses one implementation across LLMs.
- Keeps command behavior and outputs identical to local CLI semantics.

### B. GPT Actions route (OpenAPI)

Start REST endpoint:

```bash
python3 tools/run_actions_api.py
```

OpenAPI schema:
- `http://<host>:8080/openapi.json`

Main endpoints:
- `POST /doctor`
- `POST /validate/predictions`
- `POST /validate/dataset`
- `POST /eval/coco`
- `POST /run/scenarios`
- `POST /convert/dataset`

Recommendation: ship MCP first, add Actions only when ChatGPT Actions integration is required.

## 3) Claude routes

Claude integration should also use the same MCP server.

```bash
python3 tools/run_mcp_server.py
```

- Expose the same tool contract and JSON shape used by other clients.
- Keep Claude-side prompt/tool wrappers thin (no duplicated business logic).

## 4) Copilot routes

Detailed setup: [Copilot MCP integration](copilot_mcp_integration.md)

### A. Copilot Extensions (skillsets / agent)

Define skill endpoints that forward to:
- YOLOZU MCP server (preferred), or
- YOLOZU Actions API.

### B. VS Code extension route

Implement participant/commands that invoke the same backend (MCP/API) instead of re-implementing CLI logic.

This avoids duplicate business logic and keeps output parity between Copilot and other LLM clients.

## 5) Gemini route

Gemini can use the same backend in two ways:

### A. MCP route (recommended)

Connect Gemini-capable MCP client/runtime to YOLOZU MCP server:

```bash
python3 tools/run_mcp_server.py
```

Use the same core tools (`doctor`, `validate_predictions`, `validate_dataset`, `eval_coco`, `run_scenarios`, `convert_dataset`) with identical JSON outputs.

### B. API/tool-calling route

Expose the FastAPI/OpenAPI endpoint and register tool/function calls against it:

```bash
python3 tools/run_actions_api.py
```

Schema endpoint:
- `http://<host>:8080/openapi.json`

This keeps Gemini, OpenAI, and Copilot integrations aligned on one implementation.

## 6) Client matrix (recommended)

- Gemini: MCP first, API/tool-calling optional
- Claude: MCP first
- Copilot: MCP-backed extension/participant
- OpenAI: MCP first, GPT Actions optional

All four routes should share the same backend implementation in `yolozu.integrations.tool_runner`.

Generated contract reference:
- `docs/generated/mcp_actions_tool_reference.json`
- `docs/generated/mcp_actions_tool_reference.md`

## 7) Connection templates (examples)

Example MCP client profiles (template JSON):

- OpenAI: `docs/examples/mcp_clients/openai_mcp_profile.example.json`
- Claude: `docs/examples/mcp_clients/claude_mcp_profile.example.json`
- Copilot: `docs/examples/mcp_clients/copilot_mcp_profile.example.json`
- Gemini: `docs/examples/mcp_clients/gemini_mcp_profile.example.json`

These are intentionally generic templates. Adjust keys/shape to each client runtime's exact MCP config format.
