# YOLOZU LLM integrations (MCP-first)

This project standardizes LLM integrations around one backend implementation.

Architecture reference: [MCP extension architecture](mcp_extension_architecture.md)

## 1) Common base (highest priority): YOLOZU MCP server

Start server:

```bash
python3 -m pip install 'yolozu[mcp]'
yolozu-mcp
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
- recommendation: `recommend_image_pipeline` (Experimental, MCP-only, read-only)
- pinned local processing: `process_images` (Experimental, MCP-only, dry-run by default)
- inference/calibration: `predict_images`, `parity_check`, `calibrate_predictions`
- evaluation: `eval_instance_seg`, `eval_long_tail`
- async jobs: `train_job`, `export_predictions_job`, `test_job`, `ttt_job`, `ctta_job`
- compatibility alias: `export_onnx_job` (same behavior as `export_predictions_job`)
- job/run control: `jobs_list`, `jobs_status`, `jobs_cancel`, `runs_list`, `runs_describe`

Guaranteed AI-safe support:
- `doctor`, `generate_config`, `review_config`, `validate_predictions`

The server also registers a broader 27-tool live MCP surface. The generated
reference distinguishes that live set from the four guaranteed tools, the two
config-review tools, and the 21 canonical Actions operations. Registration does
not promote environment-dependent tools into the guaranteed set.

Installed MCP quickstart (copy-paste):

```bash
# Start MCP server (stdio)
yolozu-mcp

# Inspect the four guaranteed AI-safe tools as JSON
yolozu-mcp --print-tools --guaranteed --ids-only > reports/mcp_tool_ids.json

# Inspect all 27 registered MCP operations
yolozu-mcp --print-tools --supported --ids-only > reports/mcp_live_tool_ids.json

# Deterministic sample I/O (useful for client wiring tests)
yolozu-mcp --sample-generate-config > reports/ai_generate_config.json
yolozu-mcp --sample-review-config reports/ai_generate_config.json > reports/ai_review_config.json

# MCP settings check (manifest + generated reference sync)
python3 tools/check_mcp_settings.py --output reports/mcp_settings_check.json
```

Here “registered” means discoverable through the live MCP schema. It does not
promise that environment-dependent execution will succeed; only the four-tool
`guaranteed_ai_safe` set carries the lightweight deterministic guarantee.

`recommend_image_pipeline` accepts a structured image job and local input. It
returns a selected or abstained SelectionDecision without inference, downloads,
writes, network access, or natural-language parsing. The packaged registry contains
three non-promoted Candidate records, while the screening and public evidence streams
are empty. The default installed call therefore abstains. A selected decision would
require matching governed evidence and does
not itself execute the selected pipeline.

`process_images` requires that complete selected decision and the same local
job/input roots. It revalidates current lifecycle, evidence, environment,
workload, input, artifact, resolver, and class-mapping identities. `dry_run=true`
performs no runner call or write. Explicit execution is available only through a
registered code-owned network-free route; none is registered in the empty public
baseline, so this surface does not claim a currently runnable model.

Best-effort only (environment-dependent):
- training jobs, TensorRT pipelines, OpenCV CUDA/OpenVINO paths
- TTT/CTTA jobs, which require torch and a checkpoint with
  `status=full` for the selected RT-DETR config

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

- Path policy: caller-relative paths use the process current working directory;
  `..`, home-directory shortcuts, symlink escapes, and absolute paths outside
  that workspace are rejected by integration layer guards, including
  `--flag=path` argument values.
- Scenario `extra_args` accept only declared long-form flags with one value;
  short, unknown, empty, and missing-value flags are rejected.
- Use workspace-relative paths whenever possible.
- For long-running tasks, use `job_id` + `jobs_status` instead of waiting on one synchronous call.
- `jobs_status.ok` becomes `false` when the underlying command returns
  `ok=false` or a nonzero `exit_code`; the nested `job.result` is retained for
  diagnostics.
- Treat `ok/tool/summary/exit_code` as canonical status and `meta` as optional provenance.
- Set `dry_run`, `strict`, and `force` explicitly to avoid client-specific default drift.
- `validate_predictions(strict=true)` is fail-closed. Use `strict=false` only
  when compatibility repair is intended; the response identifies repair mode
  and returns up to 100 repair warnings plus
  `limits.warnings_truncated`. `eval_coco` is also strict unless `repair=true`.
- `ttt_job` and `ctta_job` require workspace-relative `dataset` and
  `checkpoint` inputs. They fail before queueing unless strict checkpoint
  preflight reports `status=full` and `load.loaded=true`.
- Treat their predictions and TTT log as local diagnostics. A completed job
  does not establish efficacy.

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
yolozu-mcp
```

- Expose the same tool interface contract and JSON shape used by other clients.
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
yolozu-mcp
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
- Ollama (local): use an MCP-capable client with an OpenAI-compatible base URL

Ollama note (local LLM):
- Run Ollama locally and point an OpenAI-compatible client to `http://127.0.0.1:11434/v1`.
- This only changes the LLM provider/model; the backend tool surface remains the same YOLOZU MCP server.

All four routes should share the same backend implementation in `yolozu.integrations.tool_runner`.

Generated interface contract reference:
- `docs/generated/mcp_actions_tool_reference.json`
- `docs/generated/mcp_actions_tool_reference.md`

The JSON reference's `surfaces` object is the machine-readable source for
`mcp_live`, `guaranteed_ai_safe`, `config_review`, and `actions_public`.
The same generated JSON is packaged in the wheel for checkout-independent
discovery. It provides the exact live input schemas and summaries. Surface
membership does not infer maturity; filters expose counts for excluded
unclassified metadata.

The installed `yolozu export` path uses the exporter packaged in the wheel;
it does not require `tools/export_predictions.py` from a source checkout.
Candidate-artifact CI installs the wheel into a clean environment and completes
both `ttt_job` and `ctta_job` from an external consumer directory.

## 7) Connection templates (examples)

Example MCP client profiles (template JSON):

- OpenAI: `docs/examples/mcp_clients/openai_mcp_profile.example.json`
- Claude: `docs/examples/mcp_clients/claude_mcp_profile.example.json`
- Copilot: `docs/examples/mcp_clients/copilot_mcp_profile.example.json`
- Gemini: `docs/examples/mcp_clients/gemini_mcp_profile.example.json`

These are intentionally generic templates. Adjust keys/shape to each client runtime's exact MCP config format.
