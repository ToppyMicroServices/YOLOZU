# MCP extension architecture (YOLOZU v2)

This document fixes the extension layering and operational policy.

## Layer design

1. Core layer (`yolozu.integrations.layers.core`)
   - request/response shaping
   - error normalization
   - short summary generation
2. YOLOZU API layer (`yolozu.integrations.layers.api`)
   - CLI/API wrapper execution
   - argument/path guards
   - allowlist enforcement
3. Job layer (`yolozu.integrations.layers.jobs`)
   - async execution with `job_id`
   - queue / running / completed / failed / cancelled tracking
4. Artifact layer (`yolozu.integrations.layers.artifacts`)
   - run listing/description
   - metadata (`git_sha`, `manifest_sha256`, runtime info)

## Tool roadmap status

Data entry tools (A1-A3) are implemented in shared backend `yolozu.integrations.tool_runner` and exposed via both MCP and Actions API:
- A1: `validate_predictions`
- A2: `validate_dataset`
- A3: `convert_dataset`

Inference tools (B4-B6) are implemented in the same shared backend and exposed via both MCP and Actions API:
- B4: `predict_images`
- B5: `parity_check`
- B6: `calibrate_predictions`

Evaluation tools (C7-C9) are implemented in the same shared backend and exposed via both MCP and Actions API:
- C7: `eval_coco`
- C8: `eval_instance_seg`
- C9: `eval_long_tail`

Training tools (D10-D12) are implemented as async jobs in the same shared backend and exposed via both MCP and Actions API:
- D10: `train_job`
- D11: `export_predictions_job` (compatibility alias: `export_onnx_job`)
- D12: `test_job`

TTT/CTTA tools (E13-E14) are implemented as async jobs in the same shared backend and exposed via both MCP and Actions API:
- E13: `ttt_job`
- E14: `ctta_job`

## Long-running jobs

Long tools should return quickly with `job_id`.
Job states are persisted under `runs/mcp_jobs/*.json` and restored on restart.
States restored as `queued`/`running` are converted to `unknown` to avoid false in-flight claims.
Job storage is initialized lazily; importing the AI surface, requesting
`--help`, or running read-only discovery does not create `runs/mcp_jobs`.

Current API surface:
- `jobs.list`
- `jobs.status(job_id)`
- `jobs.cancel(job_id)`
- `runs.list`
- `runs.describe(run_id)`

## Security policy

- Only allowlisted `yolozu` top-level subcommands are executable.
   - Allowlist is manifest-first (the manifest packaged in `yolozu`), with conservative fallback defaults.
- Path traversal (`..`) is rejected.
- The caller process's current working directory is the default workspace.
- Absolute paths are allowed only within that workspace; home-directory
  shortcuts, symlink escapes, and absolute paths outside it are rejected,
  including values embedded in `--flag=path`.
- Scenario `extra_args` use a declared long-form one-value flag allowlist;
  short, unknown, empty, or missing-value flags are rejected.
- CLI execution has a timeout guard (default 600s).
- MCP route: `stdout`/`stderr` are capped and marked with truncation metadata in response payloads.
- Actions API route: CLI `stdout`/`stderr` are redacted by default (`limits.stdio_redacted=true`) and errors are genericized to avoid leaking exception details.

## AI clarity checklist (recommended)

- Keep tool names and underlying CLI behavior semantically aligned (avoid name/behavior drift).
- Keep MCP tool signatures aligned with Actions API request models for the same operation.
- Prefer explicit parameter passing (`split`, `dry_run`, `strict`, `force`) over implicit defaults.
- Keep response interface contracts stable (`ok`, `tool`, `summary`, `exit_code`) and add fields compatibly.
- Include actionable error categories for common failures: allowlist, path guard, timeout, command failure.

Generated parity reference:
- `docs/generated/mcp_actions_tool_reference.json`
- `docs/generated/mcp_actions_tool_reference.md`
- regenerate/check: `python3 tools/generate_integration_tool_reference.py --check`

Its `surfaces` object separates live MCP registration from the guaranteed
AI-safe subset, config-review helpers, and Actions endpoints. CI also calls the
installed MCP SDK's live `app.list_tools()` API and compares canonical names and
input schemas with this reference.
Live registration is discovery evidence, not an execution guarantee.
The JSON reference is copied into the wheel as a package resource so full
discovery remains available outside a checkout. Its `surface_tiers` are not
maturity classifications; missing explicit maturity/tag metadata stays
unclassified and filter diagnostics report any resulting exclusions.

## CI no-abandon rule

When MCP-related changes are pushed:
1. run local checks (`manifest`, unit tests)
2. push and monitor latest CI
3. if failure appears, patch and re-run immediately
