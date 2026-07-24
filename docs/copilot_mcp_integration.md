# Copilot integration (Extensions / VS Code)

Use the same backend as other LLM clients to avoid duplicated logic.

## Backend options

Preferred backend:

```bash
yolozu-mcp
```

Fallback API backend:

```bash
python3 tools/run_actions_api.py
```

## Option A: Copilot Extensions (GitHub App)

- Define skill endpoints that call YOLOZU MCP tools (preferred) or Actions API.
- Keep skills thin wrappers around backend tool calls.
- Return backend JSON directly, plus a short `summary`.

Recommended skill intents:
- Validate predictions
- Validate dataset
- Evaluate COCO dry-run
- Run scenario suite

## Option B: VS Code extension route

- Add command(s)/participant(s) that call MCP/API backend.
- Avoid re-implementing YOLOZU business logic in extension code.
- Surface `ok`, `summary`, and artifact paths in chat responses.

## Parity policy

Copilot, OpenAI, Claude, and Gemini should all consume the same tool interface contract and response shape from `yolozu.integrations.tool_runner`.
