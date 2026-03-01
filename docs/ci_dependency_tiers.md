# CI dependency tiers

YOLOZU uses a tiered CI dependency model to keep signal high while avoiding optional-extras combination explosion.

## Tiers

1. `core`
- Install: `python -m pip install .`
- Purpose: packaging and CLI/runtime smoke only.
- Jobs: `smoke_gate`, `pip_smoke`.

2. `docs_mcp`
- Install: `python -m pip install -e '.[mcp,actions]'`
- Purpose: docs/manual/README and MCP/Actions surface checks without running full runtime regression gates.
- Jobs: `docs_mcp_gate`.

3. `recommended`
- Install: `python -m pip install -r requirements-ci.lock`
- Purpose: pinned interface contract/behavior gates (`schema`, `manifest`, `reference regression`, focused tests).
- Jobs: `quality_gate`, `test`.

4. `full`
- Purpose: GPU/backend matrix (TensorRT/CUDA/provider parity, full reference regression profile).
- Workflows: `gpu_smoke_machine.yml`, `gpu_zisn_pipeline.yml`, `reference_adapter_full.yml`.

## Why this split

- `core` catches packaging/runtime breakages cheaply.
- `docs_mcp` validates docs/MCP changes quickly without spending runtime-heavy CI budget.
- `recommended` gives stable regression signals with a lock file.
- `full` is intentionally separated because GPU/provider stacks are expensive and noisy for every PR.

## Optional extras policy

Optional extras are defined in `pyproject.toml` and their rationale is recorded under:

- `[tool.yolozu.optional_extras_rationale]`

CI does not depend on `full` extras as a single install target. Instead, it uses the tiered installs above so failures are easier to localize.
