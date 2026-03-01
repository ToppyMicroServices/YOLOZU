# CI dependency tiers

YOLOZU uses a three-tier CI dependency model to keep signal high while avoiding optional-extras combination explosion.

## Tiers

1. `core`
- Install: `python -m pip install .`
- Purpose: packaging and CLI/runtime smoke only.
- Jobs: `smoke_gate`, `pip_smoke`.

2. `recommended`
- Install: `python -m pip install -r requirements-ci.lock`
- Purpose: pinned interface contract/behavior gates (`schema`, `manifest`, `reference regression`, focused tests).
- Jobs: `quality_gate`, `test`.

3. `full`
- Purpose: GPU/backend matrix (TensorRT/CUDA/provider parity, full reference regression profile).
- Workflows: `gpu_smoke_machine.yml`, `gpu_zisn_pipeline.yml`, `reference_adapter_full.yml`.

## Why this split

- `core` catches packaging/runtime breakages cheaply.
- `recommended` gives stable regression signals with a lock file.
- `full` is intentionally separated because GPU/provider stacks are expensive and noisy for every PR.

## Optional extras policy

Optional extras are defined in `pyproject.toml` and their rationale is recorded under:

- `[tool.yolozu.optional_extras_rationale]`

CI does not depend on `full` extras as a single install target. Instead, it uses the tiered installs above so failures are easier to localize.
