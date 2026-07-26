# CI dependency tiers

YOLOZU uses a tiered CI dependency model to keep signal high while avoiding optional-extras combination explosion.

Default cost policy:

- Pull requests run lightweight Ubuntu checks: docs/metadata gates, runtime smoke, and focused quality tests.
- Pushes to `main` run the full CPU evaluation gate, including Python matrix and full unittest discovery.
- GPU checks stay on Ubuntu/Linux GPU runners and run only by manual dispatch.
- macOS is reserved for release-time wheel build validation only.
- Expensive full/fuzz/regression sweeps are manual-only unless there is an explicit release or incident reason to re-enable a schedule.

## Tiers

1. `core`
- Install: `python3 tools/ci/install_with_hashes.py --requirements requirements-locks/requirements-runtime.lock --install-local-wheel`
- Purpose: packaging and CLI/runtime smoke only.
- Jobs: `smoke_gate`; `pip_smoke` is main-push only.

2. `docs_mcp`
- Install: `python3 tools/ci/install_with_hashes.py --requirements requirements-locks/requirements-docs-actions.lock`, then `python3 tools/ci/install_with_hashes.py --requirements requirements-locks/requirements-web-docs.lock`
- The web-docs candidate gate sets `YOLOZU_REQUIRE_REAL_COCO=1`; a missing
  `pycocotools` installation is a failure, not a dry-run fallback.
- Purpose: docs/manual/README and MCP/Actions surface checks without running full runtime regression gates.
- Jobs: `docs_mcp_gate`.

3. `workflows_meta`
- Install: none beyond stock Python on the runner.
- Purpose: release/security workflow regression checks for `.github/workflows/**`-only changes.
- Jobs: `workflows_meta`.

4. `recommended`
- Install: `python3 tools/ci/install_with_hashes.py --requirements requirements-locks/requirements-ci.lock`
- Purpose: pinned interface contract/behavior gates (`schema`, `manifest`, `reference regression`, deep smoke walkthrough, focused tests).
- Jobs: `quality_gate`; `test` is main-push only.

5. `full`
- Purpose: GPU/backend matrix (TensorRT/CUDA/provider parity, full reference regression profile).
- Trigger: manual dispatch only.
- Workflows: `gpu_smoke_machine.yml`, `gpu_practical_suite_machine.yml`, `gpu_zisn_pipeline.yml`, `reference_adapter_full.yml`, `cflite_batch.yml`.

6. `release`
- Purpose: publish-time packaging confidence, including macOS wheel build validation before the Ubuntu publish job.
- Workflows: `publish.yml`, `container.yml`, `manual_doi.yml`, `announce_release.yml`.

## Why this split

- `core` catches packaging/runtime breakages cheaply.
- `docs_mcp` validates docs/MCP changes quickly without spending runtime-heavy CI budget.
- `recommended` gives stable regression signals with an exact-version lock that is installed via a generated `--require-hashes` wheelhouse.
- `full` is intentionally separated because GPU/provider stacks are expensive and noisy for every PR or nightly schedule. The GPU shell helpers and container images now also layer exact-version extras on top of `requirements-locks/requirements-runtime.lock` via `tools/ci/install_with_hashes.py`.

## Optional extras policy

Optional extras are defined in `pyproject.toml` and their rationale is recorded under:

- `[tool.yolozu.optional_extras_rationale]`

CI does not depend on `full` extras as a single install target. Instead, it uses the tiered installs above so failures are easier to localize.
