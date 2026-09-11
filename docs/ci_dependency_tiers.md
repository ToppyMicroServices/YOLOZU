# CI dependency tiers

YOLOZU uses a tiered CI dependency model to keep signal high while avoiding optional-extras combination explosion.

Default cost policy:

- Pull requests run lightweight Ubuntu checks: docs/metadata gates, runtime smoke, and focused quality tests.
- Runtime-affecting pushes to `main` run the full CPU evaluation gate and full unittest discovery.
- GPU checks stay on Ubuntu/Linux GPU runners and run only by manual dispatch.
- The main `ci` workflow uses Ubuntu. Release wheel checks and the separate labeled-sample compatibility workflow also cover macOS; sample compatibility additionally covers Windows.
- Expensive full/fuzz/regression sweeps are manual-only unless there is an explicit release or incident reason to re-enable a schedule.

## Change routing

`.github/workflows/build_and_test.yml` is the routing source of truth. The workflow
starts on every PR and `main` push, then selects jobs by changed files:

| Changed files | Selected checks in `ci` |
| --- | --- |
| Runtime code, tools, scripts, tests, configs, data, baselines, dependency locks, packaging config | Core and recommended gates; full CPU suite only on `main` |
| README, `docs/**`, listed MCP/Actions files and adaptive-roadmap reports | Docs/MCP and AI interface contract gates |
| `manual/**` | Manual build |
| `.github/workflows/**`, routing regression test | Workflow regression tests, including the pinned paths-filter bundle |
| Other reports, `.beads/**`, CHANGELOG, other metadata | Metadata fast path; no runtime dependency installation |

Mixed changes select the union of these checks. The explicit MCP-only exclusions
in the runtime filter still select docs/MCP checks. Dependency files under
`requirements-locks/**` select runtime gates as well as root `requirements*` files.
The docs/Actions and web-docs locks also select the docs/MCP gates they configure.
New runtime/build locations must be added to the filter and its regression tests.

The runtime filter uses one positive brace union with `predicate-quantifier: every`
so all exclusions apply to each file. The other filters use `some` for positive
alternatives. Do not combine independent negative rules with `some`: the pinned
[paths-filter implementation](https://github.com/dorny/paths-filter/blob/fbd0ab8f3e69293af611ebaee6363fc25e6d187d/src/filter.ts)
treats each rule as an independent match, which previously selected full CI for
report-only changes.

Older revisions of the same PR are canceled. Each `main` push has its own
concurrency group, so a later metadata update cannot cancel an in-flight runtime
build. Main builds can therefore overlap. Independent workflows such as CodeQL
and scheduled security checks keep their own triggers; this table does not disable
them. CLI inputs, outputs, and the packaged tool manifest are unchanged.

To run the behavioral routing tests locally, check out the same paths-filter
revision used in the workflow, then run (Python, Git, and Node are required):

```bash
YOLOZU_PATHS_FILTER_ENTRYPOINT=/path/to/paths-filter/dist/index.js \
  python3 -m unittest tests.test_ci_path_routing -v
```

The tests run the actual bundled action in temporary Git repositories, including
added, modified, deleted, renamed, excluded, and mixed-scope files. Without the
environment variable, only structural tests run and behavioral tests are marked
skipped. The workflow regression job always supplies the bundle.

## Tiers

1. `core`
- Install: `python3 tools/ci/install_with_hashes.py --requirements requirements-locks/requirements-runtime.lock --install-local-wheel`
- Purpose: packaging and CLI/runtime smoke only.
- Jobs: `smoke_gate`; `pip_smoke` is main-push only.

2. `docs_mcp`
- Install: `python3 tools/ci/install_with_hashes.py --requirements requirements-locks/requirements-docs-actions.lock`, then `python3 tools/ci/install_with_hashes.py --requirements requirements-locks/requirements-web-docs.lock`
- The web-docs candidate gate sets `YOLOZU_REQUIRE_REAL_COCO=1`; a missing
  `pycocotools` installation is a failure, not a dry-run fallback.
- Purpose: docs/README and MCP/Actions surface checks without running full runtime regression gates. The manual has a separate build job.
- Jobs: `docs_mcp_gate`.

3. `workflows_meta`
- Install: none beyond stock Python on the runner.
- Purpose: release/security and path-routing regression checks whenever workflow files or the routing test change, including mixed-scope changes.
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
