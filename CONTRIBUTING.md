# Contributing to YOLOZU

Thanks for considering a contribution.

## Quick start (dev setup)

```bash
python3 -m venv .venv
. .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m unittest -q
```

## What to contribute

- Bug fixes (especially anything that affects `pip install yolozu` users).
- Docs: clearer quickstarts, end-to-end examples, and “pip users vs repo users” alignment.
- New evaluators / validators that fit the contract-first design (predictions JSON interoperability).

## Issue tracking

Maintainers use **bd (beads)** internally for task tracking. External contributors can:
- open a GitHub Issue describing the problem/feature, or
- open a PR directly with context and a minimal repro/test when applicable.

## Code style

- Lint: `ruff check .`
- Tests: `python -m unittest`

## Test policy

YOLOZU has a general policy that when major new functionality is added, corresponding automated tests should be added in the same pull request whenever practical.

Expected contribution behavior:
- new functionality: add focused automated tests that exercise the new behavior
- behavior changes: update existing tests and add regression coverage for the changed path
- docs-only or comment-only changes: explain in the PR why no automated tests were needed
- parser / canonicalization / predictions interface contract changes: add or update tests under `tests/`, and update any related fuzz or compatibility coverage when applicable

If a change affects CLI behavior, schema/interface contract behavior, or manifest-driven docs, contributors should also update the relevant docs in the same pull request.

## Local pre-push gates (recommended)

To catch CI failures before pushing, run:

```bash
bash scripts/pre_push.sh
```

To enable automatic pre-push checks, install repo-local git hooks:

```bash
bash scripts/install_git_hooks.sh
```

## Pull requests

PRs should include:
- a clear description of the change and why it’s needed
- tests (or a note explaining why tests aren’t practical)
- docs updates when behavior/CLI changes
- manifest/manual updates when the change affects machine-readable tool docs or published operator guidance

Maintainers and contributors should treat the test suite as part of the release surface:
- major new functionality should land with automated tests in the same pull request whenever practical
- bug fixes should add regression coverage for the observed failure mode whenever practical
- docs-only changes may omit tests, but the PR should say so explicitly
- interface contract, schema, canonicalization, and workflow changes should update related tests in the same pull request

External-facing hygiene (recommended):
- Keep `main` stable and release-friendly (small PRs, minimal churn).
- Large work: use a feature branch and link an issue (context + scope).
- Quality gates before review: `ruff check .`, `python -m unittest`, and manifest/packaging checks when applicable.
- If behavior/perf changes: attach a small benchmark/report artifact (JSON/CSV) or a short repro script.

## License

By contributing, you agree that your contributions will be licensed under the
Apache License 2.0 (see `LICENSE`).
