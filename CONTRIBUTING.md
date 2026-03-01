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

External-facing hygiene (recommended):
- Keep `main` stable and release-friendly (small PRs, minimal churn).
- Large work: use a feature branch and link an issue (context + scope).
- Quality gates before review: `ruff check .`, `python -m unittest`, and manifest/packaging checks when applicable.
- If behavior/perf changes: attach a small benchmark/report artifact (JSON/CSV) or a short repro script.

## License

By contributing, you agree that your contributions will be licensed under the
Apache License 2.0 (see `LICENSE`).
