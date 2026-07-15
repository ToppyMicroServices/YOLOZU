# Releasing to PyPI

YOLOZU is configured for **PyPI Trusted Publishing** (OIDC) via GitHub Actions.
This avoids long-lived PyPI API tokens.

## Trigger model (authoritative)

- PyPI publish workflow: `.github/workflows/publish.yml`
- Actual trigger: **GitHub Release → published** (plus manual `workflow_dispatch`)
- **Tag push alone does not publish to PyPI**
- Container images are separate: `.github/workflows/container.yml` can publish on tag/manual runs to GHCR and mirror to `nvcr.io/yolozu/...` when `NGC_API_KEY` is configured
- `publish.yml` validates release tag/manual inputs against `yolozu/__init__.py::__version__`, requires a matching `CHANGELOG.md` section, and confirms that PyPI exposes both the wheel and sdist after upload.

## One-time setup (PyPI)

1) Create a PyPI account and enable **2FA**.

2) Configure a **Trusted Publisher** for this repo.
   - For first release of a new project name, use a **pending publisher** so the project is created on first publish.
   - Recommended:
     - Project name: `yolozu`
     - Repository: `<owner>/YOLOZU`
     - Workflow file: `.github/workflows/publish.yml`
     - Environment: `pypi`

## Release quality gates (must be green)

- Local checks: `docs/release_reliability_checklist.md`
- Required CI: `.github/workflows/build_and_test.yml`
- Compatibility gates:
  - schema compatibility
  - golden compatibility (`python3 tools/check_golden_compatibility.py`)
  - wheel/sdist contents gates

## Each release

1) Run the release helper from `main`:

```bash
bash release.sh
```

The helper bumps `yolozu/__init__.py`, inserts a matching `CHANGELOG.md` release section, runs release checks, commits, tags, pushes, creates the GitHub Release, and dispatches the manual DOI workflow.

For a preview:

```bash
bash release.sh --dry-run --output reports/release_report.dry_run.json
```

2) Verify publish result:
   - GitHub Actions `publish` job is green; its final gate confirms that PyPI exposes the new version with both wheel and sdist
   - PyPI project page shows the new version (operator confirmation)
   - manual DOI workflow either published a record or produced an explicit skip reason

Manual fallback:
- If you bypass `release.sh`, bump `yolozu/__init__.py`, add `## [X.Y.Z] - YYYY-MM-DD` to `CHANGELOG.md`, push to `main`, create an annotated tag, then publish a GitHub Release.
- Tag push alone prepares release metadata but does **not** trigger PyPI publish.

Use `.github/release_notes_template.md` as the minimal release note template.

## Optional manual publish trigger

`publish.yml` also supports manual execution (`workflow_dispatch`) for operational recovery.
Use only when release metadata is already aligned, and provide:

- `expected_version`: exact package version to publish
- `release_tag` (optional): `vX.Y.Z` tag to validate against when re-running release automation

`container.yml` also supports manual execution (`workflow_dispatch`) for container republish/recovery.
Provide:

- `release_tag`: exact `vX.Y.Z` tag to reuse for GHCR + NGC image tags on a manual run

Container notes:
- CPU images: `nvcr.io/yolozu/yolozu`, `nvcr.io/yolozu/yolozu-demo`
- RunPod/TensorRT images: `nvcr.io/yolozu/yolozu-trt`, `nvcr.io/yolozu/yolozu-rtdetr-pose`
- NGC publication requires repository secret `NGC_API_KEY`

Manual runs now fail fast if `__version__`, the optional tag, and `CHANGELOG.md` do not agree.

## Manual PDF DOI (separate from software DOI)

YOLOZU publishes software artifacts via GitHub Release/PyPI and can publish the manual PDF as a separate Zenodo record:

- Workflow: `.github/workflows/manual_doi.yml`
- Trigger: GitHub Release `published` (or manual `workflow_dispatch`)
- Build: `manual/build/yolozu_manual.pdf`
- Zenodo flow:
  - If `YOLOZU_MANUAL_CONCEPTRECID` exists: resolve latest record id via Zenodo Records API, then call `actions/newversion`
  - Else: create a new deposition
  - Upload PDF to bucket, set metadata, and publish (or keep draft on manual runs)
- Linkage:
  - `related_identifiers` is written with relation `isSupplementTo`
  - Identifier source: repo variable `YOLOZU_SOFTWARE_CONCEPT_DOI`

Required repository secrets:

- `ZENODO_TOKEN` (production Zenodo)
- `ZENODO_SANDBOX_TOKEN` (optional, for sandbox runs)

Required repository variables:

- `YOLOZU_SOFTWARE_CONCEPT_DOI` (software concept DOI; stable cross-version reference)

Recommended repository variable:

- `YOLOZU_MANUAL_CONCEPTRECID` (manual conceptrecid; workflow resolves this to the latest record id before calling `actions/newversion`)

First-time release note:

- If you do not yet have a software concept DOI (no prior Zenodo record), the release-triggered `manual-doi` workflow will skip automatically.
  After the software DOI exists, re-run `manual-doi` via `workflow_dispatch` with `software_concept_doi`, then set `YOLOZU_SOFTWARE_CONCEPT_DOI` for future releases.

Workflow artifact:

- `reports/manual_doi_publish.json` (DOI, concept DOI, conceptrecid, deposition id, URL, state)

## Optional SNS announcement automation (LinkedIn / X / Reddit)

YOLOZU can announce GitHub Releases via `.github/workflows/announce_release.yml`.

- Trigger: GitHub Release `published` (tag push alone does not post).
- Bundle artifacts (always): `reports/announce/announcement.json`, `reports/announce/announcement.md`, `reports/announce/post_report.json`
- Posting occurs only when platform secrets/vars are configured.
- `python3 tools/announce_release.py --x-max-len 280` can be used to override X text truncation for dry-runs or ops recovery.

Required secrets/vars:
- LinkedIn:
  - `LINKEDIN_ACCESS_TOKEN`
  - `LINKEDIN_AUTHOR_URN` (e.g. `urn:li:person:...` or `urn:li:organization:...`)
- X:
  - `X_API_KEY`, `X_API_KEY_SECRET`
  - `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET`
- Reddit:
  - `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_REFRESH_TOKEN`, `REDDIT_USER_AGENT`
  - Repo variable: `YOLOZU_REDDIT_SUBREDDIT` (e.g. `MachineLearning`)
  - Optional repo variable: `YOLOZU_REDDIT_KIND` (`link` or `text`)

## Notes

- You cannot upload the same version twice to PyPI. Always bump `__version__` before publishing.
- If you prefer manual Twine publish, use `python -m build` + `python -m twine upload dist/*` with a PyPI API token.
- If publish is blocked by environment protection, update GitHub **Settings → Environments → pypi** deployment rules and rerun the release workflow.
- 1.0 compatibility boundary is defined in `docs/release_1_0_stability.md`.
- Security reporting/support policy is defined in `SECURITY.md`; dependency/license policy is in `docs/license_policy.md`.
