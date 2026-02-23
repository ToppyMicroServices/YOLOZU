# Releasing to PyPI

YOLOZU is configured for **PyPI Trusted Publishing** (OIDC) via GitHub Actions.
This avoids long-lived PyPI API tokens.

## Trigger model (authoritative)

- PyPI publish workflow: `.github/workflows/publish.yml`
- Actual trigger: **GitHub Release → published** (plus manual `workflow_dispatch`)
- **Tag push alone does not publish to PyPI**
- Container images are separate: `.github/workflows/container.yml` can publish on tag/manual runs
- `publish.yml` now validates that release tag `vX.Y.Z` matches `yolozu/__init__.py::__version__`.

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
- Required CI: `.github/workflows/ci.yml`
- Compatibility gates:
  - schema compatibility
  - golden compatibility (`python3 tools/check_golden_compatibility.py`)
  - wheel/sdist contents gates

## Each release

1) Bump version in `yolozu/__init__.py` (`__version__`).

2) Update `CHANGELOG.md`:
   - create/update `[X.Y.Z] - YYYY-MM-DD`
   - keep `[Unreleased]` empty (or with new post-release changes only)
   - make `Breaking` / `Changed` / `Deprecated` explicit

3) Push to `main`.

4) Create and push an annotated tag:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

5) Create a GitHub Release from that tag and click **Publish release**.

This action triggers `.github/workflows/publish.yml`, which builds and publishes to PyPI.

6) Verify publish result:
   - GitHub Actions `publish` job is green
   - PyPI project page shows the new version

## Optional manual publish trigger

`publish.yml` also supports manual execution (`workflow_dispatch`) for operational recovery.
Use only when release metadata (tag/release/version/changelog) is already aligned.

## Manual PDF DOI (separate from software DOI)

YOLOZU publishes software artifacts via GitHub Release/PyPI and can publish the manual PDF as a separate Zenodo record:

- Workflow: `.github/workflows/manual_doi.yml`
- Trigger: GitHub Release `published` (or manual `workflow_dispatch`)
- Build: `manual/build/yolozu_manual.pdf`
- Zenodo flow:
  - If `YOLOZU_MANUAL_CONCEPTRECID` exists: call `actions/newversion`
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

- `YOLOZU_MANUAL_CONCEPTRECID` (manual conceptrecid used for versioned `newversion` publishing)

Workflow artifact:

- `reports/manual_doi_publish.json` (DOI, concept DOI, conceptrecid, deposition id, URL, state)

## Notes

- You cannot upload the same version twice to PyPI. Always bump `__version__` before publishing.
- If you prefer manual Twine publish, use `python -m build` + `python -m twine upload dist/*` with a PyPI API token.
- If publish is blocked by environment protection, update GitHub **Settings → Environments → pypi** deployment rules and rerun the release workflow.
- 1.0 compatibility boundary is defined in `docs/release_1_0_stability.md`.
- Security reporting/support policy is defined in `SECURITY.md`; dependency/license policy is in `docs/license_policy.md`.
