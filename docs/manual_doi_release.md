# Manual PDF DOI release (Zenodo)

Use `.github/workflows/manual_doi.yml` to publish the manual PDF (`manual/build/yolozu_manual.pdf`) to Zenodo as a **separate** DOI record from the software release.

## Goal

- Keep software DOI lifecycle independent from manual PDF DOI lifecycle.
- Maintain machine-readable linkage from manual record to software concept DOI.
- Version the manual record with Zenodo `actions/newversion`.

## Required repo configuration

Secrets:

- `ZENODO_TOKEN` (production Zenodo API token)
- `ZENODO_SANDBOX_TOKEN` (optional, for sandbox runs)

Variables:

- `YOLOZU_SOFTWARE_CONCEPT_DOI` (required): stable software concept DOI, e.g. `10.5281/zenodo.xxxxxxx`
- `YOLOZU_MANUAL_CONCEPTRECID`: required for release-triggered automatic updates after the first manual deposition

## Trigger model

- `release: published` is the single automatic trigger. It builds and publishes the manual record in production Zenodo.
- `workflow_dispatch` is reserved for first-time setup, recovery, sandbox checks, overrides, and draft mode. Do not dispatch it in addition to a normal successful release.
- Runs for the same manual version are serialized. When the configured concept record already contains that published version, the workflow exits idempotently with `state=already_published` before creating a deposition.

Recommended pre-step for release operations:

```bash
bash release.sh --dry-run --allow-dirty --allow-non-main --output reports/release_report.dry_run.json
```

Workflow inputs:

- `zenodo_environment`: `production` or `sandbox`
- `manual_version`: optional override (defaults to release tag/ref; leading `v` is removed)
- `manual_conceptrecid`: optional override (otherwise uses `YOLOZU_MANUAL_CONCEPTRECID`)
- `software_concept_doi`: optional override (otherwise uses `YOLOZU_SOFTWARE_CONCEPT_DOI`)
- `publish_record`: `true` publishes, `false` keeps a draft
- `create_first_deposition`: explicit `true` is required only when creating the first manual record without a conceptrecid

## Zenodo behavior

1. Build manual PDF.
2. If conceptrecid exists, resolve its published records via Zenodo Records API:
  - `GET /api/records/?q=conceptrecid:<conceptrecid>&sort=mostrecent&size=100`
  - If a record already has the requested version, report `already_published` and stop without writes.
  - Otherwise call `POST /api/deposit/depositions/<latest record id>/actions/newversion`.
3. Without a conceptrecid, fail closed unless a manual `workflow_dispatch` explicitly sets `create_first_deposition=true`; release-triggered runs skip rather than create an implicit first deposition.
4. Upload PDF to deposition bucket.
5. Set metadata with:
   - `version` = release version
   - `prereserve_doi = true`
   - `related_identifiers[]` with:
     - `identifier = <software concept DOI>`
     - `relation = isSupplementTo`
     - `scheme = doi`
6. Publish record (unless draft mode).

## Output artifacts

- Workflow artifact: `manual_pdf`
- Workflow artifact: `manual_doi_publish` (`reports/manual_doi_publish.json`)

`reports/manual_doi_publish.json` includes:

- `doi`
- `conceptdoi`
- `conceptrecid`
- `deposition_id`
- `record_url`
- `state` (`published`, `draft`, or idempotent `already_published`)
