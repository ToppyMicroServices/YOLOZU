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
- `YOLOZU_MANUAL_CONCEPTRECID` (recommended): manual conceptrecid for newversion publishing

## Trigger model

- `release: published`: builds and publishes manual record in production Zenodo.
- `workflow_dispatch`: supports production/sandbox, overrides, and draft mode.

Workflow inputs:

- `zenodo_environment`: `production` or `sandbox`
- `manual_version`: optional override (defaults to release tag/ref; leading `v` is removed)
- `manual_conceptrecid`: optional override (otherwise uses `YOLOZU_MANUAL_CONCEPTRECID`)
- `software_concept_doi`: optional override (otherwise uses `YOLOZU_SOFTWARE_CONCEPT_DOI`)
- `publish_record`: `true` publishes, `false` keeps a draft

## Zenodo behavior

1. Build manual PDF.
2. If conceptrecid exists, resolve the latest **record id** via Zenodo Records API:
  - `GET /api/records/?q=conceptrecid:<conceptrecid>&sort=mostrecent&size=1`
  - Then call `POST /api/deposit/depositions/<record id>/actions/newversion`.
3. Else create a new deposition.
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
- `state` (`published` or `draft`)
