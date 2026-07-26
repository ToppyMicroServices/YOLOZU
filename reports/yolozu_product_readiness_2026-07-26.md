# YOLOZU product readiness — 2026-07-26

## Confirmed

- The stable product lane remains strict predictions validation and comparable
  evaluation through the predictions interface contract.
- The supported Python surface is the typed, in-process `yolozu.api`. Compact
  AI discovery is available through
  `yolozu-mcp --print-tools --guaranteed --ids-only`.
- PRs #232, #233, and #214 merged the installed AI/MCP surface, fail-closed TTT
  jobs and evidence boundaries, and the searchable web-docs source bundle.
- Web generation accepts only repository-local manifest, schema, content, and
  referenced artifact sources. It rejects unsafe URLs, symlink escapes, unsafe
  output targets, and replacement of non-empty output not wholly owned by its
  provenance inventory.
- Generated web provenance binds the generator, curated content, assets, all
  checked-in JSON Schemas, and every manifest-referenced implementation and
  documentation source except the generated bundle itself.
- TTT remains an opt-in Research lane. The bundled TTT image is explicitly a
  synthetic documentation fixture and does not establish efficacy.

## Observed

- A candidate wheel built from committed source was installed outside the
  checkout and completed the documented strict validation, typed API, and real
  `pycocotools` COCO evaluation path.
- Another external candidate-wheel run completed live CPU `ttt_job` and
  `ctta_job` operations with a compatible RT-DETR checkpoint, predictions,
  reports, terminal exit state, and persisted job state.
- The web safety, provenance, drift, link, search, and replacement suite passed
  22 tests. CI treats the real COCO candidate-wheel route as required.
- Browser dogfood loaded all generated assets, returned 12 results for a `TTT`
  search, displayed the synthetic-evidence boundary, and loaded its 1800×900
  image without console warnings.
- After company-site PR #9 merged as `2e3c2f4`, GitHub Pages reported `built`
  and `https://www.toppymicros.com/yolozu/docs/` served all 14 generated files
  byte-for-byte equal to the bundle on YOLOZU `main`, including provenance,
  search index, and favicon.

## Unknown / risk

- Confirmed external users, repeat use, time to first useful report, and
  design-partner conversion remain unknown. Repository activity is not proof
  of adoption.
- TTT task efficacy, a recommended method, and comparative ranking remain
  unavailable until the evidence required by `YOLOZU-ll2.53` is reproduced.
- Current TTA candidates remain unimplemented and unqualified under the common
  protocol in `YOLOZU-ll2.54`.
- Public PyPI remains on the prior release until `YOLOZU-ll2.30` publishes and
  verifies the next version.

## Recommendation

Publish one verified YOLOZU release that includes the installed API/MCP, TTT
safety boundary, and web-docs source.
Keep TTT efficacy claims blocked. After release verification, prioritize three
observed design-partner onboarding sessions and weekly privacy-safe adoption
snapshots over adding more Research methods.

## Change trigger

- Reopen web-readiness work if the production bundle stops matching YOLOZU
  `main`, provenance validation fails, or live search/assets regress.
- Change the TTT efficacy conclusion only after `YOLOZU-ll2.53` is satisfied
  with release-addressable, independently reproduced artifacts.
- Expand the TTA method list only after `YOLOZU-ll2.54` records primary-source,
  license, RT-DETR applicability, compute, and common-protocol decisions.
- Change the adoption conclusion only after consented external workflow
  observations or privacy-safe repeat-use evidence exists.

## Confidence

- Stable API/CLI, packaging, web-generation safety, and provenance: high.
- TTT execution safety and Research-only boundary: high.
- TTT efficacy and comparative quality: unavailable.
- External adoption: unavailable from current evidence.
