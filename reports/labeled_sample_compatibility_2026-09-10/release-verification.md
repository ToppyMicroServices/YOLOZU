# YOLOZU 4.8.0 release verification

Status: GitHub, PyPI, the public manual, and hosted documentation are published
and verified within the scopes below. This is not an all-hardware or all-CI
qualification claim.

## Source and release

- PR: https://github.com/ToppyMicroServices/YOLOZU/pull/301 (merged).
- Release commit: `bd51c77c267fd6e699d1fae406c7b5cca131e1c6`.
- Release: https://github.com/ToppyMicroServices/YOLOZU/releases/tag/v4.8.0
  (published 2026-09-10 23:35:13 UTC, not a draft or prerelease).
- The annotated remote tag resolves to the release commit. Release metadata
  and all four release-helper quality checks pass.
  See `release-tag-report.json`.

The final PR rerun was still queued at merge time. The runtime and build paths
listed in the retained source comparison were unchanged from commit `658e9bf`,
whose PR CI passed; its full-gate job was skipped. Later changes corrected historical
documentation and a stale manual assertion, retained test evidence, and included
the existing main adoption snapshot. The final source suite ran 1,736 tests with
zero failures/errors and 18 environment skips; focused document checks passed.
Normal squash merge was used without changing branch protection or an admin
bypass. An intentional single-program string concatenation warning was checked
by inspecting the AST and rerunning the isolated metadata probe; its PR thread
was resolved with that evidence, not by suppressing the analysis.

## Public package and reusable sample

[PyPI 4.8.0](https://pypi.org/project/yolozu/4.8.0/) is published. Both downloads
match the PyPI size and SHA-256 metadata and are not yanked:

- Wheel: `36894d84deac82971a88bbc07efa2fbe66fe33b85db0a4219919fe9c096d8391`.
- Source archive: `14ba05a02477fe33df843ded2c9294e253a6251d37793af1658e2e516931de18`.

A fresh, non-editable public-package installation on macOS arm64 / Python
3.14.6 passed `pip check`, verified all 360 installed package files against the
public wheel, and completed both 13-check sample compatibility and 13-check
installed-user journeys. The sample still evaluates after moving to a path
with spaces; all 23 files and COCO metrics are unchanged. Source-archive metadata
and four relevant source files match the wheel; a local sdist build was not run.
See `public-v4.8-artifacts/public-verification.json`,
`public-v4.8/compatibility-report.json`, and
`public-v4.8-journey/installed-journey.json`.
The journey's short per-file hash block is a subset, not a complete evaluator
provenance manifest. `public-evaluator-verification.json` separately records
the official COCO evaluator's source files and their public-wheel identity.

The [labeled sample ZIP](https://github.com/ToppyMicroServices/YOLOZU/releases/download/v4.8.0/yolozu-labeled-sample-v4.8.0.zip)
and its checksum are attached to the release. A fresh release-asset download
matches all 23 source files, has safe paths and valid ZIP CRCs, and has SHA-256
`9d439c4cb6408ce955740292e360893adbc5c7d40079cd77111c6fb5c5b75652`.
See `release-assets-verification.json`; the earlier pre-upload package report
is retained as a historical capture. `release-zip-verification.json` repeats
the public-download CRC and path checks with explicit outcomes.

The actual release commit also passed all eight candidate CI lanes in run
34542754702: Linux Python 3.10–3.14, macOS Python 3.14, Windows Python 3.12,
and Linux Python 3.10 with declared core/COCO dependency floors. Each lane checks
its own wheel identity, installation, strict sample evaluation, relocation, and
unchanged sample files. These are release-source candidate builds, not eight
downloads of the public PyPI wheel. See `release-ci-summary.json`.

## Public manual

[Manual 4.8.0](https://zenodo.org/records/22699038) is published with DOI
`10.5281/zenodo.22699038`. The downloaded public PDF is byte-identical to the
release workflow artifact: 175 pages, 3,476,875 bytes, SHA-256
`9270069c713804b42b78aa1a0ab3047e4f9dfae53f2042fe6427026248da136d`.
Its hash differs from the pre-publication local PDF; the public file was checked
independently rather than treating the local build as publication evidence.

The final-public-file QA script exited 0. qpdf and Ghostscript reported no
structural diagnostics. Poppler, PDFium, and macOS PDFKit each rendered all
175 pages. All 27 contact sheets and 20 full-size images were inspected;
no layout blocker was observed at those scales. This is not line-by-line
proofreading, full-size review of every page, or user approval. The public PDF,
all 525 page images, renderer versions, hashes, and detailed logs remain in the
local audit directory. Compact records are in
`public-manual-v4.8/publication-verification.json` and
`public-manual-v4.8-qa/{qa-report.json,visual-review.json}`.

## Hosted documentation

The first deployment, site PR 13 and Pages run 34543622713, served all 14
generated files correctly. Its checks found a real discoverability bug:
the global search omitted manifest example commands, so `demo dataset` had no
global result. That failed check remains in `hosted-docs-v4.8/`.

YOLOZU PR 303 fixed the generator and added two regression tests; its CI and
CodeQL passed before normal squash merge at
`2388280ffcbb154428ff1e91a1695f5b980b2def`. The focused local set passed all
30 tests, the manifest and generated-doc drift gates, Ruff, and `--help`.
Site PR 14 deployed the correction at
`0f746afa3c839568eef5d9dc4fa38671454b02e4`; Pages run 34544824895 succeeded.
All 14 public file hashes match the source, 115 references and seven anchors
resolve, and two outside-bundle site routes plus four selected external URLs
return HTTP 200. See `hosted-docs-v4.8-search-fixed/`.

The delegated HTTP verifier had no browser connection. A subsequent root-agent
check did connect to the in-app browser: entering `demo dataset` in the public
overview search displayed the YOLOZU command result; clicking it reached
`commands.html#tool-yolozu`; filtering and expanding that entry displayed
`python3 -m yolozu demo dataset --run-dir reports/labeled_sample --seed 0`
and the labeled-sample guide link. The overview search screenshot had no
visible overlap at 1280×720. This is a bounded interaction check, not a
responsive-device matrix or a first-time human usability study. See
`hosted-docs-v4.8-search-fixed/browser-verification.json`.

This web-only correction does not change the released Python package or the
version tag. The unrelated dirty canonical website checkout was not modified.

## Workflow readback and remaining work

The release `publish`, `manual-doi`, `container`, and `announce_release` runs
completed successfully on the tagged source. All four container build/push
steps and all four optional NGC mirror steps succeeded; the published containers
were not pulled or executed in this verification. Announcement logs establish
bundle generation and the optional posting command, not delivery to each social
platform. See `release-workflows-verification.json`.

The release-commit main CI run 34542754706 was canceled during the PyInstaller
smoke, after its unit and earlier full-gate checks passed. The newer main run
started 14 seconds before the cancellation; this is consistent with the
configured same-branch cancellation policy, but the initiator is not directly
identified in the retained logs. The older source-equivalent PR run skipped
its full-gate job and does not supply the missing PyInstaller evidence.
The newer main run was still in progress at the recorded snapshot. None of
these pending, skipped, or canceled checks is reported as passing.

A later readback of that same newer main run, 34544605121 at `2388280`,
confirmed success, including the PyInstaller smoke. The Python runtime trees,
packaging metadata, deployment inputs, scripts, dependency locks, and CI
workflow match the release source; the changed files are the web generator,
its search regressions, and generated search data. This supplies a completed
frozen-binary check for the release-equivalent runtime, without rewriting the
earlier canceled result. Optional PyArmor remained skipped. See the separate
`main-ci-final.json` supplement.

`YOLOZU-ll2.85` (actual release surfaces) and `YOLOZU-ll2.86` (compatibility
qualification) are closed. `YOLOZU-ll2.87` now retains only the investigation of
report-only path routing; its frozen-binary evidence gap is resolved by the
supplemental run. Its filter-semantics finding
is separated from a different-version local reproduction; no speculative
workflow or permission changes were made. `YOLOZU-ll2.84` remains open for
consented first-use observation. Beads state was shared at `f7785ae`, with the
successful frozen-binary follow-up at `912fec2`, preserving
all remote rows, tombstones, and interactions and excluding the unrelated
local-only issue from publication.

## Published evidence views

Machine-local paths in the public JSON views are normalized to placeholders.
Exact raw captures remain in the ignored local `private-evidence/` directory.
`redaction-report.json` records the normalization rules and before/after JSON
hashes. Package, PDF, sample, and source-code hashes still refer to the actual
verified bytes. The CI supplement distinguishes the unchanged historical raw
capture from its normalized public view; normalization does not change a
canceled result into a passing one.
This applies to the published head, not retroactive redaction of earlier PR
commits or repository history.

## Limits

Sample predictions come from the labels; AP 1.0 validates the evaluation path,
not model accuracy. The real-model check covers two CPU images and runtime
parity, not a speed ranking, broad accuracy, GPU/MPS qualification, or external
training runtimes. Compatibility evidence is limited to the recorded versions
and platforms. Consented first-use observation remains `YOLOZU-ll2.84`; the
cause of low adoption is not established by these tests.

The release-commit `gpu-ngc` run 34542754748 reports success, but its
`trt_and_opencv_cuda` job was skipped. Runner discovery returned HTTP 403
with the workflow token, so runner availability is unknown. Only the discovery
attempt and summary publication ran; this is not GPU evidence.
