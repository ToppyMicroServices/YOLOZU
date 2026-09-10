# Labeled sample and compatibility audit

Date: 2026-09-10. Scope: the CPU validation/evaluation product path, not model
inference, GPU qualification, or evidence about why people stop using YOLOZU.

## Changes

- `yolozu demo dataset` produces eight visible synthetic images, matching YOLO
  bbox labels, a negative validation image, class maps, known predictions, a
  preview, and a hashed/versioned manifest. Existing output paths are refused.
- Loading and migration reject unsupported wrapper versions before normalization;
  explicit null entry versions are rejected. Strict class IDs exclude booleans.
  Metadata-free migration no longer creates incomplete inference metadata.
- Bbox COCO evaluation distinguishes exact paths, basenames, and stems. It
  rejects ambiguous fallback matches and resolves full-dataset identity before
  selecting a `max_images` prefix.
- Candidate-wheel CI checks Linux Python 3.10–3.14, macOS Python 3.14, Windows
  Python 3.12, and Linux/Python 3.10 with the declared core/COCO dependency floors.
  Workflow configuration alone is not evidence of successful execution.

The [candidate run](https://github.com/ToppyMicroServices/YOLOZU/actions/runs/34489258251)
passed all eight lanes for PR head `689d682` (tested merge `d6ccc93`). The floor
lane used exactly the declared NumPy 1.24.0, PyYAML 6.0, Pillow 12.2.0,
typing_extensions 4.8.0, and pycocotools 2.0.7. All lanes reported unchanged
sample files and identical evaluation after relocation. The Windows wheel's
content hash differs from Unix-built wheels; each lane verifies its own exact
installed wheel, not cross-platform byte identity.

## Completed local checks

The non-editable candidate 4.8.0 wheel passed `pip check`, installed-file identity
checks, sample generation, strict train/val/YAML/predictions validation, and
official COCO evaluation. Copying the whole sample into a path containing spaces
and evaluating from an unrelated working directory gave identical metrics and
unchanged file hashes. AP50, AP50:95, AP75, and AR100 were all 1.0 because the
predictions come from the labels.

- Candidate wheel SHA-256:
  `2e2c20f3dfb90f13b18041837c9f9ae86ed9308691a7c4e278e58a31083429f7`.
- Environment: macOS arm64, Python 3.14.6, NumPy 2.5.3, Pillow 12.3.0,
  PyYAML 6.0.3, pycocotools 2.0.11.
- Detailed local candidate evidence: `candidate-v4.8/compatibility-report.json`.
- Public 4.6.0 and 4.7.0 both consumed one earlier candidate-generated sample
  without changing its files, with the same strict validation and metrics before
  and after relocation. See `public-release-reuse/release-reuse-report.json`.
  Installed package files were checked against each public wheel's bytes.

The 4.8.0 candidate also passed all 13 installed evaluate/export/debug/demo
commands from the prior audit, including proof PNG decoding and official COCO
evaluation. Public docs (123 examples) and new sample/version docs (6 examples)
passed independent command checks; all 21 manual chapters and 132 manifest tool
help surfaces passed. Two combined audit attempts exceeded their nested
30-second timeout under concurrent local load; the same subchecks were run
separately, and the combined audit also passed in the PR's docs CI lane.

Final manual review found three incomplete optional `meta` examples. Detection,
Keypoints, and 6DoF examples now pass strict prediction validation without
warnings, enforced by `tests.test_manual_prediction_examples`. Paragraph/listing
spacing was fixed where the frame top edge extended into the right margin.

The 175-page local distribution manual has SHA-256
`bdc64b35cd527b5d9e8f1caa7ea1d64a973cc1c83e19b6cac4f9b2a6ef0f579c`.
qpdf and Ghostscript found no structural diagnostics; Poppler, PDFium, and
PDFKit each rendered all pages. The final visual review covers all-page contact
sheets and selected full-size pages, with the five changed pages reviewed at
full size in all three renderers. See `manual-v4.8-corrected-qa/qa-report.json`
and its visual-review evidence. The release-triggered manual rebuild is a
separate artifact and needs its own downloaded-byte check.

## Historical and current runtime evidence

The first full source suite ran 1,734 tests, with one failure and 18 skips. Its
failure compared the current evaluator source to hashes recorded by a July
case study. All nine recorded hashes still match that historical Git commit.
The historical artifact is unchanged; the test now checks its own source and
protocol identity, and verifies historical Git bytes when that commit is
available, without fetching in shallow CI.

A fresh, separate Mask R-CNN eager/TorchScript run on CPU processed the same two
real images using the cached, hash-checked official weights. Both paths emitted
12 detections, passed parity, and produced identical COCO metrics, also matching
the historical results. This is a two-image execution check, not broad accuracy
or GPU qualification. Its single-run timings were observed while other local
checks were running and must not be used to rank runtimes. See
`runtime-parity-v4.8/summary.json`; the output records the dirty worktree state
rather than inventing a clean revision. A final full-suite rerun is recorded
separately so the earlier failed run is not overwritten.

The second full run executed 1,736 tests with one failure, no errors, and 18
skips. Its only failure expected the removed incomplete `keypoints_format`
metadata field in the manual. That text-presence check now looks for the actual
`keypoints` payload; all 22 manual tests pass, including strict validation of
the JSON examples. The second failure log is retained in `full-suite/final-2`.

The legacy development `.venv` contains stale installed distribution metadata
and Pillow below the current declared floor. Source-checkout unit tests in that
environment are regression evidence, not proof that a fresh current package
installation has compatible dependencies. The separate candidate environment
above has current, non-editable metadata.

## Release boundary

This report initially records local candidate evidence. Release publication,
cross-platform CI, the public manual's final bytes, and hosted-doc deployment
must be checked separately and recorded in the release verification report.
Human first-use observation remains `YOLOZU-ll2.84`; no consented user session
has been replaced with a synthetic run. Work is tracked by `.85` and `.86`.

Large local environments, images, wheel files, and full renderer outputs remain
in this directory but are not all committed. Compact JSON evidence and the
reproducibility helpers are retained in version control.
