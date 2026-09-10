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
