## Summary

YOLOZU 4.8.0 adds a reusable labeled sample and improves first-run evaluation
and prediction-file compatibility.

Install from [PyPI](https://pypi.org/project/yolozu/4.8.0/):

```bash
python -m pip install "yolozu[coco]==4.8.0"
yolozu demo dataset --run-dir sample-data --seed 0
```

Or download the [ready-made labeled sample ZIP](https://github.com/ToppyMicroServices/YOLOZU/releases/download/v4.8.0/yolozu-labeled-sample-v4.8.0.zip).
It includes images, YOLO labels, known predictions, a preview, and reuse instructions.

## Changes

- Generate eight synthetic images with YOLO bbox labels, a preview, known
  predictions, and a versioned manifest using `yolozu demo dataset`.
- Keep samples portable across directories and older compatible readers.
- Repair doctor proof images and improve installation and evaluation guidance.
- Reuse proxy mAP preparation across thresholds; reject ambiguous COCO image
  matches while preserving exact-path and unique-basename behavior.
- Reject unsupported prediction versions and invalid strict class IDs; keep
  metadata-free migration output valid.
- Synchronize the CLI manifests, generated documentation, and TeX/PDF manual.

## Testing

- The final source-checkout suite ran 1,736 tests with 18 environment skips,
  zero failures, and zero errors; Ruff and whitespace checks passed.
- The 4.8.0 candidate passed installed-wheel identity checks, `pip check`, strict
  sample validation, and official COCO evaluation before and after relocation.
- Published 4.6.0 and 4.7.0 both read the same generated sample on macOS arm64 /
  Python 3.14.6 without changing files or evaluation results.
- All eight candidate CI lanes passed: Linux Python 3.10–3.14, macOS Python
  3.14, Windows Python 3.12, and Linux Python 3.10 with declared core/COCO
  dependency floors.
- A two-image CPU Mask R-CNN execution check passed eager/TorchScript parity
  with identical COCO metrics. This is not a performance benchmark.
- The 175-page local manual passed qpdf/Ghostscript structural checks and
  all-page rendering with Poppler, PDFium, and PDFKit. The release rebuild is
  verified separately after publication.

## Post-publication checks

- The actual public wheel and source archive match PyPI hashes. A fresh macOS
  arm64 / Python 3.14.6 public-wheel installation passed both 13-check sample
  compatibility and 13-check first-use journeys, including official COCO
  evaluation and relocation. All 360 installed package files match the wheel.
- All eight compatibility lanes passed again on the exact release commit.
- The release ZIP was downloaded again and all 23 sample files matched.
- The [public Manual 4.8.0](https://zenodo.org/records/22699038) matches the
  release workflow PDF. Its 175 pages passed structural checks and rendering
  with all three engines. All contact sheets and selected full-size pages were
  reviewed; no layout blocker was observed at those scales.

## Notes

The sample's predictions come from its labels. Its perfect score is a workflow
check, not model accuracy. This release does not establish GPU/MPS performance,
support for every newer dependency, or the cause of low adoption.
