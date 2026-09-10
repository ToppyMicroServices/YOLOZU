## Summary

YOLOZU 4.8.0 adds a reusable labeled sample and improves first-run evaluation
and prediction-file compatibility.

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

- The 4.8.0 candidate passed installed-wheel identity checks, `pip check`, strict
  sample validation, and official COCO evaluation before and after relocation.
- Published 4.6.0 and 4.7.0 both read the same generated sample on macOS arm64 /
  Python 3.14.6 without changing files or evaluation results.
- Candidate CI exercises multiple Python versions, operating systems, and the
  declared core/COCO dependency floors; see the release verification evidence
  for the completed run outcomes.

## Notes

The sample's predictions come from its labels. Its perfect score is a workflow
check, not model accuracy. This release does not establish GPU/MPS performance,
support for every newer dependency, or the cause of low adoption.
