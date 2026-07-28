# Non-TTT Artifact Research Evidence — 2026-07-28

## Confirmed scope

This report qualifies two artifact-only Research lanes:

- offline prediction distillation
- Hessian offset refinement

It does not qualify training-time SFT/SDFT, model improvement, or production
promotion. The run started from source commit
`e745253ec7d034e83bdcecd8141d3922aa31586d` with no tracked changes.

## Reproduction

```bash
./.venv/bin/python tools/qualify_artifact_research.py \
  --output-dir /tmp/yolozu-artifact-research
```

The command refuses repository-external student, teacher, dataset, or config
inputs and refuses to replace an existing output directory.

Environment:

- macOS 26.5.2 arm64
- Python 3.14.6
- torch 2.12.0.dev20260330
- pycocotools 2.0.11

Inputs:

- student: `reports/predictions_rtdetr_pose_baseline.json`
  - SHA-256: `2099efb5d5c46877c84a3576a7d7bedc5cf36f80062a2539ca7cb481d4fb456a`
- teacher: `data/smoke/predictions/predictions_dummy.json`
  - SHA-256: `e028f2d9fb9f567527cd2b38f94ad7ed366665cec840af61bec62c736013e6fe`
  - coverage: 10 prediction images
- dataset: `data/coco128`, split `train2017`, 128 images
  - tree SHA-256: `2b20f0b2541de8f01167fb71cdd0eed01aac7ec4bdf6c7036fb106332721c71d`

## Results

| Lane | Repetitions | Canonical prediction hash | Total transform time (s) | mAP50-95 | Delta | Decision |
|---|---:|---|---|---:|---:|---|
| Stable baseline | 1 | `fbba36208e5e7acdfc08e4b30a63ae52a76e2a8f6307505b2acca03a975e7f18` | n/a | 0.000000 | n/a | retained |
| Offline distillation | 3 | `2141a153aef6ec4925ba884e7ed0c448215e13332e8c4bada3e3fdb5c32f8d8e` in all runs | 0.036328 / 0.036963 / 0.035906 | 0.073351 in all runs | +0.073351 | hold |
| Hessian refinement | 3 | `3ed633d28f6d031a612c00844be8146c795b15d67620e38edc313243201ed2e0` in all runs | 0.822072 / 0.681246 / 0.684983 | 0.000000 in all runs | 0.000000 | hold |

COCOeval succeeded for the baseline and every transformed artifact. Normalized
prediction content was deterministic across all three repeated inputs for both
lanes.

## Interpretation and boundaries

Distillation added teacher-only detections from a ten-image checked-in smoke
fixture. The positive metric delta verifies artifact transformation and stable
evaluation, but it is not evidence of general model efficacy or an
independently inferred teacher. Promotion is therefore held.

COCO128 has no per-instance depth/mask auxiliary signal for the current
Hessian offset objective. Each repetition recorded `no_signal` for all 1,280
detections and left task metrics unchanged. This is a measured negative
control, not a successful refinement result.

Both methods write separate transformed artifacts and leave the stable input
unchanged, so rollback is selecting the retained baseline artifact.

## Discovered and corrected defect

The first real run failed closed because Hessian `wrap: true` output omitted
required stable `meta` fields. The CLI now emits a strict-valid predictions
interface contract with `adapter`, string `config`, `images`, `tta`, and `ttt`
metadata. A regression test validates the wrapped output before evaluation.

## Evidence artifacts

- machine-readable summary:
  `reports/artifact_research_evidence_2026-07-28.json`
  - SHA-256: `099a0fbd5f0c5b6a7b631e7e0c4a1a371bf353c7037436b892ff1f4414d43c9c`
- full compressed bundle:
  `reports/artifact_research_evidence_2026-07-28.tgz`
  - SHA-256: `705b8f76e1131924f4049952a00efbbb9dc2ce7d27c82dcdd5be64ea59ac06a8`

The bundle contains the normalized baseline, every transformed predictions
artifact, every lane `research_report`, every COCOeval report, and the
qualification summary.

## Promotion trigger

- Distillation: rerun with an independently inferred teacher over the complete
  fixed evaluation set and reproduce a positive metric/cost trade-off outside
  this repository.
- Hessian: use a fixed dataset that supplies the depth/mask signal required by
  the objective, observe actual bounded updates, improve the target metric, and
  reproduce the result independently.
