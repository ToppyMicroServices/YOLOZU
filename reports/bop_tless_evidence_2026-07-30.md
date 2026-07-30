# BOP T-LESS object 6DoF evidence — 2026-07-30

Status: **real T-LESS data and strict GT independently reproduced; diagnostic
metrics show no matched predictions; Research; efficacy not established**.

## Dataset and protocol

The run downloaded the official T-LESS BOP archives and verified them before
extraction:

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `tless_base.zip` | 49,597 | `dd70ca884b7c471a530a952f70c5ab2c212f3d2c2f371be86397442b97d70a7e` |
| `tless_models.zip` | 33,468,762 | `6a29c59766b8d2af05e62e71739f0ca7243ad81bc3c7f9a24925504a8cb37928` |
| `tless_train_primesense.zip` | 2,486,152,818 | `7262fdf0e9de09cf5d051ca108d860c7c5fd563ee58496751e970458c6897ba1` |

The download manifest SHA-256 was
`d6829d3355956de84d9fccccbcf03d6137fb9b98aab0370976b5d935cd9afbc1`.
T-LESS data is CC BY 4.0; YOLOZU code remains Apache-2.0.

This is a preregistered diagnostic frame holdout from
`train_primesense`, not the official BOP test benchmark:

- training: first 24 eligible frames after excluding `frame_id % 5 == 0`;
- validation: first 8 held-out frames where `frame_id % 5 == 0`;
- seeds: 11, 22, 33;
- baseline: deterministic zero-epoch initialization;
- trained: one epoch, maximum four optimizer steps;
- image size 96, batch size 2, CPU.

## Strict ground truth

No model inference generated labels. The validation set contained:

| Supervision | Source | Count |
|---|---|---:|
| bbox | BOP `bbox_visib` GT | 8 instances |
| segmentation | BOP `mask_visib` GT | 8 instances |
| 2D keypoints | deterministic CAD anchors projected with BOP GT `K/R/t`, visibility confirmed by instance `mask_visib` | 27 visible points |
| depth | BOP depth GT | 8 images |
| object 6DoF | BOP object-to-camera `R/t` GT | 8 instances |

CAD points and object symmetry metadata were retained for metric-aware
ADD/ADD-S evaluation. Translations and CAD points use metres. Human 3D
skeleton pose is unsupported.

## Before/after task-native result

All three seeds emitted separate bbox and task-native before/after fields.

| Metric family | Before | After | Interpretation |
|---|---:|---:|---|
| bbox mAP50:95 | 0.0 | 0.0 | no improvement |
| segmentation mAP50:95 | 0.0 | 0.0 | no predicted masks |
| keypoint PCK | null | null | no GT/prediction instance match |
| depth absolute error | null | null | no matched pose/depth pair |
| rotation/translation/pose success | null | null | no matched pose pair |
| ADD / ADD-S | null | null | no matched pose pair |

Null is intentional and is not converted to zero. It means that the metric was
not measurable because the bounded checkpoints produced no matched
predictions. Thresholds were IoU 0.5, rotation success 15 degrees,
translation success 0.1 m, with ADD/ADD-S in metres.

Across the six baseline/trained jobs, peak RSS ranged from 344,670,208 B to
1,976,844,288 B and each job completed in 3–9 seconds. Every run recorded checkpoint,
config, run-metadata, resource, and dataset hashes.

## Independent reproduction

A clean Python 3.12.13 environment independently extracted the same three
archives, reconverted the data to a separate output root, and reran all
baseline/trained jobs for seeds 11/22/33.

- primary qualification summary SHA-256:
  `6ee80f6ee18af9985bb004c6239303b246054f6a7a4a89b81c9e86e5758b4a45`
- independent qualification summary SHA-256:
  `9c253dbae2bceaf83358b6e9abcc1535c6b28efb196998c99a80fa14c30b0c6f`
- independent comparison: `semantic_match=true`

The summaries follow
[`../docs/schemas/bop_tless_qualification.schema.json`](../docs/schemas/bop_tless_qualification.schema.json).
The exact machine-readable
[`primary`](bop_tless_qualification_primary_2026-07-30.json) and
[`independent`](bop_tless_qualification_independent_2026-07-30.json)
summaries are tracked with this report.

## Decision

Real acquisition, strict ground-truth provenance, three-seed before/after
evaluation, artifact/resource evidence, and independent reproduction are
complete. The result remains `hold` and `not_established` because the
diagnostic holdout is not the official BOP test protocol and the bounded
checkpoints produced no matched pose predictions. This evidence must not be
reported as human 3D pose support or as a positive BOP benchmark result.
