# External runtime training evidence — 2026-07-30

Status: **three available lanes independently reproduced; five tested lanes
unavailable on this host; Experimental; efficacy not established**.

The machine-readable record is
[`external_runtime_qualification_2026-07-30.json`](external_runtime_qualification_2026-07-30.json),
validated by
[`../docs/schemas/external_runtime_qualification.schema.json`](../docs/schemas/external_runtime_qualification.schema.json).

## Environments

| Role | Python | Torch | Device |
|---|---|---|---|
| Primary | 3.12.13 | 2.14.0.dev20260729 | Apple CPU |
| Independent | 3.14.6 | 2.12.0.dev20260330 | Apple CPU |

The primary environment installed the external runtimes and official source
launchers. The independent environment was separately provisioned for each
successful lane. Runtime packages remain outside YOLOZU and are not bundled.

## Real training and handoff

| Lane | Pinned runtime | Primary training | Independent training | Result |
|---|---|---:|---:|---|
| Ultralytics | 8.4.112 | 1 epoch / 5 batches | 1 epoch / 5 batches | predictions interface contract valid; bbox mAP50:95 0.0 |
| HF DETR | Transformers 5.14.1 | 1 optimizer step | 1 optimizer step | checkpoint and predictions byte-identical; bbox mAP50:95 0.0 |
| Detectron2 | 0.6, source `b4a4a3b` | 1 optimizer step | 1 optimizer step | predictions interface contract valid; bbox mAP50:95 0.0 |
| YOLOX | 0.3.0, source `6ddff48` | failed | not applicable | upstream trainer requires CUDA on the tested host |
| MMDetection | 3.3.0 | failed | not applicable | `mmcv._ext` unavailable for the installed nightly Torch |
| MMPose | 1.3.2 | failed | not applicable | `xtcocotools` unavailable on Python 3.12 |
| MMSeg | 1.2.2 | failed | not applicable | `mmcv._ext` unavailable for the installed nightly Torch |
| TAO | not installed | failed | not applicable | structured `E_EXTERNAL_RUNTIME_MISSING` |

“Failed” means a real non-dry launcher was invoked and produced a
machine-readable runtime failure. Config projection is not counted as
training. YOLOX printed an uncaught traceback but exited zero; YOLOZU now
rejects that pattern instead of reporting false success.

## Resource and artifact evidence

| Lane | Primary wall / peak RSS | Independent wall / peak RSS | Checkpoint reproduction |
|---|---|---|---|
| Ultralytics | 22.45 s / 782,204,928 B | 21.42 s / 751,239,168 B | semantic result matched; checkpoint differed across Torch versions |
| HF DETR | 0.071 s / 393,904,128 B | 0.048 s / 417,234,944 B | exact SHA-256 match |
| Detectron2 | 1.525 s / 783,843,328 B | 1.458 s / 902,430,720 B | semantic result matched; checkpoint differed across Torch versions |

The complete report contains config, checkpoint, predictions, evaluation, and
wrapper-report SHA-256 values. The HF DETR checkpoint and predictions matched
exactly across Python/Torch environments. Ultralytics and Detectron2 produced
the same zero-detection/zero-mAP evaluation semantics, but their serialized
checkpoint bytes differed.

## License boundary

- Ultralytics 8.4.112 reported `AGPL-3.0` in installed package metadata.
- Transformers 5.14.1 and the tested OpenMMLab packages reported Apache 2.0
  metadata.
- Detectron2 and YOLOX were installed from their official Apache-2.0 source
  repositories at the recorded commits.
- TAO was not installed, so its deployment terms were not qualified.

These runtimes are optional external dependencies. A downstream deployment
must review its selected model, runtime, and distribution terms separately.

## Decision

The task proves real execution, failure normalization, resource recording, and
predictions interface contract handoff for the tested bounded fixtures. It
does not establish training quality: all successful evaluation metrics were
zero and five runtimes remain unavailable on this macOS CPU host. All eight
lanes remain Experimental, with `hold` and `not_established`.
