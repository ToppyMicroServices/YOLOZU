# Compatible-Host External Runtime Qualification — 2026-07-30

Status: **complete and independently reproduced for runtime availability**.

Decision: **Experimental / `hold`**. The two runs establish that the pinned
YOLOX, MMDetection, MMPose, MMSeg, and NVIDIA TAO training paths can execute on
the tested Linux/CUDA host. This evidence does not establish training quality,
checkpoint byte determinism, or readiness for promotion.

## Scope and immutable runs

The compatible-host workflow supplements, rather than rewrites, the macOS CPU
availability result in
[`external_runtime_evidence_2026-07-30.md`](external_runtime_evidence_2026-07-30.md).
Both successful runs used source commit
`806496d453f8adfb550a9cc1e994182fa04e64b2`:

- [Primary run 30546919180](https://github.com/ToppyMicroServices/YOLOZU/actions/runs/30546919180)
- [Independent run 30548569775](https://github.com/ToppyMicroServices/YOLOZU/actions/runs/30548569775)

Both ran on Linux with a Tesla T4, Python 3.10.13, CUDA-enabled Torch 2.1.2,
MMEngine 0.10.7, MMCV 2.1.0, MMDetection 3.3.0, MMPose 1.3.2, MMSeg 1.2.2,
and YOLOX 0.3.0. The pinned source commits were:

- YOLOX: `6ddff4824372906469a7fae2dc3206c7aa4bbaee`
- MMDetection: `44ebd17b145c2372c4b700bfb9cb20dbd28ab64a`
- MMPose: `5408bc76f5b848cf925a0d1857899011d8c5b497`
- MMSeg: `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8`

## Dataset boundary

The repository-owned preparation step produced two images, 28 instances, and
five classes. Both runs recorded dataset tree SHA-256
`b36d5af592faedf2ae7ca04a91f2495a0569debd6e8b5681bebef0a8270d1435`.
Detection uses the repository fixture bbox labels. Keypoints and segmentation
use bbox-derived runtime-only labels and are not task-quality ground truth.

## Open-source runtime results

All four lanes recorded `returncode=0`, `training_executed=true`, a non-empty
checkpoint, launcher-process wall/CPU/RSS measurements, and valid structural
resume/export/eval/parity handoffs in both runs.

| Lane | Checkpoint bytes (primary / independent) | Wall seconds (primary / independent) | Peak RSS bytes (primary / independent) |
|---|---:|---:|---:|
| YOLOX | 71,850,046 / 71,850,046 | 9.451 / 9.274 | 1,594,777,600 / 1,594,912,768 |
| MMDetection | 333,525,378 / 333,525,378 | 12.537 / 12.370 | 2,170,593,280 / 2,197,729,280 |
| MMPose | 408,488,850 / 408,488,850 | 12.192 / 11.964 | 2,015,174,656 / 2,015,809,536 |
| MMSeg | 392,241,368 / 392,241,304 | 92.161 / 93.848 | 2,202,738,688 / 2,201,714,688 |

Checkpoint SHA-256 values:

| Lane | Primary | Independent |
|---|---|---|
| YOLOX | `d74e7f8a7eb2ab17a99b243b12bbe6ce330946c59a56094096b11ffc24f2bad4` | `c9b4006f86fa31b6b0d43286b3a7f43eb6be149dafed95eca633a2c354b350a1` |
| MMDetection | `7c0fe1f2c6c0f3b7c449bf8764b79a0c57f1b3ae8630c26900ea4581dde25157` | `e682598122baaf9ae63a1e1891ad37ec9d06095f7144fc6ee6a8365c4ee91321` |
| MMPose | `9c790e83833d69f92046206442ae6b0d78b6ee3fa20e062663b325097e96596b` | `a84f46e41b056be6dfd569504799090f5f085ee7c72c4ce20bb0dd518d44a595` |
| MMSeg | `fd64bc621438c883a10ef1bd0b0b26f0c304793d96a06bdebb1ef41d8bf165f7` | `c228b1277e408465b4d277979246928acd74a98788ca11a6a4cfd97ba3b4b378` |

The structural handoff check verifies that each lane declares a command and
output type for resume, export, evaluation, and parity. YOLOX, MMDetection,
and MMPose declare a predictions interface contract export; MMSeg declares a
segmentation predictions interface contract export. These checks validate the
handoff structure. They do not claim that downstream prediction export,
evaluation, or parity execution occurred in this qualification.

## NVIDIA TAO result

Both runs used
`nvcr.io/nvidia/tao/tao-toolkit@sha256:d0d24bc5608832246ed6f7f768b8dbbe429e0e41c580582a0b89606bb9e752a9`.
The vendor status stream contained no `FAILURE` and ended with
`Train finished successfully.`. Each run produced a 367,044,426-byte
checkpoint:

- Primary SHA-256:
  `379f7a518cfc70ff558dc0cef1384ae16dcb7d9bc6b8e309e11fe7f1a13dc82e`
  in 50.782 seconds.
- Independent SHA-256:
  `5ccbd8302fbf77166b416821edb21525d5071d76040509dac921500026c06c5d`
  in 50.092 seconds.

Both vendor completion records reported training loss
`17.32905387878418`. This equality is recorded as observed behavior, not as a
task-quality result.

## Reproduction assessment

The independent run reproduced:

- the source commit, external source pins, runtime versions, GPU family, and
  dataset tree hash;
- all four open-source `training_executed` and checkpoint requirements;
- all resource-use and structural handoff requirements;
- the TAO image digest, vendor completion condition, and checkpoint presence;
- the workflow-level `open_source_lanes=success` and `tao_lane=success`
  outcomes.

Checkpoint SHA-256 values differed in every lane, and the MMSeg checkpoint
size differed by 64 bytes. The evidence therefore supports semantic
reproduction of runtime availability, not checkpoint byte reproducibility.

## Reproduction command

```bash
bash scripts/run_external_runtime_gpu_qualification.sh \
  --output-dir reports/compatible_host_external_runtimes \
  --dataset-root data/real_multitask_fewshot
```

NVIDIA TAO remains a separate vendor-container workflow step so its runtime
and license boundary is explicit.

## Machine-readable evidence

- [Primary open-source qualification](external_runtime_compatible_host_primary_2026-07-30.json)
- [Independent open-source qualification](external_runtime_compatible_host_independent_2026-07-30.json)
- [Primary dataset preparation](external_runtime_compatible_host_primary_dataset_2026-07-30.json)
- [Independent dataset preparation](external_runtime_compatible_host_independent_dataset_2026-07-30.json)
- [Primary TAO evidence](external_runtime_compatible_host_primary_tao_2026-07-30.json)
- [Independent TAO evidence](external_runtime_compatible_host_independent_tao_2026-07-30.json)
- [Primary workflow inventory](external_runtime_compatible_host_primary_workflow_2026-07-30.json)
- [Independent workflow inventory](external_runtime_compatible_host_independent_workflow_2026-07-30.json)

## Evidence boundary

The compatible-host result changes the observed status of these five runtimes
from unavailable on the tested macOS CPU host to executable on the pinned T4
host. It does not invalidate the macOS observation. It also does not establish
training quality because the keypoint/segmentation labels are runtime-only,
the bounded detection lane has no promotion-quality task-native comparison,
and the structural handoff checks do not execute downstream evaluation. All
five lanes therefore remain Experimental with decision `hold`.
