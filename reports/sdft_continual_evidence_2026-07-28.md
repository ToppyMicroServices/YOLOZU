# SDFT-Style Continual Qualification Evidence — 2026-07-28

## Confirmed scope

This report records a clean three-seed execution of YOLOZU's detector-specific
checkpoint-distillation continual-learning lane. It compares naive sequential
fine-tuning with the repository's SDFT-style regularizer. It is not a faithful
reproduction of demonstration-conditioned, on-policy language-model SDFT, and
it does not support a production or efficacy claim.

The run started from source commit
`d54ea3d2e8647c4742562e639872b2d9c4b83b7c` with no tracked changes.

## Reproduction

```bash
./.venv/bin/python tools/qualify_sdft_continual.py \
  --output-dir /tmp/yolozu_sdft_qualification_20260728_d54ea3d \
  --archive /tmp/yolozu_sdft_qualification_20260728_d54ea3d.tgz
```

Environment:

- macOS 26.5.2 arm64
- Python 3.14.6
- torch 2.12.0.dev20260330
- pycocotools 2.0.11

Protocol:

- source task: repository-local COCO128, all 128 images
- target task: deterministic Gaussian blur, severity 3, seed 2026
- methods: naive sequential fine-tuning and SDFT-style checkpoint distillation
- training seeds: 11, 22, 33
- budget per task: 20 steps, batch size 2, image size 64, CPU
- evaluation: real `pycocotools` COCOeval, mAP50-95
- fairness: common initial checkpoint and identical task data/order/budget
  within each seed

## Observed results

| Seed | Method | Task matrix | Avg. accuracy | Forgetting | BWT | FWT | Decision |
|---:|---|---|---:|---:|---:|---:|---|
| 11 | naive | `[[0, 0], [0, 0]]` | 0.0 | 0.0 | 0.0 | 0.0 | hold |
| 11 | SDFT-style | `[[0, 0], [0, 0]]` | 0.0 | 0.0 | 0.0 | 0.0 | hold |
| 22 | naive | `[[0, 0], [0, 0]]` | 0.0 | 0.0 | 0.0 | 0.0 | hold |
| 22 | SDFT-style | `[[0, 0], [0, 0]]` | 0.0 | 0.0 | 0.0 | 0.0 | hold |
| 33 | naive | `[[0, 0], [0, 0]]` | 0.0 | 0.0 | 0.0 | 0.0 | hold |
| 33 | SDFT-style | `[[0, 0], [0, 0]]` | 0.0 | 0.0 | 0.0 | 0.0 | hold |

All SDFT-minus-naive deltas were zero. For every seed, the task-0 state and
the task-0/task-1 record-order hashes matched between methods. The SDFT task-1
run recorded the prior task checkpoint as its teacher; the naive run recorded
no teacher.

Mean recorded train/evaluation/decision time was 13.974 seconds per naive run
and 14.067 seconds per SDFT-style run. Maximum recorded child peak RSS was
317.44 MiB for naive and 319.23 MiB for SDFT-style.

## Interpretation

This is a measured negative result. The fixed workflow completed with real
COCOeval and comparable provenance, but the deliberately small random-initial
20-step protocol produced no detections that scored under COCOeval. Therefore:

- execution, artifact integrity, and fail-closed promotion behavior are
  established for this bounded diagnostic;
- a retention/adaptation advantage is not observed;
- efficacy remains `not_established`;
- the aggregate decision remains `hold`.

The zero FWT/BWT/forgetting values must not be interpreted as successful
continual learning: every task score was also zero.

## Artifact integrity

The full evidence bundle is published as a
[GitHub prerelease](https://github.com/ToppyMicroServices/YOLOZU/releases/tag/sdft-evidence-2026-07-28).

- archive SHA-256:
  `5dddd416f6b65c817cbdb4cd2bcef81d59a67a329c4f6497ea6dbd16e68e8bba`
- embedded `checksums.sha256` SHA-256:
  `46d93638a71e1161cb4726dd637b1bef0c7f567e41be815795f977bd9ddbaa1a`
- machine-readable summary SHA-256:
  `bc4463cde9e2cb609c521405921945400dafc349f1bdd8786e39d48b568c512f`
- embedded checks: 434 files, all matched before packaging
- clean extraction: 435 archive members; all 434 embedded file checks matched

The bundle contains the schema-defined summary, all per-seed configs, initial
and trained checkpoints, COCO reports, continual run records, promotion
decisions, and file checksums.

## Independent reproduction (2026-07-30)

A separate Python 3.12.13 environment verified the release archive SHA-256 and
all 434 embedded hashes, checked out source commit
`d54ea3d2e8647c4742562e639872b2d9c4b83b7c`, and reran all 12 train/eval
paths (two methods, two tasks, three seeds) with actual `pycocotools`
COCOeval.

- source dataset canonical tree SHA-256:
  `a47ad492999ffad7eec820c39eaa54514fc44fd6b751fe09373b1020e12ff0cc`
- all 256 canonical data-file hashes matched;
- semantic qualification results matched for every seed and method;
- rerun archive SHA-256:
  `a9ba5e748fadb1a28b0e503e71c02258c3e1d8ed28908db6c94abef8693cf3bb`
- rerun qualification summary SHA-256:
  `41015c0a40b76f423ed2a7f15f9e0412dc6e0aa1833c8b8cf5d3431977b0a712`
- rerun checksum manifest SHA-256:
  `c7bd9644f19f7a690840bd8a7382970e5b13e355e91d43ab78fa85f3bee51bd5`

Path-bearing JSON fields differ because the clean extraction used a different
absolute root; content and semantic comparisons exclude those relocation-only
fields. Independent reproduction is established for the bounded diagnostic.

## Remaining promotion conditions

Promotion requires a preregistered task sequence and budget that produce
non-zero task metrics, a positive retention/adaptation trade-off against the
naive baseline across all fixed seeds. The release-addressable independent
reproduction gate is complete, but the all-zero result does not support
efficacy. The lane remains Research and `hold`.
