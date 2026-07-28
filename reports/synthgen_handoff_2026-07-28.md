# SynthGen handoff qualification — 2026-07-28

## Decision

The fresh one-command handoff passed for Open3D `render_only` and the
deterministic `internal_stub` implementation of `bg_only_inpaint`. The result
establishes local execution, strict QA, label preservation, loader
interoperability, overlay generation, and task-native interface evaluation.

It does not qualify an external generative provider or downstream model
accuracy. `appearance_only_conditioned` remains internal experimental and
`full_regen` remains unsupported.

## Reproduction

```bash
./.venv/bin/python tools/smoke_synthgen.py \
  --synthgen-repo ../YOLOZU-synthgen \
  --output-dir /tmp/yolozu_synthgen_one_command_20260728c
```

The output directory must not already contain `generated_handoff`; the command
does not delete or replace it.

## Pinned inputs

- YOLOZU: `0413e9eab41e5560e6a9c0b9102811c5371d1cb7` plus this qualification change
- YOLOZU-synthgen: `b1c15d6a506d056b591b542eaa6b8bace7fceea4`
- global seed: `20260727`
- Python 3.12.13, Open3D 0.19.0, NumPy 2.4.3
- provider: local deterministic `internal_stub`
- external requests, prompt transfer, and provider cost: none; USD 0
- assets: self-authored primitive demo records, CC-BY-4.0, with computed
  canonical-record SHA-256 values in the generator manifest

## Measured results

- generated scene/sample count: 5 (3 train, 2 validation)
- selected appearance sample count: 1
- mode QA: accepted
- background pixels changed: 422,518 / 422,518
- foreground pixels changed: 0 / 167,306
- renderer-owned truth equality across appearance edit:
  `depth_ndc`, `inst_id`, `sem_id`, and `kpts2d` all true
- generator training-loader batch: 2 images, `[2,3,768,768]`
- YOLOZU bridge batch: 2 images, `[2,768,768,3]`; keypoints
  `[2,2,9,3]`
- oracle interface self-check: visible-keypoint error 0 px, depth MAE 0,
  instance pixel accuracy 1.0, semantic mIoU 1.0

The prediction artifact is an oracle copy of renderer-owned truth. These
metrics check the interface path and must not be reported as trained-model
quality.

SHA-256:

- selected background-edited image:
  `a1c72bbddd884832a0ec9168a50ccec455d298635c0108f7bb7afc1718b5a525`
- oracle interface predictions:
  `a20c7e30d9ffe2d77df6462abca94474fe267d58fd4d13de043fe5e8eee77423`
- YOLOZU eval report:
  `c4adae13792df57e6b7a4274e924c5eef644cd5cb821385adbd50dcd0f97e873`
- YOLOZU overlay:
  `f6229431dccccd6e87f1d3e50ece663eb11f46283994d1fedd0f6bb810567c81`
- machine-readable summary:
  `08d48c41f61ffb700ac68ffd8c93d3fc229aef8a445da7147ab0676687db3054`

## API and agent use

Python code should use `SynthGenShardDataset` and
`collate_synthgen_batch`; stream consumers can use
`SynthGenStreamDataset`. Filter independent schema families with `schema_id`.

Agents should inspect tool id `smoke_synthgen` in `tools/manifest.json`, call
`--help`, then parse `smoke_synthgen_summary.json`. A run is unusable when its
top-level status is not `ok`, QA is rejected, or any `truth_equal` value is
false.

## Rejected path and residual boundary

The placeholder renderer has no background pixels, so background-only editing
is now rejected and the generator exporter fails closed. A real external
provider still requires separate evidence for provider/model version, prompts,
seeds, privacy and retention terms, cost, failures, and the same pixel/truth
invariants.
