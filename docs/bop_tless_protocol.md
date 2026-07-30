# BOP T-LESS object 6DoF protocol

This page is the source of truth for YOLOZU's Research-stage BOP T-LESS
conversion and object-pose workflow. It does not establish model efficacy.

## Scope and terminology

| Term | Meaning in YOLOZU |
|---|---|
| 2D keypoints | Image-plane points represented as `(x, y, visibility)` |
| Object-space 3D keypoints | Optional `(X, Y, Z)` points in an object or CAD coordinate frame, exposed as `kpts3d_object` by the SynthGen interface contract |
| Object 6DoF pose | Object-to-camera rotation `R` and translation `t`; BOP translations are converted from millimetres to metres by default |
| Human 3D skeleton pose | Not implemented; human 3D skeleton pose is unsupported |

The reference trainer's pose fields and this protocol refer to rigid-object
pose. They must not be described as general 3D pose or human pose support.

## Dataset source and license

- Dataset: BOP T-LESS.
- Upstream terms: CC BY 4.0, as listed by the
  [BOP dataset page](https://bop.felk.cvut.cz/datasets/) and the
  [official BOP T-LESS dataset card](https://huggingface.co/datasets/bop-benchmark/tless).
- YOLOZU code remains Apache-2.0; the dataset license is separate.
- `download_manifest.json` records the fixed upstream URLs, archive byte
  sizes, SHA-256 values, extraction status, overall completion, and license
  source. The opt-in quota-smoke path records `complete: false` and
  `partial_quota`; it is not completed-dataset evidence.

The downloader accepts plain ZIP filenames only, uses the fixed
`bop-benchmark/<dataset>` host path, and rejects absolute, parent-traversing,
backslash, and symlink archive members before extraction.

## Concise CLI

Inspect the entrypoints before a network or write operation:

```bash
bash deploy/runpod/bootstrap_bop_tless_train_primesense.sh --help
bash deploy/runpod/run_rtdetr_pose_bop_tless_pose_eval.sh --help
python3 tools/export_bop19_rtdetr_pose.py --help
python3 tools/summarize_bop19_pose_evidence.py --help
```

Download and convert directly:

```bash
python3 tools/download_bop_dataset.py \
  --dataset tless --out /workspace/bop
python3 tools/prepare_bop_yolozu.py \
  --bop-root /workspace/bop --split train_primesense \
  --out /workspace/bop-yolozu-tless --out-split train2017 \
  --partition-modulus 5 --partition-remainder 0 --partition-mode exclude \
  --cad-keypoints 4 \
  --link-images
```

Add the deterministic validation partition only to an owned conversion root:

```bash
python3 tools/prepare_bop_yolozu.py \
  --bop-root /workspace/bop --split train_primesense \
  --out /workspace/bop-yolozu-tless --out-split val2017 \
  --partition-modulus 5 --partition-remainder 0 --partition-mode include \
  --cad-keypoints 4 \
  --link-images --append-owned
```

`--overwrite` deletes only a non-symlink output bearing
`.yolozu_bop_output.json`. Existing unowned, protected, source-overlapping, and
non-directory outputs are refused. `--append-owned` refuses an existing split.

## Python and agent-facing use

The conversion functions can be invoked without a subprocess:

```python
from tools.download_bop_dataset import main as download_bop
from tools.prepare_bop_yolozu import main as prepare_bop

download_bop(["--dataset", "tless", "--out", "/workspace/bop"])
prepare_bop([
    "--bop-root", "/workspace/bop",
    "--split", "train_primesense",
    "--out", "/workspace/bop-yolozu-tless",
    "--out-split", "train2017",
    "--cad-keypoints", "4",
])
```

An agent should inspect the declarative registry before execution:

```bash
python3 -m yolozu registry show download_bop_dataset -j
python3 -m yolozu registry show prepare_bop_yolozu -j
python3 -m yolozu registry run -n --allow-network download_bop_dataset -- \
  --dataset tless --out reports/bop
```

The dry-run prints the resolved command and declared write/network effects.
Remove `-n` only after reviewing the dataset terms and output root.

## Recorded output

The converter preserves image/class/bbox data, camera intrinsics, object-to-camera
`R_gt`/`t_gt`, source depth references, and deterministic frame partitions. If
a BOP model directory is present, it copies a deterministic, metre-scaled CAD
point subset per object, records model hashes, and attaches per-instance CAD
paths so `eval_pose.py` can report ADD and ADD-S. Available BOP symmetry
metadata is preserved in the sidecar. `--cad-keypoints N` selects deterministic
object-space CAD anchors and projects them with the BOP ground-truth `K/R/t`
into ordinary YOLO keypoint labels. An in-frame anchor receives
`visibility=2` only when its projected pixel is also present in that
instance's BOP `mask_visib`; an in-frame but mask-occluded anchor receives
`visibility=1`. This is strict object-pose GT; it is not a human skeleton
annotation.

The declared JSON outputs are:

- `download_manifest.json`, schema:
  [`schemas/bop_download_manifest.schema.json`](schemas/bop_download_manifest.schema.json)
- `conversion_reports/<split>.json`, schema:
  [`schemas/bop_conversion_report.schema.json`](schemas/bop_conversion_report.schema.json)
- `qualification_summary.json`, schema:
  [`schemas/bop_tless_qualification.schema.json`](schemas/bop_tless_qualification.schema.json)
- official BOP19 three-seed summary, schema:
  [`schemas/bop19_tless_pose_qualification.schema.json`](schemas/bop19_tless_pose_qualification.schema.json)

## Official BOP19 test qualification

The official-test path is separate from the diagnostic
`train_primesense` frame holdout. It trains only from strict real
`train_primesense` GT, exports one target-conditioned estimate for every entry
in `test_targets_bop19.json`, and evaluates the result with the pinned official
BOP toolkit:

```bash
python3 tools/export_bop19_rtdetr_pose.py \
  --bop-root /workspace/tless \
  --targets /workspace/tless/test_targets_bop19.json \
  --config rtdetr_pose/configs/bop_tless_official.json \
  --checkpoint /workspace/run/checkpoint.pt \
  --output reports/yolozu-rtdetrpose-s11_tless-test.csv
```

`tools/summarize_bop19_pose_evidence.py` combines the official VSD, MSSD, and
MSPD scores with matched rotation/translation errors and BOP toolkit
ADD/ADD-S-style errors. No test GT is read during inference. The summary keeps
unmeasurable values as `null`, records the toolkit commit and every input hash,
and supports a separate `--role independent --source-summary ...` replay.

The official T-LESS archive set used for this qualification consists of the
base, models, and `test_primesense` archives. Its license remains CC BY 4.0 and
separate from YOLOZU's Apache-2.0 code license.

## Qualification boundary

`deploy/runpod/run_rtdetr_pose_bop_tless_pose_eval.sh` currently performs a
diagnostic frame holdout from `train_primesense`: frame IDs whose remainder is
zero form validation, and the remaining frames form training. It evaluates a
deterministic zero-epoch initialization baseline for seeds `11,22,33`, followed
by epoch budgets `1,5,20` by default. Each run records config/checkpoint hashes,
elapsed seconds, metrics, and license boundaries. This diagnostic frame holdout
is not the official BOP test protocol and must not be reported as a BOP
benchmark result.

The 2026-07-30 run downloaded and hash-verified the real base, models, and
`train_primesense` archives; built strict bbox/mask/CAD-keypoint/depth/object-
pose GT; evaluated baseline and trained checkpoints for seeds 11/22/33; and
repeated the protocol in a clean Python 3.12 environment. Primary and
independent summaries matched semantically.

All bbox and segmentation mAP values were zero. Keypoint, depth, rotation,
translation, pose-success, ADD, and ADD-S values were null because no predicted
instance matched GT. Null is preserved rather than rewritten as zero. The lane
therefore remains Research with `hold` and `not_established`. See
[`../reports/bop_tless_evidence_2026-07-30.md`](../reports/bop_tless_evidence_2026-07-30.md).

The later official-test qualification is recorded separately in
[`../reports/bop19_tless_official_evidence_2026-07-30.md`](../reports/bop19_tless_official_evidence_2026-07-30.md).
It supersedes only the protocol gap; it does not rewrite the diagnostic
frame-holdout result or promote the lane.
