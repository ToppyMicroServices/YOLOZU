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

Inspect the two entrypoints before a network or write operation:

```bash
bash deploy/runpod/bootstrap_bop_tless_train_primesense.sh --help
bash deploy/runpod/run_rtdetr_pose_bop_tless_pose_eval.sh --help
```

Download and convert directly:

```bash
python3 tools/download_bop_dataset.py \
  --dataset tless --out /workspace/bop
python3 tools/prepare_bop_yolozu.py \
  --bop-root /workspace/bop/tless --split train_primesense \
  --out /workspace/bop-yolozu-tless --out-split train2017 \
  --partition-modulus 5 --partition-remainder 0 --partition-mode exclude \
  --link-images
```

Add the deterministic validation partition only to an owned conversion root:

```bash
python3 tools/prepare_bop_yolozu.py \
  --bop-root /workspace/bop/tless --split train_primesense \
  --out /workspace/bop-yolozu-tless --out-split val2017 \
  --partition-modulus 5 --partition-remainder 0 --partition-mode include \
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
    "--bop-root", "/workspace/bop/tless",
    "--split", "train_primesense",
    "--out", "/workspace/bop-yolozu-tless",
    "--out-split", "train2017",
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
metadata is preserved in the sidecar.

The declared JSON outputs are:

- `download_manifest.json`, schema:
  [`schemas/bop_download_manifest.schema.json`](schemas/bop_download_manifest.schema.json)
- `conversion_reports/<split>.json`, schema:
  [`schemas/bop_conversion_report.schema.json`](schemas/bop_conversion_report.schema.json)

## Qualification boundary

`deploy/runpod/run_rtdetr_pose_bop_tless_pose_eval.sh` currently performs a
diagnostic frame holdout from `train_primesense`: frame IDs whose remainder is
zero form validation, and the remaining frames form training. It evaluates a
deterministic zero-epoch initialization baseline for seeds `11,22,33`, followed
by epoch budgets `1,5,20` by default. Each run records config/checkpoint hashes,
elapsed seconds, metrics, and license boundaries. This diagnostic frame holdout
is not the official BOP test protocol and must not be reported as a BOP
benchmark result.

No checked-in run currently satisfies all of the following:

- baseline-versus-trained results on the pinned real subset;
- three completed seeds with detection, rotation, translation, pose success,
  ADD, and ADD-S metrics;
- recorded checkpoint/config hashes, runtime/cost, and failure cases;
- independent reproduction from a release-addressable evidence archive.

Until those items are complete, this lane remains Research and model efficacy
is `not_established`. See
[`../reports/bop_pose_readiness_2026-07-28.md`](../reports/bop_pose_readiness_2026-07-28.md).
