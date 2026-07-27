# Dataset processing qualification — 2026-07-27

## Scope and result

This report qualifies the currently claimed dataset routes and records the
boundaries that remain unqualified. Dataset I/O remains deferred as a
standalone capability; the evidence applies only to the rows marked qualified
in `docs/dataset_processing_matrix.md`.

- Source base: `fe880c93d185a5261edaf78980882c2e184570dd`
- Candidate wheel: `yolozu-4.6.0-py3-none-any.whl`
- Candidate wheel SHA-256:
  `ebd3224a6a8b793c2586f7cee06612653df3fea06689dff7af3884c3d1afd87c`
- Installed interpreter: CPython 3.12 in an isolated virtual environment
- Local full-training interpreter: CPython 3.14 with Torch
- Platform: macOS arm64

## Detection round trip from an installed wheel

The candidate wheel was run from `/tmp`, outside the source checkout:

```bash
yolozu migrate dataset --from coco \
  --coco-root /Users/akira/YOLOZU/data/conversion_tiny_coco \
  --split val2017 --output /tmp/yolozu_matrix_installed_20260727/wrapper --force
yolozu validate dataset /tmp/yolozu_matrix_installed_20260727/wrapper \
  --split val2017 --strict
yolozu export-dataset yolo --dataset /tmp/yolozu_matrix_installed_20260727/wrapper \
  --split val2017 --out-dir /tmp/yolozu_matrix_installed_20260727/yolo \
  --image-mode copy --force
yolozu export-dataset coco --dataset /tmp/yolozu_matrix_installed_20260727/wrapper \
  --split val2017 --out-dir /tmp/yolozu_matrix_installed_20260727/coco \
  --image-mode copy --force
yolozu export-dataset kitti --dataset /tmp/yolozu_matrix_installed_20260727/wrapper \
  --split val2017 --out-dir /tmp/yolozu_matrix_installed_20260727/kitti \
  --image-mode copy --force
```

Observed:

- Source, wrapper, YOLO export, and COCO export each contained 2 images and 3
  labels.
- Contiguous class IDs remained `[0, 1]`; class names remained `crate`, `cone`.
- Maximum normalized bbox coordinate delta after six-decimal YOLO
  serialization was `4.999999999866223e-7`, within the declared `1e-6`
  round-off policy. No clipping occurred.
- KITTI export contained 2 images and 2 label files and is intentionally
  export-only.
- Wrapper creation took 0.66 seconds on this machine.
- Source annotations SHA-256:
  `242652132b66a7945f95a5d71bc545055e2daa9e7eaf808810d89601ba6054ee`.

## Multi-task subset and training-loader step

```bash
python3 tools/make_subset_dataset.py \
  --dataset data/real_multitask_fewshot --split val --n 2 --strategy first \
  --copy --out /tmp/yolozu_real_multitask_subset
python3 -m yolozu doctor train-dataset --from depth \
  --dataset /tmp/yolozu_real_multitask_subset --split val --output -
python3 -m yolozu doctor train-dataset --from pose6d \
  --dataset /tmp/yolozu_real_multitask_subset --split val --output -
```

Observed:

- 2 images, 3 instances, and 18 keypoint entries were loaded.
- Both records retained mask and depth paths, `K_gt`, `R_gt`, and `t_gt`.
- Keypoint metadata retained 6 names and the 5-edge skeleton.
- Depth and object-pose doctors both reported
  `direct_train_ready=true`, strict validation `scope=all`, and zero errors.
- The copy subset contained 13 hashed artifacts totaling 2,982,167 bytes and
  was created in 0.0146 seconds.
- `dataset.json` SHA-256:
  `cdfcfaf483f2483a56710a6de7c51e85a6925cb73f17df1476d56c3f30e98965`.
- `prepare_summary.json` SHA-256:
  `44bbb9d7df60f1986b952aff06da1e985433d0e98349ad1d7fecc08401d56658`.
- `PROVENANCE.md` SHA-256:
  `99625976ee0f12a75a453d2a03630314e8aea9e738359abe9d6f9954ba311f13`.

A bounded two-sample `ManifestDataset` plus `collate` step completed with:

```text
images       (2, 3, 64, 64)
gt_count     [2, 1]
gt_M         (2, 2, 64, 64)
gt_D_obj     (2, 2, 64, 64)
gt_keypoints (2, 2, 6, 2)
gt_R         (2, 2, 3, 3)
gt_t         (2, 2, 3)
depth        (2, 1, 64, 64)
```

This exercise found and fixed two real loss paths: subset creation omitted
referenced mask/depth sidecars and keypoint metadata, and variable-resolution
auxiliary arrays could not be collated. PNG masks now load as discrete arrays;
masks use nearest-neighbor resizing and depth uses bilinear resizing before
batch collation.

## Adversarial and failure checks

- Empty selected splits fail validation.
- Missing referenced sidecars fail subset creation.
- Existing non-empty outputs require `--overwrite`.
- `--overwrite` refuses symlink, protected, unowned, and source-overlapping
  paths.
- The subset ownership marker itself may not be a symlink.
- Every materialized payload artifact is hashed in `subset.json`; the hash
  manifest does not recursively hash itself.

## Provenance and license boundary

`data/real_multitask_fewshot/prepare_summary.json` records:

- bbox: COCO instances ground truth
- masks: COCO polygon ground truth
- keypoints: bbox-derived anchors
- depth: bbox-scale heuristic
- object 6DoF pose: bbox/depth/intrinsics heuristic

The last three fields are suitable for interface and loader preservation
checks, not scientific accuracy claims. The tracked source annotations do not
contain the license lookup table needed to interpret the selected images'
numeric license IDs. The exact gap is recorded in
`data/real_multitask_fewshot/PROVENANCE.md`; this report therefore does not
claim that redistribution-license review is complete. YOLOZU's Apache-2.0 code
license does not replace dataset-specific terms.

COCO-keypoints and semantic-segmentation conversions pass deterministic
fixture tests, but no bundled licensed real upstream subset is published.
Those rows remain implemented/fixture-tested rather than real-data-qualified.

## Automated reproduction

```text
python -m unittest \
  tests.test_export_dataset_cli \
  tests.test_dataset_auto_detection_extended_cli \
  tests.test_make_subset_dataset \
  rtdetr_pose.tests.test_train_minimal_mask_depth_collate

80 tests passed, including sidecar preservation, external-sidecar collision,
owned-output, source/output overlap, variable-resolution auxiliary collation,
and PNG-mask regressions.

The repository-wide top-level unittest discovery also completed:
`1279 tests passed; 18 skipped`. The skips are environment or optional-backend
lanes declared by those tests, not newly suppressed dataset checks.
```
