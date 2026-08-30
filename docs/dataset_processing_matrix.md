# Dataset processing and round-trip matrix

This page is the source of truth for what YOLOZU preserves across dataset
preflight, wrappers, materialized exports, deterministic subsets, and reference
training intake. A Stable parent command does not make every row Stable.

## Support matrix

| Source | Operation | Result | Preserved fields | Qualification |
|---|---|---|---|---|
| YOLO detection layout or `data.yaml` | `validate dataset`, `doctor train-dataset` | Direct reference-trainer input | image, class, bbox | Qualified on `data/coco128` and `data/smoke`; empty splits fail closed |
| COCO instances root or explicit JSON/images paths | `doctor`, `migrate/import dataset` | Read-only YOLOZU wrapper | class mapping, bbox geometry, image references | Qualified on the tracked two-image COCO fixture; generated conformance coverage includes crowd filtering and nested Unicode paths |
| YOLOZU/YOLO detection wrapper | `export-dataset yolo|coco|kitti` | Materialized or symlink/copy export | image count, class mapping, bbox geometry | Qualified on the tracked two-image COCO fixture; KITTI is export-only |
| COCO keypoints root | `import dataset`, then `export-dataset yolo|coco` | Read-only wrapper, then materialized export | bbox, keypoint coordinates/visibility, keypoint names, skeleton | Implemented and covered by deterministic fixtures; a tracked licensed real-keypoints qualification is not yet published |
| VOC, Cityscapes, ADE20K, or YOLOZU segmentation descriptor | `import dataset`, `export-dataset segmentation` | Read-only descriptor or images/masks export | image/mask pairing, class metadata, ignore index where available | Pixel-valid conformance fixtures cover all three upstream layouts; validation rejects unreadable assets and image/mask size mismatches; no bundled upstream real dataset is redistributed |
| YOLO multi-task layout with sidecar JSON | `make_subset_dataset.py` | Owned symlink/copy subset | bbox, keypoints, masks, depth maps/units, intrinsics, object-pose sidecars, provenance metadata | Qualified on `data/real_multitask_fewshot`; bbox/masks are COCO-derived, while keypoints/depth/object pose are explicitly heuristic |
| Classification folder or OBB labels | `doctor train-dataset` | Recognized external-lane intake | source metadata only | External-only; not a direct RT-DETR reference-trainer input |
| SynthGen shard/stream | SynthGen loaders and validation tools | Experimental intake records | renderer-owned truth fields | See `synthgen_contract.md`; image generation remains outside YOLOZU |
| BOP object pose | Safe BOP download, owned conversion, official-target pose export, and object-pose evaluation | Research/object-pose records | bbox, intrinsics, object rotation/translation, source depth, deterministic metre-scaled CAD subsets, model/archive/result hashes, available symmetry metadata | See [`bop_tless_protocol.md`](bop_tless_protocol.md); official BOP19 protocol execution is tracked, but this is not human 3D skeleton pose and no positive multi-seed efficacy result is claimed |

## Wrapper versus materialized output

- `doctor` is read-only.
- `import dataset` and `migrate dataset` normally create a small
  `dataset.json` wrapper plus normalized labels/metadata as required by the
  selected adapter. Referenced source images are not silently duplicated.
- `export-dataset` creates a target-layout tree. `--image-mode copy`
  materializes image bytes; `--image-mode symlink` keeps links.
- `make_subset_dataset.py` creates an ownership-marked subset. By default it
  links selected assets; `--copy` materializes them. Referenced mask, depth, and
  CAD sidecars and `classes.json` keypoint metadata are retained. `subset.json`
  records the SHA-256 of every materialized payload artifact plus source
  metadata hashes. As the hash manifest, `subset.json` does not hash itself.
- Replacement is allowed only for a non-symlink subset directory bearing
  `.yolozu_subset_output.json`. Protected or unowned paths are refused.

## Concise CLI

```bash
# Native source -> preflight -> wrapper -> materialized target
yolozu doctor train-dataset --from auto --dataset /path/to/source --split val2017 --output -
yolozu import dataset --from auto --dataset /path/to/source --split val2017 \
  --output reports/source_wrapper --force
yolozu export-dataset coco --dataset reports/source_wrapper --split val2017 \
  --out-dir reports/source_coco --image-mode copy --force

# Sidecar-safe deterministic subset
python3 tools/make_subset_dataset.py \
  --dataset data/real_multitask_fewshot --split val --n 2 --strategy first \
  --copy --out reports/real_multitask_subset

# BOP rigid-object pose conversion (Research)
python3 tools/prepare_bop_yolozu.py \
  --bop-root /workspace/bop --split train_primesense \
  --out reports/bop_tless --out-split train2017 \
  --cad-keypoints 4
```

## Python use

```python
from pathlib import Path

from rtdetr_pose.dataset import build_manifest
from yolozu.dataset_validator import validate_dataset_records

manifest = build_manifest(Path("reports/real_multitask_subset"), split="val")
records = manifest["images"]
validation_records = [{**record, "image": record["image_path"]} for record in records]
result = validate_dataset_records(validation_records, strict=True, check_images=True)
result.raise_if_errors()
```

## Agent use

An agent should inspect the manifest before running a write:

```bash
python3 -c 'import json; d=json.load(open("tools/manifest.json")); print(next(t for t in d["tools"] if t["id"]=="make_subset_dataset"))'
```

Use the declared `effects.writes`, require both the explicit source and output,
run `--help`, and verify `subset.json.artifacts.sha256` after completion. Do not
infer that an implemented or external-only row is production-qualified.

## Evidence and boundaries

The adapter conformance lane uses generated, readable image and mask files to
exercise COCO, Pascal VOC, Cityscapes, and ADE20K layout handling without a
large download. It checks paths, class and bbox mapping, crowd/difficult
policies, mask values, and paired dimensions. This is interface and parser
regression evidence, not evidence of model quality or generalization across
real-world dataset distributions. Registry fetch tests also reject malformed or
mismatched declared SHA-256 values, including cached and multi-part assets;
entries with no published checksum remain explicitly unpinned.

The dated reproduction is
[`reports/dataset_roundtrip_2026-07-27.md`](../reports/dataset_roundtrip_2026-07-27.md).
The real multi-task fixture records per-field label provenance in
`data/real_multitask_fewshot/prepare_summary.json`; its explicit license gap is
recorded in `data/real_multitask_fewshot/PROVENANCE.md`. It is suitable for
loader and preservation checks, not accuracy claims for its heuristic
keypoint, depth, or object-pose labels. Dataset licenses remain separate from
YOLOZU's Apache-2.0 code license; callers must retain and review the upstream
dataset terms.
