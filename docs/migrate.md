# Migration helpers (`yolozu migrate`)

YOLOZU aims to accept common dataset layouts “as-is” and convert them into a stable internal contract.
The `yolozu migrate` subcommands generate small **descriptor artifacts** so downstream commands can treat
external datasets uniformly.

## 1) YOLO-style data.yaml / YOLO (detect/pose/segment)

YOLO-style datasets are typically described by a `data.yaml`:

```yaml
path: /path/to/dataset_root
train: images/train
val: images/val
```

### `dataset.json` wrapper

`yolozu migrate dataset --from auto` writes a `dataset.json` that points at the resolved
`images_dir` / `labels_dir` and optionally pins how to parse label txt files.

Example (segment / polygon labels):

```bash
yolozu migrate dataset \
  --from auto \
  --data /path/to/data.yaml \
  --args /path/to/yolo_runtime/runs/.../args.yaml \
  --task segment \
  --output data/yolo_segment_wrapper
```

Outputs:
- `data/yolo_segment_wrapper/dataset.json`

`dataset.json` keys (subset):
- `images_dir`: absolute/relative path to `.../images/<split>`
- `labels_dir`: absolute/relative path to `.../labels/<split>`
- `split`: default split name
- `label_format`: `detect` (default) or `segment` (YOLO polygon labels)

You can then pass the wrapper directory anywhere YOLOZU expects a dataset root:

```bash
yolozu validate dataset data/yolo_segment_wrapper
```

### YOLO polygon labels

When `label_format=segment`, YOLOZU expects label txt lines like:

```
class_id x1 y1 x2 y2 ... xn yn
```

YOLOZU derives a bbox from the polygon and keeps the raw polygon under `label["polygon"]`.

## 2) VOC / Cityscapes / ADE20K (semantic segmentation)

These datasets use class-id PNG masks rather than YOLO bbox labels.

`yolozu migrate seg-dataset` generates a semantic-segmentation dataset descriptor JSON compatible with:
- `yolozu.segmentation_dataset.load_seg_dataset_descriptor()`
- `tools/eval_segmentation.py`

Examples:

```bash
yolozu migrate seg-dataset --from voc --root /path/to/VOCdevkit/VOC2012 --split val --output reports/voc_seg_dataset.json
yolozu migrate seg-dataset --from cityscapes --root /path/to/cityscapes --split val --output reports/cityscapes_seg_dataset.json
yolozu migrate seg-dataset --from ade20k --root /path/to/ade20k --split val --output reports/ade20k_seg_dataset.json
```

The output descriptor contains:
- `samples`: `[{ "id": "...", "image": "...", "mask": "..." }, ...]`
- `classes` (when known)
- `ignore_index`
- `path_type`: `absolute` (default) or `relative`

## 3) COCO JSON datasets (YOLOX / Detectron2 / MMDetection)

Many ecosystems (YOLOX, Detectron2, MMDetection) use COCO-style `instances_*.json` for detection datasets.

YOLOZU consumes **YOLO-style bbox labels** (`labels/<split>/*.txt`), so the simplest migration path is:
COCO JSON → YOLO labels + `dataset.json` wrapper.

### `yolozu migrate dataset --from coco`

```bash
yolozu migrate dataset \
  --from coco \
  --coco-root /path/to/coco_like_root \
  --split val2017 \
  --output data/coco_yolo_like \
  --mode manifest
```

This writes:
- `data/coco_yolo_like/labels/val2017/*.txt`
- `data/coco_yolo_like/dataset.json` (points at `images_dir` and `labels_dir`)

Validate (label-only):

```bash
yolozu validate dataset data/coco_yolo_like --split val2017 --no-check-images
```

If you want a self-contained dataset directory, use `--mode copy` (or `--mode symlink`) to populate
`data/coco_yolo_like/images/val2017/`.

### Legacy helper script (still available)

If you prefer a standalone script, this does the same conversion:

```bash
python3 tools/prepare_coco_yolo.py \
  --coco-root /path/to/coco_like_root \
  --split val2017 \
  --out data/coco_yolo_like
```

This writes:
- `data/coco_yolo_like/labels/val2017/*.txt`
- `data/coco_yolo_like/dataset.json` (points at `images_dir` and `labels_dir`)

Then validate:

```bash
yolozu validate dataset data/coco_yolo_like --split val2017
```

### YOLOZU wrapper -> YOLO / KITTI

Once a dataset is in YOLOZU wrapper form, you can export it into small external training layouts:

```bash
yolozu export-dataset yolo \
  --dataset data/coco_yolo_like \
  --split val2017 \
  --out-dir data/coco_yolo_export \
  --force

yolozu export-dataset kitti \
  --dataset data/coco_yolo_like \
  --split val2017 \
  --out-dir data/coco_kitti_export \
  --force
```

YOLO export writes:
- `data/coco_yolo_export/data.yaml`
- `data/coco_yolo_export/images/<split>/*`
- `data/coco_yolo_export/labels/<split>/*.txt`

KITTI export writes:
- `data/coco_kitti_export/image_2/*`
- `data/coco_kitti_export/label_2/*.txt`
- `data/coco_kitti_export/ImageSets/Main/<split>.txt`

Repo-bundled tiny sample:

```bash
python3 -m yolozu migrate dataset --from coco --coco-root data/conversion_tiny_coco --split val2017 --output reports/conversion_tiny_wrapper --force
python3 -m yolozu export-dataset yolo --dataset reports/conversion_tiny_wrapper --split val2017 --out-dir reports/conversion_tiny_yolo --force
python3 -m yolozu export-dataset kitti --dataset reports/conversion_tiny_wrapper --split val2017 --out-dir reports/conversion_tiny_kitti --force
```

### COCO results → `predictions.json`

Detectron2/MMDetection/YOLOX often export **COCO detection results** (list of `image_id/category_id/bbox/score`).
YOLOZU can convert those into `predictions.json`:

```bash
yolozu migrate predictions \
  --from coco-results \
  --results /path/to/coco_results.json \
  --instances /path/to/instances_val2017.json \
  --output reports/predictions.json \
  --force
```

Then:

```bash
yolozu validate predictions reports/predictions.json --strict
```

### Entry schema migration (`v1` -> `v2`)

Use this when old predictions entries are missing entry-level `schema_version`
or explicitly use `schema_version: 1`.

```bash
yolozu predictions migrate \
  --input reports/predictions_legacy.json \
  --output reports/predictions_v2.json \
  --from v1 \
  --to v2 \
  --force
```

Strict source check (reject mixed/newer inputs):

```bash
yolozu predictions migrate \
  --input reports/predictions_mixed.json \
  --output reports/predictions_v2.json \
  --from v1 \
  --to v2 \
  --strict-source \
  --force
```

## Troubleshooting (common migration failures)

- `yolozu migrate dataset` fails to resolve layouts:
  - Ensure `data.yaml` contains `path:` and `train:`/`val:` point at `images/<split>` directories.
- Polygon labels (`label_format=segment`) errors:
  - Label line must be `class_id` + even-length coords `[x1 y1 x2 y2 ...]` with at least 3 points (>= 6 numbers).
  - Coordinates should be normalized to `[0,1]` (typical YOLO-runtime default).
  - If your labels include `class cx cy w h + poly(...)`, YOLOZU skips the first 4 numbers automatically.
- `yolozu validate dataset` complains about image size:
  - Ensure images are real `.jpg/.png/.bmp/.tif/.tiff/.webp/.gif` files (not empty placeholders), or use `--no-check-images` for label-only validation.
- `yolozu migrate seg-dataset` (VOC/Cityscapes/ADE20K) can’t find files:
  - Double-check `--root` points at the dataset root (see each dataset’s expected layout) and `--split` exists.
