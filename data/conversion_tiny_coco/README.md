# Tiny COCO Conversion Sample

This fixture is a repo-bundled, two-image COCO-style dataset for dataset conversion smoke checks.

## Layout

- `images/val2017/*.jpg` - copied sample images
- `annotations/instances_val2017.json` - minimal COCO instances annotations with 2 categories

## Example flow

```bash
python3 -m yolozu validate dataset data/conversion_tiny_coco --split val2017 --strict
python3 -m yolozu doctor import --dataset-from auto --dataset data/conversion_tiny_coco --split val2017 --output -
python3 -m yolozu migrate dataset --from auto --dataset data/conversion_tiny_coco --split val2017 --output reports/conversion_tiny_wrapper --force
python3 -m yolozu export-dataset yolo --dataset reports/conversion_tiny_wrapper --split val2017 --out-dir reports/conversion_tiny_yolo --force
python3 -m yolozu export-dataset coco --dataset reports/conversion_tiny_wrapper --split val2017 --out-dir reports/conversion_tiny_coco --force
python3 -m yolozu export-dataset kitti --dataset reports/conversion_tiny_wrapper --split val2017 --out-dir reports/conversion_tiny_kitti --force
```

## Purpose

Use this fixture when you want a tiny, fully local sample for:

- `COCO -> YOLOZU`
- `YOLOZU -> YOLO`
- `YOLOZU -> COCO`
- `YOLOZU -> KITTI`
