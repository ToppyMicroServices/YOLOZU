import argparse
import json
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.coco_convert import convert_coco_instances_to_yolo_labels  # noqa: E402
from yolozu.core.diagnostics import format_cli_error  # noqa: E402

_NUMERIC_ERRORS = (TypeError, ValueError, OverflowError)


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco-root",
        required=True,
        help="COCO root containing images/ (train2017/val2017) and annotations/*.json",
    )
    parser.add_argument("--split", choices=("train2017", "val2017"), default="val2017")
    parser.add_argument(
        "--instances-json",
        default=None,
        help="Override path to instances_*.json (default uses coco-root/annotations/instances_{split}.json).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output root for YOLO-format dataset (will create images/<split>/, labels/<split>/, masks/<split>/).",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into out/images/<split>/ (default: do not copy; expects you to point eval to original images).",
    )
    parser.add_argument("--include-crowd", action="store_true", help="Include iscrowd annotations (default: skip).")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images to process.")
    parser.add_argument(
        "--max-instances-per-image",
        type=int,
        default=None,
        help="Optional cap for number of instances exported per image.",
    )
    return parser.parse_args(argv)


def _try_import_deps():  # pragma: no cover
    try:
        import numpy as np
        from PIL import Image
        from pycocotools.coco import COCO  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "prepare_coco_instance_seg requires numpy, Pillow, and pycocotools.\n"
            "Install: pip install numpy Pillow pycocotools"
        ) from exc
    return np, Image, COCO


def _to_int_key_map(mapping):
    out = {}
    if not isinstance(mapping, dict):
        return out
    for k, v in mapping.items():
        try:
            out[int(k)] = int(v)
        except _NUMERIC_ERRORS:
            continue
    return out


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    coco_root = Path(args.coco_root)
    split = args.split
    images_src = coco_root / "images" / split
    if not images_src.exists():
        # Common COCO layout: coco_root/<split>/*.jpg
        fallback = coco_root / split
        if fallback.exists():
            images_src = fallback
        else:
            raise SystemExit(f"images not found for split={split} under {coco_root}")

    if args.instances_json:
        instances_path = Path(args.instances_json)
    else:
        instances_path = coco_root / "annotations" / f"instances_{split}.json"
    if not instances_path.exists():
        raise SystemExit(f"instances JSON not found: {instances_path}")

    out_root = Path(args.out)
    images_out = out_root / "images" / split
    labels_out = out_root / "labels" / split
    masks_out = out_root / "masks" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    instance_images = instances.get("images") or []
    if not isinstance(instance_images, list) or not instance_images:
        raise SystemExit(
            format_cli_error(
                code="E_COCO_IMAGES_EMPTY",
                message=f"instances JSON has no images entries: {instances_path}",
                hint="check source annotations or split selection",
            )
        )
    convert_coco_instances_to_yolo_labels(
        instances_json=instances,
        images_dir=images_src,
        labels_dir=labels_out,
        include_crowd=bool(args.include_crowd),
    )

    if args.copy_images:
        # Copy only images referenced in COCO JSON.
        for img in instance_images:
            file_name = str(img.get("file_name") or "")
            if not file_name:
                continue
            src = images_src / file_name
            dst = images_out / Path(file_name).name
            if dst.exists():
                continue
            if src.exists():
                shutil.copy2(src, dst)

    classes_path = labels_out / "classes.json"
    class_map = {}
    if classes_path.exists():
        try:
            class_map = json.loads(classes_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            class_map = {}
    cat_to_cls = _to_int_key_map((class_map or {}).get("category_id_to_class_id"))

    np, Image, COCO = _try_import_deps()
    coco = COCO(str(instances_path))

    images = list(instance_images)
    if args.max_images is not None:
        images = images[: int(args.max_images)]

    for img in images:
        try:
            img_id = int(img.get("id"))
        except _NUMERIC_ERRORS:
            continue
        file_name = str(img.get("file_name") or "")
        if not file_name:
            continue
        stem = Path(file_name).stem

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        anns = sorted([a for a in anns if isinstance(a, dict)], key=lambda a: int(a.get("id", 0)))

        mask_paths: list[str] = []
        mask_classes: list[int] = []
        for ann in anns:
            if not bool(args.include_crowd) and int(ann.get("iscrowd", 0) or 0) == 1:
                continue
            if not ann.get("segmentation"):
                continue
            try:
                cat_id = int(ann.get("category_id", 0))
            except _NUMERIC_ERRORS:
                continue
            class_id = cat_to_cls.get(cat_id)
            if class_id is None:
                continue
            try:
                ann_id = int(ann.get("id", 0))
            except _NUMERIC_ERRORS:
                ann_id = 0

            try:
                m = coco.annToMask(ann)
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                continue
            try:
                arr = (np.asarray(m).astype("uint8") * 255)
            except (AttributeError, TypeError, ValueError):
                continue
            if arr.ndim != 2:
                continue

            out_name = f"{stem}_{ann_id}.png"
            (masks_out / out_name).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr, mode="L").save(masks_out / out_name)

            rel = Path("masks") / split / out_name
            mask_paths.append(str(rel))
            mask_classes.append(int(class_id))

            if args.max_instances_per_image is not None and len(mask_paths) >= int(args.max_instances_per_image):
                break

        if mask_paths:
            (labels_out / f"{stem}.json").write_text(
                json.dumps({"mask_path": mask_paths, "mask_classes": mask_classes}, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    # Write a minimal dataset descriptor for convenience.
    (out_root / "dataset.json").write_text(
        json.dumps(
            {
                "split": split,
                "images_dir": str(images_out if args.copy_images else images_src),
                "labels_dir": str(labels_out),
                "masks_dir": str(masks_out),
                "instances_json": str(instances_path),
                "copied_images": bool(args.copy_images),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(out_root)


if __name__ == "__main__":
    main()
