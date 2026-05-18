import argparse
import json
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.coco_convert import convert_coco_instances_to_yolo_labels, resolve_coco_file_path
from yolozu.core.diagnostics import format_cli_error


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
        help="Output root for YOLO-format dataset (will create images/<split>/ and labels/<split>/).",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into out/images/<split>/ (default: do not copy; expects you to point eval to original images).",
    )
    parser.add_argument("--include-crowd", action="store_true", help="Include iscrowd annotations (default: skip).")
    return parser.parse_args(argv)


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
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    instances = json.loads(instances_path.read_text())
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
            src = resolve_coco_file_path(images_src, file_name)
            dst = images_out / Path(file_name).name
            if dst.exists():
                continue
            if src.exists():
                shutil.copy2(src, dst)

    # Write a minimal dataset descriptor for convenience.
    (out_root / "dataset.json").write_text(
        json.dumps(
            {
                "split": split,
                "images_dir": str((images_out if args.copy_images else images_src).resolve()),
                "labels_dir": str(labels_out.resolve()),
                "instances_json": str(instances_path.resolve()),
                "copied_images": bool(args.copy_images),
            },
            indent=2,
            sort_keys=True,
        )
    )

    print(out_root)


if __name__ == "__main__":
    main()
