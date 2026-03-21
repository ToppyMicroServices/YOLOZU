import argparse
import json
import os
from pathlib import Path

_NUMERIC_ERRORS = (TypeError, ValueError, OverflowError)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert COCO keypoints annotations to YOLOZU YOLO-format labels.")
    p.add_argument("--coco-root", required=True, help="COCO root containing val2017/ and annotations/.")
    p.add_argument(
        "--annotations",
        default="annotations/person_keypoints_val2017.json",
        help="Path to person_keypoints_*.json (relative to --coco-root).",
    )
    p.add_argument("--images-dir", default="val2017", help="Images directory under --coco-root (default: val2017).")
    p.add_argument("--out", required=True, help="Output dataset root (writes labels/ + dataset.json).")
    p.add_argument("--out-split", default="val2017", help="Output split name under labels/ (default: val2017).")
    p.add_argument("--min-kps", type=int, default=1, help="Require at least N labeled keypoints (default: 1).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")
    p.add_argument("--link-images", action="store_true", help="Symlink images into out/images/<split>/ (optional).")
    p.add_argument("--category-id", type=int, default=None, help="Target COCO category id (optional).")
    p.add_argument("--category-name", default=None, help="Target COCO category name (optional).")
    p.add_argument("--class-id", type=int, default=0, help="YOLO class_id to write for the target category (default: 0).")
    return p.parse_args(argv)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _bbox_xywh_to_cxcywh_norm(bbox_xywh: list[float], *, width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    cx = (x + w / 2.0) / float(width)
    cy = (y + h / 2.0) / float(height)
    bw = w / float(width)
    bh = h / float(height)
    return float(cx), float(cy), float(bw), float(bh)


def _kps_to_norm_triplets(kps: list[float], *, width: int, height: int) -> list[float]:
    # COCO keypoints: [x1,y1,v1, x2,y2,v2, ...] in pixels.
    out: list[float] = []
    if len(kps) % 3 != 0:
        return out
    for i in range(0, len(kps), 3):
        x = float(kps[i + 0]) / float(width)
        y = float(kps[i + 1]) / float(height)
        v = float(kps[i + 2])
        out.extend([x, y, v])
    return out


def _count_labeled(kps: list[object]) -> int:
    if len(kps) % 3 != 0:
        return 0
    cnt = 0
    for i in range(0, len(kps), 3):
        v = kps[i + 2]
        try:
            if int(v) > 0:
                cnt += 1
        except _NUMERIC_ERRORS:
            continue
    return cnt


def _normalize_keypoint_names(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            names.append(text)
    return names


def _normalize_skeleton(value: object, *, num_keypoints: int) -> list[list[int]]:
    if not isinstance(value, list):
        return []
    out: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for edge in value:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        try:
            a = int(edge[0])
            b = int(edge[1])
        except _NUMERIC_ERRORS:
            continue
        if a <= 0 or b <= 0 or a == b:
            continue
        if a > int(num_keypoints) or b > int(num_keypoints):
            continue
        key = (a, b) if a <= b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        out.append([int(key[0]), int(key[1])])
    return out


def _pick_category(categories: list[dict], *, category_id: int | None, category_name: str | None) -> dict:
    keypoint_categories: list[dict] = []
    for cat in categories:
        if not isinstance(cat, dict):
            continue
        if _normalize_keypoint_names(cat.get("keypoints")):
            keypoint_categories.append(cat)

    if not keypoint_categories:
        raise SystemExit("no category with keypoints found in annotations categories[]")

    if category_id is not None:
        for cat in keypoint_categories:
            try:
                if int(cat.get("id")) == int(category_id):
                    return cat
            except _NUMERIC_ERRORS:
                continue
        raise SystemExit(f"category_id={category_id} not found among keypoint-enabled categories")

    if isinstance(category_name, str) and category_name.strip():
        needle = category_name.strip().lower()
        for cat in keypoint_categories:
            if str(cat.get("name", "")).strip().lower() == needle:
                return cat
        raise SystemExit(f"category_name='{category_name}' not found among keypoint-enabled categories")

    for cat in keypoint_categories:
        if str(cat.get("name", "")).strip().lower() == "person":
            return cat

    if len(keypoint_categories) == 1:
        return keypoint_categories[0]

    names = [str(cat.get("name") or cat.get("id")) for cat in keypoint_categories]
    raise SystemExit(
        "multiple keypoint-enabled categories found; specify --category-id or --category-name. "
        f"Candidates: {', '.join(names)}"
    )


def main(argv: list[str] | None = None) -> int:
    import sys

    args = _parse_args(sys.argv[1:] if argv is None else argv)

    coco_root = Path(str(args.coco_root))
    if not coco_root.is_absolute():
        coco_root = Path.cwd() / coco_root
    ann_path = coco_root / str(args.annotations)
    images_dir = coco_root / str(args.images_dir)
    if not ann_path.exists():
        raise SystemExit(f"annotations not found: {ann_path}")
    if not images_dir.exists():
        raise SystemExit(f"images dir not found: {images_dir}")

    out_root = Path(str(args.out))
    if not out_root.is_absolute():
        out_root = Path.cwd() / out_root

    out_split = str(args.out_split)
    out_labels = out_root / "labels" / out_split
    out_images = out_root / "images" / out_split
    _ensure_dir(out_labels)
    if bool(args.link_images):
        _ensure_dir(out_images)

    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    images = payload.get("images") or []
    annotations = payload.get("annotations") or []
    categories = payload.get("categories") or []

    categories_dicts = [cat for cat in categories if isinstance(cat, dict)]
    target_cat = _pick_category(
        categories_dicts,
        category_id=args.category_id,
        category_name=args.category_name,
    )
    try:
        target_cat_id = int(target_cat.get("id"))
    except _NUMERIC_ERRORS:
        raise SystemExit("selected category has invalid id")
    target_cat_name = str(target_cat.get("name") or target_cat_id)
    keypoint_names = _normalize_keypoint_names(target_cat.get("keypoints"))
    if not keypoint_names:
        raise SystemExit("selected category has no keypoints schema")
    skeleton = _normalize_skeleton(target_cat.get("skeleton") or [], num_keypoints=len(keypoint_names))

    images_by_id: dict[int, dict] = {}
    for im in images:
        if not isinstance(im, dict):
            continue
        try:
            images_by_id[int(im["id"])] = im
        except (KeyError, TypeError, ValueError, OverflowError):
            continue

    ann_by_image: dict[int, list[dict]] = {}
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        try:
            if int(ann.get("category_id")) != int(target_cat_id):
                continue
        except _NUMERIC_ERRORS:
            continue
        try:
            image_id = int(ann.get("image_id"))
        except _NUMERIC_ERRORS:
            continue
        ann_by_image.setdefault(image_id, []).append(ann)

    written = 0
    for image_id, im in sorted(images_by_id.items(), key=lambda kv: kv[0]):
        if args.max_images is not None and written >= int(args.max_images):
            break
        file_name = im.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            continue
        try:
            width = int(im.get("width"))
            height = int(im.get("height"))
        except _NUMERIC_ERRORS:
            continue
        src_img = images_dir / file_name
        if not src_img.exists():
            continue

        anns = ann_by_image.get(int(image_id), [])
        lines: list[str] = []
        for ann in anns:
            bbox = ann.get("bbox")
            kps = ann.get("keypoints")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            if not (isinstance(kps, list) and len(kps) >= 3):
                continue
            try:
                kps_float = [float(v) for v in kps]
                bbox_float = [float(v) for v in bbox]
            except _NUMERIC_ERRORS:
                continue
            if _count_labeled(kps_float) < int(args.min_kps):
                continue
            cx, cy, bw, bh = _bbox_xywh_to_cxcywh_norm(bbox_float, width=width, height=height)
            # YOLOZU keypoints: append x y v triplets (normalized).
            kps_norm = _kps_to_norm_triplets(kps_float, width=width, height=height)
            if not kps_norm:
                continue
            parts = [f"{int(args.class_id):d}", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"] + [f"{v:.6f}" for v in kps_norm]
            lines.append(" ".join(parts))

        stem = Path(file_name).stem
        (out_labels / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        if bool(args.link_images):
            dst = out_images / src_img.name
            if not dst.exists():
                os.symlink(src_img, dst)

        written += 1

    classes_json = {
        "class_names": [target_cat_name],
        "keypoint_names": keypoint_names,
        "num_keypoints": int(len(keypoint_names)),
        "keypoint_category_id": int(target_cat_id),
    }
    if skeleton:
        classes_json["skeleton"] = skeleton
    (out_labels / "classes.json").write_text(json.dumps(classes_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_labels / "classes.txt").write_text(f"{target_cat_name}\n", encoding="utf-8")

    # Prefer dataset.json so we don't need to copy images.
    dataset_json = {
        "images_dir": str(images_dir),
        "labels_dir": str(out_labels),
        "split": out_split,
        "task": "keypoints",
        "source": str(ann_path),
        "category_id": int(target_cat_id),
        "category_name": target_cat_name,
        "class_id": int(args.class_id),
        "keypoint_names": keypoint_names,
        "num_keypoints": int(len(keypoint_names)),
        "notes": "single-category keypoints; class_id is configurable; keypoints appended as x y v triplets (normalized).",
    }
    if skeleton:
        dataset_json["skeleton"] = skeleton
    (out_root / "dataset.json").write_text(json.dumps(dataset_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
