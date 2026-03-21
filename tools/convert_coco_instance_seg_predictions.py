import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
_NUMERIC_ERRORS = (TypeError, ValueError, OverflowError)


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Convert COCO instance-seg predictions (polygons/RLE) into YOLOZU PNG-mask contract.")
    p.add_argument("--predictions", required=True, help="COCO predictions JSON (list of {image_id,category_id,score,segmentation}).")
    p.add_argument("--instances-json", required=True, help="COCO instances_{split}.json (used for image_id→file_name and sizes).")
    p.add_argument("--output", default="reports/instance_seg_predictions.json", help="Output YOLOZU instance-seg predictions JSON.")
    p.add_argument("--masks-dir", default="reports/instance_seg_masks", help="Directory to write per-instance PNG masks.")
    p.add_argument("--min-score", type=float, default=0.0, help="Drop predictions with score < min_score (default: 0.0).")
    p.add_argument("--max-preds", type=int, default=None, help="Optional cap for number of predictions to convert.")
    return p.parse_args(argv)


def _try_import_deps():  # pragma: no cover
    try:
        import numpy as np
        from PIL import Image
        from pycocotools import mask as mask_utils  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "convert_coco_instance_seg_predictions requires numpy, Pillow, and pycocotools.\n"
            "Install: pip install numpy Pillow pycocotools"
        ) from exc
    return np, Image, mask_utils


def _build_maps(instances: dict[str, Any]):
    images = instances.get("images") or []
    image_id_to_meta: dict[int, dict[str, Any]] = {}
    for im in images:
        if not isinstance(im, dict):
            continue
        try:
            image_id = int(im.get("id"))
        except _NUMERIC_ERRORS:
            continue
        file_name = str(im.get("file_name") or "")
        if not file_name:
            continue
        try:
            w = int(im.get("width") or 0)
            h = int(im.get("height") or 0)
        except _NUMERIC_ERRORS:
            w, h = 0, 0
        image_id_to_meta[image_id] = {"file_name": file_name, "width": w, "height": h}

    categories = instances.get("categories") or []
    cats = []
    for c in categories:
        if not isinstance(c, dict):
            continue
        try:
            cid = int(c.get("id"))
        except _NUMERIC_ERRORS:
            continue
        name = c.get("name")
        cats.append((cid, str(name) if name is not None else str(cid)))
    cats.sort(key=lambda x: x[0])
    cat_to_cls = {int(cid): int(i) for i, (cid, _) in enumerate(cats)}

    return image_id_to_meta, cat_to_cls


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    np, Image, mask_utils = _try_import_deps()

    pred_path = Path(args.predictions)
    inst_path = Path(args.instances_json)
    out_path = Path(args.output)
    masks_dir = Path(args.masks_dir)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    if not masks_dir.is_absolute():
        masks_dir = repo_root / masks_dir

    preds_raw = json.loads(pred_path.read_text(encoding="utf-8"))
    if not isinstance(preds_raw, list):
        raise SystemExit("COCO predictions JSON must be a list")
    instances = json.loads(inst_path.read_text(encoding="utf-8"))
    if not isinstance(instances, dict):
        raise SystemExit("instances-json must be a COCO instances JSON object")

    image_id_to_meta, cat_to_cls = _build_maps(instances)

    masks_dir.mkdir(parents=True, exist_ok=True)
    by_image: dict[str, list[dict[str, Any]]] = {}

    max_preds = int(args.max_preds) if args.max_preds is not None else None
    converted = 0
    for idx, pred in enumerate(preds_raw):
        if max_preds is not None and converted >= max_preds:
            break
        if not isinstance(pred, dict):
            continue
        try:
            score = float(pred.get("score", 0.0))
        except _NUMERIC_ERRORS:
            score = 0.0
        if float(score) < float(args.min_score):
            continue
        try:
            image_id = int(pred.get("image_id"))
        except _NUMERIC_ERRORS:
            continue
        meta = image_id_to_meta.get(int(image_id))
        if meta is None:
            continue
        file_name = str(meta.get("file_name") or "")
        if not file_name:
            continue
        try:
            cat_id = int(pred.get("category_id"))
        except _NUMERIC_ERRORS:
            continue
        class_id = cat_to_cls.get(int(cat_id))
        if class_id is None:
            continue

        seg = pred.get("segmentation")
        if seg is None:
            continue

        h = int(meta.get("height") or 0)
        w = int(meta.get("width") or 0)
        if h <= 0 or w <= 0:
            continue

        try:
            if isinstance(seg, list):
                rles = mask_utils.frPyObjects(seg, h, w)
                rle = mask_utils.merge(rles)
            else:
                rle = seg
            m = mask_utils.decode(rle)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            continue
        try:
            arr = np.asarray(m)
        except (TypeError, ValueError):
            continue
        if arr.ndim == 3:
            # Some decoders return (H,W,1).
            arr = arr[..., 0]
        if arr.ndim != 2:
            continue

        base = Path(file_name).stem
        digest = hashlib.md5(f"{image_id}:{idx}:{cat_id}:{score}".encode("utf-8")).hexdigest()[:10]
        mask_name = f"{base}_{digest}.png"
        mask_path = masks_dir / mask_name
        Image.fromarray((arr.astype("uint8") * 255), mode="L").save(mask_path)

        # YOLOZU expects image key to be the filename (or full path), and mask to be path relative to --pred-root.
        rel_mask = str(Path(masks_dir.name) / mask_name)
        by_image.setdefault(Path(file_name).name, []).append(
            {
                "class_id": int(class_id),
                "score": float(score),
                "mask": rel_mask,
            }
        )
        converted += 1

    entries = [{"image": img, "instances": insts} for img, insts in sorted(by_image.items())]
    payload = {
        "predictions": entries,
        "meta": {
            "source": "coco_instance_seg_predictions",
            "predictions": str(pred_path),
            "instances_json": str(inst_path),
            "masks_dir": str(masks_dir),
            "min_score": float(args.min_score),
            "converted": int(converted),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
