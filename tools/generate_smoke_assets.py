from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.datasets.coco import COCO_80_CATEGORY_IDS, COCO_80_CLASSES


def _sanitize_bbox(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float] | None:
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    max_w = min(2.0 * cx, 2.0 * (1.0 - cx))
    max_h = min(2.0 * cy, 2.0 * (1.0 - cy))
    w = min(w, max_w)
    h = min(h, max_h)

    if w <= 0.0 or h <= 0.0:
        return None
    return cx, cy, w, h


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the bundled smoke images, YOLO labels, standard COCO80 classes "
            "mapping, and deterministic predictions from local data/coco128."
        )
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _parse_args(sys.argv[1:] if argv is None else argv)
    source_root = repo_root / "data" / "coco128"
    source_images = source_root / "images" / "train2017"
    source_labels = source_root / "labels" / "train2017"

    smoke_root = repo_root / "data" / "smoke"
    smoke_images = smoke_root / "images" / "val"
    smoke_labels = smoke_root / "labels" / "val"
    smoke_predictions = smoke_root / "predictions"

    for path in (smoke_images, smoke_labels, smoke_predictions):
        path.mkdir(parents=True, exist_ok=True)

    image_map = {path.stem: path for path in source_images.glob("*.jpg")}
    label_map = {path.stem: path for path in source_labels.glob("*.txt")}
    common_stems = sorted(set(image_map) & set(label_map))
    selected_stems = common_stems[:10]
    if len(selected_stems) < 10:
        raise SystemExit(f"not enough coco128 pairs for smoke assets: {len(selected_stems)}")

    predictions: list[dict[str, object]] = []
    for stem in selected_stems:
        image_src = image_map[stem]
        label_src = label_map[stem]
        image_dst = smoke_images / image_src.name
        label_dst = smoke_labels / label_src.name
        shutil.copy2(image_src, image_dst)

        detections: list[dict[str, object]] = []
        label_lines: list[str] = []
        for line in label_src.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            sanitized = _sanitize_bbox(cx, cy, w, h)
            if sanitized is None:
                continue
            cx, cy, w, h = sanitized
            label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            detections.append(
                {
                    "class_id": class_id,
                    "score": 0.9,
                    "bbox": {"cx": cx, "cy": cy, "w": w, "h": h},
                }
            )

        label_dst.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        predictions.append({"image": image_dst.name, "detections": detections})

    classes_payload = {
        "schema_version": 1,
        "names": list(COCO_80_CLASSES),
        "class_to_category_id": {
            str(class_id): category_id for class_id, category_id in enumerate(COCO_80_CATEGORY_IDS)
        },
        "category_id_to_class_id": {
            str(category_id): class_id for class_id, category_id in enumerate(COCO_80_CATEGORY_IDS)
        },
    }
    (smoke_labels / "classes.json").write_text(
        json.dumps(classes_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    predictions_payload = {
        "schema_version": 1,
        "predictions": predictions,
    }
    (smoke_predictions / "predictions_dummy.json").write_text(
        json.dumps(predictions_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"wrote smoke assets to: {smoke_root}")
    print(f"images: {len(selected_stems)}")


if __name__ == "__main__":
    main()
