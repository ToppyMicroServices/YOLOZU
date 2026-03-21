#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


SUPPORTED_FORMATS: tuple[str, ...] = ("auto", "yolo_pose", "coco", "cvat_xml")
UNSUPPORTED_FORMATS: dict[str, str] = {
    "detectron2_dataset_dict": "Export to COCO keypoints JSON first, then use --format coco.",
    "labelme_keypoints": "Convert to COCO keypoints first, then use --format coco.",
}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prepare a keypoints dataset for YOLOZU with one command. "
            "Supports YOLO Pose layout and COCO keypoints export."
        )
    )
    p.add_argument("--source", required=True, help="Source path (dataset root for YOLO Pose, or COCO root for COCO mode).")
    p.add_argument("--out", required=True, help="Output dataset root (writes dataset.json and/or labels).")
    p.add_argument(
        "--format",
        default="auto",
        help="Input format (supported: auto, yolo_pose, coco; default: auto).",
    )
    p.add_argument(
        "--list-formats",
        action="store_true",
        help="Print supported/unsupported formats and conversion routes, then exit.",
    )

    p.add_argument("--split", default=None, help="Split name for YOLO Pose mode (default: auto-detect).")
    p.add_argument(
        "--num-keypoints",
        type=int,
        default=None,
        help="Optional num_keypoints metadata to write into dataset.json/classes.json (YOLO Pose mode).",
    )
    p.add_argument(
        "--keypoint-names",
        default=None,
        help="Optional comma-separated keypoint names for metadata (YOLO Pose mode).",
    )

    p.add_argument(
        "--annotations",
        default="annotations/person_keypoints_val2017.json",
        help="COCO keypoints annotations path (relative to --source if not absolute).",
    )
    p.add_argument("--images-dir", default="val2017", help="Images dir under COCO root (default: val2017).")
    p.add_argument("--out-split", default="val2017", help="Output split for COCO conversion (default: val2017).")
    p.add_argument("--min-kps", type=int, default=1, help="Min labeled keypoints in COCO mode (default: 1).")
    p.add_argument("--max-images", type=int, default=None, help="Optional max images in COCO mode.")
    p.add_argument("--link-images", action="store_true", help="Symlink images in COCO mode.")
    p.add_argument("--category-id", type=int, default=None, help="COCO mode: target category id.")
    p.add_argument("--category-name", default=None, help="COCO mode: target category name.")
    p.add_argument("--class-id", type=int, default=0, help="COCO mode: YOLO class_id to emit (default: 0).")
    p.add_argument(
        "--cvat-images-dir",
        default=None,
        help="CVAT XML mode: optional images root override (default: infer from --source).",
    )
    return p.parse_args(argv)


def _print_format_support() -> None:
    print("supported formats:")
    for name in SUPPORTED_FORMATS:
        print(f"- {name}")
    print("unsupported direct inputs (convert first):")
    for name, hint in UNSUPPORTED_FORMATS.items():
        print(f"- {name}: {hint}")


def _auto_detect(source: Path) -> str:
    if (source / "images").exists() and (source / "labels").exists():
        return "yolo_pose"
    if (source / "annotations").exists():
        return "coco"
    if source.is_file() and source.suffix.lower() == ".xml":
        return "cvat_xml"
    if source.is_dir() and (source / "annotations.xml").exists():
        return "cvat_xml"
    if source.is_file() and source.suffix.lower() == ".json":
        return "coco"
    raise SystemExit(
        "failed to auto-detect format; pass --format yolo_pose, --format coco or --format cvat_xml explicitly. "
        "run with --list-formats for supported/unsupported formats and conversion hints."
    )


def _normalize_format(raw: str) -> str:
    fmt = str(raw).strip().lower()
    if fmt in SUPPORTED_FORMATS:
        return fmt
    if fmt in UNSUPPORTED_FORMATS:
        raise SystemExit(
            f"unsupported direct format: {fmt}. {UNSUPPORTED_FORMATS[fmt]} "
            "run with --list-formats for full matrix."
        )
    raise SystemExit(
        f"unknown format: {fmt}. supported: {', '.join(SUPPORTED_FORMATS)}. "
        "run with --list-formats for full matrix."
    )


def _pick_split_for_yolo_pose(source: Path, requested: str | None) -> str:
    if isinstance(requested, str) and requested.strip():
        return requested.strip()
    candidates = ["train2017", "val2017", "train", "val", "valid", "validation"]
    for split in candidates:
        if (source / "images" / split).exists() and (source / "labels" / split).exists():
            return split
    try:
        for child in sorted((source / "images").iterdir()):
            if child.is_dir() and (source / "labels" / child.name).exists():
                return child.name
    except OSError as exc:
        logger.debug("failed to scan YOLO Pose image splits under %s: %s", source, exc)
    raise SystemExit("could not detect split in YOLO Pose source; pass --split explicitly")


def _parse_keypoint_names(raw: str | None) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def _write_yolo_pose_wrapper(args: argparse.Namespace, source: Path, out: Path) -> int:
    split = _pick_split_for_yolo_pose(source, args.split)
    images_dir = source / "images" / split
    labels_dir = source / "labels" / split
    if not images_dir.exists() or not labels_dir.exists():
        raise SystemExit(f"YOLO Pose layout not found for split={split}: {images_dir} / {labels_dir}")

    out.mkdir(parents=True, exist_ok=True)
    out_dataset_json = out / "dataset.json"

    payload: dict[str, Any] = {
        "images_dir": str(images_dir.resolve()),
        "labels_dir": str(labels_dir.resolve()),
        "split": split,
        "task": "keypoints",
        "source": str(source.resolve()),
    }

    kp_names = _parse_keypoint_names(args.keypoint_names)
    if kp_names:
        payload["keypoint_names"] = kp_names
        payload["num_keypoints"] = len(kp_names)
    elif isinstance(args.num_keypoints, int) and args.num_keypoints > 0:
        payload["num_keypoints"] = int(args.num_keypoints)

    out_dataset_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if kp_names:
        labels_out = out / "labels" / split
        labels_out.mkdir(parents=True, exist_ok=True)
        classes_json = {
            "keypoint_names": kp_names,
            "num_keypoints": len(kp_names),
        }
        (labels_out / "classes.json").write_text(json.dumps(classes_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(str(out.resolve()))
    return 0


def _run_coco_conversion(args: argparse.Namespace, source: Path, out: Path) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    converter = repo_root / "tools" / "prepare_coco_keypoints_yolozu.py"
    cmd = [
        sys.executable,
        str(converter),
        "--coco-root",
        str(source),
        "--annotations",
        str(args.annotations),
        "--images-dir",
        str(args.images_dir),
        "--out",
        str(out),
        "--out-split",
        str(args.out_split),
        "--min-kps",
        str(int(args.min_kps)),
        "--class-id",
        str(int(args.class_id)),
    ]
    if args.max_images is not None:
        cmd.extend(["--max-images", str(int(args.max_images))])
    if bool(args.link_images):
        cmd.append("--link-images")
    if args.category_id is not None:
        cmd.extend(["--category-id", str(int(args.category_id))])
    if isinstance(args.category_name, str) and args.category_name.strip():
        cmd.extend(["--category-name", str(args.category_name.strip())])

    subprocess.run(cmd, check=True)
    return 0


def _ordered_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = str(item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _parse_points_xy(raw: str) -> tuple[float, float] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    head = text.split(";", 1)[0].strip()
    if not head:
        return None
    parts = [p.strip() for p in head.split(",")]
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except Exception:
        return None


def _resolve_cvat_xml_paths(args: argparse.Namespace, source: Path) -> tuple[Path, Path]:
    if source.is_file() and source.suffix.lower() == ".xml":
        xml_path = source
        base = source.parent
    elif source.is_dir():
        if (source / "annotations.xml").exists():
            xml_path = source / "annotations.xml"
        else:
            candidates = sorted(source.glob("*.xml"))
            if not candidates:
                raise SystemExit(f"no XML file found under source: {source}")
            xml_path = candidates[0]
        base = source
    else:
        raise SystemExit("CVAT XML mode requires --source as .xml file or directory containing XML")

    if isinstance(args.cvat_images_dir, str) and args.cvat_images_dir.strip():
        images_root = Path(args.cvat_images_dir).expanduser()
        if not images_root.is_absolute():
            images_root = (Path.cwd() / images_root).resolve()
    else:
        images_root = base / "images"
        if not images_root.exists():
            images_root = base

    if not xml_path.exists():
        raise SystemExit(f"CVAT XML not found: {xml_path}")
    if not images_root.exists():
        raise SystemExit(f"CVAT images root not found: {images_root}")
    return xml_path, images_root


def _run_cvat_xml_conversion(args: argparse.Namespace, source: Path, out: Path) -> int:
    xml_path, images_root = _resolve_cvat_xml_paths(args, source)
    root = ET.parse(str(xml_path)).getroot()

    meta_label_order: list[str] = []
    for label in root.findall("./meta/task/labels/label"):
        name = str(label.findtext("name") or "").strip()
        if name:
            meta_label_order.append(name)

    image_nodes = list(root.findall(".//image"))
    if not image_nodes:
        raise SystemExit(f"no <image> entries found in CVAT XML: {xml_path}")

    split = str(args.out_split or args.split or "val2017")
    out_labels = out / "labels" / split
    out.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    seen_kp_labels: list[str] = []
    for image in image_nodes:
        for point in image.findall("points"):
            label_name = str(point.get("label") or "").strip()
            if label_name:
                seen_kp_labels.append(label_name)

    keypoint_names = _parse_keypoint_names(args.keypoint_names)
    if not keypoint_names:
        ordered_seen = _ordered_unique(seen_kp_labels)
        if meta_label_order:
            keypoint_names = [name for name in meta_label_order if name in set(ordered_seen)]
            for name in ordered_seen:
                if name not in keypoint_names:
                    keypoint_names.append(name)
        else:
            keypoint_names = ordered_seen
    if not keypoint_names:
        raise SystemExit("CVAT XML mode could not infer keypoint labels; pass --keypoint-names explicitly")

    kp_index = {name: idx for idx, name in enumerate(keypoint_names)}
    min_kps = int(args.min_kps)
    written = 0

    for image in image_nodes:
        image_name = str(image.get("name") or "").strip()
        if not image_name:
            continue
        try:
            width = int(float(str(image.get("width"))))
            height = int(float(str(image.get("height"))))
        except Exception:
            continue
        if width <= 0 or height <= 0:
            continue

        image_path = images_root / image_name
        if not image_path.exists() and (images_root / Path(image_name).name).exists():
            image_path = images_root / Path(image_name).name
        if not image_path.exists():
            continue

        points_by_group: dict[str, dict[str, tuple[float, float, float]]] = {}
        for idx, point in enumerate(image.findall("points")):
            label_name = str(point.get("label") or "").strip()
            if label_name not in kp_index:
                continue
            xy = _parse_points_xy(str(point.get("points") or ""))
            if xy is None:
                continue
            group_key = str(point.get("group_id") or f"__default_{idx}")
            occluded = str(point.get("occluded") or "0").strip() == "1"
            v = 1.0 if occluded else 2.0
            points_by_group.setdefault(group_key, {})[label_name] = (float(xy[0]), float(xy[1]), float(v))

        boxes_by_group: dict[str, tuple[float, float, float, float]] = {}
        for box in image.findall("box"):
            try:
                xtl = float(str(box.get("xtl")))
                ytl = float(str(box.get("ytl")))
                xbr = float(str(box.get("xbr")))
                ybr = float(str(box.get("ybr")))
            except Exception:
                continue
            if xbr <= xtl or ybr <= ytl:
                continue
            group_key = str(box.get("group_id") or "")
            label_name = str(box.get("label") or "").strip()
            if args.category_name and label_name and label_name != str(args.category_name).strip():
                continue
            if group_key:
                boxes_by_group[group_key] = (xtl, ytl, xbr, ybr)

        lines: list[str] = []
        for group_key, kp_map in points_by_group.items():
            labeled = [item for item in kp_map.values() if float(item[2]) > 0.0]
            if len(labeled) < min_kps:
                continue

            if group_key in boxes_by_group:
                xtl, ytl, xbr, ybr = boxes_by_group[group_key]
            else:
                xs = [float(item[0]) for item in labeled]
                ys = [float(item[1]) for item in labeled]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                pad = 2.0
                xtl = max(0.0, xmin - pad)
                ytl = max(0.0, ymin - pad)
                xbr = min(float(width), xmax + pad)
                ybr = min(float(height), ymax + pad)
                if xbr <= xtl:
                    xbr = min(float(width), xtl + 1.0)
                if ybr <= ytl:
                    ybr = min(float(height), ytl + 1.0)

            bw = max(1e-6, xbr - xtl)
            bh = max(1e-6, ybr - ytl)
            cx = (xtl + bw / 2.0) / float(width)
            cy = (ytl + bh / 2.0) / float(height)
            bw_n = bw / float(width)
            bh_n = bh / float(height)

            kps: list[float] = []
            for kp_name in keypoint_names:
                if kp_name in kp_map:
                    x, y, v = kp_map[kp_name]
                    kps.extend([x / float(width), y / float(height), v])
                else:
                    kps.extend([0.0, 0.0, 0.0])

            parts = [f"{int(args.class_id):d}", f"{cx:.6f}", f"{cy:.6f}", f"{bw_n:.6f}", f"{bh_n:.6f}"] + [
                f"{v:.6f}" for v in kps
            ]
            lines.append(" ".join(parts))

        stem = Path(image_name).stem
        (out_labels / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1

    classes_json = {
        "class_names": [str(args.category_name).strip() if args.category_name else "object"],
        "keypoint_names": keypoint_names,
        "num_keypoints": int(len(keypoint_names)),
        "format": "cvat_xml",
        "source_xml": str(xml_path),
    }
    (out_labels / "classes.json").write_text(json.dumps(classes_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_labels / "classes.txt").write_text(f"{classes_json['class_names'][0]}\n", encoding="utf-8")

    dataset_json = {
        "images_dir": str(images_root),
        "labels_dir": str(out_labels),
        "split": split,
        "task": "keypoints",
        "source": str(xml_path),
        "class_id": int(args.class_id),
        "keypoint_names": keypoint_names,
        "num_keypoints": int(len(keypoint_names)),
        "notes": "converted from CVAT XML; bbox from group box when available, otherwise derived from visible keypoints",
        "images_written": int(written),
    }
    (out / "dataset.json").write_text(json.dumps(dataset_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(out.resolve()))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if bool(getattr(args, "list_formats", False)):
        _print_format_support()
        return 0

    source = Path(args.source).expanduser()
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()
    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()

    fmt = _normalize_format(args.format)
    if fmt == "auto":
        fmt = _auto_detect(source)

    if fmt == "yolo_pose":
        return _write_yolo_pose_wrapper(args, source, out)
    if fmt == "coco":
        return _run_coco_conversion(args, source, out)
    if fmt == "cvat_xml":
        return _run_cvat_xml_conversion(args, source, out)

    raise SystemExit(f"unsupported format: {fmt}")


if __name__ == "__main__":
    raise SystemExit(main())
