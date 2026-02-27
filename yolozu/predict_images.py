from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from yolozu.export import export_dummy_predictions, write_predictions_json


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _iter_images(input_dir: Path, *, patterns: Iterable[str]) -> list[Path]:
    images: list[Path] = []
    for pattern in patterns:
        images.extend(sorted(input_dir.glob(pattern)))

    seen: set[str] = set()
    out: list[Path] = []
    for image_path in images:
        key = str(image_path.resolve()) if image_path.exists() else str(image_path)
        if key in seen:
            continue
        seen.add(key)
        out.append(image_path)
    return out


def _ensure_wrapper(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
        return payload
    if isinstance(payload, list):
        return {"schema_version": 1, "predictions": payload}
    raise ValueError("unsupported predictions payload shape")


def _rewrite_image_paths(payload: dict[str, Any], mapping: dict[str, str]) -> None:
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        return
    for entry in predictions:
        if not isinstance(entry, dict):
            continue
        image_value = entry.get("image")
        if not isinstance(image_value, str) or not image_value:
            continue
        replacement = mapping.get(image_value)
        if replacement is not None:
            entry["image"] = replacement
            continue
        try:
            replacement = mapping.get(str(Path(image_value).resolve()))
        except Exception:
            replacement = None
        if replacement is not None:
            entry["image"] = replacement


def _render_overlays(
    *,
    payload: dict[str, Any],
    overlays_dir: Path,
    max_images: int | None,
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Pillow is required for overlays: {exc}") from exc

    overlays_dir.mkdir(parents=True, exist_ok=True)
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("invalid predictions payload: missing predictions[]")

    written = 0
    items: list[dict[str, Any]] = []
    for entry in predictions:
        if max_images is not None and int(written) >= int(max_images):
            break
        if not isinstance(entry, dict):
            continue
        image_value = entry.get("image")
        if not isinstance(image_value, str) or not image_value:
            continue

        image_path = Path(image_value)
        if not image_path.exists():
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        detections = entry.get("detections")
        if not isinstance(detections, list):
            detections = []

        draw = ImageDraw.Draw(image)
        width, height = image.size
        for det in detections:
            if not isinstance(det, dict):
                continue
            bbox = det.get("bbox")
            if not isinstance(bbox, dict):
                continue
            try:
                cx = float(bbox.get("cx"))
                cy = float(bbox.get("cy"))
                box_w = float(bbox.get("w"))
                box_h = float(bbox.get("h"))
            except Exception:
                continue

            x1 = (cx - box_w / 2.0) * float(width)
            y1 = (cy - box_h / 2.0) * float(height)
            x2 = (cx + box_w / 2.0) * float(width)
            y2 = (cy + box_h / 2.0) * float(height)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

        out_path = overlays_dir / f"{written:06d}_{image_path.name}"
        image.save(out_path)
        items.append({"image": str(image_path), "overlay": str(out_path), "detections": int(len(detections))})
        written += 1

    return {"overlays_dir": str(overlays_dir), "count": int(written), "items": items}


def _write_html_report(*, html_path: Path, overlays: dict[str, Any], title: str) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    raw_items = overlays.get("items")
    items = raw_items if isinstance(raw_items, list) else []

    def _relative(path_value: str) -> str:
        path_obj = Path(path_value)
        try:
            return str(path_obj.relative_to(html_path.parent))
        except Exception:
            return str(path_obj)

    lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '  <meta charset="utf-8" />',
        f"  <title>{title}</title>",
        "  <style>",
        "    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:16px;}",
        "    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px;}",
        "    .card{border:1px solid #ddd;border-radius:8px;padding:8px;}",
        "    img{max-width:100%;height:auto;border-radius:6px;}",
        "    .meta{color:#666;font-size:12px;overflow-wrap:anywhere;}",
        "  </style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        f"<p class='meta'>Generated: {_now_utc()}</p>",
        "<div class='grid'>",
    ]
    for item in items:
        if not isinstance(item, dict):
            continue
        overlay_path = item.get("overlay")
        if not isinstance(overlay_path, str) or not overlay_path:
            continue
        image_path = item.get("image")
        detections = item.get("detections")
        lines.extend(
            [
                "<div class='card'>",
                f"  <img src='{_relative(overlay_path)}' />",
                f"  <div class='meta'>image: {image_path}</div>",
                f"  <div class='meta'>detections: {detections}</div>",
                "</div>",
            ]
        )
    lines.extend(["</div>", "</body>", "</html>"])
    html_path.write_text("\n".join(lines), encoding="utf-8")


def predict_images(
    *,
    backend: str,
    input_dir: str | Path,
    output: str | Path,
    score: float,
    max_images: int | None,
    force: bool,
    glob_patterns: list[str] | None = None,
    overlays_dir: str | Path | None = None,
    html: str | Path | None = None,
    title: str = "YOLOZU predict-images report",
    onnx: str | Path | None = None,
    input_name: str = "images",
    boxes_output: str = "boxes",
    scores_output: str = "scores",
    class_output: str | None = None,
    combined_output: str | None = None,
    combined_format: str = "xyxy_score_class",
    raw_output: str | None = None,
    raw_format: str = "yolo_84",
    raw_postprocess: str = "native",
    boxes_format: str = "xyxy",
    boxes_scale: str = "norm",
    min_score: float = 0.001,
    topk: int = 300,
    nms_iou: float = 0.7,
    agnostic_nms: bool = False,
    imgsz: int = 640,
    dry_run: bool = False,
    strict: bool = False,
) -> tuple[Path, Path | None]:
    input_dir_path = Path(input_dir).expanduser()
    if not input_dir_path.is_absolute():
        input_dir_path = Path.cwd() / input_dir_path
    if not input_dir_path.is_dir():
        raise FileNotFoundError(f"input dir not found: {input_dir_path}")

    patterns = glob_patterns if glob_patterns else ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.gif"]
    images = _iter_images(input_dir_path, patterns=patterns)
    if max_images is not None:
        images = images[: max(0, int(max_images))]
    if not images:
        raise FileNotFoundError(f"no images matched under: {input_dir_path}")

    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    overlays_path: Path | None = None
    if overlays_dir is not None:
        overlays_path = Path(overlays_dir).expanduser()
        if not overlays_path.is_absolute():
            overlays_path = Path.cwd() / overlays_path

    html_path: Path | None = None
    if html is not None:
        html_path = Path(html).expanduser()
        if not html_path.is_absolute():
            html_path = Path.cwd() / html_path

    with tempfile.TemporaryDirectory(prefix="yolozu_predict_images_") as temp_dir:
        temp_root = Path(temp_dir)
        split = "train2017"
        temp_images = temp_root / "images" / split
        temp_labels = temp_root / "labels" / split
        temp_images.mkdir(parents=True, exist_ok=True)
        temp_labels.mkdir(parents=True, exist_ok=True)

        mapping: dict[str, str] = {}
        for index, src in enumerate(images):
            dst = temp_images / f"{index:06d}_{src.name}"
            try:
                os.symlink(str(src.resolve()), str(dst))
            except Exception:
                shutil.copy2(src, dst)
            mapping[str(dst)] = str(src.resolve())
            mapping[str(dst.resolve())] = str(src.resolve())
            # Keep a valid YOLO layout even when no labels are available.
            (temp_labels / f"{dst.stem}.txt").touch()

        payload: dict[str, Any]
        if backend == "dummy":
            payload, _ = export_dummy_predictions(
                dataset_root=temp_root,
                split=split,
                max_images=max_images,
                score=float(score),
            )
        elif backend == "onnxrt":
            from yolozu.onnxrt_export import export_predictions_onnxrt

            payload = export_predictions_onnxrt(
                dataset_root=temp_root,
                split=split,
                max_images=max_images,
                onnx=str(onnx) if onnx else None,
                input_name=str(input_name),
                boxes_output=str(boxes_output),
                scores_output=str(scores_output),
                class_output=(str(class_output) if class_output else None),
                combined_output=(str(combined_output) if combined_output else None),
                combined_format=str(combined_format),
                raw_output=(str(raw_output) if raw_output else None),
                raw_format=str(raw_format),
                raw_postprocess=str(raw_postprocess),
                boxes_format=str(boxes_format),
                boxes_scale=str(boxes_scale),
                min_score=float(min_score),
                topk=int(topk),
                nms_iou=float(nms_iou),
                agnostic_nms=bool(agnostic_nms),
                imgsz=int(imgsz),
                dry_run=bool(dry_run),
                strict=bool(strict),
            )
        else:
            raise ValueError(f"unsupported backend: {backend}")

    wrapped_payload = _ensure_wrapper(payload)
    _rewrite_image_paths(wrapped_payload, mapping)
    out_path = write_predictions_json(output=output_path, payload=wrapped_payload, force=bool(force))

    if overlays_path is None:
        return out_path, None
    overlay_index = _render_overlays(payload=wrapped_payload, overlays_dir=overlays_path, max_images=max_images)
    if html_path is not None:
        _write_html_report(html_path=html_path, overlays=overlay_index, title=str(title))
    return out_path, html_path
