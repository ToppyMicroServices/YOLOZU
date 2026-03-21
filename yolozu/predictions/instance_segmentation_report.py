"""End-to-end instance-segmentation evaluation pipeline.

Loads dataset + predictions, runs instance-map matching, and
generates JSON / HTML evaluation reports.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["run_instance_segmentation_eval"]

from yolozu.dataset import build_manifest
from yolozu.instance_segmentation_eval import extract_gt_instances_from_record, load_mask_bool, evaluate_instance_map
from yolozu.image_keys import image_basename, image_key_aliases, lookup_image_alias
from yolozu.instance_segmentation_predictions import iter_instances, load_instance_segmentation_predictions_entries
from yolozu.metrics_report import build_report, write_json

_JSON_LOAD_ERRORS = (OSError, UnicodeDecodeError, json.JSONDecodeError)
_NUMERIC_ERRORS = (TypeError, ValueError, OverflowError)
_MASK_LOAD_ERRORS = (OSError, ValueError, TypeError, RuntimeError)


def _class_colors(num_classes: int) -> list[tuple[int, int, int]]:
    import colorsys

    colors: list[tuple[int, int, int]] = []
    for i in range(int(num_classes)):
        h = (float(i) * 0.6180339887498949) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.70, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def _load_class_id_to_name(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".txt":
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return {int(i): str(name) for i, name in enumerate(lines)}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except _JSON_LOAD_ERRORS:
        return {}

    if isinstance(data, dict):
        names = data.get("class_names") or data.get("names")
        if isinstance(names, list):
            out: dict[int, str] = {}
            for i, name in enumerate(names):
                if name is None:
                    continue
                out[int(i)] = str(name)
            return out
        id_to_name = data.get("id_to_name")
        if isinstance(id_to_name, dict):
            out: dict[int, str] = {}
            for k, v in id_to_name.items():
                try:
                    out[int(k)] = str(v)
                except _NUMERIC_ERRORS:
                    continue
            return out

    return {}


def _overlay_instances(
    image_rgb,
    instances: list[dict[str, Any]],
    *,
    colors: list[tuple[int, int, int]],
    alpha: float,
    allow_rgb_masks: bool,
):
    import numpy as np
    from PIL import Image

    img = image_rgb.convert("RGB")
    base = np.asarray(img).astype("float32")
    out = base.copy()

    a = float(alpha)
    for inst in instances:
        try:
            class_id = int(inst.get("class_id", 0))
        except _NUMERIC_ERRORS:
            continue
        color = colors[int(class_id) % len(colors)] if colors else (255, 0, 0)
        mask_val = inst.get("mask")
        if mask_val is None:
            continue
        try:
            m = load_mask_bool(mask_val, allow_rgb=allow_rgb_masks)
        except _MASK_LOAD_ERRORS:
            continue
        mask = np.asarray(m, dtype=bool)
        if mask.shape[0] != base.shape[0] or mask.shape[1] != base.shape[1]:
            continue
        c = np.array(color, dtype="float32")
        out[mask] = (1.0 - a) * out[mask] + a * c

    out = np.clip(out, 0.0, 255.0).astype("uint8")
    return Image.fromarray(out, mode="RGB")


def _resize_max(image_rgb, *, max_size: int):
    from PIL import Image

    img = image_rgb.convert("RGB")
    w, h = img.size
    m = max(int(w), int(h))
    if m <= int(max_size):
        return img
    scale = float(max_size) / float(m)
    new_w = max(1, int(round(float(w) * scale)))
    new_h = max(1, int(round(float(h) * scale)))
    return img.resize((new_w, new_h), resample=Image.BILINEAR)


def _concat_h(images):
    from PIL import Image

    imgs = [im.convert("RGB") for im in images if im is not None]
    if not imgs:
        raise ValueError("no images to concat")
    heights = [im.size[1] for im in imgs]
    widths = [im.size[0] for im in imgs]
    h = max(heights)
    w = sum(widths)
    out = Image.new("RGB", (w, h), color=(0, 0, 0))
    x = 0
    for im in imgs:
        out.paste(im, (x, 0))
        x += im.size[0]
    return out


def _write_html(*, html_path: Path, title: str, report: dict[str, Any], overlays: list[dict[str, Any]] | None) -> None:
    metrics = report.get("metrics") if isinstance(report, dict) else None
    if not isinstance(metrics, dict):
        metrics = {}
    meta = report.get("meta") if isinstance(report, dict) else None
    if not isinstance(meta, dict):
        meta = {}

    classes = metrics.get("classes")
    if not isinstance(classes, list):
        classes = []
    per_image = meta.get("per_image_top")
    if not isinstance(per_image, list):
        per_image = []

    def rel(p: str) -> str:
        try:
            return str(Path(p).relative_to(html_path.parent))
        except ValueError:
            return str(p)

    lines: list[str] = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '  <meta charset="utf-8" />',
        f"  <title>{title}</title>",
        "  <style>",
        "    body{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding:16px;}",
        "    .meta{color:#666; font-size:12px; overflow-wrap:anywhere;}",
        "    table{border-collapse:collapse; width:100%; margin-top:12px;}",
        "    th,td{border:1px solid #ddd; padding:6px 8px; font-size:12px; text-align:right;}",
        "    th:first-child, td:first-child{text-align:left;}",
        "    .grid{display:grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap:16px; margin-top:16px;}",
        "    .card{border:1px solid #ddd; border-radius:8px; padding:8px;}",
        "    img{max-width:100%; height:auto; border-radius:6px;}",
        "  </style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        f"<p class='meta'>dataset: {meta.get('dataset')}</p>",
        f"<p class='meta'>split: {meta.get('split')}</p>",
        f"<p class='meta'>predictions: {meta.get('predictions')}</p>",
        f"<p class='meta'>output_json: {rel(str(meta.get('output_json', '')))}</p>",
        f"<p class='meta'>images_evaluated: {meta.get('images_evaluated')}</p>",
        f"<p class='meta'>min_score: {meta.get('min_score')}</p>",
        f"<p class='meta'>diag_iou: {meta.get('diag_iou')}</p>",
        f"<p class='meta'>overlay_sort: {meta.get('overlay_sort')}</p>",
        f"<p class='meta'>map50: {metrics.get('map50')}</p>",
        f"<p class='meta'>map50_95: {metrics.get('map50_95')}</p>",
        "<h2>Per-class AP</h2>",
        "<table>",
        "<tr><th>class_id</th><th>name</th><th>ap@0.50</th><th>mean_ap</th></tr>",
    ]

    for c in classes:
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        name = c.get("name")
        ap50 = c.get("ap@0.50")
        mean_ap = c.get("map50_95")
        lines.append(f"<tr><td>{cid}</td><td>{name}</td><td>{ap50}</td><td>{mean_ap}</td></tr>")

    lines.extend(["</table>"])

    if per_image:
        lines.append("<h2>Per-image diagnostics (top)</h2>")
        lines.append("<table>")
        lines.append(
            "<tr>"
            "<th>image</th><th>gt</th><th>pred</th><th>tp</th><th>fp</th><th>fn</th>"
            "<th>precision</th><th>recall</th><th>mean_iou</th><th>error_rate</th>"
            "</tr>"
        )
        for it in per_image:
            if not isinstance(it, dict):
                continue
            lines.append(
                "<tr>"
                f"<td>{it.get('image')}</td>"
                f"<td>{it.get('gt_instances')}</td>"
                f"<td>{it.get('pred_instances')}</td>"
                f"<td>{it.get('tp')}</td>"
                f"<td>{it.get('fp')}</td>"
                f"<td>{it.get('fn')}</td>"
                f"<td>{it.get('precision')}</td>"
                f"<td>{it.get('recall')}</td>"
                f"<td>{it.get('mean_iou')}</td>"
                f"<td>{it.get('error_rate')}</td>"
                "</tr>"
            )
        lines.append("</table>")

    if overlays:
        lines.append("<h2>Overlays</h2>")
        lines.append("<div class='grid'>")
        for it in overlays:
            if not isinstance(it, dict):
                continue
            overlay_path = it.get("overlay")
            if not isinstance(overlay_path, str) or not overlay_path:
                continue
            lines.append("<div class='card'>")
            lines.append(f"<div class='meta'>{it.get('image')}</div>")
            lines.append(f"<img src='{rel(overlay_path)}' />")
            lines.append("</div>")
        lines.append("</div>")

    lines.extend(["</body>", "</html>"])

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_instance_segmentation_eval(
    *,
    dataset_root: str | Path,
    split: str | None,
    predictions: str | Path,
    pred_root: str | Path | None,
    classes: str | Path | None,
    output: str | Path,
    html: str | Path | None,
    title: str,
    overlays_dir: str | Path | None,
    max_overlays: int,
    overlay_sort: str,
    overlay_max_size: int,
    overlay_alpha: float,
    min_score: float,
    max_images: int | None,
    diag_iou: float,
    per_image_limit: int,
    allow_rgb_masks: bool,
) -> tuple[Path, Path | None]:
    cwd = Path.cwd()

    dataset_root = Path(dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = cwd / dataset_root

    pred_json_path = Path(predictions)
    if not pred_json_path.is_absolute():
        pred_json_path = cwd / pred_json_path

    effective_pred_root = Path(pred_root) if pred_root else pred_json_path.parent
    if not effective_pred_root.is_absolute():
        effective_pred_root = cwd / effective_pred_root

    class_names: dict[int, str] = {}
    classes_path: Path | None = None
    if classes:
        classes_path = Path(classes)
        if not classes_path.is_absolute():
            classes_path = cwd / classes_path
        class_names = _load_class_id_to_name(classes_path)

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = cwd / output_path

    manifest = build_manifest(dataset_root, split=split)
    records = list(manifest.get("images") or [])
    if max_images is not None:
        records = records[: int(max_images)]

    entries, pred_meta = load_instance_segmentation_predictions_entries(pred_json_path)

    result = evaluate_instance_map(
        records=records,
        predictions_entries=entries,
        min_score=float(min_score),
        pred_root=effective_pred_root,
        allow_rgb_masks=bool(allow_rgb_masks),
        return_per_image=True,
        diagnostics_iou=float(diag_iou),
    )

    per_class_rows: list[dict[str, Any]] = []
    for cid in sorted(result.per_class.keys()):
        ap_map = result.per_class.get(int(cid), {}) or {}
        vals = [float(v) for v in ap_map.values() if isinstance(v, (int, float))]
        name = class_names.get(int(cid))
        per_class_rows.append(
            {
                "id": int(cid),
                "name": str(name) if name is not None else None,
                **ap_map,
                "map50": float(ap_map.get("ap@0.50", 0.0)) if ap_map else 0.0,
                "map50_95": float(sum(vals) / float(len(vals))) if vals else 0.0,
            }
        )

    report = build_report(
        losses={},
        metrics={
            "map50": float(result.map50),
            "map50_95": float(result.map50_95),
            "classes": per_class_rows,
            "counts": dict(result.counts),
        },
        meta={
            "dataset": str(dataset_root),
            "split": str(manifest.get("split")),
            "predictions": str(pred_json_path),
            "pred_root": str(effective_pred_root),
            "classes": str(classes_path) if classes_path is not None else None,
            "output_json": str(output_path),
            "images_evaluated": int(len(records)),
            "min_score": float(min_score),
            "diag_iou": float(diag_iou),
            "per_image_limit": int(per_image_limit),
            "overlay_sort": str(overlay_sort),
            "allow_rgb_masks": bool(allow_rgb_masks),
            "predictions_meta": pred_meta,
            "warnings": list(result.warnings),
            "per_image_top": [],
        },
    )

    per_image = list(result.per_image or [])
    per_image.sort(
        key=lambda d: (
            -int(d.get("badness", 0)),
            -float(d.get("error_rate", 0.0)),
            float(d.get("mean_iou") if d.get("mean_iou") is not None else 0.0),
        )
    )
    report["meta"]["per_image_top"] = per_image[: max(0, int(per_image_limit))]

    write_json(output_path, report)

    html_path: Path | None = None
    if html is None:
        return output_path, None

    html_path = Path(html)
    if not html_path.is_absolute():
        html_path = cwd / html_path

    overlays_index: list[dict[str, Any]] | None = None
    if overlays_dir and int(max_overlays) > 0:
        import numpy as np
        from PIL import Image

        overlays_index = []
        overlays_dir_p = Path(overlays_dir)
        if not overlays_dir_p.is_absolute():
            overlays_dir_p = cwd / overlays_dir_p
        overlays_dir_p.mkdir(parents=True, exist_ok=True)

        # Build a lightweight prediction index: image key -> instances.
        pred_index: dict[str, list[dict[str, Any]]] = {}
        for inst in iter_instances(entries):
            image = inst.get("image")
            if not isinstance(image, str) or not image:
                continue
            try:
                score = float(inst.get("score", 1.0))
            except _NUMERIC_ERRORS:
                score = 1.0
            if score < float(min_score):
                continue
            mask = inst.get("mask")
            if isinstance(mask, (str, Path)):
                p = Path(mask)
                if not p.is_absolute():
                    p = effective_pred_root / p
                inst = dict(inst)
                inst["mask"] = str(p)
            aliases = image_key_aliases(image)
            if not aliases:
                continue
            bucket = pred_index.setdefault(aliases[0], [])
            bucket.append(inst)
            for alias in aliases[1:]:
                pred_index.setdefault(alias, bucket)

        # Select overlay order based on per-image diagnostics.
        max_class_id = 0
        if result.per_class:
            max_class_id = max(int(k) for k in result.per_class.keys())
        colors = _class_colors(max(1, int(max_class_id) + 1))

        diag_by_path: dict[str, dict[str, Any]] = {}
        for d in (result.per_image or []):
            if isinstance(d, dict) and isinstance(d.get("image_path"), str):
                diag_by_path[str(d["image_path"])] = d

        ordered_records = list(records)
        if overlay_sort in ("worst", "best"):
            reverse = bool(overlay_sort == "worst")

            def _sort_key(rec: dict[str, Any]):
                key = str(rec.get("image", ""))
                diag = diag_by_path.get(key, {})
                badness = int(diag.get("badness", 0))
                error_rate = float(diag.get("error_rate", 0.0))
                mean_iou = float(diag.get("mean_iou") if diag.get("mean_iou") is not None else 0.0)
                if reverse:
                    return (-badness, -error_rate, mean_iou)
                return (badness, error_rate, -mean_iou)

            ordered_records.sort(key=_sort_key)

        for idx, rec in enumerate(ordered_records):
            if len(overlays_index) >= int(max_overlays):
                break
            image_path = rec.get("image")
            if not isinstance(image_path, str) or not image_path:
                continue

            gt_instances, _ = extract_gt_instances_from_record(rec, allow_rgb_masks=bool(allow_rgb_masks))
            pred_instances = lookup_image_alias(pred_index, image_path) or []
            if not gt_instances and not pred_instances:
                continue

            try:
                img = Image.open(image_path).convert("RGB")
            except OSError:
                continue

            # Downscale image for overlays only.
            img_r = _resize_max(img, max_size=int(overlay_max_size))
            scale_x = float(img_r.size[0]) / float(img.size[0]) if img.size[0] else 1.0
            scale_y = float(img_r.size[1]) / float(img.size[1]) if img.size[1] else 1.0

            def _resize_mask_bool(mask_bool):
                m = np.asarray(mask_bool, dtype=bool)
                if m.shape[0] != img.size[1] or m.shape[1] != img.size[0]:
                    return None
                if scale_x == 1.0 and scale_y == 1.0:
                    return m
                im = Image.fromarray((m.astype("uint8") * 255), mode="L")
                im_r = im.resize(img_r.size, resample=Image.NEAREST)
                return (np.asarray(im_r).astype("int64", copy=False) != 0)

            # Inline-resize GT masks (they're already loaded arrays).
            gt_r: list[dict[str, Any]] = []
            for inst in gt_instances:
                m = inst.get("mask")
                if m is None:
                    continue
                m_r = _resize_mask_bool(m)
                if m_r is None:
                    continue
                gt_r.append({"class_id": inst.get("class_id", 0), "mask": m_r})

            pred_r: list[dict[str, Any]] = []
            for inst in pred_instances:
                mask_path = inst.get("mask")
                if mask_path is None:
                    continue
                try:
                    m = load_mask_bool(mask_path, allow_rgb=bool(allow_rgb_masks))
                except _MASK_LOAD_ERRORS:
                    continue
                m_r = _resize_mask_bool(m)
                if m_r is None:
                    continue
                pred_r.append({"class_id": inst.get("class_id", 0), "mask": m_r})

            ov_gt = _overlay_instances(img_r, gt_r, colors=colors, alpha=float(overlay_alpha), allow_rgb_masks=bool(allow_rgb_masks))
            ov_pred = _overlay_instances(img_r, pred_r, colors=colors, alpha=float(overlay_alpha), allow_rgb_masks=bool(allow_rgb_masks))

            combined = _concat_h([img_r, ov_gt, ov_pred])
            stem = Path(image_path).stem
            digest = hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()[:8]
            out_path = overlays_dir_p / f"{idx:06d}_{stem}_{digest}.png"
            try:
                combined.save(out_path)
            except OSError:
                continue

            diag = diag_by_path.get(image_path)
            overlays_index.append(
                {
                    "image": image_basename(image_path) or image_path,
                    "overlay": str(out_path),
                    "badness": int(diag.get("badness", 0)) if isinstance(diag, dict) else None,
                    "tp": int(diag.get("tp", 0)) if isinstance(diag, dict) else None,
                    "fp": int(diag.get("fp", 0)) if isinstance(diag, dict) else None,
                    "fn": int(diag.get("fn", 0)) if isinstance(diag, dict) else None,
                    "mean_iou": float(diag.get("mean_iou")) if isinstance(diag, dict) and diag.get("mean_iou") is not None else None,
                }
            )

    _write_html(html_path=html_path, title=str(title), report=report, overlays=overlays_index if overlays_index else None)
    return output_path, html_path
