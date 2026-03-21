import argparse
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.metrics_report import build_report, write_json  # noqa: E402
from yolozu.segmentation_dataset import load_seg_dataset_descriptor, resolve_dataset_path  # noqa: E402
from yolozu.segmentation_eval import compute_confusion_matrix, compute_iou_metrics, load_mask_array  # noqa: E402
from yolozu.segmentation_predictions import build_id_to_mask, load_segmentation_predictions_entries  # noqa: E402

_NUMERIC_ERRORS = (TypeError, ValueError, OverflowError)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate semantic segmentation predictions (mIoU / per-class IoU).")
    p.add_argument("--dataset-json", required=True, help="Path to dataset.json produced by tools/prepare_*_seg.py.")
    p.add_argument("--predictions", required=True, help="Segmentation predictions JSON (id->mask path).")
    p.add_argument("--output", default="reports/seg_eval.json", help="Output JSON report path.")
    p.add_argument("--html", default=None, help="Optional HTML report path.")
    p.add_argument("--overlays-dir", default=None, help="Optional directory to write overlay images for HTML.")
    p.add_argument("--max-overlays", type=int, default=0, help="Max overlays to render (default: 0).")
    p.add_argument("--overlay-max-size", type=int, default=512, help="Max size (max(H,W)) for overlay images (default: 512).")
    p.add_argument("--overlay-alpha", type=float, default=0.5, help="Mask overlay alpha (default: 0.5).")

    p.add_argument("--num-classes", type=int, default=None, help="Override number of classes (default: len(dataset.classes)).")
    p.add_argument("--ignore-index", type=int, default=None, help="Override ignore_index (default: dataset.ignore_index).")
    p.add_argument(
        "--allow-gt-out-of-range",
        action="store_true",
        help="Ignore GT pixels outside [0,num_classes-1] instead of failing.",
    )
    p.add_argument(
        "--miou-ignore-background",
        action="store_true",
        help="If classes[0]=='background', exclude it from mIoU aggregation.",
    )
    p.add_argument("--max-samples", type=int, default=None, help="Optional cap for number of samples to evaluate.")
    p.add_argument("--pred-root", default=None, help="Optional root to resolve relative prediction mask paths.")
    p.add_argument(
        "--allow-rgb-masks",
        action="store_true",
        help="Allow 3-channel masks (uses channel 0; intended for grayscale stored as RGB).",
    )
    p.add_argument(
        "--skip-missing-pred",
        action="store_true",
        help="Skip samples missing in predictions JSON instead of failing.",
    )
    return p.parse_args(argv)


def _resolve_path(value: str, *, base: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return base / p


def _class_colors(num_classes: int):
    import colorsys

    colors: list[tuple[int, int, int]] = []
    for i in range(int(num_classes)):
        h = (float(i) * 0.6180339887498949) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.70, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    # Unknown prediction color.
    colors.append((160, 160, 160))
    return colors


def _overlay_image(image_rgb, mask, *, colors, ignore_index: int | None, alpha: float):
    import numpy as np
    from PIL import Image

    img = image_rgb.convert("RGB")
    img_np = np.asarray(img).astype("float32")

    m = np.asarray(mask).astype("int64", copy=False)
    unknown = int(len(colors) - 1)
    in_range = (m >= 0) & (m < unknown)
    idx = np.where(in_range, m, unknown)

    lut = np.array(colors, dtype="uint8")
    mask_rgb = lut[idx].astype("float32")

    a = float(alpha)
    if ignore_index is not None:
        valid = (m != int(ignore_index)).astype("float32")
        valid = valid[..., None]
        out = img_np * (1.0 - valid * a) + mask_rgb * (valid * a)
    else:
        out = img_np * (1.0 - a) + mask_rgb * a

    out = np.clip(out, 0.0, 255.0).astype("uint8")
    return Image.fromarray(out)


def _mask_image_to_array(mask_img, *, allow_rgb: bool):
    import numpy as np

    arr = np.array(mask_img)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and allow_rgb:
        if arr.shape[2] >= 3:
            if (arr[..., 0] == arr[..., 1]).all() and (arr[..., 0] == arr[..., 2]).all():
                return arr[..., 0]
        return arr[..., 0]
    raise ValueError(f"mask must be 2D class-id image; got shape={getattr(arr, 'shape', None)}")


def _resize_triplet(image_rgb, gt_mask, pred_mask, *, max_size: int):
    from PIL import Image

    img = image_rgb.convert("RGB")
    w, h = img.size
    m = int(max_size)
    if m <= 0:
        return img, gt_mask, pred_mask
    scale = min(1.0, float(m) / float(max(h, w)))
    if scale >= 1.0:
        return img, gt_mask, pred_mask
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img_r = img.resize((new_w, new_h), resample=Image.BILINEAR)
    gt_r = gt_mask.resize((new_w, new_h), resample=Image.NEAREST)
    pred_r = pred_mask.resize((new_w, new_h), resample=Image.NEAREST)
    return img_r, gt_r, pred_r


def _concat_h(images):
    from PIL import Image

    widths = [im.size[0] for im in images]
    heights = [im.size[1] for im in images]
    out = Image.new("RGB", (int(sum(widths)), int(max(heights))))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += int(im.size[0])
    return out


def _write_html(
    *,
    html_path: Path,
    title: str,
    report: dict[str, Any],
    overlays: list[dict[str, Any]] | None,
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = report.get("metrics") if isinstance(report, dict) else None
    meta = report.get("meta") if isinstance(report, dict) else None
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(meta, dict):
        meta = {}

    classes = metrics.get("classes")
    if not isinstance(classes, list):
        classes = []

    # Use relative paths for portability.
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
        f"<p class='meta'>dataset_json: {meta.get('dataset_json')}</p>",
        f"<p class='meta'>predictions: {meta.get('predictions')}</p>",
        f"<p class='meta'>output_json: {rel(str(meta.get('output_json', '')))}</p>",
        f"<p class='meta'>samples_evaluated: {meta.get('samples_evaluated')}</p>",
        f"<p class='meta'>pixels_valid: {(meta.get('stats') or {}).get('pixels_valid')}</p>",
        f"<p class='meta'>mIoU: {metrics.get('miou')}</p>",
        f"<p class='meta'>pixel_accuracy: {metrics.get('pixel_accuracy')}</p>",
        "<h2>Per-class IoU</h2>",
        "<table>",
        "<tr><th>class</th><th>iou</th><th>tp</th><th>fp</th><th>fn</th><th>gt_pixels</th><th>pred_pixels</th><th>union</th></tr>",
    ]

    for c in classes:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        iou = c.get("iou")
        lines.append(
            "<tr>"
            f"<td>{name}</td>"
            f"<td>{iou}</td>"
            f"<td>{c.get('tp')}</td>"
            f"<td>{c.get('fp')}</td>"
            f"<td>{c.get('fn')}</td>"
            f"<td>{c.get('gt_pixels')}</td>"
            f"<td>{c.get('pred_pixels')}</td>"
            f"<td>{c.get('union')}</td>"
            "</tr>"
        )

    lines.extend(["</table>"])

    if overlays:
        lines.append("<h2>Overlays (worst samples)</h2>")
        lines.append("<div class='grid'>")
        for it in overlays:
            if not isinstance(it, dict):
                continue
            overlay_path = it.get("overlay")
            if not isinstance(overlay_path, str) or not overlay_path:
                continue
            lines.extend(
                [
                    "<div class='card'>",
                    f"  <img src='{rel(overlay_path)}' />",
                    f"  <div class='meta'>id: {it.get('id')}</div>",
                    f"  <div class='meta'>mismatch_rate: {it.get('mismatch_rate')}</div>",
                    "</div>",
                ]
            )
        lines.append("</div>")

    lines.extend(["</body>", "</html>"])
    html_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    dataset_json_path = Path(args.dataset_json)
    if not dataset_json_path.is_absolute():
        dataset_json_path = repo_root / dataset_json_path
    dataset_root = dataset_json_path.parent

    pred_json_path = Path(args.predictions)
    if not pred_json_path.is_absolute():
        pred_json_path = repo_root / pred_json_path
    pred_root = Path(args.pred_root) if args.pred_root else pred_json_path.parent
    if not pred_root.is_absolute():
        pred_root = repo_root / pred_root

    desc = load_seg_dataset_descriptor(dataset_json_path)
    ignore_index = int(args.ignore_index) if args.ignore_index is not None else int(desc.ignore_index)

    class_names = desc.classes
    if args.num_classes is not None:
        num_classes = int(args.num_classes)
        if class_names is not None and len(class_names) != num_classes:
            # Keep names we have; fall back to string ids for any missing.
            pass
    else:
        if class_names is None:
            raise SystemExit("dataset.classes is null; provide --num-classes")
        num_classes = int(len(class_names))

    entries, pred_meta = load_segmentation_predictions_entries(pred_json_path)
    id_to_mask = build_id_to_mask(entries)

    import numpy as np

    global_conf = np.zeros((int(num_classes), int(num_classes) + 1), dtype="int64")
    stats_sum = {
        "pixels_total": 0,
        "pixels_ignored": 0,
        "pixels_gt_out_of_range": 0,
        "pixels_pred_out_of_range": 0,
        "pixels_valid": 0,
        "pixels_mismatched": 0,
    }

    samples_total = int(len(desc.samples))
    samples_evaluated = 0
    missing_gt = 0
    missing_pred = 0

    per_sample: list[dict[str, Any]] = []
    for idx, s in enumerate(desc.samples):
        if args.max_samples is not None and idx >= int(args.max_samples):
            break
        if s.mask is None:
            missing_gt += 1
            continue

        try:
            pred_mask_rel = id_to_mask.get(s.sample_id)
        except AttributeError:
            pred_mask_rel = None
        if not pred_mask_rel:
            missing_pred += 1
            if args.skip_missing_pred:
                continue
            raise SystemExit(f"missing prediction for sample id: {s.sample_id}")

        gt_mask_path = resolve_dataset_path(s.mask, dataset_root=dataset_root, path_type=desc.path_type)
        pred_mask_path = _resolve_path(str(pred_mask_rel), base=pred_root)

        gt_arr = load_mask_array(gt_mask_path, allow_rgb=bool(args.allow_rgb_masks))
        pred_arr = load_mask_array(pred_mask_path, allow_rgb=bool(args.allow_rgb_masks))

        conf, st = compute_confusion_matrix(
            gt_arr,
            pred_arr,
            num_classes=int(num_classes),
            ignore_index=int(ignore_index),
            allow_gt_out_of_range=bool(args.allow_gt_out_of_range),
        )
        global_conf += conf

        for k in stats_sum.keys():
            stats_sum[k] += int(getattr(st, k))

        mismatch_rate = float(st.pixels_mismatched) / float(st.pixels_valid) if st.pixels_valid > 0 else 0.0
        per_sample.append(
            {
                "id": s.sample_id,
                "image": s.image,
                "gt_mask": str(gt_mask_path),
                "pred_mask": str(pred_mask_path),
                "valid_pixels": int(st.pixels_valid),
                "mismatched_pixels": int(st.pixels_mismatched),
                "mismatch_rate": float(mismatch_rate),
            }
        )
        samples_evaluated += 1

    metrics = compute_iou_metrics(
        global_conf,
        class_names=class_names,
        miou_ignore_background=bool(args.miou_ignore_background),
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    report = build_report(
        losses={},
        metrics=metrics,
        meta={
            "dataset_json": str(dataset_json_path),
            "predictions": str(pred_json_path),
            "output_json": str(output_path),
            "samples_total": int(samples_total),
            "samples_evaluated": int(samples_evaluated),
            "missing_gt": int(missing_gt),
            "missing_pred": int(missing_pred),
            "ignore_index": int(ignore_index),
            "num_classes": int(num_classes),
            "stats": dict(stats_sum),
            "predictions_meta": pred_meta,
            "per_sample_top": sorted(per_sample, key=lambda x: float(x.get("mismatch_rate", 0.0)), reverse=True)[:100],
        },
    )
    write_json(output_path, report)
    print(output_path)

    overlays_index: list[dict[str, Any]] | None = None
    if args.html:
        overlays_index = []
        if args.overlays_dir and int(args.max_overlays) > 0:
            try:
                from PIL import Image
            except ImportError as exc:  # pragma: no cover
                raise SystemExit(f"Pillow required for overlays: {exc}") from exc

            overlays_dir = Path(args.overlays_dir)
            if not overlays_dir.is_absolute():
                overlays_dir = repo_root / overlays_dir
            overlays_dir.mkdir(parents=True, exist_ok=True)

            colors = _class_colors(int(num_classes))
            # Render worst samples by mismatch rate.
            worst = sorted(per_sample, key=lambda x: float(x.get("mismatch_rate", 0.0)), reverse=True)[: int(args.max_overlays)]
            for item in worst:
                try:
                    sample_id = str(item["id"])
                    img_path = resolve_dataset_path(str(item["image"]), dataset_root=dataset_root, path_type=desc.path_type)
                    gt_path = Path(str(item["gt_mask"]))
                    pred_path = Path(str(item["pred_mask"]))
                    img = Image.open(img_path).convert("RGB")
                    gt_img = Image.open(gt_path)
                    pred_img = Image.open(pred_path)
                    img_r, gt_r, pred_r = _resize_triplet(img, gt_img, pred_img, max_size=int(args.overlay_max_size))
                except (KeyError, OSError, ValueError):
                    continue

                try:
                    gt_r_arr = _mask_image_to_array(gt_r, allow_rgb=bool(args.allow_rgb_masks))
                    pred_r_arr = _mask_image_to_array(pred_r, allow_rgb=bool(args.allow_rgb_masks))
                except (TypeError, ValueError):
                    continue

                try:
                    ov_gt = _overlay_image(img_r, gt_r_arr, colors=colors, ignore_index=int(ignore_index), alpha=float(args.overlay_alpha))
                    ov_pred = _overlay_image(img_r, pred_r_arr, colors=colors, ignore_index=None, alpha=float(args.overlay_alpha))
                    # Error overlay: red where mismatched (valid gt only).
                    import numpy as np

                    err = (np.asarray(gt_r_arr).astype("int64", copy=False) != np.asarray(pred_r_arr).astype("int64", copy=False)) & (
                        np.asarray(gt_r_arr).astype("int64", copy=False) != int(ignore_index)
                    )
                    err_mask = np.where(err, 1, int(ignore_index)).astype("int64")
                    err_colors = [(0, 0, 0), (255, 0, 0)]
                    ov_err = _overlay_image(img_r, err_mask, colors=err_colors, ignore_index=int(ignore_index), alpha=0.7)

                    combined = _concat_h([img_r, ov_gt, ov_pred, ov_err])
                    out_path = overlays_dir / f"{sample_id}.png"
                    combined.save(out_path)
                    overlays_index.append(
                        {
                            "id": sample_id,
                            "overlay": str(out_path),
                            "mismatch_rate": float(item.get("mismatch_rate", 0.0)),
                        }
                    )
                except (OSError, TypeError, ValueError, KeyError):
                    continue

        html_path = Path(args.html)
        if not html_path.is_absolute():
            html_path = repo_root / html_path
        _write_html(
            html_path=html_path,
            title="YOLOZU segmentation eval report",
            report=report,
            overlays=overlays_index if overlays_index else None,
        )
        print(html_path)


if __name__ == "__main__":
    main()
