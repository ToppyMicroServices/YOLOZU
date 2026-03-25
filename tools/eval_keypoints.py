import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.coco_keypoints_eval import (  # noqa: E402
    COCO17_KPT_OKS_SIGMAS,
    build_coco_keypoints_ground_truth,
    evaluate_coco_oks_map,
    predictions_to_coco_keypoints,
)
from yolozu.cli_args import (  # noqa: E402
    require_float_in_range,
    require_non_negative_float,
    require_non_negative_int,
    require_positive_int,
    resolve_input_path,
    resolve_output_path,
)
from yolozu.dataset import build_manifest  # noqa: E402
from yolozu.image_keys import add_image_aliases, lookup_image_alias  # noqa: E402
from yolozu.keypoints import keypoints_to_pixels, normalize_keypoints  # noqa: E402
from yolozu.keypoints_eval import evaluate_keypoints_pck, match_keypoints_detections  # noqa: E402
from yolozu.metrics_report import build_report, write_json  # noqa: E402
from yolozu.predictions import load_predictions_entries, load_predictions_index  # noqa: E402


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate keypoint predictions using PCK (bbox-normalized distance).")
    p.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    p.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p.add_argument("--predictions", required=True, help="Predictions JSON (detections may include keypoints).")
    p.add_argument("--output", default="reports/keypoints_eval.json", help="Output JSON report path.")

    p.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching det→GT (default: 0.5).")
    p.add_argument("--pck-threshold", type=float, default=0.1, help="PCK threshold (normalized by max(bbox_w,bbox_h)) (default: 0.1).")
    p.add_argument("--min-score", type=float, default=0.0, help="Minimum score threshold for predictions (default: 0.0).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images to evaluate.")
    p.add_argument("--per-image-limit", type=int, default=100, help="How many per-image rows to store in report/HTML (default: 100).")

    p.add_argument("--oks", action="store_true", help="Also compute COCO OKS mAP (requires pycocotools).")
    p.add_argument(
        "--oks-sigmas",
        default=None,
        help="OKS sigmas: 'coco17' or comma-separated floats (len=K). Default: coco17 when K==17.",
    )
    p.add_argument("--oks-sigmas-file", default=None, help="JSON file containing list[float] sigmas (len=K).")
    p.add_argument("--oks-max-dets", type=int, default=20, help="COCOeval maxDets for keypoints (default: 20).")

    p.add_argument("--html", default=None, help="Optional HTML report path.")
    p.add_argument("--title", default="YOLOZU keypoints eval report", help="HTML title.")
    p.add_argument("--overlays-dir", default=None, help="Optional directory to write overlay images for HTML.")
    p.add_argument("--max-overlays", type=int, default=0, help="Max overlays to render (default: 0).")
    p.add_argument(
        "--overlay-sort",
        choices=("worst", "best", "first"),
        default="worst",
        help="How to select overlay samples (default: worst).",
    )
    p.add_argument("--overlay-max-size", type=int, default=768, help="Max size (max(H,W)) for overlay images (default: 768).")
    p.add_argument("--kp-radius", type=int, default=3, help="Keypoint marker radius in pixels (default: 3).")
    p.add_argument("--kp-line", action="store_true", help="Draw gt→pred keypoint error lines.")
    return p.parse_args(argv)


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_oks_sigmas(value: str) -> list[float]:
    raw = value.strip().lower()
    if raw in ("coco", "coco17", "coco-17"):
        return list(COCO17_KPT_OKS_SIGMAS)
    parts = [p.strip() for p in value.replace(" ", ",").split(",") if p.strip()]
    return [float(p) for p in parts]


def _resolve_oks_sigmas(
    args: argparse.Namespace,
    *,
    keypoints_count: int,
    cwd: Path,
) -> list[float] | None:
    if not bool(args.oks):
        return None

    if args.oks_sigmas_file:
        path = resolve_input_path(str(args.oks_sigmas_file), cwd=cwd, repo_root=repo_root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("--oks-sigmas-file must contain a JSON list")
        sigmas = [float(v) for v in payload]
    elif args.oks_sigmas:
        sigmas = _parse_oks_sigmas(str(args.oks_sigmas))
    else:
        sigmas = list(COCO17_KPT_OKS_SIGMAS) if int(keypoints_count) == 17 else None

    if sigmas is None:
        raise ValueError(
            f"OKS requires sigmas for K={keypoints_count}. Pass --oks-sigmas or --oks-sigmas-file (or use K=17)."
        )
    if len(sigmas) != int(keypoints_count):
        raise ValueError(f"OKS sigmas length mismatch: expected {keypoints_count}, got {len(sigmas)}")
    return sigmas


def _write_html(
    *,
    html_path: Path,
    title: str,
    report: dict[str, Any],
    overlays: list[dict[str, Any]] | None,
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)

    def rel(p: str) -> str:
        try:
            return str(Path(p).relative_to(html_path.parent))
        except ValueError:
            return str(p)

    meta = report.get("meta") if isinstance(report, dict) else None
    meta = meta if isinstance(meta, dict) else {}
    metrics = report.get("metrics") if isinstance(report, dict) else None
    metrics = metrics if isinstance(metrics, dict) else {}
    warnings = meta.get("warnings")
    warnings = warnings if isinstance(warnings, list) else []
    per_image = meta.get("per_image")
    per_image = per_image if isinstance(per_image, list) else []

    lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '  <meta charset="utf-8" />',
        f"  <title>{title}</title>",
        "  <style>",
        "    body{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding:16px;}",
        "    code,pre{background:#f6f8fa; padding:2px 4px; border-radius:4px;}",
        "    table{border-collapse:collapse; width:100%; margin:12px 0;}",
        "    th,td{border:1px solid #ddd; padding:6px 8px; font-size:13px;}",
        "    th{background:#fafafa; text-align:left;}",
        "    .grid{display:grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap:16px;}",
        "    .card{border:1px solid #ddd; border-radius:8px; padding:8px;}",
        "    img{max-width:100%; height:auto; border-radius:6px;}",
        "    .meta{color:#666; font-size:12px; overflow-wrap:anywhere;}",
        "  </style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        f"<p class='meta'>Generated: {_now_utc()}</p>",
        "<h2>Metrics</h2>",
        "<table>",
        "<tr><th>key</th><th>value</th></tr>",
    ]
    for key in sorted(metrics.keys()):
        val = metrics.get(key)
        lines.append(f"<tr><td>{key}</td><td><code>{json.dumps(val)}</code></td></tr>")
    lines.append("</table>")

    if warnings:
        lines.append("<h2>Warnings</h2>")
        lines.append("<pre>" + "\n".join(str(w) for w in warnings) + "</pre>")

    if per_image:
        lines.append("<h2>Per-image (truncated)</h2>")
        lines.append("<table>")
        lines.append("<tr><th>image</th><th>gt</th><th>pred</th><th>matched</th><th>labeled</th><th>correct</th><th>pck</th></tr>")
        for row in per_image:
            if not isinstance(row, dict):
                continue
            lines.append(
                "<tr>"
                f"<td class='meta'>{row.get('image')}</td>"
                f"<td>{row.get('gt_instances')}</td>"
                f"<td>{row.get('pred_instances')}</td>"
                f"<td>{row.get('matched_instances')}</td>"
                f"<td>{row.get('keypoints_labeled')}</td>"
                f"<td>{row.get('keypoints_correct')}</td>"
                f"<td>{row.get('pck')}</td>"
                "</tr>"
            )
        lines.append("</table>")

    if overlays:
        lines.append("<h2>Overlays</h2>")
        lines.append("<div class='grid'>")
        for it in overlays:
            if not isinstance(it, dict):
                continue
            overlay = it.get("overlay")
            image = it.get("image")
            pck = it.get("pck")
            if not isinstance(overlay, str) or not overlay:
                continue
            lines.extend(
                [
                    "<div class='card'>",
                    f"  <img src='{rel(overlay)}' />",
                    f"  <div class='meta'>image: {image}</div>",
                    f"  <div class='meta'>pck: {pck}</div>",
                    "</div>",
                ]
            )
        lines.append("</div>")

    lines.extend(["</body>", "</html>"])
    html_path.write_text("\n".join(lines), encoding="utf-8")


def _bbox_xyxy_abs(bbox: dict[str, Any], *, width: int, height: int) -> tuple[float, float, float, float] | None:
    try:
        cx = float(bbox.get("cx"))
        cy = float(bbox.get("cy"))
        bw = float(bbox.get("w"))
        bh = float(bbox.get("h"))
    except (TypeError, ValueError):
        return None
    x1 = (cx - bw / 2.0) * float(width)
    y1 = (cy - bh / 2.0) * float(height)
    x2 = (cx + bw / 2.0) * float(width)
    y2 = (cy + bh / 2.0) * float(height)
    return float(x1), float(y1), float(x2), float(y2)


def _render_overlay(
    *,
    image_path: str,
    record: dict[str, Any],
    pred_dets: list[dict[str, Any]],
    iou_threshold: float,
    min_score: float,
    overlay_max_size: int,
    kp_radius: int,
    kp_line: bool,
    out_path: Path,
) -> dict[str, Any] | None:
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Pillow is required for overlays: {exc}") from exc

    labels = record.get("labels") or []
    if not isinstance(labels, list):
        labels = []
    gt_labels = [lab for lab in labels if isinstance(lab, dict) and lab.get("keypoints") is not None]

    matches = match_keypoints_detections(
        gt_labels=gt_labels,
        pred_detections=pred_dets,
        iou_threshold=float(iou_threshold),
        min_score=float(min_score),
    )
    if not matches:
        return None

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    orig_w, orig_h = img.size
    scale = 1.0
    m = int(overlay_max_size)
    if m > 0 and max(orig_w, orig_h) > m:
        scale = float(m) / float(max(orig_w, orig_h))
        img = img.resize((max(1, int(orig_w * scale)), max(1, int(orig_h * scale))), resample=Image.BILINEAR)

    draw = ImageDraw.Draw(img)

    def draw_kps(kps: list[dict[str, Any]], *, color: tuple[int, int, int]):
        pts = keypoints_to_pixels(kps, width=int(orig_w), height=int(orig_h))
        r = max(1, int(kp_radius))
        for x, y, v in pts:
            if v is not None:
                try:
                    if float(v) <= 0.0:
                        continue
                except (TypeError, ValueError):
                    pass
            x = float(x) * float(scale)
            y = float(y) * float(scale)
            draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2, fill=None)

    for m in matches:
        gt_bbox = {"cx": m.gt.get("cx"), "cy": m.gt.get("cy"), "w": m.gt.get("w"), "h": m.gt.get("h")}
        pred_bbox = m.pred.get("bbox")
        if isinstance(pred_bbox, dict):
            pb = _bbox_xyxy_abs(pred_bbox, width=int(orig_w), height=int(orig_h))
            if pb is not None:
                x1, y1, x2, y2 = pb
                draw.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale], outline=(255, 0, 0), width=2)
        gb = _bbox_xyxy_abs(gt_bbox, width=int(orig_w), height=int(orig_h))
        if gb is not None:
            x1, y1, x2, y2 = gb
            draw.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale], outline=(0, 255, 0), width=2)

        gt_kps = normalize_keypoints(m.gt.get("keypoints"), where="gt.keypoints")
        pred_kps = normalize_keypoints(m.pred.get("keypoints"), where="pred.keypoints")
        draw_kps(gt_kps, color=(0, 255, 0))
        draw_kps(pred_kps, color=(255, 0, 0))

        if kp_line:
            gt_pts = keypoints_to_pixels(gt_kps, width=int(orig_w), height=int(orig_h))
            pred_pts = keypoints_to_pixels(pred_kps, width=int(orig_w), height=int(orig_h))
            n = min(len(gt_pts), len(pred_pts))
            for i in range(n):
                gx, gy, gv = gt_pts[i]
                if gv is not None:
                    try:
                        if float(gv) <= 0.0:
                            continue
                    except (TypeError, ValueError):
                        pass
                px, py, _pv = pred_pts[i]
                draw.line(
                    [float(gx) * scale, float(gy) * scale, float(px) * scale, float(py) * scale],
                    fill=(255, 128, 0),
                    width=1,
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return {"image": image_path, "overlay": str(out_path)}


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        max_images = require_non_negative_int(args.max_images, flag_name="--max-images")
        per_image_limit = require_non_negative_int(args.per_image_limit, flag_name="--per-image-limit")
        max_overlays = require_non_negative_int(args.max_overlays, flag_name="--max-overlays")
        overlay_max_size = require_positive_int(args.overlay_max_size, flag_name="--overlay-max-size")
        kp_radius = require_positive_int(args.kp_radius, flag_name="--kp-radius")
        oks_max_dets = require_positive_int(args.oks_max_dets, flag_name="--oks-max-dets")
        iou_threshold = require_float_in_range(args.iou_threshold, flag_name="--iou-threshold", minimum=0.0, maximum=1.0)
        pck_threshold = require_non_negative_float(args.pck_threshold, flag_name="--pck-threshold")
        min_score = require_non_negative_float(args.min_score, flag_name="--min-score")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    cwd = Path.cwd()
    dataset_root = resolve_input_path(args.dataset, cwd=cwd, repo_root=repo_root)
    predictions_path = resolve_input_path(args.predictions, cwd=cwd, repo_root=repo_root)
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    split_effective = manifest["split"]
    if max_images is not None:
        records = records[: int(max_images)]

    pred_index = load_predictions_index(predictions_path)

    # If overlays are requested, compute per-image stats for all images first, then truncate in the final report.
    eval_limit = int(len(records)) if int(max_overlays or 0) > 0 else int(per_image_limit or 0)
    result = evaluate_keypoints_pck(
        records=records,
        predictions_index=pred_index,
        iou_threshold=float(iou_threshold or 0.0),
        pck_threshold=float(pck_threshold or 0.0),
        min_score=float(min_score or 0.0),
        per_image_limit=eval_limit,
    )

    oks_stats: list[Any] | None = None
    oks_sigmas: list[float] | None = None
    if bool(args.oks):
        gt, coco_index = build_coco_keypoints_ground_truth(records, keypoints_format="xy_norm")
        if int(coco_index.keypoints_count) <= 0:
            result.setdefault("warnings", []).append("oks_no_gt_keypoints")
        else:
            oks_sigmas = _resolve_oks_sigmas(args, keypoints_count=int(coco_index.keypoints_count), cwd=cwd)
            image_sizes = {img["id"]: (int(img["width"]), int(img["height"])) for img in gt.get("images", []) or []}
            preds_entries = load_predictions_entries(predictions_path)
            dt = predictions_to_coco_keypoints(
                preds_entries,
                coco_index=coco_index,
                image_sizes=image_sizes,
                keypoints_format="xy_norm",
                min_score=float(min_score or 0.0),
            )
            oks = evaluate_coco_oks_map(
                gt,
                dt,
                sigmas=oks_sigmas,
                max_dets=int(oks_max_dets or 1),
            )
            oks_stats = oks.get("stats") if isinstance(oks, dict) else None
            oks_metrics = oks.get("metrics") if isinstance(oks, dict) else None
            if isinstance(oks_metrics, dict):
                result.setdefault("metrics", {}).update(oks_metrics)

    per_image = result.get("per_image") if isinstance(result, dict) else None
    per_image = per_image if isinstance(per_image, list) else []
    total_per_image = int(len(per_image))

    kept = per_image[: int(per_image_limit or 0)]
    truncated = bool(total_per_image > int(per_image_limit or 0))
    result["per_image"] = kept
    result["per_image_total"] = total_per_image
    result["per_image_kept"] = int(len(kept))
    result["per_image_truncated"] = truncated

    report = build_report(
        metrics=result.get("metrics") if isinstance(result, dict) else {},
        meta={
            "dataset": str(args.dataset),
            "split": str(split_effective),
            "predictions": str(predictions_path),
            "warnings": result.get("warnings", []) if isinstance(result, dict) else [],
            "oks": {
                "enabled": bool(args.oks),
                "keypoints_count": int(coco_index.keypoints_count) if bool(args.oks) else None,
                "sigmas": oks_sigmas,
                "max_dets": int(oks_max_dets or 1),
                "stats": oks_stats,
            },
            "per_keypoint": result.get("per_keypoint"),
            "per_class": result.get("per_class"),
            "per_image": result.get("per_image"),
            "per_image_total": result.get("per_image_total"),
            "per_image_kept": result.get("per_image_kept"),
            "per_image_truncated": result.get("per_image_truncated"),
        },
    )

    out_path = resolve_output_path(args.output, cwd=cwd)
    write_json(out_path, report)
    print(out_path)

    overlays_index: list[dict[str, Any]] | None = None
    if int(max_overlays or 0) > 0 and args.overlays_dir:
        overlays_dir = resolve_output_path(args.overlays_dir, cwd=cwd)

        # Rebuild a candidate list for overlay selection from the full per-image stats.
        # Note: per-image stats in `result` were computed with eval_limit=len(records) above.
        candidates = [
            row
            for row in per_image
            if isinstance(row, dict) and int(row.get("keypoints_labeled") or 0) > 0 and row.get("pck") is not None
        ]
        if args.overlay_sort == "worst":
            candidates.sort(key=lambda r: float(r.get("pck", 1.0)))
        elif args.overlay_sort == "best":
            candidates.sort(key=lambda r: float(r.get("pck", 0.0)), reverse=True)
        else:
            # Keep original order.
            pass

        record_by_image: dict[str, dict[str, Any]] = {}
        for r in records:
            if not isinstance(r, dict) or not r.get("image"):
                continue
            add_image_aliases(record_by_image, str(r.get("image")), r)

        overlays_index = []
        for idx, row in enumerate(candidates[: int(max_overlays or 0)]):
            image_path = str(row.get("image"))
            record = lookup_image_alias(record_by_image, image_path)
            if record is None:
                continue

            pred_dets = lookup_image_alias(pred_index, image_path)
            pred_dets = pred_dets or []
            if not isinstance(pred_dets, list):
                pred_dets = [pred_dets]
            pred_dets = [d for d in pred_dets if isinstance(d, dict) and d.get("keypoints") is not None]

            out_img = overlays_dir / f"{idx:03d}_{Path(image_path).stem}.png"
            overlay = _render_overlay(
                image_path=image_path,
                record=record,
                pred_dets=pred_dets,
                iou_threshold=float(iou_threshold or 0.0),
                min_score=float(min_score or 0.0),
                overlay_max_size=int(overlay_max_size or 1),
                kp_radius=int(kp_radius or 1),
                kp_line=bool(args.kp_line),
                out_path=out_img,
            )
            if overlay:
                overlay["pck"] = row.get("pck")
                overlays_index.append(overlay)

    if args.html:
        html_path = resolve_output_path(args.html, cwd=cwd)
        _write_html(html_path=html_path, title=str(args.title), report=report, overlays=overlays_index)
        print(html_path)


if __name__ == "__main__":
    main()
