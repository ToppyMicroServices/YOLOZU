#!/usr/bin/env python3
"""Render beginner-friendly TTT manual figures from fixed result artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ASSETS = REPO_ROOT / "docs" / "assets"
MANUAL_FIGURES = REPO_ROOT / "manual" / "figures"


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


FONT_18 = _load_font(18)
FONT_22 = _load_font(22)
FONT_28 = _load_font(28)
FONT_34 = _load_font(34)
FONT_40 = _load_font(40)


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, font, fill=(20, 20, 20)) -> None:
    draw.text(xy, text, font=font, fill=fill)


def _wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word]).strip()
        if len(trial) <= width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font,
    fill=(20, 20, 20),
    width: int = 48,
    line_gap: int = 6,
) -> int:
    x, y = xy
    lines = _wrap_text(text, width)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y = bbox[3] + line_gap
    return y


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], *, title: str) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, outline=(150, 160, 175), width=3, fill=(249, 250, 252))
    draw.rounded_rectangle((x0 + 12, y0 + 12, x1 - 12, y0 + 58), radius=12, fill=(232, 238, 246))
    _text(draw, (x0 + 24, y0 + 20), title, font=FONT_28)


def _scale_values(values: list[float]) -> tuple[list[float], str]:
    if not values:
        return values, ""
    max_value = max(values)
    if max_value < 0.01:
        return [v * 1000.0 for v in values], "×10^-3"
    if max_value < 0.1:
        return [v * 100.0 for v in values], "×10^-2"
    return values, ""


def _draw_bar_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    methods: list[str],
    values: list[float],
    baseline_idx: int = 0,
) -> None:
    x0, y0, x1, y1 = box
    chart_x0 = x0 + 54
    chart_y0 = y0 + 70
    chart_x1 = x1 - 30
    chart_y1 = y1 - 50
    _text(draw, (x0 + 12, y0 + 12), title, font=FONT_22)

    scaled, unit = _scale_values(values)
    max_value = max(scaled) if scaled else 1.0
    max_value = max(max_value, 1e-6)
    axis_color = (120, 130, 145)
    draw.line((chart_x0, chart_y0, chart_x0, chart_y1), fill=axis_color, width=2)
    draw.line((chart_x0, chart_y1, chart_x1, chart_y1), fill=axis_color, width=2)

    ticks = 4
    for idx in range(ticks + 1):
        frac = idx / ticks
        y = int(chart_y1 - frac * (chart_y1 - chart_y0))
        draw.line((chart_x0 - 8, y, chart_x1, y), fill=(228, 232, 238), width=1)
        value = max_value * frac
        label = f"{value:.2f}"
        _text(draw, (x0 + 2, y - 10), label, font=FONT_18, fill=(80, 90, 100))

    count = len(methods)
    bar_gap = 22
    width_total = chart_x1 - chart_x0 - bar_gap * (count - 1)
    bar_w = max(width_total // max(count, 1), 30)

    for idx, (method, value) in enumerate(zip(methods, scaled)):
        bar_x0 = chart_x0 + idx * (bar_w + bar_gap) + 18
        bar_x1 = bar_x0 + bar_w
        frac = value / max_value if max_value else 0.0
        bar_y0 = int(chart_y1 - frac * (chart_y1 - chart_y0))
        fill = (88, 132, 226) if idx != baseline_idx else (150, 150, 150)
        draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, chart_y1), radius=8, fill=fill)
        label = method.upper() if method != "baseline" else "BASE"
        text_bbox = draw.textbbox((0, 0), label, font=FONT_18)
        draw.text((bar_x0 + (bar_w - (text_bbox[2] - text_bbox[0])) // 2, chart_y1 + 10), label, font=FONT_18, fill=(30, 30, 30))
        value_label = f"{value:.2f}"
        vb = draw.textbbox((0, 0), value_label, font=FONT_18)
        draw.text((bar_x0 + (bar_w - (vb[2] - vb[0])) // 2, max(bar_y0 - 26, chart_y0)), value_label, font=FONT_18, fill=(20, 20, 20))

    if unit:
        _text(draw, (chart_x1 - 60, chart_y0 - 32), unit, font=FONT_18, fill=(80, 90, 100))


def _draw_summary_figure(source: dict[str, Any], out_path: Path) -> None:
    img = Image.new("RGB", (1800, 1420), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    _text(draw, (70, 40), "TTT result summary (real shifted probe + fixed-probe MIM)", font=FONT_40)
    subtitle = (
        "Real probe: same few-shot checkpoint + same 10 shifted images (gaussian_noise severity 3, seed 2026). "
        "All five methods below use a fixed before/after protocol. In this runtime, MIM quality numbers come from "
        "the built-in simple_map_proxy fallback because pycocotools is not installed."
    )
    _draw_wrapped(draw, (70, 95), subtitle, font=FONT_22, width=125, line_gap=6)

    left = (70, 190, 900, 720)
    right = (930, 190, 1760, 720)
    _panel(draw, left, title="Real probe metric delta")
    _panel(draw, right, title="Method execution notes")

    methods = ["baseline", "tent", "mim", "cotta", "eata", "sar"]
    map50 = [source["metrics"][m]["map50"] for m in methods]
    map5095 = [source["metrics"][m]["map50_95"] for m in methods]
    _draw_bar_chart(draw, (95, 270, 860, 480), title="mAP50 on the 10-image shifted probe", methods=methods, values=map50)
    _draw_bar_chart(draw, (95, 495, 860, 705), title="mAP50-95 on the same probe", methods=methods, values=map5095)

    x = 955
    y = 280
    bullets = [
        f"Tent / MIM / CoTTA / EATA / SAR all improved from {source['metrics']['baseline']['map50']:.6f} to {source['metrics']['tent']['map50']:.6f} (mAP50).",
        f"The same five methods improved from {source['metrics']['baseline']['map50_95']:.6f} to {source['metrics']['tent']['map50_95']:.6f} (mAP50-95).",
    ]
    for bullet in bullets:
        draw.ellipse((x, y + 8, x + 10, y + 18), fill=(88, 132, 226))
        y = _draw_wrapped(draw, (x + 24, y), bullet, font=FONT_22, width=48, line_gap=8) + 10

    mim = source["metrics"]["mim"]
    card = (955, 600, 1735, 970)
    draw.rounded_rectangle(card, radius=18, outline=(147, 157, 168), width=2, fill=(245, 247, 250))
    _text(draw, (980, 620), "MIM fixed probe note", font=FONT_28)
    mim_lines = [
        f"Fixture: fixed real probe ({mim['images']} images)",
        f"steps_run={mim['steps_run']}, mean_final_loss={mim['mean_final_loss']:.6f}",
        f"changed_images={mim['changed_images']} / {mim['images']}",
        f"map50={mim['map50']:.6f}, map50_95={mim['map50_95']:.6f}",
    ]
    y = 665
    for line in mim_lines:
        y = _draw_wrapped(draw, (980, y), line, font=FONT_22, width=48, line_gap=8) + 4
    _draw_wrapped(
        draw,
        (980, y + 8),
        "Interpretation: MIM now has a real fixed-probe example. The metric backend is simple_map_proxy in this CPU-only runtime.",
        font=FONT_22,
        width=48,
        line_gap=8,
    )

    footer = (
        "Source artifacts: reports/ttt_improvement_probe/ttt_improvement_report.json and "
        "reports/ttt_compare/mim_probe_cpu/mim_before_after_compare.json"
    )
    _draw_wrapped(draw, (70, 1340), footer, font=FONT_18, fill=(95, 105, 115), width=155, line_gap=4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _draw_pipeline_figure(out_path: Path) -> None:
    img = Image.new("RGB", (1800, 620), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    _text(draw, (70, 35), "TTT compare workflow (what to run and what to open first)", font=FONT_40)
    subtitle = "The compare flow is intentionally short: freeze the subset, export once without TTT, export once with one boilerplate, then open the before/after report before touching raw logs."
    _draw_wrapped(draw, (70, 92), subtitle, font=FONT_22, width=140, line_gap=6)

    boxes = [
        ("1. Freeze subset", "Deterministic domain-shift target or fixed subset. Keep the exact same images for every method."),
        ("2. Baseline export", "Run one export without TTT. This writes baseline_predictions.json."),
        ("3. Adapted export", "Run Tent / MIM / CoTTA / EATA / SAR once. This writes <method>_predictions.json and <method>_ttt_log.json."),
        ("4. Compare", "Open <method>_before_after_compare.md first. Use raw logs only for deeper diagnosis."),
    ]
    start_x = 70
    width = 390
    gap = 50
    y0 = 220
    for idx, (title, body) in enumerate(boxes):
        x0 = start_x + idx * (width + gap)
        x1 = x0 + width
        draw.rounded_rectangle((x0, y0, x1, 470), radius=24, outline=(136, 148, 168), width=3, fill=(246, 248, 251))
        _text(draw, (x0 + 24, y0 + 24), title, font=FONT_28)
        _draw_wrapped(draw, (x0 + 24, y0 + 78), body, font=FONT_18, width=24, line_gap=7)
        if idx < len(boxes) - 1:
            cx = x1 + 15
            cy = y0 + 125
            draw.line((cx, cy, cx + 26, cy), fill=(88, 132, 226), width=6)
            draw.polygon([(cx + 26, cy), (cx + 6, cy - 10), (cx + 6, cy + 10)], fill=(88, 132, 226))

    notes = [
        "Why this order works:",
        "• the compare markdown already answers whether adaptation ran, whether predictions changed, and what file to inspect next",
        "• the raw TTT log is for diagnostics, not for first-pass interpretation",
        "• use one boilerplate per method instead of typing long --ttt-* option sets by hand",
    ]
    y = 505
    for line in notes:
        font = FONT_22 if line.startswith("•") else FONT_28
        y = _draw_wrapped(draw, (70, y), line, font=font, width=145, line_gap=5) + 4

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _norm_to_xyxy(det: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    bbox = det["bbox"]
    cx = float(bbox["cx"]) * width
    cy = float(bbox["cy"]) * height
    w = float(bbox["w"]) * width
    h = float(bbox["h"]) * height
    x0 = int(cx - w / 2)
    y0 = int(cy - h / 2)
    x1 = int(cx + w / 2)
    y1 = int(cy + h / 2)
    return x0, y0, x1, y1


def _load_prediction_entry(path: Path, image_name: str) -> dict[str, Any]:
    preds = json.loads(path.read_text(encoding="utf-8"))["predictions"]
    for entry in preds:
        if Path(entry["image"]).name == image_name:
            return entry
    raise KeyError(f"missing prediction entry for {image_name} in {path}")


def _load_gt_detections(label_path: Path) -> list[dict[str, Any]]:
    detections = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        cls, cx, cy, w, h = raw.split()[:5]
        detections.append(
            {
                "class_id": int(cls),
                "score": None,
                "bbox": {"cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h)},
            }
        )
    return detections


def _top_detection(detections: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not detections:
        return None
    return max(detections, key=lambda det: float(det.get("score") or 0.0))


def _draw_boxes(base: Image.Image, detections: list[dict[str, Any]], *, color: tuple[int, int, int], title: str, top_k: int | None = None) -> Image.Image:
    img = base.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    filtered = detections
    if top_k is not None:
        filtered = sorted(detections, key=lambda d: float(d.get("score") or 0.0), reverse=True)[:top_k]
    for det in filtered:
        x0, y0, x1, y1 = _norm_to_xyxy(det, width, height)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
        score = det.get("score")
        label = f"id={det['class_id']}" if score is None else f"id={det['class_id']} {float(score):.3f}"
        label_top = max(0, y0 - 28)
        label_bottom = max(label_top + 24, y0)
        draw.rounded_rectangle((x0, label_top, x0 + max(90, len(label) * 9), label_bottom), radius=8, fill=color)
        draw.text((x0 + 6, label_top + 4), label, font=FONT_18, fill=(255, 255, 255))
    banner_h = 40
    draw.rounded_rectangle((0, 0, width, banner_h), radius=0, fill=(248, 248, 250))
    draw.text((12, 8), title, font=FONT_22, fill=(20, 20, 20))
    return img


def _draw_top_box(draw: ImageDraw.ImageDraw, det: dict[str, Any] | None, *, width: int, height: int, color: tuple[int, int, int]) -> None:
    if det is None:
        return
    x0, y0, x1, y1 = _norm_to_xyxy(det, width, height)
    draw.rectangle((x0, y0, x1, y1), outline=color, width=4)


def _draw_probe_grid(source: dict[str, Any], out_path: Path) -> None:
    probe_root = REPO_ROOT / source["probe_dataset"]
    pred_no_ttt = json.loads((REPO_ROOT / "reports" / "ttt_improvement_probe" / "pred_no_ttt.json").read_text(encoding="utf-8"))["predictions"]
    pred_ttt = json.loads((REPO_ROOT / "reports" / "ttt_improvement_probe" / "pred_ttt.json").read_text(encoding="utf-8"))["predictions"]
    by_no_ttt = {Path(entry["image"]).name: entry for entry in pred_no_ttt}
    by_ttt = {Path(entry["image"]).name: entry for entry in pred_ttt}

    rows: list[dict[str, Any]] = []
    for image_name, entry in by_no_ttt.items():
        label_path = probe_root / "labels" / "val" / image_name.replace(".jpg", ".txt")
        gt = _load_gt_detections(label_path)
        ttt_entry = by_ttt[image_name]
        top_no_ttt = _top_detection(entry.get("detections", []))
        top_ttt = _top_detection(ttt_entry.get("detections", []))
        score_no_ttt = float(top_no_ttt.get("score") or 0.0) if top_no_ttt else 0.0
        score_ttt = float(top_ttt.get("score") or 0.0) if top_ttt else 0.0
        rows.append(
            {
                "image": image_name,
                "gt_count": len(gt),
                "no_ttt_count": len(entry.get("detections", [])),
                "ttt_count": len(ttt_entry.get("detections", [])),
                "top_no_ttt": top_no_ttt,
                "top_ttt": top_ttt,
                "score_no_ttt": score_no_ttt,
                "score_ttt": score_ttt,
                "score_abs_delta": abs(score_ttt - score_no_ttt),
            }
        )
    rows.sort(key=lambda row: (row["score_abs_delta"], row["image"]), reverse=True)

    cols = 5
    tile_w = 330
    tile_h = 320
    thumb_w = 290
    thumb_h = 170
    gutter = 20
    title_h = 190
    footer_h = 85
    canvas = Image.new("RGB", (cols * tile_w + (cols + 1) * gutter, title_h + 2 * tile_h + 3 * gutter + footer_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    _text(draw, (50, 25), "Per-image shifted probe view (same 10-image subset used in the metric chart)", font=FONT_34)
    intro = (
        "Each tile is one image from the fixed shifted probe. Red = highest-score baseline detection. "
        "Blue = highest-score detection after Tent. The text under each tile shows the top-score movement."
    )
    _draw_wrapped(draw, (50, 78), intro, font=FONT_22, width=112, line_gap=6)
    legend_y = 170
    draw.rounded_rectangle((50, legend_y, 250, legend_y + 38), radius=12, fill=(248, 248, 250), outline=(180, 185, 190))
    draw.rectangle((65, legend_y + 11, 85, legend_y + 27), outline=(210, 73, 70), width=3)
    _text(draw, (95, legend_y + 6), "baseline top1", font=FONT_18)
    draw.rectangle((215, legend_y + 11, 235, legend_y + 27), outline=(53, 101, 216), width=3)
    _text(draw, (245, legend_y + 6), "TTT top1", font=FONT_18)

    start_y = title_h + gutter
    for idx, row in enumerate(rows):
        col = idx % cols
        row_idx = idx // cols
        x0 = gutter + col * (tile_w + gutter)
        y0 = start_y + row_idx * (tile_h + gutter)
        x1 = x0 + tile_w
        y1 = y0 + tile_h
        draw.rounded_rectangle((x0, y0, x1, y1), radius=20, outline=(150, 160, 175), width=2, fill=(249, 250, 252))
        image_path = probe_root / "images" / "val" / row["image"]
        thumb = Image.open(image_path).convert("RGB").resize((thumb_w, thumb_h))
        thumb_draw = ImageDraw.Draw(thumb)
        _draw_top_box(thumb_draw, row["top_no_ttt"], width=thumb_w, height=thumb_h, color=(210, 73, 70))
        _draw_top_box(thumb_draw, row["top_ttt"], width=thumb_w, height=thumb_h, color=(53, 101, 216))
        canvas.paste(thumb, (x0 + 20, y0 + 18))
        _text(draw, (x0 + 20, y0 + 196), row["image"], font=FONT_18)
        score_line = f"score {row['score_no_ttt']:.3f} -> {row['score_ttt']:.3f}   |Δ| {row['score_abs_delta']:.3f}"
        det_line = f"dets {row['no_ttt_count']} -> {row['ttt_count']}   GT {row['gt_count']}"
        _text(draw, (x0 + 20, y0 + 228), score_line, font=FONT_18)
        _text(draw, (x0 + 20, y0 + 264), det_line, font=FONT_18)

    footer = (
        f"All ten images are from {source['probe_dataset']}. The figure shows only the top-1 baseline and top-1 TTT boxes per image "
        "to keep the grid readable; use the prediction JSON and compare markdown for full detections."
    )
    _draw_wrapped(draw, (40, canvas.height - 64), footer, font=FONT_18, width=150, line_gap=4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _copy_outputs(paths: list[Path], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, target_dir / path.name)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render beginner-friendly TTT manual figures from fixed result artifacts.")
    p.add_argument("--source-json", default="docs/assets/ttt_method_results_source.json", help="Source JSON containing fixed TTT summary results.")
    p.add_argument("--docs-assets-dir", default="docs/assets", help="Directory to receive generated docs asset PNGs.")
    p.add_argument("--manual-figures-dir", default="manual/figures", help="Directory to receive copied manual figure PNGs.")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    source = json.loads((REPO_ROOT / args.source_json).read_text(encoding="utf-8"))
    docs_dir = REPO_ROOT / args.docs_assets_dir
    manual_dir = REPO_ROOT / args.manual_figures_dir

    summary_png = docs_dir / "ttt_method_results_summary.png"
    pipeline_png = docs_dir / "ttt_compare_pipeline.png"
    example_png = docs_dir / "ttt_probe_example_panel.png"

    _draw_summary_figure(source, summary_png)
    _draw_pipeline_figure(pipeline_png)
    _draw_probe_grid(source, example_png)
    _copy_outputs([summary_png, pipeline_png, example_png], manual_dir)

    print(summary_png)
    print(pipeline_png)
    print(example_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
