"""Demo command handlers for the YOLOZU CLI."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _debug_demo_summary(context: str, exc: Exception) -> None:
    logger.debug("%s: %s", context, exc, exc_info=True)


def _optional_module_status(module: str) -> dict[str, Any]:
    try:
        __import__(module)
        return {"module": module, "available": True, "error": None}
    except Exception as exc:
        return {"module": module, "available": False, "error": f"{type(exc).__name__}: {exc}"}


def _write_demo_overview_report(*, output: str | None) -> Path:
    ts = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    out_path = Path(output) if output else (Path("demo_output") / "overview" / ts / "demo_overview_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dependency_checks = [
        _optional_module_status("torch"),
        _optional_module_status("torchvision"),
        _optional_module_status("cv2"),
        _optional_module_status("numpy"),
        _optional_module_status("transformers"),
    ]
    dep_map = {str(d["module"]): bool(d["available"]) for d in dependency_checks}

    coverage = [
        {
            "capability": "bbox",
            "status": "supported",
            "entrypoints": ["yolozu demo instance-seg", "python3 tools/run_real_multitask_finetune_demo.py"],
            "notes": "Detection outputs are covered in instance-seg and multitask finetune smoke flows.",
        },
        {
            "capability": "segmentation",
            "status": "supported",
            "entrypoints": ["yolozu demo instance-seg", "yolozu demo instance-seg-tta", "python3 tools/eval_instance_segmentation.py"],
            "notes": "Polygon and mask-centric flows are available in demo/eval tooling, including a visible raw-vs-TTA compare demo on real COCO images.",
        },
        {
            "capability": "keypoints",
            "status": "supported" if dep_map.get("torch", False) and dep_map.get("torchvision", False) else "deps_missing",
            "entrypoints": ["yolozu demo keypoints", "python3 tools/eval_keypoints.py"],
            "notes": "Torch/Torchvision are required for keypoint inference demo.",
        },
        {
            "capability": "depth",
            "status": "supported" if dep_map.get("torch", False) else "deps_missing",
            "entrypoints": ["yolozu demo depth", "python3 tools/export_predictions.py --adapter rtdetr_pose --ttt ..."],
            "notes": "Depth demo relies on torch; compare mode adds MiDaS/DPT model checks.",
        },
        {
            "capability": "pose6d",
            "status": "supported" if dep_map.get("cv2", False) else "deps_missing",
            "entrypoints": ["yolozu demo pose", "python3 tools/run_real_multitask_finetune_demo.py"],
            "notes": "6D pose demo supports chessboard/ArUco and optional DenseFusion backend.",
        },
        {
            "capability": "external_finetune_audit",
            "status": "supported",
            "entrypoints": ["python3 tools/run_external_finetune_smoke.py", "python3 tools/audit_backend_support.py"],
            "notes": "Audits YOLOv/MMDetection/Detectron2/RT-DETR entrypoints with reportable interface contract outputs.",
        },
        {
            "capability": "ttt_improvement",
            "status": "supported" if dep_map.get("torch", False) else "deps_missing",
            "entrypoints": ["yolozu demo ttt", "python3 tools/export_predictions.py --adapter rtdetr_pose --ttt ..."],
            "notes": "Runs a deterministic domain shift + few-shot train, then measures simple mAP proxy delta with TTT on/off.",
        },
    ]

    recommended_commands = [
        "yolozu demo instance-seg --run-dir reports/quickstart_instance_seg --progress",
        "yolozu demo overview",
        "yolozu demo",
        "yolozu demo ttt",
        "yolozu demo instance-seg-tta",
        "yolozu demo keypoints",
        "yolozu demo depth",
        "yolozu demo pose",
        "python3 tools/run_real_multitask_finetune_demo.py --dataset-root data/real_multitask_fewshot --out reports/real_multitask_finetune_demo --device cpu --epochs 1 --max-steps 1 --batch-size 2 --image-size 96 --strict-provenance --force",
        "python3 tools/run_external_finetune_smoke.py --dataset-root data/smoke --split train --output reports/external_finetune_smoke.json",
    ]

    warnings: list[str] = []
    for dep in dependency_checks:
        if not bool(dep.get("available")):
            warnings.append(f"optional dependency missing: {dep.get('module')} ({dep.get('error')})")

    payload = {
        "kind": "demo_overview",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "coverage": coverage,
        "dependency_checks": dependency_checks,
        "visible_quickstart": {
            "command": "yolozu demo instance-seg --run-dir reports/quickstart_instance_seg --progress",
            "settings": "configs/quickstart/instance_seg_demo.yaml",
            "report": "reports/quickstart_instance_seg/instance_seg_demo_report.json",
            "png_overlay": "reports/quickstart_instance_seg/overlays/overlay_img_0000.png",
            "notes": "Use this when you want a folder with images, masks, overlays, and a JSON report.",
        },
        "recommended_commands": recommended_commands,
        "docs": [
            "README.md",
            "docs/tools_index.md",
            "docs/external_finetune_smoke.md",
            "docs/training_inference_export.md",
        ],
        "warnings": warnings,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def handle_demo_command(args: argparse.Namespace) -> int:
    def _print_instance_seg_report(*, out_path: Path, label: str | None = None) -> None:
        try:
            payload = json.loads(Path(out_path).read_text(encoding="utf-8"))
            res = payload.get("result", {})
            counts = res.get("counts", {})
            meta = payload.get("meta", {})
            if label:
                print(label)
            print(
                "instance-seg demo: "
                f"mAP50-95={res.get('map50_95'):.3f} mAP50={res.get('map50'):.3f} "
                f"(images={counts.get('images')} gt={counts.get('gt_instances')} pred={counts.get('pred_instances')} classes={counts.get('classes')})"
            )
            if isinstance(meta, dict):
                inf = meta.get("inference")
                if isinstance(inf, dict):
                    used = bool(inf.get("used"))
                    backend = inf.get("backend")
                    weights = inf.get("weights")
                    dev = inf.get("device")
                    mode = inf.get("mode")
                    if used and backend:
                        print(f"inference: {backend} weights={weights} device={dev}")
                    elif mode:
                        print(f"inference: {mode} (used={used})")
            if isinstance(meta, dict) and meta.get("run_dir"):
                print(f"output_dir: {meta.get('run_dir')}")
            artifacts = payload.get("artifacts", {})
            if isinstance(artifacts, dict):
                overlays_dir = artifacts.get("overlays_dir")
                if isinstance(overlays_dir, str) and overlays_dir:
                    try:
                        overlays = sorted(Path(overlays_dir).glob("*.png"))
                    except Exception as exc:
                        _debug_demo_summary("instance-seg overlays directory scan skipped", exc)
                        overlays = []
                    if overlays:
                        print(f"first_overlay: {overlays[0]}")
        except Exception as exc:
            _debug_demo_summary("instance-seg report summary skipped", exc)
            if label:
                print(label)
        print(f"report: {out_path}")
        print(str(out_path))

    if args.demo_command is None:
        from yolozu.demos.instance_seg import run_instance_seg_demo

        suite_id = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
        suite_root = Path("demo_output") / "instance_seg" / f"suite_{suite_id}"
        ok = 0
        overview_path = _write_demo_overview_report(output=None)
        print("== overview ==")
        print(str(overview_path))

        suite_instances = getattr(args, "coco_instances_json", None)
        suite_images = getattr(args, "coco_images_dir", None)
        if suite_instances or suite_images:
            if not (suite_instances and suite_images):
                raise ValueError("demo suite requires both --coco-instances-json and --coco-images-dir")
            has_real_coco_instances = True
        else:
            default_instances = Path("data") / "coco" / "annotations" / "instances_val2017.json"
            default_images = Path("data") / "coco" / "images" / "val2017"
            has_real_coco_instances = default_instances.exists() and default_images.exists()

        if not has_real_coco_instances:
            # 1) Synthetic instance-seg
            syn_run_dir = suite_root / "synthetic"
            syn_run_dir.mkdir(parents=True, exist_ok=True)
            syn_cfg_path = syn_run_dir / "demo_config.json"
            syn_cfg_path.write_text(
                json.dumps(
                    {
                        "kind": "demo_config",
                        "demo": "instance-seg",
                        "background": "synthetic",
                        "seed": 0,
                        "num_images": 4,
                        "image_size": 96,
                        "max_instances": 2,
                    },
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            out_syn = run_instance_seg_demo(
                run_dir=syn_run_dir,
                seed=0,
                num_images=4,
                image_size=96,
                max_instances=2,
                background="synthetic",
            )
            print(f"config: {syn_cfg_path}")
            _print_instance_seg_report(out_path=Path(out_syn), label="== instance-seg (synthetic) ==")
            ok += 1

            # 2) COCO128-backed instance-seg (skip if dataset missing)
            try:
                coco128_run_dir = suite_root / "coco128"
                coco128_run_dir.mkdir(parents=True, exist_ok=True)
                coco128_cfg_path = coco128_run_dir / "demo_config.json"
                coco128_cfg_path.write_text(
                    json.dumps(
                        {
                            "kind": "demo_config",
                            "demo": "instance-seg",
                            "background": "coco128",
                            "seed": 0,
                            "num_images": 2,
                            "image_size": 96,
                            "max_instances": 2,
                        },
                        indent=2,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                out_coco = run_instance_seg_demo(
                    run_dir=coco128_run_dir,
                    seed=0,
                    num_images=2,
                    image_size=96,
                    max_instances=2,
                    background="coco128",
                )
                print(f"config: {coco128_cfg_path}")
                _print_instance_seg_report(out_path=Path(out_coco), label="== instance-seg (coco128) ==")
                ok += 1
            except FileNotFoundError as exc:
                print("== instance-seg (coco128) ==")
                print(f"skipped: {exc}")

        # 2b) COCO instances (polygon) instance-seg
        try:
            default_instances = Path("data") / "coco" / "annotations" / "instances_val2017.json"
            default_images = Path("data") / "coco" / "images" / "val2017"

            def _ensure_tiny_coco_instances_fixture(*, fixture_dir: Path) -> tuple[Path, Path]:
                fixture_dir.mkdir(parents=True, exist_ok=True)
                images_dir = fixture_dir / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                instances_path = fixture_dir / "instances_val.json"

                # Keep this dependency-light (Pillow only), but generate something that
                # looks more like a real segmentation dataset.
                #
                # If coco128 is available locally, prefer using real photos + bbox labels
                # to create polygon masks that at least match *object locations*.
                # (coco128 does not ship true segmentation polygons in this repo.)
                import math
                import random
                import shutil

                from PIL import Image, ImageDraw

                rng = random.Random(0)
                fallback_width, fallback_height = 320, 240
                fallback_num_images = 6

                categories = [
                    {"id": 1, "name": "person"},
                    {"id": 2, "name": "bicycle"},
                    {"id": 3, "name": "dog"},
                ]

                def _jitter_color(base: tuple[int, int, int], j: int = 25) -> tuple[int, int, int]:
                    return (
                        max(0, min(255, base[0] + rng.randint(-j, j))),
                        max(0, min(255, base[1] + rng.randint(-j, j))),
                        max(0, min(255, base[2] + rng.randint(-j, j))),
                    )

                def _make_textured_bg(*, width: int, height: int) -> Image.Image:
                    # Procedural "photo-like" background: combine noise layers,
                    # colorize, blur, add gentle gradient + vignette.
                    from PIL import ImageChops, ImageEnhance, ImageFilter, ImageOps

                    # Base noise (low frequency)
                    n1 = Image.effect_noise((width, height), rng.uniform(35.0, 65.0)).convert("L")
                    n1 = n1.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.5, 3.5)))
                    # Detail noise (high frequency)
                    n2 = Image.effect_noise((width, height), rng.uniform(10.0, 25.0)).convert("L")
                    n2 = n2.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.1)))

                    mix = ImageChops.add(n1, n2, scale=2.0)

                    # Choose a palette that looks like real-world scenes.
                    palettes = [
                        ((35, 55, 90), (210, 205, 170)),  # blue -> tan
                        ((25, 70, 45), (200, 200, 205)),  # green -> gray
                        ((70, 50, 35), (215, 210, 195)),  # brown -> beige
                        ((40, 45, 55), (205, 210, 220)),  # slate -> light gray
                    ]
                    dark, light = rng.choice(palettes)
                    base = ImageOps.colorize(mix, black=dark, white=light).convert("RGB")

                    # Gentle global gradient (simulate lighting)
                    grad = Image.new("L", (width, height), 0)
                    gd = ImageDraw.Draw(grad)
                    top = rng.randint(40, 90)
                    bottom = rng.randint(170, 230)
                    for y in range(height):
                        v = int(top + (bottom - top) * (y / max(1, height - 1)))
                        gd.line([0, y, width, y], fill=v)
                    grad_rgb = ImageOps.colorize(grad, black=(0, 0, 0), white=(255, 255, 255)).convert("RGB")
                    base = ImageChops.multiply(base, grad_rgb)

                    # Mild sharpening/contrast like a camera pipeline
                    base = base.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))
                    base = ImageEnhance.Contrast(base).enhance(rng.uniform(1.05, 1.25))
                    base = ImageEnhance.Color(base).enhance(rng.uniform(1.05, 1.20))

                    # Vignette
                    vig = Image.new("L", (width, height), 255)
                    vd = ImageDraw.Draw(vig)
                    inset = rng.randint(10, 35)
                    vd.ellipse([inset, inset, width - inset, height - inset], fill=210)
                    vig = vig.filter(ImageFilter.GaussianBlur(radius=rng.uniform(10.0, 18.0)))
                    base = ImageChops.multiply(base, ImageOps.colorize(vig, black=(0, 0, 0), white=(255, 255, 255)).convert("RGB"))

                    return base

                def _make_polygon(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
                    n = rng.randint(8, 14)
                    pts: list[tuple[float, float]] = []
                    phase = rng.random() * (2.0 * math.pi)
                    for k in range(n):
                        a = phase + (2.0 * math.pi * k / n)
                        rr = r * (0.65 + 0.55 * rng.random())
                        x = cx + rr * math.cos(a)
                        y = cy + rr * math.sin(a)
                        x = max(2.0, min(float(width) - 3.0, x))
                        y = max(2.0, min(float(height) - 3.0, y))
                        pts.append((x, y))
                    return pts

                def _iter_coco128_pairs() -> list[tuple[Path, Path]]:
                    coco128_root = Path("data") / "coco128"
                    images_base = coco128_root / "images"
                    labels_base = coco128_root / "labels"
                    if not images_base.exists() or not labels_base.exists():
                        return []
                    pairs: list[tuple[Path, Path]] = []
                    for img in sorted(images_base.rglob("*")):
                        if not img.is_file() or img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"):
                            continue
                        rel = img.relative_to(images_base)
                        lab = labels_base / rel.with_suffix(".txt")
                        if lab.exists():
                            pairs.append((img, lab))
                    return pairs

                def _yolo_bbox_to_xyxy(*, w: int, h: int, xc: float, yc: float, bw: float, bh: float) -> tuple[int, int, int, int]:
                    x0 = int((float(xc) - float(bw) / 2.0) * float(w))
                    y0 = int((float(yc) - float(bh) / 2.0) * float(h))
                    x1 = int((float(xc) + float(bw) / 2.0) * float(w))
                    y1 = int((float(yc) + float(bh) / 2.0) * float(h))
                    x0 = max(0, min(int(w) - 1, x0))
                    y0 = max(0, min(int(h) - 1, y0))
                    x1 = max(0, min(int(w), x1))
                    y1 = max(0, min(int(h), y1))
                    if x1 <= x0:
                        x1 = min(int(w), x0 + 1)
                    if y1 <= y0:
                        y1 = min(int(h), y0 + 1)
                    return x0, y0, x1, y1

                def _bbox_to_polygon(*, x0: int, y0: int, x1: int, y1: int) -> list[float]:
                    # Create a jittered ellipse-like polygon inside bbox.
                    cx = (x0 + x1) / 2.0
                    cy = (y0 + y1) / 2.0
                    rx = max(2.0, (x1 - x0) / 2.0)
                    ry = max(2.0, (y1 - y0) / 2.0)
                    n = rng.randint(10, 18)
                    phase = rng.random() * (2.0 * math.pi)
                    pts: list[float] = []
                    for k in range(n):
                        a = phase + (2.0 * math.pi * k / n)
                        # Alternate radius a bit to avoid perfect ellipse.
                        s = 0.75 + 0.30 * rng.random()
                        if (k % 2) == 0:
                            s *= 0.92
                        x = cx + (rx * s) * math.cos(a)
                        y = cy + (ry * s) * math.sin(a)
                        x = max(float(x0), min(float(x1 - 1), x))
                        y = max(float(y0), min(float(y1 - 1), y))
                        pts.extend([float(x), float(y)])
                    return pts

                images: list[dict[str, Any]] = []
                annotations: list[dict[str, Any]] = []
                ann_id = 1

                coco128_pairs = _iter_coco128_pairs()
                if coco128_pairs:
                    # Use real photos + their bbox labels (converted into polygons).
                    rng.shuffle(coco128_pairs)
                    selected = coco128_pairs[: min(6, len(coco128_pairs))]
                    seen_class_ids: set[int] = set()
                    for image_id, (src_img, src_lab) in enumerate(selected, start=1):
                        file_name = f"{image_id:012d}{src_img.suffix.lower()}"
                        dst_img = images_dir / file_name
                        shutil.copyfile(src_img, dst_img)
                        img = Image.open(dst_img).convert("RGB")
                        w, h = img.size

                        lines = [ln.strip() for ln in src_lab.read_text(encoding="utf-8").splitlines() if ln.strip()]
                        # Bound runtime.
                        lines = lines[:10]
                        for ln in lines:
                            parts = ln.split()
                            if len(parts) < 5:
                                continue
                            class_id = int(float(parts[0]))
                            xc, yc, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                            x0, y0, x1, y1 = _yolo_bbox_to_xyxy(w=w, h=h, xc=xc, yc=yc, bw=bw, bh=bh)
                            if (x1 - x0) < 4 or (y1 - y0) < 4:
                                continue

                            seg = [_bbox_to_polygon(x0=x0, y0=y0, x1=x1, y1=y1)]
                            # Map YOLO class_id to category_id space (just keep it as-is).
                            cat_id = int(class_id)
                            seen_class_ids.add(cat_id)

                            annotations.append(
                                {
                                    "id": ann_id,
                                    "image_id": image_id,
                                    "category_id": cat_id,
                                    "iscrowd": 0,
                                    "segmentation": seg,
                                }
                            )
                            ann_id += 1

                        images.append({"id": image_id, "file_name": file_name, "width": w, "height": h})

                    # Provide minimal categories list for the ids we used.
                    used_categories = [{"id": int(cid), "name": f"class_{int(cid)}"} for cid in sorted(seen_class_ids)]
                    if used_categories:
                        categories = used_categories
                else:
                    # Fallback: fully synthetic photos + shapes.
                    width, height = fallback_width, fallback_height
                    num_images = fallback_num_images
                    for image_id in range(1, num_images + 1):
                        file_name = f"{image_id:012d}.jpg"
                        img = _make_textured_bg(width=width, height=height)
                        draw = ImageDraw.Draw(img)

                        num_inst = rng.randint(2, 5)
                        for _ in range(num_inst):
                            cat = rng.choice(categories)
                            cx = rng.uniform(60.0, float(width) - 60.0)
                            cy = rng.uniform(50.0, float(height) - 50.0)
                            r = rng.uniform(18.0, 55.0)

                            # Some instances are multi-polygons (COCO supports list of polygons).
                            segs: list[list[float]] = []
                            poly1 = _make_polygon(cx, cy, r)
                            segs.append([v for xy in poly1 for v in xy])
                            if rng.random() < 0.25:
                                poly2 = _make_polygon(
                                    cx + rng.uniform(-25.0, 25.0),
                                    cy + rng.uniform(-20.0, 20.0),
                                    r * 0.55,
                                )
                                segs.append([v for xy in poly2 for v in xy])

                            fill = _jitter_color((60, 140, 220) if (int(cat["id"]) % 2) == 0 else (220, 120, 60), 20)
                            outline = (0, 0, 0)
                            for poly in segs:
                                pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly) - 1, 2)]
                                draw.polygon(pts, fill=fill, outline=outline)
                                draw.line(pts + [pts[0]], fill=outline, width=2)

                            annotations.append(
                                {
                                    "id": ann_id,
                                    "image_id": image_id,
                                    "category_id": int(cat["id"]),
                                    "iscrowd": 0,
                                    "segmentation": segs,
                                }
                            )
                            ann_id += 1

                        img.save(images_dir / file_name)
                        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})

                coco = {"images": images, "annotations": annotations, "categories": categories}
                instances_path.write_text(json.dumps(coco), encoding="utf-8")
                return instances_path, images_dir

            suite_instances = getattr(args, "coco_instances_json", None)
            suite_images = getattr(args, "coco_images_dir", None)
            if suite_instances or suite_images:
                instances_path = Path(str(suite_instances)) if suite_instances else None
                images_path = Path(str(suite_images)) if suite_images else None
                if instances_path is None or images_path is None:
                    raise ValueError("demo suite requires both --coco-instances-json and --coco-images-dir")
                coco_source = "cli"
            else:
                if default_instances.exists() and default_images.exists():
                    instances_path = default_instances
                    images_path = default_images
                    coco_source = "data/coco"
                else:
                    instances_path, images_path = _ensure_tiny_coco_instances_fixture(
                        fixture_dir=Path("demo_output")
                        / "instance_seg"
                        / f"suite_{suite_id}"
                        / "_coco_instances_fixture"
                    )
                    coco_source = "fixture"

            ci_run_dir = suite_root / "coco_instances"
            ci_run_dir.mkdir(parents=True, exist_ok=True)
            ci_cfg_path = ci_run_dir / "demo_config.json"
            ci_cfg_path.write_text(
                json.dumps(
                    {
                        "kind": "demo_config",
                        "demo": "instance-seg",
                        "background": "coco-instances",
                        "seed": 0,
                        "num_images": 2,
                        "image_size": 96,
                        "max_instances": 2,
                        "coco_instances_json": str(instances_path),
                        "coco_images_dir": str(images_path),
                        "coco_source": coco_source,
                        "inference": "auto",
                        "device": "cpu",
                        "score_threshold": 0.5,
                    },
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            if instances_path.exists() and images_path.exists():
                out_ci = run_instance_seg_demo(
                    run_dir=ci_run_dir,
                    seed=0,
                    num_images=2,
                    image_size=96,
                    max_instances=2,
                    background="coco-instances",
                    coco_instances_json=instances_path,
                    coco_images_dir=images_path,
                    inference="auto",
                    device="cpu",
                    score_threshold=0.5,
                )
                print(f"config: {ci_cfg_path}")
                _print_instance_seg_report(out_path=Path(out_ci), label="== instance-seg (coco-instances) ==")
                ok += 1
            else:
                print("== instance-seg (coco-instances) ==")
                print(f"skipped: not found: instances={instances_path} images_dir={images_path}")
        except Exception as exc:
            print("== instance-seg (coco-instances) ==")
            print(f"skipped: {exc}")

        try:
            suite_root.mkdir(parents=True, exist_ok=True)
            suite_cfg_path = suite_root / "suite_config.json"
            suite_cfg_path.write_text(
                json.dumps(
                    {
                        "kind": "demo_suite_config",
                        "suite_id": suite_id,
                        "has_real_coco_instances": bool(has_real_coco_instances),
                        "cli_coco_instances_json": str(suite_instances) if suite_instances else None,
                        "cli_coco_images_dir": str(suite_images) if suite_images else None,
                    },
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"suite_config: {suite_cfg_path}")
        except Exception as exc:
            _debug_demo_summary("instance-seg suite config write skipped", exc)

        # 3) Continual demo (skip if torch missing)
        try:
            from yolozu.demos.continual import run_continual_demo

            out = run_continual_demo(
                output=None,
                seed=0,
                device="cpu",
                method="naive",
                steps_a=50,
                steps_b=50,
                batch_size=64,
                hidden=32,
                lr=1e-2,
                corr=2.0,
                noise=0.6,
                n_train=1024,
                n_eval=256,
                ewc_lambda=20.0,
                fisher_batches=8,
                replay_capacity=256,
                replay_k=32,
            )
            if out is not None:
                print("== continual ==")
                print(f"output_dir: {Path(out).parent}")
                print(str(out))
                ok += 1
        except Exception as exc:
            print("== continual ==")
            print(f"skipped: {exc}")

        return 0 if ok > 0 else 1

    if args.demo_command == "instance-seg":
        from yolozu.demos.instance_seg import run_instance_seg_demo

        bg = str(getattr(args, "background", "synthetic"))
        resolved_instances = getattr(args, "coco_instances_json", None)
        resolved_images = getattr(args, "coco_images_dir", None)
        yolo_root = getattr(args, "yolo_root", None)
        yolo_split = getattr(args, "yolo_split", "val")
        if str(bg).strip().lower() == "coco-instances":
            instances_was_default = resolved_instances is None
            images_was_default = resolved_images is None
            if resolved_instances is None:
                resolved_instances = str(Path("data") / "coco" / "annotations" / "instances_val2017.json")
            if resolved_images is None:
                resolved_images = str(Path("data") / "coco" / "images" / "val2017")

            # Keep the UX short: `yolozu demo instance-seg` should run even if
            # the user hasn't downloaded COCO instances yet.
            try:
                instances_ok = Path(str(resolved_instances)).exists() if resolved_instances else False
                images_ok = Path(str(resolved_images)).exists() if resolved_images else False
            except Exception as exc:
                _debug_demo_summary("COCO path availability probe failed", exc)
                instances_ok = False
                images_ok = False

            if instances_was_default and images_was_default and (not (instances_ok and images_ok)):
                print(
                    "note: COCO instances not found at the default paths; falling back to --background synthetic. "
                    "To enable coco-instances: python3 scripts/download_coco_instances_tiny.py"
                )
                bg = "synthetic"
                resolved_instances = None
                resolved_images = None

        resolved_inference = getattr(args, "inference", None)
        bg_norm = str(bg).strip().lower()
        inf_norm = None if resolved_inference is None else str(resolved_inference).strip().lower()

        if bg_norm == "yolo-bbox" and not yolo_root:
            raise SystemExit("background=yolo-bbox requires --yolo-root <dataset> (with images/<split> and labels/<split>)")

        # Instance-seg "inference" is only meaningful for background=coco-instances.
        # Some entrypoints set a default like "auto"; keep the demo runnable by
        # forcing inference=none unless coco-instances is explicitly requested.
        if bg_norm != "coco-instances":
            if inf_norm not in (None, "none", "auto"):
                raise SystemExit(
                    "instance-seg inference is only supported for --background coco-instances "
                    f"(got --background {bg!r} --inference {resolved_inference!r})"
                )
            resolved_inference = "none"
            inf_norm = "none"

        if inf_norm is None:
            resolved_inference = "auto"
            inf_norm = "auto"

        if inf_norm in ("auto", "torchvision"):
            try:
                import torch  # noqa: F401
                import torchvision  # noqa: F401
            except Exception as exc:
                if resolved_inference == "torchvision":
                    raise SystemExit(
                        "instance-seg inference requires torch+torchvision. "
                        "Install: python3 -m pip install -U 'yolozu[demo]' (pip) or python3 -m pip install -e '.[demo]' (repo checkout)"
                    ) from exc
                print(
                    "note: torch/torchvision not available; falling back to --inference none "
                    "(synthetic predictions). To enable inference: python3 -m pip install -U 'yolozu[demo]'",
                    file=sys.stderr,
                )
                resolved_inference = "none"

        out = run_instance_seg_demo(
            run_dir=getattr(args, "run_dir", None),
            seed=int(getattr(args, "seed", 0)),
            num_images=int(getattr(args, "num_images", 8)),
            image_size=int(getattr(args, "image_size", 96)),
            max_instances=int(getattr(args, "max_instances", 2)),
            background=bg,
            coco_instances_json=resolved_instances,
            coco_images_dir=resolved_images,
            yolo_root=yolo_root,
            yolo_split=str(yolo_split),
            inference=str(resolved_inference),
            device=str(getattr(args, "device", "cpu")),
            score_threshold=float(getattr(args, "score_threshold", 0.5)),
            progress=getattr(args, "progress", None),
        )
        try:
            cfg_path = Path(out).parent / "demo_config.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "kind": "demo_config",
                        "demo": "instance-seg",
                        "background": bg,
                        "seed": int(getattr(args, "seed", 0)),
                        "num_images": int(getattr(args, "num_images", 8)),
                        "image_size": int(getattr(args, "image_size", 96)),
                        "max_instances": int(getattr(args, "max_instances", 2)),
                        "coco_instances_json": (
                            str(resolved_instances)
                            if resolved_instances
                            else None
                        ),
                        "coco_images_dir": (
                            str(resolved_images)
                            if resolved_images
                            else None
                        ),
                        "yolo_root": str(yolo_root) if yolo_root else None,
                        "yolo_split": str(yolo_split),
                        "inference": str(resolved_inference),
                        "device": str(getattr(args, "device", "cpu")),
                        "score_threshold": float(getattr(args, "score_threshold", 0.5)),
                    },
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"config: {cfg_path}")
        except Exception as exc:
            _debug_demo_summary("instance-seg config echo skipped", exc)
        _print_instance_seg_report(out_path=Path(out))
        return 0

    if args.demo_command == "instance-seg-tta":
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except Exception as exc:
            raise SystemExit(
                "demo instance-seg-tta requires torch+torchvision. "
                "Install: python3 -m pip install -U 'yolozu[demo]' (pip) or python3 -m pip install -e '.[demo]' (repo checkout)"
            ) from exc

        from yolozu.demos.instance_seg_tta import run_instance_seg_tta_demo

        out = run_instance_seg_tta_demo(
            run_dir=getattr(args, "run_dir", None),
            seed=int(getattr(args, "seed", 0)),
            coco_instances_json=getattr(args, "coco_instances_json", None),
            coco_images_dir=getattr(args, "coco_images_dir", None),
            device=str(getattr(args, "device", "auto")),
            score_threshold=float(getattr(args, "score_threshold", 0.25)),
            max_instances=int(getattr(args, "max_instances", 8)),
            num_images=int(getattr(args, "num_images", 8)),
            corruption=str(getattr(args, "corruption", "brightness")),
            severity=int(getattr(args, "severity", 5)),
            image_id=getattr(args, "image_id", None),
        )
        try:
            payload = json.loads(Path(out).read_text(encoding="utf-8"))
            selected = payload.get("selected") or {}
            settings = payload.get("settings") or {}
            artifacts = payload.get("artifacts") or {}
            raw = selected.get("raw") or {}
            tta = selected.get("tta") or {}
            print(
                "instance-seg TTA demo: "
                f"image_id={selected.get('image_id')} "
                f"tp {raw.get('tp')}→{tta.get('tp')} "
                f"fp {raw.get('fp')}→{tta.get('fp')} "
                f"fn {raw.get('fn')}→{tta.get('fn')} "
                f"mean_iou {float(raw.get('mean_iou', 0.0)):.3f}→{float(tta.get('mean_iou', 0.0)):.3f} "
                f"(corruption={settings.get('corruption')} severity={settings.get('severity')})"
            )
            overlay_raw = artifacts.get("overlay_raw") if isinstance(artifacts, dict) else None
            overlay_tta = artifacts.get("overlay_tta") if isinstance(artifacts, dict) else None
            if overlay_raw:
                print(str(overlay_raw))
            if overlay_tta:
                print(str(overlay_tta))
        except Exception as exc:
            _debug_demo_summary("instance-seg TTA artifact summary skipped", exc)
        print(str(out))
        return 0

    if args.demo_command == "overview":
        out = _write_demo_overview_report(output=getattr(args, "output", None))
        print("demo overview:")
        print(str(out))
        print("visible quickstart (writes PNG overlays):")
        print("yolozu demo instance-seg --run-dir reports/quickstart_instance_seg --progress")
        print("settings checklist:")
        print("configs/quickstart/instance_seg_demo.yaml")
        print("expected PNG:")
        print("reports/quickstart_instance_seg/overlays/overlay_img_0000.png")
        return 0

    if args.demo_command == "continual":
        from yolozu.demos.continual import (
            format_continual_demo_suite_markdown,
            run_continual_demo,
            run_continual_demo_suite,
        )

        def _friendly_demo_deps_error(exc: Exception) -> SystemExit:
            msg = str(exc)
            if "requires torchvision" in msg:
                return SystemExit(
                    "demo continual (--problem mnist_rotate) requires torchvision. "
                    "Install: python3 -m pip install -U 'yolozu[demo]' (pip) or python3 -m pip install -e '.[demo]' (repo checkout)"
                )
            if "requires torch" in msg:
                return SystemExit(
                    "demo continual requires torch. "
                    "Install: python3 -m pip install -U 'yolozu[demo]' (pip) or python3 -m pip install -e '.[demo]' (repo checkout)"
                )
            return SystemExit(msg)

        methods = None
        if args.methods:
            methods = [str(m) for m in args.methods]
        elif args.compare:
            methods = ["naive", "ewc", "replay", "ewc_replay"]

        # Convenience presets to keep the command short.
        problem = str(getattr(args, "problem", "toy2d"))
        method_override = None
        steps_a = int(args.steps_a)
        steps_b = int(args.steps_b)
        n_train = int(args.n_train)
        n_eval = int(args.n_eval)
        fisher_batches = int(args.fisher_batches)
        replay_capacity = int(args.replay_capacity)
        replay_k = int(args.replay_k)

        if bool(getattr(args, "practical", False)):
            problem = "mnist_rotate"
            # Keep this simple and fast on CPU.
            method_override = "ewc"
            steps_a = min(steps_a, 60)
            steps_b = min(steps_b, 60)
            n_train = min(n_train, 2048)
            n_eval = min(n_eval, 512)
            fisher_batches = min(fisher_batches, 16)

        if bool(getattr(args, "fast", False)):
            steps_a = min(steps_a, 60)
            steps_b = min(steps_b, 60)
            n_train = min(n_train, 2048)
            n_eval = min(n_eval, 512)
            fisher_batches = min(fisher_batches, 16)
            replay_capacity = min(replay_capacity, 256)
            replay_k = min(replay_k, 32)

        if methods and len(methods) > 1:
            try:
                out = run_continual_demo_suite(
                    methods=methods,
                    output=args.output,
                    seed=int(args.seed),
                    device=str(args.device),
                    problem=str(problem),
                    data_dir=str(getattr(args, "data_dir", str(Path("data") / "torchvision"))),
                    steps_a=int(steps_a),
                    steps_b=int(steps_b),
                    batch_size=int(args.batch_size),
                    hidden=int(args.hidden),
                    lr=float(args.lr),
                    corr=float(args.corr),
                    noise=float(args.noise),
                    n_train=int(n_train),
                    n_eval=int(n_eval),
                    ewc_lambda=float(args.ewc_lambda),
                    fisher_batches=int(fisher_batches),
                    replay_capacity=int(replay_capacity),
                    replay_k=int(replay_k),
                )
            except RuntimeError as exc:
                raise _friendly_demo_deps_error(exc)
            try:
                payload = json.loads(Path(out).read_text(encoding="utf-8"))
                md = format_continual_demo_suite_markdown(payload)
                print(md, end="")
                if args.markdown:
                    md_path = Path(out).with_suffix(".md")
                    md_path.write_text(md, encoding="utf-8")
                    print(str(md_path))
            except Exception as exc:
                _debug_demo_summary("continual demo markdown write skipped", exc)
            print(str(out))
            return 0

        method = str(args.method)
        if methods and len(methods) == 1:
            method = str(methods[0])
        if method_override is not None:
            method = str(method_override)

        try:
            out = run_continual_demo(
                output=args.output,
                seed=int(args.seed),
                device=str(args.device),
                method=method,
                problem=str(problem),
                data_dir=str(getattr(args, "data_dir", str(Path("data") / "torchvision"))),
                steps_a=int(steps_a),
                steps_b=int(steps_b),
                batch_size=int(args.batch_size),
                hidden=int(args.hidden),
                lr=float(args.lr),
                corr=float(args.corr),
                noise=float(args.noise),
                n_train=int(n_train),
                n_eval=int(n_eval),
                ewc_lambda=float(args.ewc_lambda),
                fisher_batches=int(fisher_batches),
                replay_capacity=int(replay_capacity),
                replay_k=int(replay_k),
            )
        except RuntimeError as exc:
            raise _friendly_demo_deps_error(exc)
        try:
            payload = json.loads(Path(out).read_text(encoding="utf-8"))
            metrics = payload.get("metrics", {})
            settings = payload.get("settings", {})
            a = metrics.get("after_task_a", {})
            b = metrics.get("after_task_b", {})
            forgetting = metrics.get("forgetting_acc_a")
            gain = metrics.get("gain_acc_b")
            prob = ""
            model = ""
            bb = ""
            if isinstance(settings, dict):
                prob = str(settings.get("problem") or "")
                model = str(settings.get("model") or "")
                bb = str(settings.get("backbone") or "")
            msg = f"continual demo ({method}): "
            if prob or model:
                msg += f"problem={prob} model={model} "
            if bb:
                msg += f"backbone={bb} "
            msg += (
                f"accA {a.get('acc_a'):.3f}→{b.get('acc_a'):.3f} "
                f"accB {a.get('acc_b'):.3f}→{b.get('acc_b'):.3f} "
                f"forget={forgetting:.3f} gain={gain:.3f} "
                f"(output_dir={Path(out).parent})"
            )
            print(msg)
            if args.markdown:
                md = format_continual_demo_suite_markdown({"runs": [{"method": method, "metrics": metrics}]})
                md_path = Path(out).with_suffix(".md")
                md_path.write_text(md, encoding="utf-8")
                print(str(md_path))
        except Exception as exc:
            _debug_demo_summary("continual suite markdown write skipped", exc)
        print(str(out))
        return 0

    if args.demo_command == "ttt":
        try:
            import torch  # noqa: F401
        except Exception as exc:
            raise SystemExit(
                "demo ttt requires torch. "
                "Install: python3 -m pip install -U 'yolozu[demo]' (pip) or python3 -m pip install -e '.[demo]' (repo checkout)"
            ) from exc

        from yolozu.demos.ttt_improvement import run_ttt_improvement_demo

        suite_id = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
        run_dir = getattr(args, "run_dir", None)
        if run_dir is None:
            run_dir = str(Path("demo_output") / "ttt" / suite_id)

        res = run_ttt_improvement_demo(
            run_dir=run_dir,
            dataset_root=str(getattr(args, "dataset_root")),
            split=str(getattr(args, "split", "val")),
            max_images=int(getattr(args, "max_images", 10)),
            corruption=str(getattr(args, "corruption", "gaussian_noise")),
            severity=int(getattr(args, "severity", 3)),
            seed=int(getattr(args, "seed", 2026)),
            train_seed=int(getattr(args, "train_seed", 0)),
            train_epochs=int(getattr(args, "train_epochs", 30)),
            train_batch_size=int(getattr(args, "train_batch_size", 2)),
            train_lr=float(getattr(args, "train_lr", 1e-3)),
            image_size=int(getattr(args, "image_size", 320)),
            device=str(getattr(args, "device", "cpu")),
            adapter_config=str(getattr(args, "adapter_config", "configs/yolo26_rtdetr_pose/yolo26n.json")),
            score_threshold=float(getattr(args, "score_threshold", 0.01)),
            max_detections=int(getattr(args, "max_detections", 100)),
            ttt_preset=str(getattr(args, "ttt_preset", "safe")),
            force=bool(getattr(args, "force", False)),
        )

        m0 = res.metrics_no_ttt
        m1 = res.metrics_ttt
        d = {
            "map50": float(m1.get("map50", 0.0)) - float(m0.get("map50", 0.0)),
            "map50_95": float(m1.get("map50_95", 0.0)) - float(m0.get("map50_95", 0.0)),
        }
        print(
            "ttt demo: "
            f"map50 {m0.get('map50'):.9g}→{m1.get('map50'):.9g} (Δ={d['map50']:.9g}) "
            f"map50_95 {m0.get('map50_95'):.9g}→{m1.get('map50_95'):.9g} (Δ={d['map50_95']:.9g}) "
            f"(output_dir={res.run_dir})"
        )
        if res.overlay_no_ttt_path:
            print(str(res.overlay_no_ttt_path))
        if res.overlay_ttt_path:
            print(str(res.overlay_ttt_path))
        print(str(res.report_path))
        return 0

    if args.demo_command == "keypoints":
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except Exception as exc:
            raise SystemExit(
                "demo keypoints requires torch+torchvision. "
                "Install: python3 -m pip install -U 'yolozu[demo]' (pip) or python3 -m pip install -e '.[demo]' (repo checkout)"
            ) from exc

        from yolozu.demos.keypoints import run_keypoints_demo

        out = run_keypoints_demo(
            image=getattr(args, "image", None),
            run_dir=getattr(args, "run_dir", None),
            device=str(getattr(args, "device", "auto")),
            score_threshold=float(getattr(args, "score_threshold", 0.7)),
            max_persons=int(getattr(args, "max_persons", 3)),
        )
        try:
            payload = json.loads(Path(out).read_text(encoding="utf-8"))
            res = payload.get("result", {})
            settings = payload.get("settings", {})
            n = res.get("num_persons")
            img = settings.get("image")
            run_dir = settings.get("run_dir")
            print(f"keypoints demo: persons={n} image={img} (output_dir={run_dir})")
            artifacts = res.get("artifacts", {}) if isinstance(res, dict) else {}
            overlay = artifacts.get("overlay") if isinstance(artifacts, dict) else None
            if overlay:
                print(str(overlay))
        except Exception as exc:
            _debug_demo_summary("pose demo summary skipped", exc)
        print(str(out))
        return 0

    if args.demo_command == "pose":
        try:
            import cv2  # noqa: F401
            import numpy as np  # noqa: F401
        except Exception as exc:
            raise SystemExit(
                "demo pose requires opencv-python and numpy (aruco backend needs opencv-contrib-python; "
                "densefusion backend requires CUDA + DenseFusion assets). "
                "Install: python3 -m pip install -U 'yolozu[demo]' (pip) or python3 -m pip install -e '.[demo]' (repo checkout)"
            ) from exc

        from yolozu.demos.pose6d import run_pose6d_demo

        out = run_pose6d_demo(
            image=getattr(args, "image", None),
            run_dir=getattr(args, "run_dir", None),
            backend=str(getattr(args, "backend", "chessboard")),
            densefusion_root=getattr(args, "densefusion_root", None),
            densefusion_object=str(getattr(args, "densefusion_object", "ape")),
            densefusion_auto_download=bool(getattr(args, "densefusion_auto_download", True)),
            densefusion_model=getattr(args, "densefusion_model", None),
            densefusion_refine_model=getattr(args, "densefusion_refine_model", None),
            pattern_cols=getattr(args, "pattern_cols", None),
            pattern_rows=getattr(args, "pattern_rows", None),
            square_size=float(getattr(args, "square_size", 0.04)),
            aruco_dict=str(getattr(args, "aruco_dict", "DICT_4X4_50")),
            aruco_id=int(getattr(args, "aruco_id", 23)),
            marker_length=float(getattr(args, "marker_length", 0.05)),
            camera_fx=getattr(args, "camera_fx", None),
            camera_fy=getattr(args, "camera_fy", None),
            camera_cx=getattr(args, "camera_cx", None),
            camera_cy=getattr(args, "camera_cy", None),
            sample_source=str(getattr(args, "sample_source", "auto")),
        )
        try:
            payload = json.loads(Path(out).read_text(encoding="utf-8"))
            res = payload.get("result", {})
            settings = payload.get("settings", {})
            run_dir = settings.get("run_dir")
            t_xyz = res.get("t_xyz")
            backend = settings.get("backend")
            print(f"pose demo ({backend}): t_xyz={t_xyz} (output_dir={run_dir})")
            artifacts = res.get("artifacts", {}) if isinstance(res, dict) else {}
            overlay = artifacts.get("overlay") if isinstance(artifacts, dict) else None
            if overlay:
                print(str(overlay))
        except Exception as exc:
            _debug_demo_summary("keypoints demo summary skipped", exc)
        print(str(out))
        return 0

    if args.demo_command == "depth":
        try:
            import torch  # noqa: F401
        except Exception as exc:
            raise SystemExit(
                "demo depth requires torch. "
                "Install: python3 -m pip install -U 'yolozu[demo]' (pip) or python3 -m pip install -e '.[demo]' (repo checkout)"
            ) from exc

        from yolozu.demos.depth import run_depth_demo

        try:
            out = run_depth_demo(
                image=getattr(args, "image", None),
                run_dir=getattr(args, "run_dir", None),
                device=str(getattr(args, "device", "auto")),
                model=str(getattr(args, "model", "midas_small")),
                invert=bool(getattr(args, "invert", True)),
                compare=bool(getattr(args, "compare", False)),
            )
        except Exception as exc:
            msg = str(exc)
            if "intel-isl/MiDaS" in msg or "torch.hub" in msg:
                raise SystemExit(
                    "demo depth uses MiDaS via torch.hub and downloads weights on first run; ensure network access. "
                    f"error: {exc}"
                )
            raise

        try:
            payload = json.loads(Path(out).read_text(encoding="utf-8"))
            res = payload.get("result", {})
            settings = payload.get("settings", {})
            run_dir = settings.get("run_dir")

            models = res.get("models")
            if isinstance(models, list) and models:
                names = ",".join([str(m.get("model")) for m in models])
                print(f"depth demo: compare=[{names}] (output_dir={run_dir})")
            else:
                d = (res.get("depth") or {})
                print(
                    f"depth demo: model={settings.get('model')} depth_range=[{d.get('min'):.3g}, {d.get('max'):.3g}] (output_dir={run_dir})"
                )
        except Exception as exc:
            _debug_demo_summary("depth demo summary skipped", exc)
        print(str(out))
        return 0

    if args.demo_command == "train":
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except Exception as exc:
            raise SystemExit(
                "demo train requires torch+torchvision. "
                "Install: python3 -m pip install -U 'yolozu[demo]' (pip) or python3 -m pip install -e '.[demo]' (repo checkout)"
            ) from exc

        from yolozu.demos.train import run_train_demo

        out = run_train_demo(
            output=getattr(args, "output", None),
            seed=int(getattr(args, "seed", 0)),
            device=str(getattr(args, "device", "cpu")),
            data_dir=str(getattr(args, "data_dir", str(Path("data") / "torchvision"))),
            epochs=int(getattr(args, "epochs", 1)),
            max_steps=int(getattr(args, "max_steps", 80)),
            batch_size=int(getattr(args, "batch_size", 64)),
            lr=float(getattr(args, "lr", 3e-4)),
        )
        try:
            payload = json.loads(Path(out).read_text(encoding="utf-8"))
            res = payload.get("result", {})
            train = res.get("train", {})
            val = res.get("val", {})
            settings = payload.get("settings", {})
            print(
                "train demo: "
                f"steps={train.get('steps')} loss_mean={train.get('loss_mean'):.3f} "
                f"val_acc={val.get('acc'):.3f} (output_dir={settings.get('run_dir')})"
            )
        except Exception as exc:
            _debug_demo_summary("train demo summary skipped", exc)
        print(str(out))
        return 0
    raise SystemExit("unknown demo command")
