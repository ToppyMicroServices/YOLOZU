"""Adapter-based image inference.

Runs an adapter’s ``predict()`` on a directory of images, producing
schema-correct ``predictions.json`` and optional overlay images.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

__all__ = ["predict_images", "predict_images_with_namespace"]
from typing import Any, Callable, Iterable

from yolozu.export import write_predictions_json
from yolozu.inference.export_orchestrator import export_with_backend, load_json, sha256_json


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


def _default_subprocess_or_die(cmd: list[str]) -> str:
    if len(cmd) >= 2:
        candidate = Path(str(cmd[1]))
        if candidate.suffix == ".py":
            script_path = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
            if not script_path.is_file():
                raise SystemExit(f"required script not found: {candidate}")
    proc = subprocess.run(
        cmd,
        cwd=str(Path.cwd()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.stderr and proc.stderr.strip():
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def _default_base_run_meta(*, seed: int | None, notes: str | None, config_fingerprint: dict[str, Any]) -> dict[str, Any]:
    from yolozu.core import doctor as doctor_mod

    cwd = Path.cwd()
    git = doctor_mod._gather_git_info(cwd=cwd)
    return {
        "timestamp": _now_utc(),
        "seed": seed,
        "notes": notes,
        "config_hash": sha256_json(config_fingerprint),
        "git": {"head": git.get("head"), "dirty": git.get("dirty")},
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "gpu": doctor_mod._gather_gpu_info(),
        "env": {
            "python_executable": sys.executable,
            "cwd": str(cwd),
        },
    }


def predict_images_with_namespace(
    args: argparse.Namespace,
    *,
    subprocess_or_die: Callable[[list[str]], str] | None = None,
    base_run_meta: Callable[..., dict[str, Any]] | None = None,
) -> tuple[Path, Path | None]:
    subprocess_fn = _default_subprocess_or_die if subprocess_or_die is None else subprocess_or_die
    run_meta_fn = _default_base_run_meta if base_run_meta is None else base_run_meta

    input_dir_path = Path(str(args.input_dir)).expanduser()
    if not input_dir_path.is_absolute():
        input_dir_path = Path.cwd() / input_dir_path
    if not input_dir_path.is_dir():
        raise FileNotFoundError(f"input dir not found: {input_dir_path}")

    patterns = list(args.glob) if getattr(args, "glob", None) else ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.gif"]
    images = _iter_images(input_dir_path, patterns=patterns)
    max_images = getattr(args, "max_images", None)
    if max_images is not None:
        images = images[: max(0, int(max_images))]
    if not images:
        raise FileNotFoundError(f"no images matched under: {input_dir_path}")

    output_path = Path(str(args.output)).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    overlays_path: Path | None = None
    if getattr(args, "overlays_dir", None) is not None:
        overlays_path = Path(str(args.overlays_dir)).expanduser()
        if not overlays_path.is_absolute():
            overlays_path = Path.cwd() / overlays_path

    html_path: Path | None = None
    if getattr(args, "html", None) is not None:
        html_path = Path(str(args.html)).expanduser()
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
            (temp_labels / f"{dst.stem}.txt").touch()

        export_args = argparse.Namespace(**vars(args))
        export_args.dataset = str(temp_root)
        export_args.split = split
        export_args.output = str(output_path)
        export_path = export_with_backend(
            export_args,
            subprocess_or_die=subprocess_fn,
            base_run_meta=run_meta_fn,
            dataset_override=str(temp_root),
            dataset_meta=str(input_dir_path),
        )

        wrapped_payload = _ensure_wrapper(load_json(export_path))
        _rewrite_image_paths(wrapped_payload, mapping)
        out_path = write_predictions_json(output=output_path, payload=wrapped_payload, force=True)

    if overlays_path is None:
        return out_path, None
    overlay_index = _render_overlays(payload=wrapped_payload, overlays_dir=overlays_path, max_images=max_images)
    if html_path is not None:
        _write_html_report(html_path=html_path, overlays=overlay_index, title=str(getattr(args, "title", "YOLOZU predict-images report")))
    return out_path, html_path


def predict_images(
    *,
    backend: str,
    input_dir: str | Path,
    output: str | Path,
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
    args = argparse.Namespace(
        backend=backend,
        input_dir=str(input_dir),
        output=str(output),
        max_images=max_images,
        force=force,
        glob=list(glob_patterns) if glob_patterns else None,
        overlays_dir=str(overlays_dir) if overlays_dir is not None else None,
        html=str(html) if html is not None else None,
        title=title,
        onnx=str(onnx) if onnx is not None else None,
        input_name=input_name,
        boxes_output=boxes_output,
        scores_output=scores_output,
        class_output=class_output,
        combined_output=combined_output,
        combined_format=combined_format,
        raw_output=raw_output,
        raw_format=raw_format,
        raw_postprocess=raw_postprocess,
        boxes_format=boxes_format,
        boxes_scale=boxes_scale,
        min_score=min_score,
        topk=topk,
        nms_iou=nms_iou,
        agnostic_nms=agnostic_nms,
        imgsz=imgsz,
        dry_run=dry_run,
        strict=strict,
        dataset=None,
        split=None,
        run_dir=None,
        cache=False,
        cache_dir="runs/yolozu_runs",
        notes=None,
        seed=None,
        config="rtdetr_pose/configs/base.json",
        checkpoint=None,
        device="cpu",
        infer_batch_size=1,
        image_size=None,
        score_threshold=0.3,
        max_detections=50,
        torch_compile=False,
        torch_compile_backend="inductor",
        torch_compile_mode="reduce-overhead",
        torch_amp="off",
        torch_channels_last=False,
        torch_inference_mode=True,
        lora_r=0,
        lora_alpha=None,
        lora_dropout=0.0,
        lora_target="head",
        lora_freeze_base=False,
        lora_train_bias="none",
        tta=False,
        tta_mode="postprocess",
        tta_seed=None,
        tta_flip_prob=0.5,
        tta_norm_only=False,
        tta_keypoint_swap_pairs=None,
        tta_model_merge_iou=0.55,
        tta_flip_keypoints=True,
        tta_flip_pose_offsets=True,
        tta_log_out=None,
        ttt=False,
        ttt_preset=None,
        ttt_method="tent",
        ttt_reset="stream",
        ttt_steps=1,
        ttt_batch_size=1,
        ttt_lr=1e-4,
        ttt_stop_on_non_finite=True,
        ttt_rollback_on_stop=True,
        ttt_max_grad_norm=None,
        ttt_max_update_norm=None,
        ttt_max_total_update_norm=None,
        ttt_max_loss_ratio=None,
        ttt_max_loss_increase=None,
        ttt_update_filter="all",
        ttt_include=None,
        ttt_exclude=None,
        ttt_max_batches=1,
        ttt_seed=None,
        ttt_mask_prob=0.6,
        ttt_patch_size=16,
        ttt_mask_value=0.0,
        ttt_cotta_ema_momentum=0.999,
        ttt_cotta_augmentations=None,
        ttt_cotta_aggregation="confidence_weighted_mean",
        ttt_cotta_restore_prob=0.01,
        ttt_cotta_restore_interval=1,
        ttt_eata_conf_min=0.2,
        ttt_eata_entropy_min=0.05,
        ttt_eata_entropy_max=3.0,
        ttt_eata_min_valid_dets=1,
        ttt_eata_anchor_lambda=1e-3,
        ttt_eata_selected_ratio_min=0.0,
        ttt_eata_max_skip_streak=3,
        ttt_sar_rho=0.05,
        ttt_sar_adaptive=False,
        ttt_sar_first_step_scale=1.0,
        ttt_sdft_task=None,
        ttt_aux_pose_weight=0.0,
        ttt_aux_keypoints_weight=0.0,
        ttt_aux_depth_weight=0.0,
        ttt_aux_seg_weight=0.0,
        ttt_aux_temperature=1.0,
        ttt_log_out=None,
        ttt_lite_non_torch=False,
        ttt_lite_temperature=1.0,
        ttt_lite_entropy_weight=0.0,
        ttt_lite_minmax=True,
        model=None,
        exp=None,
        weights=None,
        score_thr=0.01,
        keep_aspect=False,
        dnn_backend="opencv",
        dnn_target="cpu",
        decode="auto",
        preprocess=None,
        dump_io=None,
    )
    return predict_images_with_namespace(args)
