"""TTT improvement micro-demo (deterministic, real images when available).

This demo is designed to *show an improvement* when enabling test-time training (TTT)
under a fixed, deterministic domain shift. It intentionally keeps the setup
small/fast (few-shot + few images) so it can be used as a reproducible smoke-style
check by users.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from yolozu.dataset import build_manifest
from yolozu.predictions import canonicalize_predictions
from yolozu.simple_map import evaluate_map
from yolozu.tta.config import TTTConfig
from yolozu.tta.integration import run_ttt

try:
    from yolozu.adapter import RTDETRPoseAdapter
except Exception:  # pragma: no cover
    RTDETRPoseAdapter = None  # type: ignore[assignment]

__all__ = ["TTTDemoResult", "run_ttt_improvement_demo"]

SUPPORTED_CORRUPTIONS = (
    "gaussian_blur",
    "gaussian_noise",
    "brightness",
    "contrast",
    "jpeg",
)


def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}::{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _apply_corruption(image: Image.Image, *, corruption: str, severity: int, seed: int) -> Image.Image:
    sev = max(1, min(5, int(severity)))
    rng = random.Random(int(seed))
    rgb = image.convert("RGB")

    if corruption == "gaussian_blur":
        radius = float(0.5 * sev + rng.uniform(0.0, 0.15))
        return rgb.filter(ImageFilter.GaussianBlur(radius=radius))

    if corruption == "brightness":
        factor = float(max(0.2, 1.0 - 0.11 * sev + rng.uniform(-0.02, 0.02)))
        return ImageEnhance.Brightness(rgb).enhance(factor)

    if corruption == "contrast":
        factor = float(max(0.2, 1.0 + 0.18 * sev + rng.uniform(-0.03, 0.03)))
        return ImageEnhance.Contrast(rgb).enhance(factor)

    if corruption == "gaussian_noise":
        sigma = float(6.0 * sev)
        width, height = rgb.size
        noise_bytes = bytearray(width * height)
        for idx in range(width * height):
            value = int(128.0 + rng.gauss(0.0, sigma))
            noise_bytes[idx] = max(0, min(255, value))
        noise_l = Image.frombytes("L", rgb.size, bytes(noise_bytes))
        noise_rgb = Image.merge("RGB", (noise_l, noise_l, noise_l))
        alpha = float(min(0.75, 0.12 * sev + rng.uniform(0.0, 0.02)))
        return Image.blend(rgb, noise_rgb, alpha=alpha)

    if corruption == "jpeg":
        quality = max(10, 95 - 15 * sev)
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=int(quality), optimize=False, progressive=False)
        buffer.seek(0)
        with Image.open(buffer) as tmp:
            return tmp.convert("RGB")

    raise ValueError(f"unsupported corruption: {corruption}")


def _dataset_images_digest(paths: list[tuple[str, Path]]) -> str:
    hasher = hashlib.sha256()
    for rel_key, path in sorted(paths, key=lambda item: item[0]):
        hasher.update(str(rel_key).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_sha256_file(path).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _prepare_domain_shift_dataset(
    *,
    dataset_root: Path,
    split: str,
    out_root: Path,
    corruption: str,
    severity: int,
    seed: int,
    max_images: int,
    force: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    if corruption not in SUPPORTED_CORRUPTIONS:
        raise ValueError(f"unsupported corruption: {corruption} (supported: {', '.join(SUPPORTED_CORRUPTIONS)})")
    if int(severity) < 1 or int(severity) > 5:
        raise ValueError("--severity must be in [1, 5]")
    if out_root.exists():
        if not force:
            raise FileExistsError(f"output already exists: {out_root} (use force=True)")
        shutil.rmtree(out_root)

    manifest = build_manifest(str(dataset_root), split=str(split))
    split_effective = str(manifest.get("split") or split)
    records = list(manifest.get("images") or [])
    if max_images is not None:
        records = records[: max(0, int(max_images))]
    if not records:
        raise FileNotFoundError(f"no images found for split '{split_effective}' in {dataset_root}")

    out_images_dir = out_root / "images" / split_effective
    out_labels_dir = out_root / "labels" / split_effective
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    written_images: list[tuple[str, Path]] = []
    for rec in records:
        image_raw = str(rec.get("image") or "")
        if not image_raw:
            continue
        src_image = Path(image_raw)
        if not src_image.is_absolute():
            src_image = (Path.cwd() / src_image).resolve()
        if not src_image.exists():
            raise FileNotFoundError(f"missing source image: {src_image}")

        # Stable key for per-image RNG and output location.
        out_image = out_images_dir / src_image.name
        image_key = str(Path("images") / split_effective / out_image.name)

        with Image.open(src_image) as img:
            rgb = ImageOps.exif_transpose(img).convert("RGB")
            img_seed = _stable_seed(int(seed), image_key)
            shifted = _apply_corruption(rgb, corruption=str(corruption), severity=int(severity), seed=int(img_seed))
            shifted.save(out_image)
        written_images.append((image_key, out_image))

    source_labels_split = dataset_root / "labels" / split_effective
    if not source_labels_split.exists():
        raise FileNotFoundError(f"labels split not found: {source_labels_split}")
    shutil.copytree(source_labels_split, out_labels_dir, dirs_exist_ok=True)

    # Best-effort: copy classes.json if present.
    classes_json = source_labels_split / "classes.json"
    if classes_json.exists():
        shutil.copy2(classes_json, out_labels_dir / "classes.json")

    recipe_path = out_root / "domain_shift_recipe.json"
    recipe_id = f"{corruption}_s{int(severity)}_seed{int(seed)}"
    images_digest = _dataset_images_digest(written_images)
    domain_shift_target = {
        "id": str(recipe_id),
        "split": str(split_effective),
        "corruption": str(corruption),
        "severity": int(severity),
        "seed": int(seed),
        "source_dataset_root": str(dataset_root),
        "target_dataset_root": str(out_root),
        "image_count": int(len(written_images)),
        "images_sha256": str(images_digest),
        "deterministic": True,
    }
    recipe = {
        "kind": "yolozu_domain_shift_recipe",
        "version": 1,
        "timestamp": _now_utc_iso(),
        "domain_shift_target": domain_shift_target,
        "export_settings": {"domain_shift_target": dict(domain_shift_target)},
    }
    recipe_path.write_text(json.dumps(recipe, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_root, recipe_path, domain_shift_target


def _render_bbox_overlay(*, image_path: Path, detections: list[dict[str, Any]], out_path: Path, max_dets: int = 50) -> None:
    from PIL import ImageDraw

    with Image.open(image_path) as img:
        canvas = img.convert("RGB")

    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    for det in (detections or [])[: int(max_dets)]:
        if not isinstance(det, dict):
            continue
        bbox = det.get("bbox")
        if not isinstance(bbox, dict):
            continue
        try:
            cx = float(bbox.get("cx"))
            cy = float(bbox.get("cy"))
            bw = float(bbox.get("w"))
            bh = float(bbox.get("h"))
        except Exception:
            continue

        x1 = (cx - bw / 2.0) * float(w)
        y1 = (cy - bh / 2.0) * float(h)
        x2 = (cx + bw / 2.0) * float(w)
        y2 = (cy + bh / 2.0) * float(h)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


@dataclass(frozen=True)
class TTTDemoResult:
    report_path: Path
    run_dir: Path
    checkpoint_path: Path
    shift_dataset_root: Path
    domain_shift_recipe_path: Path
    pred_no_ttt_path: Path
    pred_ttt_path: Path
    overlay_no_ttt_path: Path | None
    overlay_ttt_path: Path | None
    metrics_no_ttt: dict[str, float]
    metrics_ttt: dict[str, float]


def run_ttt_improvement_demo(
    *,
    run_dir: str | Path,
    dataset_root: str | Path,
    split: str = "val",
    max_images: int = 10,
    corruption: str = "gaussian_noise",
    severity: int = 3,
    seed: int = 2026,
    train_seed: int = 0,
    train_epochs: int = 30,
    train_batch_size: int = 2,
    train_lr: float = 1e-3,
    image_size: int = 320,
    device: str = "cpu",
    adapter_config: str = "configs/yolo26_rtdetr_pose/yolo26n.json",
    score_threshold: float = 0.01,
    max_detections: int = 100,
    ttt_preset: str = "safe",
    force: bool = False,
) -> TTTDemoResult:
    """Run a deterministic "TTT improves metrics" micro-demo.

    Notes:
    - This demo trains a tiny few-shot checkpoint on the clean split first. This makes the
      improvement reproducible without requiring external model downloads.
    - Metrics are the built-in simple mAP proxy (mAP50 and mAP50-95) computed over bbox labels.
    """

    if RTDETRPoseAdapter is None:  # pragma: no cover
        raise RuntimeError("RTDETRPoseAdapter unavailable (torch missing?)")

    run_dir_p = Path(run_dir)
    if not run_dir_p.is_absolute():
        run_dir_p = (Path.cwd() / run_dir_p).resolve()
    run_dir_p.mkdir(parents=True, exist_ok=True)

    dataset_root_p = Path(dataset_root)
    if not dataset_root_p.is_absolute():
        dataset_root_p = (Path.cwd() / dataset_root_p).resolve()
    if not dataset_root_p.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root_p}")

    checkpoint_path = run_dir_p / "checkpoint.pt"
    if checkpoint_path.exists() and not force:
        pass
    else:
        # Train a fast few-shot model so the demo remains self-contained.
        try:
            import rtdetr_pose.train_minimal as train_minimal
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"failed to import rtdetr_pose.train_minimal: {exc}") from exc

        args = [
            "--config",
            str(adapter_config),
            "--dataset-root",
            str(dataset_root_p),
            "--split",
            str(split),
            "--device",
            str(device),
            "--real-images",
            "--use-matcher",
            "--seed",
            str(int(train_seed)),
            "--epochs",
            str(int(train_epochs)),
            "--batch-size",
            str(int(train_batch_size)),
            "--lr",
            str(float(train_lr)),
            "--image-size",
            str(int(image_size)),
            "--max-steps",
            "1000000",
            "--val-every",
            "0",
            "--no-export-onnx",
            "--checkpoint-out",
            str(checkpoint_path),
            "--metrics-json",
            str(run_dir_p / "train_metrics.json"),
            "--metrics-jsonl",
            str(run_dir_p / "train_metrics.jsonl"),
        ]
        rc = int(train_minimal.main(args))
        if rc != 0:
            raise RuntimeError(f"train_minimal failed with exit code {rc}")
        if not checkpoint_path.exists():
            raise RuntimeError("train_minimal did not produce checkpoint.pt")

    shift_root = run_dir_p / "domain_shift_dataset"
    shift_root, recipe_path, domain_shift_target = _prepare_domain_shift_dataset(
        dataset_root=dataset_root_p,
        split=str(split),
        out_root=shift_root,
        corruption=str(corruption),
        severity=int(severity),
        seed=int(seed),
        max_images=int(max_images),
        force=True,
    )

    shift_manifest = build_manifest(str(shift_root), split=str(split))
    shift_records = list(shift_manifest.get("images") or [])
    if max_images is not None:
        shift_records = shift_records[: max(0, int(max_images))]

    thresholds = [0.5 + 0.05 * i for i in range(10)]

    def _predict(*, enable_ttt: bool) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        adapter = RTDETRPoseAdapter(
            config_path=str(adapter_config),
            checkpoint_path=str(checkpoint_path),
            device=str(device),
            image_size=(int(image_size), int(image_size)),
            score_threshold=float(score_threshold),
            max_detections=int(max_detections),
            infer_batch_size=1,
            init_seed=int(train_seed),
            repro_policy="relaxed",
        )
        ttt_report = None
        if enable_ttt:
            if str(ttt_preset) != "safe":
                raise ValueError("demo currently supports only --ttt-preset safe (to keep the claim stable)")
            ttt_cfg = TTTConfig(
                enabled=True,
                method="tent",
                reset="stream",
                steps=1,
                batch_size=1,
                lr=1e-4,
                update_filter="norm_only",
                max_batches=1,
                seed=int(seed),
                max_grad_norm=1.0,
                max_update_norm=1.0,
                max_total_update_norm=1.0,
                max_loss_ratio=3.0,
            )
            ttt_report = run_ttt(adapter, shift_records, config=ttt_cfg).to_dict()
        entries = adapter.predict(shift_records)
        canonical = canonicalize_predictions(entries, strict=False, policy="clamp")
        return canonical.entries, ttt_report

    preds_no_ttt, _ = _predict(enable_ttt=False)
    preds_ttt, ttt_report = _predict(enable_ttt=True)

    map_no_ttt = evaluate_map(shift_records, preds_no_ttt, iou_thresholds=thresholds)
    map_ttt = evaluate_map(shift_records, preds_ttt, iou_thresholds=thresholds)
    metrics_no_ttt = {"map50": float(map_no_ttt.map50), "map50_95": float(map_no_ttt.map50_95)}
    metrics_ttt = {"map50": float(map_ttt.map50), "map50_95": float(map_ttt.map50_95)}

    pred_no_ttt_path = run_dir_p / "pred_no_ttt.json"
    pred_ttt_path = run_dir_p / "pred_ttt.json"
    pred_no_ttt_path.write_text(
        json.dumps(
            {
                "predictions": preds_no_ttt,
                "meta": {
                    "timestamp": _now_utc_iso(),
                    "adapter": "rtdetr_pose",
                    "device": str(device),
                    "checkpoint": str(checkpoint_path),
                    "dataset_root": str(shift_root),
                    "domain_shift_target": domain_shift_target,
                    "ttt": {"enabled": False},
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pred_ttt_path.write_text(
        json.dumps(
            {
                "predictions": preds_ttt,
                "meta": {
                    "timestamp": _now_utc_iso(),
                    "adapter": "rtdetr_pose",
                    "device": str(device),
                    "checkpoint": str(checkpoint_path),
                    "dataset_root": str(shift_root),
                    "domain_shift_target": domain_shift_target,
                    "ttt": {"enabled": True, "preset": str(ttt_preset), "report": ttt_report},
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    overlay_no_ttt_path = None
    overlay_ttt_path = None
    try:
        if shift_records:
            image0 = Path(str(shift_records[0].get("image") or ""))
            if image0.exists():
                overlay_no_ttt_path = run_dir_p / "overlay_no_ttt.png"
                overlay_ttt_path = run_dir_p / "overlay_ttt.png"
                # Find matching entries by image key.
                dets_no = []
                dets_ttt = []
                for e in preds_no_ttt:
                    if str(e.get("image")) == str(image0):
                        dets_no = list(e.get("detections") or [])
                        break
                for e in preds_ttt:
                    if str(e.get("image")) == str(image0):
                        dets_ttt = list(e.get("detections") or [])
                        break
                _render_bbox_overlay(image_path=image0, detections=dets_no, out_path=overlay_no_ttt_path, max_dets=50)
                _render_bbox_overlay(image_path=image0, detections=dets_ttt, out_path=overlay_ttt_path, max_dets=50)
    except Exception:
        overlay_no_ttt_path = None
        overlay_ttt_path = None

    report = {
        "kind": "ttt_improvement_demo",
        "timestamp": _now_utc_iso(),
        "interface_contract": "predictions interface contract",
        "dataset": {
            "source_root": str(dataset_root_p),
            "split": str(split),
            "max_images": int(max_images),
        },
        "domain_shift": {
            "recipe": str(recipe_path),
            "target": domain_shift_target,
        },
        "train": {
            "adapter_config": str(adapter_config),
            "epochs": int(train_epochs),
            "batch_size": int(train_batch_size),
            "lr": float(train_lr),
            "image_size": int(image_size),
            "seed": int(train_seed),
            "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256_file(checkpoint_path)},
        },
        "ttt": {
            "enabled": True,
            "preset": str(ttt_preset),
            "seed": int(seed),
        },
        "metrics": {
            "name": "simple_map_proxy",
            "iou_thresholds": [float(t) for t in thresholds],
            "no_ttt": dict(metrics_no_ttt),
            "with_ttt": dict(metrics_ttt),
            "delta": {
                "map50": float(metrics_ttt["map50"] - metrics_no_ttt["map50"]),
                "map50_95": float(metrics_ttt["map50_95"] - metrics_no_ttt["map50_95"]),
            },
        },
        "artifacts": {
            "run_dir": str(run_dir_p),
            "shift_dataset_root": str(shift_root),
            "pred_no_ttt": str(pred_no_ttt_path),
            "pred_ttt": str(pred_ttt_path),
            "overlay_no_ttt": (str(overlay_no_ttt_path) if overlay_no_ttt_path else None),
            "overlay_ttt": (str(overlay_ttt_path) if overlay_ttt_path else None),
        },
    }
    report_path = run_dir_p / "ttt_improvement_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    return TTTDemoResult(
        report_path=report_path,
        run_dir=run_dir_p,
        checkpoint_path=checkpoint_path,
        shift_dataset_root=shift_root,
        domain_shift_recipe_path=recipe_path,
        pred_no_ttt_path=pred_no_ttt_path,
        pred_ttt_path=pred_ttt_path,
        overlay_no_ttt_path=overlay_no_ttt_path,
        overlay_ttt_path=overlay_ttt_path,
        metrics_no_ttt=metrics_no_ttt,
        metrics_ttt=metrics_ttt,
    )
