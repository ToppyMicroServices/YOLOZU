#!/usr/bin/env python3
"""Export target-conditioned RT-DETR pose estimates in official BOP19 CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - exercised on platforms without resource.
    resource = None

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "rtdetr_pose"))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run RT-DETR pose on official BOP19 localization targets and write "
            "scene_id,im_id,obj_id,score,R,t,time CSV plus provenance JSON."
        )
    )
    parser.add_argument("--bop-root", required=True, help="Extracted T-LESS root containing test_primesense/.")
    parser.add_argument("--targets", required=True, help="Official test_targets_bop19.json.")
    parser.add_argument("--config", required=True, help="RT-DETR pose model config.")
    parser.add_argument("--checkpoint", required=True, help="Compatible trained checkpoint.")
    parser.add_argument("--output", required=True, help="Fresh BOP19 result CSV.")
    parser.add_argument("--report", default=None, help="Evidence JSON (default: <output>.report.json).")
    parser.add_argument("--split", default="test_primesense", help="BOP image split (default: test_primesense).")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu).")
    parser.add_argument("--image-size", type=int, default=96, help="Square model input size (default: 96).")
    parser.add_argument("--max-images", type=int, default=None, help="Optional bounded unique-image limit.")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _rot6d_to_matrix(values: list[float]) -> list[list[float]]:
    if len(values) != 6:
        raise ValueError("rot6d must contain six values")

    def normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1e-12:
            return [1.0, 0.0, 0.0]
        return [value / norm for value in vector]

    first = normalize(values[:3])
    second_raw = values[3:]
    projection = sum(first[index] * second_raw[index] for index in range(3))
    second = normalize(
        [second_raw[index] - projection * first[index] for index in range(3)]
    )
    third = [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]
    return [first, second, third]


def _scaled_intrinsics(
    camera_matrix: list[float],
    *,
    original_width: int,
    original_height: int,
    image_size: int,
) -> tuple[float, float, float, float]:
    if len(camera_matrix) != 9:
        raise ValueError("cam_K must contain nine values")
    scale_x = float(image_size) / float(original_width)
    scale_y = float(image_size) / float(original_height)
    return (
        float(camera_matrix[0]) * scale_x,
        float(camera_matrix[4]) * scale_y,
        float(camera_matrix[2]) * scale_x,
        float(camera_matrix[5]) * scale_y,
    )


def _translation_mm(
    *,
    bbox: list[float],
    offsets: list[float],
    log_z: float,
    k_delta: list[float],
    intrinsics: tuple[float, float, float, float],
    image_size: int,
) -> list[float]:
    fx, fy, cx, cy = intrinsics
    if len(k_delta) == 4:
        fx = fx * (1.0 + float(k_delta[0]))
        fy = fy * (1.0 + float(k_delta[1]))
        cx = cx + float(k_delta[2])
        cy = cy + float(k_delta[3])
    fx = max(abs(fx), 1e-6)
    fy = max(abs(fy), 1e-6)
    u = float(bbox[0]) * float(image_size) + float(offsets[0])
    v = float(bbox[1]) * float(image_size) + float(offsets[1])
    z_m = max(math.exp(float(log_z)), 1e-6)
    return [
        ((u - cx) / fx) * z_m * 1000.0,
        ((v - cy) / fy) * z_m * 1000.0,
        z_m * 1000.0,
    ]


def _target_groups(
    targets: list[dict[str, Any]],
    *,
    max_images: int | None,
) -> list[tuple[tuple[int, int], list[dict[str, Any]]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for target in targets:
        key = (int(target["scene_id"]), int(target["im_id"]))
        grouped.setdefault(key, []).append(target)
    items = sorted(grouped.items())
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("--max-images must be positive")
        items = items[: int(max_images)]
    return items


def _find_image(split_root: Path, scene_id: int, image_id: int) -> Path:
    scene = split_root / f"{scene_id:06d}" / "rgb"
    for extension in (".png", ".jpg", ".jpeg"):
        candidate = scene / f"{image_id:06d}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"BOP RGB image not found: scene={scene_id} image={image_id}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    bop_root = Path(args.bop_root).expanduser().resolve()
    targets_path = Path(args.targets).expanduser().resolve()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else output_path.with_suffix(output_path.suffix + ".report.json")
    )
    for path, label in (
        (bop_root, "BOP root"),
        (targets_path, "targets"),
        (config_path, "config"),
        (checkpoint_path, "checkpoint"),
    ):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")
    if output_path.exists() or output_path.is_symlink():
        raise SystemExit(f"refusing to replace existing output: {output_path}")
    if report_path.exists() or report_path.is_symlink():
        raise SystemExit(f"refusing to replace existing report: {report_path}")
    image_size = int(args.image_size)
    if image_size <= 0:
        raise SystemExit("--image-size must be positive")

    try:
        import numpy as np
        import torch
        from PIL import Image
        from rtdetr_pose.config import load_config
        from rtdetr_pose.factory import build_model
        from yolozu.inference.checkpoint_compatibility import load_checkpoint_compatible
    except ImportError as exc:
        raise SystemExit(f"required inference dependency missing: {exc}") from exc

    targets = _load_json(targets_path)
    if not isinstance(targets, list):
        raise SystemExit("targets must be a JSON array")
    groups = _target_groups(targets, max_images=args.max_images)
    split_root = bop_root / str(args.split)
    if not split_root.is_dir():
        raise SystemExit(f"BOP split not found: {split_root}")

    torch.use_deterministic_algorithms(True)
    config = load_config(config_path)
    model = build_model(config.model).eval()
    checkpoint_report = load_checkpoint_compatible(
        model,
        checkpoint_path,
        config_identity=str(config_path),
        allow_partial=False,
    )
    model.to(str(args.device))

    scene_cameras: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for (scene_id, image_id), image_targets in groups:
        image_path = _find_image(split_root, scene_id, image_id)
        if scene_id not in scene_cameras:
            camera_path = split_root / f"{scene_id:06d}" / "scene_camera.json"
            scene_cameras[scene_id] = _load_json(camera_path)
        camera = scene_cameras[scene_id][str(image_id)]
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            original_width, original_height = rgb.size
            resized = rgb.resize((image_size, image_size), resample=Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = (
            torch.from_numpy(array)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
            .to(str(args.device))
        )
        infer_started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(tensor)
        image_seconds = float(time.perf_counter() - infer_started)
        probabilities = outputs["logits"][0].softmax(dim=-1)
        boxes = outputs["bbox"][0].sigmoid()
        rotations = outputs["rot6d"][0]
        depths = outputs["log_z"][0].squeeze(-1)
        offsets = outputs["offsets"][0]
        k_delta = outputs["k_delta"][0]
        intrinsics = _scaled_intrinsics(
            camera["cam_K"],
            original_width=original_width,
            original_height=original_height,
            image_size=image_size,
        )
        for target in sorted(image_targets, key=lambda item: int(item["obj_id"])):
            object_id = int(target["obj_id"])
            class_index = object_id - 1
            instance_count = max(1, int(target.get("inst_count", 1)))
            if not 0 <= class_index < int(probabilities.shape[-1]) - 1:
                raise SystemExit(f"target object id outside model classes: {object_id}")
            class_scores = probabilities[:, class_index]
            count = min(instance_count, int(class_scores.shape[0]))
            selected_scores, selected_queries = torch.topk(class_scores, k=count)
            for score, query in zip(selected_scores.tolist(), selected_queries.tolist()):
                rotation = _rot6d_to_matrix(
                    [float(value) for value in rotations[query].tolist()]
                )
                translation = _translation_mm(
                    bbox=[float(value) for value in boxes[query].tolist()],
                    offsets=[float(value) for value in offsets[query].tolist()],
                    log_z=float(depths[query].item()),
                    k_delta=[float(value) for value in k_delta.tolist()],
                    intrinsics=intrinsics,
                    image_size=image_size,
                )
                rows.append(
                    {
                        "scene_id": scene_id,
                        "im_id": image_id,
                        "obj_id": object_id,
                        "score": max(float(score), 1e-12),
                        "R": rotation,
                        "t": translation,
                        "time": image_seconds,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
        for row in rows:
            writer.writerow(
                [
                    row["scene_id"],
                    row["im_id"],
                    row["obj_id"],
                    f"{row['score']:.12g}",
                    " ".join(f"{value:.12g}" for line in row["R"] for value in line),
                    " ".join(f"{value:.12g}" for value in row["t"]),
                    f"{row['time']:.12g}",
                ]
            )

    report = {
        "schema_version": 1,
        "kind": "bop19_rtdetr_pose_export",
        "protocol": {
            "task": "BOP19 6D object localization",
            "conditioning": "official target object identity",
            "test_ground_truth_used_for_inference": False,
            "target_query_selection": "top class-probability queries per target inst_count",
            "translation_unit": "millimetre",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "device": str(args.device),
        },
        "inputs": {
            "bop_root": str(bop_root),
            "split": str(args.split),
            "targets": str(targets_path),
            "targets_sha256": _sha256(targets_path),
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_compatibility": checkpoint_report,
        },
        "output": {
            "csv": str(output_path),
            "csv_sha256": _sha256(output_path),
            "rows": len(rows),
            "images": len(groups),
            "target_entries": sum(len(items) for _, items in groups),
            "target_instances": sum(
                max(1, int(item.get("inst_count", 1)))
                for _, items in groups
                for item in items
            ),
        },
        "resources": {
            "wall_seconds": float(time.perf_counter() - started),
            "peak_rss_bytes": _peak_rss_bytes(),
        },
        "licenses": {
            "dataset": "CC-BY-4.0",
            "model_implementation": "Apache-2.0",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
