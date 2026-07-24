from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Mapping
from xml.sax.saxutils import escape


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest
from yolozu.datasets.coco import COCO_80_CATEGORY_IDS, COCO_80_CLASSES
from yolozu.predictions import validate_predictions_payload
from yolozu.predictions.predictions import CURRENT_ENTRY_SCHEMA_VERSION
from yolozu.predictions.schema_governance import CURRENT_SCHEMA_VERSION


CASE_STUDY_ID = "torchvision-maskrcnn-eager-torchscript-v1"
DEFAULT_OUTPUT_DIR = "docs/assets/case_studies/maskrcnn_eager_torchscript"
OFFICIAL_WEIGHTS_SHA256 = "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e"
ARTIFACT_NAMES = (
    "checksums.sha256",
    "commands.json",
    "comparison.svg",
    "environment.json",
    "eval_eager.json",
    "eval_torchscript.json",
    "parity.json",
    "predictions_eager.json",
    "predictions_torchscript.json",
    "protocol.json",
    "reproduction_check.json",
    "summary.json",
)
REQUIRED_ARTIFACT_NAMES = frozenset(ARTIFACT_NAMES) - {
    "checksums.sha256",
    "reproduction_check.json",
}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one real Torchvision Mask R-CNN checkpoint through PyTorch eager and "
            "TorchScript on the same images, then evaluate both wrapped prediction "
            "artifacts through YOLOZU's stable COCO evaluation lane."
        )
    )
    parser.add_argument("--dataset", default="data/smoke", help="YOLO-format dataset root.")
    parser.add_argument("--split", default="val", help="Dataset split under images/ and labels/.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=2,
        help="Number of deterministically ordered images to evaluate (default: 2).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Post-model score threshold applied identically to both outputs (default: 0.5).",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=20,
        help="Maximum retained detections per image and runtime (default: 20).",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Torch seed (default: 2026).")
    parser.add_argument("--threads", type=int, default=1, help="CPU inference threads (default: 1).")
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "Optional local official Mask R-CNN v2 state_dict. Its full SHA-256 is always "
            "checked against the pinned COCO_V1 checkpoint. Without this flag, the Torch Hub "
            "cache is used; a cache miss requires --allow-download."
        ),
    )
    parser.add_argument(
        "--expected-weights-sha256",
        default=None,
        help="Optional explicit assertion of the pinned official COCO_V1 full SHA-256.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading the official Torchvision checkpoint on a cache miss.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            f"Artifact directory (default: {DEFAULT_OUTPUT_DIR}). An existing bundle is "
            "used as an implicit baseline and is replaced only when reproduction checks pass."
        ),
    )
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help=(
            "Optional prior artifact directory to verify a clean reproduction against. "
            "May equal --output-dir because generation uses a separate staging directory."
        ),
    )
    parser.add_argument(
        "--metric-atol",
        type=float,
        default=1e-8,
        help="Absolute metric tolerance for --baseline-dir comparison (default: 1e-8).",
    )
    return parser.parse_args(argv)


def _resolve_repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _clean_known_artifacts(output_dir: Path) -> None:
    if output_dir.is_symlink():
        raise SystemExit(f"--output-dir must not be a symlink: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_NAMES:
        path = output_dir / name
        if path.is_dir():
            raise SystemExit(f"refusing to replace artifact directory: {path}")
        if path.exists() or path.is_symlink():
            path.unlink()


def _publish_staged_artifacts(*, staging_dir: Path, output_dir: Path) -> None:
    if output_dir.is_symlink():
        raise SystemExit(f"--output-dir must not be a symlink: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise SystemExit(f"--output-dir must be a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_NAMES:
        destination = output_dir / name
        if destination.is_dir():
            raise SystemExit(f"refusing to replace artifact directory: {destination}")
    published_names = {
        name for name in ARTIFACT_NAMES if name != "checksums.sha256" and (staging_dir / name).is_file()
    }
    for name in sorted(published_names):
        destination = output_dir / name
        os.replace(staging_dir / name, destination)
    for name in ARTIFACT_NAMES:
        if name in published_names or name == "checksums.sha256":
            continue
        stale = output_dir / name
        if stale.exists() or stale.is_symlink():
            stale.unlink()
    os.replace(staging_dir / "checksums.sha256", output_dir / "checksums.sha256")


def _publish_after_checks(
    *,
    staging_dir: Path,
    output_dir: Path,
    parity_ok: bool,
    reproduction_ok: bool,
) -> int:
    if not parity_ok or not reproduction_ok:
        print("case-study checks failed; existing output bundle was preserved", file=sys.stderr)
        return 2
    _publish_staged_artifacts(staging_dir=staging_dir, output_dir=output_dir)
    print(output_dir)
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.max_images) < 1:
        raise SystemExit("--max-images must be >= 1")
    if not 0.0 <= float(args.score_threshold) <= 1.0:
        raise SystemExit("--score-threshold must be in [0, 1]")
    if int(args.max_detections) < 1:
        raise SystemExit("--max-detections must be >= 1")
    if int(args.threads) < 1:
        raise SystemExit("--threads must be >= 1")
    if not math.isfinite(float(args.metric_atol)) or float(args.metric_atol) < 0.0:
        raise SystemExit("--metric-atol must be finite and >= 0")
    if args.expected_weights_sha256 is not None:
        value = str(args.expected_weights_sha256).strip().lower()
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise SystemExit("--expected-weights-sha256 must be a full 64-character hexadecimal SHA-256")
        if value != OFFICIAL_WEIGHTS_SHA256:
            raise SystemExit(
                "--expected-weights-sha256 must match the pinned official COCO_V1 checkpoint: "
                f"{OFFICIAL_WEIGHTS_SHA256}"
            )
        args.expected_weights_sha256 = value


def _dataset_class_mapping(*, dataset_root: Path, split: str) -> dict[str, Any]:
    classes_path = dataset_root / "labels" / split / "classes.json"
    if not classes_path.is_file():
        raise SystemExit(
            "the runtime case study requires labels/<split>/classes.json with the standard "
            f"COCO80 class order: {classes_path}"
        )
    try:
        payload = json.loads(classes_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to read COCO80 classes mapping: {classes_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"COCO80 classes mapping must be a JSON object: {classes_path}")

    names = payload.get("names")
    if names != list(COCO_80_CLASSES):
        raise SystemExit(
            "dataset class order does not match the standard contiguous COCO80 order "
            f"declared by YOLOZU: {classes_path}"
        )
    expected_category_to_class = {
        str(category_id): class_id for class_id, category_id in enumerate(COCO_80_CATEGORY_IDS)
    }
    expected_class_to_category = {
        str(class_id): category_id for class_id, category_id in enumerate(COCO_80_CATEGORY_IDS)
    }
    if payload.get("category_id_to_class_id") != expected_category_to_class:
        raise SystemExit(f"dataset category_id_to_class_id is not the standard sparse COCO mapping: {classes_path}")
    class_to_category = payload.get("class_to_category_id", payload.get("class_id_to_category_id"))
    if class_to_category != expected_class_to_category:
        raise SystemExit(f"dataset class_to_category_id is not the standard sparse COCO mapping: {classes_path}")

    mapping = {
        "path": _display_path(classes_path),
        "sha256": _sha256(classes_path),
        "class_count": len(names),
        "class_names": names,
        "category_ids_by_class_id": list(COCO_80_CATEGORY_IDS),
    }
    mapping["semantic_sha256"] = _json_hash(
        {
            "class_names": mapping["class_names"],
            "category_ids_by_class_id": mapping["category_ids_by_class_id"],
        }
    )
    return mapping


def _selected_inputs(
    *,
    dataset_root: Path,
    split: str,
    max_images: int,
) -> tuple[str, list[dict[str, Any]], str]:
    manifest = build_manifest(dataset_root, split=split)
    split_effective = str(manifest["split"])
    records = list(manifest["images"])[: int(max_images)]
    if len(records) != int(max_images):
        raise SystemExit(
            f"dataset split contains fewer than --max-images records: requested={max_images}, found={len(records)}"
        )

    images_base = dataset_root / "images" / split_effective
    labels_base = dataset_root / "labels" / split_effective
    selected: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        image_path = Path(str(record.get("image") or "")).expanduser()
        if not image_path.is_absolute():
            image_path = (repo_root / image_path).resolve()
        if not image_path.is_file():
            raise SystemExit(f"selected image not found: {image_path}")
        try:
            relative_image = image_path.relative_to(images_base)
        except ValueError as exc:
            raise SystemExit(f"selected image escaped dataset split: {image_path}") from exc
        label_path = labels_base / relative_image.with_suffix(".txt")
        if not label_path.is_file():
            raise SystemExit(f"selected label not found: {label_path}")
        selected.append(
            {
                "index": int(index),
                "image": _display_path(image_path),
                "image_path": image_path,
                "image_sha256": _sha256(image_path),
                "label": _display_path(label_path),
                "label_sha256": _sha256(label_path),
                "labels": record.get("labels") or [],
            }
        )

    hash_view = [
        {
            "image": item["image"],
            "image_sha256": item["image_sha256"],
            "label": item["label"],
            "label_sha256": item["label_sha256"],
        }
        for item in selected
    ]
    return split_effective, selected, _json_hash(hash_view)


def _load_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        import torchvision
        from PIL import Image
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_V2_Weights,
            maskrcnn_resnet50_fpn_v2,
        )
    except Exception as exc:
        raise SystemExit(
            "torch, torchvision, and Pillow are required; install the documented yolozu demo and coco extras"
        ) from exc
    return torch, torchvision, Image, MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2


def _official_weights(
    *,
    torch: Any,
    weights_enum: Any,
    local_weights: str | None,
    allow_download: bool,
    expected_sha256: str | None,
) -> tuple[Mapping[str, Any], Path, dict[str, Any]]:
    weights = weights_enum.COCO_V1
    url = str(weights.url)
    if local_weights is not None:
        weights_path = _resolve_repo_path(local_weights)
        if not weights_path.is_file():
            raise SystemExit(f"--weights file not found: {weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise SystemExit(f"safe checkpoint load failed for --weights: {exc}") from exc
        source = "local"
    else:
        filename = url.rsplit("/", 1)[-1]
        cache_dir = Path(torch.hub.get_dir()).expanduser().resolve() / "checkpoints"
        weights_path = cache_dir / filename
        cache_hit = weights_path.is_file()
        if not cache_hit and not bool(allow_download):
            raise SystemExit(
                "official checkpoint is not in the Torch Hub cache; pass --weights /path/to/state_dict "
                "or explicitly permit the official download with --allow-download"
            )
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                url,
                model_dir=str(cache_dir),
                map_location="cpu",
                progress=False,
                check_hash=True,
                weights_only=True,
            )
        except Exception as exc:
            raise SystemExit(f"official checkpoint load failed: {exc}") from exc
        source = "torch_hub_cache" if cache_hit else "official_download"

    if not isinstance(state_dict, Mapping):
        raise SystemExit("checkpoint must contain a state_dict mapping")
    weights_sha256 = _sha256(weights_path)
    if weights_sha256 != OFFICIAL_WEIGHTS_SHA256:
        raise SystemExit(
            "checkpoint is not the pinned official COCO_V1 state_dict: "
            f"expected={OFFICIAL_WEIGHTS_SHA256}, actual={weights_sha256}, path={weights_path}"
        )
    if expected_sha256 is not None and weights_sha256 != expected_sha256:
        raise SystemExit(
            "checkpoint SHA-256 mismatch: "
            f"expected={expected_sha256}, actual={weights_sha256}, path={weights_path}"
        )
    categories = [str(value) for value in weights.meta.get("categories", [])]
    if len(categories) != 91:
        raise SystemExit(f"unexpected official category metadata length: {len(categories)}")
    return state_dict, weights_path, {
        "enum": "MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
        "source": source,
        "url": url,
        "path": str(weights_path),
        "sha256": weights_sha256,
        "categories": categories,
    }


def _category_id_map(categories: list[str]) -> dict[int, int]:
    valid_indices = [
        index
        for index, name in enumerate(categories)
        if index != 0 and str(name).strip().lower() not in {"n/a", "__background__"}
    ]
    if len(valid_indices) != 80:
        raise ValueError(f"expected 80 COCO categories after removing metadata gaps, found {len(valid_indices)}")
    ordered_names = [categories[index] for index in valid_indices]
    if ordered_names != list(COCO_80_CLASSES):
        raise ValueError("Torchvision checkpoint metadata does not match YOLOZU's standard COCO80 class order")
    if tuple(valid_indices) != COCO_80_CATEGORY_IDS:
        raise ValueError("Torchvision checkpoint metadata does not match the standard sparse COCO category ids")
    return {category_id: class_id for class_id, category_id in enumerate(valid_indices)}


def _input_tensors(*, selected: list[dict[str, Any]], torch: Any, Image: Any) -> tuple[list[Any], list[dict[str, int]]]:
    try:
        from torchvision.transforms.functional import pil_to_tensor
    except Exception as exc:
        raise SystemExit("torchvision.transforms.functional.pil_to_tensor is required") from exc

    tensors: list[Any] = []
    sizes: list[dict[str, int]] = []
    for item in selected:
        with Image.open(item["image_path"]) as handle:
            image = handle.convert("RGB")
            width, height = image.size
            tensor = pil_to_tensor(image).to(dtype=torch.float32).div(255.0)
        tensors.append(tensor)
        sizes.append({"width": int(width), "height": int(height)})
    return tensors, sizes


def _scripted_detections(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, tuple) or len(raw) != 2:
        raise RuntimeError("scripted Mask R-CNN must return the documented (losses, detections) tuple")
    detections = raw[1]
    if not isinstance(detections, list):
        raise RuntimeError("scripted Mask R-CNN detections output must be a list")
    return detections


def _to_predictions(
    *,
    outputs: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    sizes: list[dict[str, int]],
    category_map: dict[int, int],
    score_threshold: float,
    max_detections: int,
) -> list[dict[str, Any]]:
    if len(outputs) != len(selected):
        raise RuntimeError(f"runtime result count mismatch: results={len(outputs)}, inputs={len(selected)}")
    entries: list[dict[str, Any]] = []
    for output, item, size in zip(outputs, selected, sizes):
        for key in ("boxes", "labels", "scores"):
            if key not in output:
                raise RuntimeError(f"runtime result is missing {key!r}")
        boxes = output["boxes"].detach().cpu().tolist()
        labels = output["labels"].detach().cpu().tolist()
        scores = output["scores"].detach().cpu().tolist()
        if not (len(boxes) == len(labels) == len(scores)):
            raise RuntimeError("runtime boxes/labels/scores lengths differ")

        width = int(size["width"])
        height = int(size["height"])
        detections: list[dict[str, Any]] = []
        for box, raw_label, raw_score in zip(boxes, labels, scores):
            score = float(raw_score)
            if score < float(score_threshold):
                continue
            label = int(raw_label)
            if label not in category_map:
                raise RuntimeError(f"runtime emitted an unmapped COCO category id: {label}")
            if not isinstance(box, list) or len(box) != 4:
                raise RuntimeError(f"runtime emitted an invalid xyxy box: {box!r}")
            x1, y1, x2, y2 = (float(value) for value in box)
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2, score)):
                raise RuntimeError("runtime emitted a non-finite detection")
            x1 = max(0.0, min(float(width), x1))
            x2 = max(0.0, min(float(width), x2))
            y1 = max(0.0, min(float(height), y1))
            y2 = max(0.0, min(float(height), y2))
            if x2 <= x1 or y2 <= y1:
                raise RuntimeError(f"runtime emitted a degenerate box after bounds normalization: {box!r}")
            detections.append(
                {
                    "class_id": int(category_map[label]),
                    "score": score,
                    "bbox": {
                        "cx": ((x1 + x2) / 2.0) / float(width),
                        "cy": ((y1 + y2) / 2.0) / float(height),
                        "w": (x2 - x1) / float(width),
                        "h": (y2 - y1) / float(height),
                    },
                }
            )
            if len(detections) >= int(max_detections):
                break
        entries.append(
            {
                "schema_version": int(CURRENT_ENTRY_SCHEMA_VERSION),
                "image": str(item["image"]),
                "detections": detections,
            }
        )
    return entries


def _default_tta() -> dict[str, Any]:
    return {
        "enabled": False,
        "seed": None,
        "flip_prob": 0.0,
        "norm_only": False,
        "warnings": [],
        "summary": None,
    }


def _default_ttt() -> dict[str, Any]:
    return {
        "enabled": False,
        "method": "none",
        "steps": 0,
        "batch_size": 0,
        "lr": 0.0,
        "update_filter": "none",
        "include": None,
        "exclude": None,
        "max_batches": 0,
        "seed": None,
        "mim": {"mask_prob": 0.0, "patch_size": 0, "mask_value": 0.0},
        "report": None,
    }


def _wrapped_predictions(
    *,
    runtime: str,
    entries: list[dict[str, Any]],
    timestamp: str,
    protocol: dict[str, Any],
    weights: dict[str, Any],
    input_set_sha256: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    payload = {
        "schema_version": int(CURRENT_SCHEMA_VERSION),
        "predictions": entries,
        "meta": {
            "timestamp": timestamp,
            "adapter": f"torchvision_maskrcnn_{runtime}",
            "config": "maskrcnn_resnet50_fpn_v2",
            "images": int(len(entries)),
            "tta": _default_tta(),
            "ttt": _default_ttt(),
            "extra": {
                "case_study_id": CASE_STUDY_ID,
                "runtime": runtime,
                "runtime_executed": True,
                "execution_status": "completed",
                "inference_calls": 1,
                "result_count": int(len(entries)),
                "weights": {
                    "enum": weights["enum"],
                    "url": weights["url"],
                    "sha256": weights["sha256"],
                },
                "input_set_sha256": input_set_sha256,
                "elapsed_seconds": float(elapsed_seconds),
                "export_settings": protocol["fixed_conditions"],
            },
        },
    }
    validate_predictions_payload(payload, strict=True)
    return payload


def _run_eval(
    *,
    dataset: Path,
    split: str,
    predictions: Path,
    published_predictions: Path,
    max_images: int,
    output: Path,
) -> tuple[dict[str, Any], list[str]]:
    dataset_arg = _display_path(dataset)
    predictions_arg = _display_path(predictions)
    output_arg = _display_path(output)
    command = [
        sys.executable,
        str(repo_root / "tools" / "eval_coco.py"),
        "--dataset",
        dataset_arg,
        "--split",
        split,
        "--predictions",
        predictions_arg,
        "--bbox-format",
        "cxcywh_norm",
        "--max-images",
        str(int(max_images)),
        "--output",
        output_arg,
    ]
    result = subprocess.run(command, cwd=repo_root, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        if output.exists():
            output.unlink()
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"stable eval-coco lane failed for {predictions.name}: {detail}")
    report = _read_json_object(output)
    report["dataset"] = _display_path(dataset)
    report["predictions"] = _display_path(published_predictions)
    _write_json(output, report)
    return report, command


def _run_parity(
    *,
    reference: Path,
    candidate: Path,
    published_reference: Path,
    published_candidate: Path,
    max_images: int,
    output: Path,
) -> tuple[dict[str, Any], list[str], int]:
    command = [
        sys.executable,
        str(repo_root / "tools" / "check_predictions_parity.py"),
        "--reference",
        _display_path(reference),
        "--candidate",
        _display_path(candidate),
        "--bbox-format",
        "cxcywh_norm",
        "--iou-thresh",
        "0.999999",
        "--score-atol",
        "1e-7",
        "--bbox-atol",
        "1e-7",
        "--max-images",
        str(int(max_images)),
    ]
    result = subprocess.run(command, cwd=repo_root, text=True, capture_output=True, check=False)
    if result.returncode not in (0, 2):
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"parity comparison failed without a report: {detail}")
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"parity comparison returned invalid JSON: {result.stdout!r}") from exc
    report["reference"] = _display_path(published_reference)
    report["candidate"] = _display_path(published_candidate)
    _write_json(output, report)
    return report, command, int(result.returncode)


def _metrics(report: dict[str, Any]) -> dict[str, float]:
    values = report.get("metrics")
    if not isinstance(values, dict):
        raise RuntimeError("evaluation report is missing metrics")
    out: dict[str, float] = {}
    for key in ("map50_95", "map50", "map75", "ar100"):
        value = values.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise RuntimeError(f"evaluation metric {key} is not finite: {value!r}")
        out[key] = float(value)
    return out


def _comparison_svg(*, eager: dict[str, float], scripted: dict[str, float], output: Path) -> None:
    labels = [
        ("map50_95", "mAP 50-95"),
        ("map50", "mAP 50"),
        ("map75", "mAP 75"),
        ("ar100", "AR 100"),
    ]
    width, height = 880, 500
    plot_left, plot_top, plot_width, plot_height = 90, 105, 720, 280
    group_width = plot_width / len(labels)
    bar_width = 42
    bar_center_offset = 32
    label_center_offset = 44
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1220"/>',
        '<text x="44" y="48" fill="#f8fafc" font-family="system-ui,sans-serif" font-size="24" font-weight="700">'
        + "Mask R-CNN: eager and TorchScript</text>",
        '<text x="44" y="76" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="14">'
        + "Same checkpoint, images, preprocessing, filtering, and YOLOZU evaluation lane</text>",
    ]
    for tick in range(6):
        value = tick / 5
        y = plot_top + plot_height - value * plot_height
        lines.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_left + plot_width}" y2="{y:.1f}" '
            'stroke="#263449" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{plot_left - 14}" y="{y + 5:.1f}" text-anchor="end" fill="#94a3b8" '
            f'font-family="system-ui,sans-serif" font-size="12">{value:.1f}</text>'
        )
    for index, (key, label) in enumerate(labels):
        center = plot_left + group_width * (index + 0.5)
        for bar_offset, label_offset, values, color in (
            (-bar_center_offset, -label_center_offset, eager, "#38bdf8"),
            (bar_center_offset, label_center_offset, scripted, "#a78bfa"),
        ):
            value = max(0.0, min(1.0, float(values[key])))
            bar_height = value * plot_height
            x = center + bar_offset - bar_width / 2
            y = plot_top + plot_height - bar_height
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" '
                f'rx="5" fill="{color}"/>'
            )
            lines.append(
                f'<text x="{center + label_offset:.1f}" y="{max(plot_top + 14, y - 7):.1f}" '
                f'text-anchor="middle" fill="#e2e8f0" font-family="system-ui,sans-serif" '
                f'font-size="11">{value:.6f}</text>'
            )
        lines.append(
            f'<text x="{center:.1f}" y="{plot_top + plot_height + 28}" text-anchor="middle" '
            f'fill="#cbd5e1" font-family="system-ui,sans-serif" font-size="13">{escape(label)}</text>'
        )
    lines.extend(
        [
            '<rect x="246" y="450" width="14" height="14" rx="3" fill="#38bdf8"/>',
            '<text x="268" y="462" fill="#cbd5e1" font-family="system-ui,sans-serif" font-size="13">PyTorch eager</text>',
            '<rect x="442" y="450" width="14" height="14" rx="3" fill="#a78bfa"/>',
            '<text x="464" y="462" fill="#cbd5e1" font-family="system-ui,sans-serif" font-size="13">TorchScript</text>',
            '<text x="836" y="488" text-anchor="end" fill="#64748b" font-family="system-ui,sans-serif" font-size="11">'
            + "Descriptive two-image case study; not a performance ranking.</text>",
            "</svg>",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to read case-study artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"case-study artifact must be a JSON object: {path}")
    return value


def _environment_identity(environment: dict[str, Any]) -> dict[str, Any]:
    python_version = environment.get("python")
    if isinstance(python_version, str):
        python_parts = python_version.split(".")
        python_version = ".".join(python_parts[:2]) if len(python_parts) >= 2 else python_version

    packages = environment.get("packages")
    if isinstance(packages, dict):
        packages = {
            key: value.split("+", 1)[0] if isinstance(value, str) else value
            for key, value in packages.items()
            if key != "yolozu"
        }
    torch_state = environment.get("torch")
    if isinstance(torch_state, dict):
        torch_state = {
            key: torch_state.get(key)
            for key in ("device", "threads", "deterministic_algorithms")
        }
    return {
        "python_major_minor": python_version,
        "packages": packages,
        "torch": torch_state,
    }


def _source_identity(source: Any) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {"available": False, "head": None, "file_sha256": {}}
    hashes = source.get("file_sha256")
    if not isinstance(hashes, dict):
        hashes = {}
    return {
        "available": bool(source.get("available")),
        "head": source.get("head"),
        "file_sha256": {str(key): str(value) for key, value in sorted(hashes.items())},
    }


def _parity_identity(parity: dict[str, Any]) -> dict[str, Any]:
    return {
        key: parity.get(key)
        for key in ("bbox_atol", "bbox_format", "images", "iou_thresh", "ok", "results", "score_atol")
    }


def _artifact_identity(artifact_dir: Path) -> dict[str, Any]:
    summary_path = artifact_dir / "summary.json"
    summary = _read_json_object(summary_path)
    protocol = _read_json_object(artifact_dir / "protocol.json")
    environment = _read_json_object(artifact_dir / "environment.json")
    parity = _read_json_object(artifact_dir / "parity.json")
    eager = _read_json_object(artifact_dir / "predictions_eager.json")
    scripted = _read_json_object(artifact_dir / "predictions_torchscript.json")
    source = _source_identity(summary.get("source"))
    environment_view = _environment_identity(environment)
    return {
        "case_study_id": summary.get("case_study_id"),
        "summary_sha256": _sha256(summary_path),
        "input_set_sha256": summary.get("inputs", {}).get("input_set_sha256"),
        "weights_sha256": summary.get("weights", {}).get("sha256"),
        "protocol_sha256": _json_hash(protocol),
        "environment_sha256": _json_hash(environment_view),
        "environment": environment_view,
        "source_sha256": _json_hash(source["file_sha256"]),
        "source": source,
        "source_clean_before_run": summary.get("source", {}).get("clean_before_run"),
        "parity_sha256": _json_hash(_parity_identity(parity)),
        "predictions": {
            "eager_sha256": _json_hash(eager.get("predictions")),
            "torchscript_sha256": _json_hash(scripted.get("predictions")),
        },
    }


def _verify_checksum_manifest(artifact_dir: Path) -> dict[str, Any]:
    checksum_path = artifact_dir / "checksums.sha256"
    errors: list[str] = []
    entries: dict[str, str] = {}
    if not checksum_path.is_file():
        return {
            "ok": False,
            "manifest_sha256": None,
            "entries": entries,
            "errors": ["missing checksums.sha256"],
        }
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            digest, filename = line.split("  ", 1)
        except ValueError:
            errors.append(f"line {line_number}: expected '<sha256>  <filename>'")
            continue
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            errors.append(f"line {line_number}: invalid SHA-256")
            continue
        if Path(filename).name != filename or filename == "checksums.sha256":
            errors.append(f"line {line_number}: invalid artifact filename {filename!r}")
            continue
        if filename in entries:
            errors.append(f"line {line_number}: duplicate artifact filename {filename!r}")
            continue
        entries[filename] = digest
        target = artifact_dir / filename
        if not target.is_file():
            errors.append(f"{filename}: missing artifact")
        elif _sha256(target) != digest:
            errors.append(f"{filename}: checksum mismatch")

    for filename in sorted(REQUIRED_ARTIFACT_NAMES - set(entries)):
        errors.append(f"{filename}: missing checksum entry")
    present_known = {
        name
        for name in ARTIFACT_NAMES
        if name != "checksums.sha256" and (artifact_dir / name).is_file()
    }
    for filename in sorted(present_known - set(entries)):
        errors.append(f"{filename}: present but not checksummed")
    return {
        "ok": not errors,
        "manifest_sha256": _sha256(checksum_path),
        "entries": entries,
        "errors": errors,
    }


def _baseline_check(
    *,
    baseline_dir: Path,
    candidate_dir: Path,
    metric_atol: float,
) -> dict[str, Any]:
    baseline_checksums = _verify_checksum_manifest(baseline_dir)
    baseline = _read_json_object(baseline_dir / "summary.json")
    candidate = _read_json_object(candidate_dir / "summary.json")
    baseline_identity = _artifact_identity(baseline_dir)
    candidate_identity = _artifact_identity(candidate_dir)
    checks: list[dict[str, Any]] = []

    def exact(name: str, actual: Any, expected: Any) -> None:
        checks.append({"name": name, "actual": actual, "expected": expected, "ok": actual == expected})

    exact("baseline_checksums_valid", baseline_checksums["ok"], True)
    exact("candidate_source_clean_before_run", candidate_identity["source_clean_before_run"], True)
    for key in (
        "case_study_id",
        "input_set_sha256",
        "weights_sha256",
        "protocol_sha256",
        "environment_sha256",
        "source_sha256",
        "parity_sha256",
        "predictions",
    ):
        exact(key, candidate_identity.get(key), baseline_identity.get(key))
    exact(
        "eager_detection_count",
        candidate.get("results", {}).get("eager", {}).get("detections"),
        baseline.get("results", {}).get("eager", {}).get("detections"),
    )
    exact(
        "torchscript_detection_count",
        candidate.get("results", {}).get("torchscript", {}).get("detections"),
        baseline.get("results", {}).get("torchscript", {}).get("detections"),
    )
    exact(
        "parity_ok",
        candidate.get("parity", {}).get("ok"),
        baseline.get("parity", {}).get("ok"),
    )
    for runtime in ("eager", "torchscript"):
        actual_metrics = candidate.get("results", {}).get(runtime, {}).get("metrics", {})
        expected_metrics = baseline.get("results", {}).get(runtime, {}).get("metrics", {})
        for key in ("map50_95", "map50", "map75", "ar100"):
            try:
                actual = float(actual_metrics.get(key))
                expected = float(expected_metrics.get(key))
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"baseline metric {runtime}.{key} must be numeric") from exc
            if not math.isfinite(actual) or not math.isfinite(expected):
                raise SystemExit(f"baseline metric {runtime}.{key} must be finite")
            delta = abs(actual - expected)
            checks.append(
                {
                    "name": f"{runtime}_{key}",
                    "actual": actual,
                    "expected": expected,
                    "absolute_delta": delta,
                    "atol": float(metric_atol),
                    "ok": delta <= float(metric_atol),
                }
            )
    return {
        "schema_version": 2,
        "baseline": baseline_identity,
        "baseline_checksums": baseline_checksums,
        "candidate": candidate_identity,
        "metric_atol": float(metric_atol),
        "checks": checks,
        "ok": all(bool(item["ok"]) for item in checks),
    }


def _portable_commands(
    *,
    args: argparse.Namespace,
    split: str,
    output_dir: Path,
    predictions_eager: Path,
    predictions_scripted: Path,
) -> dict[str, Any]:
    output_display = _display_path(output_dir)
    dataset_display = _display_path(_resolve_repo_path(args.dataset))
    generate = [
        "python3",
        "tools/generate_runtime_parity_case_study.py",
        "--dataset",
        dataset_display,
        "--split",
        split,
        "--max-images",
        str(int(args.max_images)),
        "--score-threshold",
        f"{float(args.score_threshold):g}",
        "--max-detections",
        str(int(args.max_detections)),
        "--seed",
        str(int(args.seed)),
        "--threads",
        str(int(args.threads)),
        "--output-dir",
        output_display,
    ]
    if args.weights:
        generate.extend(["--weights", _display_path(_resolve_repo_path(args.weights))])
    else:
        generate.append("--allow-download")
    if args.expected_weights_sha256:
        generate.extend(["--expected-weights-sha256", args.expected_weights_sha256])
    if args.baseline_dir:
        generate.extend(
            [
                "--baseline-dir",
                _display_path(_resolve_repo_path(args.baseline_dir)),
                "--metric-atol",
                f"{float(args.metric_atol):g}",
            ]
        )

    eager_display = _display_path(predictions_eager)
    scripted_display = _display_path(predictions_scripted)
    eager_eval = [
        "python3",
        "tools/eval_coco.py",
        "--dataset",
        dataset_display,
        "--split",
        split,
        "--predictions",
        eager_display,
        "--bbox-format",
        "cxcywh_norm",
        "--max-images",
        str(int(args.max_images)),
        "--output",
        f"{output_display}/eval_eager.json",
    ]
    scripted_eval = [
        "python3",
        "tools/eval_coco.py",
        "--dataset",
        dataset_display,
        "--split",
        split,
        "--predictions",
        scripted_display,
        "--bbox-format",
        "cxcywh_norm",
        "--max-images",
        str(int(args.max_images)),
        "--output",
        f"{output_display}/eval_torchscript.json",
    ]
    parity = [
        "python3",
        "tools/check_predictions_parity.py",
        "--reference",
        eager_display,
        "--candidate",
        scripted_display,
        "--bbox-format",
        "cxcywh_norm",
        "--iou-thresh",
        "0.999999",
        "--score-atol",
        "1e-7",
        "--bbox-atol",
        "1e-7",
        "--max-images",
        str(int(args.max_images)),
    ]
    return {
        "schema_version": 1,
        "generate": shlex.join(generate),
        "validate": [
            shlex.join(
                ["python3", "tools/validate_predictions.py", eager_display, "--strict"]
            ),
            shlex.join(
                ["python3", "tools/validate_predictions.py", scripted_display, "--strict"]
            ),
        ],
        "evaluate": [shlex.join(eager_eval), shlex.join(scripted_eval)],
        "parity": shlex.join(parity),
        "note": "Run from the YOLOZU repository root in an environment with the documented demo and coco extras.",
    }


def _artifact_index(*, include_reproduction: bool) -> dict[str, str]:
    artifacts = {
        "predictions_eager": "predictions_eager.json",
        "predictions_torchscript": "predictions_torchscript.json",
        "eval_eager": "eval_eager.json",
        "eval_torchscript": "eval_torchscript.json",
        "parity": "parity.json",
        "protocol": "protocol.json",
        "environment": "environment.json",
        "comparison": "comparison.svg",
        "commands": "commands.json",
        "checksums": "checksums.sha256",
    }
    if include_reproduction:
        artifacts["reproduction_check"] = "reproduction_check.json"
    return artifacts


def _logical_command(
    command: list[str],
    *,
    staging_dir: Path,
    published_output_dir: Path,
) -> str:
    staging_display = _display_path(staging_dir)
    published_display = _display_path(published_output_dir)
    staging_aliases = {
        str(staging_dir),
        str(staging_dir.resolve()),
        staging_display,
    }
    logical_args = []
    for value in command[1:]:
        value = value.replace(str(repo_root) + "/", "")
        for staging_alias in sorted(staging_aliases, key=len, reverse=True):
            value = value.replace(staging_alias, published_display)
        logical_args.append(value)
    return shlex.join(logical_args)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_source_state() -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    source_files = (
        "data/smoke/labels/val/classes.json",
        "tools/generate_runtime_parity_case_study.py",
        "tools/generate_smoke_assets.py",
        "tools/eval_coco.py",
        "tools/check_predictions_parity.py",
        "yolozu/api.py",
        "yolozu/datasets/coco.py",
        "yolozu/eval/coco_eval.py",
        "yolozu/predictions/predictions_parity.py",
    )
    hashes = {
        relative: _sha256(repo_root / relative)
        for relative in source_files
        if (repo_root / relative).is_file()
    }
    if status.returncode != 0:
        return {
            "available": False,
            "head": head.stdout.strip() if head.returncode == 0 else None,
            "clean_before_run": None,
            "file_sha256": hashes,
        }
    return {
        "available": True,
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "clean_before_run": not bool(status.stdout.strip()),
        "file_sha256": hashes,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _validate_args(args)
    dataset_root = _resolve_repo_path(args.dataset)
    if not dataset_root.is_dir():
        raise SystemExit(f"--dataset directory not found: {dataset_root}")
    requested_output_dir = Path(args.output_dir).expanduser()
    if not requested_output_dir.is_absolute():
        requested_output_dir = repo_root / requested_output_dir
    if requested_output_dir.is_symlink():
        raise SystemExit(f"--output-dir must not be a symlink: {requested_output_dir}")
    published_output_dir = requested_output_dir.resolve()
    if published_output_dir.exists() and not published_output_dir.is_dir():
        raise SystemExit(f"--output-dir must be a directory: {published_output_dir}")
    baseline_dir = _resolve_repo_path(args.baseline_dir) if args.baseline_dir is not None else None
    baseline_mode = "explicit" if baseline_dir is not None else None
    if baseline_dir is None and published_output_dir.is_dir():
        has_existing_bundle = any(
            (published_output_dir / name).exists() for name in ARTIFACT_NAMES
        )
        if has_existing_bundle:
            baseline_dir = published_output_dir
            baseline_mode = "implicit_existing_output"
    source_state = _git_source_state()
    published_output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{published_output_dir.name}.staging-",
        dir=published_output_dir.parent,
    ) as staging_dir:
        return _generate_case_study(
            args=args,
            dataset_root=dataset_root,
            published_output_dir=published_output_dir,
            source_state=source_state,
            output_dir=Path(staging_dir),
            baseline_dir=baseline_dir,
            baseline_mode=baseline_mode,
        )


def _generate_case_study(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    published_output_dir: Path,
    source_state: dict[str, Any],
    output_dir: Path,
    baseline_dir: Path | None,
    baseline_mode: str | None,
) -> int:
    _clean_known_artifacts(output_dir)

    split, selected, records_sha256 = _selected_inputs(
        dataset_root=dataset_root,
        split=str(args.split),
        max_images=int(args.max_images),
    )
    class_mapping = _dataset_class_mapping(dataset_root=dataset_root, split=split)
    input_set_sha256 = _json_hash(
        {
            "records_sha256": records_sha256,
            "classes_sha256": class_mapping["sha256"],
            "class_semantic_sha256": class_mapping["semantic_sha256"],
        }
    )
    torch, torchvision, Image, weights_enum, model_factory = _load_dependencies()
    torch.manual_seed(int(args.seed))
    torch.set_num_threads(int(args.threads))
    torch.use_deterministic_algorithms(True, warn_only=True)

    state_dict, weights_path, weights_meta = _official_weights(
        torch=torch,
        weights_enum=weights_enum,
        local_weights=args.weights,
        allow_download=bool(args.allow_download),
        expected_sha256=args.expected_weights_sha256,
    )
    model = model_factory(weights=None, weights_backbone=None)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "official checkpoint is incompatible with Mask R-CNN v2: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.to(torch.device("cpu"))
    model.eval()

    category_map = _category_id_map(weights_meta["categories"])
    tensors, sizes = _input_tensors(selected=selected, torch=torch, Image=Image)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    started = time.perf_counter()
    with torch.inference_mode():
        eager_outputs = model(tensors)
    eager_seconds = time.perf_counter() - started

    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="RCNN always returns a .* tuple in scripting")
        scripted_model = torch.jit.script(model)
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with torch.inference_mode():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="RCNN always returns a .* tuple in scripting")
            scripted_raw = scripted_model(tensors)
    scripted_seconds = time.perf_counter() - started
    scripted_outputs = _scripted_detections(scripted_raw)

    eager_entries = _to_predictions(
        outputs=eager_outputs,
        selected=selected,
        sizes=sizes,
        category_map=category_map,
        score_threshold=float(args.score_threshold),
        max_detections=int(args.max_detections),
    )
    scripted_entries = _to_predictions(
        outputs=scripted_outputs,
        selected=selected,
        sizes=sizes,
        category_map=category_map,
        score_threshold=float(args.score_threshold),
        max_detections=int(args.max_detections),
    )

    transform = model.transform
    roi_heads = model.roi_heads
    protocol = {
        "schema_version": 1,
        "id": CASE_STUDY_ID,
        "task": "detection",
        "evaluation_lane": "tools/eval_coco.py",
        "canonical_protocol_preset": None,
        "canonical_protocol_preset_reason": (
            "Torchvision applies its embedded GeneralizedRCNNTransform rather than the "
            "640-pixel letterbox preprocessing pinned by YOLOZU's named COCO presets."
        ),
        "dataset": _display_path(dataset_root),
        "split": split,
        "fixed_conditions": {
            "checkpoint": weights_meta["enum"],
            "checkpoint_sha256": weights_meta["sha256"],
            "device": "cpu",
            "seed": int(args.seed),
            "threads": int(args.threads),
            "dataset_class_mapping": class_mapping,
            "input_color": "RGB",
            "external_preprocess": "PIL RGB to float32 tensor in [0,1]; no external resize",
            "model_transform": {
                "min_size": [int(value) for value in transform.min_size],
                "max_size": int(transform.max_size),
                "image_mean": [float(value) for value in transform.image_mean],
                "image_std": [float(value) for value in transform.image_std],
            },
            "model_postprocess": {
                "score_threshold": float(roi_heads.score_thresh),
                "nms_iou_threshold": float(roi_heads.nms_thresh),
                "detections_per_image": int(roi_heads.detections_per_img),
            },
            "export_filter": {
                "score_threshold": float(args.score_threshold),
                "max_detections": int(args.max_detections),
                "bbox_format": "cxcywh_norm",
                "class_id_space": "COCO 80-class contiguous index",
            },
            "evaluation": {
                "max_images": int(args.max_images),
                "bbox_format": "cxcywh_norm",
            },
        },
    }
    _write_json(output_dir / "protocol.json", protocol)

    eager_payload = _wrapped_predictions(
        runtime="eager",
        entries=eager_entries,
        timestamp=timestamp,
        protocol=protocol,
        weights=weights_meta,
        input_set_sha256=input_set_sha256,
        elapsed_seconds=eager_seconds,
    )
    scripted_payload = _wrapped_predictions(
        runtime="torchscript",
        entries=scripted_entries,
        timestamp=timestamp,
        protocol=protocol,
        weights=weights_meta,
        input_set_sha256=input_set_sha256,
        elapsed_seconds=scripted_seconds,
    )
    predictions_eager = output_dir / "predictions_eager.json"
    predictions_scripted = output_dir / "predictions_torchscript.json"
    _write_json(predictions_eager, eager_payload)
    _write_json(predictions_scripted, scripted_payload)

    eval_eager, eval_eager_command = _run_eval(
        dataset=dataset_root,
        split=split,
        predictions=predictions_eager,
        published_predictions=published_output_dir / "predictions_eager.json",
        max_images=int(args.max_images),
        output=output_dir / "eval_eager.json",
    )
    eval_scripted, eval_scripted_command = _run_eval(
        dataset=dataset_root,
        split=split,
        predictions=predictions_scripted,
        published_predictions=published_output_dir / "predictions_torchscript.json",
        max_images=int(args.max_images),
        output=output_dir / "eval_torchscript.json",
    )
    parity, parity_command, parity_exit = _run_parity(
        reference=predictions_eager,
        candidate=predictions_scripted,
        published_reference=published_output_dir / "predictions_eager.json",
        published_candidate=published_output_dir / "predictions_torchscript.json",
        max_images=int(args.max_images),
        output=output_dir / "parity.json",
    )
    eager_metrics = _metrics(eval_eager)
    scripted_metrics = _metrics(eval_scripted)
    metric_deltas = {
        key: float(scripted_metrics[key] - eager_metrics[key])
        for key in ("map50_95", "map50", "map75", "ar100")
    }

    environment = {
        "schema_version": 1,
        "timestamp": timestamp,
        "python": platform.python_version(),
        "python_executable_name": Path(sys.executable).name,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {
            "yolozu": _package_version("yolozu"),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "Pillow": _package_version("Pillow"),
            "numpy": _package_version("numpy"),
            "pycocotools": _package_version("pycocotools"),
        },
        "torch": {
            "device": "cpu",
            "threads": int(torch.get_num_threads()),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        },
        "source": source_state,
    }
    _write_json(output_dir / "environment.json", environment)

    summary = {
        "schema_version": 1,
        "case_study_id": CASE_STUDY_ID,
        "timestamp": timestamp,
        "status": "completed",
        "runtime_execution": {"eager": True, "torchscript": True},
        "source": source_state,
        "inputs": {
            "dataset": _display_path(dataset_root),
            "split": split,
            "count": int(len(selected)),
            "input_set_sha256": input_set_sha256,
            "records_sha256": records_sha256,
            "class_mapping": {
                "path": class_mapping["path"],
                "sha256": class_mapping["sha256"],
                "semantic_sha256": class_mapping["semantic_sha256"],
            },
            "records": [
                {
                    "image": item["image"],
                    "image_sha256": item["image_sha256"],
                    "label": item["label"],
                    "label_sha256": item["label_sha256"],
                }
                for item in selected
            ],
        },
        "weights": {
            "enum": weights_meta["enum"],
            "source": weights_meta["source"],
            "url": weights_meta["url"],
            "filename": weights_path.name,
            "sha256": weights_meta["sha256"],
        },
        "results": {
            "eager": {
                "inference_seconds": float(eager_seconds),
                "detections": int(sum(len(item["detections"]) for item in eager_entries)),
                "metrics": eager_metrics,
            },
            "torchscript": {
                "compile_seconds": float(compile_seconds),
                "inference_seconds": float(scripted_seconds),
                "detections": int(sum(len(item["detections"]) for item in scripted_entries)),
                "metrics": scripted_metrics,
            },
        },
        "metric_deltas_torchscript_minus_eager": metric_deltas,
        "parity": {
            "ok": bool(parity.get("ok")),
            "exit_code": int(parity_exit),
            "images": int(parity.get("images", 0)),
        },
        "interpretation": (
            "The two real runtime paths produced parity-equivalent exported detections "
            "under the pinned case-study conditions."
            if bool(parity.get("ok"))
            else "The two runtime paths produced a recorded output difference under the pinned conditions."
        ),
        "performance_claim": (
            "Inference times are single-run environment observations and are not used as a runtime ranking."
        ),
        "artifacts": _artifact_index(include_reproduction=baseline_dir is not None),
        "execution_evidence": {
            "eval_eager_exit_code": 0,
            "eval_torchscript_exit_code": 0,
            "parity_exit_code": int(parity_exit),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    commands = _portable_commands(
        args=args,
        split=split,
        output_dir=published_output_dir,
        predictions_eager=published_output_dir / "predictions_eager.json",
        predictions_scripted=published_output_dir / "predictions_torchscript.json",
    )
    commands["executed_by_generator"] = {
        "eval_eager": _logical_command(
            eval_eager_command,
            staging_dir=output_dir,
            published_output_dir=published_output_dir,
        ),
        "eval_torchscript": _logical_command(
            eval_scripted_command,
            staging_dir=output_dir,
            published_output_dir=published_output_dir,
        ),
        "parity": _logical_command(
            parity_command,
            staging_dir=output_dir,
            published_output_dir=published_output_dir,
        ),
    }
    commands["execution_path_note"] = (
        "Commands use the final logical artifact paths; files were generated in a sibling "
        "staging directory and published only after all checks passed."
    )
    _write_json(output_dir / "commands.json", commands)
    _comparison_svg(eager=eager_metrics, scripted=scripted_metrics, output=output_dir / "comparison.svg")

    reproduction_ok = True
    if baseline_dir is not None:
        reproduction = _baseline_check(
            baseline_dir=baseline_dir,
            candidate_dir=output_dir,
            metric_atol=float(args.metric_atol),
        )
        reproduction["baseline_mode"] = baseline_mode
        _write_json(output_dir / "reproduction_check.json", reproduction)
        reproduction_ok = bool(reproduction["ok"])

    checksum_paths = [
        output_dir / name
        for name in ARTIFACT_NAMES
        if name != "checksums.sha256" and (output_dir / name).is_file()
    ]
    checksum_lines = [f"{_sha256(path)}  {path.name}" for path in sorted(checksum_paths, key=lambda value: value.name)]
    (output_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    return _publish_after_checks(
        staging_dir=output_dir,
        output_dir=published_output_dir,
        parity_ok=bool(parity.get("ok")),
        reproduction_ok=reproduction_ok,
    )


if __name__ == "__main__":
    raise SystemExit(main())
