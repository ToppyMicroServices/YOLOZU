from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from yolozu.core.boxes import iou_cxcywh_norm_dict
from yolozu.core.cli_args import resolve_output_path
from yolozu.core.image_keys import require_image_key
from yolozu.eval.simple_map import evaluate_map
from yolozu.predictions import canonicalize_predictions

GATE_SCHEMA = "schema_drift"
GATE_CONSISTENCY = "consistency_drift"
GATE_METRIC = "metric_drift"
GATE_SPEED = "speed_drift"
RUNTIME_LOCK_KEYS = ("torch", "onnxruntime")
ERR_IO = "E_IO"
ERR_DECODE = "E_DECODE"
ERR_PREPROC = "E_PREPROC"

repo_root = Path(__file__).resolve().parents[2]
def _image_size(values: list[int]) -> tuple[int, int]:
    if len(values) == 1:
        return int(values[0]), int(values[0])
    if len(values) == 2:
        return int(values[0]), int(values[1])
    raise SystemExit("--image-size expects 1 or 2 integers")


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_relative_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()).as_posix())
    except (OSError, ValueError):
        return str(path)


def _sha256_json(payload: Any) -> str:
    dumped = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(dumped).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _sanitize_segment(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    clean = clean.strip("-")
    return clean or "unknown"


def _resolve_baseline_path(*, args: argparse.Namespace, cwd: Path) -> Path:
    if str(args.baseline_layout) != "matrix":
        return resolve_output_path(args.baseline, cwd=cwd)
    root = resolve_output_path(args.baseline_root, cwd=cwd)
    adapter_id = _sanitize_segment(str(args.adapter_id))
    backend_id = _sanitize_segment(str(args.backend_id))
    device_id = _sanitize_segment(str(args.matrix_device or args.device))
    version_id = _sanitize_segment(str(args.baseline_version))
    profile_id = _sanitize_segment(str(args.profile))
    return root / adapter_id / backend_id / device_id / version_id / f"{profile_id}.json"


def _capture_cmd_lines(cmd: list[str]) -> list[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _cpu_flags() -> list[str]:
    linux_cpuinfo = Path("/proc/cpuinfo")
    if linux_cpuinfo.exists():
        for raw in linux_cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if raw.lower().startswith("flags"):
                _, _, rhs = raw.partition(":")
                return sorted({x.strip() for x in rhs.split() if x.strip()})
    candidates = [
        ["sysctl", "-n", "machdep.cpu.features"],
        ["sysctl", "-n", "machdep.cpu.leaf7_features"],
    ]
    out: set[str] = set()
    for cmd in candidates:
        for line in _capture_cmd_lines(cmd):
            out.update({x.strip() for x in line.split() if x.strip()})
    return sorted(out)


def _safe_torch_build_info(capture_mode: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}

    info: dict[str, Any] = {
        "available": True,
        "version": str(getattr(torch, "__version__", "")),
        "cuda": str(getattr(getattr(torch, "version", object()), "cuda", None)),
        "git_version": str(getattr(getattr(torch, "version", object()), "git_version", None)),
        "cuda_available": bool(getattr(torch.cuda, "is_available", lambda: False)()),
    }
    if hasattr(torch.backends, "cudnn"):
        try:
            info["cudnn_version"] = torch.backends.cudnn.version()
        except (AttributeError, RuntimeError, TypeError):
            info["cudnn_version"] = None
    try:
        cfg_text = str(torch.__config__.show())
    except (AttributeError, RuntimeError, TypeError):
        cfg_text = ""
    info["config_sha256"] = _sha256_text(cfg_text)
    if capture_mode == "full" and cfg_text:
        info["config"] = cfg_text
    return info


def _git_tag() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "describe", "--tags", "--exact-match"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return None


def _ci_metadata() -> dict[str, Any]:
    env = os.environ
    run_id = env.get("GITHUB_RUN_ID")
    run_attempt = env.get("GITHUB_RUN_ATTEMPT")
    repository = env.get("GITHUB_REPOSITORY")
    server = env.get("GITHUB_SERVER_URL")
    ci_url: str | None = None
    if server and repository and run_id:
        ci_url = f"{server.rstrip('/')}/{repository}/actions/runs/{run_id}"
    return {
        "ci": bool(env.get("CI")),
        "provider": ("github_actions" if run_id else None),
        "run_id": run_id,
        "run_attempt": run_attempt,
        "workflow": env.get("GITHUB_WORKFLOW"),
        "job": env.get("GITHUB_JOB"),
        "run_url": ci_url,
    }


def _collect_provenance(*, capture_mode: str, runtime_lock: dict[str, Any]) -> dict[str, Any]:
    base: dict[str, Any] = {
        "generator": {
            "tool": "tools/run_reference_adapter_regression.py",
            "generated_utc": _now_utc(),
            "user": os.environ.get("USER") or os.environ.get("USERNAME"),
        },
        "git": {
            "sha": _git_sha(),
            "tag": _git_tag(),
            "ref": os.environ.get("GITHUB_REF"),
            "ref_name": os.environ.get("GITHUB_REF_NAME"),
        },
        "ci": _ci_metadata(),
        "capture_mode": str(capture_mode),
    }
    if capture_mode == "off":
        base["snapshot"] = {"enabled": False}
        return base

    freeze_lines = _capture_cmd_lines([sys.executable, "-m", "pip", "freeze"])
    python_vv = "".join(_capture_cmd_lines([sys.executable, "-VV"])) or None
    cpu_flags = _cpu_flags()
    platform_payload = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    snapshot: dict[str, Any] = {
        "enabled": True,
        "python_vv": python_vv,
        "python_vv_sha256": (_sha256_text(python_vv) if python_vv else None),
        "pip_freeze_sha256": _sha256_text("\n".join(freeze_lines)),
        "pip_package_count": int(len(freeze_lines)),
        "platform": platform_payload,
        "cpu_flags_sha256": _sha256_text("\n".join(cpu_flags)),
        "runtime_lock_sha256": runtime_lock.get("sha256"),
        "runtime_lock_path": runtime_lock.get("path"),
        "torch_build": _safe_torch_build_info(capture_mode),
    }
    if capture_mode == "full":
        snapshot["pip_freeze"] = freeze_lines
        snapshot["cpu_flags"] = cpu_flags
    base["snapshot"] = snapshot
    return base


def _normalize_pkg_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _parse_runtime_lock(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    if not path.exists():
        return pins
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            continue
        base = line.split(";", 1)[0].strip()
        if "==" not in base:
            continue
        name, version = base.split("==", 1)
        key = _normalize_pkg_name(name)
        if key in RUNTIME_LOCK_KEYS:
            pins[key] = version.strip()
    return pins


def _hash_list(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def _safe_package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except OSError:
        return None


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        value = out.strip()
        return value if value else None
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return None


def _ensure_repo_write_target(path: Path, *, flag_name: str) -> None:
    try:
        path.resolve().relative_to(repo_root.resolve())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{flag_name} must be under repository root: {path}") from exc


def _canonical_image_key(image: str, *, dataset_root: Path) -> str:
    text = str(image)
    try:
        img = Path(text).resolve()
        root = dataset_root.resolve()
        return str(img.relative_to(root).as_posix())
    except (OSError, ValueError):
        return text


def _dataset_label_path(image_path: Path, *, dataset_root: Path) -> Path | None:
    try:
        rel = image_path.resolve().relative_to(dataset_root.resolve())
    except (OSError, ValueError):
        return None

    parts = rel.parts
    if len(parts) < 2 or parts[0] != "images":
        return None

    label_rel = Path("labels", *parts[1:]).with_suffix(".txt")
    return dataset_root / label_rel


def _dataset_fingerprint(
    records: list[dict[str, Any]],
    *,
    dataset_root: Path,
    split: str,
    max_images: int,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    missing_files: list[str] = []

    for idx, record in enumerate(records):
        where = f"records[{idx}].image"
        image_key = require_image_key(record.get("image"), where=where)
        image_path = Path(image_key)
        if not image_path.is_absolute():
            image_path = (repo_root / image_path).resolve()

        image_rel = _canonical_image_key(str(image_path), dataset_root=dataset_root)
        image_hash: str | None
        if image_path.exists():
            image_hash = _sha256_file(image_path)
        else:
            image_hash = None
            missing_files.append(f"missing image file: {image_path}")

        label_path = _dataset_label_path(image_path, dataset_root=dataset_root)
        label_rel: str | None = None
        label_hash: str | None = None
        if label_path is not None:
            label_rel = _repo_relative_display(label_path)
            if label_path.exists():
                label_hash = _sha256_file(label_path)

        items.append(
            {
                "image": image_rel,
                "image_sha256": image_hash,
                "label": label_rel,
                "label_sha256": label_hash,
            }
        )

    items_sorted = sorted(items, key=lambda row: str(row.get("image", "")))
    payload = {
        "dataset_root": _repo_relative_display(dataset_root),
        "split": str(split),
        "max_images": int(max_images),
        "files": items_sorted,
    }
    return {
        "hash": _sha256_json(payload),
        "payload": payload,
        "count": int(len(items_sorted)),
        "missing": missing_files,
    }


def _resolve_record_image_path(image_key: str) -> Path:
    p = Path(str(image_key))
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def _preflight_records(
    records: list[dict[str, Any]],
    *,
    dataset_root: Path,
    image_size: tuple[int, int],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:
        return {}, [f"{ERR_PREPROC}: Pillow unavailable for record preflight: {exc}"]

    out: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for idx, record in enumerate(records):
        where = f"records[{idx}].image"
        try:
            image_key = require_image_key(record.get("image"), where=where)
        except ValueError as exc:
            errors.append(f"{ERR_IO}: {exc}")
            continue

        image_path = _resolve_record_image_path(image_key)
        canonical_image = _canonical_image_key(str(image_path), dataset_root=dataset_root)

        if not image_path.exists() or not image_path.is_file():
            errors.append(f"{ERR_IO}: {where} not found: {image_path}")
            continue

        try:
            with Image.open(image_path) as src:
                normalized = ImageOps.exif_transpose(src)
                width, height = normalized.size
                mode = str(getattr(normalized, "mode", ""))
                normalized.load()
        except FileNotFoundError:
            errors.append(f"{ERR_IO}: {where} not found during decode: {image_path}")
            continue
        except UnidentifiedImageError as exc:
            errors.append(f"{ERR_DECODE}: {where} unsupported/corrupt image: {image_path} ({exc})")
            continue
        except (OSError, ValueError) as exc:
            errors.append(f"{ERR_DECODE}: {where} decode failed: {image_path} ({exc})")
            continue

        if int(width) <= 0 or int(height) <= 0:
            errors.append(f"{ERR_DECODE}: {where} invalid image size: {width}x{height} ({image_path})")
            continue

        if canonical_image in out:
            errors.append(f"{ERR_IO}: duplicate canonical image key in records: {canonical_image}")
            continue

        out[canonical_image] = {
            "record_index": int(idx),
            "image_path": str(image_path),
            "orig_w": int(width),
            "orig_h": int(height),
            "image_w": int(image_size[0]),
            "image_h": int(image_size[1]),
            "model_input_w": int(image_size[0]),
            "model_input_h": int(image_size[1]),
            "decode": {
                "library": "Pillow",
                "mode": mode,
                "exif_orientation": "normalized",
                "color_order": "RGB",
            },
            "preproc_requirements": {
                "method": "resize",
                "resize_algorithm": "bilinear",
                "pad": {"left": 0, "top": 0, "right": 0, "bottom": 0},
                "letterbox": False,
                "dtype": "float32",
                "normalize": "0_1",
            },
        }
    return out, errors


def _entry_preproc(entry: dict[str, Any]) -> dict[str, Any] | None:
    pp = entry.get("preproc")
    if isinstance(pp, dict):
        return pp
    pp = entry.get("preprocess")
    if isinstance(pp, dict):
        return pp
    return None


def _validate_reference_entry_metadata(
    predictions: list[dict[str, Any]],
    *,
    record_preflight: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    required_int_keys = (
        "image_w",
        "image_h",
        "orig_w",
        "orig_h",
        "model_input_w",
        "model_input_h",
    )

    for idx, entry in enumerate(predictions):
        where = f"predictions[{idx}]"
        image = str(entry.get("image") or "")
        ref = record_preflight.get(image)
        if ref is None:
            errors.append(f"{ERR_IO}: {where}.image missing from preflight map: {image}")
            continue

        for key in required_int_keys:
            value = entry.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or int(value) <= 0:
                errors.append(f"{ERR_PREPROC}: {where}.{key} must be positive int")

        if int(entry.get("orig_w", -1)) != int(ref.get("orig_w", -2)):
            errors.append(
                f"{ERR_PREPROC}: {where}.orig_w mismatch (entry={entry.get('orig_w')} preflight={ref.get('orig_w')})"
            )
        if int(entry.get("orig_h", -1)) != int(ref.get("orig_h", -2)):
            errors.append(
                f"{ERR_PREPROC}: {where}.orig_h mismatch (entry={entry.get('orig_h')} preflight={ref.get('orig_h')})"
            )
        if int(entry.get("model_input_w", -1)) != int(ref.get("model_input_w", -2)):
            errors.append(
                f"{ERR_PREPROC}: {where}.model_input_w mismatch (entry={entry.get('model_input_w')} preflight={ref.get('model_input_w')})"
            )
        if int(entry.get("model_input_h", -1)) != int(ref.get("model_input_h", -2)):
            errors.append(
                f"{ERR_PREPROC}: {where}.model_input_h mismatch (entry={entry.get('model_input_h')} preflight={ref.get('model_input_h')})"
            )

        pp = _entry_preproc(entry)
        if pp is None:
            errors.append(f"{ERR_PREPROC}: {where} missing preprocess/preproc metadata object")
            continue

        method = str(pp.get("method") or "").strip().lower()
        if method != "resize":
            errors.append(f"{ERR_PREPROC}: {where}.preproc.method must be 'resize' (got '{method or '<missing>'}')")

        resize = pp.get("resize")
        if not isinstance(resize, dict):
            errors.append(f"{ERR_PREPROC}: {where}.preproc.resize must be an object")
        else:
            algo = str(resize.get("algorithm") or "").strip().lower()
            if algo != "bilinear":
                errors.append(
                    f"{ERR_PREPROC}: {where}.preproc.resize.algorithm must be 'bilinear' (got '{algo or '<missing>'}')"
                )

        pad = pp.get("pad")
        if not isinstance(pad, dict):
            errors.append(f"{ERR_PREPROC}: {where}.preproc.pad must be an object")
        else:
            for pad_key in ("left", "top", "right", "bottom"):
                try:
                    pad_value = int(pad.get(pad_key, -1))
                except (TypeError, ValueError):
                    errors.append(f"{ERR_PREPROC}: {where}.preproc.pad.{pad_key} must be integer")
                    continue
                if pad_value != 0:
                    errors.append(f"{ERR_PREPROC}: {where}.preproc.pad.{pad_key} must be 0 for resize mode")

        if bool(pp.get("letterbox", False)):
            errors.append(f"{ERR_PREPROC}: {where}.preproc.letterbox must be false for resize mode")

        if str(pp.get("color_order") or "").upper() != "RGB":
            errors.append(f"{ERR_PREPROC}: {where}.preproc.color_order must be RGB")
        if str(pp.get("dtype") or "").lower() != "float32":
            errors.append(f"{ERR_PREPROC}: {where}.preproc.dtype must be float32")
        if str(pp.get("normalize") or "") != "0_1":
            errors.append(f"{ERR_PREPROC}: {where}.preproc.normalize must be 0_1")

    return errors


def _canonicalize_predictions(
    predictions: list[dict[str, Any]],
    *,
    dataset_root: Path,
    decimals: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        canonical = canonicalize_predictions(
            predictions,
            policy="error",
            strict=True,
            unknown_keys="warn",
        )
    except ValueError as exc:
        return [], {
            "schema_errors": [str(exc)],
            "consistency_errors": [],
            "canonical_decimals": int(decimals),
            "bbox_format": "cxcywh_norm",
            "bbox_range": [0.0, 1.0],
            "stable_sort": ["score_desc", "class_id_asc", "bbox_lexicographic"],
            "warnings": [],
        }

    entries: list[dict[str, Any]] = []
    for idx, entry in enumerate(canonical.entries):
        image_key = require_image_key(entry.get("image"), where=f"predictions[{idx}].image")
        out = dict(entry)
        out["image"] = _canonical_image_key(image_key, dataset_root=dataset_root)
        entries.append(out)

    details = {
        "schema_errors": [],
        "consistency_errors": [],
        "canonical_decimals": int(decimals),
        "bbox_format": "cxcywh_norm",
        "bbox_range": [0.0, 1.0],
        "stable_sort": ["score_desc", "class_id_asc", "bbox_lexicographic"],
        "warnings": list(canonical.warnings),
    }
    return entries, details


def _build_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    detections_per_image: list[int] = []
    class_hist: dict[str, int] = {}
    total_detections = 0
    score_sum = 0.0
    bbox_checksum = 0.0

    for entry in predictions:
        dets = entry.get("detections") or []
        detections_per_image.append(int(len(dets)))
        for det in dets:
            total_detections += 1
            score_sum += float(det.get("score", 0.0))
            bbox = det.get("bbox") or {}
            bbox_checksum += float(bbox.get("cx", 0.0))
            bbox_checksum += float(bbox.get("cy", 0.0))
            bbox_checksum += float(bbox.get("w", 0.0))
            bbox_checksum += float(bbox.get("h", 0.0))
            class_id = int(det.get("class_id", -1))
            key = str(class_id)
            class_hist[key] = int(class_hist.get(key, 0) + 1)

    images = int(len(predictions))
    images_with_detections = int(sum(1 for n in detections_per_image if n > 0))
    score_mean = (score_sum / float(total_detections)) if total_detections > 0 else 0.0
    return {
        "images": images,
        "images_with_detections": images_with_detections,
        "total_detections": int(total_detections),
        "detections_per_image": detections_per_image,
        "class_hist": dict(sorted(class_hist.items(), key=lambda kv: int(kv[0]))),
        "score_sum": float(round(score_sum, 6)),
        "score_mean": float(round(score_mean, 6)),
        "bbox_checksum": float(round(bbox_checksum, 6)),
    }


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    q = float(max(0.0, min(1.0, q)))
    data = sorted(float(v) for v in values)
    if len(data) == 1:
        return float(data[0])
    pos = q * float(len(data) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(data[lo])
    frac = pos - float(lo)
    return float(data[lo] * (1.0 - frac) + data[hi] * frac)


def _label_bbox(label: dict[str, Any]) -> dict[str, float] | None:
    bbox_obj = label.get("bbox")
    if isinstance(bbox_obj, dict):
        try:
            return {
                "cx": float(bbox_obj.get("cx")),
                "cy": float(bbox_obj.get("cy")),
                "w": float(bbox_obj.get("w")),
                "h": float(bbox_obj.get("h")),
            }
        except (TypeError, ValueError):
            return None
    try:
        return {
            "cx": float(label.get("cx")),
            "cy": float(label.get("cy")),
            "w": float(label.get("w")),
            "h": float(label.get("h")),
        }
    except (TypeError, ValueError):
        return None


def _extract_gt_records(
    records: list[dict[str, Any]],
    *,
    dataset_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    eval_records: list[dict[str, Any]] = []
    gt_by_image: dict[str, list[dict[str, Any]]] = {}
    for idx, record in enumerate(records):
        where = f"records[{idx}].image"
        try:
            image_key = require_image_key(record.get("image"), where=where)
        except ValueError:
            continue
        canonical_image = _canonical_image_key(image_key, dataset_root=dataset_root)
        labels_out: list[dict[str, Any]] = []
        for label in record.get("labels", []) or []:
            bbox = _label_bbox(label)
            if bbox is None:
                continue
            try:
                class_id = int(label.get("class_id"))
            except (TypeError, ValueError):
                continue
            labels_out.append(
                {
                    "class_id": class_id,
                    "cx": float(bbox["cx"]),
                    "cy": float(bbox["cy"]),
                    "w": float(bbox["w"]),
                    "h": float(bbox["h"]),
                }
            )
        eval_records.append({"image": canonical_image, "labels": labels_out})
        gt_by_image[canonical_image] = [
            {"class_id": int(lbl["class_id"]), "bbox": {"cx": lbl["cx"], "cy": lbl["cy"], "w": lbl["w"], "h": lbl["h"]}}
            for lbl in labels_out
        ]
    return eval_records, gt_by_image


def _preds_by_image(predictions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for entry in predictions:
        image = str(entry.get("image", ""))
        if not image:
            continue
        rows: list[dict[str, Any]] = out.setdefault(image, [])
        for det in entry.get("detections", []) or []:
            bbox = det.get("bbox") or {}
            try:
                rows.append(
                    {
                        "class_id": int(det.get("class_id")),
                        "score": float(det.get("score", 0.0)),
                        "bbox": {
                            "cx": float(bbox.get("cx")),
                            "cy": float(bbox.get("cy")),
                            "w": float(bbox.get("w")),
                            "h": float(bbox.get("h")),
                        },
                    }
                )
            except (TypeError, ValueError):
                continue
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return out


def _class_aware_counts(
    *,
    gt_by_image: dict[str, list[dict[str, Any]]],
    pred_by_image: dict[str, list[dict[str, Any]]],
    iou_threshold: float,
    top_k: int | None,
) -> tuple[int, int, int, list[float]]:
    tp = 0
    fp = 0
    fn = 0
    matched_ious: list[float] = []

    image_ids = sorted(set(gt_by_image.keys()) | set(pred_by_image.keys()))
    for image in image_ids:
        gt_items = list(gt_by_image.get(image) or [])
        pred_items = list(pred_by_image.get(image) or [])
        if top_k is not None:
            pred_items = pred_items[: max(0, int(top_k))]

        used_gt: set[int] = set()
        for pred in pred_items:
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gt_items):
                if idx in used_gt:
                    continue
                if int(gt.get("class_id", -1)) != int(pred.get("class_id", -2)):
                    continue
                iou = float(iou_cxcywh_norm_dict(pred.get("bbox") or {}, gt.get("bbox") or {}))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= float(iou_threshold):
                used_gt.add(best_idx)
                tp += 1
                matched_ious.append(float(best_iou))
            else:
                fp += 1
        fn += max(0, len(gt_items) - len(used_gt))
    return tp, fp, fn, matched_ious


def _class_mismatch_count(
    *,
    gt_by_image: dict[str, list[dict[str, Any]]],
    pred_by_image: dict[str, list[dict[str, Any]]],
    iou_threshold: float,
) -> int:
    mismatch = 0
    for image, preds in pred_by_image.items():
        gts = list(gt_by_image.get(image) or [])
        if not gts:
            continue
        for pred in preds:
            best_iou = 0.0
            best_class = None
            for gt in gts:
                iou = float(iou_cxcywh_norm_dict(pred.get("bbox") or {}, gt.get("bbox") or {}))
                if iou > best_iou:
                    best_iou = iou
                    best_class = int(gt.get("class_id", -1))
            if best_iou >= float(iou_threshold) and best_class is not None:
                if int(pred.get("class_id", -2)) != int(best_class):
                    mismatch += 1
    return int(mismatch)


def _build_robust_metrics(
    *,
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    dataset_root: Path,
    worst_k: int,
    recall_k: int,
) -> dict[str, Any]:
    eval_records, gt_by_image = _extract_gt_records(records, dataset_root=dataset_root)
    pred_by_image = _preds_by_image(predictions)

    thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    map_result = evaluate_map(eval_records, predictions, iou_thresholds=thresholds)
    per_class_map50 = {
        str(int(class_id)): float(round(metrics.get("ap@0.50", 0.0), 6))
        for class_id, metrics in (map_result.per_class or {}).items()
    }
    class_scores = sorted(float(v) for v in per_class_map50.values())
    worst_k = max(1, int(worst_k))
    worst_values = class_scores[: min(worst_k, len(class_scores))]
    worst_k_map50 = (sum(worst_values) / float(len(worst_values))) if worst_values else 0.0
    median_class_map50 = _quantile(class_scores, 0.5) if class_scores else 0.0

    tp_all, fp_all, fn_all, ious = _class_aware_counts(
        gt_by_image=gt_by_image,
        pred_by_image=pred_by_image,
        iou_threshold=0.5,
        top_k=None,
    )
    tp_k, _fp_k, fn_k, _ious_k = _class_aware_counts(
        gt_by_image=gt_by_image,
        pred_by_image=pred_by_image,
        iou_threshold=0.5,
        top_k=max(1, int(recall_k)),
    )
    gt_count = int(tp_all + fn_all)
    recall_at_k = (float(tp_k) / float(tp_k + fn_k)) if (tp_k + fn_k) > 0 else 0.0
    mismatch = _class_mismatch_count(gt_by_image=gt_by_image, pred_by_image=pred_by_image, iou_threshold=0.5)
    pred_count = int(sum(len(v) for v in pred_by_image.values()))

    return {
        "map50": float(round(float(map_result.map50), 6)),
        "map50_95": float(round(float(map_result.map50_95), 6)),
        "per_class_map50": per_class_map50,
        "class_count": int(len(class_scores)),
        "worst_k": int(worst_k),
        "worst_k_map50": float(round(worst_k_map50, 6)),
        "median_class_map50": float(round(median_class_map50, 6)),
        "recall_k": int(max(1, int(recall_k))),
        "recall_at_k": float(round(recall_at_k, 6)),
        "iou_p10": float(round(_quantile(ious, 0.10), 6)),
        "iou_p50": float(round(_quantile(ious, 0.50), 6)),
        "iou_p90": float(round(_quantile(ious, 0.90), 6)),
        "matched_count": int(tp_all),
        "missing_count": int(fn_all),
        "extra_count": int(fp_all),
        "class_mismatch_count": int(mismatch),
        "gt_count": int(gt_count),
        "pred_count": int(pred_count),
    }


def _find_duplicates(items: list[str]) -> list[str]:
    seen: set[str] = set()
    dupes: list[str] = []
    for item in items:
        if item in seen:
            dupes.append(item)
        seen.add(item)
    return dupes


def _build_contract(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    dataset_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []

    record_images: list[str] = []
    for idx, record in enumerate(records):
        where = f"records[{idx}].image"
        try:
            image_key = require_image_key(record.get("image"), where=where)
        except ValueError as exc:
            errors.append(str(exc))
            image_key = f"__invalid_record__/{idx}"
        record_images.append(_canonical_image_key(image_key, dataset_root=dataset_root))

    prediction_images = [
        _canonical_image_key(str((entry or {}).get("image")), dataset_root=dataset_root)
        for entry in predictions
    ]

    if len(predictions) != len(records):
        errors.append(
            f"entry count mismatch: predictions={len(predictions)} records={len(records)}"
        )

    if prediction_images != record_images:
        errors.append("image order mismatch between records and predictions")

    record_dupes = _find_duplicates(record_images)
    if record_dupes:
        errors.append(f"duplicate record images detected: {sorted(set(record_dupes))}")

    prediction_dupes = _find_duplicates(prediction_images)
    if prediction_dupes:
        errors.append(f"duplicate prediction images detected: {sorted(set(prediction_dupes))}")

    contract = {
        "record_images_sha256": _hash_list(record_images),
        "prediction_images_sha256": _hash_list(prediction_images),
        "record_images": record_images,
        "prediction_images": prediction_images,
        "records": int(len(record_images)),
        "entries": int(len(prediction_images)),
    }
    return contract, errors


def _thresholds_from_args(args: argparse.Namespace) -> dict[str, Any]:
    metric_common = {
        "total_detections_abs": float(args.metric_total_detections_abs),
        "score_sum_abs": float(args.metric_score_sum_abs),
        "score_mean_abs": float(args.metric_score_mean_abs),
        "bbox_checksum_abs": float(args.metric_bbox_checksum_abs),
        "map50_abs": float(args.metric_map50_abs),
        "map50_95_abs": float(args.metric_map50_95_abs),
        "worst_k_map50_abs": float(args.metric_worst_k_map50_abs),
        "median_class_map50_abs": float(args.metric_median_class_map50_abs),
        "recall_at_k_abs": float(args.metric_recall_at_k_abs),
        "iou_p10_abs": float(args.metric_iou_p10_abs),
        "iou_p50_abs": float(args.metric_iou_p50_abs),
        "missing_count_abs": float(args.metric_missing_count_abs),
        "extra_count_abs": float(args.metric_extra_count_abs),
        "class_mismatch_count_abs": float(args.metric_class_mismatch_abs),
    }
    backend_id = str(args.backend_id)
    parity_common = {
        "mode": str(args.backend_parity_mode),
        "map50_abs": float(args.backend_parity_map50_abs),
        "map50_95_abs": float(args.backend_parity_map50_95_abs),
    }
    return {
        "metric": metric_common,
        "metric_by_backend": {
            backend_id: metric_common,
        },
        "backend_parity": parity_common,
        "backend_parity_by_backend": {
            backend_id: parity_common,
        },
        "speed": {
            "min_fps_ratio": float(args.min_fps_ratio),
            "absolute_floor_fps": float(args.absolute_floor_fps),
        },
    }


def _gate_policy_from_args(args: argparse.Namespace) -> dict[str, str]:
    return {
        GATE_SCHEMA: str(args.schema_gate_mode),
        GATE_CONSISTENCY: str(args.consistency_gate_mode),
        GATE_METRIC: str(args.score_gate_mode),
        GATE_SPEED: str(args.perf_gate_mode),
    }


def _protocol_spec(*, canonical_decimals: int) -> dict[str, Any]:
    return {
        "name": "reference_adapter_regression",
        "interface_contract": {
            "hard_invariants": [
                "records preflight requires readable image files with successful decode and positive dimensions",
                "entries schema validation with strict mode",
                "record/prediction image mapping and ordering",
                "image identifier canonicalization to dataset-relative keys",
                "bbox coordinate system is cxcywh_norm and values must be finite",
                "duplicate image IDs and duplicate detections are rejected",
                "empty detections are normalized to []",
                "stable detection sort key is fixed",
                "reference-adapter entries must include image_w/h, orig_w/h, model_input_w/h, and preproc metadata",
            ],
            "forbidden_values": ["NaN", "+inf", "-inf"],
        },
        "behavior": {
            "soft_invariants": [
                "metric drift (total_detections, score_sum, score_mean, bbox_checksum)",
                "robust metric drift (map50/map50_95, worst-k, quantiles, recall@K, count diagnostics)",
                "backend parity drift against peer backend report",
                "performance drift (fps ratio and absolute floor)",
            ],
            "gate_adoption": "warn_then_hard",
        },
        "canonicalization": {
            "bbox_format": "cxcywh_norm",
            "bbox_range": [0.0, 1.0],
            "round_decimals": int(canonical_decimals),
            "stable_sort": ["score_desc", "class_id_asc", "bbox_lexicographic"],
            "empty_detections": "[]",
        },
    }


def _default_diff_summary_path(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.diff_summary.json"


def _default_topk_examples_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_topk_examples"


def _safe_slug(value: str, *, max_len: int = 64) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    clean = clean.strip("._")
    if not clean:
        clean = "image"
    return clean[: max(1, int(max_len))]


def _bbox_xyxy_from_norm(bbox: dict[str, Any], *, width: int, height: int) -> tuple[int, int, int, int] | None:
    try:
        cx = float(bbox.get("cx"))
        cy = float(bbox.get("cy"))
        bw = float(bbox.get("w"))
        bh = float(bbox.get("h"))
    except (TypeError, ValueError):
        return None
    x0 = int(round((cx - bw / 2.0) * float(width)))
    y0 = int(round((cy - bh / 2.0) * float(height)))
    x1 = int(round((cx + bw / 2.0) * float(width)))
    y1 = int(round((cy + bh / 2.0) * float(height)))
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _build_record_lookup(
    records: list[dict[str, Any]],
    *,
    dataset_root: Path,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for idx, record in enumerate(records):
        try:
            image_key = require_image_key(record.get("image"), where=f"records[{idx}].image")
        except ValueError:
            continue
        image_path = _resolve_record_image_path(image_key)
        canonical = _canonical_image_key(str(image_path), dataset_root=dataset_root)
        out[canonical] = {
            "record_index": int(idx),
            "image_path": str(image_path),
            "labels": list(record.get("labels") or []),
        }
    return out


def _collect_counterexample_images(
    *,
    report: dict[str, Any],
    predictions: list[dict[str, Any]],
    limit: int,
) -> list[str]:
    images: list[str] = []
    gates = report.get("gates") or {}
    for gate_key in (GATE_CONSISTENCY, GATE_METRIC, GATE_SCHEMA, GATE_SPEED):
        gate = gates.get(gate_key) or {}
        details = gate.get("details") or {}
        first = details.get("first_counterexample") if isinstance(details, dict) else None
        if isinstance(first, dict):
            image = str(first.get("image") or "").strip()
            if image:
                images.append(image)
    for entry in predictions:
        image = str(entry.get("image") or "").strip()
        if image:
            images.append(image)
        if len(images) >= int(limit) * 3:
            break
    unique: list[str] = []
    seen: set[str] = set()
    for image in images:
        if image in seen:
            continue
        seen.add(image)
        unique.append(image)
        if len(unique) >= int(limit):
            break
    return unique


def _write_topk_examples(
    *,
    out_dir: Path,
    report: dict[str, Any],
    predictions: list[dict[str, Any]],
    records: list[dict[str, Any]],
    dataset_root: Path,
    topk: int,
) -> list[str]:
    if int(topk) <= 0:
        return []
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return []

    record_lookup = _build_record_lookup(records, dataset_root=dataset_root)
    pred_lookup: dict[str, dict[str, Any]] = {str(e.get("image")): e for e in predictions if isinstance(e, dict)}
    targets = _collect_counterexample_images(report=report, predictions=predictions, limit=int(topk))
    if not targets:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for rank, image_key in enumerate(targets, start=1):
        rec = record_lookup.get(image_key)
        if rec is None:
            continue
        image_path = Path(str(rec.get("image_path")))
        if not image_path.exists():
            continue
        pred_entry = pred_lookup.get(image_key) or {}
        labels = list(rec.get("labels") or [])
        detections = list(pred_entry.get("detections") or [])

        try:
            with Image.open(image_path) as src:
                canvas = src.convert("RGB")
        except (OSError, ValueError):
            continue

        draw = ImageDraw.Draw(canvas)
        width, height = canvas.size
        for label in labels:
            bbox = _label_bbox(label) or {}
            xyxy = _bbox_xyxy_from_norm(bbox, width=width, height=height)
            if xyxy is None:
                continue
            class_id = label.get("class_id")
            draw.rectangle(xyxy, outline=(0, 200, 0), width=2)
            draw.text((xyxy[0] + 2, max(0, xyxy[1] - 12)), f"gt:{class_id}", fill=(0, 200, 0))
        for det in detections[:20]:
            bbox = det.get("bbox") or {}
            xyxy = _bbox_xyxy_from_norm(bbox, width=width, height=height)
            if xyxy is None:
                continue
            class_id = det.get("class_id")
            try:
                score = float(det.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            draw.rectangle(xyxy, outline=(220, 30, 30), width=2)
            draw.text((xyxy[0] + 2, xyxy[1] + 2), f"pred:{class_id}@{score:.2f}", fill=(220, 30, 30))

        out_path = out_dir / f"{int(rank):02d}_{_safe_slug(image_key)}.png"
        canvas.save(out_path)
        written.append(_repo_relative_display(out_path))
    return written


def _build_diff_summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    gates = report.get("gates") or {}
    gate_status = {
        str(k): {
            "ok": bool((v or {}).get("ok", True)),
            "mode": str((v or {}).get("mode", "")),
            "category": str((v or {}).get("category", "")),
        }
        for k, v in gates.items()
    }
    counterexamples: dict[str, Any] = {}
    for gate_key, gate in gates.items():
        details = (gate or {}).get("details") or {}
        first = details.get("first_counterexample") if isinstance(details, dict) else None
        if isinstance(first, dict):
            counterexamples[str(gate_key)] = first

    return {
        "schema_version": 1,
        "generated_utc": _now_utc(),
        "ok": bool(report.get("ok")),
        "run": report.get("run"),
        "baseline_path": report.get("baseline_path"),
        "gate_status": gate_status,
        "hard_failure_count": int(len(report.get("hard_failures") or [])),
        "soft_failure_count": int(len(report.get("soft_failures") or [])),
        "first_hard_failure": ((report.get("hard_failures") or [None])[0]),
        "first_soft_failure": ((report.get("soft_failures") or [None])[0]),
        "counterexamples": counterexamples,
        "failure_records_topk": list((report.get("failure_records") or [])[:10]),
    }
