#!/usr/bin/env python3
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

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.adapter import RTDETRPoseAdapter
from yolozu.core.boxes import iou_cxcywh_norm_dict
from yolozu.core.cli_args import (
    require_float_in_range,
    require_non_negative_int,
    require_positive_int,
    resolve_input_path,
    resolve_output_path,
)
from yolozu.core.image_keys import require_image_key
from yolozu.dataset import build_manifest
from yolozu.eval.simple_map import evaluate_map
from yolozu.predictions import canonicalize_predictions, validate_predictions_entries

DEFAULT_BASELINE = "baselines/reference_adapter/rtdetr_pose_smoke_val.json"
DEFAULT_DATASET = "data/smoke"
DEFAULT_SPLIT = "val"
DEFAULT_BASELINE_ROOT = "baselines/reference_adapter"
DEFAULT_PROFILE = "micro"

GATE_SCHEMA = "schema_drift"
GATE_CONSISTENCY = "consistency_drift"
GATE_METRIC = "metric_drift"
GATE_SPEED = "speed_drift"
RUNTIME_LOCK_KEYS = ("torch", "onnxruntime")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run RT-DETR reference-adapter regression on a fixed real-image dataset with "
            "separated contract/behavior gates and reproducibility metadata."
        )
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="YOLO-format dataset root.")
    p.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split (default: val).")
    p.add_argument("--max-images", type=int, default=2, help="Max images to evaluate (default: 2).")
    p.add_argument(
        "--profile",
        choices=("micro", "full", "custom"),
        default=DEFAULT_PROFILE,
        help="Regression profile label used for matrix baselines/reporting (default: micro).",
    )
    p.add_argument("--baseline", default=DEFAULT_BASELINE, help="Baseline JSON path.")
    p.add_argument(
        "--baseline-layout",
        choices=("flat", "matrix"),
        default="flat",
        help="Baseline path layout: flat (use --baseline) or matrix (derive from backend/device/version).",
    )
    p.add_argument(
        "--baseline-root",
        default=DEFAULT_BASELINE_ROOT,
        help="Root directory for matrix baseline layout (default: baselines/reference_adapter).",
    )
    p.add_argument(
        "--adapter-id",
        default="rtdetr_pose",
        help="Adapter id for matrix baseline layout (default: rtdetr_pose).",
    )
    p.add_argument(
        "--backend-id",
        default="torch",
        help="Backend id for matrix baseline layout (default: torch).",
    )
    p.add_argument(
        "--matrix-device",
        default=None,
        help="Optional device id override for matrix baseline layout (default: --device).",
    )
    p.add_argument(
        "--baseline-version",
        default="v1",
        help="Baseline version segment used by matrix baseline layout (default: v1).",
    )
    p.add_argument(
        "--output",
        default="reports/reference_adapter_regression.json",
        help="Regression report output JSON path.",
    )
    p.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write/update baseline JSON from current run instead of comparing.",
    )
    p.add_argument(
        "--runtime-lock",
        default="requirements-ci.lock",
        help="Pinned runtime lock file used for CI/runtime reproducibility checks.",
    )
    p.add_argument(
        "--enforce-runtime-lock",
        action="store_true",
        help="Fail if run-time torch/onnxruntime versions differ from --runtime-lock pins.",
    )
    p.add_argument(
        "--enforce-weights-hash",
        action="store_true",
        help="Fail consistency gate if baseline/current weights_hash differ (even checkpoint-free).",
    )
    p.add_argument(
        "--expected-dataset-hash",
        default=None,
        help="Optional expected dataset SHA256. Mismatch fails consistency gate.",
    )
    p.add_argument(
        "--expected-weights-hash",
        default=None,
        help="Optional expected weights SHA256. Mismatch fails consistency gate.",
    )
    p.add_argument(
        "--expected-checkpoint-hash",
        default=None,
        help="Optional expected checkpoint SHA256. Mismatch fails consistency gate.",
    )
    p.add_argument(
        "--capture-provenance",
        choices=("full", "minimal", "off"),
        default="full",
        help="Capture SBOM/environment provenance snapshot in baseline_meta (default: full).",
    )

    p.add_argument("--config", default="rtdetr_pose/configs/base.json", help="RT-DETR config path.")
    p.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    p.add_argument("--device", default="cpu", help="Device for adapter inference (default: cpu).")
    p.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=[160],
        help="Image size for adapter (one value or two values).",
    )
    p.add_argument("--score-threshold", type=float, default=0.05, help="Adapter score threshold.")
    p.add_argument("--max-detections", type=int, default=20, help="Max detections per image.")
    p.add_argument(
        "--init-seed",
        type=int,
        default=2026,
        help="Deterministic model-init seed for reference baseline (default: 2026).",
    )
    p.add_argument(
        "--repro-policy",
        choices=("strict", "relaxed", "off"),
        default="relaxed",
        help="Reproducibility policy: strict (deterministic), relaxed (seed-only), off (speed).",
    )

    p.add_argument(
        "--schema-gate-mode",
        choices=("hard", "off"),
        default="hard",
        help="Schema gate mode (default: hard).",
    )
    p.add_argument(
        "--consistency-gate-mode",
        choices=("hard", "off"),
        default="hard",
        help="Consistency gate mode (default: hard).",
    )
    p.add_argument(
        "--score-gate-mode",
        choices=("warn", "hard", "off"),
        default="warn",
        help="Behavior score gate mode (default: warn).",
    )
    p.add_argument(
        "--perf-gate-mode",
        choices=("warn", "hard", "off"),
        default="warn",
        help="Behavior performance gate mode (default: warn).",
    )

    p.add_argument(
        "--canonical-decimals",
        type=int,
        default=6,
        help="Decimal places used by canonicalization (default: 6).",
    )

    p.add_argument(
        "--metric-total-detections-abs",
        type=float,
        default=0.0,
        help="Allowed absolute drift for total detections.",
    )
    p.add_argument(
        "--metric-score-sum-abs",
        type=float,
        default=0.01,
        help="Allowed absolute drift for score_sum.",
    )
    p.add_argument(
        "--metric-score-mean-abs",
        type=float,
        default=0.001,
        help="Allowed absolute drift for score_mean.",
    )
    p.add_argument(
        "--metric-bbox-checksum-abs",
        type=float,
        default=0.1,
        help="Allowed absolute drift for bbox_checksum.",
    )
    p.add_argument(
        "--metric-map50-abs",
        type=float,
        default=0.03,
        help="Allowed absolute drift for robust metric map50.",
    )
    p.add_argument(
        "--metric-map50-95-abs",
        type=float,
        default=0.03,
        help="Allowed absolute drift for robust metric map50_95.",
    )
    p.add_argument(
        "--metric-worst-k-map50-abs",
        type=float,
        default=0.05,
        help="Allowed absolute drift for robust metric worst_k_map50.",
    )
    p.add_argument(
        "--metric-median-class-map50-abs",
        type=float,
        default=0.05,
        help="Allowed absolute drift for robust metric median_class_map50.",
    )
    p.add_argument(
        "--metric-recall-at-k-abs",
        type=float,
        default=0.05,
        help="Allowed absolute drift for robust metric recall_at_k.",
    )
    p.add_argument(
        "--metric-iou-p10-abs",
        type=float,
        default=0.1,
        help="Allowed absolute drift for robust metric iou_p10.",
    )
    p.add_argument(
        "--metric-iou-p50-abs",
        type=float,
        default=0.1,
        help="Allowed absolute drift for robust metric iou_p50.",
    )
    p.add_argument(
        "--metric-missing-count-abs",
        type=float,
        default=2.0,
        help="Allowed absolute drift for robust metric missing_count.",
    )
    p.add_argument(
        "--metric-extra-count-abs",
        type=float,
        default=2.0,
        help="Allowed absolute drift for robust metric extra_count.",
    )
    p.add_argument(
        "--metric-class-mismatch-abs",
        type=float,
        default=2.0,
        help="Allowed absolute drift for robust metric class_mismatch_count.",
    )
    p.add_argument(
        "--metric-worst-k",
        type=int,
        default=3,
        help="Worst-k class mAP aggregation size for robust metrics (default: 3).",
    )
    p.add_argument(
        "--metric-recall-k",
        type=int,
        default=20,
        help="K for recall@K robust metric (default: 20).",
    )
    p.add_argument(
        "--peer-report",
        default=None,
        help="Optional peer backend regression report for backend parity drift checks.",
    )
    p.add_argument(
        "--backend-parity-mode",
        choices=("off", "warn", "hard"),
        default="off",
        help="Backend parity drift policy against --peer-report (default: off).",
    )
    p.add_argument(
        "--backend-parity-map50-abs",
        type=float,
        default=0.03,
        help="Allowed absolute map50 delta vs peer backend report.",
    )
    p.add_argument(
        "--backend-parity-map50-95-abs",
        type=float,
        default=0.03,
        help="Allowed absolute map50_95 delta vs peer backend report.",
    )
    p.add_argument(
        "--min-fps-ratio",
        type=float,
        default=0.25,
        help="Perf gate lower bound as ratio against baseline fps (default: 0.25).",
    )
    p.add_argument(
        "--absolute-floor-fps",
        type=float,
        default=0.2,
        help="Absolute minimum fps floor for perf gate (default: 0.2).",
    )
    return p.parse_args(argv)


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
    except Exception:
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
    except Exception:
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
    except Exception:
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
        except Exception:
            info["cudnn_version"] = None
    try:
        cfg_text = str(torch.__config__.show())
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
        return None


def _ensure_repo_write_target(path: Path, *, flag_name: str) -> None:
    try:
        path.resolve().relative_to(repo_root.resolve())
    except Exception as exc:
        raise SystemExit(f"{flag_name} must be under repository root: {path}") from exc


def _canonical_image_key(image: str, *, dataset_root: Path) -> str:
    text = str(image)
    try:
        img = Path(text).resolve()
        root = dataset_root.resolve()
        return str(img.relative_to(root).as_posix())
    except Exception:
        return text


def _dataset_label_path(image_path: Path, *, dataset_root: Path) -> Path | None:
    try:
        rel = image_path.resolve().relative_to(dataset_root.resolve())
    except Exception:
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
        except Exception:
            return None
    try:
        return {
            "cx": float(label.get("cx")),
            "cy": float(label.get("cy")),
            "w": float(label.get("w")),
            "h": float(label.get("h")),
        }
    except Exception:
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
            except Exception:
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
            except Exception:
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
    return {
        "metric": {
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
        },
        "backend_parity": {
            "mode": str(args.backend_parity_mode),
            "map50_abs": float(args.backend_parity_map50_abs),
            "map50_95_abs": float(args.backend_parity_map50_95_abs),
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
                "entries schema validation with strict mode",
                "record/prediction image mapping and ordering",
                "image identifier canonicalization to dataset-relative keys",
                "bbox coordinate system is cxcywh_norm and values must be finite",
                "duplicate image IDs and duplicate detections are rejected",
                "empty detections are normalized to []",
                "stable detection sort key is fixed",
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


def _configure_repro_policy(*, policy: str, seed: int | None) -> dict[str, Any]:
    details: dict[str, Any] = {
        "policy": str(policy),
        "seed": (int(seed) if seed is not None else None),
        "actions": [],
    }

    try:
        import torch
    except Exception:
        details["actions"].append("torch_unavailable")
        return details

    if policy in ("strict", "relaxed") and seed is not None:
        torch.manual_seed(int(seed))
        details["actions"].append("torch.manual_seed")
        if bool(getattr(torch.cuda, "is_available", lambda: False)()):
            try:
                torch.cuda.manual_seed_all(int(seed))
                details["actions"].append("torch.cuda.manual_seed_all")
            except Exception:
                details["actions"].append("torch.cuda.manual_seed_all_failed")

    if policy == "strict":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        details["actions"].append("env:CUBLAS_WORKSPACE_CONFIG")
        try:
            torch.use_deterministic_algorithms(True)
            details["actions"].append("torch.use_deterministic_algorithms(True)")
        except Exception:
            details["actions"].append("torch.use_deterministic_algorithms_failed")
        if hasattr(torch.backends, "cudnn"):
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                details["actions"].append("torch.backends.cudnn.deterministic=True")
                details["actions"].append("torch.backends.cudnn.benchmark=False")
            except Exception:
                details["actions"].append("cudnn_flags_failed")

    if policy == "off":
        details["actions"].append("repro_disabled")

    try:
        details["deterministic_algorithms_enabled"] = bool(torch.are_deterministic_algorithms_enabled())
    except Exception:
        details["deterministic_algorithms_enabled"] = None

    return details


def _hash_model_state_dict(model: Any, torch_module: Any) -> str | None:
    try:
        state = model.state_dict()
    except Exception:
        return None

    digest = hashlib.sha256()
    for key in sorted(state.keys()):
        tensor = state[key]
        if not hasattr(tensor, "detach"):
            continue
        t = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(getattr(t, "shape", ()))).encode("utf-8"))
        digest.update(str(getattr(t, "dtype", "unknown")).encode("utf-8"))
        try:
            digest.update(t.numpy().tobytes(order="C"))
        except Exception:
            try:
                digest.update(bytes(t.view(torch_module.uint8).tolist()))
            except Exception:
                return None
    return digest.hexdigest()


def _collect_run_meta(
    *,
    adapter: RTDETRPoseAdapter,
    args: argparse.Namespace,
    config_path: Path,
    checkpoint_path: Path | None,
    dataset_fingerprint: dict[str, Any],
    repro_details: dict[str, Any],
    canonicalization: dict[str, Any],
    runtime_lock: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    model_dtype: str | None = None
    model_state_hash: str | None = None
    model_hash_source: str | None = None
    backend = str(args.backend_id)

    try:
        model = adapter.get_model()
        torch_module = (adapter._backend or {}).get("torch") if hasattr(adapter, "_backend") else None
        if model is not None:
            try:
                param = next(model.parameters())
                model_dtype = str(getattr(param, "dtype", None))
            except Exception:
                model_dtype = None
        if model is not None and torch_module is not None:
            model_state_hash = _hash_model_state_dict(model, torch_module)
            model_hash_source = "model_state_dict"
    except Exception:
        model_dtype = None

    checkpoint_hash = _sha256_file(checkpoint_path) if checkpoint_path is not None and checkpoint_path.exists() else None
    config_hash = _sha256_file(config_path) if config_path.exists() else None

    weights_hash = checkpoint_hash if checkpoint_hash else model_state_hash
    weights_source = "checkpoint" if checkpoint_hash else model_hash_source

    versions = {
        "python": platform.python_version(),
        "torch": _safe_package_version("torch"),
        "onnxruntime": _safe_package_version("onnxruntime"),
        "ultralytics": _safe_package_version("ultralytics"),
        "numpy": _safe_package_version("numpy"),
        "pillow": _safe_package_version("Pillow"),
        "yolozu": _safe_package_version("yolozu"),
    }

    return {
        "generated_utc": _now_utc(),
        "git_sha": _git_sha(),
        "versions": versions,
        "backend": backend,
        "device": str(args.device),
        "dtype": model_dtype,
        "repro_policy": str(args.repro_policy),
        "seed": (None if str(args.repro_policy) == "off" else int(args.init_seed)),
        "repro_actions": list(repro_details.get("actions") or []),
        "deterministic_algorithms_enabled": repro_details.get("deterministic_algorithms_enabled"),
        "weights_hash": weights_hash,
        "weights_source": weights_source,
        "checkpoint_hash": checkpoint_hash,
        "config_hash": config_hash,
        "dataset_hash": dataset_fingerprint.get("hash"),
        "dataset_count": dataset_fingerprint.get("count"),
        "dataset_missing": list(dataset_fingerprint.get("missing") or []),
        "canonical_decimals": int(canonicalization.get("canonical_decimals", 6)),
        "bbox_format": str(canonicalization.get("bbox_format", "cxcywh_norm")),
        "runtime_lock_path": runtime_lock.get("path"),
        "runtime_lock_sha256": runtime_lock.get("sha256"),
        "runtime_lock_versions": dict(runtime_lock.get("versions") or {}),
        "profile": str(args.profile),
        "baseline_layout": str(args.baseline_layout),
        "provenance": provenance,
    }


def _build_baseline_payload(
    *,
    args: argparse.Namespace,
    baseline_path: Path,
    dataset_root: Path,
    split: str,
    summary: dict[str, Any],
    robust_metrics: dict[str, Any],
    speed: dict[str, Any],
    contract: dict[str, Any],
    predictions_sha256: str,
    run_meta: dict[str, Any],
    protocol: dict[str, Any],
    gate_policy: dict[str, str],
    canonicalization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "reference_adapter": str(args.adapter_id),
        "generated_utc": _now_utc(),
        "profile": str(args.profile),
        "baseline_layout": str(args.baseline_layout),
        "baseline_path": _repo_relative_display(baseline_path),
        "dataset": {
            "path": _repo_relative_display(dataset_root),
            "split": str(split),
            "max_images": int(args.max_images),
            "hash": run_meta.get("dataset_hash"),
        },
        "adapter": {
            "config": str(args.config),
            "checkpoint": (str(args.checkpoint) if args.checkpoint else None),
            "device": str(args.device),
            "image_size": [int(v) for v in _image_size(list(args.image_size))],
            "score_threshold": float(args.score_threshold),
            "max_detections": int(args.max_detections),
            "init_seed": (None if str(args.repro_policy) == "off" else int(args.init_seed)),
            "repro_policy": str(args.repro_policy),
        },
        "thresholds": _thresholds_from_args(args),
        "gate_policy": gate_policy,
        "protocol": protocol,
        "canonicalization": {
            "canonical_decimals": int(canonicalization.get("canonical_decimals", 6)),
            "bbox_format": str(canonicalization.get("bbox_format", "cxcywh_norm")),
            "stable_sort": list(canonicalization.get("stable_sort") or []),
        },
        "baseline_meta": run_meta,
        "baseline": {
            "summary": summary,
            "robust_metrics": robust_metrics,
            "speed": speed,
            "contract": contract,
            "predictions_sha256": predictions_sha256,
        },
    }


def _new_gate(*, mode: str, category: str) -> dict[str, Any]:
    return {
        "mode": str(mode),
        "category": str(category),
        "ok": True,
        "details": {},
    }


def _failure_code(gate_key: str, message: str) -> str:
    text = str(message).lower()
    if gate_key == GATE_SCHEMA:
        if "unknown keys" in text:
            return "E_SCHEMA_UNKNOWN_KEYS"
        if "missing" in text:
            return "E_SCHEMA_MISSING_FIELD"
        return "E_SCHEMA_VALIDATION"
    if gate_key == GATE_CONSISTENCY:
        if "runtime lock" in text:
            return "E_CANON_RUNTIME_LOCK"
        if "weights_hash mismatch" in text:
            return "E_CANON_WEIGHTS_HASH"
        if "image order mismatch" in text:
            return "E_CANON_IMAGE_ORDER"
        if "duplicate" in text:
            return "E_CANON_DUPLICATE_IMAGE"
        return "E_CANON_CONSISTENCY"
    if gate_key == GATE_METRIC:
        if "parity" in text:
            return "E_SCORE_PARITY"
        return "E_SCORE_DRIFT"
    if gate_key == GATE_SPEED:
        return "E_PERF_DRIFT"
    return "E_UNKNOWN"


def _append_gate_failure(
    *,
    gate_key: str,
    message: str,
    gate_policy: dict[str, str],
    hard_failures: list[str],
    soft_failures: list[str],
    failure_records: list[dict[str, Any]],
    mode_override: str | None = None,
) -> None:
    mode = str(mode_override or gate_policy.get(gate_key, "hard"))
    code = _failure_code(gate_key, message)
    line = f"[{gate_key}][{code}] {message}"
    if mode == "hard":
        hard_failures.append(line)
    elif mode == "warn":
        soft_failures.append(line)
    failure_records.append(
        {
            "gate": gate_key,
            "mode": mode,
            "code": code,
            "message": str(message),
        }
    )


def _compare_against_baseline(
    *,
    baseline_payload: dict[str, Any],
    summary: dict[str, Any],
    robust_metrics: dict[str, Any],
    speed: dict[str, Any],
    contract: dict[str, Any],
    run_meta: dict[str, Any],
    schema_warnings: list[str],
    schema_errors: list[str],
    consistency_errors: list[str],
    contract_errors: list[str],
    gate_policy: dict[str, str],
    predictions: list[dict[str, Any]],
    enforce_runtime_lock: bool,
    enforce_weights_hash: bool,
    peer_robust_metrics: dict[str, Any] | None,
    backend_parity: dict[str, Any],
    failure_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str], list[str]]:
    hard_failures: list[str] = []
    soft_failures: list[str] = []

    gates: dict[str, Any] = {
        GATE_SCHEMA: _new_gate(mode=gate_policy[GATE_SCHEMA], category="contract"),
        GATE_CONSISTENCY: _new_gate(mode=gate_policy[GATE_CONSISTENCY], category="contract"),
        GATE_METRIC: _new_gate(mode=gate_policy[GATE_METRIC], category="behavior"),
        GATE_SPEED: _new_gate(mode=gate_policy[GATE_SPEED], category="behavior"),
    }

    baseline = baseline_payload.get("baseline") or {}
    thresholds = baseline_payload.get("thresholds") or {}
    baseline_summary = baseline.get("summary") or {}
    baseline_robust = baseline.get("robust_metrics") or {}
    baseline_speed = baseline.get("speed") or {}
    baseline_contract = baseline.get("contract") or {}
    baseline_meta = baseline_payload.get("baseline_meta") or {}

    metric_thr = thresholds.get("metric") or {}
    speed_thr = thresholds.get("speed") or {}

    schema_gate = gates[GATE_SCHEMA]
    schema_gate["details"] = {
        "warnings": list(schema_warnings),
        "errors": list(schema_errors),
    }
    if str(schema_gate["mode"]) != "off":
        for msg in schema_warnings:
            schema_gate["ok"] = False
            _append_gate_failure(
                gate_key=GATE_SCHEMA,
                message=f"schema warning: {msg}",
                gate_policy=gate_policy,
                hard_failures=hard_failures,
                soft_failures=soft_failures,
                failure_records=failure_records,
            )
        for msg in schema_errors:
            schema_gate["ok"] = False
            _append_gate_failure(
                gate_key=GATE_SCHEMA,
                message=f"schema error: {msg}",
                gate_policy=gate_policy,
                hard_failures=hard_failures,
                soft_failures=soft_failures,
                failure_records=failure_records,
            )
    else:
        schema_gate["details"]["skipped"] = True

    consistency_gate = gates[GATE_CONSISTENCY]
    consistency_mismatches: list[str] = []
    consistency_warnings: list[str] = []
    if str(consistency_gate["mode"]) != "off":
        for key in (
            "record_images_sha256",
            "prediction_images_sha256",
            "record_images",
            "prediction_images",
        ):
            if key in baseline_contract and baseline_contract.get(key) != contract.get(key):
                consistency_mismatches.append(
                    f"{key} mismatch: baseline={baseline_contract.get(key)} current={contract.get(key)}"
                )

        for key in (
            "dataset_hash",
            "config_hash",
            "checkpoint_hash",
            "repro_policy",
            "canonical_decimals",
        ):
            ref = baseline_meta.get(key)
            cur = run_meta.get(key)
            if ref is None:
                continue
            if ref != cur:
                consistency_mismatches.append(f"{key} mismatch: baseline={ref} current={cur}")

        ref_lock_sha = baseline_meta.get("runtime_lock_sha256")
        cur_lock_sha = run_meta.get("runtime_lock_sha256")
        ref_lock_versions = dict(baseline_meta.get("runtime_lock_versions") or {})
        cur_lock_versions = dict(run_meta.get("runtime_lock_versions") or {})

        if bool(enforce_runtime_lock):
            if not ref_lock_sha:
                consistency_mismatches.append("runtime lock baseline missing runtime_lock_sha256")
            elif not cur_lock_sha:
                consistency_mismatches.append("runtime lock current run missing runtime_lock_sha256")
            elif ref_lock_sha != cur_lock_sha:
                consistency_mismatches.append(
                    f"runtime lock sha mismatch: baseline={ref_lock_sha} current={cur_lock_sha}"
                )

            if not ref_lock_versions:
                consistency_mismatches.append("runtime lock baseline missing runtime_lock_versions")
            elif not cur_lock_versions:
                consistency_mismatches.append("runtime lock current run missing runtime_lock_versions")
            elif ref_lock_versions != cur_lock_versions:
                consistency_mismatches.append(
                    f"runtime lock versions mismatch: baseline={ref_lock_versions} current={cur_lock_versions}"
                )
        else:
            if ref_lock_sha is not None and ref_lock_sha != cur_lock_sha:
                consistency_mismatches.append(
                    f"runtime_lock_sha256 mismatch: baseline={ref_lock_sha} current={cur_lock_sha}"
                )
            if ref_lock_versions and ref_lock_versions != cur_lock_versions:
                consistency_mismatches.append(
                    f"runtime_lock_versions mismatch: baseline={ref_lock_versions} current={cur_lock_versions}"
                )

        ref_weights_hash = baseline_meta.get("weights_hash")
        cur_weights_hash = run_meta.get("weights_hash")
        ref_checkpoint_hash = baseline_meta.get("checkpoint_hash")
        cur_checkpoint_hash = run_meta.get("checkpoint_hash")
        if bool(enforce_weights_hash):
            if not ref_weights_hash:
                consistency_mismatches.append("weights_hash missing in baseline_meta")
            elif not cur_weights_hash:
                consistency_mismatches.append("weights_hash missing in run_meta")
            elif ref_weights_hash != cur_weights_hash:
                consistency_mismatches.append(
                    f"weights_hash mismatch: baseline={ref_weights_hash} current={cur_weights_hash}"
                )
        elif ref_weights_hash is not None:
            # Default mode: checkpoint-backed comparisons are hard, checkpoint-free are warnings.
            if ref_checkpoint_hash is not None and cur_checkpoint_hash is not None:
                if ref_weights_hash != cur_weights_hash:
                    consistency_mismatches.append(
                        f"weights_hash mismatch: baseline={ref_weights_hash} current={cur_weights_hash}"
                    )
            elif ref_weights_hash != cur_weights_hash:
                consistency_warnings.append(
                    "weights_hash differs in checkpoint-free comparison; skipped hard consistency check"
                )

        consistency_mismatches.extend(contract_errors)
        consistency_mismatches.extend(consistency_errors)

        if consistency_mismatches:
            consistency_gate["ok"] = False
            for msg in consistency_mismatches:
                _append_gate_failure(
                    gate_key=GATE_CONSISTENCY,
                    message=msg,
                    gate_policy=gate_policy,
                    hard_failures=hard_failures,
                    soft_failures=soft_failures,
                    failure_records=failure_records,
                )
    else:
        consistency_gate["details"]["skipped"] = True
    if consistency_mismatches:
        first_image = None
        for key in ("prediction_images", "record_images"):
            images = contract.get(key)
            if isinstance(images, list) and images:
                first_image = str(images[0])
                break
        excerpt: list[dict[str, Any]] = []
        if predictions:
            first = predictions[0]
            for det in list(first.get("detections") or [])[:2]:
                excerpt.append(
                    {
                        "class_id": det.get("class_id"),
                        "score": det.get("score"),
                        "bbox": det.get("bbox"),
                    }
                )
            if first_image is None:
                first_image = str(first.get("image"))
        consistency_gate["details"]["first_counterexample"] = {
            "image": first_image,
            "detections_excerpt": excerpt,
            "mismatch": consistency_mismatches[0],
        }
    consistency_gate["details"]["mismatches"] = consistency_mismatches
    consistency_gate["details"]["warnings"] = consistency_warnings

    metric_gate = gates[GATE_METRIC]
    if str(metric_gate["mode"]) == "off":
        metric_gate["details"] = {"skipped": True}
    else:
        metric_checks = {
            "total_detections": float(metric_thr.get("total_detections_abs", 0.0)),
            "score_sum": float(metric_thr.get("score_sum_abs", 0.0)),
            "score_mean": float(metric_thr.get("score_mean_abs", 0.0)),
            "bbox_checksum": float(metric_thr.get("bbox_checksum_abs", 0.0)),
        }
        metric_deltas: dict[str, Any] = {}
        for key, tol in metric_checks.items():
            cur = float(summary.get(key, 0.0))
            ref = float(baseline_summary.get(key, 0.0))
            delta = abs(cur - ref)
            ok = bool(delta <= tol)
            metric_deltas[key] = {
                "baseline": ref,
                "current": cur,
                "abs_delta": delta,
                "allowed_abs": tol,
                "ok": ok,
            }
            if not ok:
                metric_gate["ok"] = False
                _append_gate_failure(
                    gate_key=GATE_METRIC,
                    message=f"{key} abs_delta={delta:.6f} exceeds tolerance={tol:.6f}",
                    gate_policy=gate_policy,
                    hard_failures=hard_failures,
                    soft_failures=soft_failures,
                    failure_records=failure_records,
                )
        robust_checks = {
            "map50": float(metric_thr.get("map50_abs", 0.0)),
            "map50_95": float(metric_thr.get("map50_95_abs", 0.0)),
            "worst_k_map50": float(metric_thr.get("worst_k_map50_abs", 0.0)),
            "median_class_map50": float(metric_thr.get("median_class_map50_abs", 0.0)),
            "recall_at_k": float(metric_thr.get("recall_at_k_abs", 0.0)),
            "iou_p10": float(metric_thr.get("iou_p10_abs", 0.0)),
            "iou_p50": float(metric_thr.get("iou_p50_abs", 0.0)),
            "missing_count": float(metric_thr.get("missing_count_abs", 0.0)),
            "extra_count": float(metric_thr.get("extra_count_abs", 0.0)),
            "class_mismatch_count": float(metric_thr.get("class_mismatch_count_abs", 0.0)),
        }
        robust_deltas: dict[str, Any] = {}
        missing_baseline_keys: list[str] = []
        for key, tol in robust_checks.items():
            if key not in baseline_robust:
                missing_baseline_keys.append(key)
                continue
            cur = float(robust_metrics.get(key, 0.0))
            ref = float(baseline_robust.get(key, 0.0))
            delta = abs(cur - ref)
            ok = bool(delta <= tol)
            robust_deltas[key] = {
                "baseline": ref,
                "current": cur,
                "abs_delta": delta,
                "allowed_abs": tol,
                "ok": ok,
            }
            if not ok:
                metric_gate["ok"] = False
                _append_gate_failure(
                    gate_key=GATE_METRIC,
                    message=f"{key} abs_delta={delta:.6f} exceeds tolerance={tol:.6f}",
                    gate_policy=gate_policy,
                    hard_failures=hard_failures,
                    soft_failures=soft_failures,
                    failure_records=failure_records,
                )

        parity_details: dict[str, Any] = {"mode": str(backend_parity.get("mode", "off"))}
        parity_mode = str(backend_parity.get("mode", "off"))
        if parity_mode == "off" or not peer_robust_metrics:
            parity_details["skipped"] = True
        else:
            checks = {
                "map50": float(backend_parity.get("map50_abs", 0.0)),
                "map50_95": float(backend_parity.get("map50_95_abs", 0.0)),
            }
            rows: dict[str, Any] = {}
            parity_ok = True
            for key, tol in checks.items():
                if key not in peer_robust_metrics:
                    rows[key] = {"missing_peer_metric": True, "allowed_abs": tol}
                    continue
                cur = float(robust_metrics.get(key, 0.0))
                peer = float(peer_robust_metrics.get(key, 0.0))
                delta = abs(cur - peer)
                ok = bool(delta <= tol)
                rows[key] = {
                    "current": cur,
                    "peer": peer,
                    "abs_delta": delta,
                    "allowed_abs": tol,
                    "ok": ok,
                }
                if not ok:
                    parity_ok = False
                    metric_gate["ok"] = False
                    _append_gate_failure(
                        gate_key=GATE_METRIC,
                        message=f"parity {key} abs_delta={delta:.6f} exceeds tolerance={tol:.6f}",
                        gate_policy=gate_policy,
                        hard_failures=hard_failures,
                        soft_failures=soft_failures,
                        failure_records=failure_records,
                        mode_override=parity_mode,
                    )
            parity_details["metrics"] = rows
            parity_details["ok"] = parity_ok

        class_hist_cur = summary.get("class_hist") or {}
        class_hist_ref = baseline_summary.get("class_hist") or {}
        class_delta_rows: list[dict[str, Any]] = []
        for class_id in sorted(set(class_hist_cur.keys()) | set(class_hist_ref.keys()), key=lambda x: int(str(x))):
            cur_v = int(class_hist_cur.get(class_id, 0))
            ref_v = int(class_hist_ref.get(class_id, 0))
            delta = cur_v - ref_v
            if delta != 0:
                class_delta_rows.append(
                    {
                        "class_id": int(class_id),
                        "baseline": ref_v,
                        "current": cur_v,
                        "delta": delta,
                        "abs_delta": abs(delta),
                    }
                )
        class_delta_rows.sort(key=lambda row: int(row["abs_delta"]), reverse=True)

        failing_metrics = [name for name, row in metric_deltas.items() if not bool(row.get("ok"))]
        failing_robust_metrics = [name for name, row in robust_deltas.items() if not bool(row.get("ok"))]
        metric_gate["details"] = {
            "metrics": metric_deltas,
            "robust_metrics": {
                "current": robust_metrics,
                "baseline": baseline_robust,
                "deltas": robust_deltas,
                "missing_baseline_metric_keys": missing_baseline_keys,
                "failed_metric_names": failing_robust_metrics,
            },
            "failed_metric_names": failing_metrics,
            "class_hist_topk": class_delta_rows[:5],
            "backend_parity": parity_details,
        }

    speed_gate = gates[GATE_SPEED]
    if str(speed_gate["mode"]) == "off":
        speed_gate["details"] = {"skipped": True}
    else:
        baseline_fps = float(baseline_speed.get("fps", 0.0))
        current_fps = float(speed.get("fps", 0.0))
        min_ratio = float(speed_thr.get("min_fps_ratio", 0.0))
        floor = float(speed_thr.get("absolute_floor_fps", 0.0))
        required_fps = max(floor, baseline_fps * min_ratio)
        speed_ok = bool(current_fps >= required_fps)
        speed_gate["ok"] = speed_ok
        ratio_vs_baseline = (current_fps / baseline_fps) if baseline_fps > 0 else None
        speed_gate["details"] = {
            "baseline_fps": baseline_fps,
            "current_fps": current_fps,
            "ratio_vs_baseline": ratio_vs_baseline,
            "required_min_fps": required_fps,
            "min_fps_ratio": min_ratio,
            "absolute_floor_fps": floor,
            "measurement": {
                "mode": "single_shot",
                "images": int(speed.get("images", 0)),
                "seconds": float(speed.get("seconds", 0.0)),
                "percentiles_fps": {
                    "p50": current_fps,
                    "p95": current_fps,
                },
            },
            "ok": speed_ok,
        }
        if not speed_ok:
            _append_gate_failure(
                gate_key=GATE_SPEED,
                message=(
                    "current_fps="
                    f"{current_fps:.3f} below required_min_fps={required_fps:.3f}"
                ),
                gate_policy=gate_policy,
                hard_failures=hard_failures,
                soft_failures=soft_failures,
                failure_records=failure_records,
            )

    return gates, hard_failures, soft_failures


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        max_images = require_non_negative_int(args.max_images, flag_name="--max-images")
        max_detections = require_positive_int(args.max_detections, flag_name="--max-detections")
        score_threshold = require_float_in_range(
            args.score_threshold,
            flag_name="--score-threshold",
            minimum=0.0,
            maximum=1.0,
        )
        init_seed = require_non_negative_int(args.init_seed, flag_name="--init-seed")
        canonical_decimals = require_non_negative_int(
            args.canonical_decimals,
            flag_name="--canonical-decimals",
        )
        metric_worst_k = require_positive_int(args.metric_worst_k, flag_name="--metric-worst-k")
        metric_recall_k = require_positive_int(args.metric_recall_k, flag_name="--metric-recall-k")
        require_float_in_range(
            args.min_fps_ratio,
            flag_name="--min-fps-ratio",
            minimum=0.0,
            maximum=1.0,
        )
        require_float_in_range(
            args.absolute_floor_fps,
            flag_name="--absolute-floor-fps",
            minimum=0.0,
            maximum=100000.0,
        )
        for flag, value in (
            ("--metric-total-detections-abs", args.metric_total_detections_abs),
            ("--metric-score-sum-abs", args.metric_score_sum_abs),
            ("--metric-score-mean-abs", args.metric_score_mean_abs),
            ("--metric-bbox-checksum-abs", args.metric_bbox_checksum_abs),
            ("--metric-map50-abs", args.metric_map50_abs),
            ("--metric-map50-95-abs", args.metric_map50_95_abs),
            ("--metric-worst-k-map50-abs", args.metric_worst_k_map50_abs),
            ("--metric-median-class-map50-abs", args.metric_median_class_map50_abs),
            ("--metric-recall-at-k-abs", args.metric_recall_at_k_abs),
            ("--metric-iou-p10-abs", args.metric_iou_p10_abs),
            ("--metric-iou-p50-abs", args.metric_iou_p50_abs),
            ("--metric-missing-count-abs", args.metric_missing_count_abs),
            ("--metric-extra-count-abs", args.metric_extra_count_abs),
            ("--metric-class-mismatch-abs", args.metric_class_mismatch_abs),
            ("--backend-parity-map50-abs", args.backend_parity_map50_abs),
            ("--backend-parity-map50-95-abs", args.backend_parity_map50_95_abs),
        ):
            require_float_in_range(value, flag_name=flag, minimum=0.0, maximum=100000.0)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    cwd = Path.cwd()
    dataset_root = resolve_input_path(args.dataset, cwd=cwd, repo_root=repo_root)
    config_path = resolve_input_path(args.config, cwd=cwd, repo_root=repo_root)
    checkpoint_path = (
        resolve_input_path(args.checkpoint, cwd=cwd, repo_root=repo_root)
        if args.checkpoint
        else None
    )
    baseline_path = _resolve_baseline_path(args=args, cwd=cwd)
    output_path = resolve_output_path(args.output, cwd=cwd)
    runtime_lock_path = resolve_input_path(args.runtime_lock, cwd=cwd, repo_root=repo_root)
    peer_report_path = (
        resolve_input_path(args.peer_report, cwd=cwd, repo_root=repo_root)
        if args.peer_report
        else None
    )
    _ensure_repo_write_target(baseline_path, flag_name="--baseline")
    _ensure_repo_write_target(output_path, flag_name="--output")
    if args.enforce_runtime_lock and not runtime_lock_path.exists():
        raise SystemExit(f"--runtime-lock not found: {runtime_lock_path}")
    if peer_report_path is not None and not peer_report_path.exists():
        raise SystemExit(f"--peer-report not found: {peer_report_path}")

    runtime_lock_meta = {
        "path": _repo_relative_display(runtime_lock_path),
        "sha256": (_sha256_file(runtime_lock_path) if runtime_lock_path.exists() else None),
        "versions": _parse_runtime_lock(runtime_lock_path),
    }

    image_size = _image_size(list(args.image_size))
    manifest = build_manifest(dataset_root, split=args.split)
    records = list(manifest.get("images") or [])
    if max_images is not None:
        records = records[: int(max_images)]
    if not records:
        raise SystemExit("no records to evaluate; check --dataset/--split/--max-images")

    repro_details = _configure_repro_policy(
        policy=str(args.repro_policy),
        seed=(None if str(args.repro_policy) == "off" else int(init_seed)),
    )

    adapter = RTDETRPoseAdapter(
        config_path=str(config_path),
        checkpoint_path=(str(checkpoint_path) if checkpoint_path is not None else None),
        device=str(args.device),
        image_size=image_size,
        score_threshold=float(score_threshold),
        max_detections=int(max_detections),
        init_seed=(None if str(args.repro_policy) == "off" else int(init_seed)),
        repro_policy=str(args.repro_policy),
    )

    started_utc = _now_utc()
    start = time.perf_counter()
    predictions_raw = adapter.predict(records)
    elapsed = time.perf_counter() - start
    finished_utc = _now_utc()

    predictions, canonicalization = _canonicalize_predictions(
        predictions_raw,
        dataset_root=dataset_root,
        decimals=int(canonical_decimals),
    )

    schema_errors = list(canonicalization.get("schema_errors") or [])
    schema_warnings = list(canonicalization.get("warnings") or [])
    try:
        validate_result = validate_predictions_entries(predictions, strict=True)
        schema_warnings.extend(list(validate_result.warnings))
    except ValueError as exc:
        schema_errors.append(str(exc))

    summary = _build_summary(predictions)
    robust_metrics = _build_robust_metrics(
        records=records,
        predictions=predictions,
        dataset_root=dataset_root,
        worst_k=int(metric_worst_k),
        recall_k=int(metric_recall_k),
    )
    contract, contract_errors = _build_contract(
        records,
        predictions,
        dataset_root=dataset_root,
    )

    consistency_errors = list(canonicalization.get("consistency_errors") or [])

    speed = {
        "images": int(len(records)),
        "seconds": float(round(elapsed, 6)),
        "fps": float(round((len(records) / elapsed) if elapsed > 0 else 0.0, 6)),
    }
    predictions_sha256 = _sha256_json(predictions)

    dataset_fingerprint = _dataset_fingerprint(
        records,
        dataset_root=dataset_root,
        split=str(manifest.get("split")),
        max_images=int(len(records)),
    )
    provenance = _collect_provenance(
        capture_mode=str(args.capture_provenance),
        runtime_lock=runtime_lock_meta,
    )

    run_meta = _collect_run_meta(
        adapter=adapter,
        args=args,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        dataset_fingerprint=dataset_fingerprint,
        repro_details=repro_details,
        canonicalization=canonicalization,
        runtime_lock=runtime_lock_meta,
        provenance=provenance,
    )
    if bool(args.enforce_runtime_lock):
        pinned_versions = dict(runtime_lock_meta.get("versions") or {})
        for key in RUNTIME_LOCK_KEYS:
            expected = pinned_versions.get(key)
            if expected is None:
                consistency_errors.append(f"runtime lock missing pin for package '{key}'")
                continue
            actual = (run_meta.get("versions") or {}).get(key)
            if actual is None:
                consistency_errors.append(f"runtime package '{key}' not installed (expected {expected})")
                continue
            if str(actual) != str(expected):
                consistency_errors.append(f"runtime lock mismatch for {key}: expected={expected} actual={actual}")
    if args.expected_dataset_hash:
        expected_dataset_hash = str(args.expected_dataset_hash).strip()
        current_dataset_hash = str(run_meta.get("dataset_hash") or "")
        if expected_dataset_hash != current_dataset_hash:
            consistency_errors.append(
                f"dataset_hash mismatch: expected={expected_dataset_hash} current={current_dataset_hash}"
            )
    if args.expected_weights_hash:
        expected_weights_hash = str(args.expected_weights_hash).strip()
        current_weights_hash = str(run_meta.get("weights_hash") or "")
        if expected_weights_hash != current_weights_hash:
            consistency_errors.append(
                f"weights_hash mismatch: expected={expected_weights_hash} current={current_weights_hash}"
            )
    if args.expected_checkpoint_hash:
        expected_checkpoint_hash = str(args.expected_checkpoint_hash).strip()
        current_checkpoint_hash = str(run_meta.get("checkpoint_hash") or "")
        if expected_checkpoint_hash != current_checkpoint_hash:
            consistency_errors.append(
                f"checkpoint_hash mismatch: expected={expected_checkpoint_hash} current={current_checkpoint_hash}"
            )

    run_context = {
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "dataset": _repo_relative_display(dataset_root),
        "split": str(manifest.get("split")),
        "adapter": str(args.adapter_id),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "device": str(args.device),
        "profile": str(args.profile),
        "baseline_layout": str(args.baseline_layout),
        "baseline_path": str(baseline_path),
        "peer_report": (str(peer_report_path) if peer_report_path is not None else None),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "score_threshold": float(score_threshold),
        "max_detections": int(max_detections),
        "init_seed": (None if str(args.repro_policy) == "off" else int(init_seed)),
        "repro_policy": str(args.repro_policy),
        "records": int(len(records)),
    }

    gate_policy = _gate_policy_from_args(args)
    protocol = _protocol_spec(canonical_decimals=int(canonical_decimals))

    hard_failures: list[str] = []
    soft_failures: list[str] = []
    failure_records: list[dict[str, Any]] = []
    gates: dict[str, Any]
    baseline_payload: dict[str, Any]
    peer_robust_metrics: dict[str, Any] | None = None
    if peer_report_path is not None:
        peer_payload = json.loads(peer_report_path.read_text(encoding="utf-8"))
        peer_robust_metrics = (
            peer_payload.get("robust_metrics")
            or ((peer_payload.get("baseline") or {}).get("robust_metrics"))
            or ((peer_payload.get("summary") or {}).get("robust_metrics"))
        )

    if args.write_baseline:
        baseline_payload = _build_baseline_payload(
            args=args,
            baseline_path=baseline_path,
            dataset_root=dataset_root,
            split=str(manifest.get("split")),
            summary=summary,
            robust_metrics=robust_metrics,
            speed=speed,
            contract=contract,
            predictions_sha256=predictions_sha256,
            run_meta=run_meta,
            protocol=protocol,
            gate_policy=gate_policy,
            canonicalization=canonicalization,
        )
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(baseline_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        gates = {
            GATE_SCHEMA: _new_gate(mode=gate_policy[GATE_SCHEMA], category="contract"),
            GATE_CONSISTENCY: _new_gate(mode=gate_policy[GATE_CONSISTENCY], category="contract"),
            GATE_METRIC: _new_gate(mode="off", category="behavior"),
            GATE_SPEED: _new_gate(mode="off", category="behavior"),
        }

        if gate_policy[GATE_SCHEMA] != "off":
            gates[GATE_SCHEMA]["details"] = {
                "warnings": schema_warnings,
                "errors": schema_errors,
                "mode": "baseline_write",
            }
            for msg in schema_warnings:
                gates[GATE_SCHEMA]["ok"] = False
                _append_gate_failure(
                    gate_key=GATE_SCHEMA,
                    message=f"schema warning: {msg}",
                    gate_policy=gate_policy,
                    hard_failures=hard_failures,
                    soft_failures=soft_failures,
                    failure_records=failure_records,
                )
            for msg in schema_errors:
                gates[GATE_SCHEMA]["ok"] = False
                _append_gate_failure(
                    gate_key=GATE_SCHEMA,
                    message=f"schema error: {msg}",
                    gate_policy=gate_policy,
                    hard_failures=hard_failures,
                    soft_failures=soft_failures,
                    failure_records=failure_records,
                )
        else:
            gates[GATE_SCHEMA]["details"] = {"skipped": True, "mode": "baseline_write"}

        baseline_write_consistency = [*contract_errors, *consistency_errors]
        if gate_policy[GATE_CONSISTENCY] != "off":
            gates[GATE_CONSISTENCY]["details"] = {
                "errors": baseline_write_consistency,
                "mode": "baseline_write",
            }
            if baseline_write_consistency:
                gates[GATE_CONSISTENCY]["ok"] = False
                for msg in baseline_write_consistency:
                    _append_gate_failure(
                        gate_key=GATE_CONSISTENCY,
                        message=msg,
                        gate_policy=gate_policy,
                        hard_failures=hard_failures,
                        soft_failures=soft_failures,
                        failure_records=failure_records,
                    )
        else:
            gates[GATE_CONSISTENCY]["details"] = {"skipped": True, "mode": "baseline_write"}

        gates[GATE_METRIC]["details"] = {"mode": "baseline_write", "skipped": True}
        gates[GATE_SPEED]["details"] = {"mode": "baseline_write", "skipped": True}
    else:
        if not baseline_path.exists():
            raise SystemExit(
                f"baseline not found: {baseline_path} (run with --write-baseline to create it)"
            )
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        gates, hard_failures, soft_failures = _compare_against_baseline(
            baseline_payload=baseline_payload,
            summary=summary,
            robust_metrics=robust_metrics,
            speed=speed,
            contract=contract,
            run_meta=run_meta,
            schema_warnings=schema_warnings,
            schema_errors=schema_errors,
            consistency_errors=consistency_errors,
            contract_errors=contract_errors,
            gate_policy=gate_policy,
            predictions=predictions,
            enforce_runtime_lock=bool(args.enforce_runtime_lock),
            enforce_weights_hash=bool(args.enforce_weights_hash),
            peer_robust_metrics=peer_robust_metrics,
            backend_parity=(_thresholds_from_args(args).get("backend_parity") or {}),
            failure_records=failure_records,
        )

    report = {
        "schema_version": 2,
        "ok": len(hard_failures) == 0,
        "run": run_context,
        "run_meta": run_meta,
        "baseline_meta": (baseline_payload.get("baseline_meta") if not args.write_baseline else run_meta),
        "protocol": protocol,
        "gate_policy": gate_policy,
        "summary": summary,
        "robust_metrics": robust_metrics,
        "thresholds": _thresholds_from_args(args),
        "speed": speed,
        "contract": contract,
        "canonicalization": canonicalization,
        "predictions_sha256": predictions_sha256,
        "gates": gates,
        "failures": hard_failures,
        "hard_failures": hard_failures,
        "soft_failures": soft_failures,
        "failure_records": failure_records,
        "warnings": soft_failures,
        "baseline_path": str(baseline_path),
        "peer_report_path": (str(peer_report_path) if peer_report_path is not None else None),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path)

    if soft_failures:
        print("reference adapter regression soft warnings:", file=sys.stderr)
        for item in soft_failures:
            print(f"- {item}", file=sys.stderr)

    if hard_failures:
        raise SystemExit("reference adapter regression failed:\n- " + "\n- ".join(hard_failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
