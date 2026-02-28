#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.adapter import RTDETRPoseAdapter
from yolozu.core.cli_args import (
    require_float_in_range,
    require_non_negative_int,
    require_positive_int,
    resolve_input_path,
    resolve_output_path,
)
from yolozu.core.image_keys import require_image_key
from yolozu.dataset import build_manifest
from yolozu.predictions import canonicalize_predictions, validate_predictions_entries

DEFAULT_BASELINE = "baselines/reference_adapter/rtdetr_pose_smoke_val.json"
DEFAULT_DATASET = "data/smoke"
DEFAULT_SPLIT = "val"

GATE_SCHEMA = "schema_drift"
GATE_CONSISTENCY = "consistency_drift"
GATE_METRIC = "metric_drift"
GATE_SPEED = "speed_drift"


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
    p.add_argument("--baseline", default=DEFAULT_BASELINE, help="Baseline JSON path.")
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
) -> dict[str, Any]:
    model_dtype: str | None = None
    model_state_hash: str | None = None
    model_hash_source: str | None = None
    backend = "torch"

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
    }


def _build_baseline_payload(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    split: str,
    summary: dict[str, Any],
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
        "reference_adapter": "rtdetr_pose",
        "generated_utc": _now_utc(),
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


def _append_gate_failure(
    *,
    gate_key: str,
    message: str,
    gate_policy: dict[str, str],
    hard_failures: list[str],
    soft_failures: list[str],
) -> None:
    mode = str(gate_policy.get(gate_key, "hard"))
    line = f"[{gate_key}] {message}"
    if mode == "hard":
        hard_failures.append(line)
    elif mode == "warn":
        soft_failures.append(line)


def _compare_against_baseline(
    *,
    baseline_payload: dict[str, Any],
    summary: dict[str, Any],
    speed: dict[str, Any],
    contract: dict[str, Any],
    run_meta: dict[str, Any],
    schema_warnings: list[str],
    schema_errors: list[str],
    consistency_errors: list[str],
    contract_errors: list[str],
    gate_policy: dict[str, str],
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
            )
        for msg in schema_errors:
            schema_gate["ok"] = False
            _append_gate_failure(
                gate_key=GATE_SCHEMA,
                message=f"schema error: {msg}",
                gate_policy=gate_policy,
                hard_failures=hard_failures,
                soft_failures=soft_failures,
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

        ref_weights_hash = baseline_meta.get("weights_hash")
        cur_weights_hash = run_meta.get("weights_hash")
        ref_checkpoint_hash = baseline_meta.get("checkpoint_hash")
        cur_checkpoint_hash = run_meta.get("checkpoint_hash")
        if ref_weights_hash is not None:
            # Only enforce weights hash as a hard contract when both sides are checkpoint-backed.
            # Checkpoint-free runs may vary across torch/python versions even with fixed seed.
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
                )
    else:
        consistency_gate["details"]["skipped"] = True
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
                )
        metric_gate["details"] = {"metrics": metric_deltas}

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
        speed_gate["details"] = {
            "baseline_fps": baseline_fps,
            "current_fps": current_fps,
            "required_min_fps": required_fps,
            "min_fps_ratio": min_ratio,
            "absolute_floor_fps": floor,
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
    baseline_path = resolve_output_path(args.baseline, cwd=cwd)
    output_path = resolve_output_path(args.output, cwd=cwd)
    _ensure_repo_write_target(baseline_path, flag_name="--baseline")
    _ensure_repo_write_target(output_path, flag_name="--output")

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

    run_meta = _collect_run_meta(
        adapter=adapter,
        args=args,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        dataset_fingerprint=dataset_fingerprint,
        repro_details=repro_details,
        canonicalization=canonicalization,
    )

    run_context = {
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "dataset": _repo_relative_display(dataset_root),
        "split": str(manifest.get("split")),
        "adapter": "rtdetr_pose",
        "config": str(config_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "device": str(args.device),
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
    gates: dict[str, Any]
    baseline_payload: dict[str, Any]

    if args.write_baseline:
        baseline_payload = _build_baseline_payload(
            args=args,
            dataset_root=dataset_root,
            split=str(manifest.get("split")),
            summary=summary,
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
                )
            for msg in schema_errors:
                gates[GATE_SCHEMA]["ok"] = False
                _append_gate_failure(
                    gate_key=GATE_SCHEMA,
                    message=f"schema error: {msg}",
                    gate_policy=gate_policy,
                    hard_failures=hard_failures,
                    soft_failures=soft_failures,
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
            speed=speed,
            contract=contract,
            run_meta=run_meta,
            schema_warnings=schema_warnings,
            schema_errors=schema_errors,
            consistency_errors=consistency_errors,
            contract_errors=contract_errors,
            gate_policy=gate_policy,
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
        "speed": speed,
        "contract": contract,
        "canonicalization": canonicalization,
        "predictions_sha256": predictions_sha256,
        "gates": gates,
        "failures": hard_failures,
        "hard_failures": hard_failures,
        "soft_failures": soft_failures,
        "warnings": soft_failures,
        "baseline_path": str(baseline_path),
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
