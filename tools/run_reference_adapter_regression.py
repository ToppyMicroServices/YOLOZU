#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
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
from yolozu.dataset import build_manifest
from yolozu.predictions import validate_predictions_entries

DEFAULT_BASELINE = "baselines/reference_adapter/rtdetr_pose_real_multitask_fewshot.json"
DEFAULT_DATASET = "data/real_multitask_fewshot"
DEFAULT_SPLIT = "val"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run RT-DETR reference-adapter regression on a fixed real-image dataset and "
            "enforce schema/consistency/metric/speed gates."
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
        help="Speed gate lower bound as ratio against baseline fps (default: 0.25).",
    )
    p.add_argument(
        "--absolute-floor-fps",
        type=float,
        default=0.2,
        help="Absolute minimum fps floor for speed gate (default: 0.2).",
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


def _hash_list(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


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


def _canonical_image_key(image: str, *, dataset_root: Path) -> str:
    text = str(image)
    try:
        img = Path(text).resolve()
        root = dataset_root.resolve()
        return str(img.relative_to(root).as_posix())
    except Exception:
        return text


def _build_contract(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    dataset_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    record_images = [
        _canonical_image_key(str(r.get("image")), dataset_root=dataset_root)
        for r in records
    ]
    prediction_images = [
        _canonical_image_key(str((e or {}).get("image")), dataset_root=dataset_root)
        for e in predictions
    ]

    if len(predictions) != len(records):
        errors.append(
            f"entry count mismatch: predictions={len(predictions)} records={len(records)}"
        )

    if prediction_images != record_images:
        errors.append("image order mismatch between records and predictions")

    contract = {
        "record_images_sha256": _hash_list(record_images),
        "prediction_images_sha256": _hash_list(prediction_images),
        "record_images": record_images,
        "prediction_images": prediction_images,
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


def _build_baseline_payload(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    split: str,
    summary: dict[str, Any],
    speed: dict[str, Any],
    contract: dict[str, Any],
    predictions_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "reference_adapter": "rtdetr_pose",
        "generated_utc": _now_utc(),
        "dataset": {
            "path": _repo_relative_display(dataset_root),
            "split": str(split),
            "max_images": int(args.max_images),
        },
        "adapter": {
            "config": str(args.config),
            "checkpoint": (str(args.checkpoint) if args.checkpoint else None),
            "device": str(args.device),
            "image_size": [int(v) for v in _image_size(list(args.image_size))],
            "score_threshold": float(args.score_threshold),
            "max_detections": int(args.max_detections),
            "init_seed": int(args.init_seed),
        },
        "thresholds": _thresholds_from_args(args),
        "baseline": {
            "summary": summary,
            "speed": speed,
            "contract": contract,
            "predictions_sha256": predictions_sha256,
        },
    }


def _compare_against_baseline(
    *,
    baseline_payload: dict[str, Any],
    summary: dict[str, Any],
    speed: dict[str, Any],
    contract: dict[str, Any],
    schema_warnings: list[str],
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    gates: dict[str, Any] = {
        "schema_drift": {"ok": True, "details": {}},
        "consistency_drift": {"ok": True, "details": {}},
        "metric_drift": {"ok": True, "details": {}},
        "speed_drift": {"ok": True, "details": {}},
    }

    baseline = baseline_payload.get("baseline") or {}
    thresholds = baseline_payload.get("thresholds") or {}

    baseline_summary = baseline.get("summary") or {}
    baseline_speed = baseline.get("speed") or {}
    baseline_contract = baseline.get("contract") or {}

    metric_thr = thresholds.get("metric") or {}
    speed_thr = thresholds.get("speed") or {}

    gates["schema_drift"]["details"]["warnings"] = list(schema_warnings)

    # Consistency gate: zero tolerance for interface contract order/mapping drift.
    consistency_mismatches: list[str] = []
    for key in ("record_images_sha256", "prediction_images_sha256"):
        if baseline_contract.get(key) != contract.get(key):
            consistency_mismatches.append(
                f"{key} mismatch: baseline={baseline_contract.get(key)} current={contract.get(key)}"
            )
    if baseline_summary.get("detections_per_image") != summary.get("detections_per_image"):
        consistency_mismatches.append(
            "detections_per_image mismatch: "
            f"baseline={baseline_summary.get('detections_per_image')} current={summary.get('detections_per_image')}"
        )
    if baseline_summary.get("class_hist") != summary.get("class_hist"):
        consistency_mismatches.append(
            f"class_hist mismatch: baseline={baseline_summary.get('class_hist')} current={summary.get('class_hist')}"
        )
    if consistency_mismatches:
        gates["consistency_drift"]["ok"] = False
        gates["consistency_drift"]["details"]["mismatches"] = consistency_mismatches
        failures.extend([f"[consistency_drift] {msg}" for msg in consistency_mismatches])

    # Metric gate with explicit tolerances.
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
        metric_deltas[key] = {
            "baseline": ref,
            "current": cur,
            "abs_delta": delta,
            "allowed_abs": tol,
            "ok": bool(delta <= tol),
        }
        if delta > tol:
            gates["metric_drift"]["ok"] = False
            failures.append(
                f"[metric_drift] {key} abs_delta={delta:.6f} exceeds tolerance={tol:.6f}"
            )
    gates["metric_drift"]["details"]["metrics"] = metric_deltas

    # Speed gate.
    baseline_fps = float(baseline_speed.get("fps", 0.0))
    current_fps = float(speed.get("fps", 0.0))
    min_ratio = float(speed_thr.get("min_fps_ratio", 0.0))
    floor = float(speed_thr.get("absolute_floor_fps", 0.0))
    required_fps = max(floor, baseline_fps * min_ratio)
    speed_ok = bool(current_fps >= required_fps)
    gates["speed_drift"]["ok"] = speed_ok
    gates["speed_drift"]["details"] = {
        "baseline_fps": baseline_fps,
        "current_fps": current_fps,
        "required_min_fps": required_fps,
        "min_fps_ratio": min_ratio,
        "absolute_floor_fps": floor,
        "ok": speed_ok,
    }
    if not speed_ok:
        failures.append(
            "[speed_drift] current_fps="
            f"{current_fps:.3f} below required_min_fps={required_fps:.3f}"
        )

    return gates, failures


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
        require_non_negative_int(args.init_seed, flag_name="--init-seed")
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

    image_size = _image_size(list(args.image_size))
    manifest = build_manifest(dataset_root, split=args.split)
    records = list(manifest.get("images") or [])
    if max_images is not None:
        records = records[: int(max_images)]
    if not records:
        raise SystemExit("no records to evaluate; check --dataset/--split/--max-images")

    adapter = RTDETRPoseAdapter(
        config_path=str(config_path),
        checkpoint_path=(str(checkpoint_path) if checkpoint_path is not None else None),
        device=str(args.device),
        image_size=image_size,
        score_threshold=float(score_threshold),
        max_detections=int(max_detections),
        init_seed=int(args.init_seed),
    )

    started_utc = _now_utc()
    start = time.perf_counter()
    predictions = adapter.predict(records)
    elapsed = time.perf_counter() - start
    finished_utc = _now_utc()

    validate_result = validate_predictions_entries(predictions, strict=True)
    schema_warnings = list(validate_result.warnings)

    summary = _build_summary(predictions)
    contract, contract_errors = _build_contract(
        records,
        predictions,
        dataset_root=dataset_root,
    )

    speed = {
        "images": int(len(records)),
        "seconds": float(round(elapsed, 6)),
        "fps": float(round((len(records) / elapsed) if elapsed > 0 else 0.0, 6)),
    }
    predictions_sha256 = _sha256_json(predictions)

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
        "init_seed": int(args.init_seed),
        "records": int(len(records)),
    }

    failures: list[str] = []
    gates: dict[str, Any]

    if args.write_baseline:
        baseline_payload = _build_baseline_payload(
            args=args,
            dataset_root=dataset_root,
            split=str(manifest.get("split")),
            summary=summary,
            speed=speed,
            contract=contract,
            predictions_sha256=predictions_sha256,
        )
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(baseline_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        gates = {
            "schema_drift": {"ok": True, "details": {"warnings": schema_warnings}},
            "consistency_drift": {"ok": len(contract_errors) == 0, "details": {"errors": contract_errors}},
            "metric_drift": {"ok": True, "details": {"mode": "baseline_write"}},
            "speed_drift": {"ok": True, "details": {"mode": "baseline_write"}},
        }
        if contract_errors:
            failures.extend([f"[consistency_drift] {msg}" for msg in contract_errors])
    else:
        if not baseline_path.exists():
            raise SystemExit(
                f"baseline not found: {baseline_path} (run with --write-baseline to create it)"
            )
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        gates, failures = _compare_against_baseline(
            baseline_payload=baseline_payload,
            summary=summary,
            speed=speed,
            contract=contract,
            schema_warnings=schema_warnings,
        )
        if contract_errors:
            gates["consistency_drift"]["ok"] = False
            gates["consistency_drift"]["details"].setdefault("errors", [])
            gates["consistency_drift"]["details"]["errors"].extend(contract_errors)
            failures.extend([f"[consistency_drift] {msg}" for msg in contract_errors])

    report = {
        "schema_version": 1,
        "ok": len(failures) == 0,
        "run": run_context,
        "summary": summary,
        "speed": speed,
        "contract": contract,
        "predictions_sha256": predictions_sha256,
        "gates": gates,
        "failures": failures,
        "baseline_path": str(baseline_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path)

    if failures:
        raise SystemExit("reference adapter regression failed:\n- " + "\n- ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
