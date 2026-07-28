import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest
from yolozu.core.cli_args import (require_float_in_range, require_non_negative_float, require_non_negative_int)
from yolozu.distillation import distill_predictions
from yolozu.metrics_report import build_report, write_json
from yolozu.predictions import canonicalize_predictions, normalize_predictions_payload
from yolozu.predictions import validate_predictions_payload
from yolozu.simple_map import evaluate_map


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Blend teacher/student prediction artifacts and emit a bounded research report."
    )
    p.add_argument("--student", required=True, help="Student predictions JSON.")
    p.add_argument("--teacher", required=True, help="Teacher predictions JSON.")
    p.add_argument("--output", default="reports/predictions_distilled.json", help="Output predictions JSON.")
    p.add_argument("--output-report", default="reports/distill_report.json", help="Output report JSON.")
    p.add_argument("--dataset", default=None, help="Optional dataset root for mAP evaluation.")
    p.add_argument("--split", default=None, help="Dataset split override.")
    p.add_argument("--config", default=None, help="Optional JSON/YAML config (enabled + params).")
    p.add_argument("--iou-threshold", type=float, default=0.7, help="IoU threshold for matching.")
    p.add_argument("--alpha", type=float, default=0.5, help="Blend factor for student/teacher scores.")
    p.add_argument("--add-missing", action="store_true", help="Add unmatched teacher detections.")
    p.add_argument("--add-score-scale", type=float, default=0.5, help="Scale for added teacher scores.")
    p.add_argument(
        "--teacher-min-score",
        type=float,
        default=0.0,
        help="Minimum teacher score for unmatched detection injection.",
    )
    p.add_argument(
        "--max-added-per-image",
        type=int,
        default=None,
        help="Maximum unmatched teacher detections to inject per image.",
    )
    p.add_argument(
        "--add-duplicate-iou-threshold",
        type=float,
        default=0.9,
        help="IoU threshold for duplicate-suppression when injecting missing detections.",
    )
    return p.parse_args(argv)


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_simple_yaml_value(raw: str) -> Any:
    value = raw.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml_object(text: str, *, source: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line[:1].isspace():
            raise SystemExit(f"Unsupported nested YAML in {source}:{lineno}; use a flat top-level mapping")
        if ":" not in line:
            raise SystemExit(f"Invalid YAML line in {source}:{lineno}: {line}")
        key, raw = line.split(":", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid YAML key in {source}:{lineno}")
        data[key] = _parse_simple_yaml_value(raw)
    return data


def _supplement_wrapped_meta(payload: Any, *, source: str) -> Any:
    if not isinstance(payload, dict) or "predictions" not in payload:
        return payload
    meta = payload.get("meta")
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        return payload

    supplemented = dict(meta)
    preds = payload.get("predictions", [])
    supplemented.setdefault("timestamp", _now_utc())
    supplemented.setdefault("adapter", "unknown")
    supplemented.setdefault("config", "")
    supplemented.setdefault("images", len(preds) if isinstance(preds, list) else 0)
    supplemented.setdefault(
        "tta",
        {
            "enabled": False,
            "seed": None,
            "flip_prob": 0.0,
            "norm_only": False,
            "warnings": [f"meta supplemented for distillation input: {source}"],
            "summary": None,
        },
    )
    supplemented.setdefault(
        "ttt",
        {
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
            "mim": {"mask_prob": 0.0, "patch_size": 1, "mask_value": 0.0},
            "report": None,
        },
    )

    wrapped = dict(payload)
    wrapped["meta"] = supplemented
    return wrapped


def _load_config(path_str: str | None) -> dict[str, Any]:
    if not path_str:
        return {}
    config_path = Path(path_str)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        data = _load_simple_yaml_object(text, source=config_path)
    else:
        data = json.loads(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"--config must point to a JSON/YAML object: {config_path}")
    return data


def _load_prediction_entries(path_str: str) -> list[dict[str, Any]]:
    path = Path(path_str)
    if not path.is_absolute():
        path = repo_root / path
    raw = json.loads(path.read_text(encoding="utf-8"))
    supplemented = _supplement_wrapped_meta(raw, source=str(path))
    validate_predictions_payload(supplemented, strict=False)
    entries, _ = normalize_predictions_payload(supplemented)
    canonical = canonicalize_predictions(entries, strict=False, policy="clamp")
    return canonical.entries


def _safe_metrics(records: list[dict[str, Any]], entries: list[dict[str, Any]]) -> dict[str, float]:
    result = evaluate_map(records, entries, iou_thresholds=[0.5 + 0.05 * i for i in range(10)])
    return {"map50": result.map50, "map50_95": result.map50_95}


def _validate_distill_params(params: dict[str, Any]) -> None:
    try:
        require_float_in_range(params.get("iou_threshold"), flag_name="--iou-threshold", minimum=0.0, maximum=1.0)
        require_float_in_range(params.get("alpha"), flag_name="--alpha", minimum=0.0, maximum=1.0)
        require_non_negative_float(params.get("add_score_scale"), flag_name="--add-score-scale")
        require_float_in_range(
            params.get("teacher_min_score"),
            flag_name="--teacher-min-score",
            minimum=0.0,
            maximum=1.0,
        )
        require_non_negative_int(params.get("max_added_per_image"), flag_name="--max-added-per-image")
        require_float_in_range(
            params.get("add_duplicate_iou_threshold"),
            flag_name="--add-duplicate-iou-threshold",
            minimum=0.0,
            maximum=1.0,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    started = time.perf_counter()

    config = _load_config(args.config)

    enabled = bool(config.get("enabled", True))

    student_entries = _load_prediction_entries(args.student)
    teacher_entries = _load_prediction_entries(args.teacher)

    distill_params = {
        "iou_threshold": float(config.get("iou_threshold", args.iou_threshold)),
        "alpha": float(config.get("alpha", args.alpha)),
        "add_missing": bool(config.get("add_missing", args.add_missing)),
        "add_score_scale": float(config.get("add_score_scale", args.add_score_scale)),
        "teacher_min_score": float(config.get("teacher_min_score", args.teacher_min_score)),
        "max_added_per_image": config.get("max_added_per_image", args.max_added_per_image),
        "add_duplicate_iou_threshold": float(
            config.get("add_duplicate_iou_threshold", args.add_duplicate_iou_threshold)
        ),
    }
    _validate_distill_params(distill_params)

    if enabled:
        distilled_entries, stats = distill_predictions(student_entries, teacher_entries, **distill_params)
    else:
        distilled_entries = student_entries
        stats = None

    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_predictions_payload(distilled_entries, strict=False)
    output_path.write_text(json.dumps(distilled_entries, indent=2, sort_keys=True), encoding="utf-8")
    elapsed_seconds = time.perf_counter() - started

    metrics = {}
    if args.dataset:
        dataset_root = Path(args.dataset)
        if not dataset_root.is_absolute():
            dataset_root = repo_root / dataset_root
        manifest = build_manifest(dataset_root, split=args.split)
        records = manifest["images"]
        metrics["student"] = _safe_metrics(records, student_entries)
        metrics["distilled"] = _safe_metrics(records, distilled_entries)

    report = build_report(
        losses={"distill_score_gap": getattr(stats, "avg_score_gap", 0.0) if stats else 0.0},
        metrics=metrics,
        meta={
            "enabled": enabled,
            "student": args.student,
            "teacher": args.teacher,
            "output": str(output_path),
            "distill": distill_params,
            "matched": getattr(stats, "matched", 0) if stats else 0,
            "added": getattr(stats, "added", 0) if stats else 0,
        },
    )

    report_path = _resolve(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    student_path = _resolve(args.student)
    teacher_path = _resolve(args.teacher)
    report["research_report"] = {
        "kind": "research_lane_report",
        "lane": "distillation",
        "stable_baseline_artifact": str(student_path),
        "research_output_artifact": str(output_path),
        "report_artifact": str(report_path),
        "latency_overhead": {
            "source": "measured_by_distill_predictions",
            "unit": "seconds",
            "total": float(elapsed_seconds),
            "prediction_images": len(distilled_entries),
            "seconds_per_image": float(elapsed_seconds / max(1, len(distilled_entries))),
        },
        "artifact_hashes": {
            "student_sha256": _sha256(student_path),
            "teacher_sha256": _sha256(teacher_path),
            "output_sha256": _sha256(output_path),
        },
        "rollback": {
            "status": "separate_artifact",
            "reason": "Distillation writes a separate output and does not mutate either input artifact.",
        },
        "promotion_gate": {
            "decision": "review_required",
            "reason": "Evaluate the distilled artifact with the stable task evaluator before any promotion.",
        },
    }
    write_json(report_path, report)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
