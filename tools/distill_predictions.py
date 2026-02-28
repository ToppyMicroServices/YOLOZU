import argparse
import json
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest
from yolozu.core.cli_args import (require_float_in_range, require_non_negative_float, require_non_negative_int)
from yolozu.distillation import distill_predictions
from yolozu.metrics_report import build_report, write_json
from yolozu.predictions import load_predictions_entries
from yolozu.predictions import validate_predictions_payload
from yolozu.simple_map import evaluate_map


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True, help="Student predictions JSON.")
    p.add_argument("--teacher", required=True, help="Teacher predictions JSON.")
    p.add_argument("--output", default="reports/predictions_distilled.json", help="Output predictions JSON.")
    p.add_argument("--output-report", default="reports/distill_report.json", help="Output report JSON.")
    p.add_argument("--dataset", default=None, help="Optional dataset root for mAP evaluation.")
    p.add_argument("--split", default=None, help="Dataset split override.")
    p.add_argument("--config", default=None, help="Optional JSON config (enabled + params).")
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

    config = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = repo_root / config_path
        config = json.loads(config_path.read_text())

    enabled = bool(config.get("enabled", True))

    student_raw = json.loads(Path(args.student).read_text(encoding="utf-8"))
    teacher_raw = json.loads(Path(args.teacher).read_text(encoding="utf-8"))
    validate_predictions_payload(student_raw, strict=False)
    validate_predictions_payload(teacher_raw, strict=False)
    student_entries = load_predictions_entries(args.student)
    teacher_entries = load_predictions_entries(args.teacher)

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

    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_predictions_payload(distilled_entries, strict=False)
    output_path.write_text(json.dumps(distilled_entries, indent=2, sort_keys=True))

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

    report_path = repo_root / args.output_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
