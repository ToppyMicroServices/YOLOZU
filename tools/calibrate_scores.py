import argparse
import json
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
_NUMERIC_ERRORS = (TypeError, ValueError, OverflowError)

from yolozu.calibration import calibrate_predictions_entries
from yolozu.dataset import build_manifest
from yolozu.metrics_report import build_report, write_json
from yolozu.predictions import load_predictions_entries
from yolozu.simple_map import evaluate_map


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    p.add_argument("--predictions", required=True, help="Path to predictions JSON.")
    p.add_argument("--output", default="reports/predictions_calibrated.json", help="Output predictions JSON.")
    p.add_argument("--output-report", default="reports/calibration_report.json", help="Output report JSON.")
    p.add_argument("--output-artifact", default="reports/calibration_artifact.json", help="Output artifact JSON.")
    p.add_argument("--split", default=None, help="Dataset split override.")
    p.add_argument("--temperatures", default="0.5,1.0,1.5,2.0", help="Comma-separated temperature grid.")
    p.add_argument("--min-score", type=float, default=None, help="Optional min score clamp.")
    p.add_argument("--max-score", type=float, default=None, help="Optional max score clamp.")
    return p.parse_args(argv)


def _parse_grid(text: str) -> list[float]:
    out = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except _NUMERIC_ERRORS:
            continue
    return out


def _safe_metrics(records: list[dict[str, Any]], entries: list[dict[str, Any]]) -> dict[str, float]:
    result = evaluate_map(records, entries, iou_thresholds=[0.5 + 0.05 * i for i in range(10)])
    return {"map50": result.map50, "map50_95": result.map50_95}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    dataset_root = Path(args.dataset)
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root

    entries = load_predictions_entries(args.predictions)
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]

    base_metrics = _safe_metrics(records, entries)

    grid = _parse_grid(args.temperatures)
    if not grid:
        raise SystemExit("temperature grid is empty")

    best = {"temperature": None, "metrics": None}
    grid_results = []
    for temp in grid:
        calibrated = calibrate_predictions_entries(
            entries,
            temperature=temp,
            min_score=args.min_score,
            max_score=args.max_score,
        )
        metrics = _safe_metrics(records, calibrated)
        grid_results.append({"temperature": temp, "metrics": metrics})
        if best["metrics"] is None or metrics["map50_95"] > best["metrics"]["map50_95"]:
            best = {"temperature": temp, "metrics": metrics}

    best_temp = float(best["temperature"])
    calibrated = calibrate_predictions_entries(
        entries,
        temperature=best_temp,
        min_score=args.min_score,
        max_score=args.max_score,
    )

    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(calibrated, indent=2, sort_keys=True))

    artifact = {
        "best_temperature": best_temp,
        "base_metrics": base_metrics,
        "best_metrics": best["metrics"],
        "grid": grid_results,
    }
    artifact_path = repo_root / args.output_artifact
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))

    report = build_report(
        metrics={
            "base": base_metrics,
            "best": best["metrics"],
            "best_temperature": best_temp,
            "grid": grid_results,
        },
        meta={
            "dataset": str(dataset_root),
            "split": manifest.get("split"),
            "predictions": args.predictions,
            "output": str(output_path),
            "artifact": str(artifact_path),
            "score_clamp": {
                "min": args.min_score,
                "max": args.max_score,
            },
        },
    )

    report_path = repo_root / args.output_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
