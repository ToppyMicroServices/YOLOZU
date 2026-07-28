#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.continual_metrics import summarize_continual_matrix  # noqa: E402
from yolozu.api import evaluate_coco  # noqa: E402
from yolozu.dataset import build_manifest  # noqa: E402
from yolozu.pose_eval import evaluate_pose  # noqa: E402
from yolozu.predictions import load_predictions_entries  # noqa: E402
from yolozu.simple_map import evaluate_map  # noqa: E402


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _subprocess_or_die(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.stderr and proc.stderr.strip():
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_html(*, html_path: Path, report: dict[str, Any]) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)

    meta = report.get("meta") if isinstance(report.get("meta"), dict) else {}
    tasks = report.get("tasks") if isinstance(report.get("tasks"), list) else []
    rows = report.get("matrix") if isinstance(report.get("matrix"), list) else []
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    cell_details = report.get("cell_details") if isinstance(report.get("cell_details"), list) else None

    def esc(s: Any) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def _fmt(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int) and not isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            return f"{float(v):.6g}"
        return str(v)

    def _cell_title(t: int, i: int) -> str | None:
        if cell_details is None:
            return None
        if not (0 <= int(t) < len(cell_details)):
            return None
        row = cell_details[int(t)]
        if not isinstance(row, list) or not (0 <= int(i) < len(row)):
            return None
        cell = row[int(i)]
        if not isinstance(cell, dict):
            return None

        metric = cell.get("metric")
        metrics = cell.get("metrics") if isinstance(cell.get("metrics"), dict) else {}
        counts = cell.get("counts") if isinstance(cell.get("counts"), dict) else {}

        parts: list[str] = []
        if metric:
            parts.append(f"metric={metric}")

        if metric == "simple_map":
            parts.append(f"map50={_fmt(metrics.get('map50'))}")
            parts.append(f"map50_95={_fmt(metrics.get('map50_95'))}")
        elif metric == "pose":
            for k in (
                "pose_success",
                "rot_success",
                "trans_success",
                "match_rate",
                "iou_mean",
                "rot_deg_mean",
                "trans_l2_mean",
                "depth_abs_mean",
                "depth_abs_median",
                "add_mean",
                "add_median",
                "adds_mean",
                "adds_median",
            ):
                if k in metrics:
                    parts.append(f"{k}={_fmt(metrics.get(k))}")
            for k in (
                "gt_instances",
                "pred_instances",
                "matches",
                "pose_measured",
                "rot_measured",
                "trans_measured",
                "depth_measured",
                "add_measured",
                "adds_measured",
            ):
                if k in counts:
                    parts.append(f"{k}={_fmt(counts.get(k))}")
        else:
            # Generic fallback: show the keys relevant to downstream selection.
            for k, v in sorted(metrics.items()):
                parts.append(f"{k}={_fmt(v)}")
            for k, v in sorted(counts.items()):
                parts.append(f"{k}={_fmt(v)}")

        return " | ".join(parts) if parts else None

    lines: list[str] = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '  <meta charset="utf-8" />',
        "  <title>Continual Eval</title>",
        "  <style>",
        "    body{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding:16px;}",
        "    .meta{color:#666; font-size:12px; overflow-wrap:anywhere;}",
        "    table{border-collapse:collapse; width:100%; margin-top:12px;}",
        "    th,td{border:1px solid #ddd; padding:6px 8px; font-size:12px; text-align:right;}",
        "    th:first-child, td:first-child{text-align:left;}",
        "  </style>",
        "</head>",
        "<body>",
        "<h1>Continual learning evaluation</h1>",
        f"<div class='meta'>run_json: {esc(meta.get('run_json'))}</div>",
        f"<div class='meta'>metric: {esc(meta.get('metric'))} key={esc(meta.get('metric_key'))}</div>",
        f"<div class='meta'>timestamp: {esc(meta.get('timestamp_utc'))}</div>",
        "<h2>Summary</h2>",
        "<table>",
        "<tr><th>metric</th><th>value</th></tr>",
        f"<tr><td>avg_acc</td><td>{esc(summary.get('avg_acc'))}</td></tr>",
        f"<tr><td>forgetting</td><td>{esc(summary.get('forgetting'))}</td></tr>",
        f"<tr><td>bwt</td><td>{esc(summary.get('bwt'))}</td></tr>",
        f"<tr><td>fwt</td><td>{esc(summary.get('fwt'))}</td></tr>",
        "</table>",
        "<h2>Matrix</h2>",
        "<table>",
        "<tr><th>checkpoint</th>",
    ]
    for t in tasks:
        lines.append(f"<th>{esc(t.get('name'))}</th>")
    lines.append("</tr>")

    for t, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        label = row.get("label")
        vals = row.get("values") if isinstance(row.get("values"), list) else []
        lines.append(f"<tr><td>{esc(label)}</td>")
        for i, v in enumerate(vals):
            title = _cell_title(int(t), int(i))
            if title:
                lines.append(f"<td title=\"{esc(title)}\">{esc(v)}</td>")
            else:
                lines.append(f"<td>{esc(v)}</td>")
        lines.append("</tr>")

    lines.extend(["</table>", "</body>", "</html>"])
    html_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a continual learning run with COCO, proxy mAP, or pose metrics.")
    p.add_argument("--run-json", required=True, help="Path to runs/.../continual_run.json produced by train_continual.py")
    p.add_argument("--device", default="cpu", help="Torch device for export (default: cpu).")
    p.add_argument("--image-size", type=int, default=320, help="Adapter image size (square, default: 320).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for export/eval.")
    p.add_argument(
        "--metric",
        choices=("coco", "simple_map", "pose"),
        default="simple_map",
        help="Metric backend to use (default: simple_map). Use coco for real pycocotools COCOeval.",
    )
    p.add_argument(
        "--metric-key",
        default=None,
        help="Which metric value to use for CL summaries (default depends on --metric).",
    )
    p.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for pose matching (default: 0.5).")
    p.add_argument("--min-score", type=float, default=0.0, help="Min detection score for pose eval (default: 0.0).")
    p.add_argument("--success-rot-deg", type=float, default=15.0, help="Pose success: rot error <= deg (default: 15).")
    p.add_argument("--success-trans", type=float, default=0.1, help="Pose success: translation L2 <= meters (default: 0.1).")
    p.add_argument("--keep-per-image", type=int, default=0, help="Keep N per-image summaries in JSON (default: 0).")
    p.add_argument(
        "--baseline-checkpoint",
        default=None,
        help="Initial checkpoint used to define FWT (default: run_json.initial_checkpoint).",
    )
    p.add_argument("--output", default=None, help="Output JSON path (default: <run_dir>/continual_eval.json).")
    p.add_argument("--html", default=None, help="Optional HTML report path (default: <run_dir>/continual_eval.html).")
    p.add_argument("--force", action="store_true", help="Overwrite existing prediction/eval outputs.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    run_json_path = _resolve(args.run_json)
    if run_json_path is None or not run_json_path.exists():
        raise SystemExit(f"run json not found: {args.run_json}")
    run = json.loads(run_json_path.read_text(encoding="utf-8"))

    tasks = run.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit("run_json.tasks must be a non-empty list")
    model_config = run.get("model_config")
    if not isinstance(model_config, str) or not model_config:
        raise SystemExit("run_json.model_config missing")
    lora_cfg = run.get("lora") if isinstance(run.get("lora"), dict) else {}
    lora_enabled = bool(lora_cfg.get("enabled", False))

    run_dir = run_json_path.parent
    out_json = _resolve(args.output) if args.output else (run_dir / "continual_eval.json")
    out_html = _resolve(args.html) if args.html else (run_dir / "continual_eval.html")
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Matrix keyed by (t,i).
    matrix_values: list[list[float | None]] = []
    matrix_rows: list[dict[str, Any]] = []
    cell_details: list[list[dict[str, Any] | None]] = []

    metric_key = args.metric_key
    if metric_key is None:
        metric_key = "map50_95" if args.metric in ("coco", "simple_map") else "pose_success"
    if args.metric == "coco":
        if metric_key not in ("map50_95", "map50", "map75", "ar100"):
            raise SystemExit("--metric-key must be one of: map50_95, map50, map75, ar100 (for --metric coco)")
    elif args.metric == "simple_map":
        if metric_key not in ("map50", "map50_95"):
            raise SystemExit("--metric-key must be one of: map50, map50_95 (for --metric simple_map)")
    elif args.metric == "pose":
        if metric_key not in (
            "pose_success",
            "rot_success",
            "trans_success",
            "match_rate",
            "iou_mean",
            "depth_abs_mean",
            "depth_abs_median",
            "add_mean",
            "add_median",
            "adds_mean",
            "adds_median",
        ):
            raise SystemExit(
                "--metric-key must be one of: pose_success, rot_success, trans_success, match_rate, iou_mean, depth_abs_mean, depth_abs_median, add_mean, add_median, adds_mean, adds_median (for --metric pose)"
            )

    def checkpoint_path(raw: Any, *, field: str) -> Path:
        if isinstance(raw, dict):
            raw = raw.get("path")
        if not isinstance(raw, str) or not raw:
            raise SystemExit(f"{field} missing")
        path = Path(raw)
        if not path.is_absolute():
            path = (run_dir / path).resolve()
        if not path.exists():
            raise SystemExit(f"checkpoint not found: {path}")
        return path

    def evaluate_checkpoint(
        *,
        ckpt_path: Path,
        checkpoint_label: str,
        task_idx: int,
        task: dict[str, Any],
    ) -> tuple[float | None, dict[str, Any]]:
        dataset_root = task.get("dataset_root")
        val_split = task.get("val_split")
        if not isinstance(dataset_root, str) or not dataset_root:
            raise SystemExit(f"tasks[{task_idx}].dataset_root missing")
        if not isinstance(val_split, str) or not val_split:
            raise SystemExit(f"tasks[{task_idx}].val_split missing")

        dataset_path = Path(dataset_root)
        if not dataset_path.is_absolute():
            dataset_path = (repo_root / dataset_path).resolve()

        pred_path = eval_dir / f"pred_{checkpoint_label}_on_{task_idx:02d}.json"
        export_started = time.perf_counter()
        if args.force or not pred_path.exists():
            cmd = [
                sys.executable,
                "tools/export_predictions.py",
                "--adapter",
                "rtdetr_pose",
                "--dataset",
                str(dataset_path),
                "--split",
                str(val_split),
                "--config",
                str(model_config),
                "--checkpoint",
                str(ckpt_path),
                "--device",
                str(args.device),
                "--image-size",
                str(int(args.image_size)),
                "--wrap",
                "--output",
                str(pred_path),
            ]
            if args.max_images is not None:
                cmd.extend(["--max-images", str(int(args.max_images))])
            if lora_enabled:
                cmd.extend(["--lora-r", str(int(lora_cfg.get("r") or 0))])
                if lora_cfg.get("alpha") is not None:
                    cmd.extend(["--lora-alpha", str(float(lora_cfg.get("alpha")))])
                cmd.extend(["--lora-dropout", str(float(lora_cfg.get("dropout") or 0.0))])
                cmd.extend(["--lora-target", str(lora_cfg.get("target") or "head")])
                freeze_base = bool(lora_cfg.get("freeze_base", True))
                cmd.append("--lora-freeze-base" if freeze_base else "--no-lora-freeze-base")
                cmd.extend(["--lora-train-bias", str(lora_cfg.get("train_bias") or "none")])
            _subprocess_or_die(cmd)
        export_seconds = time.perf_counter() - export_started

        eval_started = time.perf_counter()
        if args.metric == "coco":
            result = evaluate_coco(
                dataset_path,
                pred_path,
                split=str(val_split),
                max_images=args.max_images,
                dry_run=False,
            )
            metrics = result.metrics.to_dict()
            value_raw = metrics.get(str(metric_key))
            value = float(value_raw) if value_raw is not None else None
            details: dict[str, Any] = {
                "metric": "coco",
                "metrics": metrics,
                "counts": result.counts.to_dict(),
                "warnings": list(result.warnings),
            }
        else:
            manifest = build_manifest(dataset_path, split=str(val_split))
            records = manifest["images"]
            if args.max_images is not None:
                records = records[: int(args.max_images)]
            preds = load_predictions_entries(pred_path)
            if args.metric == "simple_map":
                result = evaluate_map(records, preds, iou_thresholds=[0.5 + 0.05 * j for j in range(10)])
                metrics = {"map50": float(result.map50), "map50_95": float(result.map50_95)}
                value = float(metrics[str(metric_key)])
                details = {"metric": "simple_map", "metrics": metrics, "counts": {"images": int(len(records))}}
            else:
                result = evaluate_pose(
                    records,
                    preds,
                    iou_threshold=float(args.iou_threshold),
                    min_score=float(args.min_score),
                    success_rot_deg=float(args.success_rot_deg),
                    success_trans=float(args.success_trans),
                    keep_per_image=int(args.keep_per_image),
                )
                value_raw = result.metrics.get(str(metric_key))
                value = float(value_raw) if value_raw is not None else None
                details = {
                    "metric": "pose",
                    "metrics": dict(result.metrics),
                    "counts": dict(result.counts),
                    "warnings": list(result.warnings),
                }
                if result.per_image:
                    details["per_image"] = list(result.per_image)
        details["provenance"] = {
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": _sha256_file(ckpt_path),
            "predictions": str(pred_path),
            "predictions_sha256": _sha256_file(pred_path),
            "export_seconds": float(export_seconds),
            "evaluation_seconds": float(time.perf_counter() - eval_started),
        }
        return value, details

    baseline_raw: Any = args.baseline_checkpoint
    if baseline_raw is None:
        baseline_raw = run.get("initial_checkpoint")
    baseline_path = checkpoint_path(baseline_raw, field="baseline checkpoint") if baseline_raw else None
    baseline_values: list[float | None] | None = None
    baseline_details: list[dict[str, Any]] | None = None
    if baseline_path is not None:
        baseline_values = []
        baseline_details = []
        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                baseline_values.append(None)
                baseline_details.append({})
                continue
            value, details = evaluate_checkpoint(
                ckpt_path=baseline_path,
                checkpoint_label="baseline",
                task_idx=i,
                task=task,
            )
            baseline_values.append(value)
            baseline_details.append(details)

    for t, stage in enumerate(tasks):
        if not isinstance(stage, dict):
            continue
        label = stage.get("name") or f"task{t:02d}"
        ckpt_path = checkpoint_path(stage.get("checkpoint"), field=f"tasks[{t}].checkpoint")

        row: list[float | None] = []
        row_render: list[float | None] = []
        row_details: list[dict[str, Any] | None] = []

        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                row.append(None)
                row_render.append(None)
                row_details.append(None)
                continue
            value, details = evaluate_checkpoint(
                ckpt_path=ckpt_path,
                checkpoint_label=f"t{t:02d}",
                task_idx=i,
                task=task,
            )

            row.append(value)
            row_render.append(value)
            row_details.append(details)

        matrix_values.append(row)
        matrix_rows.append({"label": str(label), "values": row_render})
        cell_details.append(row_details)

    summary = summarize_continual_matrix(matrix_values, baseline=baseline_values)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "timestamp_utc": _now_utc(),
        "meta": {
            "run_json": str(run_json_path),
            "metric": str(args.metric),
            "metric_key": str(metric_key),
            "baseline_checkpoint": str(baseline_path) if baseline_path is not None else None,
            "baseline_checkpoint_sha256": _sha256_file(baseline_path) if baseline_path is not None else None,
            "iou_threshold": (float(args.iou_threshold) if args.metric == "pose" else None),
            "min_score": (float(args.min_score) if args.metric == "pose" else None),
            "success_rot_deg": (float(args.success_rot_deg) if args.metric == "pose" else None),
            "success_trans": (float(args.success_trans) if args.metric == "pose" else None),
            "timestamp_utc": _now_utc(),
        },
        "tasks": [{"name": str(t.get("name") or f"task{i:02d}")} for i, t in enumerate(tasks) if isinstance(t, dict)],
        "matrix": matrix_rows,
        "matrix_values": matrix_values,
        "cell_details": cell_details,
        "baseline_values": baseline_values,
        "baseline_details": baseline_details,
        "summary": {
            "avg_acc": summary.avg_acc,
            "forgetting": summary.forgetting,
            "bwt": summary.bwt,
            "fwt": summary.fwt,
            "details": summary.details,
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_html(html_path=out_html, report=payload)
    print(out_json)


if __name__ == "__main__":
    main()
