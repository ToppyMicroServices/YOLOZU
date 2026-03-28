#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest
from yolozu.eval.simple_map import evaluate_map
from yolozu.predictions.predictions_parity import compare_predictions

BOILERPLATE_DIR = repo_root / "configs" / "examples" / "ttt_compare"
KNOWN_BOILERPLATES = ("tent", "mim", "mim_probe", "cotta", "eata", "sar")


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def _short(path: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _coerce_method(token: str) -> str:
    value = str(token).strip().lower()
    if not value:
        raise ValueError("--boilerplate must not be empty")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _resolve_boilerplate(token: str) -> tuple[str, Path, dict[str, Any]]:
    value = _coerce_method(token)
    if value in KNOWN_BOILERPLATES:
        path = BOILERPLATE_DIR / f"{value}.json"
    else:
        path = _resolve(value)
    if not path.is_file():
        raise FileNotFoundError(f"boilerplate not found: {path}")
    payload = _load_json(path)
    method = str(payload.get("method") or "").strip().lower()
    if method not in KNOWN_BOILERPLATES:
        raise ValueError(f"boilerplate {path} has unsupported method: {method!r}")
    return method, path, payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run baseline-vs-adapted TTT compare using a method boilerplate and emit "
            "predictions, optional eval reports, and a before-after summary."
        )
    )
    p.add_argument("-b", "--boilerplate", required=True, help="Boilerplate name (tent/mim/mim_probe/cotta/eata/sar) or JSON path.")
    p.add_argument("-d", "--dataset", required=True, help="YOLO-format dataset root.")
    p.add_argument("-s", "--split", default="val", help="Dataset split (default: val).")
    p.add_argument("-c", "--checkpoint", required=True, help="Checkpoint path for the adapted run.")
    p.add_argument("-r", "--run-dir", default=None, help="Run directory (default: reports/ttt_compare/<method>).")
    p.add_argument("-B", "--backend", default="torch", help="Export backend (default: torch).")
    p.add_argument("-D", "--device", default="cpu", help="Inference device (default: cpu).")
    p.add_argument("-m", "--max-images", type=int, default=None, help="Optional cap for number of images.")
    p.add_argument("-p", "--protocol", choices=("yolo26", "nms_applied", "e2e_nms_free"), default=None, help="Optional eval protocol for before/after reports.")
    p.add_argument("--image-size", type=int, nargs="+", default=None, help="Optional export image size (one or two ints).")
    p.add_argument("--skip-eval", action="store_true", help="Skip eval_suite stage and only compare wrapped predictions + TTT logs.")
    p.add_argument("--dry-run", action="store_true", help="Write the execution plan without running export/eval.")
    p.add_argument("--force", action="store_true", help="Overwrite run directory outputs when they exist.")
    return p.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_images is not None and int(args.max_images) < 0:
        raise ValueError("--max-images must be >= 0")
    if args.backend != "torch" and not args.dry_run:
        raise ValueError("real TTT compare currently requires --backend torch; use --dry-run to inspect the plan for other backends")


def _run(cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return {
        "command": list(cmd),
        "returncode": int(proc.returncode),
        "stdout": str(proc.stdout),
        "stderr": str(proc.stderr),
    }


def _require_ok(result: dict[str, Any], *, label: str) -> None:
    if int(result.get("returncode") or 0) != 0:
        cmd = " ".join(shlex.quote(str(part)) for part in result.get("command") or [])
        raise RuntimeError(
            f"{label} failed (rc={result.get('returncode')}):\n"
            f"$ {cmd}\n{result.get('stdout') or ''}{result.get('stderr') or ''}"
        )


def _raise_ttt_compare_failure(result: dict[str, Any], *, method: str) -> None:
    combined = f"{result.get('stdout') or ''}{result.get('stderr') or ''}"
    if method == "mim" and "mask spatial dims must match input" in combined:
        cmd = " ".join(shlex.quote(str(part)) for part in result.get("command") or [])
        raise RuntimeError(
            "mim export failed: the selected checkpoint/config does not expose a usable MIM reconstruction path for this compare run.\n"
            "Use a MIM-enabled checkpoint/config for a real compare, or run the same boilerplate with --dry-run to capture the planned before-after protocol.\n"
            f"$ {cmd}\n{combined}"
        )
    _require_ok(result, label=f"{method} export")


def _flatten_ttt_reports(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    mode = str(report.get("reset") or report.get("mode") or "stream")
    if mode != "sample":
        return [report]
    per_sample = report.get("per_sample")
    if not isinstance(per_sample, list):
        return [report]
    out: list[dict[str, Any]] = []
    for item in per_sample:
        if not isinstance(item, dict):
            continue
        sub = item.get("report")
        if isinstance(sub, dict):
            out.append(sub)
    return out or [report]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _extract_ttt_summary(payload: dict[str, Any]) -> dict[str, Any]:
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return {"enabled": False, "reason": "missing_meta"}
    ttt = meta.get("ttt")
    if not isinstance(ttt, dict):
        return {"enabled": False, "reason": "missing_meta_ttt"}
    report = ttt.get("report")
    if not isinstance(report, dict):
        return {
            "enabled": bool(ttt.get("enabled")),
            "method": ttt.get("method"),
            "preset": ttt.get("preset"),
            "reason": "missing_report",
        }
    parts = _flatten_ttt_reports(report)
    losses: list[float] = []
    final_losses: list[float] = []
    seconds: list[float] = []
    warnings: list[str] = []
    steps_run = 0
    guard_breaches = 0
    for part in parts:
        part_losses = part.get("losses")
        if isinstance(part_losses, list):
            local = [float(x) for x in part_losses]
            losses.extend(local)
            if local:
                final_losses.append(local[-1])
        warnings_list = part.get("warnings")
        if isinstance(warnings_list, list):
            warning_text = [str(x) for x in warnings_list]
            warnings.extend(warning_text)
            guard_breaches += sum(1 for x in warning_text if ("exceeded" in x or "non_finite" in x))
        steps_run += int(part.get("steps_run") or 0)
        seconds.append(_safe_float(part.get("seconds"), 0.0))
    return {
        "enabled": True,
        "method": str(ttt.get("method") or report.get("method") or "unknown"),
        "preset": ttt.get("preset"),
        "reset": ttt.get("reset") or report.get("reset"),
        "runs_count": int(len(parts)),
        "steps_run": int(steps_run),
        "mean_loss": float(sum(losses) / len(losses)) if losses else None,
        "mean_final_loss": float(sum(final_losses) / len(final_losses)) if final_losses else None,
        "mean_seconds": float(sum(seconds) / len(seconds)) if seconds else None,
        "guard_breaches": int(guard_breaches),
        "warnings": warnings,
    }


def _prediction_counts(payload: dict[str, Any]) -> dict[str, int]:
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        return {"images": 0, "detections": 0}
    detections = 0
    for entry in predictions:
        if not isinstance(entry, dict):
            continue
        raw = entry.get("detections")
        if isinstance(raw, list):
            detections += len(raw)
    return {"images": int(len(predictions)), "detections": int(detections)}


def _extract_eval_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _load_json(path)
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return None
    item = results[0]
    if not isinstance(item, dict):
        return None
    metrics = item.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return {
        "map50": metrics.get("map50"),
        "map50_95": metrics.get("map50_95"),
        "map75": metrics.get("map75"),
        "ar100": metrics.get("ar100"),
        "dry_run": bool(item.get("dry_run")),
        "warnings": item.get("warnings") if isinstance(item.get("warnings"), list) else [],
    }


def _load_prediction_entries(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError(f"predictions payload missing list at: {path}")
    return [item for item in predictions if isinstance(item, dict)]


def _should_use_simple_map_proxy(result: dict[str, Any] | None) -> bool:
    if not isinstance(result, dict):
        return False
    if int(result.get("returncode") or 0) == 0:
        return False
    combined = f"{result.get('stdout') or ''}{result.get('stderr') or ''}"
    return "pycocotools is required" in combined or "No module named 'pycocotools'" in combined


def _build_simple_map_proxy_eval(
    *,
    dataset: Path,
    split: str,
    predictions_path: Path,
    max_images: int | None,
) -> dict[str, Any]:
    manifest = build_manifest(str(dataset), split=str(split))
    records = list(manifest.get("images") or [])
    if max_images is not None:
        records = records[: max(0, int(max_images))]
    predictions_entries = _load_prediction_entries(predictions_path)
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    result = evaluate_map(records, predictions_entries, iou_thresholds=thresholds)
    return {
        "map50": float(result.map50),
        "map50_95": float(result.map50_95),
        "map75": None,
        "ar100": None,
        "dry_run": False,
        "warnings": [
            "eval_suite fell back to simple_map_proxy because pycocotools was unavailable in this runtime"
        ],
        "metric_backend": "simple_map_proxy",
    }


def _summarize_parity(report: dict[str, Any]) -> dict[str, Any]:
    results = report.get("results")
    if not isinstance(results, list):
        return {"ok": bool(report.get("ok")), "images": int(report.get("images") or 0)}
    changed_images = 0
    missing_match = 0
    value_mismatch = 0
    extra_candidates = 0
    for item in results:
        if not isinstance(item, dict):
            continue
        failures = item.get("failures") if isinstance(item.get("failures"), list) else []
        counts = item.get("counts") if isinstance(item.get("counts"), dict) else {}
        if failures or int(counts.get("extra_cand") or 0) > 0:
            changed_images += 1
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            kind = str(failure.get("type") or "")
            if kind == "missing_match":
                missing_match += 1
            elif kind == "value_mismatch":
                value_mismatch += 1
        extra_candidates = max(extra_candidates, int(counts.get("extra_cand") or 0))
    return {
        "ok": bool(report.get("ok")),
        "images": int(report.get("images") or 0),
        "changed_images": int(changed_images),
        "missing_match_failures": int(missing_match),
        "value_mismatch_failures": int(value_mismatch),
        "max_extra_candidate_detections": int(extra_candidates),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    baseline = report.get("baseline") or {}
    adapted = report.get("adapted") or {}
    parity = report.get("prediction_delta") or {}
    baseline_eval = report.get("baseline_eval") or {}
    adapted_eval = report.get("adapted_eval") or {}
    artifacts = report.get("artifacts") or {}
    lines = [
        f"# TTT before-after compare: {report.get('method')}",
        "",
        f"- Generated UTC: {report.get('timestamp')}",
        f"- Boilerplate: `{report.get('boilerplate_name')}`",
        f"- Preset: `{report.get('preset')}`",
        f"- Reset policy: `{report.get('reset')}`",
        "",
        "## Before vs after",
        "",
        "| Variant | Images | Detections | TTT enabled | Mean final loss | Mean seconds | map50 | map50_95 |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
        (
            f"| baseline | {int((baseline.get('prediction_counts') or {}).get('images') or 0)} | "
            f"{int((baseline.get('prediction_counts') or {}).get('detections') or 0)} | "
            f"{bool((baseline.get('ttt_summary') or {}).get('enabled'))} | "
            f"{_safe_float((baseline.get('ttt_summary') or {}).get('mean_final_loss'), float('nan')):.6f} | "
            f"{_safe_float((baseline.get('ttt_summary') or {}).get('mean_seconds'), float('nan')):.6f} | "
            f"{_safe_float(baseline_eval.get('map50'), float('nan')):.6f} | "
            f"{_safe_float(baseline_eval.get('map50_95'), float('nan')):.6f} |"
        ),
        (
            f"| adapted ({report.get('method')}) | {int((adapted.get('prediction_counts') or {}).get('images') or 0)} | "
            f"{int((adapted.get('prediction_counts') or {}).get('detections') or 0)} | "
            f"{bool((adapted.get('ttt_summary') or {}).get('enabled'))} | "
            f"{_safe_float((adapted.get('ttt_summary') or {}).get('mean_final_loss'), float('nan')):.6f} | "
            f"{_safe_float((adapted.get('ttt_summary') or {}).get('mean_seconds'), float('nan')):.6f} | "
            f"{_safe_float(adapted_eval.get('map50'), float('nan')):.6f} | "
            f"{_safe_float(adapted_eval.get('map50_95'), float('nan')):.6f} |"
        ),
        "",
        "## Prediction delta summary",
        "",
        f"- changed_images: `{parity.get('changed_images')}` / `{parity.get('images')}`",
        f"- missing_match_failures: `{parity.get('missing_match_failures')}`",
        f"- value_mismatch_failures: `{parity.get('value_mismatch_failures')}`",
        f"- max_extra_candidate_detections: `{parity.get('max_extra_candidate_detections')}`",
        "",
        "## Artifacts",
        "",
        f"- baseline_predictions: `{artifacts.get('baseline_predictions')}`",
        f"- adapted_predictions: `{artifacts.get('adapted_predictions')}`",
        f"- compare_json: `{artifacts.get('compare_json')}`",
        f"- compare_md: `{artifacts.get('compare_md')}`",
    ]
    if artifacts.get("adapted_ttt_log"):
        lines.append(f"- adapted_ttt_log: `{artifacts.get('adapted_ttt_log')}`")
    if artifacts.get("baseline_eval"):
        lines.append(f"- baseline_eval: `{artifacts.get('baseline_eval')}`")
    if artifacts.get("adapted_eval"):
        lines.append(f"- adapted_eval: `{artifacts.get('adapted_eval')}`")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)

    requested_boilerplate = _coerce_method(str(args.boilerplate))
    method, boilerplate_path, boilerplate = _resolve_boilerplate(str(args.boilerplate))
    if requested_boilerplate not in KNOWN_BOILERPLATES:
        requested_boilerplate = method
    run_dir = _resolve(args.run_dir or (repo_root / "reports" / "ttt_compare" / method))
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        raise SystemExit(f"run dir already exists and is not empty: {run_dir} (use --force)")
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = _resolve(args.dataset)
    checkpoint = _resolve(args.checkpoint)
    baseline_predictions = run_dir / "baseline_predictions.json"
    adapted_predictions = run_dir / f"{method}_predictions.json"
    adapted_ttt_log = run_dir / f"{method}_ttt_log.json"
    baseline_eval = run_dir / "baseline_eval.json"
    adapted_eval = run_dir / f"{method}_eval.json"
    compare_json = run_dir / f"{method}_before_after_compare.json"
    compare_md = run_dir / f"{method}_before_after_compare.md"
    plan_json = run_dir / "plan.json"

    preset_raw = boilerplate.get("preset")
    preset = str(preset_raw).strip() if preset_raw is not None else ""
    reset = str(boilerplate.get("reset") or "sample")
    extra_export_args = boilerplate.get("extra_export_args")
    if not isinstance(extra_export_args, list):
        extra_export_args = []
    common_export_args = boilerplate.get("common_export_args")
    if not isinstance(common_export_args, list):
        common_export_args = []

    common_cmd = [
        sys.executable,
        str(repo_root / "tools" / "yolozu.py"),
        "export",
        "--backend",
        str(args.backend),
        "--dataset",
        str(dataset),
        "--split",
        str(args.split),
        "--checkpoint",
        str(checkpoint),
        "--device",
        str(args.device),
        *[str(x) for x in common_export_args],
    ]
    if args.max_images is not None:
        common_cmd.extend(["--max-images", str(int(args.max_images))])
    if args.image_size:
        common_cmd.append("--image-size")
        common_cmd.extend(str(int(x)) for x in args.image_size)

    base_cmd = list(common_cmd) + [
        "--output",
        str(baseline_predictions),
        "--force",
    ]

    adapted_cmd = list(common_cmd) + [
        "--ttt",
        "--ttt-method",
        method,
    ]
    if preset:
        adapted_cmd.extend(["--ttt-preset", preset])
    adapted_cmd.extend(
        [
            "--ttt-reset",
            reset,
            "--ttt-log-out",
            str(adapted_ttt_log),
            *[str(x) for x in extra_export_args],
            "--output",
            str(adapted_predictions),
            "--force",
        ]
    )

    plan = {
        "schema_version": 1,
        "kind": "ttt_compare_plan",
        "timestamp": _now_utc(),
        "boilerplate_name": requested_boilerplate,
        "boilerplate_path": str(boilerplate_path),
        "method": method,
        "preset": preset,
        "reset": reset,
        "dataset": str(dataset),
        "split": str(args.split),
        "checkpoint": str(checkpoint),
        "backend": str(args.backend),
        "device": str(args.device),
        "max_images": args.max_images,
        "commands": {
            "baseline_export": base_cmd,
            "adapted_export": adapted_cmd,
        },
        "artifacts": {
            "baseline_predictions": str(baseline_predictions),
            "adapted_predictions": str(adapted_predictions),
            "adapted_ttt_log": str(adapted_ttt_log),
            "baseline_eval": str(baseline_eval),
            "adapted_eval": str(adapted_eval),
            "compare_json": str(compare_json),
            "compare_md": str(compare_md),
        },
    }
    if args.protocol:
        plan["protocol"] = str(args.protocol)
    if not args.skip_eval:
        baseline_eval_cmd = [
            sys.executable,
            str(repo_root / "tools" / "eval_suite.py"),
            "--dataset",
            str(dataset),
            "--split",
            str(args.split),
            "--predictions-glob",
            str(baseline_predictions),
            "--output",
            str(baseline_eval),
        ]
        adapted_eval_cmd = [
            sys.executable,
            str(repo_root / "tools" / "eval_suite.py"),
            "--dataset",
            str(dataset),
            "--split",
            str(args.split),
            "--predictions-glob",
            str(adapted_predictions),
            "--output",
            str(adapted_eval),
        ]
        if args.max_images is not None:
            baseline_eval_cmd.extend(["--max-images", str(int(args.max_images))])
            adapted_eval_cmd.extend(["--max-images", str(int(args.max_images))])
        if args.protocol:
            baseline_eval_cmd.extend(["--protocol", str(args.protocol)])
            adapted_eval_cmd.extend(["--protocol", str(args.protocol)])
        plan["commands"]["baseline_eval"] = baseline_eval_cmd
        plan["commands"]["adapted_eval"] = adapted_eval_cmd

    plan_json.write_text(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.dry_run:
        print(plan_json)
        return 0

    baseline_result = _run(base_cmd, cwd=repo_root)
    _require_ok(baseline_result, label="baseline export")
    adapted_result = _run(adapted_cmd, cwd=repo_root)
    _raise_ttt_compare_failure(adapted_result, method=method)

    eval_results: dict[str, Any] = {}
    if not args.skip_eval:
        for label in ("baseline_eval", "adapted_eval"):
            result = _run(plan["commands"][label], cwd=repo_root)
            eval_results[label] = result
    baseline_eval_metrics = _extract_eval_metrics(baseline_eval) if baseline_eval.is_file() else None
    adapted_eval_metrics = _extract_eval_metrics(adapted_eval) if adapted_eval.is_file() else None
    max_images = int(args.max_images) if args.max_images is not None else None
    if baseline_eval_metrics is None and _should_use_simple_map_proxy(eval_results.get("baseline_eval")):
        baseline_eval_metrics = _build_simple_map_proxy_eval(
            dataset=dataset,
            split=str(args.split),
            predictions_path=baseline_predictions,
            max_images=max_images,
        )
    if adapted_eval_metrics is None and _should_use_simple_map_proxy(eval_results.get("adapted_eval")):
        adapted_eval_metrics = _build_simple_map_proxy_eval(
            dataset=dataset,
            split=str(args.split),
            predictions_path=adapted_predictions,
            max_images=max_images,
        )

    baseline_payload = _load_json(baseline_predictions)
    adapted_payload = _load_json(adapted_predictions)
    parity_report = compare_predictions(
        reference=baseline_predictions,
        candidate=adapted_predictions,
        max_images=args.max_images,
        iou_thresh=float((boilerplate.get("compare") or {}).get("iou_thresh", 0.99)),
        score_atol=float((boilerplate.get("compare") or {}).get("score_atol", 1e-4)),
        bbox_atol=float((boilerplate.get("compare") or {}).get("bbox_atol", 1e-4)),
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "ttt_before_after_compare",
        "timestamp": _now_utc(),
        "boilerplate_name": requested_boilerplate,
        "boilerplate_path": str(boilerplate_path),
        "method": method,
        "preset": preset,
        "reset": reset,
        "artifacts": {
            "plan_json": _short(plan_json),
            "baseline_predictions": _short(baseline_predictions),
            "adapted_predictions": _short(adapted_predictions),
            "adapted_ttt_log": _short(adapted_ttt_log),
            "compare_json": _short(compare_json),
            "compare_md": _short(compare_md),
            "baseline_eval": _short(baseline_eval) if baseline_eval.is_file() else None,
            "adapted_eval": _short(adapted_eval) if adapted_eval.is_file() else None,
        },
        "inputs": {
            "dataset": str(dataset),
            "split": str(args.split),
            "checkpoint": str(checkpoint),
            "backend": str(args.backend),
            "device": str(args.device),
            "boilerplate_sha256": _sha256(boilerplate_path),
        },
        "baseline": {
            "prediction_counts": _prediction_counts(baseline_payload),
            "ttt_summary": _extract_ttt_summary(baseline_payload),
        },
        "adapted": {
            "prediction_counts": _prediction_counts(adapted_payload),
            "ttt_summary": _extract_ttt_summary(adapted_payload),
        },
        "prediction_delta": _summarize_parity(parity_report),
        "prediction_delta_detail": {
            "ok": bool(parity_report.get("ok")),
            "images": int(parity_report.get("images") or 0),
            "reference": str(parity_report.get("reference") or ""),
            "candidate": str(parity_report.get("candidate") or ""),
        },
        "commands": {
            "baseline_export": " ".join(shlex.quote(x) for x in base_cmd),
            "adapted_export": " ".join(shlex.quote(x) for x in adapted_cmd),
        },
    }
    if args.protocol:
        report["protocol"] = str(args.protocol)
    report["baseline_eval"] = baseline_eval_metrics
    report["adapted_eval"] = adapted_eval_metrics
    if eval_results:
        report["eval_commands"] = {
            key: " ".join(shlex.quote(x) for x in value.get("command") or []) for key, value in eval_results.items()
        }
        report["eval_status"] = {
            key: {"returncode": int(value.get("returncode") or 0), "stderr": str(value.get("stderr") or "")} for key, value in eval_results.items()
        }

    compare_json.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    compare_md.write_text(_render_markdown(report), encoding="utf-8")
    print(compare_json)
    print(compare_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
