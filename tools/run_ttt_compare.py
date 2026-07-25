#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest
from yolozu.eval.simple_map import evaluate_map
from yolozu.predictions.predictions_parity import compare_predictions
from yolozu.tta.method_profiles import get_ttt_method_profile

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


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _write_report_pair(
    json_path: Path,
    markdown_path: Path,
    *,
    report: dict[str, Any],
    markdown: str,
) -> None:
    json_pending = json_path.with_name(f".{json_path.name}.pending")
    markdown_pending = markdown_path.with_name(f".{markdown_path.name}.pending")
    _remove_stale_artifacts(
        [json_pending, markdown_pending, json_path, markdown_path]
    )
    try:
        _atomic_write_json(json_pending, report)
        _atomic_write_text(markdown_pending, markdown)
        os.replace(json_pending, json_path)
        os.replace(markdown_pending, markdown_path)
    except BaseException:
        _remove_stale_artifacts(
            [json_pending, markdown_pending, json_path, markdown_path]
        )
        raise


def _resolve_boilerplate(token: str) -> tuple[str, Path, dict[str, Any]]:
    raw = str(token).strip()
    value = _coerce_method(raw)
    if value in KNOWN_BOILERPLATES:
        path = BOILERPLATE_DIR / f"{value}.json"
    else:
        path = _resolve(raw)
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
    p.add_argument(
        "-b",
        "--boilerplate",
        "--method",
        required=True,
        help="Boilerplate name (tent/mim/mim_probe/cotta/eata/sar) or JSON path.",
    )
    p.add_argument(
        "-d", "--dataset", "--data", required=True, help="YOLO-format dataset root."
    )
    p.add_argument("-s", "--split", default="val", help="Dataset split (default: val).")
    p.add_argument(
        "-c",
        "--checkpoint",
        "--weights",
        required=True,
        help="Checkpoint path for the adapted run.",
    )
    p.add_argument(
        "-r",
        "--run-dir",
        "--out",
        default=None,
        help="Run directory (default: reports/ttt_compare/<method>).",
    )
    p.add_argument(
        "-B", "--backend", default="torch", help="Export backend (default: torch)."
    )
    p.add_argument(
        "-D", "--device", default="cpu", help="Inference device (default: cpu)."
    )
    p.add_argument(
        "-m",
        "-n",
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for number of images.",
    )
    p.add_argument(
        "-p",
        "--protocol",
        choices=("yolo26", "nms_applied", "e2e_nms_free"),
        default=None,
        help="Optional eval protocol for before/after reports.",
    )
    p.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=None,
        help="Optional export image size (one or two ints).",
    )
    p.add_argument(
        "--skip-eval",
        "--no-eval",
        action="store_true",
        help="Skip eval_suite stage and only compare wrapped predictions + TTT logs.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the execution plan without running export/eval.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite run directory outputs when they exist.",
    )
    return p.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_images is not None and int(args.max_images) < 0:
        raise ValueError("--max-images must be >= 0")
    if args.backend != "torch" and not args.dry_run:
        raise ValueError(
            "real TTT compare currently requires --backend torch; use --dry-run to inspect the plan for other backends"
        )


def _referenced_config_paths(common_export_args: list[Any]) -> list[Path]:
    paths: list[Path] = []
    tokens = [str(value) for value in common_export_args]
    for index, token in enumerate(tokens):
        if token != "--config":
            continue
        if index + 1 >= len(tokens):
            raise ValueError("boilerplate common_export_args ends with --config")
        paths.append(_resolve(tokens[index + 1]))
    if not paths:
        paths.append(_resolve("rtdetr_pose/configs/base.json"))
    if len(paths) != 1:
        raise ValueError("TTT compare requires exactly one --config")
    return paths


def _model_checkpoint_preflight(
    *,
    config_path: Path,
    checkpoint: Path,
    method: str,
) -> dict[str, Any]:
    from yolozu.adapter import RTDETRPoseAdapter
    from yolozu.tta.ttt_mim import supports_structured_mim

    adapter = RTDETRPoseAdapter(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint),
        device="cpu",
        image_size=(32, 32),
    )
    model = adapter.get_model()
    checkpoint_report = adapter.get_checkpoint_report()
    if not isinstance(checkpoint_report, dict):
        raise RuntimeError(
            "checkpoint preflight completed without a compatibility report"
        )
    if not bool((checkpoint_report.get("load") or {}).get("loaded")):
        raise RuntimeError(
            "checkpoint preflight report does not confirm a loaded model"
        )
    if str(checkpoint_report.get("status") or "") != "compatible":
        raise RuntimeError(
            "TTT compare requires a fully compatible checkpoint; "
            f"preflight status={checkpoint_report.get('status')!r}"
        )

    structured_mim = bool(supports_structured_mim(model))
    if method == "mim" and not structured_mim:
        raise RuntimeError(
            "MIM compare requires a selected model/config with the structured MIM hook"
        )
    return {
        "config": _short(config_path),
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "checkpoint_compatibility": checkpoint_report,
        "structured_mim_supported": structured_mim,
    }


def _validate_prerequisites(
    *,
    dataset: Path,
    split: str,
    checkpoint: Path,
    boilerplate_path: Path,
    method: str,
    common_export_args: list[Any],
    skip_eval: bool,
) -> dict[str, Any]:
    if not dataset.is_dir():
        raise FileNotFoundError(f"dataset root not found or not a directory: {dataset}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found or not a file: {checkpoint}")
    if checkpoint.stat().st_size <= 0:
        raise ValueError(f"checkpoint is empty: {checkpoint}")
    if not boilerplate_path.is_file():
        raise FileNotFoundError(f"boilerplate not found: {boilerplate_path}")

    manifest = build_manifest(str(dataset), split=str(split))
    records = list(manifest.get("images") or [])
    if not records:
        raise ValueError(f"dataset split has no images: root={dataset} split={split!r}")

    config_paths = _referenced_config_paths(common_export_args)
    config_hashes: list[dict[str, str]] = []
    for config_path in config_paths:
        if not config_path.is_file():
            raise FileNotFoundError(f"boilerplate config not found: {config_path}")
        _load_json(config_path)
        config_hashes.append(
            {
                "path": _short(config_path),
                "sha256": _sha256(config_path),
            }
        )
    model_preflight = _model_checkpoint_preflight(
        config_path=config_paths[0],
        checkpoint=checkpoint,
        method=method,
    )

    required_tools = [repo_root / "tools" / "yolozu.py"]
    if not skip_eval:
        required_tools.append(repo_root / "tools" / "eval_suite.py")
    for tool_path in required_tools:
        if not tool_path.is_file():
            raise FileNotFoundError(f"required tool not found: {tool_path}")

    return {
        "dataset_images": int(len(records)),
        "checkpoint_sha256": _sha256(checkpoint),
        "boilerplate_sha256": _sha256(boilerplate_path),
        "configs": config_hashes,
        "model_preflight": model_preflight,
        "required_tools": [_short(path) for path in required_tools],
    }


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
    rollback_steps = 0
    stopped_early_count = 0
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
            guard_breaches += sum(
                1 for x in warning_text if ("exceeded" in x or "non_finite" in x)
            )
        step_metrics = part.get("step_metrics")
        if isinstance(step_metrics, list):
            rollback_steps += sum(
                1
                for step in step_metrics
                if isinstance(step, dict) and bool(step.get("rolled_back"))
            )
        if bool(part.get("stopped_early")):
            stopped_early_count += 1
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
        "mean_final_loss": float(sum(final_losses) / len(final_losses))
        if final_losses
        else None,
        "mean_seconds": float(sum(seconds) / len(seconds)) if seconds else None,
        "guard_breaches": int(guard_breaches),
        "rollback_steps": int(rollback_steps),
        "stopped_early_count": int(stopped_early_count),
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
        "warnings": item.get("warnings")
        if isinstance(item.get("warnings"), list)
        else [],
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
    return (
        "pycocotools is required" in combined
        or "No module named 'pycocotools'" in combined
    )


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
        "map50": None,
        "map50_95": None,
        "map75": None,
        "ar100": None,
        "proxy_ap50": float(result.map50),
        "proxy_ap50_95": float(result.map50_95),
        "dry_run": False,
        "warnings": [
            "pycocotools was unavailable; proxy_ap50 and proxy_ap50_95 are "
            "non-COCO diagnostics and must not be reported as COCO mAP"
        ],
        "metric_backend": "simple_map_proxy",
        "metric_semantics": "non_coco_proxy",
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
        failures = (
            item.get("failures") if isinstance(item.get("failures"), list) else []
        )
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
        f"- Method profile: `{(report.get('method_profile') or {}).get('profile_id')}`",
        f"- Reference-faithful: `{bool((report.get('method_profile') or {}).get('reference_faithful'))}`",
        f"- Efficacy: `{(report.get('method_profile') or {}).get('efficacy')}`",
        f"- Promotion eligible: `{bool(report.get('promotion_eligible'))}`",
        "",
        "## Before vs after",
        "",
        "| Variant | Images | Detections | TTT enabled | Mean final loss | Mean seconds | COCO mAP50 | COCO mAP50-95 | proxy AP50 | proxy AP50-95 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| baseline | {int((baseline.get('prediction_counts') or {}).get('images') or 0)} | "
            f"{int((baseline.get('prediction_counts') or {}).get('detections') or 0)} | "
            f"{bool((baseline.get('ttt_summary') or {}).get('enabled'))} | "
            f"{_safe_float((baseline.get('ttt_summary') or {}).get('mean_final_loss'), float('nan')):.6f} | "
            f"{_safe_float((baseline.get('ttt_summary') or {}).get('mean_seconds'), float('nan')):.6f} | "
            f"{_safe_float(baseline_eval.get('map50'), float('nan')):.6f} | "
            f"{_safe_float(baseline_eval.get('map50_95'), float('nan')):.6f} | "
            f"{_safe_float(baseline_eval.get('proxy_ap50'), float('nan')):.6f} | "
            f"{_safe_float(baseline_eval.get('proxy_ap50_95'), float('nan')):.6f} |"
        ),
        (
            f"| adapted ({report.get('method')}) | {int((adapted.get('prediction_counts') or {}).get('images') or 0)} | "
            f"{int((adapted.get('prediction_counts') or {}).get('detections') or 0)} | "
            f"{bool((adapted.get('ttt_summary') or {}).get('enabled'))} | "
            f"{_safe_float((adapted.get('ttt_summary') or {}).get('mean_final_loss'), float('nan')):.6f} | "
            f"{_safe_float((adapted.get('ttt_summary') or {}).get('mean_seconds'), float('nan')):.6f} | "
            f"{_safe_float(adapted_eval.get('map50'), float('nan')):.6f} | "
            f"{_safe_float(adapted_eval.get('map50_95'), float('nan')):.6f} | "
            f"{_safe_float(adapted_eval.get('proxy_ap50'), float('nan')):.6f} | "
            f"{_safe_float(adapted_eval.get('proxy_ap50_95'), float('nan')):.6f} |"
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


def _set_plan_status(
    plan: dict[str, Any],
    plan_path: Path,
    *,
    state: str,
    stage: str,
    error: BaseException | None = None,
) -> None:
    status: dict[str, Any] = {
        "state": str(state),
        "stage": str(stage),
        "updated_at": _now_utc(),
    }
    if state == "running":
        status["started_at"] = str(
            (plan.get("execution_status") or {}).get("started_at") or _now_utc()
        )
    elif state in {"failed", "completed", "not_executed"}:
        prior = plan.get("execution_status") or {}
        if prior.get("started_at"):
            status["started_at"] = str(prior["started_at"])
        status["finished_at"] = _now_utc()
    if error is not None:
        status["error_type"] = type(error).__name__
        status["error"] = str(error)
        error_report = getattr(error, "report", None)
        if isinstance(error_report, dict):
            status["error_report"] = error_report
    plan["execution_status"] = status
    _atomic_write_json(plan_path, plan)


def _remove_stale_artifacts(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except IsADirectoryError as exc:
            raise RuntimeError(
                f"expected TTT artifact path to be a file: {path}"
            ) from exc


def _eval_or_proxy(
    *,
    result: dict[str, Any],
    output_path: Path,
    dataset: Path,
    split: str,
    predictions_path: Path,
    max_images: int | None,
    label: str,
) -> dict[str, Any]:
    if int(result.get("returncode") or 0) == 0:
        metrics = _extract_eval_metrics(output_path)
        if metrics is None:
            raise RuntimeError(
                f"{label} completed without readable metrics: {output_path}"
            )
        return metrics
    if _should_use_simple_map_proxy(result):
        return _build_simple_map_proxy_eval(
            dataset=dataset,
            split=split,
            predictions_path=predictions_path,
            max_images=max_images,
        )
    _require_ok(result, label=label)
    raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)

    requested_boilerplate = _coerce_method(str(args.boilerplate))
    method, boilerplate_path, boilerplate = _resolve_boilerplate(str(args.boilerplate))
    if requested_boilerplate not in KNOWN_BOILERPLATES:
        requested_boilerplate = method
    run_dir = _resolve(args.run_dir or (repo_root / "reports" / "ttt_compare" / method))
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        raise SystemExit(
            f"run dir already exists and is not empty: {run_dir} (use --force)"
        )
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

    known_artifacts = [
        baseline_predictions,
        adapted_predictions,
        adapted_ttt_log,
        baseline_eval,
        adapted_eval,
        compare_json,
        compare_md,
        plan_json,
    ]
    if args.force:
        _remove_stale_artifacts(known_artifacts)

    _set_plan_status(plan, plan_json, state="running", stage="prerequisite_validation")
    stage = "prerequisite_validation"
    try:
        prerequisites = _validate_prerequisites(
            dataset=dataset,
            split=str(args.split),
            checkpoint=checkpoint,
            boilerplate_path=boilerplate_path,
            method=method,
            common_export_args=common_export_args,
            skip_eval=bool(args.skip_eval),
        )
        plan["prerequisites"] = prerequisites
        if args.dry_run:
            _set_plan_status(
                plan,
                plan_json,
                state="not_executed",
                stage="dry_run_complete",
            )
            print(plan_json)
            return 0

        stage = "baseline_export"
        _set_plan_status(plan, plan_json, state="running", stage=stage)
        baseline_result = _run(base_cmd, cwd=repo_root)
        _require_ok(baseline_result, label="baseline export")

        stage = "adapted_export"
        _set_plan_status(plan, plan_json, state="running", stage=stage)
        adapted_result = _run(adapted_cmd, cwd=repo_root)
        _raise_ttt_compare_failure(adapted_result, method=method)

        eval_results: dict[str, Any] = {}
        baseline_eval_metrics = None
        adapted_eval_metrics = None
        max_images = int(args.max_images) if args.max_images is not None else None
        if not args.skip_eval:
            stage = "baseline_eval_launcher"
            _set_plan_status(plan, plan_json, state="running", stage=stage)
            baseline_eval_result = _run(
                plan["commands"]["baseline_eval"], cwd=repo_root
            )
            eval_results["baseline_eval"] = baseline_eval_result
            stage = (
                "baseline_eval_proxy"
                if _should_use_simple_map_proxy(baseline_eval_result)
                else "baseline_eval_result"
            )
            _set_plan_status(plan, plan_json, state="running", stage=stage)
            baseline_eval_metrics = _eval_or_proxy(
                result=baseline_eval_result,
                output_path=baseline_eval,
                dataset=dataset,
                split=str(args.split),
                predictions_path=baseline_predictions,
                max_images=max_images,
                label="baseline eval",
            )

            stage = "adapted_eval_launcher"
            _set_plan_status(plan, plan_json, state="running", stage=stage)
            adapted_eval_result = _run(plan["commands"]["adapted_eval"], cwd=repo_root)
            eval_results["adapted_eval"] = adapted_eval_result
            stage = (
                "adapted_eval_proxy"
                if _should_use_simple_map_proxy(adapted_eval_result)
                else "adapted_eval_result"
            )
            _set_plan_status(plan, plan_json, state="running", stage=stage)
            adapted_eval_metrics = _eval_or_proxy(
                result=adapted_eval_result,
                output_path=adapted_eval,
                dataset=dataset,
                split=str(args.split),
                predictions_path=adapted_predictions,
                max_images=max_images,
                label="adapted eval",
            )

        stage = "artifact_validation"
        _set_plan_status(plan, plan_json, state="running", stage=stage)
        baseline_payload = _load_json(baseline_predictions)
        adapted_payload = _load_json(adapted_predictions)

        stage = "prediction_comparison"
        _set_plan_status(plan, plan_json, state="running", stage=stage)
        parity_report = compare_predictions(
            reference=baseline_predictions,
            candidate=adapted_predictions,
            max_images=args.max_images,
            iou_thresh=float(
                (boilerplate.get("compare") or {}).get("iou_thresh", 0.99)
            ),
            score_atol=float(
                (boilerplate.get("compare") or {}).get("score_atol", 1e-4)
            ),
            bbox_atol=float((boilerplate.get("compare") or {}).get("bbox_atol", 1e-4)),
        )

        method_profile = get_ttt_method_profile(method)
        report: dict[str, Any] = {
            "schema_version": 1,
            "kind": "ttt_before_after_compare",
            "evidence_kind": "local_diagnostic",
            "efficacy_conclusion": "not_established",
            "timestamp": _now_utc(),
            "boilerplate_name": requested_boilerplate,
            "boilerplate_path": str(boilerplate_path),
            "method": method,
            "method_profile": method_profile,
            "promotion_eligible": False,
            "efficacy": "not_established",
            "preset": preset,
            "reset": reset,
            "artifacts": {
                "plan_json": _short(plan_json),
                "baseline_predictions": _short(baseline_predictions),
                "adapted_predictions": _short(adapted_predictions),
                "adapted_ttt_log": _short(adapted_ttt_log),
                "compare_json": _short(compare_json),
                "compare_md": _short(compare_md),
                "baseline_eval": _short(baseline_eval)
                if baseline_eval.is_file()
                else None,
                "adapted_eval": _short(adapted_eval)
                if adapted_eval.is_file()
                else None,
            },
            "inputs": {
                "dataset": str(dataset),
                "split": str(args.split),
                "checkpoint": str(checkpoint),
                "backend": str(args.backend),
                "device": str(args.device),
                "boilerplate_sha256": prerequisites["boilerplate_sha256"],
                "checkpoint_sha256": prerequisites["checkpoint_sha256"],
                "configs": prerequisites["configs"],
            },
            "provenance": {
                "evidence_kind": "local_diagnostic",
                "model_preflight": prerequisites["model_preflight"],
                "checkpoint_sha256": prerequisites["checkpoint_sha256"],
                "boilerplate_sha256": prerequisites["boilerplate_sha256"],
                "configs": prerequisites["configs"],
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
        baseline_summary = report["baseline"]["ttt_summary"]
        adapted_summary = report["adapted"]["ttt_summary"]
        baseline_seconds = baseline_summary.get("mean_seconds")
        adapted_seconds = adapted_summary.get("mean_seconds")
        latency_delta = None
        if baseline_seconds is not None and adapted_seconds is not None:
            latency_delta = float(adapted_seconds) - float(baseline_seconds)
        report["research_report"] = {
            "kind": "research_lane_report",
            "lane": "ttt",
            "evidence_kind": "local_diagnostic",
            "efficacy_conclusion": "not_established",
            "stable_baseline_artifact": _short(baseline_predictions),
            "research_output_artifact": _short(adapted_predictions),
            "report_artifact": _short(compare_json),
            "latency_overhead": {
                "baseline_mean_seconds": baseline_seconds,
                "adapted_mean_seconds": adapted_seconds,
                "delta_mean_seconds": latency_delta,
            },
            "rollback": {
                "reset_policy": reset,
                "rollback_steps": int(adapted_summary.get("rollback_steps") or 0),
                "stopped_early_count": int(
                    adapted_summary.get("stopped_early_count") or 0
                ),
                "guard_breaches": int(adapted_summary.get("guard_breaches") or 0),
            },
            "promotion_gate": {
                "decision": "not_eligible",
                "reason": (
                    "TTT efficacy is not established; this diagnostic compare cannot "
                    "promote a checkpoint."
                ),
            },
        }
        if eval_results:
            report["eval_commands"] = {
                key: " ".join(shlex.quote(x) for x in value.get("command") or [])
                for key, value in eval_results.items()
            }
            report["eval_status"] = {
                key: {
                    "returncode": int(value.get("returncode") or 0),
                    "stderr": str(value.get("stderr") or ""),
                }
                for key, value in eval_results.items()
            }

        stage = "report_write"
        _set_plan_status(plan, plan_json, state="running", stage=stage)
        _write_report_pair(
            compare_json,
            compare_md,
            report=report,
            markdown=_render_markdown(report),
        )
        _set_plan_status(plan, plan_json, state="completed", stage="complete")
        print(compare_json)
        print(compare_md)
        return 0
    except BaseException as exc:
        _set_plan_status(plan, plan_json, state="failed", stage=stage, error=exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
