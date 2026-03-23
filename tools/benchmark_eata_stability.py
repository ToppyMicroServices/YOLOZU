#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
_NUMERIC_ERRORS = (TypeError, ValueError, OverflowError)


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except _NUMERIC_ERRORS:
        return float(default)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _extract_ttt(payload: dict[str, Any], *, label: str) -> dict[str, Any]:
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"{label}: missing meta")
    ttt = meta.get("ttt")
    if not isinstance(ttt, dict):
        raise ValueError(f"{label}: missing meta.ttt")
    report = ttt.get("report")
    if not isinstance(report, dict):
        raise ValueError(f"{label}: missing meta.ttt.report")
    return report


def _flatten_reports(report: dict[str, Any]) -> list[dict[str, Any]]:
    mode = str(report.get("mode") or "stream")
    if mode != "sample":
        return [report]
    out: list[dict[str, Any]] = []
    per_sample = report.get("per_sample")
    if not isinstance(per_sample, list):
        return out
    for item in per_sample:
        if not isinstance(item, dict):
            continue
        rep = item.get("report")
        if isinstance(rep, dict):
            out.append(rep)
    return out


def _guard_breaches(warnings: list[str]) -> int:
    total = 0
    for warning in warnings:
        w = str(warning)
        if "exceeded" in w or "non_finite" in w:
            total += 1
    return total


def _summarize(report: dict[str, Any]) -> dict[str, Any]:
    runs = _flatten_reports(report)
    all_losses: list[float] = []
    final_losses: list[float] = []
    warnings: list[str] = []
    selected_ratios: list[float] = []
    seconds: list[float] = []
    steps_total = 0

    for rep in runs:
        losses = rep.get("losses")
        if isinstance(losses, list):
            vals = [float(x) for x in losses]
            all_losses.extend(vals)
            if vals:
                final_losses.append(float(vals[-1]))
            steps_total += len(vals)

        ws = rep.get("warnings")
        if isinstance(ws, list):
            warnings.extend([str(x) for x in ws])

        sec = rep.get("seconds")
        if sec is not None:
            seconds.append(_safe_float(sec))

        step_metrics = rep.get("step_metrics")
        if isinstance(step_metrics, list):
            for step in step_metrics:
                if not isinstance(step, dict):
                    continue
                if "selected_ratio" in step and step.get("selected_ratio") is not None:
                    selected_ratios.append(_safe_float(step.get("selected_ratio")))

    mean_loss = float(sum(all_losses) / len(all_losses)) if all_losses else 0.0
    mean_final_loss = float(sum(final_losses) / len(final_losses)) if final_losses else 0.0
    mean_selected_ratio = float(sum(selected_ratios) / len(selected_ratios)) if selected_ratios else 0.0
    mean_seconds = float(sum(seconds) / len(seconds)) if seconds else _safe_float(report.get("seconds"), default=0.0)

    return {
        "runs_count": int(len(runs)),
        "steps_count": int(steps_total),
        "mean_loss": float(mean_loss),
        "mean_final_loss": float(mean_final_loss),
        "mean_selected_ratio": float(mean_selected_ratio),
        "mean_seconds": float(mean_seconds),
        "guard_breaches": int(_guard_breaches(warnings)),
        "warnings": warnings,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    baseline = report.get("baseline") or {}
    eata = report.get("eata") or {}
    trade = report.get("tradeoff") or {}
    rec = report.get("recommended_defaults") or {}

    lines = [
        "# EATA stability/efficiency benchmark",
        "",
        f"- Generated UTC: {report.get('timestamp')}",
        f"- Recommend EATA defaults: `{bool(rec.get('enabled'))}`",
        "",
        "## Baseline vs EATA",
        "",
        "| Variant | Mean Final Loss | Guard Breaches | Mean Seconds | Mean Selected Ratio |",
        "|---|---:|---:|---:|---:|",
        (
            "| baseline | "
            f"{_safe_float(baseline.get('mean_final_loss')):.6f} | "
            f"{int(baseline.get('guard_breaches') or 0)} | "
            f"{_safe_float(baseline.get('mean_seconds')):.6f} | "
            f"{_safe_float(baseline.get('mean_selected_ratio')):.6f} |"
        ),
        (
            "| eata | "
            f"{_safe_float(eata.get('mean_final_loss')):.6f} | "
            f"{int(eata.get('guard_breaches') or 0)} | "
            f"{_safe_float(eata.get('mean_seconds')):.6f} | "
            f"{_safe_float(eata.get('mean_selected_ratio')):.6f} |"
        ),
        "",
        "## Tradeoff summary",
        "",
        f"- loss_delta_vs_baseline: `{_safe_float(trade.get('loss_delta_vs_baseline')):.6f}`",
        f"- guard_breach_delta: `{int(trade.get('guard_breach_delta') or 0)}`",
        f"- overhead_ratio: `{_safe_float(trade.get('overhead_ratio')):.6f}`",
        "",
    ]

    if rec:
        lines.extend(["## Recommended defaults", ""])
        for k, v in rec.items():
            lines.append(f"- {k}: `{v}`")

    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark EATA stability/efficiency tradeoffs versus baseline TTT runs.")
    p.add_argument("--baseline", required=True, help="Baseline wrapped predictions JSON path.")
    p.add_argument("--eata", required=True, help="EATA wrapped predictions JSON path.")
    p.add_argument("--output-json", default="reports/eata_benchmark.json")
    p.add_argument("--output-md", default="reports/eata_benchmark.md")
    p.add_argument("--max-overhead-ratio", type=float, default=1.5, help="Maximum acceptable EATA runtime overhead ratio.")
    p.add_argument("--max-loss-ratio", type=float, default=1.05, help="Maximum acceptable EATA/baseline final-loss ratio.")
    p.add_argument("--min-selected-ratio", type=float, default=0.1, help="Minimum mean selected-sample ratio for stable adaptation.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    baseline_path = _resolve(str(args.baseline))
    eata_path = _resolve(str(args.eata))
    baseline_payload = _load_json(baseline_path)
    eata_payload = _load_json(eata_path)

    baseline_summary = _summarize(_extract_ttt(baseline_payload, label="baseline"))
    eata_summary = _summarize(_extract_ttt(eata_payload, label="eata"))

    baseline_final = _safe_float(baseline_summary.get("mean_final_loss"), default=0.0)
    eata_final = _safe_float(eata_summary.get("mean_final_loss"), default=0.0)
    baseline_seconds = max(1e-12, _safe_float(baseline_summary.get("mean_seconds"), default=0.0))
    eata_seconds = _safe_float(eata_summary.get("mean_seconds"), default=0.0)
    overhead_ratio = eata_seconds / baseline_seconds
    loss_ratio = eata_final / baseline_final if baseline_final > 0.0 else 0.0

    tradeoff = {
        "loss_delta_vs_baseline": float(eata_final - baseline_final),
        "guard_breach_delta": int(eata_summary.get("guard_breaches", 0)) - int(baseline_summary.get("guard_breaches", 0)),
        "overhead_ratio": float(overhead_ratio),
        "loss_ratio": float(loss_ratio),
    }

    stable_loss = bool(loss_ratio <= float(args.max_loss_ratio) if baseline_final > 0.0 else True)
    stable_guard = int(eata_summary.get("guard_breaches", 0)) <= int(baseline_summary.get("guard_breaches", 0))
    stable_select = _safe_float(eata_summary.get("mean_selected_ratio"), default=0.0) >= float(args.min_selected_ratio)
    efficient = bool(overhead_ratio <= float(args.max_overhead_ratio))
    recommend = bool(stable_loss and stable_guard and stable_select and efficient)

    recommended_defaults = {
        "enabled": bool(recommend),
        "ttt_method": "eata",
        "ttt_preset": "eata_safe",
        "ttt_update_filter": "lora_norm_only",
        "ttt_eata_conf_min": 0.2,
        "ttt_eata_entropy_min": 0.05,
        "ttt_eata_entropy_max": 3.0,
        "ttt_eata_min_valid_dets": 1,
        "ttt_eata_anchor_lambda": 0.001,
    }

    report = {
        "schema_version": 1,
        "kind": "eata_stability_efficiency_benchmark",
        "timestamp": _now_utc(),
        "inputs": {
            "baseline": {"path": str(baseline_path), "sha256": _sha256(baseline_path)},
            "eata": {"path": str(eata_path), "sha256": _sha256(eata_path)},
        },
        "thresholds": {
            "max_overhead_ratio": float(args.max_overhead_ratio),
            "max_loss_ratio": float(args.max_loss_ratio),
            "min_selected_ratio": float(args.min_selected_ratio),
        },
        "baseline": baseline_summary,
        "eata": eata_summary,
        "tradeoff": tradeoff,
        "checks": {
            "stable_loss": bool(stable_loss),
            "stable_guard": bool(stable_guard),
            "stable_select": bool(stable_select),
            "efficient": bool(efficient),
        },
        "recommended_defaults": recommended_defaults,
    }

    out_json = _resolve(str(args.output_json))
    out_md = _resolve(str(args.output_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    out_md.write_text(_render_markdown(report), encoding="utf-8")

    print(out_json)
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
