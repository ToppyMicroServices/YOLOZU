#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
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
        sub = item.get("report")
        if isinstance(sub, dict):
            out.append(sub)
    return out


def _guard_breaches(warnings: list[str]) -> int:
    total = 0
    for warning in warnings:
        w = str(warning)
        if "exceeded" in w or "non_finite" in w:
            total += 1
    return total


def _summarize_variant(report: dict[str, Any]) -> dict[str, Any]:
    runs = _flatten_reports(report)
    losses: list[float] = []
    final_losses: list[float] = []
    warnings: list[str] = []
    seconds: list[float] = []

    for rep in runs:
        rep_losses = rep.get("losses")
        if isinstance(rep_losses, list):
            vals = [float(x) for x in rep_losses]
            losses.extend(vals)
            if vals:
                final_losses.append(float(vals[-1]))

        rep_warn = rep.get("warnings")
        if isinstance(rep_warn, list):
            warnings.extend([str(x) for x in rep_warn])

        sec = rep.get("seconds")
        if sec is not None:
            seconds.append(_safe_float(sec))

    mean_loss = float(sum(losses) / len(losses)) if losses else 0.0
    mean_final_loss = float(sum(final_losses) / len(final_losses)) if final_losses else 0.0
    std_final_loss = float(statistics.pstdev(final_losses)) if len(final_losses) > 1 else 0.0
    mean_seconds = float(sum(seconds) / len(seconds)) if seconds else _safe_float(report.get("seconds"), default=0.0)

    return {
        "runs_count": int(len(runs)),
        "mean_loss": float(mean_loss),
        "mean_final_loss": float(mean_final_loss),
        "std_final_loss": float(std_final_loss),
        "mean_seconds": float(mean_seconds),
        "guard_breaches": int(_guard_breaches(warnings)),
        "warnings": warnings,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    cotta = report.get("cotta") or {}
    eata = report.get("eata") or {}
    sar = report.get("sar") or {}
    side = report.get("side_effects") or {}
    decision = report.get("decision") or {}

    lines = [
        "# SAR robustness impact report",
        "",
        f"- Generated UTC: {report.get('timestamp')}",
        f"- Go decision: `{decision.get('go')}`",
        f"- Summary: `{decision.get('summary')}`",
        "",
        "## Variants",
        "",
        "| Variant | Mean Final Loss | Final Loss Std | Mean Seconds | Guard Breaches |",
        "|---|---:|---:|---:|---:|",
        (
            "| cotta | "
            f"{_safe_float(cotta.get('mean_final_loss')):.6f} | "
            f"{_safe_float(cotta.get('std_final_loss')):.6f} | "
            f"{_safe_float(cotta.get('mean_seconds')):.6f} | "
            f"{int(cotta.get('guard_breaches') or 0)} |"
        ),
        (
            "| eata | "
            f"{_safe_float(eata.get('mean_final_loss')):.6f} | "
            f"{_safe_float(eata.get('std_final_loss')):.6f} | "
            f"{_safe_float(eata.get('mean_seconds')):.6f} | "
            f"{int(eata.get('guard_breaches') or 0)} |"
        ),
        (
            "| sar | "
            f"{_safe_float(sar.get('mean_final_loss')):.6f} | "
            f"{_safe_float(sar.get('std_final_loss')):.6f} | "
            f"{_safe_float(sar.get('mean_seconds')):.6f} | "
            f"{int(sar.get('guard_breaches') or 0)} |"
        ),
        "",
        "## Side effects",
        "",
        f"- overhead_vs_best_baseline: `{_safe_float(side.get('overhead_vs_best_baseline')):.6f}`",
        f"- loss_delta_vs_best_baseline: `{_safe_float(side.get('loss_delta_vs_best_baseline')):.6f}`",
        f"- variance_delta_vs_best_baseline: `{_safe_float(side.get('variance_delta_vs_best_baseline')):.6f}`",
        f"- guard_breach_delta_vs_best_baseline: `{int(side.get('guard_breach_delta_vs_best_baseline') or 0)}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SAR robustness impact against CoTTA and EATA baselines.")
    p.add_argument("--cotta", required=True, help="CoTTA wrapped predictions JSON path.")
    p.add_argument("--eata", required=True, help="EATA wrapped predictions JSON path.")
    p.add_argument("--sar", required=True, help="SAR wrapped predictions JSON path.")
    p.add_argument("--output-json", default="reports/sar_robustness_report.json")
    p.add_argument("--output-md", default="reports/sar_robustness_report.md")
    p.add_argument("--max-overhead-ratio", type=float, default=1.5)
    p.add_argument("--max-loss-ratio", type=float, default=1.05)
    p.add_argument("--max-variance-ratio", type=float, default=1.2)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    cotta_path = _resolve(str(args.cotta))
    eata_path = _resolve(str(args.eata))
    sar_path = _resolve(str(args.sar))

    cotta_summary = _summarize_variant(_extract_ttt(_load_json(cotta_path), label="cotta"))
    eata_summary = _summarize_variant(_extract_ttt(_load_json(eata_path), label="eata"))
    sar_summary = _summarize_variant(_extract_ttt(_load_json(sar_path), label="sar"))

    baselines = [cotta_summary, eata_summary]
    best_baseline = min(baselines, key=lambda x: _safe_float(x.get("mean_final_loss"), default=float("inf")))

    best_loss = _safe_float(best_baseline.get("mean_final_loss"), default=1e-12)
    best_var = _safe_float(best_baseline.get("std_final_loss"), default=0.0)
    best_sec = max(1e-12, _safe_float(best_baseline.get("mean_seconds"), default=1e-12))
    best_guard = int(best_baseline.get("guard_breaches") or 0)

    sar_loss = _safe_float(sar_summary.get("mean_final_loss"), default=0.0)
    sar_var = _safe_float(sar_summary.get("std_final_loss"), default=0.0)
    sar_sec = _safe_float(sar_summary.get("mean_seconds"), default=0.0)
    sar_guard = int(sar_summary.get("guard_breaches") or 0)

    overhead_ratio = sar_sec / best_sec
    loss_ratio = sar_loss / best_loss if best_loss > 0.0 else 0.0
    variance_ratio = (sar_var / best_var) if best_var > 0.0 else 0.0

    side_effects = {
        "overhead_vs_best_baseline": float(overhead_ratio),
        "loss_delta_vs_best_baseline": float(sar_loss - best_loss),
        "variance_delta_vs_best_baseline": float(sar_var - best_var),
        "guard_breach_delta_vs_best_baseline": int(sar_guard - best_guard),
    }

    robustness_gain = bool(sar_loss <= best_loss and sar_guard <= best_guard)
    acceptable_overhead = bool(overhead_ratio <= float(args.max_overhead_ratio))
    acceptable_loss = bool(loss_ratio <= float(args.max_loss_ratio)) if best_loss > 0.0 else True
    acceptable_variance = bool(variance_ratio <= float(args.max_variance_ratio)) if best_var > 0.0 else True
    go = bool(robustness_gain and acceptable_overhead and acceptable_loss and acceptable_variance)

    report = {
        "schema_version": 1,
        "kind": "sar_robustness_impact",
        "timestamp": _now_utc(),
        "inputs": {
            "cotta": {"path": str(cotta_path), "sha256": _sha256(cotta_path)},
            "eata": {"path": str(eata_path), "sha256": _sha256(eata_path)},
            "sar": {"path": str(sar_path), "sha256": _sha256(sar_path)},
        },
        "thresholds": {
            "max_overhead_ratio": float(args.max_overhead_ratio),
            "max_loss_ratio": float(args.max_loss_ratio),
            "max_variance_ratio": float(args.max_variance_ratio),
        },
        "cotta": cotta_summary,
        "eata": eata_summary,
        "sar": sar_summary,
        "side_effects": side_effects,
        "checks": {
            "robustness_gain": bool(robustness_gain),
            "acceptable_overhead": bool(acceptable_overhead),
            "acceptable_loss": bool(acceptable_loss),
            "acceptable_variance": bool(acceptable_variance),
        },
        "decision": {
            "go": bool(go),
            "summary": ("go" if go else "no-go"),
            "recommended_next": (
                "promote_sar_for_limited_rollout" if go else "keep_sar_experimental_tune_rho_or_scope"
            ),
        },
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
