#!/usr/bin/env python3
"""Run a fail-closed, multi-seed TTT evidence matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


repo_root = Path(__file__).resolve().parents[1]
SUPPORTED_METHODS = ("tent", "mim", "cotta", "eata", "sar")
SAMPLE_METHODS = {"tent", "mim", "eata", "sar"}
STREAM_METHODS = {"cotta"}


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _csv_tokens(value: str) -> list[str]:
    return [token.strip().lower() for token in str(value).split(",") if token.strip()]


def _seed_tokens(value: str) -> list[int]:
    seeds = [int(token.strip()) for token in str(value).split(",") if token.strip()]
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique integers")
    if len(seeds) < 3:
        raise ValueError("--seeds requires at least three unique integers")
    return seeds


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run clean and shifted TTT comparisons for at least three seeds and "
            "aggregate generated COCO, calibration, collapse, and cost metrics."
        )
    )
    parser.add_argument("-d", "--dataset", required=True, help="Clean YOLO dataset.")
    parser.add_argument(
        "-x", "--shifted-dataset", required=True, help="Deterministic shifted dataset."
    )
    parser.add_argument("-s", "--split", default="train2017", help="Dataset split.")
    parser.add_argument(
        "-c", "--checkpoint", required=True, help="Full-compatible base checkpoint."
    )
    parser.add_argument(
        "--mim-checkpoint", required=True, help="Full-compatible MIM checkpoint."
    )
    parser.add_argument(
        "-o", "--out", required=True, help="Evidence output directory."
    )
    parser.add_argument(
        "--methods",
        default="tent,mim,cotta,eata,sar",
        help="Comma-separated method list.",
    )
    parser.add_argument(
        "--seeds", default="11,22,33", help="At least three unique integer seeds."
    )
    parser.add_argument("-n", "--max-images", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument(
        "--protocol",
        choices=("yolo26", "nms_applied", "e2e_nms_free"),
        default=None,
        help="Optional canonical eval protocol; omit for arbitrary dataset splits.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and write child plans only."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite known child artifacts; directories are never recursively deleted.",
    )
    return parser.parse_args(argv)


def _validate(args: argparse.Namespace) -> tuple[list[str], list[int]]:
    methods = _csv_tokens(args.methods)
    if not methods:
        raise ValueError("--methods must not be empty")
    unsupported = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unsupported:
        raise ValueError(f"unsupported methods: {', '.join(unsupported)}")
    if len(set(methods)) != len(methods):
        raise ValueError("--methods must not contain duplicates")
    seeds = _seed_tokens(args.seeds)
    if int(args.max_images) <= 0:
        raise ValueError("--max-images must be > 0")
    if int(args.image_size) <= 0:
        raise ValueError("--image-size must be > 0")
    if not 0.0 <= float(args.score_threshold) <= 1.0:
        raise ValueError("--score-threshold must be in [0, 1]")
    if int(args.max_detections) <= 0:
        raise ValueError("--max-detections must be > 0")
    return methods, seeds


def _run(command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return {
        "command": list(command),
        "returncode": int(proc.returncode),
        "stdout": str(proc.stdout),
        "stderr": str(proc.stderr),
        "wall_seconds": float(time.perf_counter() - started),
    }


def _metric(report: dict[str, Any], variant: str, name: str) -> float | None:
    section = report.get(f"{variant}_eval")
    if not isinstance(section, dict):
        return None
    value = section.get(name)
    return float(value) if isinstance(value, (int, float)) else None


def _row(
    *,
    report: dict[str, Any],
    report_path: Path,
    domain: str,
    method: str,
    seed: int,
) -> dict[str, Any]:
    summary = ((report.get("adapted") or {}).get("ttt_summary") or {})
    baseline_diag = ((report.get("baseline") or {}).get("diagnostics") or {})
    adapted_diag = ((report.get("adapted") or {}).get("diagnostics") or {})
    research = report.get("research_report") or {}
    baseline_ap = _metric(report, "baseline", "map50_95")
    adapted_ap = _metric(report, "adapted", "map50_95")
    return {
        "domain": domain,
        "method": method,
        "protocol": "continual_stream" if method in STREAM_METHODS else "sample_reset",
        "seed": int(seed),
        "checkpoint_sha256": str(
            ((report.get("inputs") or {}).get("checkpoint_sha256")) or ""
        ),
        "dataset_images": int(
            ((report.get("inputs") or {}).get("dataset_images")) or 0
        ),
        "dataset_order_sha256": str(
            ((report.get("inputs") or {}).get("dataset_order_sha256")) or ""
        ),
        "dataset_content_sha256": str(
            ((report.get("inputs") or {}).get("dataset_content_sha256")) or ""
        ),
        "baseline_map50_95": baseline_ap,
        "adapted_map50_95": adapted_ap,
        "map50_95_delta": (
            float(adapted_ap) - float(baseline_ap)
            if adapted_ap is not None and baseline_ap is not None
            else None
        ),
        "baseline_calibration": baseline_diag.get("calibration"),
        "adapted_calibration": adapted_diag.get("calibration"),
        "baseline_collapse": baseline_diag.get("collapse"),
        "adapted_collapse": adapted_diag.get("collapse"),
        "update_ratio": summary.get("mean_update_ratio"),
        "forward_calls": int(summary.get("forward_calls") or 0),
        "backward_calls": int(summary.get("backward_calls") or 0),
        "optimizer_steps": int(summary.get("optimizer_steps") or 0),
        "peak_memory_bytes": summary.get("peak_memory_bytes"),
        "memory_metrics": summary.get("memory_metrics") or [],
        "latency": research.get("latency_overhead"),
        "guard_breaches": int(summary.get("guard_breaches") or 0),
        "stopped_early_count": int(summary.get("stopped_early_count") or 0),
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
    }


def _mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _stdev(values: list[float]) -> float | None:
    return float(statistics.stdev(values)) if len(values) >= 2 else None


def _aggregate(rows: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        clean = [row for row in method_rows if row["domain"] == "clean"]
        shifted = [row for row in method_rows if row["domain"] == "shifted"]
        clean_delta = [
            float(row["map50_95_delta"])
            for row in clean
            if row.get("map50_95_delta") is not None
        ]
        shifted_delta = [
            float(row["map50_95_delta"])
            for row in shifted
            if row.get("map50_95_delta") is not None
        ]
        shifted_ap = [
            float(row["adapted_map50_95"])
            for row in shifted
            if row.get("adapted_map50_95") is not None
        ]
        out[method] = {
            "protocol": (
                "continual_stream" if method in STREAM_METHODS else "sample_reset"
            ),
            "seeds": sorted({int(row["seed"]) for row in method_rows}),
            "clean_retention_delta_mean": _mean(clean_delta),
            "clean_retention_delta_stdev": _stdev(clean_delta),
            "shifted_improvement_delta_mean": _mean(shifted_delta),
            "shifted_improvement_delta_stdev": _stdev(shifted_delta),
            "worst_domain_adapted_map50_95": min(shifted_ap)
            if shifted_ap
            else None,
            "guard_breaches": sum(int(row["guard_breaches"]) for row in method_rows),
            "stopped_early_count": sum(
                int(row["stopped_early_count"]) for row in method_rows
            ),
            "efficacy_conclusion": "not_established",
        }
    return out


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TTT multi-seed evidence summary",
        "",
        f"- Generated UTC: `{summary['timestamp']}`",
        f"- Seeds: `{', '.join(str(seed) for seed in summary['seeds'])}`",
        f"- Images per run: `{summary['max_images']}`",
        f"- COCO backend required: `{summary['metric_gate']['pycocotools_required']}`",
        f"- Efficacy conclusion: `{summary['efficacy_conclusion']}`",
        "",
        "Sample-reset and continual-stream results are kept separate.",
        "",
        "| Method | Protocol | Clean retention Δ mAP50:95 | Shifted Δ mAP50:95 | Worst shifted mAP50:95 |",
        "|---|---|---:|---:|---:|",
    ]
    for method, item in summary["aggregate"].items():
        def fmt(value: Any) -> str:
            return "n/a" if value is None else f"{float(value):.6f}"

        lines.append(
            f"| {method} | {item['protocol']} | "
            f"{fmt(item['clean_retention_delta_mean'])} | "
            f"{fmt(item['shifted_improvement_delta_mean'])} | "
            f"{fmt(item['worst_domain_adapted_map50_95'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These runs are generated local diagnostics. They do not establish "
            + "method efficacy or independently reproduce a release result.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    methods, seeds = _validate(args)
    clean = _resolve(args.dataset)
    shifted = _resolve(args.shifted_dataset)
    checkpoint = _resolve(args.checkpoint)
    mim_checkpoint = _resolve(args.mim_checkpoint)
    out = _resolve(args.out)
    for path, label in (
        (clean, "clean dataset"),
        (shifted, "shifted dataset"),
        (checkpoint, "checkpoint"),
        (mim_checkpoint, "MIM checkpoint"),
    ):
        expected = path.is_dir() if "dataset" in label else path.is_file()
        if not expected:
            raise FileNotFoundError(f"{label} not found: {path}")
    if out.exists() and any(out.iterdir()) and not args.force:
        raise FileExistsError(f"output is not empty: {out} (use --force)")
    out.mkdir(parents=True, exist_ok=True)

    matrix: list[dict[str, Any]] = []
    for domain, dataset in (("clean", clean), ("shifted", shifted)):
        for method in methods:
            selected_checkpoint = mim_checkpoint if method == "mim" else checkpoint
            for seed in seeds:
                run_dir = out / "runs" / domain / method / f"seed_{seed}"
                command = [
                    sys.executable,
                    str(repo_root / "tools" / "run_ttt_compare.py"),
                    "--method",
                    method,
                    "--data",
                    str(dataset),
                    "--split",
                    str(args.split),
                    "--weights",
                    str(selected_checkpoint),
                    "--out",
                    str(run_dir),
                    "--device",
                    str(args.device),
                    "--max-images",
                    str(int(args.max_images)),
                    "--image-size",
                    str(int(args.image_size)),
                    "--seed",
                    str(int(seed)),
                    "--score-threshold",
                    str(float(args.score_threshold)),
                    "--max-detections",
                    str(int(args.max_detections)),
                    "--dataset-hash-mode",
                    "content",
                    "--force",
                ]
                if args.protocol:
                    command.extend(["--protocol", str(args.protocol)])
                if args.dry_run:
                    command.append("--dry-run")
                matrix.append(
                    {
                        "domain": domain,
                        "dataset": str(dataset),
                        "method": method,
                        "protocol": (
                            "continual_stream"
                            if method in STREAM_METHODS
                            else "sample_reset"
                        ),
                        "seed": int(seed),
                        "checkpoint": str(selected_checkpoint),
                        "run_dir": str(run_dir),
                        "command": command,
                    }
                )

    plan = {
        "schema_version": 1,
        "kind": "ttt_evidence_suite_plan",
        "timestamp": _now_utc(),
        "state": "running",
        "methods": methods,
        "seeds": seeds,
        "matrix": matrix,
    }
    plan_path = out / "plan.json"
    _atomic_json(plan_path, plan)

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(matrix):
        result = _run(item["command"])
        item["result"] = {
            "returncode": int(result["returncode"]),
            "wall_seconds": float(result["wall_seconds"]),
            "stdout": str(result["stdout"]),
            "stderr": str(result["stderr"]),
        }
        plan["completed_runs"] = int(index + 1)
        _atomic_json(plan_path, plan)
        if int(result["returncode"]) != 0:
            plan["state"] = "failed"
            plan["failed_run"] = item
            _atomic_json(plan_path, plan)
            rendered = " ".join(shlex.quote(part) for part in item["command"])
            raise RuntimeError(
                f"TTT evidence child failed:\n$ {rendered}\n"
                f"{result['stdout']}{result['stderr']}"
            )
        if args.dry_run:
            continue
        report_path = (
            Path(item["run_dir"])
            / f"{item['method']}_before_after_compare.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows.append(
            _row(
                report=report,
                report_path=report_path,
                domain=str(item["domain"]),
                method=str(item["method"]),
                seed=int(item["seed"]),
            )
        )

    plan["state"] = "not_executed" if args.dry_run else "completed"
    plan["finished_at"] = _now_utc()
    _atomic_json(plan_path, plan)
    if args.dry_run:
        print(plan_path)
        return 0

    for domain in ("clean", "shifted"):
        for method in methods:
            hashes = {
                str(row["dataset_order_sha256"])
                for row in rows
                if row["domain"] == domain and row["method"] == method
            }
            if len(hashes) != 1:
                raise RuntimeError(
                    f"dataset order changed across seeds: domain={domain} method={method}"
                )

    summary = {
        "schema_version": 1,
        "kind": "ttt_multi_seed_evidence",
        "timestamp": _now_utc(),
        "efficacy_conclusion": "not_established",
        "promotion_eligible": False,
        "methods": methods,
        "seeds": seeds,
        "max_images": int(args.max_images),
        "image_size": int(args.image_size),
        "checkpoints": {
            "base": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
            "mim": {
                "path": str(mim_checkpoint),
                "sha256": _sha256(mim_checkpoint),
            },
        },
        "protocol_separation": {
            "sample_reset": sorted(set(methods) & SAMPLE_METHODS),
            "continual_stream": sorted(set(methods) & STREAM_METHODS),
        },
        "metric_gate": {
            "pycocotools_required": True,
            "calibration_status_is_explicit_when_no_detections": True,
        },
        "independent_reproduction": {
            "status": "not_satisfied",
            "reason": "all runs were produced by one local environment",
        },
        "rows": rows,
        "aggregate": _aggregate(rows, methods),
    }
    summary_json = out / "summary.json"
    summary_md = out / "summary.md"
    _atomic_json(summary_json, summary)
    _atomic_text(summary_md, _render_markdown(summary))
    print(summary_json)
    print(summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
