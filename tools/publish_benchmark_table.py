#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(repo_root))

from yolozu.eval_protocol import eval_protocol_hash, load_eval_protocol, validate_eval_protocol


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _default_run_id() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report must be JSON object: {path}")
    return payload


def _extract_rows(report: dict[str, Any], *, source_report: str) -> list[dict[str, Any]]:
    meta = report.get("meta") or {}
    metrics = report.get("metrics") or {}
    run_id = str(meta.get("run_id") or "unknown")
    git_head = meta.get("git_head")

    rows: list[dict[str, Any]] = []
    buckets = metrics.get("buckets")
    if isinstance(buckets, list) and buckets:
        for bucket in buckets:
            if not isinstance(bucket, dict):
                continue
            bucket_name = str(bucket.get("name") or "unknown")
            bucket_metrics = bucket.get("metrics") or {}
            latency = bucket_metrics.get("latency_ms") or {}
            rows.append(
                {
                    "bucket": bucket_name,
                    "fps": float(bucket_metrics.get("fps") or 0.0),
                    "latency_ms_mean": float(latency.get("mean") or 0.0),
                    "run_id": run_id,
                    "git_head": git_head,
                    "source_report": source_report,
                }
            )
        return rows

    summary = metrics.get("summary") or {}
    rows.append(
        {
            "bucket": "summary",
            "fps": float(summary.get("fps_mean") or 0.0),
            "latency_ms_mean": float(summary.get("latency_ms_mean") or 0.0),
            "run_id": run_id,
            "git_head": git_head,
            "source_report": source_report,
        }
    )
    return rows


def _render_markdown(*, title: str, cadence: str, protocol_id: str, protocol_hash: str, rows: list[dict[str, Any]], source_commands: list[str]) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Generated UTC: {_now_utc()}",
        f"- Protocol: `{protocol_id}`",
        f"- Protocol hash: `{protocol_hash}`",
        f"- Cadence: `{cadence}`",
        "",
        "## Official benchmark table",
        "",
        "| Bucket | FPS | Latency Mean (ms) | Run ID | Source Report |",
        "|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('bucket')} | {float(row.get('fps') or 0.0):.3f} | {float(row.get('latency_ms_mean') or 0.0):.3f} | "
            f"{row.get('run_id')} | {row.get('source_report')} |"
        )

    if source_commands:
        lines.extend(["", "## Source commands", ""])
        for cmd in source_commands:
            lines.append(f"- `{cmd}`")

    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate official benchmark publication table from benchmark reports.")
    p.add_argument("--report", action="append", required=True, help="Repeatable benchmark report JSON path.")
    p.add_argument("--output-json", default="reports/benchmark_table.json")
    p.add_argument("--output-md", default="reports/benchmark_table.md")
    p.add_argument("--title", default="YOLOZU Official Benchmark Table")
    p.add_argument("--cadence", default="weekly")
    p.add_argument("--protocol-id", default="yolo26")
    p.add_argument("--source-command", action="append", default=[])
    p.add_argument("--publication-run-id", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    protocol = load_eval_protocol(str(args.protocol_id))
    validate_eval_protocol(protocol)
    protocol_hash = eval_protocol_hash(protocol)

    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for report_item in args.report:
        path = _resolve(str(report_item))
        payload = _load_json(path)
        rel = str(path)
        try:
            rel = str(path.relative_to(repo_root))
        except ValueError:
            rel = str(path)
        rows.extend(_extract_rows(payload, source_report=rel))
        meta = payload.get("meta") or {}
        sources.append(
            {
                "report": rel,
                "run_id": str(meta.get("run_id") or "unknown"),
                "git_head": meta.get("git_head"),
                "timestamp": payload.get("timestamp"),
            }
        )

    report = {
        "schema_version": 1,
        "kind": "benchmark_publication_table",
        "timestamp": _now_utc(),
        "publication_run_id": str(args.publication_run_id or _default_run_id()),
        "title": str(args.title),
        "cadence": str(args.cadence),
        "protocol": {
            "id": str(args.protocol_id),
            "hash": protocol_hash,
        },
        "source_commands": list(args.source_command or []),
        "source_reports": sources,
        "rows": rows,
    }

    out_json = _resolve(str(args.output_json))
    out_md = _resolve(str(args.output_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    out_md.write_text(
        _render_markdown(
            title=str(args.title),
            cadence=str(args.cadence),
            protocol_id=str(args.protocol_id),
            protocol_hash=protocol_hash,
            rows=rows,
            source_commands=list(args.source_command or []),
        ),
        encoding="utf-8",
    )

    print(out_json)
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
