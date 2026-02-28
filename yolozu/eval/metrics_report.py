"""Structured metrics report builder.

Builds timestamped run reports and writes them as JSON, JSONL, or CSV.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

__all__ = [
    "now_utc_iso",
    "flatten_metrics",
    "build_report",
    "append_jsonl",
    "write_json",
    "write_csv_row",
]
from typing import Any


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def flatten_metrics(data: dict[str, Any], *, prefix: str = "", sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (data or {}).items():
        k = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_metrics(value, prefix=k, sep=sep))
        else:
            out[k] = value
    return out


def build_report(*, losses: dict[str, Any] | None = None, metrics: dict[str, Any] | None = None, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp": now_utc_iso(),
        "losses": losses or {},
        "metrics": metrics or {},
        "meta": meta or {},
    }


def append_jsonl(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def write_csv_row(path: str | Path, obj: dict[str, Any]) -> None:
    """Write a single-row CSV with stable flattened columns."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat = flatten_metrics(obj)
    keys = sorted(flat.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerow({k: flat.get(k) for k in keys})

