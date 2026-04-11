"""Training run registry helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_training_registry_entry(
    *,
    summary: dict[str, Any],
    summary_path: str | Path,
    orchestration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    backend = summary.get("backend") if isinstance(summary, dict) else {}
    backend = backend if isinstance(backend, dict) else {}
    dataset = summary.get("dataset") if isinstance(summary, dict) else {}
    dataset = dataset if isinstance(dataset, dict) else {}
    train_cfg = summary.get("canonical_train_config") if isinstance(summary, dict) else {}
    train_cfg = train_cfg if isinstance(train_cfg, dict) else {}
    steps = summary.get("steps") if isinstance(summary, dict) else {}
    steps = steps if isinstance(steps, dict) else {}
    run_output_contract = summary.get("run_output_contract") if isinstance(summary, dict) else {}
    run_output_contract = run_output_contract if isinstance(run_output_contract, dict) else {}

    entry: dict[str, Any] = {
        "format": "yolozu_training_registry_entry_v1",
        "timestamp": _now_utc(),
        "summary_json": str(Path(summary_path)),
        "backend_id": backend.get("backend_id"),
        "backend_display_name": backend.get("display_name"),
        "maturity": backend.get("maturity"),
        "task": train_cfg.get("task") or summary.get("task_family") or summary.get("task"),
        "dataset_root": dataset.get("root"),
        "split": dataset.get("split"),
        "dry_run": bool(summary.get("dry_run")),
        "ok": bool(summary.get("ok")),
        "training_executed": bool(summary.get("training_executed")),
        "work_dir": summary.get("work_dir"),
        "run_output_contract_kind": run_output_contract.get("kind"),
        "next_steps": summary.get("next_steps") if isinstance(summary.get("next_steps"), list) else [],
        "steps": {
            name: {
                "status": (value.get("status") if isinstance(value, dict) else None),
                "ok": (value.get("ok") if isinstance(value, dict) else None),
                "executed": (value.get("executed") if isinstance(value, dict) else None),
            }
            for name, value in steps.items()
            if isinstance(name, str)
        },
    }
    if orchestration:
        entry["orchestration"] = dict(orchestration)
    return entry


def write_training_registry_entry(path: str | Path, payload: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def append_training_registry(path: str | Path, payload: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
    return out_path
