#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.training.platform import get_training_backend_spec  # noqa: E402
from yolozu.training.registry import append_training_registry, build_training_registry_entry  # noqa: E402


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _extract_output_path(command: list[str]) -> Path | None:
    for idx, part in enumerate(command):
        if part in {"--output", "-o"} and idx + 1 < len(command):
            return Path(str(command[idx + 1])).resolve()
    return None


def _build_command(exp: dict[str, Any], *, force_dry_run: bool) -> list[str]:
    backend = str(exp.get("backend") or "").strip().lower()
    if backend in {"reference", "reference-rtdetr-pose"}:
        config = str(exp.get("config") or "").strip()
        if not config:
            raise SystemExit("reference experiment requires `config`")
        cmd = [sys.executable, "-m", "yolozu", "train", config]
    else:
        get_training_backend_spec(backend)
        config = str(exp.get("config") or "").strip()
        if not config:
            raise SystemExit(f"{backend} experiment requires `config`")
        dataset = str(exp.get("dataset") or "").strip()
        if not dataset:
            raise SystemExit(f"{backend} experiment requires `dataset`")
        cmd = [sys.executable, "-m", "yolozu", "train", "--external-backend", backend, config, "--dataset", dataset]
        split = str(exp.get("split") or "").strip()
        if split:
            cmd.extend(["--split", split])
    extra_args = [str(x) for x in (exp.get("extra_args") or [])]
    if force_dry_run and "--dry-run" not in extra_args:
        extra_args.append("--dry-run")
    cmd.extend(extra_args)
    return cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchestrate_train.py",
        description="Plan or execute a small multi-backend training batch from one orchestration spec.",
    )
    parser.add_argument("--spec", required=True, help="JSON spec describing experiments[].")
    parser.add_argument(
        "--output",
        default="reports/training_orchestration_report.json",
        help="Output report JSON path (default: reports/training_orchestration_report.json).",
    )
    parser.add_argument("--execute", action="store_true", help="Actually run each planned command.")
    parser.add_argument("--dry-run", action="store_true", help="Append --dry-run to planned commands when missing.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop the batch after the first failing execution.")
    parser.add_argument("--registry-out", default=None, help="Optional JSONL registry file to append executed training runs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec_path = Path(str(args.spec)).resolve()
    if not spec_path.exists():
        raise SystemExit(f"--spec not found: {spec_path}")
    spec = _load_json(spec_path)
    experiments = list(spec.get("experiments") or [])
    if not experiments:
        raise SystemExit("orchestration spec must contain a non-empty `experiments` list")

    results: list[dict[str, Any]] = []
    ok = True
    for idx, exp in enumerate(experiments):
        if not isinstance(exp, dict):
            raise SystemExit(f"experiment at index {idx} must be an object")
        name = str(exp.get("name") or f"exp{idx:02d}")
        backend = str(exp.get("backend") or "").strip().lower()
        command = _build_command(exp, force_dry_run=bool(args.dry_run))
        row: dict[str, Any] = {
            "name": name,
            "backend": backend,
            "command": command,
            "command_str": subprocess.list2cmdline(command),
            "planned_only": not bool(args.execute),
        }
        if args.execute:
            proc = subprocess.run(
                command,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            row["returncode"] = int(proc.returncode)
            row["ok"] = proc.returncode == 0
            row["stdout_tail"] = str(proc.stdout or "").splitlines()[-20:]
            row["stderr_tail"] = str(proc.stderr or "").splitlines()[-20:]
            if proc.returncode != 0:
                ok = False
                if args.stop_on_failure:
                    results.append(row)
                    break
            output_path = _extract_output_path(command)
            if output_path is not None and output_path.exists():
                try:
                    payload = _load_json(output_path)
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    row["summary_json"] = str(output_path)
                    if payload.get("work_dir") is not None:
                        row["work_dir"] = str(payload.get("work_dir"))
                    if isinstance(payload.get("next_steps"), list):
                        row["next_steps"] = payload.get("next_steps")
                    registry_out = getattr(args, "registry_out", None)
                    if registry_out:
                        registry_entry = build_training_registry_entry(
                            summary=payload,
                            summary_path=output_path,
                            orchestration={
                                "spec": str(spec_path),
                                "experiment": str(name),
                                "batch_report": str(Path(str(args.output)).resolve()),
                            },
                        )
                        append_training_registry(Path(str(registry_out)).resolve(), registry_entry)
                        row["registry_out"] = str(Path(str(registry_out)).resolve())
        else:
            row["ok"] = True
        results.append(row)

    report = {
        "format": "yolozu_training_orchestration_report_v1",
        "schema_version": 1,
        "timestamp": _now_utc(),
        "spec": str(spec_path),
        "execute": bool(args.execute),
        "forced_dry_run": bool(args.dry_run),
        "registry_out": (str(Path(str(args.registry_out)).resolve()) if getattr(args, "registry_out", None) else None),
        "ok": ok,
        "counts": {
            "experiments": len(results),
            "executed": sum(1 for row in results if not bool(row.get("planned_only"))),
        },
        "results": results,
    }
    out_path = Path(str(args.output)).resolve()
    _write_json(out_path, report)
    print(str(out_path))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
