#!/usr/bin/env python3
"""Audit adapter/export support across external detection backends.

Runs dry-run exporters for:
- YOLOX
- YOLOv8 (Ultralytics)
- Detectron2
- MMDetection

The goal is to validate the interface contract path (CLI + schema-valid
predictions artifact) without requiring heavyweight framework installs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run dry-run backend support audit for YOLOX/YOLOv8/Detectron2/MMDetection.")
    p.add_argument("--dataset-root", default="data/real_multitask_fewshot", help="YOLO-format dataset root.")
    p.add_argument("--split", default="val", help="Dataset split to use (default: val).")
    p.add_argument("--max-images", type=int, default=2, help="Cap images per backend run (default: 2).")
    p.add_argument("--output", default="reports/backend_support_audit.json", help="Audit report JSON output path.")
    p.add_argument(
        "--work-dir",
        default=None,
        help="Optional output directory for per-backend prediction artifacts (default: <output dir>/backend_support_audit).",
    )
    p.add_argument("--python", default=sys.executable, help="Python executable for subprocess calls.")
    p.add_argument("--strict", action="store_true", help="Enable strict schema validation in exporter CLIs.")
    p.add_argument(
        "--non-dry-backend",
        action="append",
        default=[],
        choices=("yolox", "yolov8_ultralytics", "detectron2", "mmdetection"),
        help="Backend to run without --dry-run (repeatable).",
    )
    p.add_argument(
        "--require-non-dry",
        action="store_true",
        help="Fail if no backend is configured for non-dry execution.",
    )
    return p


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)


def _load_predictions_count(path: Path) -> tuple[int, str | None]:
    if not path.exists():
        return 0, "output_missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return 0, f"invalid_json:{exc}"

    if isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
        return int(len(payload["predictions"])), None
    if isinstance(payload, list):
        return int(len(payload)), None
    return 0, "unsupported_payload_shape"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    dataset_root = Path(str(args.dataset_root)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (Path.cwd() / dataset_root).resolve()
    if not dataset_root.exists():
        raise SystemExit(f"dataset root not found: {dataset_root}")

    out_path = Path(str(args.output)).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    work_dir = Path(str(args.work_dir)).expanduser() if args.work_dir else (out_path.parent / "backend_support_audit")
    if not work_dir.is_absolute():
        work_dir = (Path.cwd() / work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    strict_flags = ["--strict"] if bool(args.strict) else []
    non_dry = {str(x) for x in list(args.non_dry_backend or [])}

    matrix: list[dict[str, Any]] = [
        {
            "backend": "yolox",
            "command": [
                str(args.python),
                "tools/export_predictions_yolox.py",
                "--dataset",
                str(dataset_root),
                "--split",
                str(args.split),
                "--max-images",
                str(int(args.max_images)),
                "--output",
                str(work_dir / "predictions_yolox.json"),
                *strict_flags,
            ],
        },
        {
            "backend": "yolov8_ultralytics",
            "command": [
                str(args.python),
                "tools/export_predictions_ultralytics.py",
                "--model",
                "yolov8n.pt",
                "--dataset",
                str(dataset_root),
                "--split",
                str(args.split),
                "--max-images",
                str(int(args.max_images)),
                "--output",
                str(work_dir / "predictions_yolov8.json"),
                *strict_flags,
            ],
        },
        {
            "backend": "detectron2",
            "command": [
                str(args.python),
                "tools/export_predictions_detectron2.py",
                "--dataset",
                str(dataset_root),
                "--split",
                str(args.split),
                "--config",
                "configs/detectron2_stub.yaml",
                "--weights",
                "weights/detectron2_stub.pth",
                "--max-images",
                str(int(args.max_images)),
                "--output",
                str(work_dir / "predictions_detectron2.json"),
                *strict_flags,
            ],
        },
        {
            "backend": "mmdetection",
            "command": [
                str(args.python),
                "tools/export_predictions_mmdet.py",
                "--dataset",
                str(dataset_root),
                "--split",
                str(args.split),
                "--config",
                "configs/mmdet_stub.py",
                "--checkpoint",
                "weights/mmdet_stub.pth",
                "--max-images",
                str(int(args.max_images)),
                "--output",
                str(work_dir / "predictions_mmdet.json"),
                *strict_flags,
            ],
        },
    ]

    results: list[dict[str, Any]] = []
    for row in matrix:
        backend = str(row["backend"])
        cmd = [str(x) for x in row["command"]]
        dry_run = backend not in non_dry
        if dry_run:
            cmd.append("--dry-run")
        proc = _run(cmd, cwd=repo_root)

        out_file = None
        if "--output" in cmd:
            try:
                out_file = Path(cmd[cmd.index("--output") + 1]).resolve()
            except Exception:
                out_file = None

        preds_count = None
        output_error = None
        if out_file is not None:
            preds_count, output_error = _load_predictions_count(out_file)

        results.append(
            {
                "backend": backend,
                "ok": bool(proc.returncode == 0 and output_error is None),
                "returncode": int(proc.returncode),
                "predictions_file": (str(out_file) if out_file is not None else None),
                "predictions_count": preds_count,
                "output_error": output_error,
                "stdout_tail": (proc.stdout or "").splitlines()[-10:],
                "stderr_tail": (proc.stderr or "").splitlines()[-10:],
                "command": cmd,
                "dry_run": bool(dry_run),
            }
        )

    non_dry_count = sum(1 for item in results if not bool(item.get("dry_run", True)))
    warnings: list[str] = []
    if non_dry_count <= 0:
        warnings.append("all backends were executed in dry-run mode")
    ok = all(bool(item.get("ok")) for item in results)
    if bool(args.require_non_dry) and non_dry_count <= 0:
        ok = False
        warnings.append("--require-non-dry is set but no --non-dry-backend was configured")
    report = {
        "task": "backend_support_audit",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ok": bool(ok),
        "dataset_root": str(dataset_root),
        "split": str(args.split),
        "max_images": int(args.max_images),
        "warnings": warnings,
        "non_dry_backends": sorted(non_dry),
        "results": results,
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(str(out_path))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
