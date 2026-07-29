#!/usr/bin/env python3
"""Qualify repository-local real-image and external fine-tuning lanes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


repo_root = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the repository-local real-image multitask stages, attempt every "
            "external smoke-matrix lane non-dry, probe the wider advertised "
            "runtime surface, and emit one machine-readable hold decision."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Fresh qualification directory. Existing paths are refused.",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/real_multitask_fewshot",
        help="Repository-local prepared dataset root.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python used by training subprocesses.")
    parser.add_argument("--device", default="cpu", help="Training device (default: cpu).")
    parser.add_argument("--epochs", type=int, default=1, help="Base epochs per staged task (default: 1).")
    parser.add_argument("--max-steps", type=int, default=4, help="Max steps per epoch (default: 4).")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2).")
    parser.add_argument("--image-size", type=int, default=128, help="Image size (default: 128).")
    return parser


def _confined_repo_path(value: str, *, kind: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = candidate.resolve()
    if resolved == repo_root or repo_root not in resolved.parents:
        raise SystemExit(f"{kind} must resolve inside the repository: {resolved}")
    return resolved


def _fresh_output(value: str) -> Path:
    output = Path(value).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True)
    return output


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def _git_source() -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "tracked_changes_present": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "tracked_changes": status.stdout.splitlines() if status.returncode == 0 else [],
    }


def _probe_module(python: str, module: str) -> dict[str, Any]:
    proc = _run([python, "-c", f"import {module}"])
    detail = "\n".join([proc.stderr or "", proc.stdout or ""]).strip()
    return {
        "available": bool(proc.returncode == 0),
        "returncode": int(proc.returncode),
        "detail": detail.splitlines()[-1] if detail else None,
    }


def _runtime_probes(python: str) -> dict[str, Any]:
    modules = {
        "torch": "torch",
        "ultralytics": "ultralytics",
        "yolox": "yolox",
        "mmengine": "mmengine",
        "mmdetection": "mmdet",
        "mmpose": "mmpose",
        "mmseg": "mmseg",
        "detectron2": "detectron2",
        "transformers": "transformers",
    }
    probes = {name: _probe_module(python, module) for name, module in modules.items()}
    probes["tao_cli"] = {
        "available": bool(shutil.which("tao")),
        "path": shutil.which("tao"),
    }
    return probes


def _provider_attempts(
    *,
    output_dir: Path,
    dataset_root: Path,
    python: str,
) -> list[dict[str, Any]]:
    providers = [
        {
            "provider": "mmpose",
            "command": "train-mmpose",
            "config_args": [
                "--config",
                "configs/examples/finetune_external/mmpose_finetune_smoke.py",
            ],
        },
        {
            "provider": "mmseg",
            "command": "train-mmseg",
            "config_args": [
                "--config",
                "configs/examples/finetune_external/mmseg_finetune_smoke.py",
            ],
        },
        {
            "provider": "tao",
            "command": "train-tao",
            "config_args": [
                "--config",
                "configs/examples/finetune_external/tao_finetune_smoke.yaml",
                "--task-family",
                "bbox",
            ],
        },
        {
            "provider": "hf_detr",
            "command": "train-hf-detr",
            "config_args": [
                "--model-id",
                "facebook/detr-resnet-50",
                "--epochs",
                "1",
                "--max-steps",
                "4",
            ],
        },
    ]
    attempts: list[dict[str, Any]] = []
    for provider_spec in providers:
        provider = str(provider_spec["provider"])
        command_name = str(provider_spec["command"])
        provider_dir = output_dir / "advertised_providers" / provider
        report_path = provider_dir / "training_summary.json"
        cmd = [
            python,
            "tools/support_external_training.py",
            command_name,
            "--preset",
            "none",
            *[str(value) for value in provider_spec["config_args"]],
            "--dataset",
            str(dataset_root),
            "--split",
            "train",
            "--work-dir",
            str(provider_dir / "work"),
            "--output",
            str(report_path),
        ]
        started = time.perf_counter()
        proc = _run(cmd)
        report = _load_json(report_path) if report_path.is_file() else None
        execution_status = (report or {}).get("execution_status")
        execution_status = execution_status if isinstance(execution_status, dict) else {}
        failure_code = (report or {}).get("failure_code")
        if not failure_code and execution_status.get("state") == "requires_external_train_script":
            failure_code = "E_EXTERNAL_TRAIN_SCRIPT_REQUIRED"
        elif not failure_code and execution_status.get("state") == "runtime_failed":
            failure_code = "E_EXTERNAL_RUNTIME_FAILED"
        attempts.append(
            {
                "provider": provider,
                "command": cmd,
                "returncode": int(proc.returncode),
                "wall_seconds": float(time.perf_counter() - started),
                "report": str(report_path) if report_path.is_file() else None,
                "report_sha256": _sha256_file(report_path) if report_path.is_file() else None,
                "execution_status": execution_status or None,
                "failure_code": failure_code,
                "runtime_error": (report or {}).get("runtime_error"),
                "stderr_tail": (proc.stderr or "").splitlines()[-10:],
            }
        )
    return attempts


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    real = summary["real_multitask"]
    external = summary["external_matrix"]
    lines = [
        "# Fine-Tuning Lane Qualification",
        "",
        f"- source commit: `{summary['source'].get('commit')}`",
        f"- decision: **{summary['decision']['status']}**",
        f"- maturity: **{summary['decision']['maturity']}**",
        f"- protocol complete: **{summary['protocol_complete']}**",
        f"- real stages executed: {real['tasks_ok']}/{real['tasks_total']}",
        f"- external training executed: {external['training_executed']}/{external['frameworks']}",
        "",
        "## Real-image stages",
        "",
        "| Stage | Training | Supervision | Validation scope | After |",
        "|---|---|---|---|---:|",
    ]
    for task in real["tasks"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(task.get("task")),
                    "ok" if task.get("ok") else "failed",
                    str((task.get("supervision") or {}).get("source")),
                    "task-native" if (task.get("evaluation") or {}).get("task_native") else "bbox-only",
                    str((task.get("evaluation") or {}).get("after")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## External matrix",
            "",
            "| Framework | Non-dry | Training executed | Failure code |",
            "|---|---:|---:|---|",
        ]
    )
    for row in external["results"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("framework")),
                    "yes" if not row.get("dry_run") else "no",
                    "yes" if row.get("training_executed") else "no",
                    str(row.get("failure_code") or ""),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Decision reasons", ""])
    lines.extend(f"- {reason}" for reason in summary["decision"]["reasons"])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_checksums(output_dir: Path) -> Path:
    path = output_dir / "checksums.sha256"
    lines: list[str] = []
    for candidate in sorted(p for p in output_dir.rglob("*") if p.is_file() and p != path):
        lines.append(f"{_sha256_file(candidate)}  {candidate.relative_to(output_dir).as_posix()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset_root = _confined_repo_path(str(args.dataset_root), kind="dataset root")
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root not found: {dataset_root}")
    output_dir = _fresh_output(str(args.output_dir))

    real_dir = output_dir / "real_multitask"
    real_cmd = [
        str(args.python),
        "tools/run_real_multitask_finetune_demo.py",
        "--dataset-root",
        str(dataset_root),
        "--out",
        str(real_dir),
        "--python",
        str(args.python),
        "--device",
        str(args.device),
        "--epochs",
        str(int(args.epochs)),
        "--max-steps",
        str(int(args.max_steps)),
        "--batch-size",
        str(int(args.batch_size)),
        "--image-size",
        str(int(args.image_size)),
        "--strict-provenance",
    ]
    real_started = time.perf_counter()
    real_proc = _run(real_cmd)
    real_wall = time.perf_counter() - real_started
    real_report_path = real_dir / "multitask_finetune_demo_report.json"
    if not real_report_path.is_file():
        raise SystemExit(
            "real multitask runner did not emit its report: "
            + " | ".join((real_proc.stderr or "").splitlines()[-5:])
        )
    real_report = _load_json(real_report_path)

    external_report_path = output_dir / "external_matrix.json"
    external_dir = output_dir / "external_matrix"
    external_cmd = [
        str(args.python),
        "tools/run_external_finetune_smoke.py",
        "--dataset-root",
        str(dataset_root),
        "--split",
        "train",
        "--output",
        str(external_report_path),
        "--work-dir",
        str(external_dir),
        "--python",
        str(args.python),
        "--epochs",
        str(int(args.epochs)),
        "--max-steps",
        str(int(args.max_steps)),
        "--batch-size",
        str(int(args.batch_size)),
        "--image-size",
        str(int(args.image_size)),
        "--device",
        str(args.device),
        "--non-dry-framework",
        "yolox",
        "--non-dry-framework",
        "yolov",
        "--non-dry-framework",
        "mmdetection",
        "--non-dry-framework",
        "detectron2",
        "--non-dry-framework",
        "rtdetr",
        "--require-non-dry",
        "--require-training-execution",
    ]
    external_started = time.perf_counter()
    external_proc = _run(external_cmd)
    external_wall = time.perf_counter() - external_started
    if not external_report_path.is_file():
        raise SystemExit(
            "external matrix did not emit its report: "
            + " | ".join((external_proc.stderr or "").splitlines()[-5:])
        )
    external_report = _load_json(external_report_path)

    provider_attempts = _provider_attempts(
        output_dir=output_dir,
        dataset_root=dataset_root,
        python=str(args.python),
    )
    source = _git_source()
    real_tasks = list(real_report.get("tasks") or [])
    external_results = list(external_report.get("results") or [])
    reasons = [
        "task-native before/after metrics are absent for segmentation, keypoints, depth, and pose6d",
        "keypoints, depth, and pose6d supervision in the tracked fixture is heuristic",
        "only the in-repository RT-DETR lane executed training in this environment",
        "external framework lanes without a configured runtime or launcher remain unqualified",
        "the two-image validation split is an execution fixture rather than efficacy evidence",
    ]
    if bool(source.get("tracked_changes_present")):
        reasons.append("the qualification was run with tracked source changes present")

    protocol_complete = bool(
        real_proc.returncode == 0
        and len(real_tasks) == 5
        and all(bool(task.get("ok")) for task in real_tasks)
        and len(external_results) == 5
        and len(provider_attempts) == 4
        and all(bool(attempt.get("report")) for attempt in provider_attempts)
    )
    summary = {
        "schema_version": 1,
        "kind": "finetune_lane_qualification",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_complete": protocol_complete,
        "source": source,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "selected_python": str(args.python),
            "runtime_probes": _runtime_probes(str(args.python)),
        },
        "inputs": {
            "dataset_root": str(dataset_root.relative_to(repo_root)),
            "dataset_tree_sha256": (real_report.get("dataset") or {}).get("tree_sha256"),
            "prepare_summary_sha256": (real_report.get("dataset") or {}).get("prepare_summary_sha256"),
            "label_provenance": (real_report.get("prepare_summary") or {}).get("label_provenance"),
        },
        "protocol": {
            "device": str(args.device),
            "epochs": int(args.epochs),
            "max_steps": int(args.max_steps),
            "batch_size": int(args.batch_size),
            "image_size": int(args.image_size),
            "strict_provenance": True,
        },
        "real_multitask": {
            "command": real_cmd,
            "returncode": int(real_proc.returncode),
            "wall_seconds": float(real_wall),
            "report": str(real_report_path.relative_to(output_dir)),
            "report_sha256": _sha256_file(real_report_path),
            "tasks_total": len(real_tasks),
            "tasks_ok": sum(1 for task in real_tasks if bool(task.get("ok"))),
            "tasks": real_tasks,
        },
        "external_matrix": {
            "command": external_cmd,
            "returncode": int(external_proc.returncode),
            "wall_seconds": float(external_wall),
            "report": str(external_report_path.relative_to(output_dir)),
            "report_sha256": _sha256_file(external_report_path),
            "frameworks": len(external_results),
            "training_executed": sum(1 for row in external_results if bool(row.get("training_executed"))),
            "results": external_results,
        },
        "advertised_provider_attempts": provider_attempts,
        "decision": {
            "status": "hold",
            "maturity": "experimental",
            "training_quality": "not_established",
            "reasons": reasons,
        },
    }
    summary_path = output_dir / "qualification_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(summary, output_dir / "qualification_report.md")
    checksums = _write_checksums(output_dir)
    print(str(summary_path))
    print(str(checksums))

    return 0 if protocol_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
