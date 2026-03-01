#!/usr/bin/env python3
"""Run external framework finetune smoke checks for YOLOZU.

This tool prepares a framework matrix for:
- YOLOv (Ultralytics)
- MMDetection
- Detectron2
- RT-DETR (rtdetr_pose)

Default behavior is dry-run for all frameworks (command synthesis + config
presence checks). Use --non-dry-framework to execute selected frameworks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.core.config import simple_yaml_load

FRAMEWORKS = ("yolov", "mmdetection", "detectron2", "rtdetr")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run external finetune smoke matrix for YOLOv/MMDetection/Detectron2/RT-DETR.")
    p.add_argument("--dataset-root", default="data/smoke", help="Dataset root (YOLO-style for YOLOv/RT-DETR).")
    p.add_argument("--split", default="train", help="Dataset split for smoke runs (default: train).")
    p.add_argument("--output", default="reports/external_finetune_smoke.json", help="Output report JSON path.")
    p.add_argument(
        "--work-dir",
        default=None,
        help="Optional work directory for generated helper artifacts (default: <output dir>/external_finetune_smoke).",
    )
    p.add_argument("--python", default=sys.executable, help="Python executable for subprocess calls.")
    p.add_argument("--epochs", type=int, default=1, help="Epoch override for smoke runs (default: 1).")
    p.add_argument("--max-steps", type=int, default=1, help="Max steps override for RT-DETR smoke run (default: 1).")
    p.add_argument("--batch-size", type=int, default=2, help="Batch size override (default: 2).")
    p.add_argument("--image-size", type=int, default=96, help="Image size override (default: 96).")
    p.add_argument("--device", default="cpu", help="Device string for smoke training commands (default: cpu).")
    p.add_argument(
        "--framework",
        action="append",
        default=[],
        choices=FRAMEWORKS,
        help="Framework to include (repeatable). Default: all frameworks.",
    )
    p.add_argument(
        "--non-dry-framework",
        action="append",
        default=[],
        choices=FRAMEWORKS,
        help="Framework to execute (repeatable). Others remain dry-run.",
    )
    p.add_argument("--require-non-dry", action="store_true", help="Fail if no framework is configured for non-dry execution.")
    p.add_argument(
        "--require-training-execution",
        action="store_true",
        help="Fail if no framework actually executes a training command.",
    )
    p.add_argument(
        "--mmdet-train-script",
        default=None,
        help="Optional MMDetection train launcher path (e.g., /path/to/mmdetection/tools/train.py).",
    )
    p.add_argument(
        "--detectron2-train-script",
        default=None,
        help="Optional Detectron2 train launcher path (e.g., /path/to/detectron2/tools/train_net.py).",
    )
    return p


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True, check=False)


def _resolve_path(path_like: str, *, base: Path | None = None) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    root = base if base is not None else Path.cwd()
    return (root / p).resolve()


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        data = simple_yaml_load(text)
        return data if isinstance(data, dict) else {}


def _collect_class_names(dataset_root: Path, split: str) -> list[str]:
    classes_json = dataset_root / "labels" / split / "classes.json"
    if classes_json.exists():
        try:
            payload = json.loads(classes_json.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                out = [str(x) for x in payload]
                if out:
                    return out
            if isinstance(payload, dict):
                pairs: list[tuple[int, str]] = []
                for key, value in payload.items():
                    try:
                        idx = int(key)
                    except Exception:
                        continue
                    pairs.append((idx, str(value)))
                if pairs:
                    return [name for _, name in sorted(pairs, key=lambda x: x[0])]
        except Exception:
            pass

    max_class = -1
    labels_dir = dataset_root / "labels" / split
    for txt in sorted(labels_dir.glob("*.txt")):
        try:
            lines = txt.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cid = int(float(parts[0]))
            except Exception:
                continue
            if cid > max_class:
                max_class = cid

    if max_class < 0:
        return ["class_0"]
    return [f"class_{i}" for i in range(max_class + 1)]


def _write_ultralytics_data_yaml(*, dataset_root: Path, split: str, out_path: Path) -> dict[str, Any]:
    train_rel = f"images/{split}"
    val_rel = "images/val" if (dataset_root / "images" / "val").exists() else train_rel
    names = _collect_class_names(dataset_root, split)
    payload = {
        "path": str(dataset_root),
        "train": train_rel,
        "val": val_rel,
        "names": {int(i): str(name) for i, name in enumerate(names)},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML and avoids optional PyYAML dependency.
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def _build_rtdetr_command(
    *,
    python: str,
    config: Path,
    dataset_root: Path,
    split: str,
    run_dir: Path,
    epochs: int,
    max_steps: int,
    batch_size: int,
    image_size: int,
    device: str,
) -> list[str]:
    return [
        str(python),
        "rtdetr_pose/tools/train_minimal.py",
        "--config",
        str(config),
        "--dataset-root",
        str(dataset_root),
        "--split",
        str(split),
        "--val-split",
        str(split),
        "--epochs",
        str(int(epochs)),
        "--max-steps",
        str(int(max_steps)),
        "--batch-size",
        str(int(batch_size)),
        "--image-size",
        str(int(image_size)),
        "--device",
        str(device),
        "--run-dir",
        str(run_dir),
        "--no-export-onnx",
    ]


def _build_mmdet_command(*, python: str, train_script: Path, config: Path, work_dir: Path) -> list[str]:
    return [str(python), str(train_script), str(config), "--work-dir", str(work_dir)]


def _build_detectron2_command(*, python: str, train_script: Path, config: Path, work_dir: Path) -> list[str]:
    return [
        str(python),
        str(train_script),
        "--config-file",
        str(config),
        "SOLVER.MAX_ITER",
        "50",
        "OUTPUT_DIR",
        str(work_dir),
    ]


def _tail(text: str, n: int = 10) -> list[str]:
    return str(text or "").splitlines()[-n:]


def _execute_ultralytics(
    *,
    config_path: Path,
    data_yaml: Path,
    run_dir: Path,
    epochs: int,
    batch_size: int,
    image_size: int,
    device: str,
) -> tuple[bool, dict[str, Any], str | None]:
    details: dict[str, Any] = {
        "training_executed": False,
        "trainer": "ultralytics",
        "command": [
            "python3",
            "-m",
            "ultralytics",
            "train",
            "model=<template.model>",
            f"data={data_yaml}",
            f"epochs={int(epochs)}",
            f"imgsz={int(image_size)}",
            f"batch={int(batch_size)}",
            f"device={device}",
            f"project={run_dir}",
        ],
        "stdout_tail": [],
        "stderr_tail": [],
    }

    cfg = _load_yaml_or_json(config_path)

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        return False, details, f"ultralytics unavailable: {exc}"

    model_name = str(cfg.get("model") or "yolov8n.yaml")
    workers = int(cfg.get("workers") or 0)
    kwargs = {
        "data": str(data_yaml),
        "epochs": int(epochs),
        "imgsz": int(image_size),
        "batch": int(batch_size),
        "device": str(device),
        "workers": int(workers),
        "project": str(run_dir),
        "name": str(cfg.get("name") or "ultralytics_smoke"),
        "exist_ok": True,
        "save": bool(cfg.get("save", False)),
        "val": bool(cfg.get("val", False)),
        "pretrained": bool(cfg.get("pretrained", False)),
    }

    try:
        model = YOLO(model_name)
        model.train(**kwargs)
    except Exception as exc:
        return False, details, f"ultralytics train failed: {exc}"

    details["training_executed"] = True
    details["kwargs"] = kwargs
    return True, details, None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    dataset_root = _resolve_path(str(args.dataset_root))
    if not dataset_root.exists():
        raise SystemExit(f"dataset root not found: {dataset_root}")

    out_path = _resolve_path(str(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    work_dir = _resolve_path(str(args.work_dir), base=Path.cwd()) if args.work_dir else (out_path.parent / "external_finetune_smoke")
    work_dir.mkdir(parents=True, exist_ok=True)

    split = str(args.split)
    selected = list(dict.fromkeys([str(x) for x in (args.framework or [])]))
    if not selected:
        selected = list(FRAMEWORKS)

    non_dry = {str(x) for x in list(args.non_dry_framework or [])}

    config_map = {
        "yolov": repo_root / "configs/examples/finetune_external/ultralytics_yolov8n_finetune_smoke.yaml",
        "mmdetection": repo_root / "configs/examples/finetune_external/mmdetection_finetune_smoke.py",
        "detectron2": repo_root / "configs/examples/finetune_external/detectron2_finetune_smoke.yaml",
        "rtdetr": repo_root / "configs/examples/finetune_external/rtdetr_pose_finetune_smoke.yaml",
    }

    ultra_data_yaml = work_dir / "ultralytics_data.yaml"
    ultra_data_payload = _write_ultralytics_data_yaml(dataset_root=dataset_root, split=split, out_path=ultra_data_yaml)

    results: list[dict[str, Any]] = []
    warnings: list[str] = []

    mmdet_train_script = _resolve_path(str(args.mmdet_train_script)) if args.mmdet_train_script else None
    detectron2_train_script = _resolve_path(str(args.detectron2_train_script)) if args.detectron2_train_script else None

    for framework in selected:
        cfg_path = config_map[framework]
        row_dir = work_dir / framework
        row_dir.mkdir(parents=True, exist_ok=True)
        dry_run = framework not in non_dry
        row_warnings: list[str] = []
        command: list[str] = []
        aux_commands: list[list[str]] = []
        stdout_tail: list[str] = []
        stderr_tail: list[str] = []
        ok = True
        returncode = 0
        runtime_error: str | None = None
        training_executed = False
        capability_checks: list[str] = []

        if not cfg_path.exists():
            ok = False
            returncode = 2
            runtime_error = f"missing config template: {cfg_path}"
        elif dry_run:
            if framework == "yolov":
                command = [
                    str(args.python),
                    "-m",
                    "ultralytics",
                    "train",
                    "model=yolov8n.yaml",
                    f"data={ultra_data_yaml}",
                    f"epochs={int(args.epochs)}",
                    f"imgsz={int(args.image_size)}",
                    f"batch={int(args.batch_size)}",
                    f"device={args.device}",
                ]
            elif framework == "rtdetr":
                command = _build_rtdetr_command(
                    python=str(args.python),
                    config=cfg_path,
                    dataset_root=dataset_root,
                    split=split,
                    run_dir=row_dir / "run",
                    epochs=int(args.epochs),
                    max_steps=int(args.max_steps),
                    batch_size=int(args.batch_size),
                    image_size=int(args.image_size),
                    device=str(args.device),
                )
            elif framework == "mmdetection":
                command = [
                    str(args.python),
                    "-m",
                    "yolozu",
                    "import",
                    "config",
                    "--from",
                    "mmdet",
                    "--config",
                    str(cfg_path),
                    "--output",
                    str(row_dir / "train_config_import.json"),
                    "--force",
                ]
                if mmdet_train_script is None:
                    row_warnings.append("MMDetection train script is not configured (set --mmdet-train-script for non-dry training).")
            elif framework == "detectron2":
                command = [
                    str(args.python),
                    "-m",
                    "yolozu",
                    "import",
                    "config",
                    "--from",
                    "detectron2",
                    "--config",
                    str(cfg_path),
                    "--output",
                    str(row_dir / "train_config_import.json"),
                    "--force",
                ]
                if detectron2_train_script is None:
                    row_warnings.append(
                        "Detectron2 train script is not configured (set --detectron2-train-script for non-dry training)."
                    )
        else:
            if framework == "yolov":
                ok, extra_details, runtime_error = _execute_ultralytics(
                    config_path=cfg_path,
                    data_yaml=ultra_data_yaml,
                    run_dir=row_dir,
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    image_size=int(args.image_size),
                    device=str(args.device),
                )
                command = [str(x) for x in list(extra_details.get("command") or [])]
                stdout_tail = [str(x) for x in list(extra_details.get("stdout_tail") or [])]
                stderr_tail = [str(x) for x in list(extra_details.get("stderr_tail") or [])]
                training_executed = bool(extra_details.get("training_executed", False))
                returncode = 0 if ok else 1
            elif framework == "rtdetr":
                command = _build_rtdetr_command(
                    python=str(args.python),
                    config=cfg_path,
                    dataset_root=dataset_root,
                    split=split,
                    run_dir=row_dir / "run",
                    epochs=int(args.epochs),
                    max_steps=int(args.max_steps),
                    batch_size=int(args.batch_size),
                    image_size=int(args.image_size),
                    device=str(args.device),
                )
                proc = _run(command, cwd=repo_root)
                returncode = int(proc.returncode)
                stdout_tail = _tail(proc.stdout)
                stderr_tail = _tail(proc.stderr)
                ok = proc.returncode == 0
                training_executed = ok
                if not ok:
                    runtime_error = "rtdetr_pose train command failed"
            elif framework == "mmdetection":
                command = [
                    str(args.python),
                    "-m",
                    "yolozu",
                    "import",
                    "config",
                    "--from",
                    "mmdet",
                    "--config",
                    str(cfg_path),
                    "--output",
                    str(row_dir / "train_config_import.json"),
                    "--force",
                ]
                proc = _run(command, cwd=repo_root)
                returncode = int(proc.returncode)
                stdout_tail = _tail(proc.stdout)
                stderr_tail = _tail(proc.stderr)
                ok = proc.returncode == 0
                capability_checks.append("train_config_projection")
                if not ok:
                    runtime_error = "mmdet config projection failed"
                elif mmdet_train_script is None:
                    row_warnings.append("MMDetection non-dry run executed projection only (set --mmdet-train-script for actual training).")
                else:
                    train_cmd = _build_mmdet_command(
                        python=str(args.python),
                        train_script=mmdet_train_script,
                        config=cfg_path,
                        work_dir=row_dir / "run",
                    )
                    env = dict(os.environ)
                    env["YOLOZU_DATASET_ROOT"] = str(dataset_root)
                    env["YOLOZU_SPLIT"] = split
                    env["YOLOZU_MAX_EPOCHS"] = str(int(args.epochs))
                    env["YOLOZU_BATCH_SIZE"] = str(int(args.batch_size))
                    env["YOLOZU_DEVICE"] = str(args.device)
                    proc2 = _run(train_cmd, cwd=repo_root, env=env)
                    aux_commands.append(train_cmd)
                    returncode = int(proc2.returncode)
                    stdout_tail = _tail(proc2.stdout)
                    stderr_tail = _tail(proc2.stderr)
                    ok = proc2.returncode == 0
                    training_executed = ok
                    if not ok:
                        runtime_error = "mmdetection train command failed"
            elif framework == "detectron2":
                command = [
                    str(args.python),
                    "-m",
                    "yolozu",
                    "import",
                    "config",
                    "--from",
                    "detectron2",
                    "--config",
                    str(cfg_path),
                    "--output",
                    str(row_dir / "train_config_import.json"),
                    "--force",
                ]
                proc = _run(command, cwd=repo_root)
                returncode = int(proc.returncode)
                stdout_tail = _tail(proc.stdout)
                stderr_tail = _tail(proc.stderr)
                ok = proc.returncode == 0
                capability_checks.append("train_config_projection")
                if not ok:
                    runtime_error = "detectron2 config projection failed"
                elif detectron2_train_script is None:
                    row_warnings.append(
                        "Detectron2 non-dry run executed projection only (set --detectron2-train-script for actual training)."
                    )
                else:
                    train_cmd = _build_detectron2_command(
                        python=str(args.python),
                        train_script=detectron2_train_script,
                        config=cfg_path,
                        work_dir=row_dir / "run",
                    )
                    env = dict(os.environ)
                    env["YOLOZU_DATASET_ROOT"] = str(dataset_root)
                    env["YOLOZU_SPLIT"] = split
                    proc2 = _run(train_cmd, cwd=repo_root, env=env)
                    aux_commands.append(train_cmd)
                    returncode = int(proc2.returncode)
                    stdout_tail = _tail(proc2.stdout)
                    stderr_tail = _tail(proc2.stderr)
                    ok = proc2.returncode == 0
                    training_executed = ok
                    if not ok:
                        runtime_error = "detectron2 train command failed"

        results.append(
            {
                "framework": framework,
                "ok": bool(ok),
                "dry_run": bool(dry_run),
                "training_executed": bool(training_executed),
                "returncode": int(returncode),
                "runtime_error": runtime_error,
                "config_template": str(cfg_path),
                "command": command,
                "aux_commands": aux_commands,
                "work_dir": str(row_dir),
                "capability_checks": capability_checks,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "warnings": row_warnings,
            }
        )

    non_dry_count = sum(1 for row in results if not bool(row.get("dry_run", True)))
    training_executed_count = sum(1 for row in results if bool(row.get("training_executed", False)))

    if args.require_non_dry and non_dry_count <= 0:
        warnings.append("--require-non-dry is set but no framework was configured for non-dry execution")

    if args.require_training_execution and training_executed_count <= 0:
        warnings.append("--require-training-execution is set but no framework executed training")

    ok_all = all(bool(row.get("ok", False)) for row in results)
    if args.require_non_dry and non_dry_count <= 0:
        ok_all = False
    if args.require_training_execution and training_executed_count <= 0:
        ok_all = False

    report = {
        "task": "external_finetune_smoke",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ok": bool(ok_all),
        "dataset_root": str(dataset_root),
        "split": split,
        "frameworks": selected,
        "non_dry_frameworks": sorted(non_dry),
        "counts": {
            "frameworks": int(len(results)),
            "non_dry": int(non_dry_count),
            "training_executed": int(training_executed_count),
        },
        "ultralytics_data_yaml": str(ultra_data_yaml),
        "ultralytics_data": ultra_data_payload,
        "warnings": warnings,
        "results": results,
    }

    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(out_path))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
