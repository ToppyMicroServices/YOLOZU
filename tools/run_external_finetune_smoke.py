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
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.core.config import simple_yaml_load

logger = logging.getLogger(__name__)

FRAMEWORKS = ("yolox", "yolov", "mmdetection", "detectron2", "rtdetr")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run external finetune smoke matrix for YOLOX/Ultralytics/MMDetection/Detectron2/RT-DETR.")
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
    p.add_argument(
        "--yolox-train-script",
        default=None,
        help="Optional YOLOX train launcher path (e.g., /path/to/YOLOX/tools/train.py).",
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
    except ImportError:
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
                    except (TypeError, ValueError) as exc:
                        logger.debug("ignoring non-integer class key %r: %s", key, exc)
                        continue
                    pairs.append((idx, str(value)))
                if pairs:
                    return [name for _, name in sorted(pairs, key=lambda x: x[0])]
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.debug("failed to parse class list %s: %s", classes_json, exc)

    max_class = -1
    labels_dir = dataset_root / "labels" / split
    for txt in sorted(labels_dir.glob("*.txt")):
        try:
            lines = txt.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.debug("failed to read label file %s: %s", txt, exc)
            continue
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cid = int(float(parts[0]))
            except (TypeError, ValueError, OverflowError):
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


def _build_yolox_command(*, python: str, train_script: Path, config: Path, batch_size: int) -> list[str]:
    return [
        str(python),
        str(train_script),
        "-f",
        str(config),
        "-d",
        "1",
        "-b",
        str(int(batch_size)),
    ]


def _tail(text: str, n: int = 10) -> list[str]:
    return str(text or "").splitlines()[-n:]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _git_source() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "tracked_changes_present": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "tracked_changes": status.stdout.splitlines() if status.returncode == 0 else [],
    }


def _runtime_environment(python: str) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "selected_python": str(python),
    }


def _artifact_hashes(root: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        artifacts.append(
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return artifacts


def _probe_python_module(*, python: str, module: str, cwd: Path) -> tuple[bool, str | None]:
    probe_cmd = [str(python), "-c", f"import {module}"]
    try:
        probe = _run(probe_cmd, cwd=cwd)
    except FileNotFoundError as exc:
        return False, f"python executable not found: {python} ({exc})"
    if int(probe.returncode) == 0:
        return True, None
    combined = "\n".join([str(probe.stderr or ""), str(probe.stdout or "")]).strip()
    lines = [line.strip() for line in _tail(combined, n=3) if str(line).strip()]
    detail = " | ".join(lines) if lines else f"import probe failed for module '{module}' (exit={probe.returncode})"
    return False, detail


def _missing_module(detail: str, module: str) -> bool:
    text = str(detail or "").lower()
    return f"no module named '{module.lower()}'" in text


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
        "yolox": repo_root / "configs/examples/finetune_external/yolox_s_finetune_smoke.py",
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
    yolox_train_script = _resolve_path(str(args.yolox_train_script)) if args.yolox_train_script else None

    for framework in selected:
        row_started = time.perf_counter()
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
        failure_code: str | None = None
        training_executed = False
        projection_executed = False
        projection_error: str | None = None
        train_path_audited = False
        train_script_configured: bool | None = None
        capability_checks: list[str] = []
        dependency_status: dict[str, Any] = {}

        if not cfg_path.exists():
            ok = False
            returncode = 2
            runtime_error = f"missing config template: {cfg_path}"
            failure_code = "E_CONFIG_TEMPLATE_MISSING"
        elif dry_run:
            if framework == "yolox":
                train_script_configured = yolox_train_script is not None
                command = [
                    str(args.python),
                    "-m",
                    "yolozu",
                    "import",
                    "config",
                    "--from",
                    "yolox",
                    "--config",
                    str(cfg_path),
                    "--output",
                    str(row_dir / "train_config_import.json"),
                    "--force",
                ]
                if yolox_train_script is None:
                    row_warnings.append("YOLOX train script is not configured (set --yolox-train-script for non-dry training).")
            elif framework == "yolov":
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
                train_script_configured = mmdet_train_script is not None
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
                train_script_configured = detectron2_train_script is not None
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
            if framework == "yolox":
                train_script_configured = yolox_train_script is not None
                command = [
                    str(args.python),
                    "-m",
                    "yolozu",
                    "import",
                    "config",
                    "--from",
                    "yolox",
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
                projection_ok = proc.returncode == 0
                capability_checks.append("train_config_projection")
                projection_executed = projection_ok
                if not projection_ok:
                    projection_error = "yolox config projection failed"
                if yolox_train_script is None:
                    ok = False
                    runtime_error = (
                        "YOLOX non-dry execution requires --yolox-train-script; "
                        "config projection is not training."
                    )
                    failure_code = "E_EXTERNAL_TRAIN_SCRIPT_REQUIRED"
                    row_warnings.append("YOLOX config projection completed, but no training command was executed.")
                else:
                    if not projection_ok:
                        row_warnings.append(
                            "YOLOX config projection failed (missing optional deps?) but train-path audit continues via --yolox-train-script."
                        )
                    train_path_audited = True
                    if not yolox_train_script.exists():
                        ok = False
                        returncode = 2
                        runtime_error = f"yolox train script not found: {yolox_train_script}"
                        failure_code = "E_EXTERNAL_SCRIPT_NOT_FOUND"
                        row_warnings.append("YOLOX train path audit failed: script path is missing.")
                    else:
                        train_cmd = _build_yolox_command(
                            python=str(args.python),
                            train_script=yolox_train_script,
                            config=cfg_path,
                            batch_size=int(args.batch_size),
                        )
                        env = dict(os.environ)
                        env["YOLOZU_DATASET_ROOT"] = str(dataset_root)
                        env["YOLOZU_SPLIT"] = split
                        env["YOLOZU_BATCH_SIZE"] = str(int(args.batch_size))
                        env["YOLOZU_MAX_EPOCHS"] = str(int(args.epochs))
                        env["YOLOZU_IMAGE_SIZE"] = str(int(args.image_size))
                        proc2 = _run(train_cmd, cwd=repo_root, env=env)
                        aux_commands.append(train_cmd)
                        returncode = int(proc2.returncode)
                        stdout_tail = _tail(proc2.stdout)
                        stderr_tail = _tail(proc2.stderr)
                        ok = proc2.returncode == 0
                        runtime_error = None if ok else runtime_error
                        failure_code = None if ok else failure_code
                        training_executed = ok
                        if not ok:
                            runtime_error = "yolox train command failed"
                            failure_code = "E_YOLOX_TRAIN_FAILED"
            elif framework == "yolov":
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
                if not ok:
                    err_text = str(runtime_error or "")
                    if "ultralytics unavailable" in err_text:
                        failure_code = "E_DEP_ULTRALYTICS_MISSING"
                    else:
                        failure_code = "E_ULTRALYTICS_TRAIN_FAILED"
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
                capability_checks.append("torch_dependency_probe")
                has_torch, torch_probe_error = _probe_python_module(python=str(args.python), module="torch", cwd=repo_root)
                if not has_torch:
                    ok = False
                    returncode = 3
                    failure_code = "E_DEP_TORCH_MISSING"
                    runtime_error = (
                        "rtdetr non-dry execution requires torch in the selected runtime. "
                        f"Probe detail: {torch_probe_error}"
                    )
                    row_warnings.append(
                        "Install torch for the selected runtime before non-dry rtdetr runs "
                        "(e.g. python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu)."
                    )
                else:
                    proc = _run(command, cwd=repo_root)
                    returncode = int(proc.returncode)
                    stdout_tail = _tail(proc.stdout)
                    stderr_tail = _tail(proc.stderr)
                    ok = proc.returncode == 0
                    training_executed = ok
                    if not ok:
                        combined = "\n".join([str(proc.stderr or ""), str(proc.stdout or "")])
                        if _missing_module(combined, "torch"):
                            failure_code = "E_DEP_TORCH_MISSING"
                            runtime_error = "rtdetr_pose train command failed: torch is not available in runtime"
                        else:
                            failure_code = "E_RTDETR_TRAIN_FAILED"
                            runtime_error = "rtdetr_pose train command failed"
            elif framework == "mmdetection":
                train_script_configured = mmdet_train_script is not None
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
                projection_ok = proc.returncode == 0
                capability_checks.append("train_config_projection")
                projection_executed = projection_ok
                if not projection_ok:
                    projection_error = "mmdet config projection failed"
                    has_mmengine, mmengine_error = _probe_python_module(
                        python=str(args.python), module="mmengine", cwd=repo_root
                    )
                    has_mmdet, mmdet_error = _probe_python_module(
                        python=str(args.python), module="mmdet", cwd=repo_root
                    )
                    dependency_status = {
                        "mmengine": {"available": has_mmengine, "error": mmengine_error},
                        "mmdet": {"available": has_mmdet, "error": mmdet_error},
                    }
                if mmdet_train_script is None:
                    ok = False
                    missing = sorted(name for name, state in dependency_status.items() if not state["available"])
                    runtime_error = (
                        "MMDetection non-dry execution requires --mmdet-train-script; "
                        "config projection is not training."
                    )
                    failure_code = "E_EXTERNAL_TRAIN_SCRIPT_REQUIRED"
                    if missing:
                        runtime_error += " Missing projection dependencies: " + ", ".join(missing)
                        row_warnings.append("MMDetection projection dependencies are unavailable: " + ", ".join(missing))
                else:
                    if not projection_ok:
                        row_warnings.append(
                            "MMDetection config projection failed (missing optional deps?) but train-path audit continues via --mmdet-train-script."
                        )
                    train_path_audited = True
                    if not mmdet_train_script.exists():
                        ok = False
                        returncode = 2
                        runtime_error = f"mmdetection train script not found: {mmdet_train_script}"
                        failure_code = "E_EXTERNAL_SCRIPT_NOT_FOUND"
                        row_warnings.append("MMDetection train path audit failed: script path is missing.")
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
                        runtime_error = None if ok else runtime_error
                        failure_code = None if ok else failure_code
                        training_executed = ok
                        if not ok:
                            runtime_error = "mmdetection train command failed"
                            failure_code = "E_MMDET_TRAIN_FAILED"
            elif framework == "detectron2":
                train_script_configured = detectron2_train_script is not None
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
                projection_ok = proc.returncode == 0
                capability_checks.append("train_config_projection")
                projection_executed = projection_ok
                if not projection_ok:
                    projection_error = "detectron2 config projection failed"
                    has_detectron2, detectron2_error = _probe_python_module(
                        python=str(args.python), module="detectron2", cwd=repo_root
                    )
                    dependency_status = {
                        "detectron2": {"available": has_detectron2, "error": detectron2_error}
                    }
                if detectron2_train_script is None:
                    ok = False
                    runtime_error = (
                        "Detectron2 non-dry execution requires --detectron2-train-script; "
                        "config projection is not training."
                    )
                    failure_code = "E_EXTERNAL_TRAIN_SCRIPT_REQUIRED"
                    if dependency_status and not bool(
                        (dependency_status.get("detectron2") or {}).get("available")
                    ):
                        runtime_error += " Missing projection dependency: detectron2"
                        row_warnings.append("Detectron2 projection dependency is unavailable.")
                else:
                    if not projection_ok:
                        row_warnings.append(
                            "Detectron2 config projection failed (missing optional deps?) but train-path audit continues via --detectron2-train-script."
                        )
                    train_path_audited = True
                    if not detectron2_train_script.exists():
                        ok = False
                        returncode = 2
                        runtime_error = f"detectron2 train script not found: {detectron2_train_script}"
                        failure_code = "E_EXTERNAL_SCRIPT_NOT_FOUND"
                        row_warnings.append("Detectron2 train path audit failed: script path is missing.")
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
                        runtime_error = None if ok else runtime_error
                        failure_code = None if ok else failure_code
                        training_executed = ok
                        if not ok:
                            runtime_error = "detectron2 train command failed"
                            failure_code = "E_DETECTRON2_TRAIN_FAILED"

        results.append(
            {
                "framework": framework,
                "ok": bool(ok),
                "dry_run": bool(dry_run),
                "training_executed": bool(training_executed),
                "projection_executed": bool(projection_executed),
                "projection_error": projection_error,
                "train_path_audited": bool(train_path_audited),
                "train_script_configured": train_script_configured,
                "returncode": int(returncode),
                "failure_code": failure_code,
                "runtime_error": runtime_error,
                "config_template": str(cfg_path),
                "command": command,
                "aux_commands": aux_commands,
                "work_dir": str(row_dir),
                "capability_checks": capability_checks,
                "dependency_status": dependency_status,
                "wall_seconds": float(time.perf_counter() - row_started),
                "config_sha256": _sha256_file(cfg_path) if cfg_path.is_file() else None,
                "artifacts": _artifact_hashes(row_dir),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "warnings": row_warnings,
            }
        )

    non_dry_count = sum(1 for row in results if not bool(row.get("dry_run", True)))
    training_executed_count = sum(1 for row in results if bool(row.get("training_executed", False)))
    projection_executed_count = sum(1 for row in results if bool(row.get("projection_executed", False)))
    train_path_audited_count = sum(1 for row in results if bool(row.get("train_path_audited", False)))

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
        "schema_version": 2,
        "kind": "external_finetune_execution_matrix",
        "task": "external_finetune_smoke",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": _git_source(),
        "environment": _runtime_environment(str(args.python)),
        "ok": bool(ok_all),
        "dataset_root": str(dataset_root),
        "dataset_tree_sha256": _tree_sha256(dataset_root),
        "split": split,
        "frameworks": selected,
        "non_dry_frameworks": sorted(non_dry),
        "counts": {
            "frameworks": int(len(results)),
            "non_dry": int(non_dry_count),
            "training_executed": int(training_executed_count),
            "projection_executed": int(projection_executed_count),
            "train_path_audited": int(train_path_audited_count),
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
