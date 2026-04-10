#!/usr/bin/env python3
"""External training/integration support tool.

Provides three-layer support helpers:
1) trainer/runner,
2) repo integration wrappers,
3) export/deploy (ONNX + optional TensorRT handoff).

The primary Apache-2.0-friendly training lane is YOLOX-style training via an
external YOLOX launcher. Optional bridges for Ultralytics YOLO and HF DETR are
kept explicit so the runtime/license boundary stays visible.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.integrations.min_adapter import (  # noqa: E402
    canonicalize_predictions_file,
    layered_support_matrix,
    resolve_internal_dataset,
    write_ultralytics_data_yaml,
)
from yolozu.datasets.imports import project_detectron2_config, project_yolox_exp  # noqa: E402
from yolozu.core.canonical import TrainConfig  # noqa: E402
from yolozu.training.platform import build_training_run_summary  # noqa: E402


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _external_run_contract_paths(work_dir: Path) -> dict[str, Path]:
    reports_dir = work_dir / "reports"
    configs_dir = work_dir / "configs"
    return {
        "reports_dir": reports_dir,
        "configs_dir": configs_dir,
        "training_summary": reports_dir / "training_summary.json",
        "external_run_meta": reports_dir / "external_run_meta.json",
        "launcher_plan": reports_dir / "launcher_plan.json",
        "execution": reports_dir / "execution.json",
        "train_config_projection": configs_dir / "train_config_projection.json",
    }


def _write_external_run_contract_bundle(
    *,
    backend_id: str,
    work_dir: Path,
    report_path: Path,
    report: dict[str, Any],
    dataset_root: str,
    split: str,
    command: list[str] | None,
    command_template: str | None,
    train_script: str | None = None,
    projection_payload: dict[str, Any] | None = None,
    process_payload: dict[str, Any] | None = None,
    runtime_error: str | None = None,
    extra_env: dict[str, Any] | None = None,
) -> dict[str, str]:
    paths = _external_run_contract_paths(work_dir)

    _write_json(paths["training_summary"], report)
    if report_path.resolve() != paths["training_summary"].resolve():
        _write_json(report_path, report)

    meta = {
        "format": "yolozu_external_training_run_meta_v1",
        "timestamp": _now_utc(),
        "backend_id": str(backend_id),
        "dataset": {"root": str(dataset_root), "split": str(split)},
        "dry_run": bool(report.get("dry_run")),
        "training_executed": bool(report.get("training_executed")),
        "ok": bool(report.get("ok")),
        "work_dir": str(work_dir),
        "report_json": str(report_path),
    }
    _write_json(paths["external_run_meta"], meta)

    launcher_plan = {
        "format": "yolozu_external_training_launcher_plan_v1",
        "backend_id": str(backend_id),
        "timestamp": _now_utc(),
        "train_script": (str(train_script) if train_script else None),
        "command": list(command or []),
        "command_template": (str(command_template) if command_template else None),
        "environment_hints": dict(extra_env or {}),
    }
    _write_json(paths["launcher_plan"], launcher_plan)

    execution = {
        "format": "yolozu_external_training_execution_v1",
        "backend_id": str(backend_id),
        "timestamp": _now_utc(),
        "dry_run": bool(report.get("dry_run")),
        "executed": bool(report.get("training_executed")),
        "ok": bool(report.get("ok")),
        "runtime_error": runtime_error,
        "process": process_payload or None,
    }
    _write_json(paths["execution"], execution)

    if projection_payload is not None:
        _write_json(paths["train_config_projection"], projection_payload)

    return {
        "training_summary": str(paths["training_summary"]),
        "external_run_meta": str(paths["external_run_meta"]),
        "launcher_plan": str(paths["launcher_plan"]),
        "execution": str(paths["execution"]),
        "train_config_projection": str(paths["train_config_projection"]),
    }


def _run(
    cmd: list[str],
    *,
    cwd: Path = repo_root,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


_PRESETS: dict[str, dict[str, str]] = {
    "smoke": {
        "from_format": "internal",
        "dataset": "data/smoke",
        "split": "val",
        "ultralytics_model": "yolo11n.pt",
        "hf_model_id": "facebook/detr-resnet-50",
        "provider": "ultralytics",
        "onnx_output": "models/yolo11n.onnx",
    },
    "coco128": {
        "from_format": "auto",
        "dataset": "data/coco128",
        "split": "val",
        "ultralytics_model": "yolo11n.pt",
        "hf_model_id": "facebook/detr-resnet-50",
        "provider": "ultralytics",
        "onnx_output": "models/yolo11n_coco128.onnx",
    },
}


def _default_preset_name() -> str:
    raw = str(os.environ.get("YOLOZU_CLI_PRESET", "smoke") or "smoke").strip().lower()
    if not raw:
        return "smoke"
    return raw


def _resolve_preset(name: str | None) -> tuple[str, dict[str, str]]:
    key = str(name or _default_preset_name()).strip().lower()
    if key in {"none", "off", "custom"}:
        return "none", {}
    if key not in _PRESETS:
        allowed = ", ".join(sorted(list(_PRESETS.keys()) + ["none"]))
        raise SystemExit(f"unknown --preset '{key}'. expected one of: {allowed}")
    return key, dict(_PRESETS[key])


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return value
            continue
        return value
    return None


def _resolve_value(
    cli_value: Any,
    *,
    env: str | None = None,
    preset: dict[str, str] | None = None,
    preset_key: str | None = None,
    fallback: Any = None,
) -> Any:
    env_value = None
    if env:
        env_value = os.environ.get(env)
    preset_value = None
    if preset and preset_key:
        preset_value = preset.get(preset_key)
    return _first_present(cli_value, env_value, preset_value, fallback)


def _require_nonempty(value: Any, *, message: str) -> str:
    out = _resolve_value(value)
    if out is None:
        raise SystemExit(message)
    s = str(out).strip()
    if not s:
        raise SystemExit(message)
    return s


def _preset_choices() -> tuple[str, ...]:
    return tuple(sorted(list(_PRESETS.keys()) + ["none"]))


def _cmd_layers(args: argparse.Namespace) -> int:
    matrix = layered_support_matrix()
    if bool(args.json):
        print(json.dumps(matrix, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    for layer, desc in (matrix.get("layers") or {}).items():
        print(f"[{layer}] {desc}")
    providers = matrix.get("providers") or {}
    for provider, spec in providers.items():
        print(f"\n{provider}:")
        for layer in ("trainer_runner", "repo_impl", "export_deploy"):
            print(f"  - {layer}: {spec.get(layer)}")
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    preset_name, preset = _resolve_preset(getattr(args, "preset", None))
    source_format = str(
        _resolve_value(
            args.from_format,
            env="YOLOZU_FROM_FORMAT",
            preset=preset,
            preset_key="from_format",
            fallback="auto",
        )
    )
    dataset = _resolve_value(
        args.dataset,
        env="YOLOZU_DATASET",
        preset=preset,
        preset_key="dataset",
    )
    split = _resolve_value(
        args.split,
        env="YOLOZU_SPLIT",
        preset=preset,
        preset_key="split",
    )
    output = _resolve_value(args.output, fallback="runs/support_external_training/dataset")
    report_out = _resolve_value(args.report, fallback="reports/support_external_training.dataset.json")
    resolution = resolve_internal_dataset(
        source_format=source_format,
        dataset=str(dataset) if dataset else None,
        split=str(split) if split else None,
        output=str(output),
        instances_json=str(args.instances_json) if args.instances_json else None,
        images_dir=str(args.images_dir) if args.images_dir else None,
        force=bool(args.force),
    )
    report = {
        "task": "dataset_convert",
        "timestamp": _now_utc(),
        "ok": True,
        "source_format": resolution.source_format,
        "dataset_root": str(resolution.dataset_root),
        "split": resolution.split,
        "preset": preset_name,
        "dataset_wrapper": str(resolution.dataset_wrapper) if resolution.dataset_wrapper else None,
        "notes": list(resolution.notes),
        "layers": {
            "trainer_runner": "dataset conversion pre-step",
            "repo_impl": "resolve_internal_dataset",
            "export_deploy": "compatible with ONNX/TensorRT export wrappers",
        },
    }
    report_path = Path(str(report_out)).resolve()
    _write_json(report_path, report)
    print(str(report_path))
    return 0


def _ultralytics_train_template(
    *,
    model: str,
    data_yaml: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    name: str,
) -> str:
    return (
        "yolo train "
        f"model={model} data={data_yaml} epochs={int(epochs)} imgsz={int(imgsz)} "
        f"batch={int(batch)} device={device} project={project} name={name}"
    )


def _yolox_train_template(
    *,
    python: str,
    train_script: str,
    exp_file: str,
    batch: int,
    weights: str | None,
    devices: int,
) -> str:
    cmd = [
        "YOLOZU_DATASET_ROOT=<dataset_root>",
        "YOLOZU_SPLIT=<split>",
        "YOLOZU_NUM_CLASSES=<num_classes>",
        str(python),
        str(train_script),
        "-f",
        str(exp_file),
        "-d",
        str(int(devices)),
        "-b",
        str(int(batch)),
    ]
    if weights:
        cmd.extend(["-c", str(weights)])
    return " ".join(cmd)


def _build_yolox_train_command(
    *,
    python: str,
    train_script: str,
    exp_file: str,
    batch: int,
    devices: int,
    weights: str | None,
) -> list[str]:
    cmd = [
        str(python),
        str(train_script),
        "-f",
        str(exp_file),
        "-d",
        str(int(devices)),
        "-b",
        str(int(batch)),
    ]
    if weights:
        cmd.extend(["-c", str(weights)])
    return cmd


def _infer_detectron2_task_family(config_path: Path) -> str:
    text = config_path.read_text(encoding="utf-8", errors="ignore")
    lowered = text.lower()
    if "coco-keypoints" in lowered or "keypoint_on: true" in lowered:
        return "keypoints"
    if "instance-segmentation" in lowered or "mask_on: true" in lowered:
        return "segmentation"
    return "bbox"


def _build_detectron2_train_command(
    *,
    python: str,
    train_script: str,
    config: Path,
    work_dir: Path,
    train_opts: list[tuple[str, str]],
) -> list[str]:
    cmd = [
        str(python),
        str(train_script),
        "--config-file",
        str(config),
        "SOLVER.MAX_ITER",
        "50",
        "OUTPUT_DIR",
        str(work_dir),
    ]
    for key, value in train_opts:
        cmd.extend([str(key), str(value)])
    return cmd


def _cmd_train_yolox(args: argparse.Namespace) -> int:
    preset_name, preset = _resolve_preset(getattr(args, "preset", None))
    exp_path = Path(
        _require_nonempty(
            _resolve_value(args.exp),
            message="YOLOX exp file is required (set --exp).",
        )
    ).resolve()
    if not exp_path.exists():
        raise SystemExit(f"--exp not found: {exp_path}")

    dataset_input = _require_nonempty(
        _resolve_value(
            args.dataset,
            env="YOLOZU_DATASET",
            preset=preset,
            preset_key="dataset",
        ),
        message="dataset is required (set --dataset/-d, YOLOZU_DATASET, or --preset).",
    )
    source_format = str(
        _resolve_value(
            args.from_format,
            env="YOLOZU_FROM_FORMAT",
            preset=preset,
            preset_key="from_format",
            fallback="auto",
        )
    )
    split = str(
        _resolve_value(
            args.split,
            env="YOLOZU_SPLIT",
            preset=preset,
            preset_key="split",
            fallback="train",
        )
    )
    work_dir = Path(str(_resolve_value(args.work_dir, fallback="runs/support_external_training/yolox"))).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    resolution = resolve_internal_dataset(
        source_format=source_format,
        dataset=dataset_input,
        split=split,
        output=work_dir / "dataset",
        instances_json=str(args.instances_json) if args.instances_json else None,
        images_dir=str(args.images_dir) if args.images_dir else None,
        force=bool(args.force),
    )

    train_cfg = project_yolox_exp(config=exp_path).to_dict()
    weights_path = Path(str(args.weights)).resolve() if args.weights else None
    train_script = str(args.train_script).strip() if args.train_script else ""
    command = _build_yolox_train_command(
        python=str(args.python),
        train_script=(train_script or "<YOLOX/tools/train.py>"),
        exp_file=str(exp_path),
        batch=int(args.batch),
        devices=int(args.devices),
        weights=(str(weights_path) if weights_path else None),
    )
    template = _yolox_train_template(
        python=str(args.python),
        train_script=(train_script or "<YOLOX/tools/train.py>"),
        exp_file=str(exp_path),
        batch=int(args.batch),
        weights=(str(weights_path) if weights_path else None),
        devices=int(args.devices),
    )

    projection_path = work_dir / "yolox_train_config_projection.json"
    projection_payload = {
        "format": "yolozu_external_training_projection_v1",
        "provider": "yolox",
        "train_config": train_cfg,
        "dataset_resolution": {
            "dataset_root": str(resolution.dataset_root),
            "split": str(resolution.split),
            "source_format": str(resolution.source_format),
        },
        "environment_hints": {
            "YOLOZU_DATASET_ROOT": str(resolution.dataset_root),
            "YOLOZU_SPLIT": str(resolution.split),
        },
    }
    _write_json(projection_path, projection_payload)

    training_executed = False
    runtime_error: str | None = None
    proc_info: dict[str, Any] | None = None

    if not bool(args.dry_run):
        if not train_script:
            runtime_error = (
                "YOLOX non-dry execution requires --train-script pointing to an external "
                "Apache-2.0 YOLOX launcher (for example YOLOX/tools/train.py)."
            )
        elif not Path(train_script).exists():
            runtime_error = f"YOLOX train script not found: {train_script}"
        else:
            env = dict(os.environ)
            env["YOLOZU_DATASET_ROOT"] = str(resolution.dataset_root)
            env["YOLOZU_SPLIT"] = str(resolution.split)
            env["YOLOZU_BATCH_SIZE"] = str(int(args.batch))
            proc = _run(command, cwd=repo_root, env=env)
            proc_info = {
                "returncode": int(proc.returncode),
                "stdout_tail": str(proc.stdout or "").splitlines()[-20:],
                "stderr_tail": str(proc.stderr or "").splitlines()[-20:],
            }
            training_executed = proc.returncode == 0
            if proc.returncode != 0:
                runtime_error = f"external YOLOX train script failed ({proc.returncode})"

    report_path = Path(str(_resolve_value(args.output, fallback="reports/support_external_training.train_yolox.json"))).resolve()
    report = build_training_run_summary(
        backend_id="yolox",
        report_path=report_path,
        train_config=train_cfg,
        dataset_root=str(resolution.dataset_root),
        split=str(resolution.split),
        dry_run=bool(args.dry_run),
        work_dir=str(work_dir),
        steps={
            "train": {
                "status": ("dry_run" if bool(args.dry_run) else ("ok" if training_executed else "failed")),
                "ok": bool(args.dry_run) or training_executed,
                "executed": training_executed,
                "command_template": template,
                "train_script": train_script or None,
            },
            "export": {"status": "planned", "ok": True, "executed": False},
            "eval": {"status": "planned", "ok": True, "executed": False},
            "parity": {"status": "planned", "ok": True, "executed": False},
        },
        process=proc_info,
        runtime_error=runtime_error,
        notes=[
            f"preset={preset_name}",
            f"projection={projection_path}",
        ],
        license_boundary={
            "repo_code": "Apache-2.0",
            "primary_lane": "YOLOX-style external training bridge",
            "optional_bridge": False,
        },
    )
    report.update(
        {
            "task": "train_yolox",
            "preset": preset_name,
            "exp": str(exp_path),
            "weights": str(weights_path) if weights_path else None,
            "train_script": train_script or None,
            "template_train_command": template,
            "train_config_projection": str(projection_path),
            "layers": {
                "trainer_runner": "external YOLOX train launcher",
                "repo_impl": "support_external_training train-yolox",
                "export_deploy": "export_predictions_yolox + eval/benchmark lanes",
            },
        }
    )
    bundled_paths = _write_external_run_contract_bundle(
        backend_id="yolox",
        work_dir=work_dir,
        report_path=report_path,
        report=report,
        dataset_root=str(resolution.dataset_root),
        split=str(resolution.split),
        command=command,
        command_template=template,
        train_script=train_script or None,
        projection_payload=projection_payload,
        process_payload=proc_info,
        runtime_error=runtime_error,
        extra_env=projection_payload.get("environment_hints"),
    )
    report["run_output_contract"]["stable_artifact_paths"] = bundled_paths
    report["train_config_projection"] = bundled_paths["train_config_projection"]
    _write_json(report_path, report)
    _write_json(Path(bundled_paths["training_summary"]), report)
    print(str(report_path))
    return 0 if bool(report.get("ok")) else 1


def _cmd_train_detectron2(args: argparse.Namespace) -> int:
    preset_name, preset = _resolve_preset(getattr(args, "preset", None))
    config_path = Path(
        _require_nonempty(
            _resolve_value(args.config),
            message="Detectron2 config file is required (set --config).",
        )
    ).resolve()
    if not config_path.exists():
        raise SystemExit(f"--config not found: {config_path}")

    dataset_input = _require_nonempty(
        _resolve_value(
            args.dataset,
            env="YOLOZU_DATASET",
            preset=preset,
            preset_key="dataset",
        ),
        message="dataset is required (set --dataset/-d, YOLOZU_DATASET, or --preset).",
    )
    source_format = str(
        _resolve_value(
            args.from_format,
            env="YOLOZU_FROM_FORMAT",
            preset=preset,
            preset_key="from_format",
            fallback="auto",
        )
    )
    split = str(
        _resolve_value(
            args.split,
            env="YOLOZU_SPLIT",
            preset=preset,
            preset_key="split",
            fallback="train",
        )
    )
    work_dir = Path(str(_resolve_value(args.work_dir, fallback="runs/support_external_training/detectron2"))).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    resolution = resolve_internal_dataset(
        source_format=source_format,
        dataset=dataset_input,
        split=split,
        output=work_dir / "dataset",
        instances_json=str(args.instances_json) if args.instances_json else None,
        images_dir=str(args.images_dir) if args.images_dir else None,
        force=bool(args.force),
    )

    task_family = str(getattr(args, "task_family", "auto") or "auto").strip().lower()
    if task_family == "auto":
        task_family = _infer_detectron2_task_family(config_path)

    projection_error: str | None = None
    try:
        train_cfg = project_detectron2_config(config=config_path)
    except Exception as exc:
        projection_error = str(exc)
        train_cfg = TrainConfig(
            backend="detectron2",
            task=task_family,
            model=str(config_path),
            dataset={"root": str(resolution.dataset_root), "split": str(resolution.split)},
            source={"from": "detectron2", "config": str(config_path)},
        )
    else:
        train_cfg = replace(
            train_cfg,
            task=task_family,
            model=str(config_path),
            dataset={"root": str(resolution.dataset_root), "split": str(resolution.split)},
        )

    projection_path = work_dir / "detectron2_train_config_projection.json"
    projection_payload = {
        "format": "yolozu_external_training_projection_v1",
        "provider": "detectron2",
        "projection_error": projection_error,
        "train_config": train_cfg.to_dict(),
        "dataset_resolution": {
            "dataset_root": str(resolution.dataset_root),
            "split": str(resolution.split),
            "source_format": str(resolution.source_format),
        },
        "environment_hints": {
            "YOLOZU_DATASET_ROOT": str(resolution.dataset_root),
            "YOLOZU_SPLIT": str(resolution.split),
            "YOLOZU_TASK_FAMILY": str(task_family),
        },
    }
    _write_json(projection_path, projection_payload)

    train_script = str(args.train_script).strip() if args.train_script else ""
    train_opts = [(str(k), str(v)) for k, v in (getattr(args, "train_opt", None) or [])]
    command = _build_detectron2_train_command(
        python=str(args.python),
        train_script=(train_script or "<detectron2/tools/train_net.py>"),
        config=config_path,
        work_dir=work_dir,
        train_opts=train_opts,
    )
    template = " ".join(command)

    training_executed = False
    runtime_error: str | None = None
    proc_info: dict[str, Any] | None = None

    if not bool(args.dry_run):
        if not train_script:
            runtime_error = (
                "Detectron2 non-dry execution requires --train-script pointing to an external "
                "detectron2/tools/train_net.py style launcher."
            )
        elif not Path(train_script).exists():
            runtime_error = f"Detectron2 train script not found: {train_script}"
        else:
            env = dict(os.environ)
            env["YOLOZU_DATASET_ROOT"] = str(resolution.dataset_root)
            env["YOLOZU_SPLIT"] = str(resolution.split)
            env["YOLOZU_TASK_FAMILY"] = str(task_family)
            proc = _run(command, cwd=repo_root, env=env)
            proc_info = {
                "returncode": int(proc.returncode),
                "stdout_tail": str(proc.stdout or "").splitlines()[-20:],
                "stderr_tail": str(proc.stderr or "").splitlines()[-20:],
            }
            training_executed = proc.returncode == 0
            if proc.returncode != 0:
                runtime_error = f"external Detectron2 train script failed ({proc.returncode})"

    report_path = Path(str(_resolve_value(args.output, fallback="reports/support_external_training.train_detectron2.json"))).resolve()
    report = build_training_run_summary(
        backend_id="detectron2",
        report_path=report_path,
        train_config=train_cfg,
        dataset_root=str(resolution.dataset_root),
        split=str(resolution.split),
        dry_run=bool(args.dry_run),
        work_dir=str(work_dir),
        steps={
            "train": {
                "status": ("dry_run" if bool(args.dry_run) else ("ok" if training_executed else "failed")),
                "ok": bool(args.dry_run) or training_executed,
                "executed": training_executed,
                "command_template": template,
                "train_script": train_script or None,
            },
            "export": {"status": "planned", "ok": True, "executed": False},
            "eval": {"status": "planned", "ok": True, "executed": False},
            "parity": {"status": "planned", "ok": True, "executed": False},
        },
        process=proc_info,
        runtime_error=runtime_error,
        notes=[
            f"preset={preset_name}",
            f"projection={projection_path}",
            "task is selected by the Detectron2 config (bbox / segmentation / keypoints).",
        ],
        license_boundary={
            "repo_code": "Apache-2.0",
            "optional_bridge": True,
            "note": "Detectron2 runtime remains external to YOLOZU; review upstream terms and environment setup separately.",
        },
    )
    report.update(
        {
            "task": "train_detectron2",
            "task_family": task_family,
            "config": str(config_path),
            "train_script": train_script or None,
            "template_train_command": template,
            "train_config_projection": str(projection_path),
            "projection_error": projection_error,
            "layers": {
                "trainer_runner": "external Detectron2 train launcher",
                "repo_impl": "support_external_training train-detectron2",
                "export_deploy": "export_predictions_detectron2 + eval/benchmark lanes",
            },
        }
    )
    bundled_paths = _write_external_run_contract_bundle(
        backend_id="detectron2",
        work_dir=work_dir,
        report_path=report_path,
        report=report,
        dataset_root=str(resolution.dataset_root),
        split=str(resolution.split),
        command=command,
        command_template=template,
        train_script=train_script or None,
        projection_payload=projection_payload,
        process_payload=proc_info,
        runtime_error=runtime_error,
        extra_env=projection_payload.get("environment_hints"),
    )
    report["run_output_contract"]["stable_artifact_paths"] = bundled_paths
    report["train_config_projection"] = bundled_paths["train_config_projection"]
    _write_json(report_path, report)
    _write_json(Path(bundled_paths["training_summary"]), report)
    print(str(report_path))
    return 0 if bool(report.get("ok")) else 1


def _cmd_train_ultralytics(args: argparse.Namespace) -> int:
    preset_name, preset = _resolve_preset(getattr(args, "preset", None))
    model_name = _require_nonempty(
        _resolve_value(
            args.model,
            env="YOLOZU_MODEL",
            preset=preset,
            preset_key="ultralytics_model",
        ),
        message="model is required (set --model/-m, YOLOZU_MODEL, or --preset).",
    )
    dataset_input = _require_nonempty(
        _resolve_value(
            args.dataset,
            env="YOLOZU_DATASET",
            preset=preset,
            preset_key="dataset",
        ),
        message="dataset is required (set --dataset/-d, YOLOZU_DATASET, or --preset).",
    )
    source_format = str(
        _resolve_value(
            args.from_format,
            env="YOLOZU_FROM_FORMAT",
            preset=preset,
            preset_key="from_format",
            fallback="auto",
        )
    )
    split = str(
        _resolve_value(
            args.split,
            env="YOLOZU_SPLIT",
            preset=preset,
            preset_key="split",
            fallback="train",
        )
    )
    work_dir = Path(str(_resolve_value(args.work_dir, fallback="runs/support_external_training/ultralytics"))).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    resolution = resolve_internal_dataset(
        source_format=source_format,
        dataset=dataset_input,
        split=split,
        output=work_dir / "dataset",
        instances_json=str(args.instances_json) if args.instances_json else None,
        images_dir=str(args.images_dir) if args.images_dir else None,
        force=bool(args.force),
    )

    dataset_path = Path(dataset_input).resolve()
    if dataset_path.is_file() and dataset_path.suffix.lower() in (".yaml", ".yml") and resolution.source_format == "ultralytics":
        data_yaml = dataset_path
    else:
        data_yaml = write_ultralytics_data_yaml(
            dataset_root=resolution.dataset_root,
            split=resolution.split,
            output=work_dir / "ultralytics_data.yaml",
        )

    template = _ultralytics_train_template(
        model=model_name,
        data_yaml=data_yaml,
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        project=str(args.project),
        name=str(args.name),
    )

    training_executed = False
    run_dir: str | None = None
    runtime_error: str | None = None

    if not bool(args.dry_run):
        try:
            from ultralytics import YOLO  # type: ignore

            model = YOLO(model_name)
            result = model.train(
                data=str(data_yaml),
                epochs=int(args.epochs),
                imgsz=int(args.imgsz),
                batch=int(args.batch),
                device=str(args.device),
                project=str(args.project),
                name=str(args.name),
                workers=int(args.workers),
            )
            run_dir = str(getattr(result, "save_dir", "") or "")
            training_executed = True
        except Exception as exc:
            runtime_error = str(exc)

    report_path = Path(str(_resolve_value(args.output, fallback="reports/support_external_training.train_ultralytics.json"))).resolve()
    train_cfg = TrainConfig(
        backend="ultralytics",
        model=str(model_name),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        epochs=int(args.epochs),
        device=str(args.device),
        workers=int(args.workers),
        dataset={"root": str(resolution.dataset_root), "split": str(resolution.split)},
        source={"from": "ultralytics", "model": str(model_name)},
    )
    report = build_training_run_summary(
        backend_id="ultralytics",
        report_path=report_path,
        train_config=train_cfg,
        dataset_root=str(resolution.dataset_root),
        split=str(resolution.split),
        dry_run=bool(args.dry_run),
        work_dir=str(work_dir),
        steps={
            "train": {
                "status": ("dry_run" if bool(args.dry_run) else ("ok" if training_executed else "failed")),
                "ok": bool(args.dry_run) or training_executed,
                "executed": training_executed,
                "command_template": template,
            },
            "export": {"status": "planned", "ok": True, "executed": False},
            "eval": {"status": "planned", "ok": True, "executed": False},
            "parity": {"status": "planned", "ok": True, "executed": False},
        },
        runtime_error=runtime_error,
        notes=[f"preset={preset_name}", f"data_yaml={data_yaml}"],
        license_boundary={
            "repo_code": "Apache-2.0",
            "optional_bridge": True,
            "note": "Ultralytics runtime is optional and must be reviewed under its own license terms.",
        },
    )
    report.update(
        {
            "task": "train_ultralytics",
            "preset": preset_name,
            "model": model_name,
            "data_yaml": str(data_yaml),
            "template_train_command": template,
            "template_predict_normalize_command": (
                "python3 tools/support_external_training.py predict-normalize "
                f"--ultralytics-model {model_name} --dataset {resolution.dataset_root} "
                f"--split {resolution.split} --output reports/ultralytics_predictions.normalized.json "
                "--report reports/ultralytics_predict_normalize_report.json"
            ),
            "run_dir": run_dir,
            "layers": {
                "trainer_runner": "ultralytics.YOLO.train",
                "repo_impl": "support_external_training train-ultralytics",
                "export_deploy": "support_external_training export-onnx --provider ultralytics",
            },
        }
    )
    ultralytics_projection = {
        "format": "yolozu_external_training_projection_v1",
        "provider": "ultralytics",
        "train_config": train_cfg.to_dict(),
        "dataset_resolution": {
            "dataset_root": str(resolution.dataset_root),
            "split": str(resolution.split),
            "source_format": str(resolution.source_format),
        },
        "environment_hints": {
            "YOLOZU_DATASET_ROOT": str(resolution.dataset_root),
            "YOLOZU_SPLIT": str(resolution.split),
        },
        "data_yaml": str(data_yaml),
    }
    bundled_paths = _write_external_run_contract_bundle(
        backend_id="ultralytics",
        work_dir=work_dir,
        report_path=report_path,
        report=report,
        dataset_root=str(resolution.dataset_root),
        split=str(resolution.split),
        command=None,
        command_template=template,
        projection_payload=ultralytics_projection,
        runtime_error=runtime_error,
        extra_env=ultralytics_projection.get("environment_hints"),
    )
    report["run_output_contract"]["stable_artifact_paths"] = bundled_paths
    report["train_config_projection"] = bundled_paths["train_config_projection"]
    _write_json(report_path, report)
    _write_json(Path(bundled_paths["training_summary"]), report)
    print(str(report_path))
    return 0 if bool(report.get("ok")) else 1


def _cmd_train_hf_detr(args: argparse.Namespace) -> int:
    preset_name, preset = _resolve_preset(getattr(args, "preset", None))
    model_id = _require_nonempty(
        _resolve_value(
            args.model_id,
            env="YOLOZU_HF_MODEL_ID",
            preset=preset,
            preset_key="hf_model_id",
            fallback="facebook/detr-resnet-50",
        ),
        message="model id resolution failed for HF DETR entry.",
    )
    dataset_input = _require_nonempty(
        _resolve_value(
            args.dataset,
            env="YOLOZU_DATASET",
            preset=preset,
            preset_key="dataset",
        ),
        message="dataset is required (set --dataset/-d, YOLOZU_DATASET, or --preset).",
    )
    source_format = str(
        _resolve_value(
            args.from_format,
            env="YOLOZU_FROM_FORMAT",
            preset=preset,
            preset_key="from_format",
            fallback="auto",
        )
    )
    split = str(
        _resolve_value(
            args.split,
            env="YOLOZU_SPLIT",
            preset=preset,
            preset_key="split",
            fallback="train",
        )
    )
    work_dir = Path(str(_resolve_value(args.work_dir, fallback="runs/support_external_training/hf_detr"))).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    resolution = resolve_internal_dataset(
        source_format=source_format,
        dataset=dataset_input,
        split=split,
        output=work_dir / "dataset",
        instances_json=str(args.instances_json) if args.instances_json else None,
        images_dir=str(args.images_dir) if args.images_dir else None,
        force=bool(args.force),
    )

    train_script = str(args.train_script).strip() if args.train_script else ""
    if train_script:
        command = [
            str(args.python),
            train_script,
            "--model-id",
            model_id,
            "--dataset",
            str(resolution.dataset_root),
            "--split",
            str(resolution.split),
            "--epochs",
            str(int(args.epochs)),
            "--batch-size",
            str(int(args.batch_size)),
            "--learning-rate",
            str(float(args.learning_rate)),
            "--max-steps",
            str(int(args.max_steps)),
        ]
    else:
        command = [
            "accelerate",
            "launch",
            "<hf_detr_train_script.py>",
            "--model-id",
            model_id,
            "--dataset",
            str(resolution.dataset_root),
            "--split",
            str(resolution.split),
            "--epochs",
            str(int(args.epochs)),
            "--batch-size",
            str(int(args.batch_size)),
            "--learning-rate",
            str(float(args.learning_rate)),
            "--max-steps",
            str(int(args.max_steps)),
        ]

    training_executed = False
    runtime_error: str | None = None
    proc_info: dict[str, Any] | None = None

    if not bool(args.dry_run):
        if train_script:
            proc = _run(command, cwd=repo_root)
            proc_info = {
                "returncode": int(proc.returncode),
                "stdout_tail": str(proc.stdout or "").splitlines()[-20:],
                "stderr_tail": str(proc.stderr or "").splitlines()[-20:],
            }
            training_executed = proc.returncode == 0
            if proc.returncode != 0:
                runtime_error = f"external HF train script failed ({proc.returncode})"
        else:
            runtime_error = (
                "HF DETR entry requires --train-script for non-dry execution. "
                "Use --dry-run to get template command or provide a script."
            )

    report_path = Path(str(_resolve_value(args.output, fallback="reports/support_external_training.train_hf_detr.json"))).resolve()
    train_cfg = TrainConfig(
        backend="hf-detr",
        model=str(model_id),
        batch=int(args.batch_size),
        epochs=int(args.epochs),
        steps=int(args.max_steps),
        lr=float(args.learning_rate),
        dataset={"root": str(resolution.dataset_root), "split": str(resolution.split)},
        source={"from": "hf-detr", "model_id": str(model_id)},
    )
    report = build_training_run_summary(
        backend_id="hf-detr",
        report_path=report_path,
        train_config=train_cfg,
        dataset_root=str(resolution.dataset_root),
        split=str(resolution.split),
        dry_run=bool(args.dry_run),
        work_dir=str(work_dir),
        steps={
            "train": {
                "status": ("dry_run" if bool(args.dry_run) else ("ok" if training_executed else "failed")),
                "ok": bool(args.dry_run) or training_executed,
                "executed": training_executed,
                "command_template": " ".join(command),
                "train_script": train_script or None,
            },
            "export": {"status": "planned", "ok": True, "executed": False},
            "eval": {"status": "planned", "ok": True, "executed": False},
            "parity": {"status": "planned", "ok": True, "executed": False},
        },
        process=proc_info,
        runtime_error=runtime_error,
        notes=[f"preset={preset_name}"],
        license_boundary={
            "repo_code": "Apache-2.0",
            "optional_bridge": True,
        },
    )
    report.update(
        {
            "task": "train_hf_detr",
            "preset": preset_name,
            "model_id": model_id,
            "train_script": train_script or None,
            "template_train_command": " ".join(command),
            "layers": {
                "trainer_runner": "transformers/accelerate entry script",
                "repo_impl": "support_external_training train-hf-detr",
                "export_deploy": "support_external_training export-onnx --provider hf_detr",
            },
        }
    )
    hf_projection = {
        "format": "yolozu_external_training_projection_v1",
        "provider": "hf-detr",
        "train_config": train_cfg.to_dict(),
        "dataset_resolution": {
            "dataset_root": str(resolution.dataset_root),
            "split": str(resolution.split),
            "source_format": str(resolution.source_format),
        },
        "environment_hints": {
            "YOLOZU_DATASET_ROOT": str(resolution.dataset_root),
            "YOLOZU_SPLIT": str(resolution.split),
        },
    }
    bundled_paths = _write_external_run_contract_bundle(
        backend_id="hf-detr",
        work_dir=work_dir,
        report_path=report_path,
        report=report,
        dataset_root=str(resolution.dataset_root),
        split=str(resolution.split),
        command=command,
        command_template=" ".join(command),
        train_script=train_script or None,
        projection_payload=hf_projection,
        process_payload=proc_info,
        runtime_error=runtime_error,
        extra_env=hf_projection.get("environment_hints"),
    )
    report["run_output_contract"]["stable_artifact_paths"] = bundled_paths
    report["train_config_projection"] = bundled_paths["train_config_projection"]
    _write_json(report_path, report)
    _write_json(Path(bundled_paths["training_summary"]), report)
    print(str(report_path))
    return 0 if bool(report.get("ok")) else 1


def _cmd_export_onnx(args: argparse.Namespace) -> int:
    preset_name, preset = _resolve_preset(getattr(args, "preset", None))
    provider = str(
        _resolve_value(
            args.provider,
            env="YOLOZU_PROVIDER",
            preset=preset,
            preset_key="provider",
            fallback="ultralytics",
        )
    )
    default_model_key = "ultralytics_model" if provider == "ultralytics" else "hf_model_id"
    model_name = _require_nonempty(
        _resolve_value(
            args.model,
            env="YOLOZU_MODEL",
            preset=preset,
            preset_key=default_model_key,
        ),
        message="model is required for export (set --model/-m, YOLOZU_MODEL, or --preset).",
    )
    output_path = _resolve_value(
        args.output,
        env="YOLOZU_ONNX_OUTPUT",
        preset=preset,
        preset_key="onnx_output",
        fallback="models/exported.onnx",
    )
    out_path = Path(str(output_path)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_written: str | None = None
    runtime_error: str | None = None
    trt_info: dict[str, Any] | None = None
    template = ""

    if provider == "ultralytics":
        template = (
            "yolo export "
            f"model={model_name} format=onnx imgsz={int(args.imgsz)} opset={int(args.opset)} "
            f"dynamic={bool(args.dynamic)} half={bool(args.half)}"
        )
        if not bool(args.dry_run):
            try:
                from ultralytics import YOLO  # type: ignore

                model = YOLO(model_name)
                exported = model.export(
                    format="onnx",
                    imgsz=int(args.imgsz),
                    opset=int(args.opset),
                    dynamic=bool(args.dynamic),
                    half=bool(args.half),
                    simplify=bool(args.simplify),
                    device=str(args.device),
                )
                exported_path = Path(str(exported)).resolve() if exported else None
                if exported_path is None or not exported_path.exists():
                    raise RuntimeError("Ultralytics export returned no output path")
                if exported_path != out_path:
                    out_path.write_bytes(exported_path.read_bytes())
                onnx_written = str(out_path)
            except Exception as exc:
                runtime_error = str(exc)
    elif provider == "hf_detr":
        template = (
            f"{args.python} - <<'PY'  # export {model_name} to ONNX\n"
            "# uses transformers AutoModelForObjectDetection + torch.onnx.export\nPY"
        )
        if not bool(args.dry_run):
            try:
                import torch  # type: ignore
                from transformers import AutoModelForObjectDetection  # type: ignore

                model = AutoModelForObjectDetection.from_pretrained(model_name)
                model.eval()

                class _Wrap(torch.nn.Module):
                    def __init__(self, inner: Any):
                        super().__init__()
                        self.inner = inner

                    def forward(self, pixel_values: Any):
                        out = self.inner(pixel_values=pixel_values)
                        return out.logits, out.pred_boxes

                wrapped = _Wrap(model)
                dummy = torch.randn(1, 3, int(args.imgsz), int(args.imgsz))
                dynamic_axes = None
                if bool(args.dynamic):
                    dynamic_axes = {
                        "pixel_values": {0: "batch", 2: "height", 3: "width"},
                        "logits": {0: "batch"},
                        "pred_boxes": {0: "batch"},
                    }
                torch.onnx.export(
                    wrapped,
                    (dummy,),
                    str(out_path),
                    input_names=["pixel_values"],
                    output_names=["logits", "pred_boxes"],
                    dynamic_axes=dynamic_axes,
                    opset_version=int(args.opset),
                )
                onnx_written = str(out_path)
            except Exception as exc:
                runtime_error = str(exc)
    else:
        raise SystemExit("--provider must be ultralytics|hf_detr")

    if args.trt_engine:
        trt_cmd = [
            str(args.python),
            str(repo_root / "tools" / "build_trt_engine.py"),
            "--onnx",
            str(out_path),
            "--engine",
            str(Path(str(args.trt_engine)).resolve()),
            "--precision",
            str(args.trt_precision),
        ]
        if bool(args.dry_run):
            trt_info = {"dry_run": True, "cmd": trt_cmd}
        elif runtime_error is None:
            proc = _run(trt_cmd, cwd=repo_root)
            trt_info = {
                "dry_run": False,
                "cmd": trt_cmd,
                "returncode": int(proc.returncode),
                "stdout_tail": str(proc.stdout or "").splitlines()[-20:],
                "stderr_tail": str(proc.stderr or "").splitlines()[-20:],
            }
            if proc.returncode != 0:
                runtime_error = f"TensorRT engine build failed ({proc.returncode})"

    ok = bool(args.dry_run) or (runtime_error is None and onnx_written is not None)
    report = {
        "task": "export_onnx",
        "timestamp": _now_utc(),
        "ok": ok,
        "dry_run": bool(args.dry_run),
        "provider": provider,
        "model": model_name,
        "preset": preset_name,
        "output": str(out_path),
        "template_export_command": template,
        "onnx_written": onnx_written,
        "runtime_error": runtime_error,
        "trt": trt_info,
        "layers": {
            "trainer_runner": "N/A",
            "repo_impl": "support_external_training export-onnx",
            "export_deploy": "ONNX + optional TensorRT bridge",
        },
    }
    report_path = Path(str(_resolve_value(args.report, fallback="reports/support_external_training.export_onnx.json"))).resolve()
    _write_json(report_path, report)
    print(str(report_path))
    return 0 if ok else 1


def _cmd_predict_normalize(args: argparse.Namespace) -> int:
    preset_name, preset = _resolve_preset(getattr(args, "preset", None))
    input_value = _resolve_value(args.input)
    input_path = Path(str(input_value)).resolve() if input_value else None
    output_path = Path(str(_resolve_value(args.output, fallback="reports/predictions.normalized.json"))).resolve()
    report_out = Path(str(_resolve_value(args.report, fallback="reports/support_external_training.predict_normalize.json"))).resolve()
    ultralytics_model = _resolve_value(
        args.ultralytics_model,
        env="YOLOZU_MODEL",
        preset=preset,
        preset_key="ultralytics_model",
    )
    dataset_input = _resolve_value(
        args.dataset,
        env="YOLOZU_DATASET",
        preset=preset,
        preset_key="dataset",
    )
    split = str(
        _resolve_value(
            args.split,
            env="YOLOZU_SPLIT",
            preset=preset,
            preset_key="split",
            fallback="val",
        )
    )
    work_dir = Path(str(_resolve_value(args.work_dir, fallback="runs/support_external_training/predict"))).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    export_report: dict[str, Any] | None = None

    if input_path is None and ultralytics_model:
        if not dataset_input:
            raise SystemExit("--dataset is required when using --ultralytics-model")
        raw_export = work_dir / "ultralytics_raw_predictions.json"
        cmd = [
            str(args.python),
            str(repo_root / "tools" / "export_predictions_ultralytics.py"),
            "--model",
            str(ultralytics_model),
            "--dataset",
            str(dataset_input),
            "--split",
            split,
            "--output",
            str(raw_export),
        ]
        if bool(args.ultralytics_dry_run):
            cmd.append("--dry-run")
        proc = _run(cmd, cwd=repo_root)
        export_report = {
            "cmd": cmd,
            "returncode": int(proc.returncode),
            "stdout_tail": str(proc.stdout or "").splitlines()[-20:],
            "stderr_tail": str(proc.stderr or "").splitlines()[-20:],
        }
        if proc.returncode != 0:
            report = {
                "task": "predict_normalize",
                "timestamp": _now_utc(),
                "ok": False,
                "error": f"ultralytics export failed ({proc.returncode})",
                "export": export_report,
            }
            _write_json(report_out, report)
            print(str(report_out))
            return 1
        input_path = raw_export

    if input_path is None:
        raise SystemExit("provide --input or --ultralytics-model with --dataset")

    canonical = canonicalize_predictions_file(
        input_path=input_path,
        output_path=output_path,
        strict=bool(args.strict),
        classes_json=(str(args.classes) if args.classes else None),
        assume_class_id_is_category_id=bool(args.assume_class_id_is_category_id),
    )
    report = {
        "task": "predict_normalize",
        "timestamp": _now_utc(),
        "ok": True,
        "strict": bool(args.strict),
        "preset": preset_name,
        "input": str(input_path),
        "output": str(output_path),
        "canonicalization": canonical,
        "export": export_report,
        "layers": {
            "trainer_runner": "N/A",
            "repo_impl": "canonicalize_predictions_file",
            "export_deploy": "normalized predictions usable for ONNX/TensorRT parity/eval",
        },
    }
    _write_json(report_out, report)
    print(str(report_out))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    preset_choices = _preset_choices()
    format_choices = ("auto", "internal", "ultralytics", "coco", "coco_instances")
    preset_help = (
        "Preset defaults (smoke/coco128/none). "
        "If omitted, uses YOLOZU_CLI_PRESET or 'smoke'."
    )
    p = argparse.ArgumentParser(
        description=(
            "External training support helper with fixed 3-layer interface contract: "
            "trainer/runner, repo integration, export/deploy."
        )
    )
    sub = p.add_subparsers(dest="command", required=True)

    layers = sub.add_parser("layers", aliases=["ls"], help="Show fixed 3-layer support matrix.")
    layers.add_argument("-j", "--json", action="store_true", help="Emit JSON.")
    layers.set_defaults(_fn=_cmd_layers)

    ds = sub.add_parser(
        "dataset",
        aliases=["ds", "convert"],
        help="Convert COCO/Ultralytics/internal dataset into YOLOZU internal wrapper.",
    )
    ds.add_argument("-P", "--preset", choices=preset_choices, default=None, help=preset_help)
    ds.add_argument("-f", "--from", dest="from_format", default=None, choices=format_choices)
    ds.add_argument("-d", "--dataset", default=None, help="Dataset root or data.yaml path.")
    ds.add_argument("-s", "--split", default=None, help="Split override.")
    ds.add_argument("-i", "--instances-json", default=None, help="COCO instances JSON (for coco_instances mode).")
    ds.add_argument("-g", "--images-dir", default=None, help="COCO images dir (for coco_instances mode).")
    ds.add_argument("-o", "--output", default="runs/support_external_training/dataset", help="Output directory for converted wrapper.")
    ds.add_argument("-r", "--report", default="reports/support_external_training.dataset.json", help="Report JSON output path.")
    ds.add_argument("-F", "--force", action="store_true", help="Overwrite generated wrapper outputs.")
    ds.set_defaults(_fn=_cmd_dataset)

    tyx = sub.add_parser(
        "train-yolox",
        aliases=["ty", "yolox-train"],
        help="Apache-2.0-friendly YOLOX-style external training bridge.",
    )
    tyx.add_argument("-P", "--preset", choices=preset_choices, default=None, help=preset_help)
    tyx.add_argument("-x", "--exp", required=True, help="YOLOX exp file path.")
    tyx.add_argument("-c", "--weights", default=None, help="Optional checkpoint path to resume/fine-tune from.")
    tyx.add_argument("-t", "--train-script", default=None, help="Optional external YOLOX train launcher for non-dry execution.")
    tyx.add_argument("-p", "--python", default=sys.executable, help="Python executable for --train-script.")
    tyx.add_argument("-f", "--from", dest="from_format", default=None, choices=format_choices)
    tyx.add_argument("-d", "--dataset", default=None, help="Dataset root or descriptor.")
    tyx.add_argument("-s", "--split", default=None, help="Split for training (preset/env fallback when omitted).")
    tyx.add_argument("-i", "--instances-json", default=None, help="COCO instances JSON (for coco_instances mode).")
    tyx.add_argument("-g", "--images-dir", default=None, help="COCO images dir (for coco_instances mode).")
    tyx.add_argument("-b", "--batch", type=int, default=16, help="Global batch size (default: 16).")
    tyx.add_argument("-D", "--devices", type=int, default=1, help="Device count forwarded to the external YOLOX launcher (default: 1).")
    tyx.add_argument("-W", "--work-dir", default="runs/support_external_training/yolox", help="Work/cache dir.")
    tyx.add_argument("-o", "--output", default="reports/support_external_training.train_yolox.json", help="Report JSON output path.")
    tyx.add_argument("-n", "--dry-run", action="store_true", help="Do not execute runtime training.")
    tyx.add_argument("-F", "--force", action="store_true", help="Overwrite generated wrapper outputs.")
    tyx.set_defaults(_fn=_cmd_train_yolox)

    td2 = sub.add_parser(
        "train-detectron2",
        aliases=["td2", "detectron2-train"],
        help="Detectron2 external training bridge for bbox, instance segmentation, and keypoints.",
    )
    td2.add_argument("-P", "--preset", choices=preset_choices, default=None, help=preset_help)
    td2.add_argument("-c", "--config", required=True, help="Detectron2 config YAML path.")
    td2.add_argument("-t", "--train-script", default=None, help="Optional external Detectron2 train launcher for non-dry execution.")
    td2.add_argument("-p", "--python", default=sys.executable, help="Python executable for --train-script.")
    td2.add_argument("-f", "--from", dest="from_format", default=None, choices=format_choices)
    td2.add_argument("-d", "--dataset", default=None, help="Dataset root or descriptor.")
    td2.add_argument("-s", "--split", default=None, help="Split for training (preset/env fallback when omitted).")
    td2.add_argument("-i", "--instances-json", default=None, help="COCO instances JSON (for coco_instances mode).")
    td2.add_argument("-g", "--images-dir", default=None, help="COCO images dir (for coco_instances mode).")
    td2.add_argument(
        "--task-family",
        choices=("auto", "bbox", "segmentation", "keypoints"),
        default="auto",
        help="Reported task family; auto infers from the Detectron2 config text.",
    )
    td2.add_argument(
        "--train-opt",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=[],
        help="Extra Detectron2 config override appended to the launcher command. Repeat for DATASETS.TRAIN/TEST etc.",
    )
    td2.add_argument("-W", "--work-dir", default="runs/support_external_training/detectron2", help="Work/cache dir.")
    td2.add_argument("-o", "--output", default="reports/support_external_training.train_detectron2.json", help="Report JSON output path.")
    td2.add_argument("-n", "--dry-run", action="store_true", help="Do not execute runtime training.")
    td2.add_argument("-F", "--force", action="store_true", help="Overwrite generated wrapper outputs.")
    td2.set_defaults(_fn=_cmd_train_detectron2)

    tul = sub.add_parser(
        "train-ultralytics",
        aliases=["tu", "ultra-train"],
        help="Optional Ultralytics YOLO bridge + normalized prediction template.",
    )
    tul.add_argument("-P", "--preset", choices=preset_choices, default=None, help=preset_help)
    tul.add_argument("-m", "--model", default=None, help="Ultralytics model path/id (e.g., yolo11n.pt).")
    tul.add_argument("-f", "--from", dest="from_format", default=None, choices=format_choices)
    tul.add_argument("-d", "--dataset", default=None, help="Dataset root or data.yaml path.")
    tul.add_argument("-s", "--split", default=None, help="Split for training (preset/env fallback when omitted).")
    tul.add_argument("-i", "--instances-json", default=None, help="COCO instances JSON (for coco_instances mode).")
    tul.add_argument("-g", "--images-dir", default=None, help="COCO images dir (for coco_instances mode).")
    tul.add_argument("-e", "--epochs", type=int, default=100, help="Epochs (default: 100).")
    tul.add_argument("-z", "--imgsz", type=int, default=640, help="Image size (default: 640).")
    tul.add_argument("-b", "--batch", type=int, default=16, help="Batch size (default: 16).")
    tul.add_argument("-w", "--workers", type=int, default=8, help="Dataloader workers (default: 8).")
    tul.add_argument("-D", "--device", default="auto", help="Device (default: auto).")
    tul.add_argument("-p", "--project", default="runs/ultralytics_finetune", help="Training project dir.")
    tul.add_argument("-N", "--name", default="exp", help="Run name.")
    tul.add_argument("-W", "--work-dir", default="runs/support_external_training/ultralytics", help="Work/cache dir.")
    tul.add_argument("-o", "--output", default="reports/support_external_training.train_ultralytics.json", help="Report JSON output path.")
    tul.add_argument("-n", "--dry-run", action="store_true", help="Do not execute runtime training.")
    tul.add_argument("-F", "--force", action="store_true", help="Overwrite generated wrapper outputs.")
    tul.set_defaults(_fn=_cmd_train_ultralytics)

    thf = sub.add_parser(
        "train-hf-detr",
        aliases=["th", "hf-train"],
        help="HF DETR/RT-DETR training entry wrapper (template + script bridge).",
    )
    thf.add_argument("-P", "--preset", choices=preset_choices, default=None, help=preset_help)
    thf.add_argument("-m", "--model-id", default=None, help="HF model id (e.g. facebook/detr-resnet-50).")
    thf.add_argument("-f", "--from", dest="from_format", default=None, choices=format_choices)
    thf.add_argument("-d", "--dataset", default=None, help="Dataset root or descriptor.")
    thf.add_argument("-s", "--split", default=None, help="Split for training (preset/env fallback when omitted).")
    thf.add_argument("-i", "--instances-json", default=None, help="COCO instances JSON (for coco_instances mode).")
    thf.add_argument("-g", "--images-dir", default=None, help="COCO images dir (for coco_instances mode).")
    thf.add_argument("-e", "--epochs", type=int, default=10, help="Epochs (default: 10).")
    thf.add_argument("-b", "--batch-size", type=int, default=4, help="Batch size (default: 4).")
    thf.add_argument("-l", "--learning-rate", type=float, default=1e-4, help="Learning rate (default: 1e-4).")
    thf.add_argument("-k", "--max-steps", type=int, default=200, help="Max optimization steps (default: 200).")
    thf.add_argument("-t", "--train-script", default=None, help="Optional external HF train script for non-dry execution.")
    thf.add_argument("-p", "--python", default=sys.executable, help="Python executable for --train-script.")
    thf.add_argument("-W", "--work-dir", default="runs/support_external_training/hf_detr", help="Work/cache dir.")
    thf.add_argument("-o", "--output", default="reports/support_external_training.train_hf_detr.json", help="Report JSON output path.")
    thf.add_argument("-n", "--dry-run", action="store_true", help="Do not execute runtime training.")
    thf.add_argument("-F", "--force", action="store_true", help="Overwrite generated wrapper outputs.")
    thf.set_defaults(_fn=_cmd_train_hf_detr)

    eonnx = sub.add_parser(
        "export-onnx",
        aliases=["eo", "onnx"],
        help="Export ONNX for the YOLO-family runtime/HF DETR and optionally build TensorRT engine.",
    )
    eonnx.add_argument("-P", "--preset", choices=preset_choices, default=None, help=preset_help)
    eonnx.add_argument("-p", "--provider", choices=("ultralytics", "hf_detr"), default=None, help="Model provider.")
    eonnx.add_argument("-m", "--model", default=None, help="Model path/id.")
    eonnx.add_argument("-o", "--output", default=None, help="ONNX output path.")
    eonnx.add_argument("-r", "--report", default="reports/support_external_training.export_onnx.json", help="Report JSON output path.")
    eonnx.add_argument("-z", "--imgsz", type=int, default=640, help="Image size for export (default: 640).")
    eonnx.add_argument("-O", "--opset", type=int, default=17, help="ONNX opset (default: 17).")
    eonnx.add_argument("--dynamic", default=True, action=argparse.BooleanOptionalAction, help="Dynamic ONNX axes (default: true).")
    eonnx.add_argument("-H", "--half", action="store_true", help="FP16 export where supported.")
    eonnx.add_argument("-S", "--simplify", action="store_true", help="Simplify ONNX graph where supported.")
    eonnx.add_argument("-d", "--device", default="cpu", help="Export device (default: cpu).")
    eonnx.add_argument("-y", "--python", default=sys.executable, help="Python executable for optional TensorRT build.")
    eonnx.add_argument("-t", "--trt-engine", default=None, help="Optional TensorRT engine output path.")
    eonnx.add_argument("-q", "--trt-precision", default="fp16", choices=("fp16", "fp32", "int8"), help="TensorRT precision.")
    eonnx.add_argument("-n", "--dry-run", action="store_true", help="Do not execute runtime export.")
    eonnx.set_defaults(_fn=_cmd_export_onnx)

    pnorm = sub.add_parser(
        "predict-normalize",
        aliases=["pn", "normalize"],
        help="Normalize predictions into YOLOZU interface contract.",
    )
    pnorm.add_argument("-P", "--preset", choices=preset_choices, default=None, help=preset_help)
    pnorm.add_argument("-i", "--input", default=None, help="Input predictions JSON (external/raw).")
    pnorm.add_argument("-o", "--output", default=None, help="Normalized predictions JSON output path.")
    pnorm.add_argument("-r", "--report", default="reports/support_external_training.predict_normalize.json", help="Report JSON output path.")
    pnorm.add_argument("-c", "--classes", default=None, help="Optional labels/<split>/classes.json for class mapping.")
    pnorm.add_argument("-A", "--assume-class-id-is-category-id", action="store_true", help="Treat class_id as category_id before remap.")
    pnorm.add_argument("-s", "--strict", action="store_true", help="Strict canonicalization (error on out-of-range/unknown keys).")
    pnorm.add_argument("-m", "--ultralytics-model", default=None, help="Optional Ultralytics model to export predictions before normalization.")
    pnorm.add_argument("-d", "--dataset", default=None, help="Dataset root for --ultralytics-model mode.")
    pnorm.add_argument("-S", "--split", default=None, help="Dataset split for --ultralytics-model mode.")
    pnorm.add_argument("-n", "--ultralytics-dry-run", action="store_true", help="Pass --dry-run to export_predictions_ultralytics.")
    pnorm.add_argument("-y", "--python", default=sys.executable, help="Python executable for helper subprocesses.")
    pnorm.add_argument("-w", "--work-dir", default="runs/support_external_training/predict", help="Work/cache dir.")
    pnorm.set_defaults(_fn=_cmd_predict_normalize)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    fn = getattr(args, "_fn", None)
    if fn is None:
        raise SystemExit("missing handler")
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
