#!/usr/bin/env python3
"""External training/integration support tool.

Provides three-layer support helpers:
1) trainer/runner,
2) repo integration wrappers,
3) export/deploy (ONNX + optional TensorRT handoff).

The primary Apache-2.0-friendly training lane is YOLOX-style training via an
external YOLOX launcher. MM-family bridges (MMDetection / MMPose / MMSeg) are
also supported as external lanes. Optional bridges for Ultralytics YOLO and HF
DETR are kept explicit so the runtime/license boundary stays visible.
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
from yolozu.datasets.imports import (  # noqa: E402
    _require_module,
    project_detectron2_config,
    project_mmdet_config,
    project_yolox_exp,
)
from yolozu.core.canonical import TrainConfig  # noqa: E402
from yolozu.training.platform import build_training_run_summary  # noqa: E402
from yolozu.training.registry import (  # noqa: E402
    append_training_registry,
    build_training_registry_entry,
    write_training_registry_entry,
)


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
        "execution_status": dict(report.get("execution_status") or {}),
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
        "execution_status": dict(report.get("execution_status") or {}),
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


def _external_execution_status(
    *,
    dry_run: bool,
    training_executed: bool,
    runtime_error: str | None,
    train_script: str | None = None,
    requires_train_script: bool = False,
) -> dict[str, Any]:
    train_script_value = str(train_script).strip() if train_script else ""
    if dry_run:
        state = "dry_run_handoff"
        reason = "dry-run report and handoff artifacts were generated without executing backend training"
    elif training_executed:
        state = "executed"
        reason = None
    elif requires_train_script and not train_script_value:
        state = "requires_external_train_script"
        reason = runtime_error or "non-dry execution requires an external train script"
    else:
        state = "runtime_failed"
        reason = runtime_error or "backend runtime did not complete training"

    return {
        "state": state,
        "real_training_executed": bool(training_executed),
        "handoff_ready": bool(dry_run or training_executed or runtime_error is not None),
        "artifact_backed": True,
        "requires_external_train_script": bool(requires_train_script),
        "train_script": train_script_value or None,
        "skip_reason": reason,
    }


def _write_handoff_contracts(
    *,
    work_dir: Path,
    report_path: Path,
    handoff_contracts: dict[str, Any],
) -> dict[str, str]:
    reports_dir = work_dir / "reports"
    paths = {
        "resume_handoff": reports_dir / "resume_handoff.json",
        "export_handoff": reports_dir / "export_handoff.json",
        "eval_handoff": reports_dir / "eval_handoff.json",
        "parity_handoff": reports_dir / "parity_handoff.json",
    }
    resume_payload = handoff_contracts.get("resume")
    export_payload = handoff_contracts.get("export")
    eval_payload = handoff_contracts.get("eval")
    parity_payload = handoff_contracts.get("parity")
    if isinstance(resume_payload, dict):
        _write_json(paths["resume_handoff"], resume_payload)
    if isinstance(export_payload, dict):
        _write_json(paths["export_handoff"], export_payload)
    if isinstance(eval_payload, dict):
        _write_json(paths["eval_handoff"], eval_payload)
    if isinstance(parity_payload, dict):
        _write_json(paths["parity_handoff"], parity_payload)
    return {key: str(path) for key, path in paths.items()}


def _standard_handoff_payload(
    *,
    stage: str,
    backend_id: str,
    task_family: str,
    command: str,
    description: str,
    input_contract: dict[str, Any],
    output_contract: dict[str, Any],
    supported: bool = True,
) -> dict[str, Any]:
    return {
        "format": "yolozu_training_handoff_v1",
        "stage": str(stage),
        "backend_id": str(backend_id),
        "task_family": str(task_family),
        "supported": bool(supported),
        "command": str(command),
        "description": str(description),
        "input_contract": dict(input_contract),
        "output_contract": dict(output_contract),
    }


def _step(
    label: str,
    command: str,
    description: str,
    *,
    stage: str | None = None,
    input_contract: dict[str, Any] | None = None,
    output_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"label": str(label), "command": str(command), "description": str(description)}
    if stage is not None:
        payload["stage"] = str(stage)
    if input_contract is not None:
        payload["input_contract"] = dict(input_contract)
    if output_contract is not None:
        payload["output_contract"] = dict(output_contract)
    return payload


def _normalized_task_family(task_family: str | None) -> str:
    task = str(task_family or "bbox").strip().lower()
    if task in {"auto", "detect", ""}:
        return "bbox"
    if task == "pose":
        return "keypoints"
    return task


def _build_external_resume_command(
    *,
    backend_id: str,
    dataset_root: Path,
    split: str,
    config_token: str,
    task_family: str,
    report_path: Path,
) -> str:
    cmd = (
        f"python3 -m yolozu train --external-backend {backend_id} "
        f"{config_token} --dataset {dataset_root} --split {split} "
    )
    if task_family in {"segmentation", "keypoints"} and backend_id in {
        "detectron2",
        "mmdetection",
        "mmpose",
        "mmseg",
        "tao",
    }:
        cmd += f"--task-family {task_family} "
    cmd += f"--resume-from <path/to/checkpoint> --output {report_path}"
    return cmd


def _external_handoff_contracts(
    *,
    backend_id: str,
    dataset_root: Path,
    split: str,
    work_dir: Path,
    report_path: Path,
    config_path: Path | None = None,
    exp_path: Path | None = None,
    model_name: str | None = None,
    model_id: str | None = None,
    task_family: str | None = None,
) -> dict[str, Any]:
    task = _normalized_task_family(task_family)
    predictions_out = work_dir / "reports" / f"{backend_id}_predictions.json"
    segmentation_out = work_dir / "reports" / f"{backend_id}_segmentation_predictions.json"
    eval_out = work_dir / "reports" / f"{backend_id}_eval.json"
    parity_out = work_dir / "reports" / f"{backend_id}_parity.json"
    config_token = (
        str(exp_path)
        if exp_path is not None
        else str(config_path)
        if config_path is not None
        else str(model_name)
        if model_name is not None
        else str(model_id)
        if model_id is not None
        else "<config-or-model>"
    )

    export_command = ""
    export_description = ""
    export_input: dict[str, Any] = {}
    export_output: dict[str, Any] = {}
    export_supported = True
    eval_command = ""
    eval_description = ""
    parity_command = ""
    parity_description = ""

    if backend_id == "yolox" and exp_path is not None:
        export_command = (
            "python3 tools/export_predictions_yolox.py "
            f"--dataset {dataset_root} --split {split} --exp {exp_path} "
            f"--weights <path/to/yolox_ckpt.pth> --output {predictions_out}"
        )
        export_description = "Export YOLOX detections into the predictions interface contract."
        export_input = {"type": "yolox_checkpoint", "required": ["exp", "weights", "dataset", "split"]}
        export_output = {"type": "predictions interface contract", "path": str(predictions_out)}
    elif task == "bbox" and backend_id in {"detectron2", "mmdetection", "tao"}:
        export_command = (
            "python3 -m yolozu migrate predictions "
            "--from coco-results "
            "--results <path/to/coco_results.json> "
            "--instances <path/to/coco_instances.json> "
            f"--output {predictions_out}"
        )
        export_description = (
            f"Normalize {backend_id} COCO-style detection results into the predictions interface contract."
        )
        export_input = {"type": "coco_results_json", "required": ["results", "instances"]}
        export_output = {"type": "predictions interface contract", "path": str(predictions_out)}
    elif task == "keypoints" and backend_id in {"detectron2", "mmpose", "tao"}:
        export_command = (
            "python3 tools/export_predictions_coco_keypoints.py "
            "--results-json <path/to/keypoints_results.json> "
            "--instances-json <path/to/coco_instances.json> "
            f"--output {predictions_out}"
        )
        export_description = (
            f"Normalize {backend_id} COCO-style keypoints results into the predictions interface contract."
        )
        export_input = {"type": "coco_keypoints_results", "required": ["results_json", "instances_json"]}
        export_output = {"type": "predictions interface contract", "path": str(predictions_out)}
    elif task == "segmentation" and backend_id in {"detectron2", "mmdetection", "mmseg", "tao"}:
        export_command = (
            "python3 tools/package_segmentation_predictions.py "
            "--dataset-json <path/to/seg_dataset.json> "
            "--masks-dir <path/to/pred_mask_dir> "
            f"--output {segmentation_out}"
        )
        export_description = (
            f"Package {backend_id} class-id masks into the segmentation predictions interface contract."
        )
        export_input = {"type": "segmentation_masks_dir", "required": ["dataset_json", "masks_dir"]}
        export_output = {
            "type": "segmentation predictions interface contract",
            "path": str(segmentation_out),
        }
    elif backend_id == "ultralytics" and model_name is not None:
        export_command = (
            "python3 tools/support_external_training.py predict-normalize "
            f"--ultralytics-model {model_name} --dataset {dataset_root} --split {split} "
            f"--output {predictions_out} --report {work_dir / 'reports' / 'predict_normalize.json'}"
        )
        export_description = "Run Ultralytics prediction and normalize into the predictions interface contract."
        export_input = {"type": "ultralytics_runtime_model", "required": ["model", "dataset", "split"]}
        export_output = {"type": "predictions interface contract", "path": str(predictions_out)}
    elif backend_id == "hf-detr" and model_id is not None:
        export_command = (
            "python3 tools/support_external_training.py export-onnx "
            f"--provider hf_detr --model {model_id} --output {work_dir / 'exports' / 'model.onnx'} "
            f"--report {work_dir / 'reports' / 'export_onnx.json'}"
        )
        export_description = "Export an ONNX artifact for downstream deploy/parity checks."
        export_input = {"type": "hf_model_id", "required": ["model"]}
        export_output = {"type": "onnx_artifact", "path": str(work_dir / 'exports' / 'model.onnx')}
    else:
        export_supported = False
        export_command = f"Inspect {report_path} for backend-specific export requirements."
        export_description = "No standardized export handoff is registered for this backend/task pair."
        export_input = {"type": "backend_specific"}
        export_output = {"type": "backend_specific"}

    if task == "segmentation":
        eval_command = (
            "python3 tools/eval_segmentation.py "
            "--dataset-json <path/to/seg_dataset.json> "
            f"--predictions {segmentation_out} "
            f"--pred-root {work_dir / 'reports'} --output {eval_out}"
        )
        eval_description = "Evaluate semantic segmentation predictions via the stable segmentation eval lane."
        parity_command = (
            "python3 tools/check_segmentation_parity.py "
            f"--reference <reference_seg_predictions.json> --candidate {segmentation_out} "
            f"--output {parity_out}"
        )
        parity_description = "Compare semantic segmentation predictions against a reference artifact."
    elif task == "keypoints":
        eval_command = f"python3 tools/eval_keypoints.py --dataset {dataset_root} --split {split} --predictions {predictions_out} --output {eval_out}"
        eval_description = "Evaluate keypoint predictions with the stable keypoints eval lane."
        parity_command = f"python3 tools/check_keypoints_parity.py --reference <reference_predictions.json> --candidate {predictions_out} --output {parity_out}"
        parity_description = "Compare keypoint predictions against a reference artifact."
    else:
        eval_command = f"python3 -m yolozu eval-coco --dataset {dataset_root} --split {split} --predictions {predictions_out} --output {eval_out}"
        eval_description = "Evaluate detection predictions with the stable eval lane."
        parity_command = f"python3 -m yolozu parity --reference <reference_predictions.json> --candidate {predictions_out} --output {parity_out}"
        parity_description = "Compare candidate predictions against a reference backend artifact."

    return {
        "resume": _standard_handoff_payload(
            stage="resume",
            backend_id=backend_id,
            task_family=task,
            command=_build_external_resume_command(
                backend_id=backend_id,
                dataset_root=dataset_root,
                split=split,
                config_token=config_token,
                task_family=task,
                report_path=report_path,
            ),
            description="Resume or fine-tune the external backend from a prior checkpoint using the shared --resume-from surface.",
            input_contract={
                "type": "external_checkpoint_artifact",
                "required": ["resume_from"],
                "accepted_by": "--resume-from",
            },
            output_contract={
                "type": "external_run_contract_bundle",
                "paths": [
                    str(work_dir / "reports" / "training_summary.json"),
                    str(work_dir / "reports" / "resume_handoff.json"),
                ],
            },
        ),
        "export": _standard_handoff_payload(
            stage="export",
            backend_id=backend_id,
            task_family=task,
            command=export_command,
            description=export_description,
            input_contract=export_input,
            output_contract=export_output,
            supported=export_supported,
        ),
        "eval": _standard_handoff_payload(
            stage="eval",
            backend_id=backend_id,
            task_family=task,
            command=eval_command,
            description=eval_description,
            input_contract={"type": "export_output"},
            output_contract={"type": "evaluation_report", "path": str(eval_out)},
        ),
        "parity": _standard_handoff_payload(
            stage="parity",
            backend_id=backend_id,
            task_family=task,
            command=parity_command,
            description=parity_description,
            input_contract={"type": "reference_artifact_plus_export_output"},
            output_contract={"type": "parity_report", "path": str(parity_out)},
        ),
    }


def _external_next_steps(handoff_contracts: dict[str, Any], *, report_path: Path) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = []
    for stage in ("resume", "export", "eval", "parity"):
        payload = handoff_contracts.get(stage)
        if not isinstance(payload, dict) or not bool(payload.get("supported", True)):
            continue
        steps.append(
            _step(
                "resume_training" if stage == "resume" else str(stage),
                str(payload.get("command") or ""),
                str(payload.get("description") or ""),
                stage=str(stage),
                input_contract=payload.get("input_contract") if isinstance(payload.get("input_contract"), dict) else {},
                output_contract=payload.get("output_contract") if isinstance(payload.get("output_contract"), dict) else {},
            )
        )

    steps.append(
        _step(
            "inspect_summary",
            f"python3 - <<'PY'\nimport json\nprint(json.dumps(json.load(open(r'{report_path}','r',encoding='utf-8')), indent=2)[:4000])\nPY",
            "Inspect the machine-readable training summary and wrapper-owned run bundle.",
            stage="inspect",
            input_contract={"type": "training_run_summary_json", "path": str(report_path)},
            output_contract={"type": "stdout_preview"},
        )
    )
    return steps


def _yolox_artifact_plan(
    *,
    work_dir: Path,
    handoff_contracts: dict[str, Any],
) -> dict[str, Any]:
    paths = _external_run_contract_paths(work_dir)
    export_contract = handoff_contracts.get("export") if isinstance(handoff_contracts, dict) else None
    eval_contract = handoff_contracts.get("eval") if isinstance(handoff_contracts, dict) else None
    parity_contract = handoff_contracts.get("parity") if isinstance(handoff_contracts, dict) else None

    next_commands: dict[str, str] = {}
    for stage in ("resume", "export", "eval", "parity"):
        payload = handoff_contracts.get(stage)
        if isinstance(payload, dict) and bool(payload.get("supported", True)):
            next_commands[stage] = str(payload.get("command") or "")

    return {
        "format": "yolozu_external_training_artifact_plan_v1",
        "lane": "yolox",
        "status": "artifact_backed_dry_run_plan",
        "runtime_license_boundary": {
            "repo_code": "Apache-2.0",
            "external_runtime": "YOLOX",
            "external_runtime_license": "Apache-2.0-friendly",
            "vendored": False,
        },
        "expected_inputs": {
            "exp": "YOLOX exp file",
            "weights": "External YOLOX checkpoint for export/eval handoff",
            "dataset": "YOLOZU dataset root plus split",
        },
        "expected_outputs": {
            "training_summary": str(paths["training_summary"]),
            "external_run_meta": str(paths["external_run_meta"]),
            "launcher_plan": str(paths["launcher_plan"]),
            "execution": str(paths["execution"]),
            "train_config_projection": str(paths["train_config_projection"]),
            "predictions_json": str(((export_contract or {}).get("output_contract") or {}).get("path") or ""),
            "eval_report": str(((eval_contract or {}).get("output_contract") or {}).get("path") or ""),
            "parity_report": str(((parity_contract or {}).get("output_contract") or {}).get("path") or ""),
        },
        "next_commands": next_commands,
        "dry_run_validates": [
            "dataset_resolution",
            "train_config_projection",
            "runtime_license_boundary",
            "resume_export_eval_parity_next_commands",
            "expected_outputs",
        ],
    }


def _optional_bridge_runtime_boundary(
    *,
    bridge_id: str,
    external_runtime: str,
    license_note: str,
) -> dict[str, Any]:
    return {
        "format": "yolozu_optional_bridge_runtime_boundary_v1",
        "bridge_id": str(bridge_id),
        "bridge_kind": "optional_external_runtime",
        "stable_core_boundary": "YOLOZU owns dataset resolution, reports, and predictions interface contract artifacts.",
        "external_runtime": str(external_runtime),
        "bundled_with_yolozu": False,
        "default_install_dependency": False,
        "license_review_required": True,
        "license_note": str(license_note),
    }


def _run(
    cmd: list[str],
    *,
    cwd: Path = repo_root,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=127,
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}",
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
    resume_from: str | None = None,
) -> str:
    command = (
        "yolo train "
        f"model={model} data={data_yaml} epochs={int(epochs)} imgsz={int(imgsz)} "
        f"batch={int(batch)} device={device} project={project} name={name}"
    )
    if resume_from:
        command += f" resume=False pretrained={resume_from}"
    return command


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
    resume_from: str | None,
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
    if resume_from:
        cmd.extend(["-c", str(resume_from)])
    return cmd


def _infer_detectron2_task_family(config_path: Path) -> str:
    text = config_path.read_text(encoding="utf-8", errors="ignore")
    lowered = text.lower()
    if "coco-keypoints" in lowered or "keypoint_on: true" in lowered:
        return "keypoints"
    if "instance-segmentation" in lowered or "mask_on: true" in lowered:
        return "segmentation"
    return "bbox"


def _infer_mmdetection_task_family(config_path: Path) -> str:
    text = config_path.read_text(encoding="utf-8", errors="ignore")
    lowered = text.lower()
    if "mask_head" in lowered or "instance" in lowered:
        return "segmentation"
    return "bbox"


def _infer_tao_task_family(config_path: Path) -> str:
    text = config_path.read_text(encoding="utf-8", errors="ignore")
    lowered = text.lower()
    if "keypoint" in lowered or "centerpose" in lowered or "pose" in lowered:
        return "keypoints"
    if "segmentation" in lowered or "mask" in lowered:
        return "segmentation"
    return "bbox"


def _build_detectron2_train_command(
    *,
    python: str,
    train_script: str,
    config: Path,
    work_dir: Path,
    resume_from: str | None,
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
    if resume_from:
        cmd.extend(["MODEL.WEIGHTS", str(resume_from)])
    for key, value in train_opts:
        cmd.extend([str(key), str(value)])
    return cmd


def _build_mmengine_train_command(
    *,
    python: str,
    train_script: str,
    config: Path,
    work_dir: Path,
    resume_from: str | None,
    train_opts: list[tuple[str, str]],
) -> list[str]:
    cmd = [
        str(python),
        str(train_script),
        str(config),
        "--work-dir",
        str(work_dir),
    ]
    if resume_from:
        cmd.extend(["--cfg-options", f"load_from={str(resume_from)}"])
    if train_opts:
        cmd.append("--cfg-options")
        cmd.extend([f"{str(key)}={str(value)}" for key, value in train_opts])
    return cmd


def _build_tao_train_command(
    *,
    python: str,
    train_script: str | None,
    tao_task: str,
    config: Path,
    work_dir: Path,
    resume_from: str | None,
    train_opts: list[tuple[str, str]],
) -> list[str]:
    if train_script:
        cmd = [
            str(python),
            str(train_script),
            "--spec",
            str(config),
            "--results-dir",
            str(work_dir),
            "--tao-task",
            str(tao_task),
        ]
        if resume_from:
            cmd.extend(["--resume-from", str(resume_from)])
        for key, value in train_opts:
            cmd.extend(["--set", f"{str(key)}={str(value)}"])
        return cmd

    cmd = [
        "tao",
        "model",
        str(tao_task),
        "train",
        "-e",
        str(config),
        "-r",
        str(work_dir),
    ]
    if resume_from:
        cmd.extend(["--resume-from", str(resume_from)])
    for key, value in train_opts:
        cmd.extend(["--set", f"{str(key)}={str(value)}"])
    return cmd


def _project_mmengine_training_config(
    *,
    backend_id: str,
    config_path: Path,
    dataset_root: Path,
    split: str,
    task_family: str,
) -> tuple[TrainConfig, str | None]:
    if backend_id == "mmdetection":
        try:
            train_cfg = project_mmdet_config(config=config_path)
        except Exception as exc:
            return (
                TrainConfig(
                    backend="mmdetection",
                    task=task_family,
                    model=str(config_path),
                    dataset={"root": str(dataset_root), "split": str(split)},
                    source={"from": "mmdetection", "config": str(config_path)},
                ),
                str(exc),
            )
        return (
            replace(
                train_cfg,
                backend="mmdetection",
                task=task_family,
                model=str(config_path),
                dataset={"root": str(dataset_root), "split": str(split)},
                source={"from": "mmdetection", "config": str(config_path)},
            ),
            None,
        )

    try:
        mmengine_config = _require_module("mmengine.config", pip_hint="pip install mmengine").Config
        cfg_obj = mmengine_config.fromfile(str(config_path))
        cfg = cfg_obj.to_dict() if hasattr(cfg_obj, "to_dict") else dict(cfg_obj)
    except Exception as exc:
        return (
            TrainConfig(
                backend=backend_id,
                task=task_family,
                model=str(config_path),
                dataset={"root": str(dataset_root), "split": str(split)},
                source={"from": backend_id, "config": str(config_path)},
            ),
            str(exc),
        )

    train_dataloader = cfg.get("train_dataloader") if isinstance(cfg, dict) else {}
    batch = None
    if isinstance(train_dataloader, dict):
        raw_batch = train_dataloader.get("batch_size")
        if isinstance(raw_batch, int):
            batch = int(raw_batch)
    train_cfg = cfg.get("train_cfg") if isinstance(cfg, dict) else {}
    epochs = train_cfg.get("max_epochs") if isinstance(train_cfg, dict) else None
    steps = train_cfg.get("max_iters") if isinstance(train_cfg, dict) else None
    optim_wrapper = cfg.get("optim_wrapper") if isinstance(cfg, dict) else {}
    optimizer = optim_wrapper.get("optimizer") if isinstance(optim_wrapper, dict) else {}
    lr = optimizer.get("lr") if isinstance(optimizer, dict) else None
    weight_decay = optimizer.get("weight_decay") if isinstance(optimizer, dict) else None
    return (
        TrainConfig(
            backend=backend_id,
            task=task_family,
            model=str(config_path),
            batch=(int(batch) if isinstance(batch, int) else None),
            epochs=(int(epochs) if isinstance(epochs, int) else None),
            steps=(int(steps) if isinstance(steps, int) else None),
            optimizer=(str(optimizer.get("type")) if isinstance(optimizer, dict) and optimizer.get("type") else None),
            lr=(float(lr) if isinstance(lr, (int, float)) else None),
            weight_decay=(float(weight_decay) if isinstance(weight_decay, (int, float)) else None),
            dataset={"root": str(dataset_root), "split": str(split)},
            source={"from": backend_id, "config": str(config_path)},
        ),
        None,
    )


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
    resume_value = _resolve_value(
        getattr(args, "resume_from", None),
        env="YOLOZU_RESUME_FROM",
        preset=preset,
        preset_key="resume_from",
    )
    if resume_value is None and args.weights:
        resume_value = str(args.weights)
    resume_path = Path(str(resume_value)).resolve() if resume_value else None
    train_script = str(args.train_script).strip() if args.train_script else ""
    command = _build_yolox_train_command(
        python=str(args.python),
        train_script=(train_script or "<YOLOX/tools/train.py>"),
        exp_file=str(exp_path),
        batch=int(args.batch),
        devices=int(args.devices),
        resume_from=(str(resume_path) if resume_path else None),
    )
    template = _yolox_train_template(
        python=str(args.python),
        train_script=(train_script or "<YOLOX/tools/train.py>"),
        exp_file=str(exp_path),
        batch=int(args.batch),
        weights=(str(resume_path) if resume_path else None),
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
    handoff_contracts = _external_handoff_contracts(
        backend_id="yolox",
        dataset_root=Path(str(resolution.dataset_root)),
        split=str(resolution.split),
        work_dir=work_dir,
        report_path=report_path,
        exp_path=exp_path,
        task_family="bbox",
    )
    report = build_training_run_summary(
        backend_id="yolox",
        report_path=report_path,
        train_config=train_cfg,
        dataset_root=str(resolution.dataset_root),
        split=str(resolution.split),
        raw_dataset_format=str(resolution.source_format),
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
        handoff_contracts=handoff_contracts,
    )
    report["execution_status"] = _external_execution_status(
        dry_run=bool(args.dry_run),
        training_executed=training_executed,
        runtime_error=runtime_error,
        train_script=train_script or None,
        requires_train_script=True,
    )
    report.update(
        {
            "task": "train_yolox",
            "preset": preset_name,
            "exp": str(exp_path),
            "weights": str(resume_path) if resume_path else None,
            "resume_from": str(resume_path) if resume_path else None,
            "train_script": train_script or None,
            "template_train_command": template,
            "train_config_projection": str(projection_path),
            "layers": {
                "trainer_runner": "external YOLOX train launcher",
                "repo_impl": "support_external_training train-yolox",
                "export_deploy": "export_predictions_yolox + eval/benchmark lanes",
            },
            "artifact_plan": _yolox_artifact_plan(work_dir=work_dir, handoff_contracts=handoff_contracts),
        }
    )
    report["next_steps"] = _external_next_steps(handoff_contracts, report_path=report_path)
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
    handoff_paths = _write_handoff_contracts(work_dir=work_dir, report_path=report_path, handoff_contracts=handoff_contracts)
    registry_entry = build_training_registry_entry(summary=report, summary_path=report_path)
    registry_entry_path = work_dir / "reports" / "training_registry_entry.json"
    write_training_registry_entry(registry_entry_path, registry_entry)
    if getattr(args, "registry_out", None):
        append_training_registry(Path(str(args.registry_out)).resolve(), registry_entry)
    report["run_output_contract"]["stable_artifact_paths"] = bundled_paths
    report["run_output_contract"]["stable_artifact_paths"].update(handoff_paths)
    report["run_output_contract"]["stable_artifact_paths"]["training_registry_entry"] = str(registry_entry_path)
    report["train_config_projection"] = bundled_paths["train_config_projection"]
    report["registry_entry"] = str(registry_entry_path)
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
    resume_value = _resolve_value(
        getattr(args, "resume_from", None),
        env="YOLOZU_RESUME_FROM",
        preset=preset,
        preset_key="resume_from",
    )
    resume_path = Path(str(resume_value)).resolve() if resume_value else None
    train_opts = [(str(k), str(v)) for k, v in (getattr(args, "train_opt", None) or [])]
    command = _build_detectron2_train_command(
        python=str(args.python),
        train_script=(train_script or "<detectron2/tools/train_net.py>"),
        config=config_path,
        work_dir=work_dir,
        resume_from=(str(resume_path) if resume_path else None),
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
    handoff_contracts = _external_handoff_contracts(
        backend_id="detectron2",
        dataset_root=Path(str(resolution.dataset_root)),
        split=str(resolution.split),
        work_dir=work_dir,
        report_path=report_path,
        config_path=config_path,
        task_family=task_family,
    )
    report = build_training_run_summary(
        backend_id="detectron2",
        report_path=report_path,
        train_config=train_cfg,
        dataset_root=str(resolution.dataset_root),
        split=str(resolution.split),
        raw_dataset_format=str(resolution.source_format),
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
        handoff_contracts=handoff_contracts,
    )
    report["execution_status"] = _external_execution_status(
        dry_run=bool(args.dry_run),
        training_executed=training_executed,
        runtime_error=runtime_error,
        train_script=train_script or None,
        requires_train_script=True,
    )
    report.update(
        {
            "task": "train_detectron2",
            "task_family": task_family,
            "config": str(config_path),
            "resume_from": str(resume_path) if resume_path else None,
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
    report["next_steps"] = _external_next_steps(handoff_contracts, report_path=report_path)
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
    handoff_paths = _write_handoff_contracts(work_dir=work_dir, report_path=report_path, handoff_contracts=handoff_contracts)
    registry_entry = build_training_registry_entry(summary=report, summary_path=report_path)
    registry_entry_path = work_dir / "reports" / "training_registry_entry.json"
    write_training_registry_entry(registry_entry_path, registry_entry)
    if getattr(args, "registry_out", None):
        append_training_registry(Path(str(args.registry_out)).resolve(), registry_entry)
    report["run_output_contract"]["stable_artifact_paths"] = bundled_paths
    report["run_output_contract"]["stable_artifact_paths"].update(handoff_paths)
    report["run_output_contract"]["stable_artifact_paths"]["training_registry_entry"] = str(registry_entry_path)
    report["train_config_projection"] = bundled_paths["train_config_projection"]
    report["registry_entry"] = str(registry_entry_path)
    _write_json(report_path, report)
    _write_json(Path(bundled_paths["training_summary"]), report)
    print(str(report_path))
    return 0 if bool(report.get("ok")) else 1


def _cmd_train_mmfamily(
    args: argparse.Namespace,
    *,
    backend_id: str,
    default_task_family: str,
    default_work_dir: str,
    default_output: str,
    provider_label: str,
    repo_impl_label: str,
    export_deploy_label: str,
) -> int:
    preset_name, preset = _resolve_preset(getattr(args, "preset", None))
    config_path = Path(
        _require_nonempty(
            _resolve_value(args.config),
            message=f"{provider_label} config file is required (set --config).",
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
    work_dir = Path(str(_resolve_value(args.work_dir, fallback=default_work_dir))).resolve()
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

    task_family = str(getattr(args, "task_family", default_task_family) or default_task_family).strip().lower()
    if task_family == "auto":
        task_family = default_task_family
        if backend_id == "mmdetection":
            task_family = _infer_mmdetection_task_family(config_path)

    train_cfg, projection_error = _project_mmengine_training_config(
        backend_id=backend_id,
        config_path=config_path,
        dataset_root=Path(str(resolution.dataset_root)),
        split=str(resolution.split),
        task_family=task_family,
    )

    projection_path = work_dir / f"{backend_id}_train_config_projection.json"
    projection_payload = {
        "format": "yolozu_external_training_projection_v1",
        "provider": backend_id,
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
    resume_value = _resolve_value(
        getattr(args, "resume_from", None),
        env="YOLOZU_RESUME_FROM",
        preset=preset,
        preset_key="resume_from",
    )
    resume_path = Path(str(resume_value)).resolve() if resume_value else None
    train_opts = [(str(k), str(v)) for k, v in (getattr(args, "train_opt", None) or [])]
    command = _build_mmengine_train_command(
        python=str(args.python),
        train_script=(train_script or f"<{backend_id}/tools/train.py>"),
        config=config_path,
        work_dir=work_dir,
        resume_from=(str(resume_path) if resume_path else None),
        train_opts=train_opts,
    )
    template = " ".join(command)

    training_executed = False
    runtime_error: str | None = None
    proc_info: dict[str, Any] | None = None

    if not bool(args.dry_run):
        if not train_script:
            runtime_error = (
                f"{provider_label} non-dry execution requires --train-script pointing to an external "
                f"{backend_id}/tools/train.py style launcher."
            )
        elif not Path(train_script).exists():
            runtime_error = f"{provider_label} train script not found: {train_script}"
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
                runtime_error = f"external {provider_label} train script failed ({proc.returncode})"

    report_path = Path(str(_resolve_value(args.output, fallback=default_output))).resolve()
    handoff_contracts = _external_handoff_contracts(
        backend_id=backend_id,
        dataset_root=Path(str(resolution.dataset_root)),
        split=str(resolution.split),
        work_dir=work_dir,
        report_path=report_path,
        config_path=config_path,
        task_family=task_family,
    )
    report = build_training_run_summary(
        backend_id=backend_id,
        report_path=report_path,
        train_config=train_cfg,
        dataset_root=str(resolution.dataset_root),
        split=str(resolution.split),
        raw_dataset_format=str(resolution.source_format),
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
            f"task family is reported as {task_family}.",
        ],
        license_boundary={
            "repo_code": "Apache-2.0",
            "optional_bridge": False,
            "note": f"{provider_label} runtime remains external to YOLOZU; review upstream terms and environment setup separately.",
        },
        handoff_contracts=handoff_contracts,
    )
    report["execution_status"] = _external_execution_status(
        dry_run=bool(args.dry_run),
        training_executed=training_executed,
        runtime_error=runtime_error,
        train_script=train_script or None,
        requires_train_script=True,
    )
    report.update(
        {
            "task": f"train_{backend_id}",
            "task_family": task_family,
            "config": str(config_path),
            "resume_from": str(resume_path) if resume_path else None,
            "train_script": train_script or None,
            "template_train_command": template,
            "train_config_projection": str(projection_path),
            "projection_error": projection_error,
            "layers": {
                "trainer_runner": f"external {provider_label} train launcher",
                "repo_impl": repo_impl_label,
                "export_deploy": export_deploy_label,
            },
        }
    )
    report["next_steps"] = _external_next_steps(handoff_contracts, report_path=report_path)
    bundled_paths = _write_external_run_contract_bundle(
        backend_id=backend_id,
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
    handoff_paths = _write_handoff_contracts(
        work_dir=work_dir,
        report_path=report_path,
        handoff_contracts=handoff_contracts,
    )
    registry_entry = build_training_registry_entry(summary=report, summary_path=report_path)
    registry_entry_path = work_dir / "reports" / "training_registry_entry.json"
    write_training_registry_entry(registry_entry_path, registry_entry)
    if getattr(args, "registry_out", None):
        append_training_registry(Path(str(args.registry_out)).resolve(), registry_entry)
    report["run_output_contract"]["stable_artifact_paths"] = bundled_paths
    report["run_output_contract"]["stable_artifact_paths"].update(handoff_paths)
    report["run_output_contract"]["stable_artifact_paths"]["training_registry_entry"] = str(registry_entry_path)
    report["train_config_projection"] = bundled_paths["train_config_projection"]
    report["registry_entry"] = str(registry_entry_path)
    _write_json(report_path, report)
    _write_json(Path(bundled_paths["training_summary"]), report)
    print(str(report_path))
    return 0 if bool(report.get("ok")) else 1


def _cmd_train_mmdetection(args: argparse.Namespace) -> int:
    return _cmd_train_mmfamily(
        args,
        backend_id="mmdetection",
        default_task_family="bbox",
        default_work_dir="runs/support_external_training/mmdetection",
        default_output="reports/support_external_training.train_mmdetection.json",
        provider_label="MMDetection",
        repo_impl_label="support_external_training train-mmdetection",
        export_deploy_label="task-family-specific handoff bridges normalize MMDetection outputs into YOLOZU export/eval/parity lanes.",
    )


def _cmd_train_mmpose(args: argparse.Namespace) -> int:
    return _cmd_train_mmfamily(
        args,
        backend_id="mmpose",
        default_task_family="keypoints",
        default_work_dir="runs/support_external_training/mmpose",
        default_output="reports/support_external_training.train_mmpose.json",
        provider_label="MMPose",
        repo_impl_label="support_external_training train-mmpose",
        export_deploy_label="task-family-specific handoff bridges normalize MMPose outputs into YOLOZU export/eval/parity lanes.",
    )


def _cmd_train_mmseg(args: argparse.Namespace) -> int:
    return _cmd_train_mmfamily(
        args,
        backend_id="mmseg",
        default_task_family="segmentation",
        default_work_dir="runs/support_external_training/mmseg",
        default_output="reports/support_external_training.train_mmseg.json",
        provider_label="MMSeg",
        repo_impl_label="support_external_training train-mmseg",
        export_deploy_label="task-family-specific handoff bridges normalize MMSeg outputs into YOLOZU export/eval/parity lanes.",
    )


def _cmd_train_tao(args: argparse.Namespace) -> int:
    preset_name, preset = _resolve_preset(getattr(args, "preset", None))
    config_path = Path(
        _require_nonempty(
            _resolve_value(args.config),
            message="TAO spec file is required (set --config).",
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
    work_dir = Path(str(_resolve_value(args.work_dir, fallback="runs/support_external_training/tao"))).resolve()
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
        task_family = _infer_tao_task_family(config_path)

    resume_value = _resolve_value(
        getattr(args, "resume_from", None),
        env="YOLOZU_RESUME_FROM",
        preset=preset,
        preset_key="resume_from",
    )
    resume_path = Path(str(resume_value)).resolve() if resume_value else None
    train_script = str(args.train_script).strip() if args.train_script else ""
    train_opts = [(str(k), str(v)) for k, v in (getattr(args, "train_opt", None) or [])]
    tao_task = str(
        _resolve_value(
            getattr(args, "tao_task", None),
            preset=preset,
            preset_key="tao_task",
            fallback=("centerpose" if task_family == "keypoints" else "segformer" if task_family == "segmentation" else "dino"),
        )
    )
    command = _build_tao_train_command(
        python=str(args.python),
        train_script=(train_script or None),
        tao_task=tao_task,
        config=config_path,
        work_dir=work_dir,
        resume_from=(str(resume_path) if resume_path else None),
        train_opts=train_opts,
    )
    template = " ".join(command)

    projection_payload = {
        "format": "yolozu_external_training_projection_v1",
        "provider": "tao",
        "train_config": TrainConfig(
            backend="tao",
            task=task_family,
            model=str(config_path),
            dataset={"root": str(resolution.dataset_root), "split": str(resolution.split)},
            source={"from": "tao", "config": str(config_path), "tao_task": str(tao_task)},
        ).to_dict(),
        "dataset_resolution": {
            "dataset_root": str(resolution.dataset_root),
            "split": str(resolution.split),
            "source_format": str(resolution.source_format),
        },
        "environment_hints": {
            "YOLOZU_DATASET_ROOT": str(resolution.dataset_root),
            "YOLOZU_SPLIT": str(resolution.split),
            "YOLOZU_TASK_FAMILY": str(task_family),
            "YOLOZU_TAO_TASK": str(tao_task),
        },
    }
    projection_path = work_dir / "tao_train_config_projection.json"
    _write_json(projection_path, projection_payload)

    training_executed = False
    runtime_error: str | None = None
    failure_code: str | None = None
    proc_info: dict[str, Any] | None = None
    if not bool(args.dry_run):
        proc = _run(command, cwd=repo_root, env=dict(os.environ))
        proc_info = {
            "returncode": int(proc.returncode),
            "stdout_tail": str(proc.stdout or "").splitlines()[-20:],
            "stderr_tail": str(proc.stderr or "").splitlines()[-20:],
        }
        training_executed = proc.returncode == 0
        if proc.returncode != 0:
            if proc.returncode == 127:
                detail = (proc.stderr or "").strip()
                runtime_error = f"external NVIDIA TAO runtime unavailable: {detail}"
                failure_code = "E_EXTERNAL_RUNTIME_MISSING"
            else:
                runtime_error = f"external NVIDIA TAO train command failed ({proc.returncode})"
                failure_code = "E_EXTERNAL_RUNTIME_FAILED"

    report_path = Path(str(_resolve_value(args.output, fallback="reports/support_external_training.train_tao.json"))).resolve()
    handoff_contracts = _external_handoff_contracts(
        backend_id="tao",
        dataset_root=Path(str(resolution.dataset_root)),
        split=str(resolution.split),
        work_dir=work_dir,
        report_path=report_path,
        config_path=config_path,
        task_family=task_family,
    )
    report = build_training_run_summary(
        backend_id="tao",
        report_path=report_path,
        train_config=projection_payload["train_config"],
        dataset_root=str(resolution.dataset_root),
        split=str(resolution.split),
        raw_dataset_format=str(resolution.source_format),
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
        notes=[f"preset={preset_name}", f"projection={projection_path}", f"tao_task={tao_task}"],
        license_boundary={
            "repo_code": "Apache-2.0",
            "optional_bridge": False,
            "note": "NVIDIA TAO runtime remains external to YOLOZU; review NVIDIA tooling, container, and redistribution requirements separately.",
        },
        handoff_contracts=handoff_contracts,
    )
    report["execution_status"] = _external_execution_status(
        dry_run=bool(args.dry_run),
        training_executed=training_executed,
        runtime_error=runtime_error,
        train_script=train_script or None,
        requires_train_script=False,
    )
    report.update(
        {
            "task": "train_tao",
            "task_family": task_family,
            "config": str(config_path),
            "tao_task": tao_task,
            "resume_from": str(resume_path) if resume_path else None,
            "train_script": train_script or None,
            "template_train_command": template,
            "train_config_projection": str(projection_path),
            "failure_code": failure_code,
            "layers": {
                "trainer_runner": "external NVIDIA TAO launcher",
                "repo_impl": "support_external_training train-tao",
                "export_deploy": "task-family-specific handoff bridges into YOLOZU eval/parity lanes",
            },
        }
    )
    report["next_steps"] = _external_next_steps(handoff_contracts, report_path=report_path)
    bundled_paths = _write_external_run_contract_bundle(
        backend_id="tao",
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
    handoff_paths = _write_handoff_contracts(work_dir=work_dir, report_path=report_path, handoff_contracts=handoff_contracts)
    registry_entry = build_training_registry_entry(summary=report, summary_path=report_path)
    registry_entry_path = work_dir / "reports" / "training_registry_entry.json"
    write_training_registry_entry(registry_entry_path, registry_entry)
    if getattr(args, "registry_out", None):
        append_training_registry(Path(str(args.registry_out)).resolve(), registry_entry)
    report["run_output_contract"]["stable_artifact_paths"] = bundled_paths
    report["run_output_contract"]["stable_artifact_paths"].update(handoff_paths)
    report["run_output_contract"]["stable_artifact_paths"]["training_registry_entry"] = str(registry_entry_path)
    report["train_config_projection"] = bundled_paths["train_config_projection"]
    report["registry_entry"] = str(registry_entry_path)
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

    resume_value = _resolve_value(
        getattr(args, "resume_from", None),
        env="YOLOZU_RESUME_FROM",
        preset=preset,
        preset_key="resume_from",
    )
    resume_path = Path(str(resume_value)).resolve() if resume_value else None

    template = _ultralytics_train_template(
        model=(str(resume_path) if resume_path else model_name),
        data_yaml=data_yaml,
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        project=str(args.project),
        name=str(args.name),
        resume_from=(str(resume_path) if resume_path else None),
    )

    training_executed = False
    run_dir: str | None = None
    runtime_error: str | None = None

    if not bool(args.dry_run):
        try:
            from ultralytics import YOLO  # type: ignore

            model = YOLO(str(resume_path) if resume_path else model_name)
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
    handoff_contracts = _external_handoff_contracts(
        backend_id="ultralytics",
        dataset_root=Path(str(resolution.dataset_root)),
        split=str(resolution.split),
        work_dir=work_dir,
        report_path=report_path,
        model_name=str(model_name),
        task_family="bbox",
    )
    runtime_license_boundary = _optional_bridge_runtime_boundary(
        bridge_id="ultralytics",
        external_runtime="Ultralytics",
        license_note="Ultralytics runtime is optional and must be reviewed under its own license terms.",
    )
    report = build_training_run_summary(
        backend_id="ultralytics",
        report_path=report_path,
        train_config=train_cfg,
        dataset_root=str(resolution.dataset_root),
        split=str(resolution.split),
        raw_dataset_format=str(resolution.source_format),
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
            "runtime_license_boundary": runtime_license_boundary,
        },
        handoff_contracts=handoff_contracts,
    )
    report["execution_status"] = _external_execution_status(
        dry_run=bool(args.dry_run),
        training_executed=training_executed,
        runtime_error=runtime_error,
        requires_train_script=False,
    )
    report.update(
        {
            "task": "train_ultralytics",
            "preset": preset_name,
            "model": model_name,
            "resume_from": str(resume_path) if resume_path else None,
            "data_yaml": str(data_yaml),
            "template_train_command": template,
            "template_predict_normalize_command": (
                "python3 tools/support_external_training.py predict-normalize "
                f"--ultralytics-model {model_name} --dataset {resolution.dataset_root} "
                f"--split {resolution.split} --output reports/ultralytics_predictions.normalized.json "
                "--report reports/ultralytics_predict_normalize_report.json"
            ),
            "run_dir": run_dir,
            "runtime_license_boundary": runtime_license_boundary,
            "layers": {
                "trainer_runner": "ultralytics.YOLO.train",
                "repo_impl": "support_external_training train-ultralytics",
                "export_deploy": "support_external_training export-onnx --provider ultralytics",
            },
        }
    )
    report["next_steps"] = _external_next_steps(handoff_contracts, report_path=report_path)
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
    handoff_paths = _write_handoff_contracts(work_dir=work_dir, report_path=report_path, handoff_contracts=handoff_contracts)
    registry_entry = build_training_registry_entry(summary=report, summary_path=report_path)
    registry_entry_path = work_dir / "reports" / "training_registry_entry.json"
    write_training_registry_entry(registry_entry_path, registry_entry)
    if getattr(args, "registry_out", None):
        append_training_registry(Path(str(args.registry_out)).resolve(), registry_entry)
    report["run_output_contract"]["stable_artifact_paths"] = bundled_paths
    report["run_output_contract"]["stable_artifact_paths"].update(handoff_paths)
    report["run_output_contract"]["stable_artifact_paths"]["training_registry_entry"] = str(registry_entry_path)
    report["train_config_projection"] = bundled_paths["train_config_projection"]
    report["registry_entry"] = str(registry_entry_path)
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
    resume_value = _resolve_value(
        getattr(args, "resume_from", None),
        env="YOLOZU_RESUME_FROM",
        preset=preset,
        preset_key="resume_from",
    )
    resume_path = Path(str(resume_value)).resolve() if resume_value else None
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
        if resume_path:
            command.extend(["--resume-from", str(resume_path)])
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
        if resume_path:
            command.extend(["--resume-from", str(resume_path)])

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
    handoff_contracts = _external_handoff_contracts(
        backend_id="hf-detr",
        dataset_root=Path(str(resolution.dataset_root)),
        split=str(resolution.split),
        work_dir=work_dir,
        report_path=report_path,
        model_id=str(model_id),
        task_family="bbox",
    )
    runtime_license_boundary = _optional_bridge_runtime_boundary(
        bridge_id="hf-detr",
        external_runtime="Transformers/Accelerate DETR-family runtime",
        license_note="HF DETR bridge is optional and deployment teams must review model/runtime dependency licenses separately.",
    )
    report = build_training_run_summary(
        backend_id="hf-detr",
        report_path=report_path,
        train_config=train_cfg,
        dataset_root=str(resolution.dataset_root),
        split=str(resolution.split),
        raw_dataset_format=str(resolution.source_format),
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
            "note": "HF DETR bridge is optional and deployment teams must review model/runtime dependency licenses separately.",
            "runtime_license_boundary": runtime_license_boundary,
        },
        handoff_contracts=handoff_contracts,
    )
    report["execution_status"] = _external_execution_status(
        dry_run=bool(args.dry_run),
        training_executed=training_executed,
        runtime_error=runtime_error,
        train_script=train_script or None,
        requires_train_script=True,
    )
    report.update(
        {
            "task": "train_hf_detr",
            "preset": preset_name,
            "model_id": model_id,
            "resume_from": str(resume_path) if resume_path else None,
            "train_script": train_script or None,
            "template_train_command": " ".join(command),
            "runtime_license_boundary": runtime_license_boundary,
            "layers": {
                "trainer_runner": "transformers/accelerate entry script",
                "repo_impl": "support_external_training train-hf-detr",
                "export_deploy": "support_external_training export-onnx --provider hf_detr",
            },
        }
    )
    report["next_steps"] = _external_next_steps(handoff_contracts, report_path=report_path)
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
    handoff_paths = _write_handoff_contracts(work_dir=work_dir, report_path=report_path, handoff_contracts=handoff_contracts)
    registry_entry = build_training_registry_entry(summary=report, summary_path=report_path)
    registry_entry_path = work_dir / "reports" / "training_registry_entry.json"
    write_training_registry_entry(registry_entry_path, registry_entry)
    if getattr(args, "registry_out", None):
        append_training_registry(Path(str(args.registry_out)).resolve(), registry_entry)
    report["run_output_contract"]["stable_artifact_paths"] = bundled_paths
    report["run_output_contract"]["stable_artifact_paths"].update(handoff_paths)
    report["run_output_contract"]["stable_artifact_paths"]["training_registry_entry"] = str(registry_entry_path)
    report["train_config_projection"] = bundled_paths["train_config_projection"]
    report["registry_entry"] = str(registry_entry_path)
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
    tyx.add_argument("--resume-from", default=None, help="Standardized checkpoint handoff path for resume/fine-tune.")
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
    tyx.add_argument("--registry-out", default=None, help="Optional JSONL registry path to append a training registry entry.")
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
    td2.add_argument("--resume-from", default=None, help="Standardized checkpoint handoff path for resume/fine-tune.")
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
    td2.add_argument("--registry-out", default=None, help="Optional JSONL registry path to append a training registry entry.")
    td2.add_argument("-n", "--dry-run", action="store_true", help="Do not execute runtime training.")
    td2.add_argument("-F", "--force", action="store_true", help="Overwrite generated wrapper outputs.")
    td2.set_defaults(_fn=_cmd_train_detectron2)

    tmmdet = sub.add_parser(
        "train-mmdetection",
        aliases=["tmm", "mmdet-train"],
        help="MMDetection external training bridge for bbox and instance-segmentation workflows.",
    )
    tmmdet.add_argument("-P", "--preset", choices=preset_choices, default=None, help=preset_help)
    tmmdet.add_argument("-c", "--config", required=True, help="MMDetection config Python path.")
    tmmdet.add_argument("--resume-from", default=None, help="Standardized checkpoint handoff path for resume/fine-tune.")
    tmmdet.add_argument("-t", "--train-script", default=None, help="Optional external MMDetection train launcher for non-dry execution.")
    tmmdet.add_argument("-p", "--python", default=sys.executable, help="Python executable for --train-script.")
    tmmdet.add_argument("-f", "--from", dest="from_format", default=None, choices=format_choices)
    tmmdet.add_argument("-d", "--dataset", default=None, help="Dataset root or descriptor.")
    tmmdet.add_argument("-s", "--split", default=None, help="Split for training (preset/env fallback when omitted).")
    tmmdet.add_argument("-i", "--instances-json", default=None, help="COCO instances JSON (for coco_instances mode).")
    tmmdet.add_argument("-g", "--images-dir", default=None, help="COCO images dir (for coco_instances mode).")
    tmmdet.add_argument(
        "--task-family",
        choices=("auto", "bbox", "segmentation"),
        default="auto",
        help="Reported MMDetection task family; auto infers bbox vs segmentation from the config text.",
    )
    tmmdet.add_argument(
        "--train-opt",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=[],
        help="Extra MMDetection cfg-options pair forwarded to the external launcher. Repeat for data_root/load_from/etc.",
    )
    tmmdet.add_argument("-W", "--work-dir", default="runs/support_external_training/mmdetection", help="Work/cache dir.")
    tmmdet.add_argument("-o", "--output", default="reports/support_external_training.train_mmdetection.json", help="Report JSON output path.")
    tmmdet.add_argument("--registry-out", default=None, help="Optional JSONL registry path to append a training registry entry.")
    tmmdet.add_argument("-n", "--dry-run", action="store_true", help="Do not execute runtime training.")
    tmmdet.add_argument("-F", "--force", action="store_true", help="Overwrite generated wrapper outputs.")
    tmmdet.set_defaults(_fn=_cmd_train_mmdetection)

    tmmpose = sub.add_parser(
        "train-mmpose",
        aliases=["tmp", "mmpose-train"],
        help="MMPose external training bridge for keypoints/pose workflows.",
    )
    tmmpose.add_argument("-P", "--preset", choices=preset_choices, default=None, help=preset_help)
    tmmpose.add_argument("-c", "--config", required=True, help="MMPose config Python path.")
    tmmpose.add_argument("--resume-from", default=None, help="Standardized checkpoint handoff path for resume/fine-tune.")
    tmmpose.add_argument("-t", "--train-script", default=None, help="Optional external MMPose train launcher for non-dry execution.")
    tmmpose.add_argument("-p", "--python", default=sys.executable, help="Python executable for --train-script.")
    tmmpose.add_argument("-f", "--from", dest="from_format", default=None, choices=format_choices)
    tmmpose.add_argument("-d", "--dataset", default=None, help="Dataset root or descriptor.")
    tmmpose.add_argument("-s", "--split", default=None, help="Split for training (preset/env fallback when omitted).")
    tmmpose.add_argument("-i", "--instances-json", default=None, help="COCO instances JSON (for coco_instances mode).")
    tmmpose.add_argument("-g", "--images-dir", default=None, help="COCO images dir (for coco_instances mode).")
    tmmpose.add_argument(
        "--task-family",
        choices=("keypoints",),
        default="keypoints",
        help="Reported task family for MMPose (fixed: keypoints).",
    )
    tmmpose.add_argument(
        "--train-opt",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=[],
        help="Extra MMPose cfg-options pair forwarded to the external launcher.",
    )
    tmmpose.add_argument("-W", "--work-dir", default="runs/support_external_training/mmpose", help="Work/cache dir.")
    tmmpose.add_argument("-o", "--output", default="reports/support_external_training.train_mmpose.json", help="Report JSON output path.")
    tmmpose.add_argument("--registry-out", default=None, help="Optional JSONL registry path to append a training registry entry.")
    tmmpose.add_argument("-n", "--dry-run", action="store_true", help="Do not execute runtime training.")
    tmmpose.add_argument("-F", "--force", action="store_true", help="Overwrite generated wrapper outputs.")
    tmmpose.set_defaults(_fn=_cmd_train_mmpose)

    tmmseg = sub.add_parser(
        "train-mmseg",
        aliases=["tms", "mmseg-train"],
        help="MMSeg external training bridge for semantic segmentation workflows.",
    )
    tmmseg.add_argument("-P", "--preset", choices=preset_choices, default=None, help=preset_help)
    tmmseg.add_argument("-c", "--config", required=True, help="MMSeg config Python path.")
    tmmseg.add_argument("--resume-from", default=None, help="Standardized checkpoint handoff path for resume/fine-tune.")
    tmmseg.add_argument("-t", "--train-script", default=None, help="Optional external MMSeg train launcher for non-dry execution.")
    tmmseg.add_argument("-p", "--python", default=sys.executable, help="Python executable for --train-script.")
    tmmseg.add_argument("-f", "--from", dest="from_format", default=None, choices=format_choices)
    tmmseg.add_argument("-d", "--dataset", default=None, help="Dataset root or descriptor.")
    tmmseg.add_argument("-s", "--split", default=None, help="Split for training (preset/env fallback when omitted).")
    tmmseg.add_argument("-i", "--instances-json", default=None, help="COCO instances JSON (for coco_instances mode).")
    tmmseg.add_argument("-g", "--images-dir", default=None, help="COCO images dir (for coco_instances mode).")
    tmmseg.add_argument(
        "--task-family",
        choices=("segmentation",),
        default="segmentation",
        help="Reported task family for MMSeg (fixed: segmentation).",
    )
    tmmseg.add_argument(
        "--train-opt",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=[],
        help="Extra MMSeg cfg-options pair forwarded to the external launcher.",
    )
    tmmseg.add_argument("-W", "--work-dir", default="runs/support_external_training/mmseg", help="Work/cache dir.")
    tmmseg.add_argument("-o", "--output", default="reports/support_external_training.train_mmseg.json", help="Report JSON output path.")
    tmmseg.add_argument("--registry-out", default=None, help="Optional JSONL registry path to append a training registry entry.")
    tmmseg.add_argument("-n", "--dry-run", action="store_true", help="Do not execute runtime training.")
    tmmseg.add_argument("-F", "--force", action="store_true", help="Overwrite generated wrapper outputs.")
    tmmseg.set_defaults(_fn=_cmd_train_mmseg)

    ttao = sub.add_parser(
        "train-tao",
        aliases=["ttao", "tao-train"],
        help="NVIDIA TAO external training bridge for bbox, segmentation, and keypoints workflows.",
    )
    ttao.add_argument("-P", "--preset", choices=preset_choices, default=None, help=preset_help)
    ttao.add_argument("-c", "--config", required=True, help="TAO spec/config path.")
    ttao.add_argument("--resume-from", default=None, help="Standardized checkpoint handoff path for resume/fine-tune.")
    ttao.add_argument("--tao-task", default=None, help="Optional TAO task name (for example dino, segformer, centerpose).")
    ttao.add_argument("-t", "--train-script", default=None, help="Optional external TAO launcher wrapper for non-dry execution.")
    ttao.add_argument("-p", "--python", default=sys.executable, help="Python executable for --train-script.")
    ttao.add_argument("-f", "--from", dest="from_format", default=None, choices=format_choices)
    ttao.add_argument("-d", "--dataset", default=None, help="Dataset root or descriptor.")
    ttao.add_argument("-s", "--split", default=None, help="Split for training (preset/env fallback when omitted).")
    ttao.add_argument("-i", "--instances-json", default=None, help="COCO instances JSON (for coco_instances mode).")
    ttao.add_argument("-g", "--images-dir", default=None, help="COCO images dir (for coco_instances mode).")
    ttao.add_argument(
        "--task-family",
        choices=("auto", "bbox", "segmentation", "keypoints"),
        default="auto",
        help="Reported TAO task family; auto infers from the spec text.",
    )
    ttao.add_argument(
        "--train-opt",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=[],
        help="Extra TAO wrapper option pair. Repeat for task-specific overrides.",
    )
    ttao.add_argument("-W", "--work-dir", default="runs/support_external_training/tao", help="Work/cache dir.")
    ttao.add_argument("-o", "--output", default="reports/support_external_training.train_tao.json", help="Report JSON output path.")
    ttao.add_argument("--registry-out", default=None, help="Optional JSONL registry path to append a training registry entry.")
    ttao.add_argument("-n", "--dry-run", action="store_true", help="Do not execute runtime training.")
    ttao.add_argument("-F", "--force", action="store_true", help="Overwrite generated wrapper outputs.")
    ttao.set_defaults(_fn=_cmd_train_tao)

    tul = sub.add_parser(
        "train-ultralytics",
        aliases=["tu", "ultra-train"],
        help="Optional Ultralytics YOLO bridge + normalized prediction template.",
    )
    tul.add_argument("-P", "--preset", choices=preset_choices, default=None, help=preset_help)
    tul.add_argument("-m", "--model", default=None, help="Ultralytics model path/id (e.g., yolo11n.pt).")
    tul.add_argument("--resume-from", default=None, help="Standardized checkpoint handoff path for resume/fine-tune.")
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
    tul.add_argument("--registry-out", default=None, help="Optional JSONL registry path to append a training registry entry.")
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
    thf.add_argument("--resume-from", default=None, help="Standardized checkpoint handoff path for resume/fine-tune.")
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
    thf.add_argument("--registry-out", default=None, help="Optional JSONL registry path to append a training registry entry.")
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
