#!/usr/bin/env python3
"""Audit adapter/export support across external detection backends.

Runs exporters for:
- YOLOX
- YOLOv8 (Ultralytics)
- Detectron2
- MMDetection

The default dry-run validates the interface contract path (CLI + schema-valid
predictions artifact) without heavyweight framework installs. Selected
backends can instead be required to provide verified runtime execution
evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REQUIRED_TASKS = ("bbox", "segmentation", "keypoints", "depth", "pose6d")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit YOLOX/YOLOv8/Detectron2/MMDetection export support.")
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
        help="Backend to attempt without --dry-run (repeatable).",
    )
    p.add_argument(
        "--require-non-dry",
        action="store_true",
        help="Fail unless at least one backend supplies verified non-dry execution evidence.",
    )
    p.add_argument("--yolox-exp", default=None, help="YOLOX exp file required for a non-dry YOLOX audit.")
    p.add_argument(
        "--yolox-weights",
        default=None,
        help="YOLOX checkpoint required for a non-dry YOLOX audit.",
    )
    p.add_argument(
        "--ultralytics-model",
        default="yolov8n.pt",
        help="Ultralytics model path or name (default: yolov8n.pt).",
    )
    p.add_argument(
        "--detectron2-config",
        default=None,
        help="Detectron2 config required for a non-dry Detectron2 audit.",
    )
    p.add_argument(
        "--detectron2-weights",
        default=None,
        help="Detectron2 weights required for a non-dry Detectron2 audit.",
    )
    p.add_argument(
        "--mmdet-config",
        default=None,
        help="MMDetection config required for a non-dry MMDetection audit.",
    )
    p.add_argument(
        "--mmdet-checkpoint",
        default=None,
        help="MMDetection checkpoint required for a non-dry MMDetection audit.",
    )
    return p


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)


def _load_predictions_artifact(path: Path) -> tuple[int, dict[str, Any] | None, str | None]:
    if not path.exists():
        return 0, None, "output_missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return 0, None, f"invalid_json:{exc}"

    if isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
        meta = payload.get("meta")
        return int(len(payload["predictions"])), (meta if isinstance(meta, dict) else None), None
    if isinstance(payload, list):
        return int(len(payload)), None, None
    return 0, None, "unsupported_payload_shape"


def _validate_execution_evidence(
    *,
    meta: dict[str, Any] | None,
    dry_run: bool,
    predictions_count: int | None = None,
    expected_exporter: str | None = None,
) -> str | None:
    if not isinstance(meta, dict):
        return "execution_evidence_missing"
    extra = meta.get("extra")
    if not isinstance(extra, dict):
        return "execution_evidence_missing"

    artifact_exporter = str(extra.get("exporter") or "")
    if expected_exporter is not None and artifact_exporter != expected_exporter:
        return f"exporter_mismatch:{artifact_exporter!r}"

    if not isinstance(extra.get("dry_run"), bool) or extra["dry_run"] is not dry_run:
        return "dry_run_metadata_mismatch"
    expected_status = "dry_run" if dry_run else "completed"
    if str(extra.get("execution_status") or "") != expected_status:
        return f"execution_status_mismatch:{extra.get('execution_status')!r}"
    runtime_executed = extra.get("runtime_executed")
    if not isinstance(runtime_executed, bool) or runtime_executed is dry_run:
        return "runtime_execution_metadata_mismatch"

    raw_inference_calls = extra.get("inference_calls")
    try:
        inference_calls = int(raw_inference_calls or 0)
    except (TypeError, ValueError):
        return "inference_calls_invalid"

    if artifact_exporter == "ultralytics":
        if isinstance(raw_inference_calls, bool) or not isinstance(raw_inference_calls, int):
            return "inference_calls_invalid"
        selected_input_count = extra.get("selected_input_count")
        result_count = extra.get("result_count")
        if (
            isinstance(selected_input_count, bool)
            or not isinstance(selected_input_count, int)
            or selected_input_count <= 0
        ):
            return "selected_input_count_invalid"
        selected_inputs = extra.get("selected_inputs")
        if not isinstance(selected_inputs, list) or len(selected_inputs) != selected_input_count:
            return "selected_inputs_mismatch"
        if any(not isinstance(value, str) or not value for value in selected_inputs):
            return "selected_inputs_invalid"
        if predictions_count is None or predictions_count != selected_input_count:
            return "prediction_count_mismatch"
        if isinstance(result_count, bool) or not isinstance(result_count, int) or result_count < 0:
            return "result_count_invalid"
        if dry_run:
            if result_count != 0:
                return "dry_run_recorded_results"
        elif result_count != selected_input_count or inference_calls != selected_input_count:
            return "runtime_cardinality_mismatch"

    if dry_run:
        if inference_calls != 0:
            return "dry_run_recorded_inference_calls"
        return None

    if inference_calls <= 0:
        return "non_dry_inference_calls_missing"
    if extra.get("runtime_error") not in (None, ""):
        return "non_dry_runtime_error_present"
    provenance = extra.get("model_provenance")
    if not isinstance(provenance, dict) or not any(value not in (None, "") for value in provenance.values()):
        return "model_provenance_missing"
    return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _build_multitask_coverage(repo_root: Path) -> dict[str, Any]:
    gaps: list[dict[str, str]] = []
    components: dict[str, Any] = {}

    demo_script = repo_root / "tools" / "run_real_multitask_finetune_demo.py"
    demo_text = _read_text(demo_script)
    def _task_in_matrix(task_name: str) -> bool:
        pattern = rf'["\']{re.escape(task_name)}["\']\s*,'
        return bool(re.search(pattern, demo_text))

    training_map = {task: _task_in_matrix(task) for task in REQUIRED_TASKS}
    components["training"] = {
        "task_matrix": training_map,
        "strict_task_data_flag": "--strict-task-data" in _read_text(repo_root / "rtdetr_pose" / "tools" / "train_minimal.py"),
    }
    for task, supported in training_map.items():
        if not supported:
            gaps.append({"severity": "error", "component": "training", "task": task, "reason": "missing_task_stage"})

    pred_transform = _read_text(repo_root / "yolozu" / "predictions" / "predictions_transform.py")
    inference_map = {
        "bbox": '"bbox": 0' in pred_transform,
        "segmentation": '"segmentation": 0' in pred_transform,
        "keypoints": '"keypoints": 0' in pred_transform,
        "depth": '"depth": 0' in pred_transform,
        "pose6d": '"pose6d": 0' in pred_transform,
    }
    components["inference"] = {
        "task_coverage_summary": inference_map,
        "ttt_lite_postprocess": "def apply_ttt_lite(" in pred_transform,
        "score_fusion_postprocess": "def fuse_detection_scores(" in pred_transform,
    }
    for task, supported in inference_map.items():
        if not supported:
            gaps.append({"severity": "error", "component": "inference", "task": task, "reason": "missing_task_probe"})

    regression_text = _read_text(repo_root / "tools" / "run_reference_adapter_regression.py")
    components["prediction_guardrails"] = {
        "canonicalize_predictions": "canonicalize_predictions(" in regression_text,
        "schema_gate_flag": "--schema-gate-mode" in regression_text,
        "consistency_gate_flag": "--consistency-gate-mode" in regression_text,
        "score_gate_flag": "--score-gate-mode" in regression_text,
        "perf_gate_flag": "--perf-gate-mode" in regression_text,
    }
    for key, supported in (components.get("prediction_guardrails") or {}).items():
        if not bool(supported):
            gaps.append({"severity": "error", "component": "prediction_guardrails", "task": "all", "reason": f"missing_{key}"})

    eval_files = {
        "bbox": [repo_root / "tools" / "eval_coco.py", repo_root / "tools" / "eval_suite.py"],
        "segmentation": [repo_root / "tools" / "eval_segmentation.py", repo_root / "tools" / "eval_instance_segmentation.py"],
        "keypoints": [repo_root / "tools" / "eval_keypoints.py"],
        "depth": [repo_root / "tools" / "eval_synthgen.py"],
        "pose6d": [repo_root / "tools" / "eval_pose.py"],
    }
    eval_map: dict[str, dict[str, Any]] = {}
    for task, files in eval_files.items():
        existing = [str(p.relative_to(repo_root)) for p in files if p.exists()]
        status = "supported" if existing else "missing"
        if task == "depth" and existing:
            status = "partial"
        eval_map[task] = {"status": status, "paths": existing}
        if status == "missing":
            gaps.append({"severity": "error", "component": "eval", "task": task, "reason": "missing_eval_tool"})
        if status == "partial":
            gaps.append(
                {
                    "severity": "warning",
                    "component": "eval",
                    "task": task,
                    "reason": "depth_eval_is_synthgen_specialized_only",
                }
            )
    components["eval"] = eval_map

    errors = [g for g in gaps if str(g.get("severity")) == "error"]
    warnings = [g for g in gaps if str(g.get("severity")) == "warning"]
    supported_task_count = sum(
        1 for task in REQUIRED_TASKS if bool(training_map.get(task)) and bool(inference_map.get(task))
    )
    return {
        "required_tasks": list(REQUIRED_TASKS),
        "supported_task_count": int(supported_task_count),
        "components": components,
        "gaps": gaps,
        "ok": bool(not errors),
        "error_count": int(len(errors)),
        "warning_count": int(len(warnings)),
    }


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
                *(["--exp", str(args.yolox_exp)] if args.yolox_exp else []),
                *(["--weights", str(args.yolox_weights)] if args.yolox_weights else []),
                *strict_flags,
            ],
        },
        {
            "backend": "yolov8_ultralytics",
            "command": [
                str(args.python),
                "tools/export_predictions_ultralytics.py",
                "--model",
                str(args.ultralytics_model),
                "--dataset",
                str(dataset_root),
                "--split",
                str(args.split),
                "--max-images",
                str(int(args.max_images)),
                "--output",
                str(work_dir / "predictions_yolov8.json"),
                "--wrap",
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
                str(args.detectron2_config or "configs/detectron2_stub.yaml"),
                "--weights",
                str(args.detectron2_weights or "weights/detectron2_stub.pth"),
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
                str(args.mmdet_config or "configs/mmdet_stub.py"),
                "--checkpoint",
                str(args.mmdet_checkpoint or "weights/mmdet_stub.pth"),
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

        out_file = None
        if "--output" in cmd:
            try:
                out_file = Path(cmd[cmd.index("--output") + 1]).resolve()
            except Exception:
                out_file = None
        if out_file is not None and out_file.exists():
            out_file.unlink()

        proc = _run(cmd, cwd=repo_root)

        preds_count = None
        artifact_meta = None
        output_error = None
        execution_evidence_error = None
        if out_file is not None:
            preds_count, artifact_meta, output_error = _load_predictions_artifact(out_file)
            if output_error is None:
                execution_evidence_error = _validate_execution_evidence(
                    meta=artifact_meta,
                    dry_run=bool(dry_run),
                    predictions_count=preds_count,
                    expected_exporter=("ultralytics" if backend == "yolov8_ultralytics" else None),
                )

        results.append(
            {
                "backend": backend,
                "ok": bool(
                    proc.returncode == 0
                    and output_error is None
                    and execution_evidence_error is None
                ),
                "returncode": int(proc.returncode),
                "predictions_file": (str(out_file) if out_file is not None else None),
                "predictions_count": preds_count,
                "output_error": output_error,
                "execution_evidence_error": execution_evidence_error,
                "stdout_tail": (proc.stdout or "").splitlines()[-10:],
                "stderr_tail": (proc.stderr or "").splitlines()[-10:],
                "command": cmd,
                "dry_run": bool(dry_run),
            }
        )

    non_dry_count = sum(1 for item in results if not bool(item.get("dry_run", True)))
    verified_non_dry = sorted(
        str(item.get("backend"))
        for item in results
        if not bool(item.get("dry_run", True)) and bool(item.get("ok"))
    )
    multitask_coverage = _build_multitask_coverage(repo_root)
    warnings: list[str] = []
    if non_dry_count <= 0:
        warnings.append("all backends were executed in dry-run mode")
    ok = all(bool(item.get("ok")) for item in results)
    if not bool(multitask_coverage.get("ok", False)):
        ok = False
        warnings.append("multitask coverage audit found missing hard requirements")
    if bool(args.require_non_dry) and not verified_non_dry:
        ok = False
        warnings.append("--require-non-dry is set but no backend supplied verified runtime execution evidence")
    report = {
        "task": "backend_support_audit",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ok": bool(ok),
        "dataset_root": str(dataset_root),
        "split": str(args.split),
        "max_images": int(args.max_images),
        "warnings": warnings,
        "non_dry_backends": sorted(non_dry),
        "verified_non_dry_backends": verified_non_dry,
        "results": results,
        "multitask_coverage": multitask_coverage,
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(str(out_path))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
