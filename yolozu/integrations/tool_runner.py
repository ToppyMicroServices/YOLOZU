from __future__ import annotations

from functools import lru_cache
import uuid
from pathlib import Path
from typing import Any

from .layers.api import run_cli_tool, run_cli_tool_redacted
from .layers.artifacts import collect_artifact_metadata, describe_run, list_runs
from .layers.core import fail_response
from .layers.jobs import JobManager
from .manifest_resources import resolve_workspace_path

_SCENARIO_EXTRA_VALUE_FLAGS = frozenset(
    {
        "--adapter",
        "--checkpoint",
        "--config",
        "--dataset",
        "--device",
        "--max-detections",
        "--max-images",
        "--output",
        "--predictions",
        "--score-threshold",
        "--split",
    }
)


@lru_cache(maxsize=1)
def _job_manager() -> JobManager:
    """Create persistent job storage only when a job operation is requested."""
    return JobManager()


_DEFAULT_TTT_CONFIG = "builtin:base"
_TTT_METHODS = {"tent", "mim", "cotta", "eata", "sar"}


def _workspace_path(value: str, *, label: str, kind: str) -> Path:
    token = str(value).strip()
    if not token:
        raise ValueError(f"{label} is required")
    path = Path(token)
    if path.is_absolute():
        raise ValueError(f"{label} must be workspace-relative")
    if kind not in {"file", "dir", "output"}:
        raise ValueError(f"unsupported path kind: {kind}")
    return resolve_workspace_path(path)


def _with_meta(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("meta", collect_artifact_metadata())
    return payload


def _validated_scenario_extra_args(
    name: str,
    extra_args: list[str] | None,
) -> tuple[list[str], dict[str, Any] | None]:
    """Accept only declared one-value scenario flags on AI-facing surfaces."""
    tokens = list(extra_args or [])
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--") and "=" in token:
            flag, value = token.split("=", 1)
            if flag not in _SCENARIO_EXTRA_VALUE_FLAGS or not value:
                error = ValueError(
                    f"extra_args contains undeclared or empty flag: {flag}"
                )
                return [], _with_meta(
                    fail_response(name, message=str(error), exc=error)
                )
            index += 1
            continue
        if token not in _SCENARIO_EXTRA_VALUE_FLAGS:
            error = ValueError(
                f"extra_args contains undeclared flag: {token}"
            )
            return [], _with_meta(
                fail_response(name, message=str(error), exc=error)
            )
        if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
            error = ValueError(
                f"extra_args flag requires one value: {token}"
            )
            return [], _with_meta(
                fail_response(name, message=str(error), exc=error)
            )
        index += 2
    return tokens, None


def doctor(*, output: str = "reports/doctor.json") -> dict[str, Any]:
    return _with_meta(run_cli_tool("doctor", ["doctor", "--output", output], artifacts={"doctor": output}))


def doctor_public(*, output: str = "reports/doctor.json") -> dict[str, Any]:
    return _with_meta(run_cli_tool_redacted("doctor", ["doctor", "--output", output], artifacts={"doctor": output}))


def validate_predictions(path: str, *, strict: bool = True) -> dict[str, Any]:
    """Validate strictly, or explicitly request repair with ``strict=False``."""
    args = ["validate", "predictions", path, "--json"]
    if strict:
        args.append("--strict")
    return _with_meta(
        run_cli_tool(
            "validate_predictions",
            args,
            json_result_key="validation",
        )
    )


def validate_predictions_public(path: str, *, strict: bool = True) -> dict[str, Any]:
    """Public response variant with the same strict/repair semantics."""
    args = ["validate", "predictions", path, "--json"]
    if strict:
        args.append("--strict")
    return _with_meta(
        run_cli_tool_redacted(
            "validate_predictions",
            args,
            json_result_key="validation",
        )
    )


def validate_dataset(
    dataset: str,
    *,
    split: str | None = None,
    strict: bool = True,
    mode: str = "fail",
) -> dict[str, Any]:
    args = ["validate", "dataset", dataset, "--mode", mode]
    if split:
        args.extend(["--split", split])
    if strict:
        args.append("--strict")
    return _with_meta(run_cli_tool("validate_dataset", args))


def validate_dataset_public(
    dataset: str,
    *,
    split: str | None = None,
    strict: bool = True,
    mode: str = "fail",
) -> dict[str, Any]:
    args = ["validate", "dataset", dataset, "--mode", mode]
    if split:
        args.extend(["--split", split])
    if strict:
        args.append("--strict")
    return _with_meta(run_cli_tool_redacted("validate_dataset", args))


def eval_coco(
    dataset: str,
    predictions: str,
    *,
    split: str | None = None,
    dry_run: bool = True,
    output: str = "reports/mcp_coco_eval.json",
    max_images: int | None = None,
    repair: bool = False,
) -> dict[str, Any]:
    args = [
        "eval-coco",
        "--dataset",
        dataset,
        "--predictions",
        predictions,
        "--output",
        output,
    ]
    if split:
        args.extend(["--split", split])
    if dry_run:
        args.append("--dry-run")
    if max_images is not None:
        args.extend(["--max-images", str(max_images)])
    if repair:
        args.append("--repair")
    return _with_meta(run_cli_tool("eval_coco", args, artifacts={"report": output}))


def eval_coco_public(
    dataset: str,
    predictions: str,
    *,
    split: str | None = None,
    dry_run: bool = True,
    output: str = "reports/mcp_coco_eval.json",
    max_images: int | None = None,
    repair: bool = False,
) -> dict[str, Any]:
    args = [
        "eval-coco",
        "--dataset",
        dataset,
        "--predictions",
        predictions,
        "--output",
        output,
    ]
    if split:
        args.extend(["--split", split])
    if dry_run:
        args.append("--dry-run")
    if max_images is not None:
        args.extend(["--max-images", str(max_images)])
    if repair:
        args.append("--repair")
    return _with_meta(run_cli_tool_redacted("eval_coco", args, artifacts={"report": output}))


def predict_images(
    input_dir: str,
    *,
    backend: str = "dummy",
    output: str = "reports/mcp_predict_images.json",
    max_images: int | None = None,
    dry_run: bool = True,
    strict: bool = True,
    force: bool = True,
) -> dict[str, Any]:
    args = [
        "predict-images",
        "--backend",
        backend,
        "--input-dir",
        input_dir,
        "--output",
        output,
    ]
    if max_images is not None:
        args.extend(["--max-images", str(max_images)])
    if dry_run:
        args.append("--dry-run")
    if strict:
        args.append("--strict")
    if force:
        args.append("--force")
    return _with_meta(run_cli_tool("predict_images", args, artifacts={"predictions": output}))


def predict_images_public(
    input_dir: str,
    *,
    backend: str = "dummy",
    output: str = "reports/mcp_predict_images.json",
    max_images: int | None = None,
    dry_run: bool = True,
    strict: bool = True,
    force: bool = True,
) -> dict[str, Any]:
    args = [
        "predict-images",
        "--backend",
        backend,
        "--input-dir",
        input_dir,
        "--output",
        output,
    ]
    if max_images is not None:
        args.extend(["--max-images", str(max_images)])
    if dry_run:
        args.append("--dry-run")
    if strict:
        args.append("--strict")
    if force:
        args.append("--force")
    return _with_meta(run_cli_tool_redacted("predict_images", args, artifacts={"predictions": output}))


def parity_check(
    reference: str,
    candidate: str,
    *,
    iou_thresh: float = 0.5,
    score_atol: float = 1e-6,
    bbox_atol: float = 1e-4,
    max_images: int | None = None,
    image_size: str | None = None,
) -> dict[str, Any]:
    args = [
        "parity",
        "--reference",
        reference,
        "--candidate",
        candidate,
        "--iou-thresh",
        str(iou_thresh),
        "--score-atol",
        str(score_atol),
        "--bbox-atol",
        str(bbox_atol),
    ]
    if max_images is not None:
        args.extend(["--max-images", str(max_images)])
    if image_size:
        args.extend(["--image-size", image_size])
    return _with_meta(run_cli_tool("parity_check", args))


def parity_check_public(
    reference: str,
    candidate: str,
    *,
    iou_thresh: float = 0.5,
    score_atol: float = 1e-6,
    bbox_atol: float = 1e-4,
    max_images: int | None = None,
    image_size: str | None = None,
) -> dict[str, Any]:
    args = [
        "parity",
        "--reference",
        reference,
        "--candidate",
        candidate,
        "--iou-thresh",
        str(iou_thresh),
        "--score-atol",
        str(score_atol),
        "--bbox-atol",
        str(bbox_atol),
    ]
    if max_images is not None:
        args.extend(["--max-images", str(max_images)])
    if image_size:
        args.extend(["--image-size", image_size])
    return _with_meta(run_cli_tool_redacted("parity_check", args))


def calibrate_predictions(
    dataset: str,
    predictions: str,
    *,
    method: str = "fracal",
    split: str | None = None,
    task: str = "auto",
    output: str = "reports/mcp_calibrated_predictions.json",
    output_report: str = "reports/mcp_calibration_report.json",
    max_images: int | None = None,
    force: bool = True,
) -> dict[str, Any]:
    args = [
        "calibrate",
        "--method",
        method,
        "--dataset",
        dataset,
        "--task",
        task,
        "--predictions",
        predictions,
        "--output",
        output,
        "--output-report",
        output_report,
    ]
    if split:
        args.extend(["--split", split])
    if max_images is not None:
        args.extend(["--max-images", str(max_images)])
    if force:
        args.append("--force")
    return _with_meta(
        run_cli_tool(
            "calibrate_predictions",
            args,
            artifacts={"predictions": output, "report": output_report},
        )
    )


def calibrate_predictions_public(
    dataset: str,
    predictions: str,
    *,
    method: str = "fracal",
    split: str | None = None,
    task: str = "auto",
    output: str = "reports/mcp_calibrated_predictions.json",
    output_report: str = "reports/mcp_calibration_report.json",
    max_images: int | None = None,
    force: bool = True,
) -> dict[str, Any]:
    args = [
        "calibrate",
        "--method",
        method,
        "--dataset",
        dataset,
        "--task",
        task,
        "--predictions",
        predictions,
        "--output",
        output,
        "--output-report",
        output_report,
    ]
    if split:
        args.extend(["--split", split])
    if max_images is not None:
        args.extend(["--max-images", str(max_images)])
    if force:
        args.append("--force")
    return _with_meta(
        run_cli_tool_redacted(
            "calibrate_predictions",
            args,
            artifacts={"predictions": output, "report": output_report},
        )
    )


def eval_instance_seg(
    dataset: str,
    predictions: str,
    *,
    split: str | None = None,
    output: str = "reports/mcp_instance_seg_eval.json",
    max_images: int | None = None,
    min_score: float | None = None,
    allow_rgb_masks: bool = False,
) -> dict[str, Any]:
    args = [
        "eval-instance-seg",
        "--dataset",
        dataset,
        "--predictions",
        predictions,
        "--output",
        output,
    ]
    if split:
        args.extend(["--split", split])
    if max_images is not None:
        args.extend(["--max-images", str(max_images)])
    if min_score is not None:
        args.extend(["--min-score", str(min_score)])
    if allow_rgb_masks:
        args.append("--allow-rgb-masks")
    return _with_meta(run_cli_tool("eval_instance_seg", args, artifacts={"report": output}))


def eval_instance_seg_public(
    dataset: str,
    predictions: str,
    *,
    split: str | None = None,
    output: str = "reports/mcp_instance_seg_eval.json",
    max_images: int | None = None,
    min_score: float | None = None,
    allow_rgb_masks: bool = False,
) -> dict[str, Any]:
    args = [
        "eval-instance-seg",
        "--dataset",
        dataset,
        "--predictions",
        predictions,
        "--output",
        output,
    ]
    if split:
        args.extend(["--split", split])
    if max_images is not None:
        args.extend(["--max-images", str(max_images)])
    if min_score is not None:
        args.extend(["--min-score", str(min_score)])
    if allow_rgb_masks:
        args.append("--allow-rgb-masks")
    return _with_meta(run_cli_tool_redacted("eval_instance_seg", args, artifacts={"report": output}))


def eval_long_tail(
    dataset: str,
    predictions: str,
    *,
    split: str | None = None,
    output: str = "reports/mcp_long_tail_eval.json",
    max_images: int | None = None,
    max_detections: int | None = None,
) -> dict[str, Any]:
    args = [
        "eval-long-tail",
        "--dataset",
        dataset,
        "--predictions",
        predictions,
        "--output",
        output,
    ]
    if split:
        args.extend(["--split", split])
    if max_images is not None:
        args.extend(["--max-images", str(max_images)])
    if max_detections is not None:
        args.extend(["--max-detections", str(max_detections)])
    return _with_meta(run_cli_tool("eval_long_tail", args, artifacts={"report": output}))


def eval_long_tail_public(
    dataset: str,
    predictions: str,
    *,
    split: str | None = None,
    output: str = "reports/mcp_long_tail_eval.json",
    max_images: int | None = None,
    max_detections: int | None = None,
) -> dict[str, Any]:
    args = [
        "eval-long-tail",
        "--dataset",
        dataset,
        "--predictions",
        predictions,
        "--output",
        output,
    ]
    if split:
        args.extend(["--split", split])
    if max_images is not None:
        args.extend(["--max-images", str(max_images)])
    if max_detections is not None:
        args.extend(["--max-detections", str(max_detections)])
    return _with_meta(run_cli_tool_redacted("eval_long_tail", args, artifacts={"report": output}))


def run_scenarios(config: str, *, extra_args: list[str] | None = None) -> dict[str, Any]:
    safe_extra_args, rejected = _validated_scenario_extra_args(
        "run_scenarios",
        extra_args,
    )
    if rejected is not None:
        return rejected
    args = ["test", config, *safe_extra_args]
    return _with_meta(run_cli_tool("run_scenarios", args))


def run_scenarios_public(config: str, *, extra_args: list[str] | None = None) -> dict[str, Any]:
    safe_extra_args, rejected = _validated_scenario_extra_args(
        "run_scenarios",
        extra_args,
    )
    if rejected is not None:
        return rejected
    args = ["test", config, *safe_extra_args]
    return _with_meta(run_cli_tool_redacted("run_scenarios", args))


def convert_dataset(
    from_format: str,
    output: str,
    *,
    data: str | None = None,
    args_yaml: str | None = None,
    split: str | None = None,
    task: str | None = None,
    coco_root: str | None = None,
    instances_json: str | None = None,
    mode: str = "manifest",
    include_crowd: bool = False,
    force: bool = True,
) -> dict[str, Any]:
    args = ["migrate", "dataset", "--from", from_format, "--output", output, "--mode", mode]
    if data:
        args.extend(["--data", data])
    if args_yaml:
        args.extend(["--args", args_yaml])
    if split:
        args.extend(["--split", split])
    if task:
        args.extend(["--task", task])
    if coco_root:
        args.extend(["--coco-root", coco_root])
    if instances_json:
        args.extend(["--instances-json", instances_json])
    if include_crowd:
        args.append("--include-crowd")
    if force:
        args.append("--force")
    return _with_meta(run_cli_tool("convert_dataset", args))


def convert_dataset_public(
    from_format: str,
    output: str,
    *,
    data: str | None = None,
    args_yaml: str | None = None,
    split: str | None = None,
    task: str | None = None,
    coco_root: str | None = None,
    instances_json: str | None = None,
    mode: str = "manifest",
    include_crowd: bool = False,
    force: bool = True,
) -> dict[str, Any]:
    args = ["migrate", "dataset", "--from", from_format, "--output", output, "--mode", mode]
    if data:
        args.extend(["--data", data])
    if args_yaml:
        args.extend(["--args", args_yaml])
    if split:
        args.extend(["--split", split])
    if task:
        args.extend(["--task", task])
    if coco_root:
        args.extend(["--coco-root", coco_root])
    if instances_json:
        args.extend(["--instances-json", instances_json])
    if include_crowd:
        args.append("--include-crowd")
    if force:
        args.append("--force")
    return _with_meta(run_cli_tool_redacted("convert_dataset", args))


def submit_job(name: str, args: list[str], *, artifacts: dict[str, str] | None = None) -> dict[str, Any]:
    job_id = _job_manager().submit(
        name,
        lambda: _with_meta(
            run_cli_tool(name, args, artifacts=artifacts)
        ),
    )
    return {
        "ok": True,
        "tool": "jobs.submit",
        "summary": f"job queued: {job_id}",
        "exit_code": 0,
        "job_id": job_id,
        "status": "queued",
        "meta": collect_artifact_metadata(),
    }


def submit_job_public(name: str, args: list[str], *, artifacts: dict[str, str] | None = None) -> dict[str, Any]:
    job_id = _job_manager().submit(
        name,
        lambda: _with_meta(
            run_cli_tool_redacted(name, args, artifacts=artifacts)
        ),
    )
    return {
        "ok": True,
        "tool": "jobs.submit",
        "summary": f"job queued: {job_id}",
        "exit_code": 0,
        "job_id": job_id,
        "status": "queued",
        "meta": collect_artifact_metadata(),
    }


def jobs_list() -> dict[str, Any]:
    return {
        "ok": True,
        "tool": "jobs.list",
        "summary": "listed jobs",
        "exit_code": 0,
        "jobs": _job_manager().list(),
        "meta": collect_artifact_metadata(),
    }


def jobs_status(job_id: str) -> dict[str, Any]:
    status = _job_manager().status(job_id)
    if status is None:
        return {
            "ok": False,
            "tool": "jobs.status",
            "summary": "job not found",
            "exit_code": 1,
            "job_id": job_id,
            "meta": collect_artifact_metadata(),
        }
    failed = status.get("status") == "failed"
    error = str(status.get("error") or "job failed") if failed else None
    summary = (
        f"job failed: {error}"
        if failed
        else f"job status: {status.get('status')}"
    )
    return {
        "ok": not failed,
        "tool": "jobs.status",
        "summary": summary,
        "exit_code": 1 if failed else 0,
        "job": status,
        **({"error": error} if error is not None else {}),
        "meta": collect_artifact_metadata(),
    }


def jobs_cancel(job_id: str) -> dict[str, Any]:
    out = _job_manager().cancel(job_id)
    if out is None:
        return {
            "ok": False,
            "tool": "jobs.cancel",
            "summary": "job not found",
            "exit_code": 1,
            "job_id": job_id,
            "meta": collect_artifact_metadata(),
        }
    cancelled = bool(out.get("cancelled"))
    return {
        "ok": cancelled,
        "tool": "jobs.cancel",
        "summary": f"cancelled={out.get('cancelled')}",
        "exit_code": 0 if cancelled else 1,
        **out,
        "meta": collect_artifact_metadata(),
    }


def runs_list(limit: int = 20) -> dict[str, Any]:
    return {
        "ok": True,
        "tool": "runs.list",
        "summary": "listed runs",
        "exit_code": 0,
        "runs": list_runs(limit=limit),
        "meta": collect_artifact_metadata(),
    }


def runs_describe(run_id: str) -> dict[str, Any]:
    details = describe_run(run_id)
    if details is None:
        return {
            "ok": False,
            "tool": "runs.describe",
            "summary": "run not found",
            "exit_code": 1,
            "run_id": run_id,
            "meta": collect_artifact_metadata(),
        }
    return {
        "ok": True,
        "tool": "runs.describe",
        "summary": "run described",
        "exit_code": 0,
        "run": details,
        "meta": collect_artifact_metadata(),
    }


def train_job(train_config: str, *, run_id: str | None = None, resume: str | None = None) -> dict[str, Any]:
    args = ["train", train_config]
    if run_id:
        args.extend(["--run-id", run_id])
    if resume:
        args.extend(["--resume", resume])
    return submit_job("train", args)


def train_job_public(train_config: str, *, run_id: str | None = None, resume: str | None = None) -> dict[str, Any]:
    args = ["train", train_config]
    if run_id:
        args.extend(["--run-id", run_id])
    if resume:
        args.extend(["--resume", resume])
    return submit_job_public("train", args)


def export_predictions_job(dataset: str, output: str, *, split: str | None = None, force: bool = True) -> dict[str, Any]:
    args = ["export", "--backend", "labels", "--dataset", dataset, "--output", output]
    if split:
        args.extend(["--split", split])
    if force:
        args.append("--force")
    return submit_job("export_predictions", args, artifacts={"predictions": output})


def export_predictions_job_public(dataset: str, output: str, *, split: str | None = None, force: bool = True) -> dict[str, Any]:
    args = ["export", "--backend", "labels", "--dataset", dataset, "--output", output]
    if split:
        args.extend(["--split", split])
    if force:
        args.append("--force")
    return submit_job_public("export_predictions", args, artifacts={"predictions": output})


def export_onnx_job(dataset: str, output: str, *, split: str | None = None, force: bool = True) -> dict[str, Any]:
    return export_predictions_job(dataset=dataset, output=output, split=split, force=force)


def export_onnx_job_public(dataset: str, output: str, *, split: str | None = None, force: bool = True) -> dict[str, Any]:
    return export_predictions_job_public(dataset=dataset, output=output, split=split, force=force)


def test_job(test_config: str, *, extra_args: list[str] | None = None) -> dict[str, Any]:
    safe_extra_args, rejected = _validated_scenario_extra_args(
        "test",
        extra_args,
    )
    if rejected is not None:
        return rejected
    args = ["test", test_config, *safe_extra_args]
    return submit_job("test", args)


def test_job_public(test_config: str, *, extra_args: list[str] | None = None) -> dict[str, Any]:
    safe_extra_args, rejected = _validated_scenario_extra_args(
        "test",
        extra_args,
    )
    if rejected is not None:
        return rejected
    args = ["test", test_config, *safe_extra_args]
    return submit_job_public("test", args)


def _ttt_export_job(
    *,
    job_name: str,
    dataset: str,
    checkpoint: str,
    output: str | None,
    config: str,
    split: str,
    report: str | None,
    method: str,
    preset: str | None,
    steps: int,
    reset: str,
    device: str,
    max_images: int,
    force: bool,
    public: bool,
) -> dict[str, Any]:
    run_token = uuid.uuid4().hex[:12]
    run_root = Path("runs") / f"mcp_{job_name}" / run_token
    output_path = str(output or (run_root / "predictions.json"))
    report_path = str(report or (run_root / "ttt_report.json"))
    try:
        method_value = str(method).strip().lower()
        if method_value not in _TTT_METHODS:
            raise ValueError(
                f"method must be one of: {', '.join(sorted(_TTT_METHODS))}"
            )
        if str(reset) not in {"sample", "stream"}:
            raise ValueError("reset must be 'sample' or 'stream'")
        if int(steps) < 1:
            raise ValueError("steps must be >= 1")
        if int(max_images) < 1:
            raise ValueError("max_images must be >= 1")
        dataset_path = _workspace_path(dataset, label="dataset", kind="dir")
        _workspace_path(
            checkpoint, label="checkpoint", kind="file"
        )
        if not str(config).startswith(("builtin:", "pkg:")):
            _workspace_path(config, label="config", kind="file")
        from yolozu.dataset import build_manifest

        dataset_manifest = build_manifest(str(dataset_path), split=str(split))
        if not list(dataset_manifest.get("images") or []):
            raise ValueError(
                f"dataset split has no images: dataset={dataset!r}, split={split!r}"
            )
        output_resolved = _workspace_path(
            output_path, label="output", kind="output"
        )
        report_resolved = _workspace_path(
            report_path, label="report", kind="output"
        )
        if output_resolved == report_resolved:
            raise ValueError(
                "output and report must be different workspace-relative paths"
            )

        from yolozu.adapter import RTDETRPoseAdapter

        adapter = RTDETRPoseAdapter(
            config_path=config,
            checkpoint_path=checkpoint,
            device="cpu",
            image_size=(32, 32),
        )
        model = adapter.get_model()
        checkpoint_report = adapter.get_checkpoint_report()
        if not isinstance(checkpoint_report, dict):
            raise RuntimeError("checkpoint preflight produced no compatibility report")
        if str(checkpoint_report.get("status") or "") != "full":
            raise RuntimeError(
                "checkpoint must be fully compatible with the selected RT-DETR config"
            )
        if not bool((checkpoint_report.get("load") or {}).get("loaded")):
            raise RuntimeError("checkpoint preflight did not confirm model loading")
        if method_value == "mim":
            from yolozu.tta.ttt_mim import supports_structured_mim

            if not supports_structured_mim(model):
                raise RuntimeError(
                    "MIM requires a config/checkpoint with the structured MIM hook"
                )
        preflight = {
            "status": "full",
            "config": config,
            "checkpoint_sha256": (
                (checkpoint_report.get("checkpoint") or {}).get("sha256")
            ),
            "model_class": (
                (checkpoint_report.get("model") or {}).get("class")
            ),
        }
    except Exception as exc:
        payload = fail_response(
            job_name,
            message=(
                "TTT job prerequisite check failed before queueing: "
                f"{exc}. Provide a checkpoint fully compatible with config={config!r}."
            ),
            exit_code=2,
            exc=exc,
        )
        payload["stage"] = "preflight"
        payload["queued"] = False
        return payload

    args = [
        "export",
        "--backend",
        "torch",
        "--dataset",
        dataset,
        "--split",
        split,
        "--config",
        config,
        "--device",
        device,
        "--max-images",
        str(max_images),
        "--output",
        output_path,
        "--ttt",
        "--ttt-method",
        method_value,
        "--ttt-reset",
        reset,
        "--ttt-steps",
        str(steps),
        "--ttt-log-out",
        report_path,
    ]
    args.extend(["--checkpoint", checkpoint])
    if preset:
        args.extend(["--ttt-preset", preset])
    if force:
        args.append("--force")
    artifacts = {"predictions": output_path, "ttt_report": report_path}
    submit = submit_job_public if public else submit_job
    payload = submit(job_name, args, artifacts=artifacts)
    payload["preflight"] = preflight
    return payload


def ttt_job(
    dataset: str,
    checkpoint: str,
    output: str | None = None,
    *,
    config: str = _DEFAULT_TTT_CONFIG,
    split: str = "val",
    report: str | None = None,
    method: str = "tent",
    preset: str | None = None,
    steps: int = 1,
    reset: str = "sample",
    device: str = "cpu",
    max_images: int = 1,
    force: bool = True,
) -> dict[str, Any]:
    return _ttt_export_job(
        job_name="ttt",
        dataset=dataset,
        checkpoint=checkpoint,
        output=output,
        config=config,
        split=split,
        report=report,
        method=method,
        preset=preset,
        steps=steps,
        reset=reset,
        device=device,
        max_images=max_images,
        force=force,
        public=False,
    )


def ttt_job_public(
    dataset: str,
    checkpoint: str,
    output: str | None = None,
    *,
    config: str = _DEFAULT_TTT_CONFIG,
    split: str = "val",
    report: str | None = None,
    method: str = "tent",
    preset: str | None = None,
    steps: int = 1,
    reset: str = "sample",
    device: str = "cpu",
    max_images: int = 1,
    force: bool = True,
) -> dict[str, Any]:
    return _ttt_export_job(
        job_name="ttt",
        dataset=dataset,
        checkpoint=checkpoint,
        output=output,
        config=config,
        split=split,
        report=report,
        method=method,
        preset=preset,
        steps=steps,
        reset=reset,
        device=device,
        max_images=max_images,
        force=force,
        public=True,
    )


def ctta_job(
    dataset: str,
    checkpoint: str,
    output: str | None = None,
    *,
    config: str = _DEFAULT_TTT_CONFIG,
    split: str = "val",
    report: str | None = None,
    method: str = "cotta",
    preset: str | None = None,
    steps: int = 1,
    reset: str = "stream",
    device: str = "cpu",
    max_images: int = 1,
    force: bool = True,
) -> dict[str, Any]:
    return _ttt_export_job(
        job_name="ctta",
        dataset=dataset,
        checkpoint=checkpoint,
        output=output,
        config=config,
        split=split,
        report=report,
        method=method,
        preset=preset,
        steps=steps,
        reset=reset,
        device=device,
        max_images=max_images,
        force=force,
        public=False,
    )


def ctta_job_public(
    dataset: str,
    checkpoint: str,
    output: str | None = None,
    *,
    config: str = _DEFAULT_TTT_CONFIG,
    split: str = "val",
    report: str | None = None,
    method: str = "cotta",
    preset: str | None = None,
    steps: int = 1,
    reset: str = "stream",
    device: str = "cpu",
    max_images: int = 1,
    force: bool = True,
) -> dict[str, Any]:
    return _ttt_export_job(
        job_name="ctta",
        dataset=dataset,
        checkpoint=checkpoint,
        output=output,
        config=config,
        split=split,
        report=report,
        method=method,
        preset=preset,
        steps=steps,
        reset=reset,
        device=device,
        max_images=max_images,
        force=force,
        public=True,
    )
