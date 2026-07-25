from __future__ import annotations

from functools import lru_cache
from typing import Any

from .layers.api import run_cli_tool, run_cli_tool_redacted
from .layers.artifacts import collect_artifact_metadata, describe_run, list_runs
from .layers.jobs import JobManager


@lru_cache(maxsize=1)
def _job_manager() -> JobManager:
    """Create persistent job storage only when a job operation is requested."""
    return JobManager()


def _with_meta(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("meta", collect_artifact_metadata())
    return payload


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
    args = ["test", config, *(extra_args or [])]
    return _with_meta(run_cli_tool("run_scenarios", args))


def run_scenarios_public(config: str, *, extra_args: list[str] | None = None) -> dict[str, Any]:
    args = ["test", config, *(extra_args or [])]
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
    return {
        "ok": True,
        "tool": "jobs.status",
        "summary": f"job status: {status.get('status')}",
        "exit_code": 0,
        "job": status,
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
    args = ["test", test_config, *(extra_args or [])]
    return submit_job("test", args)


def test_job_public(test_config: str, *, extra_args: list[str] | None = None) -> dict[str, Any]:
    args = ["test", test_config, *(extra_args or [])]
    return submit_job_public("test", args)


def ttt_job(
    test_config: str,
    *,
    method: str = "tent",
    preset: str | None = None,
    steps: int | None = None,
    reset: bool = False,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    args = ["test", test_config, "--ttt", "--ttt-method", method]
    if preset:
        args.extend(["--ttt-preset", preset])
    if steps is not None:
        args.extend(["--ttt-steps", str(steps)])
    if reset:
        args.append("--ttt-reset")
    args.extend(extra_args or [])
    return submit_job("test", args)


def ttt_job_public(
    test_config: str,
    *,
    method: str = "tent",
    preset: str | None = None,
    steps: int | None = None,
    reset: bool = False,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    args = ["test", test_config, "--ttt", "--ttt-method", method]
    if preset:
        args.extend(["--ttt-preset", preset])
    if steps is not None:
        args.extend(["--ttt-steps", str(steps)])
    if reset:
        args.append("--ttt-reset")
    args.extend(extra_args or [])
    return submit_job_public("test", args)


def ctta_job(
    test_config: str,
    *,
    method: str = "cotta",
    preset: str | None = None,
    steps: int | None = None,
    reset: bool = False,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    args = ["test", test_config, "--ttt", "--ttt-method", method]
    if preset:
        args.extend(["--ttt-preset", preset])
    if steps is not None:
        args.extend(["--ttt-steps", str(steps)])
    if reset:
        args.append("--ttt-reset")
    args.extend(extra_args or [])
    return submit_job("test", args)


def ctta_job_public(
    test_config: str,
    *,
    method: str = "cotta",
    preset: str | None = None,
    steps: int | None = None,
    reset: bool = False,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    args = ["test", test_config, "--ttt", "--ttt-method", method]
    if preset:
        args.extend(["--ttt-preset", preset])
    if steps is not None:
        args.extend(["--ttt-steps", str(steps)])
    if reset:
        args.append("--ttt-reset")
    args.extend(extra_args or [])
    return submit_job_public("test", args)
