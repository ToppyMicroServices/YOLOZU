from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .tool_runner import (
    ctta_job,
    calibrate_predictions,
    convert_dataset,
    doctor,
    eval_coco,
    eval_instance_seg,
    eval_long_tail,
    export_predictions_job,
    export_onnx_job,
    jobs_cancel,
    jobs_list,
    jobs_status,
    parity_check,
    predict_images,
    run_scenarios,
    runs_describe,
    runs_list,
    train_job,
    test_job,
    ttt_job,
    validate_dataset,
    validate_predictions,
)


app = FastMCP("yolozu")


@app.tool()
def doctor_tool(output: str = "reports/doctor.json") -> dict:
    """Run yolozu doctor and return JSON payload + short summary."""
    return doctor(output=output)


@app.tool()
def validate_predictions_tool(path: str, strict: bool = True) -> dict:
    """Validate predictions JSON."""
    return validate_predictions(path=path, strict=strict)


@app.tool()
def validate_dataset_tool(dataset: str, split: str | None = None, strict: bool = True, mode: str = "fail") -> dict:
    """Validate YOLO-format dataset."""
    return validate_dataset(dataset=dataset, split=split, strict=strict, mode=mode)


@app.tool()
def eval_coco_tool(
    dataset: str,
    predictions: str,
    split: str | None = None,
    dry_run: bool = True,
    output: str = "reports/mcp_coco_eval.json",
    max_images: int | None = None,
) -> dict:
    """Evaluate predictions using eval-coco."""
    return eval_coco(
        dataset=dataset,
        predictions=predictions,
        split=split,
        dry_run=dry_run,
        output=output,
        max_images=max_images,
    )


@app.tool()
def predict_images_tool(
    input_dir: str,
    backend: str = "dummy",
    output: str = "reports/mcp_predict_images.json",
    max_images: int | None = None,
    dry_run: bool = True,
    strict: bool = True,
    force: bool = True,
) -> dict:
    """Run folder inference and write predictions JSON."""
    return predict_images(
        input_dir=input_dir,
        backend=backend,
        output=output,
        max_images=max_images,
        dry_run=dry_run,
        strict=strict,
        force=force,
    )


@app.tool()
def parity_check_tool(
    reference: str,
    candidate: str,
    iou_thresh: float = 0.5,
    score_atol: float = 1e-6,
    bbox_atol: float = 1e-4,
    max_images: int | None = None,
    image_size: str | None = None,
) -> dict:
    """Compare two predictions JSON files for parity."""
    return parity_check(
        reference=reference,
        candidate=candidate,
        iou_thresh=iou_thresh,
        score_atol=score_atol,
        bbox_atol=bbox_atol,
        max_images=max_images,
        image_size=image_size,
    )


@app.tool()
def calibrate_predictions_tool(
    dataset: str,
    predictions: str,
    method: str = "fracal",
    split: str | None = None,
    task: str = "auto",
    output: str = "reports/mcp_calibrated_predictions.json",
    output_report: str = "reports/mcp_calibration_report.json",
    max_images: int | None = None,
    force: bool = True,
) -> dict:
    """Apply post-hoc calibration to prediction scores."""
    return calibrate_predictions(
        dataset=dataset,
        predictions=predictions,
        method=method,
        split=split,
        task=task,
        output=output,
        output_report=output_report,
        max_images=max_images,
        force=force,
    )


@app.tool()
def eval_instance_seg_tool(
    dataset: str,
    predictions: str,
    split: str | None = None,
    output: str = "reports/mcp_instance_seg_eval.json",
    max_images: int | None = None,
    min_score: float | None = None,
    allow_rgb_masks: bool = False,
) -> dict:
    """Evaluate instance segmentation predictions."""
    return eval_instance_seg(
        dataset=dataset,
        predictions=predictions,
        split=split,
        output=output,
        max_images=max_images,
        min_score=min_score,
        allow_rgb_masks=allow_rgb_masks,
    )


@app.tool()
def eval_long_tail_tool(
    dataset: str,
    predictions: str,
    split: str | None = None,
    output: str = "reports/mcp_long_tail_eval.json",
    max_images: int | None = None,
    max_detections: int | None = None,
) -> dict:
    """Evaluate long-tail detection metrics."""
    return eval_long_tail(
        dataset=dataset,
        predictions=predictions,
        split=split,
        output=output,
        max_images=max_images,
        max_detections=max_detections,
    )


@app.tool()
def run_scenarios_tool(config: str, extra_args: list[str] | None = None) -> dict:
    """Run scenario suite via yolozu test."""
    return run_scenarios(config=config, extra_args=extra_args)


@app.tool()
def convert_dataset_tool(
    from_format: str,
    output: str,
    data: str | None = None,
    args_yaml: str | None = None,
    split: str | None = None,
    task: str | None = None,
    coco_root: str | None = None,
    instances_json: str | None = None,
    mode: str = "manifest",
    include_crowd: bool = False,
    force: bool = True,
) -> dict:
    """Convert external dataset layout into YOLOZU descriptor via migrate dataset."""
    return convert_dataset(
        from_format=from_format,
        output=output,
        data=data,
        args_yaml=args_yaml,
        split=split,
        task=task,
        coco_root=coco_root,
        instances_json=instances_json,
        mode=mode,
        include_crowd=include_crowd,
        force=force,
    )


@app.tool()
def train_job_tool(train_config: str, run_id: str | None = None, resume: str | None = None) -> dict:
    """Queue train command as asynchronous job and return job_id."""
    return train_job(train_config=train_config, run_id=run_id, resume=resume)


@app.tool()
def export_predictions_job_tool(dataset: str, output: str, split: str | None = None, force: bool = True) -> dict:
    """Queue predictions export command as asynchronous job and return job_id."""
    return export_predictions_job(dataset=dataset, output=output, split=split, force=force)


@app.tool()
def export_onnx_job_tool(dataset: str, output: str, split: str | None = None, force: bool = True) -> dict:
    """Compatibility alias for export_predictions_job_tool."""
    return export_onnx_job(dataset=dataset, output=output, split=split, force=force)


@app.tool()
def test_job_tool(test_config: str, extra_args: list[str] | None = None) -> dict:
    """Queue test scenario command as asynchronous job and return job_id."""
    return test_job(test_config=test_config, extra_args=extra_args)


@app.tool()
def ttt_job_tool(
    test_config: str,
    method: str = "tent",
    preset: str | None = None,
    steps: int | None = None,
    reset: bool = False,
    extra_args: list[str] | None = None,
) -> dict:
    """Queue TTT scenario command as asynchronous job and return job_id."""
    return ttt_job(
        test_config=test_config,
        method=method,
        preset=preset,
        steps=steps,
        reset=reset,
        extra_args=extra_args,
    )


@app.tool()
def ctta_job_tool(
    test_config: str,
    method: str = "cotta",
    preset: str | None = None,
    steps: int | None = None,
    reset: bool = False,
    extra_args: list[str] | None = None,
) -> dict:
    """Queue CTTA scenario command as asynchronous job and return job_id."""
    return ctta_job(
        test_config=test_config,
        method=method,
        preset=preset,
        steps=steps,
        reset=reset,
        extra_args=extra_args,
    )


@app.tool()
def jobs_list_tool() -> dict:
    """List jobs."""
    return jobs_list()


@app.tool()
def jobs_status_tool(job_id: str) -> dict:
    """Get status for one job."""
    return jobs_status(job_id)


@app.tool()
def jobs_cancel_tool(job_id: str) -> dict:
    """Cancel one job if possible."""
    return jobs_cancel(job_id)


@app.tool()
def runs_list_tool(limit: int = 20) -> dict:
    """List run directories and metadata."""
    return runs_list(limit=limit)


@app.tool()
def runs_describe_tool(run_id: str) -> dict:
    """Describe run artifacts."""
    return runs_describe(run_id)


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
