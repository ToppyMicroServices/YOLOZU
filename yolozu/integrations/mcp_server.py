from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .ai_surface import (
    ai_surface_sets,
    discover_manifest_tools,
    generate_config,
    review_config,
)
from .manifest_resources import (
    resolve_workspace_path,
    workspace_root as resolved_workspace_root,
)
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
    process_images,
    recommend_image_pipeline,
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


def _rejected_input(
    tool: str,
    *,
    code: str,
    message: str,
) -> dict:
    payload = {
        "schema_version": 1,
        "ok": False,
        "tool": tool,
        "summary": "input rejected",
        "exit_code": 1,
        "error": {
            "code": code,
            "message": message,
        },
    }
    return payload


@app.tool(name="ai_tools")
def ai_tools_tool(
    manifest_path: str | None = None,
    guaranteed: bool = False,
    supported: bool = False,
    maturity: str | None = None,
    tag: str | None = None,
    ids_only: bool = False,
) -> dict:
    """List supported AI-first MCP tools plus manifest-backed metadata."""
    safe_manifest_path: str | None = None
    if manifest_path is not None:
        try:
            safe_manifest_path = str(resolve_workspace_path(manifest_path))
        except ValueError as exc:
            return _rejected_input(
                "ai_tools",
                code="unsafe_manifest_path",
                message=str(exc),
            )
    try:
        surfaces = ai_surface_sets(safe_manifest_path)
        discovery = discover_manifest_tools(
            manifest_path=safe_manifest_path,
            guaranteed=guaranteed,
            supported=supported,
            maturity=maturity,
            tag=tag,
            ids_only=ids_only,
        )
        tools = discovery["tools"]
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return _rejected_input(
            "ai_tools",
            code="invalid_manifest",
            message=str(exc),
        )
    payload = {
        "schema_version": 1,
        "ok": True,
        "tool": "ai_tools",
        "summary": "listed AI/MCP tool surface",
        "exit_code": 0,
        "supported_mcp_tools": list(
            surfaces["guaranteed_ai_safe"]["tool_ids"]
        ),
        "supported_mcp_tools_semantics": (
            "compatibility view of guaranteed_ai_safe tool ids"
        ),
        "filters": {
            "guaranteed": guaranteed,
            "supported": supported,
            "maturity": maturity,
            "tag": tag,
            "ids_only": ids_only,
        },
        "manifest_tools": tools,
    }
    if ids_only:
        payload["selected_tool_ids"] = list(tools)
        payload["surface_counts"] = {
            name: len(surface["tool_ids"])
            for name, surface in surfaces.items()
        }
    else:
        payload["guaranteed_mcp_tools"] = list(
            surfaces["guaranteed_ai_safe"]["tool_ids"]
        )
        payload["live_mcp_tools"] = list(
            surfaces["mcp_live"]["tool_ids"]
        )
        payload["surfaces"] = surfaces
    if maturity is not None or tag is not None:
        payload["filter_diagnostics"] = discovery[
            "filter_diagnostics"
        ]
    return payload


@app.tool(name="generate_config")
def generate_config_tool(
    goal: str = "evaluate_predictions",
    dataset: str = "data/smoke",
    predictions: str = "data/smoke/predictions/predictions_dummy.json",
    split: str = "val",
    output: str = "reports/ai_eval.json",
    max_images: int = 50,
    dry_run: bool = True,
) -> dict:
    """Generate deterministic, safe-default config for agent runs."""
    payload = generate_config(
        goal=goal,
        dataset=dataset,
        predictions=predictions,
        split=split,
        output=output,
        max_images=max_images,
        dry_run=dry_run,
    )
    return {
        "ok": True,
        "tool": "generate_config",
        "summary": "generated config",
        "exit_code": 0,
        "config": payload,
    }


@app.tool(name="review_config")
def review_config_tool(config_json: str, workspace_root: str = ".") -> dict:
    """Review agent config JSON and return issues/warnings."""
    import json

    try:
        p = resolve_workspace_path(config_json)
        safe_review_root = resolve_workspace_path(
            workspace_root,
            root=resolved_workspace_root(),
        )
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        return _rejected_input(
            "review_config",
            code="config_read_failed",
            message=str(exc),
        )
    except ValueError as exc:
        return _rejected_input(
            "review_config",
            code="unsafe_or_invalid_config",
            message=str(exc),
        )
    review = review_config(doc, workspace_root=str(safe_review_root))
    return {
        "ok": bool(review.get("ok")),
        "tool": "review_config",
        "summary": str(review.get("summary")),
        "exit_code": 0 if bool(review.get("ok")) else 1,
        "review": review,
    }


@app.tool(name="doctor")
def doctor_tool(output: str = "reports/doctor.json") -> dict:
    """Run yolozu doctor and return JSON payload + short summary."""
    return doctor(output=output)


@app.tool(name="validate_predictions")
def validate_predictions_tool(path: str, strict: bool = True) -> dict:
    """Validate strictly, or explicitly repair with strict=false."""
    return validate_predictions(path=path, strict=strict)


@app.tool(name="validate_dataset")
def validate_dataset_tool(dataset: str, split: str | None = None, strict: bool = True, mode: str = "fail") -> dict:
    """Validate YOLO-format dataset."""
    return validate_dataset(dataset=dataset, split=split, strict=strict, mode=mode)


@app.tool(name="eval_coco")
def eval_coco_tool(
    dataset: str,
    predictions: str,
    split: str | None = None,
    dry_run: bool = True,
    output: str = "reports/mcp_coco_eval.json",
    max_images: int | None = None,
    repair: bool = False,
) -> dict:
    """Evaluate predictions strictly unless repair is explicitly enabled."""
    return eval_coco(
        dataset=dataset,
        predictions=predictions,
        split=split,
        dry_run=dry_run,
        output=output,
        max_images=max_images,
        repair=repair,
    )


@app.tool(name="predict_images")
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


@app.tool(name="recommend_image_pipeline")
def recommend_image_pipeline_tool(
    job_spec: dict[str, object],
    input_path: str,
    registry_root: str | None = None,
    screening_root: str | None = None,
    evidence_root: str | None = None,
    artifact_root: str | None = None,
) -> dict:
    """Recommend a qualified local image pipeline without executing it."""
    return recommend_image_pipeline(
        job_spec=job_spec,
        input_path=input_path,
        registry_root=registry_root,
        screening_root=screening_root,
        evidence_root=evidence_root,
        artifact_root=artifact_root,
    )


@app.tool(name="process_images")
def process_images_tool(
    job_spec: dict[str, object],
    selection_decision: dict[str, object],
    input_path: str,
    output_dir: str,
    registry_root: str | None = None,
    evidence_root: str | None = None,
    artifact_root: str | None = None,
    dry_run: bool = True,
    force: bool = False,
) -> dict:
    """Revalidate and optionally execute one pinned selected image pipeline."""
    return process_images(
        job_spec=job_spec,
        selection_decision=selection_decision,
        input_path=input_path,
        output_dir=output_dir,
        registry_root=registry_root,
        evidence_root=evidence_root,
        artifact_root=artifact_root,
        dry_run=dry_run,
        force=force,
    )


@app.tool(name="parity_check")
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


@app.tool(name="calibrate_predictions")
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


@app.tool(name="eval_instance_seg")
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


@app.tool(name="eval_long_tail")
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


@app.tool(name="run_scenarios")
def run_scenarios_tool(config: str, extra_args: list[str] | None = None) -> dict:
    """Run scenarios with declared long-form scenario flags only."""
    return run_scenarios(config=config, extra_args=extra_args)


@app.tool(name="convert_dataset")
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


@app.tool(name="train_job")
def train_job_tool(train_config: str, run_id: str | None = None, resume: str | None = None) -> dict:
    """Queue train command as asynchronous job and return job_id."""
    return train_job(train_config=train_config, run_id=run_id, resume=resume)


@app.tool(name="export_predictions_job")
def export_predictions_job_tool(dataset: str, output: str, split: str | None = None, force: bool = True) -> dict:
    """Queue predictions export command as asynchronous job and return job_id."""
    return export_predictions_job(dataset=dataset, output=output, split=split, force=force)


@app.tool(name="export_onnx_job")
def export_onnx_job_tool(dataset: str, output: str, split: str | None = None, force: bool = True) -> dict:
    """Compatibility alias for export_predictions_job_tool."""
    return export_onnx_job(dataset=dataset, output=output, split=split, force=force)


@app.tool(name="test_job")
def test_job_tool(test_config: str, extra_args: list[str] | None = None) -> dict:
    """Queue a test using declared long-form scenario flags only."""
    return test_job(test_config=test_config, extra_args=extra_args)


@app.tool(name="ttt_job")
def ttt_job_tool(
    dataset: str,
    checkpoint: str,
    output: str | None = None,
    config: str = "builtin:base",
    split: str = "val",
    report: str | None = None,
    method: str = "tent",
    preset: str | None = None,
    steps: int = 1,
    reset: str = "sample",
    device: str = "cpu",
    max_images: int = 1,
    force: bool = True,
) -> dict:
    """Queue a local TTT export diagnostic and return its asynchronous job_id."""
    return ttt_job(
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
    )


@app.tool(name="ctta_job")
def ctta_job_tool(
    dataset: str,
    checkpoint: str,
    output: str | None = None,
    config: str = "builtin:base",
    split: str = "val",
    report: str | None = None,
    method: str = "cotta",
    preset: str | None = None,
    steps: int = 1,
    reset: str = "stream",
    device: str = "cpu",
    max_images: int = 1,
    force: bool = True,
) -> dict:
    """Queue a local continual-TTA export diagnostic and return its job_id."""
    return ctta_job(
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
    )


@app.tool(name="jobs_list")
def jobs_list_tool() -> dict:
    """List jobs."""
    return jobs_list()


@app.tool(name="jobs_status")
def jobs_status_tool(job_id: str) -> dict:
    """Get status for one job."""
    return jobs_status(job_id)


@app.tool(name="jobs_cancel")
def jobs_cancel_tool(job_id: str) -> dict:
    """Cancel one job if possible."""
    return jobs_cancel(job_id)


@app.tool(name="runs_list")
def runs_list_tool(limit: int = 20) -> dict:
    """List run directories and metadata."""
    return runs_list(limit=limit)


@app.tool(name="runs_describe")
def runs_describe_tool(run_id: str) -> dict:
    """Describe run artifacts."""
    return runs_describe(run_id)


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
