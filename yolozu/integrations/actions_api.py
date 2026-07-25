from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .tool_runner import (
    ctta_job_public as ctta_job,
    calibrate_predictions_public as calibrate_predictions,
    convert_dataset_public as convert_dataset,
    doctor_public as doctor,
    eval_coco_public as eval_coco,
    eval_instance_seg_public as eval_instance_seg,
    eval_long_tail_public as eval_long_tail,
    export_predictions_job_public as export_predictions_job,
    export_onnx_job_public as export_onnx_job,
    jobs_cancel,
    jobs_list,
    jobs_status,
    parity_check_public as parity_check,
    predict_images_public as predict_images,
    run_scenarios_public as run_scenarios,
    runs_describe,
    runs_list,
    train_job_public as train_job,
    test_job_public as test_job,
    ttt_job_public as ttt_job,
    validate_dataset_public as validate_dataset,
    validate_predictions_public as validate_predictions,
)


app = FastAPI(title="YOLOZU Actions API", version="0.1.0", debug=False)


_SENSITIVE_RESPONSE_KEYS = {
    # Can contain stack traces from CLI failures.
    "stdout",
    "stderr",
}


def _sanitize_response(payload):  # type: ignore[no-untyped-def]
    if isinstance(payload, dict):
        return {k: _sanitize_response(v) for k, v in payload.items() if k not in _SENSITIVE_RESPONSE_KEYS}
    if isinstance(payload, list):
        return [_sanitize_response(item) for item in payload]
    return payload


class DoctorRequest(BaseModel):
    output: str = "reports/doctor.json"


class ValidatePredictionsRequest(BaseModel):
    path: str
    strict: bool = True


class ValidateDatasetRequest(BaseModel):
    dataset: str
    split: str | None = None
    strict: bool = True
    mode: str = "fail"


class EvalCocoRequest(BaseModel):
    dataset: str
    predictions: str
    split: str | None = None
    dry_run: bool = True
    output: str = "reports/mcp_coco_eval.json"
    max_images: int | None = None
    repair: bool = False


class PredictImagesRequest(BaseModel):
    input_dir: str
    backend: str = "dummy"
    output: str = "reports/mcp_predict_images.json"
    max_images: int | None = None
    dry_run: bool = True
    strict: bool = True
    force: bool = True


class ParityCheckRequest(BaseModel):
    reference: str
    candidate: str
    iou_thresh: float = 0.5
    score_atol: float = 1e-6
    bbox_atol: float = 1e-4
    max_images: int | None = None
    image_size: str | None = None


class CalibratePredictionsRequest(BaseModel):
    dataset: str
    predictions: str
    method: str = "fracal"
    split: str | None = None
    task: str = "auto"
    output: str = "reports/mcp_calibrated_predictions.json"
    output_report: str = "reports/mcp_calibration_report.json"
    max_images: int | None = None
    force: bool = True


class EvalInstanceSegRequest(BaseModel):
    dataset: str
    predictions: str
    split: str | None = None
    output: str = "reports/mcp_instance_seg_eval.json"
    max_images: int | None = None
    min_score: float | None = None
    allow_rgb_masks: bool = False


class EvalLongTailRequest(BaseModel):
    dataset: str
    predictions: str
    split: str | None = None
    output: str = "reports/mcp_long_tail_eval.json"
    max_images: int | None = None
    max_detections: int | None = None


class RunScenariosRequest(BaseModel):
    config: str
    extra_args: list[str] | None = None


class ConvertDatasetRequest(BaseModel):
    from_format: str
    output: str
    data: str | None = None
    args_yaml: str | None = None
    split: str | None = None
    task: str | None = None
    coco_root: str | None = None
    instances_json: str | None = None
    mode: str = "manifest"
    include_crowd: bool = False
    force: bool = True


class TrainJobRequest(BaseModel):
    train_config: str
    run_id: str | None = None
    resume: str | None = None


class ExportPredictionsJobRequest(BaseModel):
    dataset: str
    output: str
    split: str | None = None
    force: bool = True


class ExportOnnxJobRequest(ExportPredictionsJobRequest):
    pass


class TestJobRequest(BaseModel):
    test_config: str
    extra_args: list[str] | None = None


class TTTJobRequest(BaseModel):
    dataset: str
    checkpoint: str
    output: str | None = None
    config: str = "builtin:base"
    split: str = "val"
    report: str | None = None
    method: str = "tent"
    preset: str | None = None
    steps: int = 1
    reset: str = "sample"
    device: str = "cpu"
    max_images: int = 1
    force: bool = True


class CTTAJobRequest(BaseModel):
    dataset: str
    checkpoint: str
    output: str | None = None
    config: str = "builtin:base"
    split: str = "val"
    report: str | None = None
    method: str = "cotta"
    preset: str | None = None
    steps: int = 1
    reset: str = "stream"
    device: str = "cpu"
    max_images: int = 1
    force: bool = True


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "summary": "yolozu-actions-api healthy"}


@app.post("/doctor")
def doctor_route(req: DoctorRequest) -> dict:
    return _sanitize_response(doctor(output=req.output))


@app.post("/validate/predictions")
def validate_predictions_route(req: ValidatePredictionsRequest) -> dict:
    return _sanitize_response(validate_predictions(path=req.path, strict=req.strict))


@app.post("/validate/dataset")
def validate_dataset_route(req: ValidateDatasetRequest) -> dict:
    return _sanitize_response(validate_dataset(dataset=req.dataset, split=req.split, strict=req.strict, mode=req.mode))


@app.post("/eval/coco")
def eval_coco_route(req: EvalCocoRequest) -> dict:
    return _sanitize_response(
        eval_coco(
            dataset=req.dataset,
            predictions=req.predictions,
            split=req.split,
            dry_run=req.dry_run,
            output=req.output,
            max_images=req.max_images,
            repair=req.repair,
        )
    )


@app.post("/predict/images")
def predict_images_route(req: PredictImagesRequest) -> dict:
    return _sanitize_response(
        predict_images(
            input_dir=req.input_dir,
            backend=req.backend,
            output=req.output,
            max_images=req.max_images,
            dry_run=req.dry_run,
            strict=req.strict,
            force=req.force,
        )
    )


@app.post("/parity/check")
def parity_check_route(req: ParityCheckRequest) -> dict:
    return _sanitize_response(
        parity_check(
            reference=req.reference,
            candidate=req.candidate,
            iou_thresh=req.iou_thresh,
            score_atol=req.score_atol,
            bbox_atol=req.bbox_atol,
            max_images=req.max_images,
            image_size=req.image_size,
        )
    )


@app.post("/calibrate/predictions")
def calibrate_predictions_route(req: CalibratePredictionsRequest) -> dict:
    return _sanitize_response(
        calibrate_predictions(
            dataset=req.dataset,
            predictions=req.predictions,
            method=req.method,
            split=req.split,
            task=req.task,
            output=req.output,
            output_report=req.output_report,
            max_images=req.max_images,
            force=req.force,
        )
    )


@app.post("/eval/instance-seg")
def eval_instance_seg_route(req: EvalInstanceSegRequest) -> dict:
    return _sanitize_response(
        eval_instance_seg(
            dataset=req.dataset,
            predictions=req.predictions,
            split=req.split,
            output=req.output,
            max_images=req.max_images,
            min_score=req.min_score,
            allow_rgb_masks=req.allow_rgb_masks,
        )
    )


@app.post("/eval/long-tail")
def eval_long_tail_route(req: EvalLongTailRequest) -> dict:
    return _sanitize_response(
        eval_long_tail(
            dataset=req.dataset,
            predictions=req.predictions,
            split=req.split,
            output=req.output,
            max_images=req.max_images,
            max_detections=req.max_detections,
        )
    )


@app.post("/run/scenarios")
def run_scenarios_route(req: RunScenariosRequest) -> dict:
    return _sanitize_response(run_scenarios(config=req.config, extra_args=req.extra_args))


@app.post("/convert/dataset")
def convert_dataset_route(req: ConvertDatasetRequest) -> dict:
    return _sanitize_response(
        convert_dataset(
            from_format=req.from_format,
            output=req.output,
            data=req.data,
            args_yaml=req.args_yaml,
            split=req.split,
            task=req.task,
            coco_root=req.coco_root,
            instances_json=req.instances_json,
            mode=req.mode,
            include_crowd=req.include_crowd,
            force=req.force,
        )
    )


@app.post("/jobs/train")
def train_job_route(req: TrainJobRequest) -> dict:
    return _sanitize_response(train_job(train_config=req.train_config, run_id=req.run_id, resume=req.resume))


@app.post("/jobs/export-predictions")
def export_predictions_job_route(req: ExportPredictionsJobRequest) -> dict:
    return _sanitize_response(export_predictions_job(dataset=req.dataset, output=req.output, split=req.split, force=req.force))


@app.post("/jobs/export-onnx")
def export_onnx_job_route(req: ExportOnnxJobRequest) -> dict:
    return _sanitize_response(export_onnx_job(dataset=req.dataset, output=req.output, split=req.split, force=req.force))


@app.post("/jobs/test")
def test_job_route(req: TestJobRequest) -> dict:
    return _sanitize_response(test_job(test_config=req.test_config, extra_args=req.extra_args))


@app.post("/jobs/ttt")
def ttt_job_route(req: TTTJobRequest) -> dict:
    return _sanitize_response(
        ttt_job(
            dataset=req.dataset,
            checkpoint=req.checkpoint,
            output=req.output,
            config=req.config,
            split=req.split,
            report=req.report,
            method=req.method,
            preset=req.preset,
            steps=req.steps,
            reset=req.reset,
            device=req.device,
            max_images=req.max_images,
            force=req.force,
        )
    )


@app.post("/jobs/ctta")
def ctta_job_route(req: CTTAJobRequest) -> dict:
    return _sanitize_response(
        ctta_job(
            dataset=req.dataset,
            checkpoint=req.checkpoint,
            output=req.output,
            config=req.config,
            split=req.split,
            report=req.report,
            method=req.method,
            preset=req.preset,
            steps=req.steps,
            reset=req.reset,
            device=req.device,
            max_images=req.max_images,
            force=req.force,
        )
    )


@app.get("/jobs")
def jobs_list_route() -> dict:
    return _sanitize_response(jobs_list())


@app.get("/jobs/{job_id}")
def jobs_status_route(job_id: str) -> dict:
    return _sanitize_response(jobs_status(job_id))


@app.post("/jobs/{job_id}/cancel")
def jobs_cancel_route(job_id: str) -> dict:
    return _sanitize_response(jobs_cancel(job_id))


@app.get("/runs")
def runs_list_route(limit: int = 20) -> dict:
    return _sanitize_response(runs_list(limit=limit))


@app.get("/runs/{run_id}")
def runs_describe_route(run_id: str) -> dict:
    return _sanitize_response(runs_describe(run_id))
