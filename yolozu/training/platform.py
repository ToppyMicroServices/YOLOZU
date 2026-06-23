"""Training platform primitives shared across reference and external lanes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from pathlib import Path
from typing import Any

from yolozu.core.canonical import TrainConfig
from yolozu.datasets.dataset_contract import DATASET_CONTRACT_VERSION


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass(frozen=True)
class TrainingBackendSpec:
    """Backend-agnostic description of a training lane."""

    backend_id: str
    display_name: str
    trainer_kind: str
    lane_kind: str
    maturity: str
    config_kind: str
    primary_use: str
    interface_contract_level: str
    training_family: str = "external"
    optimizer_policy: str | None = None
    preprocess_policy: str | None = None
    postprocess_policy: str | None = None
    stability_policy: tuple[str, ...] = ()
    optional_bridge: bool = False
    supports_run_contract: bool = False
    supports_export: bool = False
    supports_eval: bool = False
    supports_parity: bool = False
    supports_resume: bool = False
    supports_mps: bool = False
    supported_tasks: tuple[str, ...] = ()
    export_interface_contract: str | None = None
    eval_interface_contract: str | None = None
    parity_interface_contract: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BACKEND_SPECS: dict[str, TrainingBackendSpec] = {
    "reference-rtdetr-pose": TrainingBackendSpec(
        backend_id="reference-rtdetr-pose",
        display_name="RT-DETR pose reference trainer",
        trainer_kind="in_repo",
        lane_kind="reference",
        maturity="stable",
        config_kind="reference_config_yaml_or_json",
        primary_use="In-repo training with full run artifacts, resume, export, eval, and parity.",
        interface_contract_level="full_run_contract",
        training_family="rtdetr",
        optimizer_policy="AdamW with separate backbone/head parameter groups for stable DETR-family fine-tuning.",
        preprocess_policy="Reference-trainer transforms; do not assume YOLO-style letterbox in the DETR trainer loop.",
        postprocess_policy="NMS-free DETR-family predictions by default; record e2e_nms_free when applicable.",
        stability_policy=("gradient_clipping", "lr_warmup", "ema", "amp_optional", "strict_run_contract"),
        supports_run_contract=True,
        supports_export=True,
        supports_eval=True,
        supports_parity=True,
        supports_resume=True,
        supports_mps=True,
        supported_tasks=("bbox", "keypoints", "depth", "pose6d"),
        export_interface_contract="Reference lane exports ONNX and training artifacts under the full run contract.",
        eval_interface_contract="YOLOZU-native eval lanes consume predictions/eval artifacts from the reference trainer.",
        parity_interface_contract="Reference lane emits ONNX parity artifacts under the full run contract.",
        notes="Primary in-repo training lane; the default path behind `yolozu train`.",
    ),
    "yolox": TrainingBackendSpec(
        backend_id="yolox",
        display_name="YOLOX external lane",
        trainer_kind="external_launcher",
        lane_kind="external",
        maturity="stable",
        config_kind="yolox_exp_python",
        primary_use="Apache-2.0-friendly YOLO-style training launched through YOLOZU wrappers.",
        interface_contract_level="external_run_contract",
        training_family="yolo",
        optimizer_policy="SGD with momentum/Nesterov-style YOLO defaults unless the external exp overrides them.",
        preprocess_policy="YOLO-family letterbox resize/pad assumptions are preserved in export/eval metadata.",
        postprocess_policy="NMS-applied predictions by default; use the nms_applied eval protocol.",
        stability_policy=("external_run_contract", "dry_run_handoff", "explicit_runtime_boundary"),
        supports_run_contract=True,
        supports_export=True,
        supports_eval=True,
        supports_parity=True,
        supports_resume=True,
        supports_mps=False,
        supported_tasks=("bbox",),
        export_interface_contract="Predictions interface contract via export_predictions_yolox.py.",
        eval_interface_contract="Detection eval through yolozu eval-coco.",
        parity_interface_contract="Detection parity through yolozu parity.",
        notes="Primary external training lane for YOLO-style workflows.",
    ),
    "detectron2": TrainingBackendSpec(
        backend_id="detectron2",
        display_name="Detectron2 external lane",
        trainer_kind="external_launcher",
        lane_kind="external",
        maturity="experimental",
        config_kind="detectron2_config_yaml",
        primary_use="External Detectron2 training for bbox, instance segmentation, and keypoints via backend-native configs.",
        interface_contract_level="external_run_contract",
        supports_run_contract=True,
        supports_export=True,
        supports_eval=True,
        supports_parity=True,
        supports_resume=True,
        supports_mps=False,
        supported_tasks=("bbox", "segmentation", "keypoints"),
        export_interface_contract="Predictions interface contract via Detectron2-side export bridge.",
        eval_interface_contract="Detection/keypoints eval via YOLOZU wrappers once predictions are exported.",
        parity_interface_contract="Detection/keypoints parity through YOLOZU wrappers once predictions are exported.",
        notes="Task family is selected by the Detectron2 config and optional train-time overrides.",
    ),
    "mmdetection": TrainingBackendSpec(
        backend_id="mmdetection",
        display_name="MMDetection external lane",
        trainer_kind="external_launcher",
        lane_kind="external",
        maturity="experimental",
        config_kind="mmdetection_config_python",
        primary_use="External MMDetection training for bbox and instance-seg workflows via backend-native configs.",
        interface_contract_level="external_run_contract",
        supports_run_contract=True,
        supports_export=True,
        supports_eval=True,
        supports_parity=True,
        supports_resume=True,
        supports_mps=False,
        supported_tasks=("bbox", "segmentation"),
        export_interface_contract="BBox export uses export_predictions_mmdet.py; instance-seg export uses a standardized mask-manifest handoff.",
        eval_interface_contract="Detection eval through yolozu eval-coco or instance/seg eval through mask-manifest handoff.",
        parity_interface_contract="Detection parity through yolozu parity; mask parity through segmentation/instance parity handoff.",
        notes="Best fit when detection or instance-segmentation training already lives in an MMDetection stack.",
    ),
    "mmpose": TrainingBackendSpec(
        backend_id="mmpose",
        display_name="MMPose external lane",
        trainer_kind="external_launcher",
        lane_kind="external",
        maturity="experimental",
        config_kind="mmpose_config_python",
        primary_use="External MMPose training for keypoints and pose estimation via backend-native configs.",
        interface_contract_level="external_run_contract",
        supports_run_contract=True,
        supports_export=True,
        supports_eval=True,
        supports_parity=True,
        supports_resume=True,
        supports_mps=False,
        supported_tasks=("keypoints", "pose"),
        export_interface_contract="COCO keypoints results JSON + instances JSON can be normalized into the predictions interface contract.",
        eval_interface_contract="Keypoints eval through tools/eval_keypoints.py.",
        parity_interface_contract="Keypoints parity through tools/check_keypoints_parity.py.",
        notes="Training and export handoff are first-class at the orchestration layer via standardized keypoints normalization.",
    ),
    "mmseg": TrainingBackendSpec(
        backend_id="mmseg",
        display_name="MMSeg external lane",
        trainer_kind="external_launcher",
        lane_kind="external",
        maturity="experimental",
        config_kind="mmseg_config_python",
        primary_use="External MMSegmentation training for semantic segmentation via backend-native configs.",
        interface_contract_level="external_run_contract",
        supports_run_contract=True,
        supports_export=True,
        supports_eval=True,
        supports_parity=True,
        supports_resume=True,
        supports_mps=False,
        supported_tasks=("segmentation",),
        export_interface_contract="Class-id mask directory + dataset.json can be packaged into the segmentation predictions interface contract.",
        eval_interface_contract="Semantic segmentation eval through tools/eval_segmentation.py.",
        parity_interface_contract="Semantic segmentation parity through tools/check_segmentation_parity.py.",
        notes="Training and export handoff are first-class at the orchestration layer via standardized segmentation packaging.",
    ),
    "ultralytics": TrainingBackendSpec(
        backend_id="ultralytics",
        display_name="Ultralytics bridge",
        trainer_kind="external_runtime_bridge",
        lane_kind="external",
        maturity="experimental",
        config_kind="ultralytics_model_or_runtime_args",
        primary_use="Optional external runtime bridge when the user already depends on Ultralytics.",
        interface_contract_level="external_run_contract",
        training_family="yolo",
        optimizer_policy="Backend-native YOLO optimizer settings from the user-provided runtime/config.",
        preprocess_policy="YOLO-family letterbox assumptions should be preserved in export/eval metadata.",
        postprocess_policy="NMS-applied predictions by default; keep the runtime/license boundary explicit.",
        supports_run_contract=True,
        optional_bridge=True,
        supports_export=True,
        supports_eval=True,
        supports_parity=True,
        supports_resume=True,
        supports_mps=False,
        supported_tasks=("bbox", "segmentation", "keypoints", "pose"),
        export_interface_contract="Predictions interface contract via predict-normalize/export-onnx bridge.",
        eval_interface_contract="YOLOZU eval lanes consume normalized predictions artifacts.",
        parity_interface_contract="Parity runs consume normalized predictions artifacts.",
        notes="Keep the runtime/license boundary explicit; review upstream license terms separately.",
    ),
    "hf-detr": TrainingBackendSpec(
        backend_id="hf-detr",
        display_name="HF DETR bridge",
        trainer_kind="external_launcher",
        lane_kind="external",
        maturity="experimental",
        config_kind="hf_model_id_or_external_train_script",
        primary_use="External DETR-family fine-tuning path when a Transformers stack already exists.",
        interface_contract_level="external_run_contract",
        training_family="detr",
        optimizer_policy="DETR-family AdamW-style settings should be supplied by the external trainer.",
        preprocess_policy="Backend-native DETR preprocessing recorded through the external run handoff.",
        postprocess_policy="NMS-free DETR-family outputs when supported by the external stack.",
        supports_run_contract=True,
        optional_bridge=True,
        supports_export=True,
        supports_eval=True,
        supports_parity=True,
        supports_resume=True,
        supports_mps=False,
        supported_tasks=("bbox",),
        export_interface_contract="ONNX export via support_external_training export-onnx.",
        eval_interface_contract="Detection eval through yolozu eval-coco once predictions are exported.",
        parity_interface_contract="Detection parity through yolozu parity once predictions are exported.",
        notes="Useful when DETR-family training already lives outside YOLOZU.",
    ),
    "tao": TrainingBackendSpec(
        backend_id="tao",
        display_name="NVIDIA TAO external lane",
        trainer_kind="external_launcher",
        lane_kind="external",
        maturity="experimental",
        config_kind="tao_spec_yaml",
        primary_use="Qualified NVIDIA TAO training lane for detection, segmentation, and keypoints workflows launched through YOLOZU wrappers.",
        interface_contract_level="external_run_contract",
        supports_run_contract=True,
        supports_export=True,
        supports_eval=True,
        supports_parity=True,
        supports_resume=True,
        supports_mps=False,
        supported_tasks=("bbox", "segmentation", "keypoints"),
        export_interface_contract="Task-specific handoff bridges normalize TAO outputs into the predictions interface contract or segmentation predictions interface contract.",
        eval_interface_contract="YOLOZU eval lanes consume normalized TAO handoff artifacts.",
        parity_interface_contract="YOLOZU parity lanes compare normalized TAO artifacts against reference backends.",
        notes="Environment-qualified external lane; TAO runtime remains external to YOLOZU.",
    ),
}


def get_training_backend_spec(backend_id: str) -> TrainingBackendSpec:
    key = str(backend_id or "").strip().lower()
    if key not in BACKEND_SPECS:
        allowed = ", ".join(sorted(BACKEND_SPECS))
        raise ValueError(f"unknown training backend: {backend_id!r} (expected one of: {allowed})")
    return BACKEND_SPECS[key]


def training_capability_matrix() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in BACKEND_SPECS.values()]


def project_reference_train_config(*, args: Any) -> TrainConfig:
    source: dict[str, Any] = {"from": "reference-rtdetr-pose"}
    config_path = getattr(args, "config", None)
    if config_path:
        source["config"] = str(config_path)
    run_id = getattr(args, "run_id", None)
    if run_id:
        source["run_id"] = str(run_id)

    dataset_root = getattr(args, "dataset_root", None)
    split = getattr(args, "split", None)

    preprocess: dict[str, Any] = {
        "imgsz": int(getattr(args, "image_size", 0) or 0) or None,
        "multiscale": bool(getattr(args, "multiscale", False)),
    }
    if getattr(args, "scale_min", None) is not None:
        preprocess["scale_min"] = float(args.scale_min)
    if getattr(args, "scale_max", None) is not None:
        preprocess["scale_max"] = float(args.scale_max)
    preprocess = {k: v for k, v in preprocess.items() if v is not None}

    eval_cfg: dict[str, Any] = {}
    if getattr(args, "val_every", None) is not None:
        eval_cfg["val_every"] = int(args.val_every)
    if getattr(args, "val_every_steps", None) is not None:
        eval_cfg["val_every_steps"] = int(args.val_every_steps)

    export_cfg: dict[str, Any] = {
        "onnx_requested": bool(getattr(args, "onnx_out", None)),
        "parity_requested": bool(getattr(args, "parity_json_out", None)),
    }

    run_contract: dict[str, Any] = {
        "enabled": bool(getattr(args, "run_contract", False) or getattr(args, "run_id", None)),
        "resume": bool(getattr(args, "resume", False)),
    }
    if getattr(args, "run_dir", None):
        run_contract["run_dir"] = str(args.run_dir)

    return TrainConfig(
        backend="reference-rtdetr-pose",
        task="detect+pose",
        model=(str(getattr(args, "model_config", None) or getattr(args, "config", None) or "").strip() or None),
        imgsz=int(getattr(args, "image_size", 0) or 0) or None,
        batch=int(getattr(args, "batch_size", 0) or 0) or None,
        epochs=int(getattr(args, "epochs", 0) or 0) or None,
        steps=int(getattr(args, "max_steps", 0) or 0) or None,
        optimizer=(str(getattr(args, "optimizer", "") or "").strip() or None),
        lr=float(getattr(args, "lr", 0.0) or 0.0) or None,
        weight_decay=float(getattr(args, "weight_decay", 0.0) or 0.0) or None,
        seed=int(getattr(args, "seed", 0) or 0) or None,
        device=(str(getattr(args, "device", "") or "").strip() or None),
        precision=(str(getattr(args, "amp", "") or "").strip() or None),
        workers=int(getattr(args, "workers", 0) or 0) or None,
        grad_clip_norm=float(getattr(args, "clip_grad_norm", 0.0) or 0.0) or None,
        preprocess=preprocess or None,
        eval=eval_cfg or None,
        export=export_cfg,
        dataset=(
            {
                "root": str(dataset_root),
                "split": str(split or "train"),
            }
            if dataset_root
            else None
        ),
        run_contract=run_contract,
        source=source,
    )


def _coerce_train_config(train_config: TrainConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(train_config, TrainConfig):
        return train_config.to_dict()
    if isinstance(train_config, dict):
        return {k: v for k, v in train_config.items() if v is not None}
    raise TypeError("train_config must be TrainConfig or dict")


def training_run_output_contract(*, backend_id: str, report_path: str | Path, work_dir: str | Path | None = None) -> dict[str, Any]:
    spec = get_training_backend_spec(backend_id)
    output_contract: dict[str, Any] = {
        "kind": spec.interface_contract_level,
        "report_json": str(report_path),
    }
    if work_dir is not None:
        output_contract["work_dir"] = str(work_dir)
    if spec.supports_run_contract and spec.lane_kind == "reference":
        output_contract["stable_artifacts"] = [
            "checkpoints/last.pt",
            "checkpoints/best.pt",
            "reports/train_metrics.jsonl",
            "reports/val_metrics.jsonl",
            "reports/config_resolved.yaml",
            "reports/run_meta.json",
            "reports/training_summary.json",
        ]
        if spec.supports_export:
            output_contract["stable_artifacts"].extend(
                [
                    "exports/model.onnx",
                    "exports/model.onnx.meta.json",
                ]
            )
        if spec.supports_parity:
            output_contract["stable_artifacts"].append("reports/onnx_parity.json")
    elif spec.supports_run_contract and spec.lane_kind == "external":
        output_contract["stable_artifacts"] = [
            "dataset/",
            "configs/train_config_projection.json",
            "reports/training_summary.json",
            "reports/external_run_meta.json",
            "reports/launcher_plan.json",
            "reports/execution.json",
            "reports/resume_handoff.json",
            "reports/export_handoff.json",
            "reports/eval_handoff.json",
            "reports/parity_handoff.json",
            "reports/training_registry_entry.json",
        ]
    return output_contract


def _backend_training_bbox_view(spec: TrainingBackendSpec) -> str:
    family = str(spec.training_family or "").strip().lower()
    if family == "yolo":
        return "cxcywh_norm"
    if family in {"rtdetr", "detr"}:
        return "xyxy_abs"
    return "backend_native_from_dataset_contract"


def build_training_data_flow(
    *,
    backend_id: str,
    dataset_root: str | Path | None = None,
    split: str | None = None,
    raw_dataset_format: str | None = None,
) -> dict[str, Any]:
    """Describe the standard training-data route before backend execution."""

    spec = get_training_backend_spec(backend_id)
    backend_bbox_view = _backend_training_bbox_view(spec)
    return {
        "format": "yolozu_training_data_flow_v1",
        "schema_version": 1,
        "stages": [
            "raw_dataset",
            "DatasetAdapter",
            "YOLOZU Dataset Contract",
            "TrainingBackend",
        ],
        "raw_dataset": {
            "root": str(dataset_root) if dataset_root is not None else None,
            "split": str(split) if split is not None else None,
            "format": (str(raw_dataset_format) if raw_dataset_format else "auto"),
        },
        "dataset_adapter": {
            "role": "normalize raw dataset files into YOLOZU Dataset Contract records",
            "supported_inputs": ["YOLO data.yaml", "dataset.json", "COCO JSON", "SynthGen shards"],
        },
        "dataset_contract": {
            "version": DATASET_CONTRACT_VERSION,
            "record_contract": "YOLOZU Dataset Contract",
            "bbox_storage_preference": "xyxy_abs",
            "adapter_views": ["xyxy_abs", "xywh_abs", "cxcywh_norm"],
        },
        "training_backend": {
            "backend_id": spec.backend_id,
            "family": spec.training_family,
            "bbox_view": backend_bbox_view,
            "optimizer_policy": spec.optimizer_policy,
            "preprocess_policy": spec.preprocess_policy,
            "postprocess_policy": spec.postprocess_policy,
        },
    }


def build_training_run_summary(
    *,
    backend_id: str,
    report_path: str | Path,
    train_config: TrainConfig | dict[str, Any],
    dataset_root: str | None = None,
    split: str | None = None,
    dry_run: bool = False,
    work_dir: str | Path | None = None,
    steps: dict[str, Any] | None = None,
    process: dict[str, Any] | None = None,
    runtime_error: str | None = None,
    notes: list[str] | None = None,
    license_boundary: dict[str, Any] | None = None,
    handoff_contracts: dict[str, Any] | None = None,
    raw_dataset_format: str | None = None,
) -> dict[str, Any]:
    spec = get_training_backend_spec(backend_id)
    payload: dict[str, Any] = {
        "format": "yolozu_training_run_summary_v1",
        "schema_version": 1,
        "timestamp": _now_utc(),
        "backend": spec.to_dict(),
        "dry_run": bool(dry_run),
        "training_executed": bool((steps or {}).get("train", {}).get("executed", False)),
        "ok": runtime_error is None and (bool(dry_run) or bool((steps or {}).get("train", {}).get("ok", False))),
        "canonical_train_config": _coerce_train_config(train_config),
        "dataset": {
            "root": str(dataset_root) if dataset_root is not None else None,
            "split": str(split) if split is not None else None,
        },
        "training_data_flow": build_training_data_flow(
            backend_id=backend_id,
            dataset_root=dataset_root,
            split=split,
            raw_dataset_format=raw_dataset_format,
        ),
        "run_output_contract": training_run_output_contract(
            backend_id=backend_id,
            report_path=report_path,
            work_dir=work_dir,
        ),
        "steps": steps or {},
        "process": process or None,
        "runtime_error": runtime_error,
        "license_boundary": dict(license_boundary or {}),
        "notes": list(notes or []),
        "handoff_contracts": dict(handoff_contracts or {}),
        "next_steps": [],
    }
    if work_dir is not None:
        payload["work_dir"] = str(work_dir)
    return payload
