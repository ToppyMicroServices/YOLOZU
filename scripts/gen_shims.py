#!/usr/bin/env python3
"""Generate backward-compatibility shim files at old yolozu/<module>.py locations."""
import pathlib

SHIMS = {
    # datasets/
    "dataset": "datasets.dataset",
    "dataset_fetch": "datasets.dataset_fetch",
    "dataset_validator": "datasets.dataset_validator",
    "segmentation_dataset": "datasets.segmentation_dataset",
    "migrate": "datasets.migrate",
    "coco_convert": "datasets.coco_convert",
    "splits": "datasets.splits",
    "imports": "datasets.imports",
    # eval/
    "coco_eval": "eval.coco_eval",
    "coco_keypoints_eval": "eval.coco_keypoints_eval",
    "keypoints_eval": "eval.keypoints_eval",
    "pose_eval": "eval.pose_eval",
    "segmentation_eval": "eval.segmentation_eval",
    "instance_segmentation_eval": "eval.instance_segmentation_eval",
    "simple_map": "eval.simple_map",
    "long_tail_metrics": "eval.long_tail_metrics",
    "continual_metrics": "eval.continual_metrics",
    "synthgen_eval": "eval.synthgen_eval",
    "benchmark": "eval.benchmark",
    "metrics": "eval.metrics",
    "metrics_report": "eval.metrics_report",
    # predictions/
    "predictions": "predictions.predictions",
    "predictions_transform": "predictions.predictions_transform",
    "predictions_parity": "predictions.predictions_parity",
    "segmentation_predictions": "predictions.segmentation_predictions",
    "instance_segmentation_predictions": "predictions.instance_segmentation_predictions",
    "instance_segmentation_report": "predictions.instance_segmentation_report",
    "export": "predictions.export",
    "schema_governance": "predictions.schema_governance",
    # inference/
    "adapter": "inference.adapter",
    "runner": "inference.runner",
    "predict_images": "inference.predict_images",
    "inference": "inference.inference",
    "inference_utils": "inference.inference_utils",
    "pipeline": "inference.pipeline",
    "onnxrt_export": "inference.onnxrt_export",
    "onnxrt_quantize": "inference.onnxrt_quantize",
    "model_fetch": "inference.model_fetch",
    # geometry/
    "geometry": "geometry.geometry",
    "intrinsics": "geometry.intrinsics",
    "math3d": "geometry.math3d",
    "constraints": "geometry.constraints",
    "symmetry": "geometry.symmetry",
    "jitter": "geometry.jitter",
    "template_verification": "geometry.template_verification",
    # training/
    "distillation": "training.distillation",
    "sdft": "training.sdft",
    "continual_regularizers": "training.continual_regularizers",
    "long_tail_recipe": "training.long_tail_recipe",
    "replay_buffer": "training.replay_buffer",
    "gates": "training.gates",
    "map_targets": "training.map_targets",
    # core/
    "canonical": "core.canonical",
    "boxes": "core.boxes",
    "keypoints": "core.keypoints",
    "image_keys": "core.image_keys",
    "image_size": "core.image_size",
    "letterbox": "core.letterbox",
    "config": "core.config",
    "resources": "core.resources",
    "run_record": "core.run_record",
    "doctor": "core.doctor",
    "eval_protocol": "core.eval_protocol",
    "scenario_suite": "core.scenario_suite",
    "scenarios_cli": "core.scenarios_cli",
    "cli_args": "core.cli_args",
}

TEMPLATE = '''\
"""Backward-compatibility shim \u2014 canonical location: ``yolozu.{new_path}``."""

# Re-export everything so ``from yolozu.{old_name} import X`` keeps working.
from yolozu.{new_path} import *  # noqa: F401,F403
'''

root = pathlib.Path("yolozu")
created = 0
for old_name, new_path in SHIMS.items():
    path = root / f"{old_name}.py"
    path.write_text(TEMPLATE.format(old_name=old_name, new_path=new_path))
    created += 1

print(f"Created {created} shim files")
