"""YOLOZU: contract-first evaluation + tooling harness."""

from __future__ import annotations

import importlib
import sys
import types

__version__ = "4.4.0"

__all__ = ["__version__"]

_LEGACY_SUBMODULE_ALIASES = {
	"adapter": "yolozu.inference.adapter",
	"benchmark": "yolozu.eval.benchmark",
	"boxes": "yolozu.core.boxes",
	"canonical": "yolozu.core.canonical",
	"cli_args": "yolozu.core.cli_args",
	"coco_convert": "yolozu.datasets.coco_convert",
	"coco_eval": "yolozu.eval.coco_eval",
	"coco_keypoints_eval": "yolozu.eval.coco_keypoints_eval",
	"config": "yolozu.core.config",
	"constraints": "yolozu.geometry.constraints",
	"continual_metrics": "yolozu.eval.continual_metrics",
	"continual_regularizers": "yolozu.training.continual_regularizers",
	"dataset": "yolozu.datasets.dataset",
	"dataset_fetch": "yolozu.datasets.dataset_fetch",
	"dataset_validator": "yolozu.datasets.dataset_validator",
	"distillation": "yolozu.training.distillation",
	"doctor": "yolozu.core.doctor",
	"eval_protocol": "yolozu.core.eval_protocol",
	"export": "yolozu.predictions.export",
	"gates": "yolozu.training.gates",
	"image_keys": "yolozu.core.image_keys",
	"image_size": "yolozu.core.image_size",
	"imports": "yolozu.datasets.imports",
	"inference_utils": "yolozu.inference.inference_utils",
	"instance_segmentation_eval": "yolozu.eval.instance_segmentation_eval",
	"instance_segmentation_predictions": "yolozu.predictions.instance_segmentation_predictions",
	"instance_segmentation_report": "yolozu.predictions.instance_segmentation_report",
	"intrinsics": "yolozu.geometry.intrinsics",
	"jitter": "yolozu.geometry.jitter",
	"keypoints": "yolozu.core.keypoints",
	"keypoints_eval": "yolozu.eval.keypoints_eval",
	"letterbox": "yolozu.core.letterbox",
	"long_tail_metrics": "yolozu.eval.long_tail_metrics",
	"long_tail_recipe": "yolozu.training.long_tail_recipe",
	"map_targets": "yolozu.training.map_targets",
	"math3d": "yolozu.geometry.math3d",
	"metrics": "yolozu.eval.metrics",
	"metrics_report": "yolozu.eval.metrics_report",
	"migrate": "yolozu.datasets.migrate",
	"model_fetch": "yolozu.inference.model_fetch",
	"onnxrt_export": "yolozu.inference.onnxrt_export",
	"onnxrt_quantize": "yolozu.inference.onnxrt_quantize",
	"pipeline": "yolozu.inference.pipeline",
	"pose_eval": "yolozu.eval.pose_eval",
	"predict_images": "yolozu.inference.predict_images",
	"predictions_parity": "yolozu.predictions.predictions_parity",
	"predictions_transform": "yolozu.predictions.predictions_transform",
	"replay_buffer": "yolozu.training.replay_buffer",
	"resources": "yolozu.core.resources",
	"run_record": "yolozu.core.run_record",
	"runner": "yolozu.inference.runner",
	"scenario_suite": "yolozu.core.scenario_suite",
	"scenarios_cli": "yolozu.core.scenarios_cli",
	"schema_governance": "yolozu.predictions.schema_governance",
	"sdft": "yolozu.training.sdft",
	"segmentation_dataset": "yolozu.datasets.segmentation_dataset",
	"segmentation_eval": "yolozu.eval.segmentation_eval",
	"segmentation_predictions": "yolozu.predictions.segmentation_predictions",
	"simple_map": "yolozu.eval.simple_map",
	"splits": "yolozu.datasets.splits",
	"symmetry": "yolozu.geometry.symmetry",
	"synthgen_eval": "yolozu.eval.synthgen_eval",
	"template_verification": "yolozu.geometry.template_verification",
	"torch_utils": "yolozu.training.torch_utils",
}


def _make_legacy_module_proxy(fullname: str, target: str) -> types.ModuleType:
	module = types.ModuleType(fullname)
	module.__doc__ = f"Legacy module alias for {target}."
	module.__package__ = __name__

	def _resolve() -> types.ModuleType:
		real = importlib.import_module(target)
		sys.modules[fullname] = real
		return real

	def __getattr__(name: str):
		real = _resolve()
		return getattr(real, name)

	def __dir__():
		real = _resolve()
		return sorted(set(real.__dict__.keys()))

	module.__getattr__ = __getattr__
	module.__dir__ = __dir__
	return module


for _name, _target in _LEGACY_SUBMODULE_ALIASES.items():
	_fullname = f"{__name__}.{_name}"
	if _fullname not in sys.modules:
		_proxy = _make_legacy_module_proxy(_fullname, _target)
		sys.modules[_fullname] = _proxy
		setattr(sys.modules[__name__], _name, _proxy)
