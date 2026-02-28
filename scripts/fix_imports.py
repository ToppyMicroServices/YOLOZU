#!/usr/bin/env python3
"""Fix relative imports in moved files — convert cross-package relatives to absolute."""
import re
from pathlib import Path

# Map old module names to their new canonical location
MODULE_MAP = {
    # core/
    "config": "yolozu.core.config",
    "canonical": "yolozu.core.canonical",
    "boxes": "yolozu.core.boxes",
    "keypoints": "yolozu.core.keypoints",
    "image_keys": "yolozu.core.image_keys",
    "image_size": "yolozu.core.image_size",
    "letterbox": "yolozu.core.letterbox",
    "resources": "yolozu.core.resources",
    "run_record": "yolozu.core.run_record",
    "doctor": "yolozu.core.doctor",
    "eval_protocol": "yolozu.core.eval_protocol",
    "scenario_suite": "yolozu.core.scenario_suite",
    "scenarios_cli": "yolozu.core.scenarios_cli",
    "cli_args": "yolozu.core.cli_args",
    # datasets/
    "dataset": "yolozu.datasets.dataset",
    "dataset_fetch": "yolozu.datasets.dataset_fetch",
    "dataset_validator": "yolozu.datasets.dataset_validator",
    "segmentation_dataset": "yolozu.datasets.segmentation_dataset",
    "migrate": "yolozu.datasets.migrate",
    "coco_convert": "yolozu.datasets.coco_convert",
    "splits": "yolozu.datasets.splits",
    "imports": "yolozu.datasets.imports",
    # eval/
    "coco_eval": "yolozu.eval.coco_eval",
    "coco_keypoints_eval": "yolozu.eval.coco_keypoints_eval",
    "keypoints_eval": "yolozu.eval.keypoints_eval",
    "pose_eval": "yolozu.eval.pose_eval",
    "segmentation_eval": "yolozu.eval.segmentation_eval",
    "instance_segmentation_eval": "yolozu.eval.instance_segmentation_eval",
    "simple_map": "yolozu.eval.simple_map",
    "long_tail_metrics": "yolozu.eval.long_tail_metrics",
    "continual_metrics": "yolozu.eval.continual_metrics",
    "synthgen_eval": "yolozu.eval.synthgen_eval",
    "benchmark": "yolozu.eval.benchmark",
    "metrics": "yolozu.eval.metrics",
    "metrics_report": "yolozu.eval.metrics_report",
    # predictions/
    "predictions": "yolozu.predictions.predictions",
    "predictions_transform": "yolozu.predictions.predictions_transform",
    "predictions_parity": "yolozu.predictions.predictions_parity",
    "segmentation_predictions": "yolozu.predictions.segmentation_predictions",
    "instance_segmentation_predictions": "yolozu.predictions.instance_segmentation_predictions",
    "instance_segmentation_report": "yolozu.predictions.instance_segmentation_report",
    "export": "yolozu.predictions.export",
    "schema_governance": "yolozu.predictions.schema_governance",
    # inference/
    "adapter": "yolozu.inference.adapter",
    "runner": "yolozu.inference.runner",
    "predict_images": "yolozu.inference.predict_images",
    "inference": "yolozu.inference.inference",
    "inference_utils": "yolozu.inference.inference_utils",
    "pipeline": "yolozu.inference.pipeline",
    "onnxrt_export": "yolozu.inference.onnxrt_export",
    "onnxrt_quantize": "yolozu.inference.onnxrt_quantize",
    "model_fetch": "yolozu.inference.model_fetch",
    # geometry/
    "geometry": "yolozu.geometry.geometry",
    "intrinsics": "yolozu.geometry.intrinsics",
    "math3d": "yolozu.geometry.math3d",
    "constraints": "yolozu.geometry.constraints",
    "symmetry": "yolozu.geometry.symmetry",
    "jitter": "yolozu.geometry.jitter",
    "template_verification": "yolozu.geometry.template_verification",
    # training/
    "distillation": "yolozu.training.distillation",
    "sdft": "yolozu.training.sdft",
    "continual_regularizers": "yolozu.training.continual_regularizers",
    "long_tail_recipe": "yolozu.training.long_tail_recipe",
    "replay_buffer": "yolozu.training.replay_buffer",
    "gates": "yolozu.training.gates",
    "map_targets": "yolozu.training.map_targets",
}

# Which package does each new subpackage belong to?
def _package_of(abs_path: str) -> str:
    """e.g. 'yolozu.core.config' -> 'core'"""
    parts = abs_path.split(".")
    if len(parts) >= 3:
        return parts[1]
    return ""

def fix_file(filepath: Path) -> int:
    """Fix relative imports in a single file. Returns number of fixes."""
    # Determine which subpackage this file is in
    rel = filepath.relative_to(Path("yolozu"))
    file_pkg = str(rel.parts[0]) if len(rel.parts) > 1 else ""

    text = filepath.read_text(encoding="utf-8")
    lines = text.split("\n")
    fixes = 0

    for i, line in enumerate(lines):
        # Match: from .module_name import ...
        m = re.match(r'^(\s*from \.)([a-zA-Z_][a-zA-Z0-9_]*)(\s+import\s+.*)$', line)
        if not m:
            continue

        prefix, module_name, suffix = m.group(1), m.group(2), m.group(3)
        indent = prefix.replace("from .", "")

        if module_name not in MODULE_MAP:
            continue

        target_abs = MODULE_MAP[module_name]
        target_pkg = _package_of(target_abs)

        # If same package, the relative import is still correct
        if target_pkg == file_pkg:
            continue

        # Cross-package: convert to absolute import
        new_line = f"{indent}from {target_abs}{suffix}"
        lines[i] = new_line
        fixes += 1
        print(f"  {filepath}:{i+1}: from .{module_name} -> from {target_abs}")

    if fixes:
        filepath.write_text("\n".join(lines), encoding="utf-8")

    return fixes

# Process all moved files
total = 0
for subpkg in ("datasets", "eval", "predictions", "inference", "geometry", "training", "core"):
    pkg_dir = Path("yolozu") / subpkg
    for py_file in sorted(pkg_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        total += fix_file(py_file)

print(f"\nFixed {total} cross-package relative imports")
