#!/usr/bin/env python3
"""Smoke test: verify both shim and canonical import paths work."""

# === Old (shim) import paths ===
from yolozu.dataset import build_manifest
from yolozu.adapter import DummyAdapter
from yolozu.coco_eval import build_coco_ground_truth
from yolozu.model_fetch import list_models
from yolozu.boxes import iou_cxcywh_norm_dict
from yolozu.config import simple_yaml_load
from yolozu.symmetry import normalize_symmetry
from yolozu.gates import final_score
from yolozu.distillation import distill_predictions
from yolozu.metrics import symmetry_geodesic
from yolozu.splits import deterministic_split_paths
print("All shim imports OK")

# === New canonical import paths ===
from yolozu.datasets.dataset import build_manifest as bm2
from yolozu.inference.adapter import DummyAdapter as da2
from yolozu.eval.coco_eval import build_coco_ground_truth as bcgt2
from yolozu.core.boxes import iou_cxcywh_norm_dict as iou2
from yolozu.geometry.symmetry import normalize_symmetry as ns2
from yolozu.training.gates import final_score as fs2
from yolozu.training.distillation import distill_predictions as dp2
from yolozu.eval.metrics import symmetry_geodesic as sg2
from yolozu.datasets.splits import deterministic_split_paths as ds2
print("All canonical imports OK")

# Verify identity (shim → canonical point to same objects)
assert build_manifest is bm2
assert DummyAdapter is da2
assert build_coco_ground_truth is bcgt2
assert iou_cxcywh_norm_dict is iou2
print("Identity checks passed - shims point to same objects")
