#!/usr/bin/env python3
"""Smoke test: verify both shim and canonical import paths work."""

# === Old (shim) import paths ===
from yolozu.dataset import build_manifest
from yolozu.adapter import DummyAdapter
from yolozu.coco_eval import build_coco_ground_truth
from yolozu.boxes import iou_cxcywh_norm_dict
print("All shim imports OK")

# === New canonical import paths ===
from yolozu.datasets.dataset import build_manifest as bm2
from yolozu.inference.adapter import DummyAdapter as da2
from yolozu.eval.coco_eval import build_coco_ground_truth as bcgt2
from yolozu.core.boxes import iou_cxcywh_norm_dict as iou2
print("All canonical imports OK")

# Verify identity (shim → canonical point to same objects)
assert build_manifest is bm2
assert DummyAdapter is da2
assert build_coco_ground_truth is bcgt2
assert iou_cxcywh_norm_dict is iou2
print("Identity checks passed - shims point to same objects")
