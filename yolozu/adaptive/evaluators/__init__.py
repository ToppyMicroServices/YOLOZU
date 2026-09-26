"""Repository-owned evaluators for adaptive image qualification."""

from .coco_bbox_simple_map import (
    COCO_BBOX_SIMPLE_MAP_EVALUATOR_ID,
    CocoBBoxSimpleMapEvaluator,
    create_coco_bbox_simple_map_evaluator,
)

__all__ = [
    "COCO_BBOX_SIMPLE_MAP_EVALUATOR_ID",
    "CocoBBoxSimpleMapEvaluator",
    "create_coco_bbox_simple_map_evaluator",
]
