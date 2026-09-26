from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tests.test_adaptive_bundle_contracts import _bundle_payload
from yolozu.adaptive.bundles import validate_algorithm_bundle_spec
from yolozu.adaptive.canonical import canonical_json_v1
from yolozu.adaptive.contracts import validate_image_job_spec
from yolozu.adaptive.evaluators.coco_bbox_simple_map import (
    COCO_BBOX_SIMPLE_MAP_EVALUATOR_ID,
    EVALUATION_PROTOCOL_SHA256,
    create_coco_bbox_simple_map_evaluator,
)
from yolozu.adaptive.inventory import pin_decoded_inputs


def _ground_truth(*, width: int = 10) -> dict[str, object]:
    return {
        "info": {},
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": "one.png",
                "width": width,
                "height": 10,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [2, 2, 4, 4],
                "area": 16,
                "iscrowd": 0,
            }
        ],
        "categories": [
            {"id": 1, "name": "cat", "supercategory": "animal"},
            {"id": 2, "name": "dog", "supercategory": "animal"},
        ],
    }


def _job(dataset_sha256: str) -> object:
    return validate_image_job_spec(
        {
            "schema_version": 1,
            "task": "object_detection",
            "prompt_mode": "fixed_classes",
            "fixed_classes": ["cat", "dog"],
            "input_mode": "single_image",
            "execution_mode": "batch",
            "batch_size": 1,
            "concurrency": 1,
            "max_images": 1,
            "max_results_per_image": 100,
            "job_timeout_seconds": 60,
            "ranking_policy": "accuracy_first",
            "allowed_maturities": ["Experimental"],
            "network_policy": "deny",
            "compute_policy": "auto",
            "quality_requirement": {
                "metric_id": "simple_bbox_map50_95",
                "direction": "higher_is_better",
                "threshold": "0.2",
                "evaluation_dataset_id": "local_coco_instances_v1",
                "evaluation_dataset_sha256": dataset_sha256,
                "evaluation_protocol_sha256": EVALUATION_PROTOCOL_SHA256,
                "evaluation_vocabulary_id": "example-classes",
            },
        }
    )


class TestAdaptiveCocoEvaluator(unittest.TestCase):
    def test_perfect_box_returns_one_with_exact_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            image_path = workspace / "one.png"
            ground_truth_path = workspace / "instances.json"
            Image.new("RGB", (10, 10)).save(image_path)
            ground_truth_path.write_text(
                json.dumps(_ground_truth(), separators=(",", ":")),
                encoding="utf-8",
            )
            bundle = validate_algorithm_bundle_spec(_bundle_payload())
            with pin_decoded_inputs(
                image_path,
                input_mode="single_image",
                workspace_root=workspace,
                max_images=1,
            ) as inputs:
                evaluator = create_coco_bbox_simple_map_evaluator(
                    Path("instances.json"),
                    workspace.resolve(),
                    inputs,
                    bundle,
                )
                handoff = canonical_json_v1(
                    {
                        "schema_version": 1,
                        "handoff_id": "image_result_mask_handoff_v1",
                        "handoff_version": 1,
                        "input_index": 0,
                        "task": "object_detection",
                        "results": [
                            {
                                "result_index": 0,
                                "score": "0.9",
                                "bbox": ["0.2", "0.2", "0.6", "0.6"],
                                "request_index": 0,
                                "label": "cat",
                            }
                        ],
                    }
                )
                value = evaluator.evaluate(
                    predictions=(handoff,),
                    job=_job(evaluator.evaluation_dataset_sha256),
                    bundle=bundle,
                )
            self.assertEqual(evaluator.evaluator_id, COCO_BBOX_SIMPLE_MAP_EVALUATOR_ID)
            self.assertEqual(evaluator.evaluation_protocol_sha256, EVALUATION_PROTOCOL_SHA256)
            self.assertEqual(evaluator.evaluation_vocabulary_id, "example-classes")
            self.assertEqual(value, "1")

    def test_rejects_ground_truth_dimension_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            image_path = workspace / "one.png"
            ground_truth_path = workspace / "instances.json"
            Image.new("RGB", (10, 10)).save(image_path)
            ground_truth_path.write_text(json.dumps(_ground_truth(width=11)), encoding="utf-8")
            with pin_decoded_inputs(
                image_path,
                input_mode="single_image",
                workspace_root=workspace,
                max_images=1,
            ) as inputs, self.assertRaisesRegex(ValueError, "dimensions"):
                create_coco_bbox_simple_map_evaluator(
                    Path("instances.json"),
                    workspace.resolve(),
                    inputs,
                    validate_algorithm_bundle_spec(_bundle_payload()),
                )

    def test_rejects_ground_truth_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            image_path = workspace / "one.png"
            target = workspace / "target.json"
            linked = workspace / "instances.json"
            Image.new("RGB", (10, 10)).save(image_path)
            target.write_text(json.dumps(_ground_truth()), encoding="utf-8")
            linked.symlink_to(target.name)
            with pin_decoded_inputs(
                image_path,
                input_mode="single_image",
                workspace_root=workspace,
                max_images=1,
            ) as inputs, self.assertRaisesRegex(ValueError, "symlink"):
                create_coco_bbox_simple_map_evaluator(
                    Path("instances.json"),
                    workspace.resolve(),
                    inputs,
                    validate_algorithm_bundle_spec(_bundle_payload()),
                )


if __name__ == "__main__":
    unittest.main()
