import json
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Any

from yolozu.eval.depth_eval import evaluate_depth_arrays
from yolozu.keypoints_eval import evaluate_keypoints_pck
from yolozu.pose_eval import evaluate_pose
from yolozu.segmentation_eval import compute_confusion_matrix, compute_iou_metrics
from yolozu.simple_map import evaluate_map


GOLDEN_ROOT = Path(__file__).resolve().parent / "golden"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return value


class TestGoldenArtifactSuite(unittest.TestCase):
    def _load_case(self, task: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        case_root = GOLDEN_ROOT / task
        artifact = _load_json(case_root / "input_artifact.json")
        expected = _load_json(case_root / "expected_report.json")
        schema = _load_json(case_root / "schema.json")
        self.assertEqual(artifact.get("task"), task)
        self.assertEqual(expected.get("task"), task)
        self.assertEqual(schema.get("task"), task)
        return artifact, expected, schema

    def _assert_schema_keys(self, artifact: dict[str, Any], report: dict[str, Any], schema: dict[str, Any]) -> None:
        self.assertEqual(schema.get("schema_version"), 1)
        for key in schema.get("required_input_keys", []):
            self.assertIn(key, artifact)
        for key in schema.get("required_report_keys", []):
            self.assertIn(key, report)

    def _assert_expected_report(self, task: str, report: dict[str, Any]) -> None:
        artifact, expected, schema = self._load_case(task)
        self._assert_schema_keys(artifact, report, schema)
        self.assertEqual(_normalize(report), expected)

    def test_detect_golden_report(self) -> None:
        artifact, _, _ = self._load_case("detect")
        result = evaluate_map(
            artifact["records"],
            artifact["predictions"],
            iou_thresholds=artifact["iou_thresholds"],
        )
        report = {
            "kind": "yolozu_golden_detect_report",
            "schema_version": 1,
            "task": "detect",
            "metrics": {
                "map50": result.map50,
                "map50_95": result.map50_95,
            },
            "per_class": {str(cid): metrics for cid, metrics in sorted(result.per_class.items())},
        }
        self._assert_expected_report("detect", report)

    def test_segmentation_golden_report(self) -> None:
        np = self._require_numpy()
        artifact, _, _ = self._load_case("segmentation")
        conf, stats = compute_confusion_matrix(
            np.asarray(artifact["ground_truth_mask"]),
            np.asarray(artifact["prediction_mask"]),
            num_classes=int(artifact["num_classes"]),
            ignore_index=int(artifact["ignore_index"]),
        )
        report = {
            "kind": "yolozu_golden_segmentation_report",
            "schema_version": 1,
            "task": "segmentation",
            "metrics": compute_iou_metrics(conf, class_names=artifact["class_names"]),
            "confusion_matrix": conf.tolist(),
            "stats": asdict(stats),
        }
        self._assert_expected_report("segmentation", report)

    def test_keypoints_golden_report(self) -> None:
        artifact, _, _ = self._load_case("keypoints")
        result = evaluate_keypoints_pck(
            records=artifact["records"],
            predictions_index=artifact["predictions_index"],
            iou_threshold=float(artifact["iou_threshold"]),
            pck_threshold=float(artifact["pck_threshold"]),
            min_score=float(artifact["min_score"]),
        )
        report = {
            "kind": "yolozu_golden_keypoints_report",
            "schema_version": 1,
            "task": "keypoints",
            **result,
        }
        self._assert_expected_report("keypoints", report)

    def test_depth_golden_report(self) -> None:
        np = self._require_numpy()
        artifact, _, _ = self._load_case("depth")
        report = evaluate_depth_arrays(
            pred=np.asarray(artifact["prediction_depth"], dtype=np.float32),
            gt=np.asarray(artifact["ground_truth_depth"], dtype=np.float32),
            mask=np.asarray(artifact["mask"], dtype=bool),
            align=artifact["align"],
        )
        report = {**report, "task": "depth"}
        self._assert_expected_report("depth", report)

    def test_pose6d_golden_report(self) -> None:
        artifact, _, _ = self._load_case("pose6d")
        result = evaluate_pose(
            artifact["records"],
            artifact["predictions"],
            iou_threshold=float(artifact["iou_threshold"]),
            min_score=float(artifact["min_score"]),
            success_rot_deg=float(artifact["success_rot_deg"]),
            success_trans=float(artifact["success_trans"]),
            keep_per_image=10,
        )
        report = {
            "kind": "yolozu_golden_pose6d_report",
            "schema_version": 1,
            "task": "pose6d",
            "metrics": result.metrics,
            "counts": result.counts,
            "per_image": result.per_image,
            "warnings": result.warnings,
        }
        self._assert_expected_report("pose6d", report)

    def _require_numpy(self):
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"numpy unavailable for golden artifact suite: {exc}")
        return np


if __name__ == "__main__":
    unittest.main()
