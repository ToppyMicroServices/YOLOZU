import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from yolozu.integrations import tool_runner
from yolozu.qualification import (
    QualificationError,
    QualificationResult,
    diff_qualification_packs,
    qualify_release,
    verify_qualification_pack,
)


class _Evaluation:
    def __init__(self, value, *, dry_run=False):
        self.value = value
        self.dry_run = dry_run

    def to_dict(self):
        return {
            "report_schema_version": 1,
            "status": "ok",
            "ok": True,
            "timestamp": "2099-01-01T00:00:00Z",
            "dataset": "/private/source/dataset",
            "split": "val",
            "split_requested": "val",
            "predictions": "/private/source/predictions.json",
            "bbox_format": "cxcywh_norm",
            "max_images": None,
            "normalization": {"classes": None, "assume_class_id_is_category_id": False},
            "validation": {"mode": "strict", "repair_enabled": False},
            "metrics": {
                "map50_95": self.value,
                "map50": self.value,
                "map75": self.value,
                "ar100": self.value,
            },
            "stats": [] if self.value is None else [self.value] * 12,
            "dry_run": self.dry_run,
            "counts": {
                "dataset_images_total": 1,
                "images": 1,
                "prediction_images_total": 1,
                "prediction_images_evaluated": 1,
                "prediction_images_excluded": 0,
                "selected_images_without_predictions": 0,
                "detections_input": 1,
                "detections": 1,
                "detections_excluded": 0,
            },
            "warnings": [],
        }


class TestReleaseQualification(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        (self.dataset / "sample.txt").write_text("dataset-v1\n", encoding="utf-8")
        self.predictions = self.root / "predictions.json"
        self.predictions.write_text('{"schema_version":1,"predictions":[]}\n', encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_packaged_qualification_schema_matches_docs(self):
        repo_root = Path(__file__).resolve().parents[1]
        docs_schema = repo_root / "docs" / "schemas" / "release_qualification.schema.json"
        packaged_schema = repo_root / "yolozu" / "data" / "schemas" / "release_qualification.schema.json"
        self.assertEqual(
            docs_schema.read_bytes(),
            packaged_schema.read_bytes(),
        )

    @patch("yolozu.api.evaluate_coco", return_value=_Evaluation(0.55))
    def test_pass_pack_is_portable_and_verifiable(self, evaluate):
        pack = self.root / "pack"
        result = qualify_release(
            self.dataset,
            self.predictions,
            pack,
            split="val",
            min_map50_95=0.5,
        )

        self.assertEqual(result.decision, "pass")
        self.assertTrue(verify_qualification_pack(pack).ok)
        self.assertEqual(evaluate.call_count, 1)
        combined = "\n".join(
            path.read_text(encoding="utf-8") for path in pack.glob("*.json")
        )
        self.assertNotIn(str(self.root), combined)
        self.assertFalse((pack / "predictions.json").exists())

    @patch("yolozu.api.evaluate_coco", return_value=_Evaluation(None, dry_run=True))
    def test_dry_run_never_passes(self, _evaluate):
        result = qualify_release(
            self.dataset,
            self.predictions,
            self.root / "dry-pack",
            dry_run=True,
            min_map50_95=0.0,
        )
        self.assertEqual(result.decision, "hold")

    @patch("yolozu.api.evaluate_coco", return_value=_Evaluation(0.55))
    def test_no_quality_threshold_never_passes(self, _evaluate):
        result = qualify_release(self.dataset, self.predictions, self.root / "ungated-pack")
        self.assertEqual(result.decision, "hold")

    @patch("yolozu.api.evaluate_coco", return_value=_Evaluation(0.55))
    def test_tampering_is_detected(self, _evaluate):
        pack = self.root / "pack"
        qualify_release(self.dataset, self.predictions, pack, min_map50_95=0.5)
        candidate = pack / "candidate_evaluation.json"
        candidate.write_text(candidate.read_text(encoding="utf-8") + " ", encoding="utf-8")

        verification = verify_qualification_pack(pack)

        self.assertFalse(verification.ok)
        self.assertIn("checksum mismatch: candidate_evaluation.json", verification.errors)

    @patch("yolozu.api.evaluate_coco")
    def test_baseline_drop_gate_and_compatible_diff(self, evaluate):
        evaluate.side_effect = [_Evaluation(0.50), _Evaluation(0.60)]
        first = qualify_release(
            self.dataset,
            self.predictions,
            self.root / "first",
            min_map50_95=0.4,
        )
        baseline_predictions = self.root / "baseline.json"
        baseline_predictions.write_text('{"schema_version":1,"predictions":[]}\n', encoding="utf-8")
        second = qualify_release(
            self.dataset,
            self.predictions,
            self.root / "second",
            min_map50_95=0.4,
        )
        self.assertTrue(first.passed)
        self.assertTrue(second.passed)

        comparison = diff_qualification_packs(first.pack_dir, second.pack_dir)

        self.assertTrue(comparison["ok"])
        self.assertTrue(comparison["compatible"])
        self.assertAlmostEqual(comparison["metrics"]["map50_95"]["delta"], 0.10)

    @patch("yolozu.api.evaluate_coco")
    def test_baseline_drop_gate_fails_on_regression(self, evaluate):
        evaluate.side_effect = [_Evaluation(0.50), _Evaluation(0.60)]
        baseline_predictions = self.root / "baseline.json"
        baseline_predictions.write_text('{"schema_version":1,"predictions":[]}\n', encoding="utf-8")

        result = qualify_release(
            self.dataset,
            self.predictions,
            self.root / "regressed",
            baseline_predictions=baseline_predictions,
            max_map50_95_drop=0.04,
        )

        self.assertEqual(result.decision, "fail")
        drop_check = next(check for check in result.qualification["checks"] if check["id"] == "max_map50_95_drop")
        self.assertEqual(drop_check["status"], "fail")

    def test_drop_threshold_requires_baseline(self):
        with self.assertRaises(QualificationError):
            qualify_release(
                self.dataset,
                self.predictions,
                self.root / "pack",
                max_map50_95_drop=0.01,
            )

    def test_mcp_runner_confines_paths_and_uses_shared_engine(self):
        qualification = {
            "decision": "hold",
            "checks": [],
        }
        result = QualificationResult(self.root / "reports" / "pack", "hold", qualification)
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            with (
                patch("yolozu.qualification.qualify_release", return_value=result) as create,
                patch("yolozu.integrations.tool_runner.collect_artifact_metadata", return_value={}),
            ):
                payload = tool_runner.qualify_release(
                    "data",
                    "predictions.json",
                    output_dir="reports/pack",
                )
        finally:
            os.chdir(previous)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["result"]["pack_dir"], "reports/pack")
        self.assertTrue(create.call_args.args[0].is_absolute())
        self.assertTrue(create.call_args.args[1].is_absolute())

    def test_mcp_runner_rejects_workspace_escape(self):
        with patch("yolozu.integrations.tool_runner.collect_artifact_metadata", return_value={}):
            payload = tool_runner.qualify_release(
                "../outside",
                "predictions.json",
                output_dir="reports/pack",
            )
        self.assertFalse(payload["ok"])
        self.assertIn("path traversal", payload["error"])

    def test_mcp_runner_rejects_descendant_symlink_escape(self):
        outside = self.root.parent / f"{self.root.name}-outside.txt"
        outside.write_text("outside\n", encoding="utf-8")
        linked_dataset = self.root / "linked-dataset"
        linked_dataset.mkdir()
        (linked_dataset / "outside.txt").symlink_to(outside)
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            with patch("yolozu.integrations.tool_runner.collect_artifact_metadata", return_value={}):
                payload = tool_runner.qualify_release(
                    "linked-dataset",
                    "predictions.json",
                    output_dir="reports/pack",
                )
        finally:
            os.chdir(previous)
            outside.unlink()

        self.assertFalse(payload["ok"])
        self.assertIn("escapes the workspace", payload["error"])


if __name__ == "__main__":
    unittest.main()
