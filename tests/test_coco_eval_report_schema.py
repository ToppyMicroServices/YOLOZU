import json
from pathlib import Path
import unittest

from yolozu.api import PredictionsValidationError, _failure_report


class TestCocoEvalReportSchema(unittest.TestCase):
    def test_packaged_schema_matches_canonical_schema(self):
        repo_root = Path(__file__).resolve().parents[1]
        canonical = repo_root / "docs" / "schemas" / "coco_eval_report.schema.json"
        packaged = repo_root / "yolozu" / "data" / "schemas" / "coco_eval_report.schema.json"

        self.assertTrue(packaged.is_file())
        self.assertEqual(
            json.loads(packaged.read_text(encoding="utf-8")),
            json.loads(canonical.read_text(encoding="utf-8")),
        )

    def test_failure_report_without_split_matches_declared_nullable_contract(self):
        repo_root = Path(__file__).resolve().parents[1]
        schema = json.loads(
            (repo_root / "docs" / "schemas" / "coco_eval_report.schema.json").read_text(encoding="utf-8")
        )
        report = _failure_report(
            PredictionsValidationError("invalid"),
            dataset="data/smoke",
            predictions="predictions.json",
            split=None,
            bbox_format="cxcywh_norm",
            max_images=None,
            dry_run=True,
            repair=False,
        )

        self.assertIsNone(report["split"])
        self.assertIn("null", schema["properties"]["split"]["type"])
        for required in schema["required"]:
            self.assertIn(required, report)
        self.assertEqual(report["status"], "failed")
        self.assertIn("error", report)

    def test_eval_manifest_declares_real_coco_dependency_and_dry_run_boundary(self):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (repo_root / "tools" / "manifest.json").read_text(encoding="utf-8")
        )
        entry = next(tool for tool in manifest["tools"] if tool["id"] == "eval_coco")

        self.assertIn("pycocotools", entry["requires"]["python_packages"])
        self.assertIn("--dry-run", entry["requires"]["notes"])
        self.assertIn("does not require pycocotools", entry["summary"])


if __name__ == "__main__":
    unittest.main()
