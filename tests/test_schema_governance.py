import json
import unittest
from importlib import resources
from pathlib import Path

from yolozu.instance_segmentation_predictions import validate_instance_segmentation_predictions_payload
from yolozu.predictions import validate_predictions_payload
from yolozu.predictions.predictions import CURRENT_ENTRY_SCHEMA_VERSION
from yolozu.predictions.schema_governance import CURRENT_SCHEMA_VERSION
from yolozu.segmentation_predictions import validate_segmentation_predictions_payload


class TestSchemaGovernance(unittest.TestCase):
    def test_adaptive_routing_schema_copies_and_browser_are_current(self):
        repo_root = Path(__file__).resolve().parents[1]
        governance = (repo_root / "docs" / "schema_governance.md").read_text(encoding="utf-8")
        manifest = json.loads((repo_root / "tools" / "manifest.json").read_text(encoding="utf-8"))
        packaged_manifest = json.loads(
            (repo_root / "yolozu" / "data" / "manifest" / "tools_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        expected = {
            "image_job_spec_json": "image_job_spec.schema.json",
            "qualification_workload_profile_json": "qualification_workload_profile.schema.json",
            "environment_profile_json": "environment_profile.schema.json",
            "algorithm_bundle_spec_json": "algorithm_bundle_spec.schema.json",
            "algorithm_bundle_registry_json": "algorithm_bundle_registry.schema.json",
            "bundle_lifecycle_record_json": "bundle_lifecycle_record.schema.json",
            "support_profile_spec_json": "support_profile_spec.schema.json",
            "support_profile_record_json": "support_profile_record.schema.json",
            "local_artifact_inventory_json": "local_artifact_inventory.schema.json",
            "qualification_report_json": "qualification_report.schema.json",
            "evidence_activation_record_json": "evidence_activation_record.schema.json",
        }
        self.assertEqual(manifest, packaged_manifest)
        for contract_id, basename in expected.items():
            canonical = repo_root / "docs" / "schemas" / basename
            packaged = repo_root / "yolozu" / "data" / "schemas" / basename
            self.assertEqual(canonical.read_bytes(), packaged.read_bytes())
            self.assertIn(f"`docs/schemas/{basename}`", governance)
            self.assertEqual(
                manifest["contracts"][contract_id]["schema"],
                f"docs/schemas/{basename}",
            )

    def test_adaptive_routing_record_instances_have_one_packaged_ssot(self):
        repo_root = Path(__file__).resolve().parents[1]
        data_root = repo_root / "yolozu" / "data" / "adaptive_routing"
        registry = json.loads((data_root / "bundle_specs.json").read_text(encoding="utf-8"))
        self.assertEqual(registry["bundles"], [])
        self.assertEqual((data_root / "bundle_lifecycle.jsonl").read_bytes(), b"")
        self.assertEqual((data_root / "support_profiles.jsonl").read_bytes(), b"")
        self.assertEqual((data_root / "evidence_activation.jsonl").read_bytes(), b"")
        self.assertEqual(
            list((data_root / "qualification_reports").glob("*.json")), []
        )
        governance = (repo_root / "docs" / "schema_governance.md").read_text(
            encoding="utf-8"
        )
        for basename in (
            "bundle_specs.json",
            "bundle_lifecycle.jsonl",
            "support_profiles.jsonl",
            "evidence_activation.jsonl",
        ):
            self.assertIn(f"`yolozu/data/adaptive_routing/{basename}`", governance)
            self.assertFalse((repo_root / "docs" / "adaptive_routing" / basename).exists())
            packaged = (
                resources.files("yolozu.data")
                .joinpath("adaptive_routing")
                .joinpath(basename)
                .read_bytes()
            )
            self.assertEqual(packaged, (data_root / basename).read_bytes())

    def test_adaptive_evidence_contracts_are_required_by_ci_and_release(self):
        repo_root = Path(__file__).resolve().parents[1]
        build = (repo_root / ".github" / "workflows" / "build_and_test.yml").read_text(
            encoding="utf-8"
        )
        pre_push = (repo_root / "scripts" / "pre_push.sh").read_text(
            encoding="utf-8"
        )
        publish = (repo_root / ".github" / "workflows" / "publish.yml").read_text(
            encoding="utf-8"
        )
        for suite in (
            "tests.test_adaptive_evidence_contracts",
            "tests.test_schema_governance",
        ):
            self.assertIn(suite, build)
            self.assertIn(suite, pre_push)
        for resource in (
            "yolozu/data/schemas/local_artifact_inventory.schema.json",
            "yolozu/data/schemas/qualification_report.schema.json",
            "yolozu/data/schemas/evidence_activation_record.schema.json",
            "yolozu/data/adaptive_routing/evidence_activation.jsonl",
        ):
            self.assertGreaterEqual(publish.count(resource), 2)

    def test_predictions_schema_copies_declare_current_versions(self):
        repo_root = Path(__file__).resolve().parents[1]
        schema_paths = [
            repo_root / "docs" / "schemas" / "predictions.schema.json",
            repo_root / "schemas" / "predictions.schema.json",
            repo_root / "yolozu" / "data" / "schemas" / "predictions.schema.json",
        ]
        schemas = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in schema_paths
        ]
        self.assertEqual(schemas[1:], [schemas[0], schemas[0]])

        definitions = schemas[0]["$defs"]
        wrapper_version = definitions["predictions_wrapper"]["properties"]["schema_version"]
        entry_version = definitions["prediction_entry"]["properties"]["schema_version"]
        self.assertEqual(wrapper_version["maximum"], CURRENT_SCHEMA_VERSION)
        self.assertEqual(entry_version["maximum"], CURRENT_ENTRY_SCHEMA_VERSION)

    def test_predictions_wrapped_without_schema_version_is_legacy_warning(self):
        payload = {
            "predictions": [
                {
                    "image": "a.jpg",
                    "detections": [
                        {
                            "class_id": 0,
                            "score": 0.8,
                            "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                        }
                    ],
                }
            ]
        }
        res = validate_predictions_payload(payload, strict=False)
        self.assertTrue(any("schema_version missing" in w for w in res.warnings))

    def test_predictions_future_schema_version_rejected(self):
        payload = {
            "schema_version": 2,
            "predictions": [{"image": "a.jpg", "detections": [{"class_id": 0, "score": 0.8, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}]}],
        }
        with self.assertRaises(ValueError):
            validate_predictions_payload(payload, strict=False)

    def test_segmentation_future_schema_version_rejected(self):
        payload = {"schema_version": 2, "predictions": [{"id": "a", "mask": "a.png"}]}
        with self.assertRaises(ValueError):
            validate_segmentation_predictions_payload(payload)

    def test_instance_seg_future_schema_version_rejected(self):
        payload = {
            "schema_version": 2,
            "predictions": [{"image": "a.jpg", "instances": [{"class_id": 0, "score": 0.9, "mask": "a.png"}]}],
        }
        with self.assertRaises(ValueError):
            validate_instance_segmentation_predictions_payload(payload)


if __name__ == "__main__":
    unittest.main()
