import hashlib
import json
import re
import unittest
from pathlib import Path


class TestExternalRuntimeCompatibleHostEvidenceReports(unittest.TestCase):
    EVIDENCE_FILENAMES = tuple(
        f"external_runtime_compatible_host_{suffix}_2026-07-30.json"
        for suffix in (
            "primary",
            "primary_dataset",
            "primary_tao",
            "primary_workflow",
            "independent",
            "independent_dataset",
            "independent_tao",
            "independent_workflow",
        )
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.reports_root = cls.repo_root / "reports"
        cls.records = {}
        for role in ("primary", "independent"):
            cls.records[role] = {
                "qualification": cls._load(
                    f"external_runtime_compatible_host_{role}_2026-07-30.json"
                ),
                "dataset": cls._load(
                    f"external_runtime_compatible_host_{role}_dataset_2026-07-30.json"
                ),
                "tao": cls._load(
                    f"external_runtime_compatible_host_{role}_tao_2026-07-30.json"
                ),
                "workflow": cls._load(
                    f"external_runtime_compatible_host_{role}_workflow_2026-07-30.json"
                ),
            }

    @classmethod
    def _load(cls, name: str) -> dict:
        return json.loads((cls.reports_root / name).read_text(encoding="utf-8"))

    def test_both_runs_pass_all_fail_closed_open_source_gates(self) -> None:
        expected_ids = ["yolox", "mmdetection", "mmpose", "mmseg"]
        expected_exports = {
            "yolox": "predictions interface contract",
            "mmdetection": "predictions interface contract",
            "mmpose": "predictions interface contract",
            "mmseg": "segmentation predictions interface contract",
        }
        for role, records in self.records.items():
            qualification = records["qualification"]
            with self.subTest(role=role):
                self.assertEqual(qualification["schema_version"], 1)
                self.assertEqual(
                    qualification["kind"],
                    "compatible_host_external_runtime_qualification",
                )
                for field in (
                    "all_training_executed",
                    "all_checkpoints_recorded",
                    "all_resource_usage_recorded",
                    "all_handoff_contracts_validated",
                    "qualification_passed",
                ):
                    self.assertTrue(qualification[field], field)
                self.assertTrue(qualification["environment"]["cuda_available"])
                self.assertEqual(qualification["environment"]["gpu"], "Tesla T4")
                self.assertEqual(
                    [lane["id"] for lane in qualification["lanes"]], expected_ids
                )
                for lane in qualification["lanes"]:
                    self.assertEqual(lane["returncode"], 0)
                    self.assertTrue(lane["training_executed"])
                    self.assertEqual(
                        lane["execution_status"]["state"], "executed"
                    )
                    self.assertTrue(
                        lane["execution_status"]["real_training_executed"]
                    )
                    checkpoint = lane["checkpoint_evidence"]
                    self.assertGreater(checkpoint["bytes"], 0)
                    self.assertRegex(checkpoint["sha256"], r"^[0-9a-f]{64}$")
                    usage = lane["resource_usage"]
                    self.assertGreater(usage["wall_seconds"], 0.0)
                    self.assertGreater(usage["child_peak_rss_bytes"], 0)
                    self.assertGreaterEqual(usage["child_user_cpu_seconds"], 0.0)
                    self.assertGreaterEqual(usage["child_system_cpu_seconds"], 0.0)
                    handoff = lane["handoff_validation"]
                    self.assertTrue(handoff["passed"])
                    self.assertEqual(
                        set(handoff["stages"]), {"resume", "export", "eval", "parity"}
                    )
                    self.assertTrue(
                        all(stage["ok"] for stage in handoff["stages"].values())
                    )
                    self.assertEqual(
                        handoff["stages"]["export"]["output_type"],
                        expected_exports[lane["id"]],
                    )

    def test_dataset_and_runtime_sources_match_across_runs(self) -> None:
        primary = self.records["primary"]
        independent = self.records["independent"]
        self.assertEqual(
            primary["qualification"]["sources"],
            independent["qualification"]["sources"],
        )
        self.assertEqual(
            primary["qualification"]["environment"]["versions"],
            independent["qualification"]["environment"]["versions"],
        )
        self.assertEqual(
            primary["dataset"]["tree_sha256"],
            independent["dataset"]["tree_sha256"],
        )
        for records in (primary, independent):
            dataset = records["dataset"]
            self.assertEqual(dataset["kind"], "external_runtime_smoke_datasets")
            self.assertEqual(dataset["images"], 2)
            self.assertEqual(dataset["instances"], 28)
            self.assertEqual(dataset["classes"], 5)
            self.assertIn("runtime availability only", dataset["ground_truth"]["keypoints"])
            self.assertIn(
                "runtime availability only",
                dataset["ground_truth"]["segmentation"],
            )

    def test_tao_vendor_completion_and_checkpoint_are_reproduced(self) -> None:
        primary = self.records["primary"]["tao"]
        independent = self.records["independent"]["tao"]
        for role, payload in (
            ("primary", primary),
            ("independent", independent),
        ):
            with self.subTest(role=role):
                self.assertEqual(payload["schema_version"], 1)
                self.assertEqual(
                    payload["kind"], "tao_compatible_host_runtime_evidence"
                )
                self.assertEqual(
                    payload["vendor_completion_record"]["message"],
                    "Train finished successfully.",
                )
                self.assertNotEqual(
                    payload["vendor_completion_record"]["status"], "FAILURE"
                )
                self.assertGreater(payload["wall_seconds"], 0.0)
                self.assertGreater(payload["checkpoint"]["bytes"], 0)
                self.assertRegex(
                    payload["checkpoint"]["sha256"], r"^[0-9a-f]{64}$"
                )
        self.assertEqual(primary["image"], independent["image"])
        self.assertEqual(primary["image_digest"], independent["image_digest"])

    def test_workflows_use_the_same_source_and_bind_tracked_summaries(self) -> None:
        source_commits = set()
        for role, records in self.records.items():
            workflow = records["workflow"]
            source_commits.add(workflow["source_commit"])
            with self.subTest(role=role):
                self.assertEqual(workflow["open_source_lanes"], "success")
                self.assertEqual(workflow["tao_lane"], "success")
                artifacts = {row["path"]: row for row in workflow["artifacts"]}
                bindings = {
                    "qualification_summary.json": "qualification",
                    "datasets/preparation_report.json": "dataset",
                    "tao/runtime_evidence.json": "tao",
                }
                for artifact_path, record_name in bindings.items():
                    raw = json.dumps(
                        records[record_name], indent=2, sort_keys=True
                    ).encode("utf-8") + b"\n"
                    self.assertEqual(
                        artifacts[artifact_path]["sha256"],
                        hashlib.sha256(raw).hexdigest(),
                    )
                self.assertTrue(
                    any(
                        re.search(r"\.(pt|pth)$", artifact_path)
                        for artifact_path in artifacts
                    )
                )
        self.assertEqual(
            source_commits,
            {"806496d453f8adfb550a9cc1e994182fa04e64b2"},
        )

    def test_report_preserves_availability_quality_boundary(self) -> None:
        report = (
            self.reports_root
            / "external_runtime_compatible_host_evidence_2026-07-30.md"
        ).read_text(encoding="utf-8")
        for run_id in ("30546919180", "30548569775"):
            self.assertIn(run_id, report)
        self.assertIn("Experimental", report)
        self.assertIn("does not establish training quality", report)
        self.assertIn("structural handoff", report)
        for filename in self.EVIDENCE_FILENAMES:
            self.assertIn(filename, report)

    def test_manifest_and_web_provenance_cover_every_raw_evidence_file(self) -> None:
        expected_paths = {
            f"reports/{filename}" for filename in self.EVIDENCE_FILENAMES
        }
        for manifest_path in (
            self.repo_root / "tools" / "manifest.json",
            self.repo_root / "yolozu" / "data" / "manifest" / "tools_manifest.json",
        ):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            tool = next(
                entry
                for entry in manifest["tools"]
                if entry["id"] == "run_external_runtime_gpu_qualification"
            )
            self.assertTrue(expected_paths.issubset(set(tool["docs"])))

        provenance = json.loads(
            (
                self.repo_root
                / "docs"
                / "generated"
                / "web_docs"
                / "provenance.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(
            expected_paths.issubset(set(provenance["source_hashes"]))
        )


if __name__ == "__main__":
    unittest.main()
