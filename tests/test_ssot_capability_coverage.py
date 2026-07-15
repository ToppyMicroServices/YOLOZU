import hashlib
import json
import unittest
from pathlib import Path


class TestSsotCapabilityCoverage(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_audit_maps_every_production_capability_area(self) -> None:
        readiness = (self.repo_root / "docs" / "production_readiness.md").read_text(encoding="utf-8")
        audit = (self.repo_root / "docs" / "ssot_capability_coverage_audit.md").read_text(encoding="utf-8")

        start_heading = "## Capability map"
        end_heading = "## Version Compatibility Matrix"
        self.assertIn(start_heading, readiness, "production readiness is missing the capability map heading")
        self.assertIn(end_heading, readiness, "production readiness is missing the version matrix heading")
        table = readiness.partition(start_heading)[2].partition(end_heading)[0]
        areas = []
        for line in table.splitlines():
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if cells and cells[0] not in {"Area", "---"} and not set(cells[0]) <= {"-", ":"}:
                areas.append(cells[0])

        self.assertGreater(len(areas), 0)
        for area in areas:
            with self.subTest(area=area):
                self.assertIn(area, audit)

    def test_audit_has_required_coverage_dimensions_and_followups(self) -> None:
        audit = (self.repo_root / "docs" / "ssot_capability_coverage_audit.md").read_text(encoding="utf-8")
        for heading in (
            "Capability",
            "Maturity",
            "Implementation",
            "CLI",
            "Manifest / packaged copy",
            "Docs",
            "Tests / evidence",
            "Result / follow-up",
        ):
            with self.subTest(heading=heading):
                self.assertIn(heading, audit)

        for issue_id in (
            "YOLOZU-ll2.3",
            "YOLOZU-ll2.5",
            "YOLOZU-ll2.11",
            "YOLOZU-ll2.12",
            "YOLOZU-ll2.15",
        ):
            with self.subTest(issue_id=issue_id):
                self.assertIn(issue_id, audit)

    def test_spec_and_readiness_keep_schema_and_mixed_lane_boundaries_explicit(self) -> None:
        spec = (self.repo_root / "docs" / "yolozu_spec.md").read_text(encoding="utf-8")
        readiness = (self.repo_root / "docs" / "production_readiness.md").read_text(encoding="utf-8")

        self.assertIn("Predictions wrapper schema version `1`; canonical entry schema version `2`", spec)
        self.assertIn("not a production-maturity declaration", spec)
        self.assertIn("entrypoint-level `maturity`", readiness)
        self.assertIn("do not infer the maturity of every subcommand or flag", readiness)

    def test_reference_adapter_baseline_tracks_the_config_fingerprint(self) -> None:
        baseline = json.loads(
            (self.repo_root / "baselines" / "reference_adapter" / "rtdetr_pose_smoke_val.json").read_text(
                encoding="utf-8"
            )
        )
        config_path = self.repo_root / baseline["adapter"]["config"]
        observed = hashlib.sha256(config_path.read_bytes()).hexdigest()

        self.assertEqual(baseline["baseline_meta"]["config_hash"], observed)


if __name__ == "__main__":
    unittest.main()
