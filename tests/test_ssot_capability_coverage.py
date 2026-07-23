import hashlib
import importlib.util
import json
import unittest
from collections import Counter
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

    def test_audit_registry_and_example_counts_match_current_sources(self) -> None:
        audit = (self.repo_root / "docs" / "ssot_capability_coverage_audit.md").read_text(
            encoding="utf-8"
        )
        manifest = json.loads((self.repo_root / "tools" / "manifest.json").read_text(encoding="utf-8"))
        tools = manifest["tools"]
        maturity = Counter(str(item["maturity"]) for item in tools)
        self.assertIn(
            f"{len(tools)} entries: {maturity['stable']} stable, "
            f"{maturity['experimental']} experimental, {maturity['research']} research",
            audit,
        )
        self.assertIn(f"Strict manifest validation passes for all {len(tools)} entries.", audit)

        script = self.repo_root / "tools" / "audit_docs_examples_drift.py"
        spec = importlib.util.spec_from_file_location("audit_docs_examples_drift", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader if spec is not None else None)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        checked_examples = sum(
            1
            for path_text in module.DEFAULT_DOCS
            for line in module._shell_lines_from_markdown(self.repo_root / path_text)
            if module._interesting_command(line)
        )
        self.assertIn(f"Public docs example audit passes {checked_examples} shell examples.", audit)

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
        config_ref = Path(baseline["adapter"]["config"])
        self.assertFalse(config_ref.is_absolute(), "baseline config path must be repository-relative")
        self.assertNotIn("..", config_ref.parts, "baseline config path must stay within the repository")
        config_path = self.repo_root / config_ref
        self.assertTrue(config_path.is_file(), f"baseline config file is missing: {config_ref}")
        observed = hashlib.sha256(config_path.read_bytes()).hexdigest()

        self.assertEqual(baseline["baseline_meta"]["config_hash"], observed)


if __name__ == "__main__":
    unittest.main()
