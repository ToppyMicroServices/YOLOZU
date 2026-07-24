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

    def test_audit_records_completed_public_pypi_fresh_install_matrix(self) -> None:
        audit = (self.repo_root / "docs" / "ssot_capability_coverage_audit.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("10/10 public-PyPI jobs on Linux/macOS", audit)
        self.assertIn("workflow run 29421807474", audit)
        self.assertIn("Resolved under `YOLOZU-ll2.3`", audit)
        self.assertNotIn(
            "Public fresh-install proof is incomplete across the supported OS/Python matrix",
            audit,
        )

    def test_spec_and_readiness_keep_schema_and_mixed_lane_boundaries_explicit(self) -> None:
        spec = (self.repo_root / "docs" / "yolozu_spec.md").read_text(encoding="utf-8")
        readiness = (self.repo_root / "docs" / "production_readiness.md").read_text(encoding="utf-8")

        self.assertIn("Predictions wrapper schema version `1`; canonical entry schema version `2`", spec)
        self.assertIn("not a production-maturity declaration", spec)
        self.assertIn("entrypoint-level `maturity`", readiness)
        self.assertIn("do not infer the maturity of every subcommand or flag", readiness)

    def test_every_spec_capability_has_an_explicit_maturity_boundary(self) -> None:
        spec = (self.repo_root / "docs" / "yolozu_spec.md").read_text(encoding="utf-8")
        table = spec.partition("### Capability maturity boundaries")[2].partition("### 1) Dataset I/O")[0]
        expected_rows = (
            "| Dataset I/O | Deferred as a standalone capability;",
            "| Mask-only label derivation | Deferred as a standalone capability;",
            "| Reference trainer | Stable reference lane;",
            "| Backbone/neck swap boundary | Stable only within the reference trainer interface boundary;",
            "| Inference constraints | Deferred as a standalone capability;",
            "| Template verification and gating | Deferred as a standalone capability;",
            "| Predictions JSON interface contract | Stable |",
            "| Evaluation harness | Stable for validation/evaluation of existing wrapped predictions;",
            "| TTA | Experimental and opt-in |",
            "| TTT | Research and opt-in |",
            "| CLI convenience | Mixed by capability;",
        )
        for row in expected_rows:
            with self.subTest(row=row):
                self.assertIn(row, table)

    def test_stable_parent_entrypoints_do_not_promote_opt_in_lanes(self) -> None:
        for rel in ("tools/manifest.json", "yolozu/data/manifest/tools_manifest.json"):
            manifest = json.loads((self.repo_root / rel).read_text(encoding="utf-8"))
            tools = {tool["id"]: tool for tool in manifest["tools"]}

            self.assertEqual(tools["yolozu"]["maturity"], "stable")
            self.assertIn("does not promote Experimental or Research", tools["yolozu"]["summary"])
            self.assertEqual(tools["export_predictions"]["maturity"], "stable")
            self.assertIn(
                "acceleration flags require backend/device qualification",
                tools["export_predictions"]["summary"],
            )
            self.assertIn("TTA remains Experimental", tools["export_predictions"]["summary"])
            self.assertIn("TTT remains Research", tools["export_predictions"]["summary"])
            self.assertIn("parent maturity does not promote", tools["export_predictions"]["summary"])

        readiness = (self.repo_root / "docs" / "production_readiness.md").read_text(encoding="utf-8")
        generated = (self.repo_root / "docs" / "generated" / "cli_reference.md").read_text(encoding="utf-8")
        tools_index = (self.repo_root / "docs" / "tools_index.md").read_text(encoding="utf-8")
        support_matrix = (self.repo_root / "docs" / "tta_support_matrix.md").read_text(encoding="utf-8")
        adapter = (self.repo_root / "docs" / "adapter_contract.md").read_text(encoding="utf-8")
        training = (self.repo_root / "docs" / "training_inference_export.md").read_text(encoding="utf-8")
        manual_cli = (self.repo_root / "manual" / "chapters" / "04_cli_reference.tex").read_text(
            encoding="utf-8"
        )
        manual_manifest = (
            self.repo_root / "manual" / "chapters" / "18_manifest_driven_docs.tex"
        ).read_text(encoding="utf-8")
        self.assertIn("Maturity applies at the narrowest declared surface", readiness)
        self.assertIn("Manifest `maturity` is entrypoint-level metadata, not a transitive guarantee", readiness)
        self.assertIn(
            "segmentation, keypoints, depth, and pose6d have artifact-backed real eval/parity lanes",
            readiness,
        )
        self.assertIn("TTA remains Experimental", generated)
        self.assertIn("TTT remains Research", generated)
        self.assertIn("Maturity is entrypoint-level and is not", tools_index)
        self.assertIn("Experimental opt-in TTA extensions", tools_index)
        self.assertIn("enable with `--tta`", tools_index)
        self.assertIn("`--tta-mode model` reruns one augmented branch for `rtdetr_pose`", tools_index)
        self.assertIn("Non-parameter-updating TTA is **Experimental** and opt-in", support_matrix)
        self.assertIn("Parameter-updating TTT is **Research** and opt-in", support_matrix)
        self.assertIn("**Interface-contract-first**", support_matrix)
        self.assertNotIn("**Contract-first**", support_matrix)
        self.assertIn("`--tta` uses `postprocess` by default", support_matrix)
        self.assertIn(
            "`--tta --tta-mode model` reruns one horizontally flipped inference branch",
            support_matrix,
        )
        self.assertIn(
            "`rtdetr_pose`-only `model` mode reruns one horizontally flipped inference",
            training,
        )
        self.assertIn("parameter-updating TTT remains a separate Research lane", training)
        self.assertNotIn("It does not rerun the model on augmented inputs.", training)
        self.assertIn("retain their separately declared capability or backend", manual_cli)
        self.assertIn("retain their separately declared capability or", manual_manifest)
        self.assertIn("does not promote the Research TTT", adapter)

        audit = (self.repo_root / "docs" / "ssot_capability_coverage_audit.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Corrected under `YOLOZU-ll2.12`", audit)
        self.assertIn("Corrected under `YOLOZU-ll2.28`", audit)
        self.assertIn("reject the combination before report, artifact, or backend writes", audit)

        readme_markers = {
            "README.md": "TTA Experimental, and TTT Research.",
            "Readme_jp.md": "TTA は Experimental、TTT は Research",
            "Readme_zh.md": "TTA 为 Experimental，TTT 为 Research。",
        }
        for rel, marker in readme_markers.items():
            with self.subTest(readme=rel):
                readme = (self.repo_root / rel).read_text(encoding="utf-8")
                self.assertIn(marker, readme)
                self.assertIn("manifest entry", readme)

    def test_research_ttt_design_docs_do_not_claim_production_readiness(self) -> None:
        for rel in (
            "docs/cotta_design_spec.md",
            "docs/eata_design_spec.md",
            "docs/sar_design_spec.md",
        ):
            with self.subTest(doc=rel):
                text = (self.repo_root / rel).read_text(encoding="utf-8")
                self.assertIn("Research", text)
                self.assertIn("does not establish", text)
                self.assertIn("production readiness", text)
                self.assertNotIn("production-safe", text)

        manual = (self.repo_root / "manual" / "chapters" / "15_ttt_tent_mim.tex").read_text(
            encoding="utf-8"
        )
        self.assertIn("safety-bounded phase-1 Research scope for EATA", manual)
        self.assertIn("safety-bounded phase-1 Research rollout scope for SAR", manual)
        self.assertNotIn("production-safe", manual)

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
