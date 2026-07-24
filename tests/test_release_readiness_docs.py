import unittest
from pathlib import Path


class TestReleaseReadinessDocs(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_publish_trigger_and_release_doc_are_aligned(self):
        publish = (self.repo_root / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
        release_md = (self.repo_root / "RELEASE.md").read_text(encoding="utf-8")

        self.assertIn("release:", publish)
        self.assertIn("published", publish)
        self.assertIn("Validate synchronized release metadata", publish)
        self.assertIn("expected_version:", publish)
        self.assertIn("CITATION.cff", publish)
        self.assertIn("validate_release_metadata", publish)
        self.assertIn("Validate built wheel version matches package version", publish)
        self.assertIn("Verify published release on PyPI", publish)
        self.assertIn("GitHub Release", release_md)
        self.assertIn("Tag push alone does not publish", release_md)
        self.assertIn("wheel and sdist", release_md)
        self.assertIn(".github/workflows/build_and_test.yml", release_md)
        self.assertIn(".github/release_notes_template.md", release_md)

    def test_release_notes_template_exists(self):
        template = (self.repo_root / ".github" / "release_notes_template.md")
        self.assertTrue(template.is_file(), "missing .github/release_notes_template.md")
        text = template.read_text(encoding="utf-8")
        self.assertIn("Quickstart (3 lines)", text)
        self.assertIn("## Added", text)
        self.assertIn("## Changed", text)
        self.assertIn("## Fixed", text)
        self.assertIn("## Breaking", text)

    def test_changelog_has_one_dot_zero_cut(self):
        changelog = (self.repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn("## [Unreleased]", changelog)
        self.assertIn("## [1.0.0] - ", changelog)
        self.assertIn("### Breaking", changelog)
        self.assertIn("### Deprecated", changelog)

    def test_package_metadata_matches_lane_based_readiness(self):
        pyproject = (self.repo_root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("Development Status :: 4 - Beta", pyproject)
        self.assertIn("stable predictions interface contract", pyproject)
        self.assertNotIn("Development Status :: 3 - Alpha", pyproject)

    def test_technical_credibility_docs_are_linked(self):
        docs_index = (self.repo_root / "docs" / "README.md").read_text(encoding="utf-8")
        readiness = (self.repo_root / "docs" / "production_readiness.md").read_text(encoding="utf-8")
        schema = (self.repo_root / "docs" / "schema_governance.md").read_text(encoding="utf-8")
        template = self.repo_root / "docs" / "evaluation_protocol_template.md"

        self.assertTrue(template.is_file(), "missing evaluation protocol template")
        self.assertIn("evaluation_protocol_template.md", docs_index)
        self.assertIn("web_docs_plan.md", docs_index)
        self.assertIn("Version Compatibility Matrix", readiness)
        self.assertIn("Production Readiness Matrix", readiness)
        self.assertIn("Schema Browser Coverage", schema)

    def test_web_docs_plan_covers_expected_surfaces(self):
        plan_path = self.repo_root / "docs" / "web_docs_plan.md"
        self.assertTrue(plan_path.is_file(), "missing web docs plan")
        plan = plan_path.read_text(encoding="utf-8")
        expected = [
            "30-minute path",
            "2-hour path",
            "Command reference",
            "Schema browser",
            "Examples gallery",
            "Glossary",
            "What can go wrong",
            "Report reading guides",
        ]
        for item in expected:
            with self.subTest(item=item):
                self.assertIn(item, plan)

    def test_ci_contains_release_integrity_gates(self):
        ci = (self.repo_root / ".github" / "workflows" / "build_and_test.yml").read_text(encoding="utf-8")
        self.assertIn("Sdist contents gate", ci)
        self.assertIn("tools/check_schema_compatibility.py", ci)
        self.assertIn("tools/check_golden_compatibility.py", ci)

    def test_manifest_includes_release_and_manifest_payload(self):
        manifest_in = (self.repo_root / "MANIFEST.in").read_text(encoding="utf-8")
        self.assertIn("include RELEASE.md", manifest_in)
        self.assertIn("include tools/manifest.json", manifest_in)

    def test_manual_doi_workflow_exists_and_links_to_software_concept_doi(self):
        workflow = (self.repo_root / ".github" / "workflows" / "manual_doi.yml").read_text(encoding="utf-8")
        self.assertIn("types: [published]", workflow)
        self.assertIn("actions/newversion", workflow)
        self.assertIn("related_identifiers", workflow)
        self.assertIn("isSupplementTo", workflow)
        self.assertIn("YOLOZU_SOFTWARE_CONCEPT_DOI", workflow)

    def test_security_doc_contains_reporting_and_support_scope(self):
        security = (self.repo_root / "SECURITY.md").read_text(encoding="utf-8")
        self.assertIn("Reporting a vulnerability", security)
        self.assertIn("Supported runtime scope", security)
        self.assertIn("Dependency and third-party policy", security)

    def test_stable_entry_docs_do_not_use_placeholder_language(self):
        checked_paths = [
            "README.md",
            "Readme_jp.md",
            "Readme_zh.md",
            "docs/README.md",
            "docs/install.md",
            "deploy/docker/README.md",
            "deploy/runpod/README.md",
            "deploy/pyinstaller/README.md",
        ]
        forbidden = (
            "scafold",
            "scaffold",
            "stub",
            "placeholder",
            "planned",
            "not implemented",
            "todo",
            "fixme",
            "未作成",
            "未実装",
            "仮実装",
            "ダミー",
        )
        hits = []
        for rel in checked_paths:
            text = (self.repo_root / rel).read_text(encoding="utf-8").lower()
            for token in forbidden:
                if token in text:
                    hits.append(f"{rel}: {token}")
        self.assertEqual(hits, [], "stable entry docs should avoid placeholder/scaffold language")


if __name__ == "__main__":
    unittest.main()
