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
        self.assertIn("Validate release tag matches package version", publish)
        self.assertIn("GitHub Release", release_md)
        self.assertIn("Tag push alone does not publish", release_md)

    def test_changelog_has_one_dot_zero_cut(self):
        changelog = (self.repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn("## [Unreleased]", changelog)
        self.assertIn("## [1.0.0] - ", changelog)
        self.assertIn("### Breaking", changelog)
        self.assertIn("### Deprecated", changelog)

    def test_package_metadata_is_stable_classifier(self):
        pyproject = (self.repo_root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("Development Status :: 5 - Production/Stable", pyproject)
        self.assertNotIn("Development Status :: 3 - Alpha", pyproject)

    def test_ci_contains_release_integrity_gates(self):
        ci = (self.repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
