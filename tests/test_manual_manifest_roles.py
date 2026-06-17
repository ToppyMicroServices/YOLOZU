import unittest
from pathlib import Path


class TestManualManifestRoles(unittest.TestCase):
    def setUp(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.tool_registry = repo_root / "manual" / "chapters" / "12_tool_registry.tex"
        self.manifest_docs = repo_root / "manual" / "chapters" / "18_manifest_driven_docs.tex"
        self.assertTrue(self.tool_registry.exists())
        self.assertTrue(self.manifest_docs.exists())

    def _normalized_text(self, path):
        return " ".join(path.read_text(encoding="utf-8").split())

    def test_tool_registry_chapter_is_discovery_focused(self):
        text = self._normalized_text(self.tool_registry)
        self.assertIn("Boundary With the Maintenance Chapter", text)
        self.assertIn("read and use", text)
        self.assertIn("Chapter~18 owns the \\emph{maintenance workflow}", text)
        self.assertIn("When You Need to Change the Registry", text)

    def test_manifest_docs_chapter_is_maintenance_focused(self):
        text = self._normalized_text(self.manifest_docs)
        self.assertIn("Relationship to the Tool Registry Chapter", text)
        self.assertIn("maintainer checklist for changing", text)
        self.assertIn("Chapter~12 covers day-to-day discovery", text)


if __name__ == "__main__":
    unittest.main()
