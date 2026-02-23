import json
import unittest
from pathlib import Path

from yolozu.integrations.tool_reference import build_tool_surface_reference, render_tool_surface_markdown


class TestGeneratedIntegrationToolReference(unittest.TestCase):
    def test_generated_reference_files_are_up_to_date(self):
        repo_root = Path(__file__).resolve().parents[1]
        json_path = repo_root / "docs" / "generated" / "mcp_actions_tool_reference.json"
        md_path = repo_root / "docs" / "generated" / "mcp_actions_tool_reference.md"

        self.assertTrue(json_path.exists(), f"missing generated JSON reference: {json_path}")
        self.assertTrue(md_path.exists(), f"missing generated Markdown reference: {md_path}")

        reference = build_tool_surface_reference()
        expected_json = json.dumps(reference, indent=2, ensure_ascii=False) + "\n"
        expected_md = render_tool_surface_markdown(reference)

        self.assertEqual(
            json_path.read_text(encoding="utf-8"),
            expected_json,
            "generated JSON drifted; run tools/generate_integration_tool_reference.py",
        )
        self.assertEqual(
            md_path.read_text(encoding="utf-8"),
            expected_md,
            "generated Markdown drifted; run tools/generate_integration_tool_reference.py",
        )


if __name__ == "__main__":
    unittest.main()
