import json
import unittest
from pathlib import Path


class TestManifestToolCoverage(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.manifest_path = self.repo_root / "tools" / "manifest.json"
        self.policy_path = self.repo_root / "docs" / "manifest_unmanifested_tools_policy.json"

    def test_unmanifested_tool_files_are_triaged(self):
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        policy = json.loads(self.policy_path.read_text(encoding="utf-8"))

        manifest_entrypoints = {
            str(tool.get("entrypoint"))
            for tool in manifest.get("tools", [])
            if isinstance(tool, dict) and tool.get("entrypoint")
        }
        direct_tool_files = {
            str(path.relative_to(self.repo_root))
            for path in (self.repo_root / "tools").iterdir()
            if path.is_file() and path.suffix in {".py", ".sh"}
        }
        policy_entries = {
            str(item.get("path")): item
            for item in policy.get("tools", [])
            if isinstance(item, dict) and item.get("path")
        }

        unmanifested = direct_tool_files - manifest_entrypoints
        self.assertEqual(
            unmanifested,
            set(policy_entries),
            "Every unmanifested direct tools/ entrypoint must be listed in docs/manifest_unmanifested_tools_policy.json.",
        )

        allowed_dispositions = {
            "analysis_helper",
            "backend_operator",
            "compatibility_wrapper",
            "dataset_recipe",
            "example_helper",
            "fixture_generator",
            "internal_helper",
            "maintenance_audit",
            "maintenance_helper",
            "manual_audit",
            "manual_build_helper",
            "migration_helper",
            "optional_bridge_helper",
            "release_operator",
            "repository_operator",
            "research_helper",
            "scenario_helper",
            "smoke_helper",
            "test_helper",
        }
        for tool_path, item in policy_entries.items():
            self.assertTrue((self.repo_root / tool_path).is_file(), f"policy path does not exist: {tool_path}")
            self.assertIn(item.get("disposition"), allowed_dispositions, f"invalid disposition for {tool_path}")
            self.assertTrue(item.get("rationale"), f"missing rationale for {tool_path}")


if __name__ == "__main__":
    unittest.main()
