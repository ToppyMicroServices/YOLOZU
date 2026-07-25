import json
import unittest
from pathlib import Path


class TestPackagedToolsManifest(unittest.TestCase):
    def test_every_tool_declares_maturity(self):
        repo_root = Path(__file__).resolve().parents[1]
        for path in (
            repo_root / "tools" / "manifest.json",
            repo_root / "yolozu" / "data" / "manifest" / "tools_manifest.json",
        ):
            obj = json.loads(path.read_text(encoding="utf-8"))
            for tool in obj.get("tools") or []:
                if not isinstance(tool, dict):
                    continue
                self.assertIn("maturity", tool, f"missing maturity in {path}: {tool.get('id')}")
                self.assertIn(
                    tool.get("maturity"),
                    {"stable", "experimental", "research"},
                    f"invalid maturity in {path}: {tool.get('id')}",
                )

    def test_packaged_tools_manifest_matches_repo_manifest(self):
        repo_root = Path(__file__).resolve().parents[1]
        repo_manifest = repo_root / "tools" / "manifest.json"
        packaged_manifest = repo_root / "yolozu" / "data" / "manifest" / "tools_manifest.json"

        self.assertTrue(repo_manifest.is_file(), f"missing repo tool manifest: {repo_manifest}")
        self.assertTrue(packaged_manifest.is_file(), f"missing packaged tool manifest: {packaged_manifest}")

        repo_obj = json.loads(repo_manifest.read_text(encoding="utf-8"))
        packaged_obj = json.loads(packaged_manifest.read_text(encoding="utf-8"))

        self.assertEqual(
            repo_obj,
            packaged_obj,
            "packaged manifest is out of sync. Update yolozu/data/manifest/tools_manifest.json to match tools/manifest.json.",
        )

    def test_packaged_manifest_matches_sync_tools_manifest_canonical_output(self):
        repo_root = Path(__file__).resolve().parents[1]
        repo_manifest = repo_root / "tools" / "manifest.json"
        packaged_manifest = repo_root / "yolozu" / "data" / "manifest" / "tools_manifest.json"

        repo_obj = json.loads(repo_manifest.read_text(encoding="utf-8"))
        expected = json.dumps(repo_obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        actual = packaged_manifest.read_text(encoding="utf-8")

        self.assertEqual(
            actual,
            expected,
            "packaged manifest must match the canonical sync_tools_manifest.py output.",
        )
        self.assertEqual(
            repo_manifest.read_text(encoding="utf-8"),
            expected,
            "source manifest must also use canonical JSON formatting.",
        )

    def test_tools_are_sorted_by_id_for_deterministic_diff(self):
        repo_root = Path(__file__).resolve().parents[1]
        for path in (
            repo_root / "tools" / "manifest.json",
            repo_root / "yolozu" / "data" / "manifest" / "tools_manifest.json",
        ):
            obj = json.loads(path.read_text(encoding="utf-8"))
            tools = obj.get("tools") or []
            ids = [str(tool.get("id")) for tool in tools if isinstance(tool, dict) and tool.get("id")]
            self.assertEqual(ids, sorted(ids), f"tools array must be sorted by id in {path}")


if __name__ == "__main__":
    unittest.main()
