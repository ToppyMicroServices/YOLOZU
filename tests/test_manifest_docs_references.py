import json
import unittest
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _tool_docs_map(manifest_obj: dict) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for tool in manifest_obj.get("tools", []):
        if not isinstance(tool, dict):
            continue
        tool_id = tool.get("id")
        if not isinstance(tool_id, str) or not tool_id:
            continue
        docs = tool.get("docs")
        if docs is None:
            out[tool_id] = ()
            continue
        if not isinstance(docs, list):
            raise AssertionError(f"tool {tool_id} has non-list docs field")
        out[tool_id] = tuple(str(d) for d in docs)
    return out


class TestManifestDocsReferences(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.repo_manifest = self.repo_root / "tools" / "manifest.json"
        self.packaged_manifest = self.repo_root / "yolozu" / "data" / "manifest" / "tools_manifest.json"

    def test_docs_fields_are_in_sync_between_manifests(self):
        repo_obj = _load(self.repo_manifest)
        packaged_obj = _load(self.packaged_manifest)
        self.assertEqual(
            _tool_docs_map(repo_obj),
            _tool_docs_map(packaged_obj),
            "docs fields drifted between tools/manifest.json and packaged tools_manifest.json",
        )

    def test_docs_paths_exist_and_are_repo_relative(self):
        for manifest_path in (self.repo_manifest, self.packaged_manifest):
            obj = _load(manifest_path)
            for tool in obj.get("tools", []):
                if not isinstance(tool, dict):
                    continue
                tool_id = str(tool.get("id", "<unknown>"))
                docs = tool.get("docs") or []
                self.assertIsInstance(docs, list, f"{manifest_path}: {tool_id} docs must be list")
                self.assertEqual(len(docs), len(set(docs)), f"{manifest_path}: duplicate docs in {tool_id}")
                for doc in docs:
                    self.assertIsInstance(doc, str, f"{manifest_path}: {tool_id} docs entry must be string")
                    self.assertFalse(doc.startswith("/"), f"{manifest_path}: {tool_id} docs must be repo-relative: {doc}")
                    self.assertNotIn("..", Path(doc).parts, f"{manifest_path}: {tool_id} docs must not contain '..': {doc}")
                    target = self.repo_root / doc
                    self.assertTrue(target.exists(), f"{manifest_path}: missing docs file for {tool_id}: {doc}")

    def test_readme_docs_manual_references_resolve(self):
        obj = _load(self.repo_manifest)
        checked = 0
        for tool in obj.get("tools", []):
            if not isinstance(tool, dict):
                continue
            for doc in tool.get("docs") or []:
                if doc == "README.md" or doc.startswith("docs/") or doc.startswith("manual/"):
                    checked += 1
                    self.assertTrue((self.repo_root / doc).exists(), f"missing README/docs/manual reference: {doc}")
        self.assertGreater(checked, 0, "expected at least one README/docs/manual docs reference in manifest")


if __name__ == "__main__":
    unittest.main()
