from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestWebDocsGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.output = self.repo_root / "docs" / "generated" / "web_docs"

    def _read(self, relative: str) -> str:
        return (self.output / relative).read_text(encoding="utf-8")

    def _run_with_content(self, content: dict[str, object]) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            content_path = root / "content.json"
            content_path.write_text(
                json.dumps(content),
                encoding="utf-8",
            )
            return subprocess.run(
                [
                    sys.executable,
                    "tools/generate_web_docs.py",
                    "--content",
                    str(content_path),
                    "--output",
                    str(root / "output"),
                ],
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

    def test_content_object_lists_fail_with_explanatory_errors(self) -> None:
        source = json.loads(
            (self.repo_root / "docs" / "web_docs_content.json").read_text(
                encoding="utf-8"
            )
        )
        for group in (
            "lanes",
            "examples",
            "glossary",
            "failures",
        ):
            with self.subTest(group=group):
                content = json.loads(json.dumps(source))
                content[group][0] = "not-an-object"
                proc = self._run_with_content(content)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(
                    f"{group} entries must be objects",
                    proc.stdout + proc.stderr,
                )

        content = json.loads(json.dumps(source))
        content["tutorial"]["thirty_minute"][0] = "not-an-object"
        proc = self._run_with_content(content)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(
            "tutorial.thirty_minute entries must be objects",
            proc.stdout + proc.stderr,
        )

    def test_generated_bundle_is_current(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "tools/generate_web_docs.py",
                "--check",
                "--json",
            ],
            cwd=self.repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        result = json.loads(proc.stdout)
        self.assertTrue(result["ok"])
        self.assertEqual(result["missing"], [])
        self.assertEqual(result["stale"], [])
        self.assertEqual(result["extra"], [])

    def test_command_reference_covers_every_manifest_entry(self) -> None:
        manifest = json.loads(
            (self.repo_root / "tools" / "manifest.json").read_text(encoding="utf-8")
        )
        commands = self._read("commands.html")
        tool_ids = {tool["id"] for tool in manifest["tools"]}

        self.assertEqual(commands.count('data-tool-id="'), len(tool_ids))
        for tool_id in tool_ids:
            with self.subTest(tool_id=tool_id):
                self.assertIn(f'data-tool-id="{tool_id}"', commands)
        self.assertIn(
            "https://github.com/ToppyMicroServices/YOLOZU/blob/main/tools/manifest.json",
            commands,
        )

    def test_schema_browser_covers_every_checked_in_schema(self) -> None:
        schema_paths = sorted(
            path.relative_to(self.repo_root).as_posix()
            for path in (self.repo_root / "docs" / "schemas").glob("*.json")
        )
        schemas = self._read("schemas.html")

        self.assertEqual(schemas.count('data-schema-path="'), len(schema_paths))
        for schema_path in schema_paths:
            with self.subTest(schema_path=schema_path):
                self.assertIn(f'data-schema-path="{schema_path}"', schemas)
                self.assertIn(
                    "https://github.com/ToppyMicroServices/YOLOZU/blob/main/"
                    + schema_path,
                    schemas,
                )

    def test_surface_keeps_lanes_visually_and_semantically_separate(self) -> None:
        index = self._read("index.html")
        for lane in ("stable", "bridge", "benchmark", "research"):
            with self.subTest(lane=lane):
                self.assertIn(f"lane-card lane-{lane}", index)
                self.assertIn(f"badge-{lane}", index)
        self.assertIn("Return to the stable artifact", index)

        examples = self._read("examples.html")
        self.assertIn("Stable baseline artifact", examples)
        self.assertIn('href="start.html"', examples)

    def test_tutorial_and_failure_guide_cover_the_acceptance_path(self) -> None:
        tutorial = self._read("start.html")
        for command in (
            "python3 -m pip install -U yolozu",
            "yolozu doctor --proof",
            "yolozu demo instance-seg",
            "yolozu validate dataset",
            "yolozu validate predictions",
            "yolozu eval-coco",
        ):
            with self.subTest(command=command):
                self.assertIn(command, tutorial)
        self.assertIn('data-docs-event="YOLOZU docs completion"', tutorial)

        failures = self._read("troubleshooting.html")
        self.assertEqual(failures.count('class="badge badge-danger"'), 8)
        normalized_failures = " ".join(failures.split())
        self.assertIn(
            "A skipped or unavailable lane is not a successful result",
            normalized_failures,
        )

    def test_global_search_index_includes_commands_schemas_and_guides(self) -> None:
        search_index = json.loads(self._read("search-index.json"))
        kinds = {entry["kind"] for entry in search_index}
        self.assertTrue(
            {
                "command",
                "schema",
                "tutorial",
                "lane",
                "example",
                "glossary",
                "troubleshooting",
            }.issubset(kinds)
        )
        self.assertTrue(all(entry["href"] and entry["search_text"] for entry in search_index))

    def test_measurement_does_not_send_search_terms(self) -> None:
        script = self._read("assets/docs.js")
        self.assertIn('window.plausible(target.getAttribute("data-docs-event")', script)
        self.assertIn('target: target.getAttribute("data-event-target")', script)
        self.assertIn('page: document.body.getAttribute("data-page")', script)
        plausible_block = script[script.index("window.plausible") :]
        self.assertNotIn("searchInput.value", plausible_block)
        self.assertNotIn("query:", plausible_block)

    def test_provenance_covers_all_generated_files_and_sources(self) -> None:
        provenance = json.loads(self._read("provenance.json"))
        generated = set(provenance["generated_files"])
        actual = {
            path.relative_to(self.output).as_posix()
            for path in self.output.rglob("*")
            if path.is_file()
        }
        self.assertEqual(generated, actual)
        sources = provenance["source_hashes"]
        self.assertIn("tools/manifest.json", sources)
        self.assertIn("docs/web_docs_content.json", sources)
        self.assertIn("docs/schemas/predictions.schema.json", sources)

    def test_generated_internal_links_and_assets_resolve(self) -> None:
        for html_path in sorted(self.output.glob("*.html")):
            text = html_path.read_text(encoding="utf-8")
            targets = re.findall(r'(?:href|src)="([^"]+)"', text)
            for target in targets:
                with self.subTest(page=html_path.name, target=target):
                    if (
                        target.startswith(("http://", "https://", "/", "#", "mailto:"))
                        or target == ""
                    ):
                        continue
                    relative = target.split("#", 1)[0].split("?", 1)[0]
                    self.assertTrue(
                        (self.output / relative).is_file(),
                        f"{html_path.name}: unresolved local target {target}",
                    )


if __name__ == "__main__":
    unittest.main()
