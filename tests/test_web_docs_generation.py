from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TestWebDocsGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.output = self.repo_root / "docs" / "generated" / "web_docs"

    def _read(self, relative: str) -> str:
        return (self.output / relative).read_text(encoding="utf-8")

    def _run_with_content(
        self,
        content: dict[str, object],
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(
            prefix=".web-docs-test-",
            dir=self.repo_root,
        ) as temporary:
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

        content = json.loads(json.dumps(source))
        content["python_api"] = "not-an-object"
        proc = self._run_with_content(content)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(
            "web docs content requires python_api object",
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
        self.assertIn(
            "Installed commands and repository-only entrypoints",
            commands,
        )
        self.assertIn("<code>yolozu-mcp</code>", commands)
        self.assertIn("<code>tools/</code>", commands)

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
            "python3 -m venv .venv",
            "python -m pip install &quot;yolozu[coco]&quot;",
            "yolozu doctor --proof",
            "yolozu validate dataset",
            "yolozu validate predictions",
            "yolozu eval-coco",
            "from yolozu.api import evaluate_coco",
        ):
            with self.subTest(command=command):
                self.assertIn(command, tutorial)
        self.assertNotIn("yolozu guide", tutorial)
        self.assertNotIn("data/smoke", tutorial)
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
                "api",
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

    def test_search_normalization_and_links_are_deterministic_and_safe(self) -> None:
        script = self._read("assets/docs.js")
        self.assertIn(".toLowerCase()", script)
        self.assertNotIn(".toLocaleLowerCase()", script)
        self.assertIn('url.protocol !== "http:" && url.protocol !== "https:"', script)
        self.assertIn("safeSearchHref(item.href)", script)

    def test_provenance_covers_all_generated_files_and_sources(self) -> None:
        provenance = json.loads(self._read("provenance.json"))
        generated = set(provenance["generated_files"])
        actual = {
            path.relative_to(self.output).as_posix()
            for path in self.output.rglob("*")
            if path.is_file()
        }
        self.assertEqual(generated, actual)
        sources = set(provenance["source_hashes"])
        content = json.loads(
            (self.repo_root / "docs" / "web_docs_content.json").read_text(
                encoding="utf-8"
            )
        )
        manifest = json.loads(
            (self.repo_root / "tools" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        expected_sources = {
            "tools/generate_web_docs.py",
            "tools/manifest.json",
            "docs/web_docs_content.json",
            "docs/web_docs_assets/styles.css",
            "docs/web_docs_assets/docs.js",
            "docs/cpu_only_dod.md",
            "docs/evaluation_protocol_template.md",
            "docs/schema_governance.md",
            *(
                path.relative_to(self.repo_root).as_posix()
                for path in (self.repo_root / "docs" / "schemas").glob("*.json")
            ),
        }
        expected_sources.update(lane["source"] for lane in content["lanes"])
        expected_sources.add(content["python_api"]["source"])
        for group_name in ("examples", "glossary", "failures"):
            for entry in content[group_name]:
                expected_sources.add(entry["source"])
                for optional_key in ("stable_artifact", "image_source"):
                    if entry.get(optional_key):
                        expected_sources.add(entry[optional_key])
        for tool in manifest["tools"]:
            expected_sources.add(tool["entrypoint"])
            expected_sources.update(
                path
                for path in (tool.get("docs") or [])
                if not path.startswith("docs/generated/web_docs/")
            )
        self.assertEqual(sources, expected_sources)

    def test_content_rejects_external_or_traversing_sources(self) -> None:
        source = json.loads(
            (self.repo_root / "docs" / "web_docs_content.json").read_text(
                encoding="utf-8"
            )
        )
        unsafe_values = (
            str(Path(tempfile.gettempdir()) / "outside.png"),
            "../outside.png",
            "docs/../README.md",
            r"docs\assets\image.png",
        )
        for unsafe in unsafe_values:
            with self.subTest(source=unsafe):
                content = json.loads(json.dumps(source))
                content["examples"][0]["image_source"] = unsafe
                proc = self._run_with_content(content)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(
                    "examples[instance-seg-visible-demo].image_source",
                    proc.stdout + proc.stderr,
                )

    def test_content_rejects_unsafe_image_output_and_urls(self) -> None:
        source = json.loads(
            (self.repo_root / "docs" / "web_docs_content.json").read_text(
                encoding="utf-8"
            )
        )
        for unsafe in (
            "../outside.png",
            "nested/outside.png",
            r"..\outside.png",
            "outside.html",
        ):
            with self.subTest(image_output=unsafe):
                content = json.loads(json.dumps(source))
                content["examples"][0]["image_output"] = unsafe
                proc = self._run_with_content(content)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("safe image filename", proc.stdout + proc.stderr)

        for unsafe in (
            "javascript:alert(1)",
            "data:text/html,unsafe",
            "file:///etc/passwd",
            "//attacker.example/path",
            "https://user:pass@example.com/path",
            "https://example.com/a/../b",
            "start.html%0ajavascript:alert(1)",
        ):
            with self.subTest(url=unsafe):
                content = json.loads(json.dumps(source))
                content["lanes"][0]["start_page"] = unsafe
                proc = self._run_with_content(content)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(
                    "lanes[stable].start_page",
                    proc.stdout + proc.stderr,
                )

    def test_symlink_source_escape_is_rejected(self) -> None:
        source = json.loads(
            (self.repo_root / "docs" / "web_docs_content.json").read_text(
                encoding="utf-8"
            )
        )
        with tempfile.TemporaryDirectory(
            prefix=".web-docs-test-",
            dir=self.repo_root,
        ) as temporary:
            root = Path(temporary)
            external = Path(tempfile.gettempdir()) / "yolozu-web-source.txt"
            external.write_text("outside\n", encoding="utf-8")
            link = root / "source.png"
            try:
                link.symlink_to(external)
                source["examples"][0]["image_source"] = link.relative_to(
                    self.repo_root
                ).as_posix()
                content_path = root / "content.json"
                content_path.write_text(json.dumps(source), encoding="utf-8")
                proc = subprocess.run(
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
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("inside the repository", proc.stdout + proc.stderr)
            finally:
                external.unlink(missing_ok=True)

    def test_unowned_output_is_preserved_and_owned_output_is_replaceable(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".web-docs-test-",
            dir=self.repo_root,
        ) as temporary:
            root = Path(temporary)
            unowned = root / "unowned" / "bundle"
            unowned.mkdir(parents=True)
            sentinel = unowned / "sentinel.txt"
            sentinel.write_text("preserve me\n", encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    "tools/generate_web_docs.py",
                    "--output",
                    str(unowned),
                ],
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve me\n")

            owned = root / "owned" / "bundle"
            for _ in range(2):
                proc = subprocess.run(
                    [
                        sys.executable,
                        "tools/generate_web_docs.py",
                        "--output",
                        str(owned),
                    ],
                    cwd=self.repo_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertTrue((owned / "provenance.json").is_file())

    def test_symlink_output_is_rejected_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".web-docs-test-",
            dir=self.repo_root,
        ) as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            sentinel = target / "sentinel.txt"
            sentinel.write_text("preserve me\n", encoding="utf-8")
            output_link = root / "nested" / "output"
            output_link.parent.mkdir()
            output_link.symlink_to(target, target_is_directory=True)
            proc = subprocess.run(
                [
                    sys.executable,
                    "tools/generate_web_docs.py",
                    "--output",
                    str(output_link),
                ],
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("symlink", proc.stdout + proc.stderr)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve me\n")

    def test_atomic_replace_rolls_back_when_stage_rename_fails(self) -> None:
        module_path = self.repo_root / "tools" / "generate_web_docs.py"
        spec = importlib.util.spec_from_file_location(
            "yolozu_test_generate_web_docs",
            module_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        bundle, _ = module._build_bundle(
            manifest_path=self.repo_root / "tools" / "manifest.json",
            schemas_dir=self.repo_root / "docs" / "schemas",
            content_path=self.repo_root / "docs" / "web_docs_content.json",
        )
        with tempfile.TemporaryDirectory(
            prefix=".web-docs-test-",
            dir=self.repo_root,
        ) as temporary:
            output = Path(temporary) / "nested" / "bundle"
            module._write_bundle(output, bundle)
            before = {
                path.relative_to(output).as_posix(): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }
            replacement = dict(bundle)
            replacement["index.html"] = b"replacement\n"
            real_rename = Path.rename

            def fail_stage_rename(path: Path, target: Path) -> Path:
                if ".yolozu-web-stage-" in path.name:
                    raise OSError("injected stage rename failure")
                return real_rename(path, target)

            with mock.patch.object(Path, "rename", fail_stage_rename):
                with self.assertRaisesRegex(
                    OSError,
                    "injected stage rename failure",
                ):
                    module._write_bundle(output, replacement)
            after = {
                path.relative_to(output).as_posix(): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }
            self.assertEqual(after, before)

    def test_cli_source_inputs_must_stay_in_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            external_content = Path(temporary) / "content.json"
            external_content.write_text("{}\n", encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    "tools/generate_web_docs.py",
                    "--content",
                    str(external_content),
                    "--output",
                    str(Path(temporary) / "nested" / "output"),
                ],
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(
            "--content must resolve inside the repository",
            proc.stdout + proc.stderr,
        )

    def test_generated_text_has_no_trailing_whitespace(self) -> None:
        for path in self.output.rglob("*"):
            if path.suffix not in {".css", ".html", ".js", ".json"}:
                continue
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                with self.subTest(path=path.name, line=line_number):
                    self.assertEqual(line, line.rstrip())

    def test_tutorial_commands_are_ordered_self_contained_and_explicit(self) -> None:
        content = json.loads(
            (self.repo_root / "docs" / "web_docs_content.json").read_text(
                encoding="utf-8"
            )
        )
        commands = [
            command
            for step in content["tutorial"]["thirty_minute"]
            for command in step["commands"]
        ]
        joined = "\n".join(commands)

        proof_index = next(
            index for index, command in enumerate(commands) if "doctor --proof" in command
        )
        validation_indices = [
            index for index, command in enumerate(commands) if "yolozu validate " in command
        ]
        eval_indices = [
            index for index, command in enumerate(commands) if "yolozu eval-coco" in command
        ]
        self.assertEqual(len(validation_indices), 2)
        self.assertEqual(len(eval_indices), 1)
        self.assertLess(proof_index, min(validation_indices))
        self.assertLess(max(validation_indices), min(eval_indices))

        real_eval = commands[eval_indices[0]]
        self.assertNotIn("--dry-run", real_eval)
        fallback_eval = content["tutorial"]["dry_run_fallback"]["command"]
        self.assertIn("--dry-run", fallback_eval)
        for flag in (" -d ", " -p ", " -s ", " -o "):
            with self.subTest(flag=flag):
                self.assertIn(flag, real_eval)
                self.assertIn(flag, fallback_eval)
        for path in (
            "reports/quickstart/proof/toy_dataset",
            "reports/quickstart/proof/known_predictions.json",
        ):
            with self.subTest(path=path):
                self.assertIn(path, joined)

        self.assertIn('python -m pip install "yolozu[coco]"', commands)
        self.assertTrue(
            all("--strict" in commands[index] for index in validation_indices)
        )
        self.assertNotIn("yolozu guide", joined)
        self.assertNotIn("data/smoke", joined)
        self.assertNotIn("/path/to/", joined)

        python_example = content["python_api"]["example"]
        compile(python_example, "<web-doc-python-api-example>", "exec")
        self.assertIn("from yolozu.api import evaluate_coco", python_example)
        self.assertIn("Path.cwd().resolve()", python_example)
        self.assertNotIn("subprocess", python_example)

    def test_generated_internal_links_and_assets_resolve(self) -> None:
        for html_path in sorted(self.output.glob("*.html")):
            text = html_path.read_text(encoding="utf-8")
            targets = re.findall(r'(?:href|src)="([^"]+)"', text)
            for target in targets:
                with self.subTest(page=html_path.name, target=target):
                    if (
                        target.startswith(("http://", "https://", "/", "mailto:"))
                        or target == ""
                    ):
                        continue
                    path_part, _, fragment = target.partition("#")
                    relative = path_part.split("?", 1)[0]
                    destination = self.output / relative if relative else html_path
                    self.assertTrue(
                        destination.is_file(),
                        f"{html_path.name}: unresolved local target {target}",
                    )
                    if fragment:
                        destination_text = destination.read_text(encoding="utf-8")
                        self.assertIn(
                            f'id="{fragment}"',
                            destination_text,
                            f"{html_path.name}: unresolved fragment {target}",
                        )


if __name__ == "__main__":
    unittest.main()
