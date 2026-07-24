import json
import tempfile
from pathlib import Path
from unittest import TestCase, main, mock

import tools.release_metadata as release_metadata
from tools.release_metadata import (
    prepare_release_metadata,
    validate_release_metadata,
    write_release_metadata_atomic,
)


class TestReleaseMetadata(TestCase):
    def _make_repo(self, root: Path) -> None:
        (root / "yolozu" / "data" / "manifest").mkdir(parents=True)
        (root / "tools").mkdir()
        (root / "docs").mkdir()
        (root / "docs" / "evidence.md").write_text(
            "# Historical release evidence\n",
            encoding="utf-8",
        )
        (root / "yolozu" / "__init__.py").write_text(
            '__version__ = "1.2.3"\n',
            encoding="utf-8",
        )
        (root / "CHANGELOG.md").write_text(
            "# Changelog\n\n"
            "## [Unreleased]\n\n"
            "## [1.2.3] - 2026-07-01\n\n"
            "### Changed\n"
            "- Existing release.\n",
            encoding="utf-8",
        )
        (root / "CITATION.cff").write_text(
            "cff-version: 1.2.0\n"
            'version: "1.2.3"\n'
            'date-released: "2026-07-01"\n',
            encoding="utf-8",
        )
        manifest = {
            "manifest_version": 1,
            "tools": [
                {
                    "id": "current_example",
                    "entrypoint": "tools/current.py",
                    "runner": "python3",
                    "summary": "Current release example.",
                    "examples": [
                        {
                            "command": (
                                "python -m pip install yolozu==1.2.3 "
                                "&& mkdir /tmp/yolozu_1_2_3"
                            ),
                            "release_version_policy": "current",
                        }
                    ],
                },
                {
                    "id": "historical_example",
                    "entrypoint": "tools/history.py",
                    "runner": "python3",
                    "summary": "Historical release example.",
                    "examples": [
                        {
                            "command": "python -m pip install yolozu==1.0.0",
                            "release_version_policy": "historical",
                            "release_version_evidence": "docs/evidence.md",
                        }
                    ],
                },
            ],
        }
        manifest_text = (
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
        (root / "tools" / "manifest.json").write_text(
            manifest_text,
            encoding="utf-8",
        )
        (root / "yolozu" / "data" / "manifest" / "tools_manifest.json").write_text(
            manifest_text,
            encoding="utf-8",
        )

    def test_synchronized_metadata_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_repo(root)
            result = validate_release_metadata(
                root,
                expected_version="1.2.3",
                expected_tag="v1.2.3",
            )
            self.assertTrue(result["ok"], result["errors"])
            self.assertEqual(result["changelog_date"], "2026-07-01")
            self.assertEqual(result["citation_date_released"], "2026-07-01")
            self.assertTrue(result["source_packaged_manifests_identical"])

    def test_stale_citation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_repo(root)
            citation = root / "CITATION.cff"
            citation.write_text(
                citation.read_text(encoding="utf-8").replace("1.2.3", "1.2.2"),
                encoding="utf-8",
            )
            result = validate_release_metadata(root, expected_version="1.2.3")
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("CITATION.cff version mismatch" in error for error in result["errors"])
            )

    def test_stale_citation_date_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_repo(root)
            citation = root / "CITATION.cff"
            citation.write_text(
                citation.read_text(encoding="utf-8").replace(
                    "2026-07-01",
                    "2026-06-30",
                ),
                encoding="utf-8",
            )
            result = validate_release_metadata(root, expected_version="1.2.3")
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("release date mismatch" in error for error in result["errors"])
            )

    def test_missing_changelog_heading_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_repo(root)
            changelog = root / "CHANGELOG.md"
            changelog.write_text(
                changelog.read_text(encoding="utf-8").replace(
                    "## [1.2.3] - 2026-07-01",
                    "## [1.2.2] - 2026-07-01",
                ),
                encoding="utf-8",
            )
            result = validate_release_metadata(root, expected_version="1.2.3")
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("missing release heading" in error for error in result["errors"])
            )

    def test_mismatched_tag_and_version_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_repo(root)
            result = validate_release_metadata(
                root,
                expected_version="1.2.3",
                expected_tag="v1.2.4",
            )
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("release tag/version mismatch" in error for error in result["errors"])
            )

    def test_prepare_and_write_synchronizes_current_examples_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_repo(root)
            plan = prepare_release_metadata(
                root,
                current_version="1.2.3",
                next_version="1.3.0",
                release_date="2026-07-24",
                changelog_section=(
                    "## [1.3.0] - 2026-07-24\n\n"
                    "### Changed\n"
                    "- Synchronized release metadata.\n"
                ),
            )
            self.assertEqual(
                set(plan.changed_paths),
                {
                    "yolozu/__init__.py",
                    "CHANGELOG.md",
                    "CITATION.cff",
                    "tools/manifest.json",
                    "yolozu/data/manifest/tools_manifest.json",
                },
            )
            self.assertTrue(plan.validation_after["ok"])
            self.assertIn("yolozu==1.3.0", plan.files["tools/manifest.json"])
            self.assertIn("/tmp/yolozu_1_3_0", plan.files["tools/manifest.json"])
            self.assertIn("yolozu==1.0.0", plan.files["tools/manifest.json"])

            write_release_metadata_atomic(root, plan)
            result = validate_release_metadata(
                root,
                expected_version="1.3.0",
                expected_tag="v1.3.0",
            )
            self.assertTrue(result["ok"], result["errors"])
            self.assertEqual(
                (root / "tools" / "manifest.json").read_bytes(),
                (
                    root
                    / "yolozu"
                    / "data"
                    / "manifest"
                    / "tools_manifest.json"
                ).read_bytes(),
            )

    def test_unclassified_exact_version_manifest_example_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_repo(root)
            manifest_path = root / "tools" / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["tools"][0]["examples"][0].pop("release_version_policy")
            text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
            manifest_path.write_text(text, encoding="utf-8")
            (
                root / "yolozu" / "data" / "manifest" / "tools_manifest.json"
            ).write_text(text, encoding="utf-8")
            result = validate_release_metadata(root, expected_version="1.2.3")
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("must declare release_version_policy" in error for error in result["errors"])
            )

    def test_write_refuses_metadata_changed_after_planning(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_repo(root)
            plan = prepare_release_metadata(
                root,
                current_version="1.2.3",
                next_version="1.2.4",
                release_date="2026-07-24",
                changelog_section=(
                    "## [1.2.4] - 2026-07-24\n\n"
                    "### Changed\n"
                    "- Synchronized release metadata.\n"
                ),
            )
            citation = root / "CITATION.cff"
            citation.write_text(
                citation.read_text(encoding="utf-8") + "# concurrent change\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "release metadata changed after planning",
            ):
                write_release_metadata_atomic(root, plan)
            self.assertIn(
                '__version__ = "1.2.3"',
                (root / "yolozu" / "__init__.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "# concurrent change",
                citation.read_text(encoding="utf-8"),
            )

    def test_write_rolls_back_after_partial_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_repo(root)
            plan = prepare_release_metadata(
                root,
                current_version="1.2.3",
                next_version="1.2.4",
                release_date="2026-07-24",
                changelog_section=(
                    "## [1.2.4] - 2026-07-24\n\n"
                    "### Changed\n"
                    "- Synchronized release metadata.\n"
                ),
            )
            real_replace = release_metadata.os.replace
            replace_calls = 0

            def fail_second_replace(source: str, destination: str) -> None:
                nonlocal replace_calls
                replace_calls += 1
                if replace_calls == 2:
                    raise OSError("injected replace failure")
                real_replace(source, destination)

            with mock.patch.object(
                release_metadata.os,
                "replace",
                side_effect=fail_second_replace,
            ):
                with self.assertRaisesRegex(OSError, "injected replace failure"):
                    write_release_metadata_atomic(root, plan)

            for relative, original in plan.original_files.items():
                self.assertEqual(
                    (root / relative).read_text(encoding="utf-8"),
                    original,
                    relative,
                )


if __name__ == "__main__":
    main()
