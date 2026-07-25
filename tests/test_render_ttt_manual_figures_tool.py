import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tools.render_ttt_manual_figures as renderer


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestRenderTTTManualFiguresTool(unittest.TestCase):
    def test_help(self) -> None:
        proc = subprocess.run(
            ["python3", "tools/render_ttt_manual_figures.py", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Render beginner-friendly TTT manual figures", proc.stdout)

    def test_render_outputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            docs_dir = Path(tmp) / "docs_assets"
            manual_dir = Path(tmp) / "manual_figures"
            proc = subprocess.run(
                [
                    "python3",
                    "tools/render_ttt_manual_figures.py",
                    "--source-json",
                    "docs/assets/ttt_method_results_source.json",
                    "--docs-assets-dir",
                    str(docs_dir.relative_to(REPO_ROOT)),
                    "--manual-figures-dir",
                    str(manual_dir.relative_to(REPO_ROOT)),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            expected = {
                "ttt_method_results_summary.png",
                "ttt_compare_pipeline.png",
                "ttt_probe_example_panel.png",
            }
            self.assertEqual(expected, {p.name for p in docs_dir.glob("*.png")})
            self.assertEqual(expected, {p.name for p in manual_dir.glob("*.png")})

    def test_checked_source_is_non_promoting_synthetic_fixture(self) -> None:
        source = json.loads(
            (REPO_ROOT / "docs/assets/ttt_method_results_source.json").read_text(
                encoding="utf-8"
            )
        )
        result = renderer._validate_evidence_source(source)
        self.assertEqual(result["evidence_kind"], "synthetic_fixture")
        self.assertIs(source["promotion_eligible"], False)
        self.assertEqual(source["efficacy"], "unavailable")

    def test_synthetic_fixture_rejects_recursive_metrics_and_promotion(self) -> None:
        base = {
            "evidence_kind": "synthetic_fixture",
            "promotion_eligible": False,
            "efficacy": "unavailable",
        }
        with self.assertRaisesRegex(ValueError, "forbids measured field"):
            renderer._validate_evidence_source(
                {
                    **base,
                    "nested": [{"deeper": {"map50": 0.9}}],
                }
            )
        with self.assertRaisesRegex(ValueError, "forbids promotion claim"):
            renderer._validate_evidence_source(
                {
                    **base,
                    "nested": {"promotion": True},
                }
            )
        for key in (
            "baseline_score",
            "adapted_quality",
            "nested_accuracy",
            "metric_value",
            "ap75",
            "coco_map50",
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "forbids measured field"):
                    renderer._validate_evidence_source(
                        {
                            **base,
                            "nested": {"deeper": {key: 0.5}},
                        }
                    )

    def _measured_source(self, path: Path, sha256: str) -> dict:
        binding = {"path": str(path), "sha256": sha256}
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
        return {
            "evidence_kind": "measured",
            "promotion_eligible": False,
            "efficacy": "not_established",
            "provenance": {
                "commit": commit,
                "seed": 2026,
                "tool_versions": {"python": "3.12.0", "yolozu": "4.5.1"},
                **{name: dict(binding) for name in renderer.MEASURED_RESOURCE_NAMES},
            },
        }

    def test_measured_evidence_requires_provenance_and_hash_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires provenance"):
            renderer._validate_evidence_source(
                {
                    "evidence_kind": "measured",
                    "promotion_eligible": False,
                }
            )
        with tempfile.TemporaryDirectory() as td:
            artifact = Path(td) / "artifact.json"
            artifact.write_text("{}", encoding="utf-8")
            source = self._measured_source(artifact, "0" * 64)
            with (
                mock.patch.object(renderer, "_is_git_tracked", return_value=True),
                self.assertRaisesRegex(ValueError, "sha256 mismatch"),
            ):
                renderer._validate_evidence_source(source)

    def test_measured_evidence_accepts_complete_tracked_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifact = Path(td) / "artifact.json"
            artifact.write_text("{}", encoding="utf-8")
            source = self._measured_source(artifact, renderer._sha256(artifact))
            with mock.patch.object(renderer, "_is_git_tracked", return_value=True):
                result = renderer._validate_evidence_source(source)
            self.assertEqual(
                set(result["resources"]), set(renderer.MEASURED_RESOURCE_NAMES)
            )
            self.assertTrue(
                all(
                    item["verification"] == "local_git_tracked_hash_verified"
                    for item in result["resources"].values()
                )
            )

    def test_release_url_is_explicitly_declared_not_fetched(self) -> None:
        result = renderer._validate_resource_binding(
            "checkpoint",
            {
                "url": "https://example.com/releases/download/v1/checkpoint.pt",
                "sha256": "a" * 64,
            },
        )
        self.assertEqual(result["verification"], "declared_not_fetched")

    def test_measured_source_without_prediction_artifacts_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifact = Path(td) / "artifact.json"
            artifact.write_text("{}", encoding="utf-8")
            source = self._measured_source(artifact, renderer._sha256(artifact))
            source["provenance"]["baseline_predictions"]["path"] = str(
                Path(td) / "missing.json"
            )
            with (
                mock.patch.object(renderer, "_is_git_tracked", return_value=True),
                self.assertRaisesRegex(FileNotFoundError, "baseline_predictions"),
            ):
                renderer._validate_evidence_source(source)

    def test_simple_map_proxy_never_uses_coco_map_names(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifact = Path(td) / "artifact.json"
            artifact.write_text("{}", encoding="utf-8")
            source = self._measured_source(artifact, renderer._sha256(artifact))
            source["metrics"] = {
                "tent": {
                    "metric_backend": "simple_map_proxy",
                    "map50": 0.25,
                    "proxy_ap50": 0.25,
                }
            }
            with (
                mock.patch.object(renderer, "_is_git_tracked", return_value=True),
                self.assertRaisesRegex(ValueError, "never COCO mAP"),
            ):
                renderer._validate_evidence_source(source)

    def test_no_label_or_ground_truth_prediction_fallback_exists(self) -> None:
        source_text = (REPO_ROOT / "tools/render_ttt_manual_figures.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_load_gt_detections", source_text)
        self.assertNotIn('"detections": gt', source_text)
        self.assertNotIn("label-derived boxes", source_text)

    def test_failed_render_preserves_previous_six_file_bundle(self) -> None:
        source = {
            "evidence_kind": "synthetic_fixture",
            "promotion_eligible": False,
            "efficacy": "unavailable",
            "fixture": {"purpose": "layout only"},
        }
        validation = renderer._validate_evidence_source(source)
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as td:
            root = Path(td)
            docs_dir = root / "docs"
            manual_dir = root / "manual"
            snapshots = {}
            for directory in (docs_dir, manual_dir):
                directory.mkdir()
                for index, name in enumerate(renderer.FIGURE_NAMES):
                    path = directory / name
                    payload = f"previous-{directory.name}-{index}".encode()
                    path.write_bytes(payload)
                    snapshots[path] = payload

            with (
                mock.patch.object(
                    renderer,
                    "_draw_pipeline_figure",
                    side_effect=RuntimeError("injected render failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "injected render failure"),
            ):
                renderer._render_and_publish(
                    source,
                    validation=validation,
                    docs_dir=docs_dir,
                    manual_dir=manual_dir,
                )

            self.assertEqual(
                {path: path.read_bytes() for path in snapshots},
                snapshots,
            )


if __name__ == "__main__":
    unittest.main()
