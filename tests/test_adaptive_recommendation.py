from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from unittest.mock import patch

from yolozu.adaptive import recommendation as recommendation_module
from yolozu.adaptive.recommendation import (
    RecommendationError,
    recommend_image_pipeline,
)
from yolozu.integrations.tool_runner import (
    recommend_image_pipeline as recommend_image_pipeline_tool_runner,
)


def _job() -> dict:
    return {
        "schema_version": 1,
        "task": "object_detection",
        "prompt_mode": "fixed_classes",
        "fixed_classes": ["cat"],
        "input_mode": "single_image",
        "execution_mode": "batch",
        "batch_size": 1,
        "concurrency": 1,
        "max_images": 1,
        "max_results_per_image": 100,
        "job_timeout_seconds": 60,
        "ranking_policy": "latency_first",
        "allowed_maturities": ["Experimental", "Stable"],
        "network_policy": "deny",
        "compute_policy": "auto",
        "provider_allowlist": [],
        "precision_allowlist": [],
        "spdx_allowlist": [],
    }


class TestAdaptiveRecommendation(unittest.TestCase):
    def test_evidence_directory_walk_closes_owned_descriptors_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "first").mkdir()
            opened: list[int] = []
            closed: list[int] = []
            real_open = os.open
            real_close = os.close

            def tracked_open(path: object, flags: int, **kwargs: object) -> int:
                if path == "second":
                    raise OSError("injected directory-open failure")
                descriptor = real_open(path, flags, **kwargs)
                opened.append(descriptor)
                return descriptor

            def tracked_close(descriptor: int) -> None:
                closed.append(descriptor)
                real_close(descriptor)

            with (
                patch.object(recommendation_module.os, "open", side_effect=tracked_open),
                patch.object(recommendation_module.os, "close", side_effect=tracked_close),
                self.assertRaisesRegex(RecommendationError, "qualification report"),
            ):
                recommendation_module._read_regular_at(
                    root,
                    ("first", "second", "qualification_report.json"),
                    maximum_bytes=1024,
                    label="qualification report",
                )

            self.assertEqual(sorted(opened), sorted(closed))

    def test_packaged_candidate_baselines_abstain_without_writes_or_path_leaks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            image_path = workspace / "input.png"
            Image.new("RGB", (8, 6), color=(10, 20, 30)).save(image_path)
            before = sorted(path.relative_to(workspace) for path in workspace.rglob("*"))

            result = recommend_image_pipeline(
                _job(),
                "input.png",
                workspace_root=workspace,
                decided_at="2026-08-25T14:30:00Z",
            )

            after = sorted(path.relative_to(workspace) for path in workspace.rglob("*"))
            self.assertEqual(before, after)
            self.assertTrue(result["ok"])
            self.assertEqual(result["decision"]["status"], "abstained")
            self.assertEqual(
                result["recommendation_metadata"]["screening_source"],
                "packaged_ssot",
            )
            self.assertEqual(
                result["recommendation_metadata"]["screening_trust_domain"],
                "yolozu_managed",
            )
            self.assertEqual(result["decision"]["registry_bundle_count"], 3)
            self.assertEqual(
                [
                    evaluation["reason_codes"]
                    for evaluation in result["decision"]["candidate_evaluations"]
                ],
                [["maturity_disallowed"]] * 3,
            )
            self.assertEqual(
                [
                    observation["status"]
                    for observation in result["recommendation_metadata"][
                        "artifact_observations"
                    ]
                ],
                ["not_checked_due_to_prior_filter"] * 3,
            )
            self.assertIsNone(
                result["recommendation_metadata"][
                    "selected_artifact_resolver_state_digest"
                ]
            )
            rendered = repr(result)
            self.assertNotIn(str(workspace), rendered)
            self.assertNotIn("input.png", rendered)

    def test_prior_filter_does_not_require_artifact_root_io(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            Image.new("RGB", (4, 4)).save(workspace / "input.png")

            result = recommend_image_pipeline(
                _job(),
                "input.png",
                workspace_root=workspace,
                artifact_root="cache-not-created",
                decided_at="2026-08-25T14:30:00Z",
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["decision"]["status"], "abstained")
            self.assertFalse((workspace / "cache-not-created").exists())

    def test_custom_screening_is_path_derived_and_malformed_stream_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            Image.new("RGB", (4, 4)).save(workspace / "input.png")
            screening = workspace / "screening"
            screening.mkdir()
            (screening / "candidate_screening.jsonl").write_bytes(b"")

            result = recommend_image_pipeline(
                _job(),
                "input.png",
                workspace_root=workspace,
                screening_root="screening",
                decided_at="2026-08-25T14:30:00Z",
            )
            self.assertEqual(
                result["recommendation_metadata"]["screening_source"],
                "workspace_screening",
            )
            self.assertEqual(
                result["recommendation_metadata"]["screening_trust_domain"],
                "operator_asserted",
            )

            (screening / "candidate_screening.jsonl").write_bytes(b"{}")
            with self.assertRaisesRegex(RecommendationError, "screening") as malformed:
                recommend_image_pipeline(
                    _job(),
                    "input.png",
                    workspace_root=workspace,
                    screening_root="screening",
                    decided_at="2026-08-25T14:30:00Z",
                )
            self.assertEqual(malformed.exception.code, "invalid_screening")

            (screening / "candidate_screening.jsonl").unlink()
            with self.assertRaises(RecommendationError) as missing:
                recommend_image_pipeline(
                    _job(),
                    "input.png",
                    workspace_root=workspace,
                    screening_root="screening",
                    decided_at="2026-08-25T14:30:00Z",
                )
            self.assertEqual(missing.exception.code, "invalid_screening")

    def test_unsafe_paths_and_malformed_job_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            image_path = workspace / "input.png"
            Image.new("RGB", (4, 4)).save(image_path)
            with self.assertRaisesRegex(RecommendationError, "job_spec") as malformed:
                recommend_image_pipeline(
                    {"schema_version": 1},
                    "input.png",
                    workspace_root=workspace,
                )
            self.assertEqual(malformed.exception.code, "invalid_job_spec")

            with self.assertRaisesRegex(RecommendationError, "input_path") as unsafe:
                recommend_image_pipeline(
                    _job(),
                    "../input.png",
                    workspace_root=workspace,
                )
            self.assertEqual(unsafe.exception.code, "invalid_input")

            with self.assertRaisesRegex(RecommendationError, "artifact_root") as root:
                recommend_image_pipeline(
                    _job(),
                    "input.png",
                    workspace_root=workspace,
                    artifact_root="../models",
                )
            self.assertEqual(root.exception.code, "unsafe_artifact_root")

    def test_tool_runner_returns_structured_error_without_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            previous = Path.cwd()
            os.chdir(workspace)
            try:
                result = recommend_image_pipeline_tool_runner(
                    _job(),
                    "../outside.png",
                )
            finally:
                os.chdir(previous)
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"]["code"], "invalid_input")
            self.assertNotIn(str(workspace), repr(result))


if __name__ == "__main__":
    unittest.main()
