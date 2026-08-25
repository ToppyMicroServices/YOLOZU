from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tests.test_adaptive_bundle_contracts import _bundle_payload, _registry_payload
from tests.test_adaptive_bundle_registry import _write_custom
from tests.test_adaptive_recommendation import _job
from tests.test_adaptive_selector import _context, _select
from yolozu.adaptive.canonical import canonical_sha256_v1
from yolozu.adaptive.recommendation import recommend_image_pipeline
from yolozu.integrations.tool_runner import (
    process_images,
    recommend_image_pipeline as recommend_image_pipeline_tool,
)


class TestAdaptiveRoutingEndToEnd(unittest.TestCase):
    def test_public_and_fixture_boundaries_have_exact_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            Image.new("RGB", (8, 6), color=(10, 20, 30)).save(
                workspace / "input.png"
            )
            job = _job()

            packaged = recommend_image_pipeline(
                job,
                "input.png",
                workspace_root=workspace,
                decided_at="2026-08-25T14:30:00Z",
            )

            custom_root = _write_custom(
                workspace,
                _registry_payload(_bundle_payload()),
            )
            custom = recommend_image_pipeline(
                job,
                "input.png",
                workspace_root=workspace,
                registry_root=custom_root.name,
                decided_at="2026-08-25T14:30:00Z",
            )

            previous = Path.cwd()
            os.chdir(workspace)
            try:
                rejected = process_images(
                    job,
                    packaged["decision"],
                    "input.png",
                    "output",
                )
                corrupt_root = workspace / "corrupt-evidence"
                corrupt_root.mkdir()
                corrupt_root.joinpath("evidence_activation.jsonl").write_bytes(b"{\n")
                corrupt = recommend_image_pipeline_tool(
                    job,
                    "input.png",
                    evidence_root=corrupt_root.name,
                )
            finally:
                os.chdir(previous)

            test_only = _bundle_payload()
            test_only["test_only"] = True
            test_only["spec_digest"] = canonical_sha256_v1(
                test_only,
                own_digest_field="spec_digest",
            )
            fixture_decision = _select(_context(test_only)).to_dict()

            actual = {
                "installed_default": {
                    "status": packaged["decision"]["status"],
                    "reason_codes": [],
                },
                "operator_catalog": {
                    "status": custom["decision"]["status"],
                    "reason_codes": custom["decision"]["candidate_evaluations"][0][
                        "reason_codes"
                    ],
                },
                "public_process": {
                    "status": "rejected" if not rejected["ok"] else "accepted",
                    "reason_codes": [rejected["error"]["code"]],
                },
                "corrupt_evidence": {
                    "status": "rejected" if not corrupt["ok"] else "accepted",
                    "reason_codes": [corrupt["error"]["code"]],
                },
                "pure_selector_test_only": {
                    "status": fixture_decision["status"],
                    "reason_codes": fixture_decision["candidate_evaluations"][0][
                        "reason_codes"
                    ],
                },
            }
            self.assertEqual(
                actual,
                {
                    "installed_default": {
                        "status": "abstained",
                        "reason_codes": [],
                    },
                    "operator_catalog": {
                        "status": "abstained",
                        "reason_codes": [
                            "lifecycle_untrusted",
                            "registry_untrusted",
                        ],
                    },
                    "public_process": {
                        "status": "rejected",
                        "reason_codes": ["selection_required"],
                    },
                    "corrupt_evidence": {
                        "status": "rejected",
                        "reason_codes": ["invalid_evidence"],
                    },
                    "pure_selector_test_only": {
                        "status": "abstained",
                        "reason_codes": ["test_only"],
                    },
                },
            )
            self.assertFalse((workspace / "output").exists())


if __name__ == "__main__":
    unittest.main()
