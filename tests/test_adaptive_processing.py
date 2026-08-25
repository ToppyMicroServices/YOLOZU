from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

from PIL import Image

from tests.test_adaptive_bundle_contracts import _bundle_payload
from tests.test_adaptive_selector import _context, _job, _select
from yolozu.adaptive.artifact_resolver import ArtifactResolver
from yolozu.adaptive.bundle_registry import RunnerProbeResult
from yolozu.adaptive.bundles import validate_algorithm_bundle_spec
from yolozu.adaptive.inventory import pin_decoded_inputs
from yolozu.adaptive.processing import (
    IsolatedRunnerCapability,
    ProcessingError,
    _execute_pinned_pipeline,
    _resolve_execution_route,
    process_images,
)
from yolozu.adaptive import processing as processing_module
from yolozu.adaptive import recommendation as recommendation_module
from yolozu.adaptive.recommendation import recommend_image_pipeline


class _Session:
    runner_id = "onnxruntime"
    runner_version = "1.23.0"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def probe(self, timeout_seconds: int) -> RunnerProbeResult:
        self.calls.append(("probe", timeout_seconds))
        return RunnerProbeResult("supported")

    def load(self, timeout_seconds: int) -> None:
        self.calls.append(("load", timeout_seconds))

    def predict(
        self, index: int, timeout_seconds: int
    ) -> tuple[Mapping[str, Any], ...]:
        self.calls.append(("predict", timeout_seconds))
        return (
            {
                "native_class_index": 0,
                "score": "0.9",
                "bbox": ["0.1", "0.2", "0.8", "0.9"],
            },
            {
                "native_class_index": 1,
                "score": "0.7",
                "bbox": ["0", "0", "1", "1"],
            },
        )

    def close(self, timeout_seconds: int) -> None:
        self.calls.append(("close", timeout_seconds))


class _IsolatedService:
    def __init__(self, policy_digest: str, status: str = "available") -> None:
        self.capability = IsolatedRunnerCapability(
            runner_id="onnxruntime",
            policy_digest=policy_digest,
            status=status,
            backend_id="fixture-isolation" if status == "available" else None,
            backend_version="1" if status == "available" else None,
            image_present=True if status == "available" else None,
        )
        self.open_count = 0

    def open_session(self, **_kwargs: object) -> _Session:
        self.open_count += 1
        return _Session()


class TestAdaptiveProcessing(unittest.TestCase):
    def _pinned_context(self, root: Path):
        image_path = root / "input.png"
        Image.new("RGB", (8, 6), color=(10, 20, 30)).save(image_path)
        artifact_root = root / "models"
        artifact_path = artifact_root / "weights" / "model.onnx"
        artifact_path.parent.mkdir(parents=True)
        artifact_path.write_bytes(b"model")
        bundle_payload = _bundle_payload()
        bundle = validate_algorithm_bundle_spec(bundle_payload)
        decision = _select(_context(bundle_payload))
        job = _job()
        inputs = pin_decoded_inputs(
            "input.png",
            input_mode="single_image",
            workspace_root=root,
            max_images=1,
        )
        resolver = ArtifactResolver(workspace=root, artifact_root=artifact_root)
        artifacts = resolver.pin(bundle)
        return job, decision, bundle, inputs, resolver, artifacts

    def test_dry_run_does_not_create_session_or_write_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            job, decision, bundle, inputs, resolver, artifacts = self._pinned_context(root)
            calls = 0

            def factory() -> _Session:
                nonlocal calls
                calls += 1
                return _Session()

            try:
                result = _execute_pinned_pipeline(
                    job=job,
                    decision=decision,
                    bundle=bundle,
                    inputs=inputs,
                    artifacts=artifacts,
                    workspace=root,
                    destination="output",
                    dry_run=True,
                    force=False,
                    session_factory=factory,
                    outer_deadline_ns=2**63 - 1,
                )
            finally:
                artifacts.close()
                resolver.close()
                inputs.close()
            self.assertTrue(result["ok"])
            self.assertFalse(result["executed"])
            self.assertEqual(calls, 0)
            self.assertFalse((root / "output").exists())

    def test_public_service_requires_selected_decision_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            Image.new("RGB", (8, 6)).save(root / "input.png")
            job = _job().to_dict()
            recommendation = recommend_image_pipeline(
                job,
                "input.png",
                workspace_root=root,
            )
            self.assertEqual(recommendation["decision"]["status"], "abstained")
            with self.assertRaises(ProcessingError) as rejected:
                process_images(
                    job,
                    recommendation["decision"],
                    "input.png",
                    "output",
                    workspace_root=root,
                )
            self.assertEqual(rejected.exception.code, "selection_required")
            self.assertFalse((root / "output").exists())

    def test_execution_filters_classes_and_publishes_exact_managed_tree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            job, decision, bundle, inputs, resolver, artifacts = self._pinned_context(root)
            session = _Session()
            try:
                result = _execute_pinned_pipeline(
                    job=job,
                    decision=decision,
                    bundle=bundle,
                    inputs=inputs,
                    artifacts=artifacts,
                    workspace=root,
                    destination="output",
                    dry_run=False,
                    force=False,
                    session_factory=lambda: session,
                    outer_deadline_ns=2**63 - 1,
                )
            finally:
                artifacts.close()
                resolver.close()
                inputs.close()
            self.assertTrue(result["executed"])
            self.assertEqual(
                session.calls,
                [("probe", 30), ("load", 600), ("predict", 300), ("close", 30)],
            )
            self.assertEqual(
                sorted(path.name for path in (root / "output").iterdir()),
                ["checksums.json", "predictions.json", "provenance.json"],
            )
            predictions = json.loads((root / "output" / "predictions.json").read_text())
            detections = predictions[0]["detections"]
            self.assertEqual(len(detections), 1)
            self.assertEqual(detections[0]["class_id"], 0)
            self.assertEqual(detections[0]["meta"]["requested_label"], "cat")
            manifest = json.loads((root / "output" / "checksums.json").read_text())
            self.assertEqual(
                manifest["expected_paths"],
                ["predictions.json", "provenance.json"],
            )
            for entry in manifest["files"]:
                payload = (root / "output" / entry["path"]).read_bytes()
                self.assertEqual(entry["size_bytes"], len(payload))
                self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())

    def test_instance_mask_is_validated_referenced_and_checksummed(self) -> None:
        class MaskSession(_Session):
            def __init__(self, mask: bytes) -> None:
                super().__init__()
                self.mask = mask

            def predict(
                self, index: int, timeout_seconds: int
            ) -> tuple[Mapping[str, Any], ...]:
                self.calls.append(("predict", timeout_seconds))
                return (
                    {
                        "native_class_index": 0,
                        "score": "0.9",
                        "bbox": ["0.1", "0.2", "0.8", "0.9"],
                        "mask_png": self.mask,
                    },
                )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            Image.new("RGB", (8, 6)).save(root / "input.png")
            buffer = io.BytesIO()
            Image.new("L", (8, 6), color=255).save(buffer, format="PNG")
            bundle_payload = _bundle_payload()
            bundle_payload["tasks"] = ["instance_segmentation", "object_detection"]
            from yolozu.adaptive.canonical import canonical_sha256_v1

            bundle_payload["spec_digest"] = "0" * 64
            bundle_payload["spec_digest"] = canonical_sha256_v1(
                bundle_payload, own_digest_field="spec_digest"
            )
            bundle = validate_algorithm_bundle_spec(bundle_payload)
            job = _job(task="instance_segmentation")
            decision = _select(_context(bundle_payload, job=job))
            artifact_root = root / "models"
            artifact_path = artifact_root / "weights" / "model.onnx"
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_bytes(b"model")
            with (
                pin_decoded_inputs(
                    "input.png",
                    input_mode="single_image",
                    workspace_root=root,
                    max_images=1,
                ) as inputs,
                ArtifactResolver(workspace=root, artifact_root=artifact_root) as resolver,
                resolver.pin(bundle) as artifacts,
            ):
                _execute_pinned_pipeline(
                    job=job,
                    decision=decision,
                    bundle=bundle,
                    inputs=inputs,
                    artifacts=artifacts,
                    workspace=root,
                    destination="output",
                    dry_run=False,
                    force=False,
                    session_factory=lambda: MaskSession(buffer.getvalue()),
                    outer_deadline_ns=2**63 - 1,
                )
            predictions = json.loads((root / "output" / "predictions.json").read_text())
            mask_path = predictions[0]["detections"][0]["mask"]
            self.assertEqual(mask_path, "artifacts/masks/000000-0000.png")
            with Image.open(root / "output" / mask_path) as mask:
                self.assertEqual(mask.size, (8, 6))
            manifest = json.loads((root / "output" / "checksums.json").read_text())
            self.assertIn(mask_path, manifest["expected_paths"])

    def test_execution_closes_once_and_publishes_nothing_on_invalid_output(self) -> None:
        class InvalidSession(_Session):
            def predict(
                self, index: int, timeout_seconds: int
            ) -> tuple[Mapping[str, Any], ...]:
                self.calls.append(("predict", timeout_seconds))
                return (
                    {
                        "native_class_index": 0,
                        "score": "0.9",
                        "bbox": ["0.8", "0.2", "0.1", "0.9"],
                    },
                )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            job, decision, bundle, inputs, resolver, artifacts = self._pinned_context(root)
            session = InvalidSession()
            try:
                with self.assertRaises(Exception):
                    _execute_pinned_pipeline(
                        job=job,
                        decision=decision,
                        bundle=bundle,
                        inputs=inputs,
                        artifacts=artifacts,
                        workspace=root,
                        destination="output",
                        dry_run=False,
                        force=False,
                        session_factory=lambda: session,
                        outer_deadline_ns=2**63 - 1,
                    )
            finally:
                artifacts.close()
                resolver.close()
                inputs.close()
            self.assertEqual([call[0] for call in session.calls].count("close"), 1)
            self.assertFalse((root / "output").exists())

    def test_isolation_route_requires_matching_live_code_owned_capability(self) -> None:
        payload = _bundle_payload()
        payload["execution_trust_class"] = "third_party_isolated"
        payload["execution_isolation_policy_digest"] = "f" * 64
        payload["spec_digest"] = "0" * 64
        from yolozu.adaptive.canonical import canonical_sha256_v1

        payload["spec_digest"] = canonical_sha256_v1(
            payload, own_digest_field="spec_digest"
        )
        bundle = validate_algorithm_bundle_spec(payload)
        with self.assertRaises(ProcessingError) as unavailable:
            _resolve_execution_route(bundle, isolated_services={})
        self.assertEqual(unavailable.exception.code, "isolation_unsupported")

        mismatch = _IsolatedService("e" * 64)
        with self.assertRaises(ProcessingError) as wrong_policy:
            _resolve_execution_route(
                bundle,
                isolated_services={"onnxruntime": mismatch},
            )
        self.assertEqual(wrong_policy.exception.code, "isolation_policy_mismatch")
        self.assertEqual(mismatch.open_count, 0)

        matching = _IsolatedService("f" * 64)
        route = _resolve_execution_route(
            bundle,
            isolated_services={"onnxruntime": matching},
        )
        self.assertEqual(route.kind, "third_party_isolated")
        self.assertEqual(matching.open_count, 0)

        with patch.dict(
            processing_module._CODE_OWNED_ISOLATED_SERVICES,
            {"onnxruntime": matching},
            clear=True,
        ):
            observation = recommendation_module._isolation_observations(
                _context(payload).registry
            )[bundle.spec_digest]
        self.assertEqual(observation.status, "supported")
        self.assertEqual(observation.isolation_policy_digest, "f" * 64)
        self.assertTrue(observation.image_present)


if __name__ == "__main__":
    unittest.main()
