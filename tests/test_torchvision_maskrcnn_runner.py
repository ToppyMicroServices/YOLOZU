from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_adaptive_bundle_contracts import _bundle_payload
from yolozu.adaptive.bundles import validate_algorithm_bundle_spec
from yolozu.adaptive.canonical import canonical_sha256_v1
from yolozu.adaptive.qualification import _CODE_OWNED_RUNNER_FACTORIES
from yolozu.adaptive.runners.torchvision_maskrcnn import (
    TorchvisionMaskRCNNRunner,
    _number,
    pipeline_identities,
)

_VERSIONS = {"torchvision": "0.27.0", "torch": "2.12.0", "Pillow": "12.2.0", "safetensors": "0.6.2"}


class _Environment:
    def __init__(self, torch_version: str) -> None:
        self.torch_version = torch_version

    def to_dict(self):
        return {
            "runtimes": [
                {
                    "runtime_id": "torch",
                    "probe_status": "present",
                    "version": self.torch_version,
                    "provider_ids": ["cpu"],
                }
            ]
        }


def _bundle():
    payload = _bundle_payload(data=b"safe-weights")
    payload["tasks"] = ["object_detection"]
    payload["prompt_modes"] = ["fixed_classes"]
    payload.pop("text_prompt_support", None)
    payload["adapter_backend_id"] = "torchvision-maskrcnn"
    payload["runner_id"] = "torchvision"
    payload["runner_version"] = _VERSIONS["torchvision"]
    payload["loader_format"] = "safetensors"
    payload["unsafe_deserialization_required"] = False
    payload["runtime"] = {
        "runtime_id": "torch",
        "runtime_version": _VERSIONS["torch"],
        "provider_id": "cpu",
        "provider_version": "1",
        "precision": "fp32",
        "architecture": "any",
        "accelerator_requirement": "none",
    }
    payload.update(pipeline_identities())
    payload["artifacts"][0]["artifact_id"] = "weights"
    payload["artifacts"][0]["cache_key"] = "weights/model.safetensors"
    payload["artifact_set_digest"] = canonical_sha256_v1(payload["artifacts"])
    payload["spec_digest"] = "0" * 64
    payload["spec_digest"] = canonical_sha256_v1(
        payload,
        own_digest_field="spec_digest",
    )
    return validate_algorithm_bundle_spec(payload)


class TestTorchvisionMaskRCNNRunner(unittest.TestCase):
    def setUp(self) -> None:
        versions = patch("yolozu.adaptive.runners.torchvision_maskrcnn._runtime_version", side_effect=_VERSIONS.get)
        versions.start()
        self.addCleanup(versions.stop)

    def test_factory_is_code_owned_and_lazy(self) -> None:
        self.assertIn("torchvision", _CODE_OWNED_RUNNER_FACTORIES)
        runner = _CODE_OWNED_RUNNER_FACTORIES["torchvision"]()
        self.assertIsInstance(runner, TorchvisionMaskRCNNRunner)
        self.assertIsNone(runner._model)

    def test_probe_accepts_only_exact_safe_cpu_contract(self) -> None:
        bundle = _bundle()
        runner = TorchvisionMaskRCNNRunner()
        supported = runner.probe(
            bundle=bundle,
            environment=_Environment(_VERSIONS["torch"]),
        )
        self.assertEqual(supported.status, "supported")

        wrong_runtime = runner.probe(
            bundle=bundle,
            environment=_Environment("0.0.0"),
        )
        self.assertEqual(wrong_runtime.status, "unsupported")
        self.assertEqual(
            wrong_runtime.reason_code,
            "runtime_observation_mismatch",
        )

    def test_output_number_is_bounded_canonical_decimal(self) -> None:
        self.assertEqual(_number(0.0), "0")
        self.assertEqual(_number(0.5), "0.5")
        self.assertEqual(_number(0.123456789), "0.12345679")
        with self.assertRaises(ValueError):
            _number(float("nan"))

    def test_probe_rejects_component_drift_and_unsupported_options(self) -> None:
        for field, key, value in (
            ("decoder", "digest", "f" * 64),
            ("preprocess", "version", "2"),
            ("postprocess", "digest", "f" * 64),
            ("runtime", "runtime_version", "2.0"),
            ("runtime", "precision", "fp16"),
            ("runner_options", "optimization_level", "all"),
        ):
            with self.subTest(field=field, key=key):
                record = copy.deepcopy(_bundle().to_dict())
                record[field][key] = value
                record["spec_digest"] = canonical_sha256_v1(record, own_digest_field="spec_digest")
                result = TorchvisionMaskRCNNRunner().probe(
                    bundle=validate_algorithm_bundle_spec(record),
                    environment=_Environment(_VERSIONS["torch"]),
                )
                self.assertEqual(result.status, "unsupported")

    def test_review_proposal_is_valid_but_cannot_enter_qualification(self) -> None:
        from yolozu.adaptive.qualification import _select_bundle, QualificationError

        root = Path(__file__).resolve().parents[1]
        review = json.loads((root / "reports/image_service_candidate_review_2026-09-20.json").read_text())
        spec = validate_algorithm_bundle_spec(review["proposed_bundle_spec"])
        self.assertFalse(review["registered"])
        self.assertFalse(review["qualified"])
        self.assertFalse(review["promoted"])
        self.assertEqual(spec.to_dict()["artifacts"][0]["license_expression"], "unknown")
        with self.assertRaisesRegex(QualificationError, "bundle_not_found"):
            _select_bundle(bundle_id=spec.to_dict()["bundle_id"], bundle_version=spec.to_dict()["bundle_version"], channel="Experimental")


if __name__ == "__main__":
    unittest.main()
