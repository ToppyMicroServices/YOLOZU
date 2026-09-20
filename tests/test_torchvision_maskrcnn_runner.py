from __future__ import annotations

import importlib.metadata
import unittest

from tests.test_adaptive_bundle_contracts import _bundle_payload
from yolozu.adaptive.bundles import validate_algorithm_bundle_spec
from yolozu.adaptive.canonical import canonical_sha256_v1
from yolozu.adaptive.qualification import _CODE_OWNED_RUNNER_FACTORIES
from yolozu.adaptive.runners.torchvision_maskrcnn import (
    TorchvisionMaskRCNNRunner,
    _number,
)


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
    payload["runner_version"] = importlib.metadata.version("torchvision")
    payload["loader_format"] = "safetensors"
    payload["unsafe_deserialization_required"] = False
    payload["runtime"] = {
        "runtime_id": "torch",
        "runtime_version": importlib.metadata.version("torch"),
        "provider_id": "cpu",
        "provider_version": "1",
        "precision": "fp32",
        "architecture": "any",
        "accelerator_requirement": "none",
    }
    payload["decoder"] = {
        "id": "pillow_rgb_v1",
        "version": "1",
        "digest": "b" * 64,
    }
    payload["preprocess"] = {
        "id": "torchvision_maskrcnn_embedded_v1",
        "version": "1",
        "digest": "c" * 64,
    }
    payload["postprocess"] = {
        "id": "torchvision_coco_xyxy_v1",
        "version": "1",
        "digest": "d" * 64,
    }
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
            environment=_Environment(importlib.metadata.version("torch")),
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


if __name__ == "__main__":
    unittest.main()
