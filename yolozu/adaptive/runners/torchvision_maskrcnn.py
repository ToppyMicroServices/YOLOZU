"""Network-free Torchvision Mask R-CNN runner using safetensors weights.

The runner intentionally supports one narrow interface contract.  It never
downloads weights, accepts an import path, or loads pickle at inference time.
An immutable bundle must pin the exact Torch/Torchvision runtime and one
``weights`` safetensors artifact.
"""

from __future__ import annotations

import importlib.metadata
import io
from decimal import Decimal
from typing import Any, Mapping

from PIL import Image

from ..bundle_registry import RunnerProbeResult
from ..bundles import AlgorithmBundleSpec
from ..contracts import EnvironmentProfile

_DECODER_ID = "pillow_rgb_v1"
_PREPROCESS_ID = "torchvision_maskrcnn_embedded_v1"
_POSTPROCESS_ID = "torchvision_coco_xyxy_v1"
_MAX_WEIGHT_BYTES = 512 * 1024 * 1024
_MAX_RESULTS = 100
_MIN_SCORE = Decimal("0.5")


def _runtime_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _number(value: float) -> str:
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text or "0"


def _read_artifact(artifacts: Any, artifact_id: str) -> bytes:
    size = artifacts.artifact_size_bytes(artifact_id)
    if size < 1 or size > _MAX_WEIGHT_BYTES:
        raise ValueError("safetensors artifact exceeds the runner bound")
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = artifacts.read_artifact_chunk(
            artifact_id,
            offset_bytes=offset,
            maximum_bytes=min(16 * 1024 * 1024, size - offset),
        )
        if not chunk:
            raise ValueError("safetensors artifact ended before its pinned size")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


class TorchvisionMaskRCNNRunner:
    """One code-owned CPU Mask R-CNN inference adapter."""

    runner_id = "torchvision"

    def __init__(self) -> None:
        self.runner_version = _runtime_version("torchvision") or "unavailable"
        self._model: Any | None = None
        self._torch: Any | None = None
        self._functional: Any | None = None
        self._category_names: tuple[str, ...] = ()
        self._bundle_labels: tuple[str, ...] = ()
        self._bundle_label_index: dict[str, int] = {}

    def probe(
        self,
        *,
        bundle: AlgorithmBundleSpec,
        environment: EnvironmentProfile,
    ) -> RunnerProbeResult:
        record = bundle.to_dict()
        if (
            record.get("runner_id") != self.runner_id
            or record.get("runner_version") != self.runner_version
        ):
            return RunnerProbeResult("unsupported", "runner_version_mismatch")
        if _runtime_version("torch") is None or self.runner_version == "unavailable":
            return RunnerProbeResult("unsupported", "runtime_unavailable")
        if record.get("loader_format") != "safetensors":
            return RunnerProbeResult("unsupported", "loader_format_unsupported")
        if record.get("unsafe_deserialization_required") is not False:
            return RunnerProbeResult("unsupported", "unsafe_loader_rejected")
        if record.get("tasks") != ["object_detection"]:
            return RunnerProbeResult("unsupported", "task_contract_mismatch")
        identities = (
            ("decoder", _DECODER_ID),
            ("preprocess", _PREPROCESS_ID),
            ("postprocess", _POSTPROCESS_ID),
        )
        if any(record.get(field, {}).get("id") != expected for field, expected in identities):
            return RunnerProbeResult("unsupported", "pipeline_identity_mismatch")
        runtime = record.get("runtime") or {}
        if runtime.get("runtime_id") != "torch" or runtime.get("provider_id") != "cpu":
            return RunnerProbeResult("unsupported", "runtime_contract_mismatch")
        observed = environment.to_dict().get("runtimes") or []
        matching = [item for item in observed if item.get("runtime_id") == "torch"]
        if (
            len(matching) != 1
            or matching[0].get("probe_status") != "present"
            or matching[0].get("version") != runtime.get("runtime_version")
            or "cpu" not in set(matching[0].get("provider_ids") or [])
        ):
            return RunnerProbeResult("unsupported", "runtime_observation_mismatch")
        return RunnerProbeResult("supported")

    def load(self, *, bundle: AlgorithmBundleSpec, artifacts: Any) -> None:
        record = bundle.to_dict()
        if tuple(artifacts.artifact_ids()) != ("weights",):
            raise ValueError("Mask R-CNN runner requires exactly one weights artifact")
        if artifacts.bundle_spec_digest != bundle.spec_digest:
            raise ValueError("pinned artifact bundle identity mismatch")
        raw_weights = _read_artifact(artifacts, "weights")

        import torch
        from safetensors.torch import load as load_safetensors
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_V2_Weights,
            maskrcnn_resnet50_fpn_v2,
        )
        from torchvision.transforms import functional as functional

        options = record.get("runner_options") or {}
        torch.set_num_threads(int(options.get("intra_op_threads", 1)))
        if "inter_op_threads" in options:
            try:
                torch.set_num_interop_threads(int(options["inter_op_threads"]))
            except RuntimeError:
                pass
        state = load_safetensors(raw_weights)
        model = maskrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
        )
        model.load_state_dict(state, strict=True)
        model.to("cpu")
        model.eval()

        vocabulary = record.get("class_vocabulary") or {}
        labels = tuple(str(item) for item in vocabulary.get("labels") or [])
        categories = tuple(
            str(item)
            for item in MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta[
                "categories"
            ]
        )
        ignored_categories = {"N/A", "__background__"}
        real_categories = {
            name for name in categories if name not in ignored_categories
        }
        if not labels or set(labels) != real_categories:
            raise ValueError("bundle vocabulary does not match Torchvision COCO labels")

        self._torch = torch
        self._functional = functional
        self._model = model
        self._category_names = categories
        self._bundle_labels = labels
        self._bundle_label_index = {label: index for index, label in enumerate(labels)}

    def _input_tensor(self, input_item: Any) -> tuple[Any, int, int]:
        if self._functional is None:
            raise RuntimeError("runner is not loaded")
        source = input_item.read_source_bytes()
        with Image.open(io.BytesIO(source)) as image:
            if bool(getattr(image, "is_animated", False)):
                raise ValueError("animated input is unsupported")
            rgb = image.convert("RGB")
            width, height = rgb.size
            tensor = self._functional.pil_to_tensor(rgb).to(dtype=self._torch.float32)
            tensor = tensor / 255.0
        return tensor, width, height

    def warmup(self, *, input_item: Any) -> None:
        if self._model is None or self._torch is None:
            raise RuntimeError("runner is not loaded")
        tensor, _width, _height = self._input_tensor(input_item)
        with self._torch.inference_mode():
            self._model([tensor])

    def predict(
        self,
        *,
        input_item: Any,
        requested_labels: tuple[str, ...],
    ) -> tuple[Mapping[str, Any], ...]:
        if self._model is None or self._torch is None:
            raise RuntimeError("runner is not loaded")
        tensor, width, height = self._input_tensor(input_item)
        with self._torch.inference_mode():
            output = self._model([tensor])[0]
        requested = set(requested_labels)
        records: list[Mapping[str, Any]] = []
        for box, score_value, category_value in zip(
            output["boxes"],
            output["scores"],
            output["labels"],
            strict=True,
        ):
            score = Decimal(str(float(score_value.detach().cpu().item())))
            if score < _MIN_SCORE:
                continue
            category_index = int(category_value.detach().cpu().item())
            if not 0 <= category_index < len(self._category_names):
                continue
            category = self._category_names[category_index]
            if category in {"N/A", "__background__"} or category not in requested:
                continue
            x1, y1, x2, y2 = [
                float(value)
                for value in box.detach().cpu().tolist()
            ]
            normalized = (
                max(0.0, min(1.0, x1 / width)),
                max(0.0, min(1.0, y1 / height)),
                max(0.0, min(1.0, x2 / width)),
                max(0.0, min(1.0, y2 / height)),
            )
            if normalized[0] > normalized[2] or normalized[1] > normalized[3]:
                continue
            records.append(
                {
                    "native_class_index": self._bundle_label_index[category],
                    "score": _number(float(score)),
                    "bbox": [_number(value) for value in normalized],
                }
            )
            if len(records) >= _MAX_RESULTS:
                break
        return tuple(records)

    def close(self) -> None:
        self._model = None
        self._torch = None
        self._functional = None
        self._category_names = ()
        self._bundle_labels = ()
        self._bundle_label_index = {}


def create_torchvision_runner() -> TorchvisionMaskRCNNRunner:
    return TorchvisionMaskRCNNRunner()
