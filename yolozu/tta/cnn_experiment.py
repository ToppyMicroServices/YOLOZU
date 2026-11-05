"""Bounded CNN quantized-LoRA test-time training research experiment.

This module deliberately uses ordinary floating-point convolution after unpacking
INT4 weights. It measures a reference storage/adaptation path, not accelerated
INT4 arithmetic, and does not implement the original NF4 QLoRA recipe.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    checkpoint: str
    output: str
    split: str = "val2017"
    support_images: int = 2
    eval_images: int = 4
    steps: int = 1
    rank: int = 4
    learning_rate: float = 1e-4
    pseudo_threshold: float = 0.7
    pseudo_topk: int = 20
    min_size: int = 256
    max_size: int = 384
    brightness: float = 0.55
    seed: int = 11
    device: str = "cpu"
    threads: int = 2
    adapter_include: tuple[str, ...] = ("backbone.body.layer4",)
    quant_include: tuple[str, ...] = ("*",)
    skip_eval: bool = False

    def validate(self) -> None:
        for name, low, high in (
            ("support_images", 1, 64), ("eval_images", 1, 256),
            ("steps", 1, 10), ("rank", 1, 64), ("threads", 1, 16),
            ("min_size", 64, 1024), ("max_size", 64, 1536),
            ("pseudo_topk", 1, 100),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if self.max_size < self.min_size:
            raise ValueError("max_size must be at least min_size")
        for name in ("learning_rate", "pseudo_threshold", "brightness"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("only cpu and cuda are supported by this research runner")
        if not self.adapter_include or not self.quant_include:
            raise ValueError("adapter_include and quant_include must be nonempty")
        if Path(self.split).name != self.split or self.split in (".", ".."):
            raise ValueError("split must be one directory name")
        if not Path(self.checkpoint).is_file():
            raise ValueError("checkpoint must be an existing local file; no download is performed")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def discover_images(config: ExperimentConfig) -> tuple[list[Path], list[Path]]:
    directory = Path(config.dataset) / "images" / config.split
    paths = sorted(path.resolve() for path in directory.iterdir()
                   if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    count = config.support_images + config.eval_images
    if len(paths) < count:
        raise ValueError(f"need {count} images for disjoint support/evaluation; found {len(paths)}")
    selected = paths[:count]
    if len(set(selected)) != count:
        raise ValueError("duplicate image paths in support/evaluation")
    support, evaluation = selected[:config.support_images], selected[config.support_images:]
    support_hashes = {sha256_file(path) for path in support}
    if any(sha256_file(path) in support_hashes for path in evaluation):
        raise ValueError("support/evaluation share identical image bytes")
    return support, evaluation


def load_image(path: Path, device: str) -> Any:
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"), copy=True)
    return torch.from_numpy(rgb).permute(2, 0, 1).to(device=device, dtype=torch.float32) / 255.0


def shifted(image: Any, brightness: float) -> Any:
    """Deterministic photometric change; boxes and image dimensions stay valid."""
    return (image * brightness).clamp(0, 1)


def _build_detector(config: ExperimentConfig) -> Any:
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

    model = maskrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None, min_size=config.min_size,
        max_size=config.max_size, box_detections_per_img=100,
    )
    state = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    # Keep the official pretrained box branch. This experiment does not train
    # masks, and never treats generated boxes as ground-truth segmentation masks.
    model.roi_heads.mask_roi_pool = None
    model.roi_heads.mask_head = None
    model.roi_heads.mask_predictor = None
    return model.to(config.device).eval()


def coco_category_mapping() -> dict[int, int]:
    """Map official sparse Torchvision COCO IDs to conventional YOLO COCO80 IDs."""
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

    categories = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta["categories"]
    return {category_id: class_id for class_id, category_id in enumerate(
        index for index, name in enumerate(categories) if name not in ("N/A", "__background__")
    )}


def select_pseudo_target(output: dict[str, Any], *, threshold: float, topk: int) -> dict[str, Any]:
    import torch

    boxes, scores, labels = output["boxes"], output["scores"], output["labels"]
    valid = (torch.isfinite(scores) & torch.isfinite(boxes).all(dim=1)
             & (scores >= threshold) & (labels > 0)
             & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
    indices = torch.where(valid)[0]
    indices = indices[torch.argsort(scores[indices], descending=True)[:topk]]
    return {"boxes": boxes[indices].detach().clone(), "labels": labels[indices].detach().clone()}


def _sync(device: str) -> None:
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


def adapt(model: Any, support: list[Any], config: ExperimentConfig) -> dict[str, Any]:
    """Fixed teacher pseudo labels; support ground-truth labels never enter here."""
    import torch
    from yolozu.tta.quantized_lora import (
        adapter_state_dict, base_fingerprint, load_adapter_state_dict,
    )

    before_base = base_fingerprint(model)
    initial_adapter = adapter_state_dict(model)
    trainables = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainables or any(not name.endswith(("lora_A", "lora_B")) for name, _ in trainables):
        raise ValueError("TTT must update only nonempty LoRA parameter set")
    initial_parameters = [parameter.detach().clone() for _, parameter in trainables]
    _sync(config.device)
    started = time.perf_counter()
    model.eval()
    targets = []
    with torch.no_grad():
        for image in support:
            targets.append(select_pseudo_target(model([image])[0],
                           threshold=config.pseudo_threshold, topk=config.pseudo_topk))
    _sync(config.device)
    teacher_seconds = time.perf_counter() - started
    optimizer = torch.optim.SGD([parameter for _, parameter in trainables], lr=config.learning_rate)
    log: list[dict[str, Any]] = []
    _sync(config.device)
    update_started = time.perf_counter()
    stop_reason = None
    for step in range(config.steps):
        for index, (image, target) in enumerate(zip(support, targets)):
            count = len(target["labels"])
            if not count:
                log.append({"step": step, "support_index": index, "selected": 0, "updated": False,
                            "reason": "no_confident_pseudo_labels"})
                continue
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            optimizer.zero_grad(set_to_none=True)
            # Slight additional illumination variation shares exact geometry
            # with the teacher input; no clean counterpart or labels are used.
            losses = model([shifted(image, 0.9)], [target])
            if not isinstance(losses, dict) or not losses:
                raise ValueError("training detector must return a nonempty loss mapping")
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                stop_reason = "non_finite_loss"
                break
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainables], 1.0)
            if not torch.isfinite(grad_norm):
                stop_reason = "non_finite_gradient"
                break
            optimizer.step()
            with torch.no_grad():
                delta = sum((parameter - original).float().square().sum()
                            for (_, parameter), original in zip(trainables, initial_parameters)).sqrt()
                finite = all(bool(torch.isfinite(parameter).all()) for _, parameter in trainables)
            log.append({"step": step, "support_index": index, "selected": count, "updated": True,
                        "loss": float(loss.detach()), "gradient_norm": float(grad_norm),
                        "update_norm": float(delta) if torch.isfinite(delta) else None})
            if not finite or not torch.isfinite(delta) or float(delta) > 1.0:
                stop_reason = "adapter_update_guard"
                break
        if stop_reason:
            break
    model.eval()
    _sync(config.device)
    update_seconds = time.perf_counter() - update_started
    elapsed = time.perf_counter() - started
    if base_fingerprint(model) != before_base:
        raise RuntimeError("base parameters or buffers changed during adapter-only TTT")
    if stop_reason:
        load_adapter_state_dict(model, initial_adapter)
    with torch.no_grad():
        changed = any(not torch.equal(parameter, original)
                      for (_, parameter), original in zip(trainables, initial_parameters))
    return {
        "status": "rolled_back" if stop_reason else ("updated" if changed else "no_update"),
        "stop_reason": stop_reason, "base_unchanged": True, "base_sha256": before_base,
        "adapter_changed": changed, "optimizer_steps": sum(item["updated"] for item in log),
        "seconds": elapsed, "teacher_seconds": teacher_seconds, "update_seconds": update_seconds,
        "pseudo_label_counts": [len(target["labels"]) for target in targets],
        "teacher": "same_arm_before_adaptation_fixed_pseudo_labels", "steps": log,
    }


def predictions(model: Any, paths: list[Path], config: ExperimentConfig,
                mapping: dict[int, int], *, brightness: float) -> tuple[list[dict[str, Any]], dict[str, float]]:
    import torch

    entries = []
    times = []
    model.eval()
    for path in paths:
        image = shifted(load_image(path, config.device), brightness)
        height, width = image.shape[-2:]
        _sync(config.device)
        started = time.perf_counter()
        with torch.no_grad():
            output = model([image])[0]
        _sync(config.device)
        times.append(time.perf_counter() - started)
        detections = []
        for box, score, label in zip(output["boxes"].detach().cpu().tolist(),
                                     output["scores"].detach().cpu().tolist(),
                                     output["labels"].detach().cpu().tolist()):
            if label not in mapping:
                continue
            if not all(math.isfinite(value) for value in (*box, score)):
                raise ValueError("non-finite detector output")
            x1, y1, x2, y2 = box
            detections.append({"class_id": mapping[label], "score": score, "bbox": {
                "cx": (x1 + x2) / (2 * width), "cy": (y1 + y2) / (2 * height),
                "w": (x2 - x1) / width, "h": (y2 - y1) / height,
            }})
        entries.append({"image": str(path), "detections": detections})
    ordered = sorted(times)
    return entries, {"mean_seconds": statistics.mean(times),
                     "p95_seconds": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
                     "sample_count": len(times), "warmup_excluded": False}


def evaluation_records(paths: list[Path], config: ExperimentConfig) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        labels_path = Path(config.dataset) / "labels" / config.split / f"{path.stem}.txt"
        labels = []
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"expected five YOLO bbox fields in {labels_path}")
            values = list(map(float, fields))
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"non-finite YOLO label in {labels_path}")
            class_id, cx, cy, width, height = values
            if class_id != int(class_id) or not 0 <= class_id < 80:
                raise ValueError("this runner requires conventional zero-based COCO80 labels")
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError(f"invalid normalized YOLO bbox in {labels_path}")
            labels.append({"class_id": int(class_id), "cx": cx, "cy": cy, "w": width, "h": height})
        records.append({"image": str(path), "labels": labels})
    return records


def evaluate(entries: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    from yolozu.eval.coco_eval import build_coco_ground_truth, evaluate_coco_map, predictions_to_coco_detections

    gt, index = build_coco_ground_truth(records)
    if not gt["annotations"]:
        raise ValueError("evaluation subset has no ground-truth boxes")
    dt = predictions_to_coco_detections(entries, coco_index=index,
             image_sizes={item["id"]: (item["width"], item["height"]) for item in gt["images"]})
    return {"metric_semantics": "pycocotools_coco_bbox", **evaluate_coco_map(gt, dt, quiet=True)}


def _rss() -> int | None:
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if platform.system() == "Darwin" else value * 1024)
    except (ImportError, AttributeError):
        return None


def _overlay(path: Path, entries: dict[str, Any], destination: Path, brightness: float) -> None:
    from PIL import Image, ImageDraw, ImageEnhance

    with Image.open(path) as original:
        image = ImageEnhance.Brightness(original.convert("RGB")).enhance(brightness)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for detection in entries["detections"]:
        if detection["score"] < 0.5:
            continue
        box = detection["bbox"]
        bounds = ((box["cx"] - box["w"] / 2) * width, (box["cy"] - box["h"] / 2) * height,
                  (box["cx"] + box["w"] / 2) * width, (box["cy"] + box["h"] / 2) * height)
        draw.rectangle(bounds, outline="lime", width=2)
        draw.text(bounds[:2], f"{detection['class_id']} {detection['score']:.2f}", fill="lime")
    image.thumbnail((1000, 1000))
    image.save(destination)


def run_experiment(config: ExperimentConfig, *,
                   model_factory: Callable[[ExperimentConfig], Any] | None = None,
                   category_mapping: dict[int, int] | None = None) -> dict[str, Any]:
    """Execute four arms. Injection points are for tests, not arbitrary CLI imports."""
    config.validate()
    destination = Path(config.output)
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    support_paths, eval_paths = discover_images(config)
    destination.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": 1, "kind": "cnn_quantized_lora_ttt_experiment", "status": "running",
        "maturity": "Research", "promotion_eligible": False, "efficacy_conclusion": "not_established",
        "config": asdict(config), "model_id": "torchvision_maskrcnn_resnet50_fpn_v2_bbox",
        "checkpoint_sha256": sha256_file(Path(config.checkpoint)),
        "dataset_label_convention": "coco80_zero_based", "adaptation_uses_ground_truth": False,
        "shift": {"kind": "brightness_multiplier", "value": config.brightness, "synthetic": True},
        "split": {"support": [{"image": str(path), "sha256": sha256_file(path)} for path in support_paths],
                  "evaluation": [{"image": str(path), "sha256": sha256_file(path)} for path in eval_paths],
                  "disjoint_paths_and_bytes": True},
        "arms": {},
    }
    source_paths = [Path(__file__), Path(__file__).with_name("quantized_lora.py")]
    tool_path = Path(__file__).resolve().parents[2] / "tools" / "run_cnn_qlora_ttt.py"
    if tool_path.is_file():
        source_paths.append(tool_path)
    report["source_sha256"] = {path.name: sha256_file(path) for path in source_paths}
    _write_json(destination / "report.json", report)
    try:
        import torch
        from yolozu.tta.quantized_lora import (
            adapter_state_dict, base_fingerprint, load_adapter_state_dict, prepare_model,
        )

        if config.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        torch.set_num_threads(config.threads)
        mapping = coco_category_mapping() if category_mapping is None else category_mapping
        report["runtime"] = {"torch": torch.__version__, "python": platform.python_version(),
                             "torchvision": importlib.metadata.version("torchvision"),
                             "device": config.device, "compute_dtype": "float32",
                             "quantized_arithmetic": False,
                             "implementation": "packed_int4_dequantize_float_reference"}
        # Evaluation labels are deliberately read only after all adaptation has
        # completed. Predictions and adapter artifacts can be inspected first.
        for arm_name, quantized, training in (
            ("float_fixed", False, False), ("int4_fixed", True, False),
            ("float_lora_ttt", False, True), ("int4_lora_ttt", True, True),
        ):
            torch.manual_seed(config.seed)
            if config.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            model = (model_factory or _build_detector)(config)
            preparation = prepare_model(model, quantize=quantized, rank=config.rank if training else 0,
                                        adapter_include=config.adapter_include, quant_include=config.quant_include)
            before_base = base_fingerprint(model)
            arm_dir = destination / arm_name
            arm_dir.mkdir()
            arm: dict[str, Any] = {"status": "running", "quantized": quantized, "ttt": training,
                                  "preparation": preparation, "compute_dtype": "float32"}
            report["arms"][arm_name] = arm
            _write_json(destination / "report.json", report)
            if training:
                support = [shifted(load_image(path, config.device), config.brightness) for path in support_paths]
                arm["adaptation"] = adapt(model, support, config)
                del support
                payload = adapter_state_dict(model)
                adapter_path = arm_dir / "adapter.pt"
                torch.save(payload, adapter_path)
                arm["adapter_file_bytes"] = adapter_path.stat().st_size
                arm["adapter_sha256"] = sha256_file(adapter_path)
                # Clear adapters, then reload from disk before evaluation. Output
                # parity checks the saved bytes rather than an in-memory clone.
                probe = shifted(load_image(eval_paths[0], config.device), config.brightness)
                model.eval()
                with torch.no_grad():
                    expected = {key: value.detach().cpu().clone() for key, value in model([probe])[0].items()
                                if key in ("boxes", "scores", "labels")}
                    for name, parameter in model.named_parameters():
                        if name.endswith(("lora_A", "lora_B")):
                            parameter.zero_()
                load_adapter_state_dict(model, torch.load(adapter_path, map_location="cpu", weights_only=True))
                with torch.no_grad():
                    observed = model([probe])[0]
                parity = all(torch.equal(expected[key], observed[key].detach().cpu()) for key in expected)
                if not parity:
                    raise RuntimeError("saved adapter reload changed detector predictions")
                arm["adapter_reload_prediction_parity"] = True
                del probe, expected, observed, payload
            else:
                arm["adaptation"] = {"status": "disabled", "optimizer_steps": 0, "seconds": 0.0}
            for domain, brightness in (("clean", 1.0), ("shifted", config.brightness)):
                entries, timing = predictions(model, eval_paths, config, mapping, brightness=brightness)
                _write_json(arm_dir / f"{domain}_predictions.json", {"predictions": entries, "meta": {
                    "arm": arm_name, "domain": domain, "evidence_kind": "local_diagnostic",
                    "checkpoint_sha256": report["checkpoint_sha256"], "ttt": training,
                }})
                arm[domain] = {"latency": timing, "detections": sum(len(entry["detections"]) for entry in entries)}
                if domain == "shifted":
                    _overlay(eval_paths[0], entries[0], arm_dir / "shifted_overlay.png", brightness)
            arm["base_unchanged"] = before_base == base_fingerprint(model)
            if not arm["base_unchanged"]:
                raise RuntimeError("base changed during experiment")
            _sync(config.device)
            arm["total_seconds"] = time.perf_counter() - started
            arm["memory"] = {
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if config.device == "cuda" else None,
                "process_lifetime_peak_rss_bytes": _rss(),
                "rss_semantics": "process_lifetime_high_watermark_not_per_arm_or_model_storage",
            }
            arm["status"] = "complete"
            del model
            gc.collect()
            _write_json(destination / "report.json", report)
        if config.skip_eval:
            report["evaluation"] = {"status": "skipped", "metrics": None}
        else:
            records = evaluation_records(eval_paths, config)
            report["evaluation_label_sha256"] = {
                path.stem + ".txt": sha256_file(Path(config.dataset) / "labels" / config.split / (path.stem + ".txt"))
                for path in eval_paths
            }
            for arm_name, arm in report["arms"].items():
                for domain in ("clean", "shifted"):
                    entries = json.loads((destination / arm_name / f"{domain}_predictions.json").read_text())["predictions"]
                    arm[domain]["evaluation"] = evaluate(entries, records)
            report["evaluation"] = {"status": "complete", "metric_semantics": "pycocotools_coco_bbox"}
            report["deltas"] = {}
            for adapted, baseline in (("float_lora_ttt", "float_fixed"), ("int4_lora_ttt", "int4_fixed")):
                report["deltas"][adapted] = {domain: (
                    report["arms"][adapted][domain]["evaluation"]["metrics"]["map50_95"]
                    - report["arms"][baseline][domain]["evaluation"]["metrics"]["map50_95"]
                ) for domain in ("clean", "shifted")}
        report["status"] = "complete"
        _write_json(destination / "report.json", report)
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(destination / "report.json", report)
        raise
