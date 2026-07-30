#!/usr/bin/env python3
"""Run a bounded real Transformers DETR optimizer step on a YOLO dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a scratch-initialized tiny Hugging Face Transformers DETR "
            "model on real YOLO images and emit checkpoint/prediction evidence."
        )
    )
    parser.add_argument("--model-id", default="local-tiny-detr", help="Recorded model handle; no remote weights are fetched.")
    parser.add_argument("--dataset", required=True, help="YOLO dataset root.")
    parser.add_argument("--split", default="train", help="Dataset split (default: train).")
    parser.add_argument("--epochs", type=int, default=1, help="Epoch budget recorded in evidence.")
    parser.add_argument("--batch-size", type=int, default=2, help="Images per optimizer step (default: 2).")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW learning rate (default: 1e-4).")
    parser.add_argument("--max-steps", type=int, default=1, help="Maximum optimizer steps (default: 1).")
    parser.add_argument("--resume-from", default=None, help="Optional state-dict checkpoint.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Evidence output directory (default: $YOLOZU_HF_OUTPUT or /tmp/yolozu-hf-detr-runtime).",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and not item.name.endswith(".cache")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolve_split(root: Path, split: str) -> tuple[Path, Path]:
    candidates = [split]
    if split == "train":
        candidates.append("train2017")
    elif split == "val":
        candidates.append("val2017")
    for candidate in candidates:
        images = root / "images" / candidate
        labels = root / "labels" / candidate
        if images.is_dir() and labels.is_dir():
            return images, labels
    raise SystemExit(f"split not found under {root}: {split}")


def _load_batch(images_dir: Path, labels_dir: Path, *, batch_size: int, size: int) -> tuple[Any, list[dict[str, Any]], list[Path]]:
    import torch
    from PIL import Image

    image_paths = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:batch_size]
    if not image_paths:
        raise SystemExit(f"no images found: {images_dir}")

    pixels = []
    labels = []
    used = []
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        image = Image.open(image_path).convert("RGB").resize((size, size))
        tensor = torch.from_numpy(__import__("numpy").asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        class_labels = []
        boxes = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            class_labels.append(int(float(fields[0])))
            boxes.append([float(value) for value in fields[1:5]])
        if not boxes:
            continue
        pixels.append(tensor)
        labels.append(
            {
                "class_labels": torch.tensor(class_labels, dtype=torch.int64),
                "boxes": torch.tensor(boxes, dtype=torch.float32),
            }
        )
        used.append(image_path)
    if not pixels:
        raise SystemExit("no labeled images were available for HF DETR training")
    return torch.stack(pixels), labels, used


def main() -> int:
    args = _parse_args()
    if args.max_steps < 1 or args.batch_size < 1 or args.epochs < 1:
        raise SystemExit("--max-steps, --batch-size, and --epochs must be >= 1")

    import os
    import torch
    from transformers import DetrConfig, DetrForObjectDetection, ResNetConfig

    dataset = Path(args.dataset).resolve()
    images_dir, labels_dir = _resolve_split(dataset, str(args.split))
    output_dir = Path(
        args.output_dir
        or os.environ.get("YOLOZU_HF_OUTPUT")
        or "/tmp/yolozu-hf-detr-runtime"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pixel_values, labels, image_paths = _load_batch(
        images_dir,
        labels_dir,
        batch_size=int(args.batch_size),
        size=64,
    )
    num_labels = max(int(value) for item in labels for value in item["class_labels"]) + 1
    backbone = ResNetConfig(
        num_channels=3,
        embedding_size=16,
        hidden_sizes=[16, 32, 64, 128],
        depths=[1, 1, 1, 1],
        layer_type="basic",
        out_features=["stage4"],
    )
    config = DetrConfig(
        backbone_config=backbone,
        num_queries=10,
        encoder_layers=1,
        decoder_layers=1,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        d_model=32,
        num_labels=num_labels,
    )
    config_path = output_dir / "model_config.json"
    config_path.write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.manual_seed(11)
    model = DetrForObjectDetection(config)
    if args.resume_from:
        resume = Path(args.resume_from).resolve()
        model.load_state_dict(torch.load(resume, map_location="cpu", weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate))

    started = time.perf_counter()
    losses: list[float] = []
    model.train()
    for _ in range(int(args.max_steps)):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        if loss is None or not bool(torch.isfinite(loss)):
            raise RuntimeError("HF DETR returned a non-finite or missing loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    checkpoint = output_dir / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)

    model.eval()
    predictions = []
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        probabilities = outputs.logits.softmax(-1)[..., :-1]
        scores, classes = probabilities.max(-1)
        for batch_index, image_path in enumerate(image_paths):
            detections = []
            for query_index in scores[batch_index].topk(min(3, scores.shape[1])).indices.tolist():
                cx, cy, width, height = [
                    float(value) for value in outputs.pred_boxes[batch_index, query_index]
                ]
                detections.append(
                    {
                        "class_id": int(classes[batch_index, query_index]),
                        "score": float(scores[batch_index, query_index]),
                        "bbox": {"cx": cx, "cy": cy, "w": width, "h": height},
                    }
                )
            predictions.append(
                {
                    "schema_version": 1,
                    "image": str(image_path),
                    "image_size": [64, 64],
                    "detections": detections,
                }
            )
    predictions_path = output_dir / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            {
                "meta": {
                    "adapter": "hf_detr_runtime_smoke",
                    "bbox_format": "cxcywh_norm",
                    "model_initialization": "scratch_tiny_detr_no_remote_weights",
                },
                "predictions": predictions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    evidence = {
        "schema_version": 1,
        "kind": "hf_detr_external_runtime_training",
        "training_executed": True,
        "optimizer_steps": int(args.max_steps),
        "epochs_budget": int(args.epochs),
        "losses": losses,
        "model_id_argument": str(args.model_id),
        "model_initialization": "scratch_tiny_detr_no_remote_weights",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "dataset": str(dataset),
        "dataset_tree_sha256": _tree_sha256(dataset),
        "images": [str(path) for path in image_paths],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "predictions": str(predictions_path),
        "predictions_sha256": _sha256(predictions_path),
        "predictions_interface_contract": "yolozu_predictions_v1",
        "wall_seconds": float(time.perf_counter() - started),
        "peak_rss_bytes": peak_rss,
        "environment": {
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "transformers_license": importlib.metadata.metadata("transformers").get("License"),
            "license_boundary": "Apache-2.0 external runtime; not bundled with YOLOZU",
        },
    }
    evidence_path = output_dir / "training_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(evidence_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
