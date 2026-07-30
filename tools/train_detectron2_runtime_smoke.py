#!/usr/bin/env python3
"""Run bounded Detectron2 DefaultTrainer training on a COCO wrapper dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one bounded Detectron2 training job on the COCO wrapper "
            "prepared by YOLOZU and emit checkpoint/resource evidence."
        )
    )
    parser.add_argument("--config-file", required=True, help="Detectron2 YAML config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Explicit Detectron2 output directory; overrides config/opts OUTPUT_DIR.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume through Detectron2 DefaultTrainer.")
    parser.add_argument("--eval-only", action="store_true", help="Reserved Detectron2 launcher compatibility flag.")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--dist-url", default="auto")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Detectron2 KEY VALUE config overrides.")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_descriptor(root: Path) -> dict:
    descriptor = root / "dataset.json"
    if not descriptor.is_file():
        raise SystemExit(f"COCO wrapper dataset.json not found: {descriptor}")
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    if payload.get("format") != "coco_instances":
        raise SystemExit(f"Detectron2 smoke requires a coco_instances wrapper: {descriptor}")
    return payload


def main() -> int:
    args = _parse_args()

    from detectron2.config import get_cfg
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer, default_setup

    dataset_root = Path(os.environ["YOLOZU_DATASET_ROOT"]).resolve()
    descriptor = _dataset_descriptor(dataset_root)
    config_path = Path(args.config_file).resolve()
    dataset_name = "yolozu_external_runtime_smoke"
    register_coco_instances(
        dataset_name,
        {},
        str(Path(descriptor["instances_json"]).resolve()),
        str(Path(descriptor["images_dir"]).resolve()),
    )

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.output_dir:
        cfg.OUTPUT_DIR = str(Path(args.output_dir).resolve())
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = ""
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.CHECKPOINT_PERIOD = 1
    cfg.freeze()
    default_setup(cfg, args)

    started = time.perf_counter()
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=bool(args.resume))
    result = trainer.train()

    output_dir = Path(cfg.OUTPUT_DIR).resolve()
    checkpoints = sorted(output_dir.glob("*.pth"))
    checkpoint = checkpoints[-1] if checkpoints else None
    if checkpoint is None:
        raise RuntimeError(f"Detectron2 training emitted no checkpoint under: {output_dir}")
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    evidence = {
        "schema_version": 1,
        "kind": "detectron2_external_runtime_training",
        "training_executed": True,
        "max_iter": int(cfg.SOLVER.MAX_ITER),
        "dataset": str(dataset_root),
        "dataset_descriptor_sha256": _sha256(dataset_root / "dataset.json"),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "instances_json": str(descriptor["instances_json"]),
        "instances_json_sha256": _sha256(Path(descriptor["instances_json"]).resolve()),
        "images_dir": str(descriptor["images_dir"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "wall_seconds": float(time.perf_counter() - started),
        "peak_rss_bytes": peak_rss,
        "train_result": result if isinstance(result, dict) else None,
        "runtime": {
            "python": platform.python_version(),
            "detectron2": importlib.metadata.version("detectron2"),
            "source": "facebookresearch/detectron2",
            "license_boundary": "Apache-2.0 external runtime; not bundled with YOLOZU",
        },
    }
    evidence_path = output_dir / "training_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(evidence_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
