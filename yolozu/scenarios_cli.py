"""CLI entry point for running a YOLOZU scenario pass.

Builds an adapter + dataset manifest from argparse arguments and
executes the inference / evaluation pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["main"]

from yolozu.adapter import DummyAdapter, PrecomputedAdapter, RTDETRPoseAdapter
from yolozu.dataset import build_manifest
from yolozu.runner import run_adapter


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a YOLOZU scenario pass (adapter + dataset manifest).")
    parser.add_argument(
        "--adapter",
        choices=("dummy", "precomputed", "rtdetr_pose"),
        default="dummy",
        help="Which adapter to run (default: dummy).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="YOLO-format dataset root (recommended; defaults to ./data/coco128 when present).",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Predictions JSON path for --adapter precomputed.",
    )
    parser.add_argument(
        "--config",
        default="builtin:base",
        help="Config path for rtdetr_pose adapter (default: builtin:base).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for rtdetr_pose adapter (default: cpu).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=None,
        help="Image size for rtdetr_pose adapter (one value or two values).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Score threshold for rtdetr_pose adapter (default: 0.3).",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=50,
        help="Max detections per image for rtdetr_pose adapter (default: 50).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint for rtdetr_pose adapter.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for number of images (for quick smoke runs).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split under images/ and labels/ (e.g. val2017, train2017). Default: auto.",
    )
    parser.add_argument(
        "--output",
        default="reports/scenario_run.json",
        help="Where to write the scenario JSON report (default: reports/scenario_run.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    cwd = Path.cwd()
    dataset_root = Path(args.dataset) if args.dataset else (cwd / "data" / "coco128")
    if not dataset_root.exists():
        raise SystemExit(f"dataset root not found: {dataset_root} (set --dataset)")

    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest.get("images") or []
    if args.max_images is not None:
        records = list(records)[: int(args.max_images)]

    if args.adapter == "dummy":
        adapter = DummyAdapter()
    elif args.adapter == "precomputed":
        if not args.predictions:
            raise SystemExit("--predictions is required for --adapter precomputed")
        adapter = PrecomputedAdapter(predictions_path=args.predictions)
    else:
        image_size = None
        if args.image_size:
            if len(args.image_size) == 1:
                image_size = (int(args.image_size[0]), int(args.image_size[0]))
            elif len(args.image_size) == 2:
                image_size = (int(args.image_size[0]), int(args.image_size[1]))
            else:
                raise SystemExit("--image-size expects 1 or 2 integers")
        adapter = RTDETRPoseAdapter(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device,
            image_size=image_size or (320, 320),
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
        )

    result = run_adapter(adapter, records)

    output_path = Path(str(args.output))
    if not output_path.is_absolute():
        output_path = cwd / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

