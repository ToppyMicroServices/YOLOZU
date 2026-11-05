"""Run a bounded, network-free four-arm CNN quantized-LoRA TTT comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.tta.cnn_experiment import ExperimentConfig, run_experiment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="YOLO dataset root; labels must use COCO80 zero-based IDs.")
    parser.add_argument("--checkpoint", required=True, help="Local official Torchvision Mask R-CNN ResNet50 FPN v2 state_dict; never downloaded.")
    parser.add_argument("--output", required=True, help="New output directory; existing paths are rejected.")
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--support-images", type=int, default=2)
    parser.add_argument("--eval-images", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1, help="TTT passes over support images (1 to 10).")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--pseudo-threshold", type=float, default=0.7)
    parser.add_argument("--pseudo-topk", type=int, default=20)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=384)
    parser.add_argument("--brightness", type=float, default=0.55, help="Synthetic illumination multiplier for support and shifted evaluation.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Reference path computes in FP32 on both devices.")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--adapter-include", default="backbone.body.layer4", help="Comma-separated module prefixes; '*' selects all supported layers.")
    parser.add_argument("--quant-include", default="*", help="Comma-separated module prefixes to pack into INT4.")
    parser.add_argument("--skip-eval", action="store_true", help="Explicitly skip COCO metrics; output is not accuracy evidence.")
    args = parser.parse_args(argv)
    values = vars(args)
    for key in ("adapter_include", "quant_include"):
        values[key] = tuple(item.strip() for item in values[key].split(",") if item.strip())
    try:
        report = run_experiment(ExperimentConfig(**values))
    except Exception as exc:
        print(f"CNN QLoRA TTT failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "report": str(Path(args.output) / "report.json"),
                      "efficacy_conclusion": report["efficacy_conclusion"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
