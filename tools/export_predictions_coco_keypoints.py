#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.datasets.migrate import migrate_coco_keypoints_results_predictions  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert COCO-style keypoints results JSON into the YOLOZU predictions interface contract."
    )
    p.add_argument("--results-json", required=True, help="COCO-style keypoints results JSON (list[dict]).")
    p.add_argument("--instances-json", required=True, help="COCO instances JSON that defines images/categories.")
    p.add_argument("--output", required=True, help="Predictions JSON output path.")
    p.add_argument("--score-threshold", type=float, default=0.0, help="Minimum detection score to keep (default: 0.0).")
    p.add_argument("--force", action="store_true", help="Overwrite --output if it already exists.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    out = migrate_coco_keypoints_results_predictions(
        results_json=args.results_json,
        instances_json=args.instances_json,
        output=args.output,
        score_threshold=float(args.score_threshold),
        force=bool(args.force),
    )
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
