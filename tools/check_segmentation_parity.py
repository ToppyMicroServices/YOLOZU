#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.eval.segmentation_parity import compare_segmentation_predictions  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare two semantic-segmentation prediction artifacts for parity.")
    p.add_argument("--reference", required=True, help="Reference segmentation predictions JSON.")
    p.add_argument("--candidate", required=True, help="Candidate segmentation predictions JSON.")
    p.add_argument("--mismatch-atol", type=float, default=0.0, help="Maximum mismatch-rate tolerance per sample (default: 0.0).")
    p.add_argument("--max-samples", type=int, default=None, help="Optional cap for number of samples to compare.")
    p.add_argument("--output", default=None, help="Optional output JSON path; stdout is used when omitted.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    report = compare_segmentation_predictions(
        reference=args.reference,
        candidate=args.candidate,
        mismatch_atol=float(args.mismatch_atol),
        max_samples=(int(args.max_samples) if args.max_samples is not None else None),
    )
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        out_path = Path(str(args.output)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(str(out_path))
    else:
        sys.stdout.write(payload)
    return 0 if bool(report.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
