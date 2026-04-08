#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate one predicted depth map against one ground-truth depth map."
    )
    p.add_argument("--pred-depth", required=True, help="Predicted depth map (.npy/.npz/image).")
    p.add_argument("--gt-depth", required=True, help="Ground-truth depth map (.npy/.npz/image).")
    p.add_argument("--mask", default=None, help="Optional valid-pixel mask (.npy/.npz/image; >0 means valid).")
    p.add_argument(
        "--align",
        choices=("none", "median_scale"),
        default="median_scale",
        help="Prediction alignment method before scoring (default: median_scale).",
    )
    p.add_argument("--pred-scale", type=float, default=1.0, help="Scale applied to predicted depth before alignment.")
    p.add_argument("--gt-scale", type=float, default=1.0, help="Scale applied to ground-truth depth before scoring.")
    p.add_argument("--min-depth", type=float, default=1e-6, help="Ignore GT pixels below this threshold.")
    p.add_argument("--max-depth", type=float, default=None, help="Optional upper GT depth threshold.")
    p.add_argument("--output", default="reports/depth_eval.json", help="Output report JSON path.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    from yolozu.eval.depth_eval import evaluate_depth_arrays, load_depth_array, load_mask_array  # noqa: E402

    pred_path = Path(args.pred_depth)
    gt_path = Path(args.gt_depth)
    mask_path = Path(args.mask) if args.mask else None

    if not pred_path.is_absolute():
        pred_path = (Path.cwd() / pred_path).resolve()
    if not gt_path.is_absolute():
        gt_path = (Path.cwd() / gt_path).resolve()
    if mask_path is not None and not mask_path.is_absolute():
        mask_path = (Path.cwd() / mask_path).resolve()

    report = evaluate_depth_arrays(
        pred=load_depth_array(pred_path),
        gt=load_depth_array(gt_path),
        mask=(load_mask_array(mask_path) if mask_path is not None else None),
        align=args.align,
        pred_scale=float(args.pred_scale),
        gt_scale=float(args.gt_scale),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    report["meta"] = {
        "timestamp_utc": _now_utc(),
        "pred_depth": str(pred_path),
        "gt_depth": str(gt_path),
        "mask": (str(mask_path) if mask_path is not None else None),
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
