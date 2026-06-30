import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.cli_args import parse_image_size_arg, require_non_negative_int, resolve_input_path
from yolozu.predictions.bbox_formats import SUPPORTED_PREDICTION_BBOX_FORMATS
from yolozu.predictions.predictions_parity import compare_predictions


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="Reference predictions JSON (e.g. PyTorch).")
    p.add_argument("--candidate", required=True, help="Candidate predictions JSON (e.g. ONNXRuntime).")
    p.add_argument(
        "--bbox-format",
        choices=SUPPORTED_PREDICTION_BBOX_FORMATS,
        default="auto",
        help="BBox format stored in both JSONs, or auto to use bbox.format/keys (default: auto).",
    )
    p.add_argument("--iou-thresh", type=float, default=0.99, help="IoU threshold to consider a match.")
    p.add_argument("--score-atol", type=float, default=1e-4, help="Absolute tolerance for score differences.")
    p.add_argument("--bbox-atol", type=float, default=1e-4, help="Absolute tolerance for bbox cx/cy/w/h differences.")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")
    p.add_argument(
        "--image-size",
        default=None,
        help="Optional fixed image size (e.g. 640 or 640,640) to avoid reading image files.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        image_size = parse_image_size_arg(args.image_size, flag_name="--image-size")
        max_images = require_non_negative_int(args.max_images, flag_name="--max-images")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    ref_path = resolve_input_path(args.reference, cwd=Path.cwd(), repo_root=repo_root)
    cand_path = resolve_input_path(args.candidate, cwd=Path.cwd(), repo_root=repo_root)

    report = compare_predictions(
        reference=ref_path,
        candidate=cand_path,
        image_size=image_size,
        max_images=max_images,
        iou_thresh=float(args.iou_thresh),
        score_atol=float(args.score_atol),
        bbox_atol=float(args.bbox_atol),
        bbox_format=str(args.bbox_format),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not bool(report.get("ok")):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
