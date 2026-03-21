import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from tools import check_predictions_parity
from tools import export_predictions_trt


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="Reference predictions JSON (PyTorch).")
    p.add_argument("--engine", required=True, help="TensorRT engine plan path.")
    p.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    p.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    p.add_argument("--output", default="reports/predictions_trt.json", help="Where to write TRT predictions JSON.")
    p.add_argument("--input-name", default="images", help="TensorRT input binding name (default: images).")
    p.add_argument("--boxes-output", default="boxes", help="Output name for boxes tensor (default: boxes).")
    p.add_argument("--scores-output", default="scores", help="Output name for scores tensor (default: scores).")
    p.add_argument("--class-output", default=None, help="Optional output name for class_id tensor.")
    p.add_argument(
        "--combined-output",
        default=None,
        help="Optional single output name with (N,6) or (1,N,6) entries [x1,y1,x2,y2,score,class_id].",
    )
    p.add_argument(
        "--combined-format",
        choices=("xyxy_score_class",),
        default="xyxy_score_class",
        help="Layout for --combined-output (default: xyxy_score_class).",
    )
    p.add_argument(
        "--raw-output",
        default=None,
        help="Optional single output name with raw head output (e.g., 1x84x8400) to decode + NMS.",
    )
    p.add_argument(
        "--raw-format",
        choices=("yolo_84",),
        default="yolo_84",
        help="Layout for --raw-output (default: yolo_84).",
    )
    p.add_argument(
        "--raw-postprocess",
        choices=("native", "ultralytics", "yolo_runtime"),
        default="native",
        help="Postprocess for --raw-output (default: native).",
    )
    p.add_argument(
        "--boxes-format",
        choices=("xyxy",),
        default="xyxy",
        help="Box layout produced by the model in input-image space (default: xyxy).",
    )
    p.add_argument(
        "--boxes-scale",
        choices=("abs", "norm"),
        default="norm",
        help="Whether boxes are in pixels (abs) or normalized [0,1] (norm) wrt input_size (default: norm).",
    )
    p.add_argument("--min-score", type=float, default=0.001, help="Score threshold (no NMS).")
    p.add_argument("--topk", type=int, default=300, help="Keep top-K detections per image (no NMS).")
    p.add_argument("--nms-iou", type=float, default=0.7, help="IoU threshold for NMS when decoding raw output.")
    p.add_argument("--agnostic-nms", action="store_true", help="Use class-agnostic NMS when decoding raw output.")
    p.add_argument("--wrap", action="store_true", help="Wrap as {predictions:[...], meta:{...}}.")
    p.add_argument("--dry-run", action="store_true", help="Write schema-correct JSON without running inference.")

    p.add_argument("--image-size", default=None, help="Optional fixed image size for parity check (e.g., 640).")
    p.add_argument("--iou-thresh", type=float, default=0.99, help="IoU threshold to consider a match.")
    p.add_argument("--score-atol", type=float, default=1e-4, help="Absolute tolerance for score differences.")
    p.add_argument("--bbox-atol", type=float, default=1e-4, help="Absolute tolerance for bbox cx/cy/w/h differences.")
    p.add_argument("--report", default=None, help="Optional path to write parity report JSON (from stdout).")
    return p.parse_args(argv)


def _append_arg(args: list[str], flag: str, value: object | None):
    if value is None:
        return
    args.extend([flag, str(value)])


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    trt_args: list[str] = [
        "--dataset",
        args.dataset,
        "--engine",
        args.engine,
        "--input-name",
        args.input_name,
        "--boxes-output",
        args.boxes_output,
        "--scores-output",
        args.scores_output,
        "--boxes-format",
        args.boxes_format,
        "--boxes-scale",
        args.boxes_scale,
        "--min-score",
        str(args.min_score),
        "--topk",
        str(args.topk),
        "--nms-iou",
        str(args.nms_iou),
        "--output",
        args.output,
    ]
    _append_arg(trt_args, "--split", args.split)
    _append_arg(trt_args, "--max-images", args.max_images)
    _append_arg(trt_args, "--class-output", args.class_output)
    _append_arg(trt_args, "--combined-output", args.combined_output)
    _append_arg(trt_args, "--combined-format", args.combined_format)
    _append_arg(trt_args, "--raw-output", args.raw_output)
    _append_arg(trt_args, "--raw-format", args.raw_format)
    _append_arg(trt_args, "--raw-postprocess", args.raw_postprocess)
    if args.agnostic_nms:
        trt_args.append("--agnostic-nms")
    if args.wrap:
        trt_args.append("--wrap")
    if args.dry_run:
        trt_args.append("--dry-run")

    export_predictions_trt.main(trt_args)

    parity_args: list[str] = [
        "--reference",
        args.reference,
        "--candidate",
        args.output,
        "--iou-thresh",
        str(args.iou_thresh),
        "--score-atol",
        str(args.score_atol),
        "--bbox-atol",
        str(args.bbox_atol),
    ]
    _append_arg(parity_args, "--image-size", args.image_size)
    _append_arg(parity_args, "--max-images", args.max_images)

    try:
        check_predictions_parity.main(parity_args)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1

    if args.report:
        report_path = Path(args.report)
        if not report_path.is_absolute():
            report_path = repo_root / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("Use stdout output from check_predictions_parity.py.\n")
        print(report_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
