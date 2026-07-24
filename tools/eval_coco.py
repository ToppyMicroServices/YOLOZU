import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.api import APIError, _failure_report, _write_json_atomic, evaluate_coco
from yolozu.eval_protocol import apply_eval_protocol_args, load_eval_protocol
from yolozu.core.diagnostics import format_cli_error


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        choices=("yolo26", "nms_applied", "e2e_nms_free"),
        default=None,
        help="Apply canonical evaluation protocol presets (pins split/bbox_format).",
    )
    parser.add_argument("-d", "--dataset", default="data/coco128", help="YOLO-format dataset root (images/ + labels/).")
    parser.add_argument(
        "-s",
        "--split",
        default=None,
        help="Dataset split under images/ and labels/ (e.g. val2017, train2017). Default: auto (val2017 if present else train2017).",
    )
    parser.add_argument("-p", "--predictions", required=True, help="Predictions JSON path (YOLOZU format).")
    parser.add_argument(
        "--bbox-format",
        choices=("cxcywh_norm", "cxcywh_abs", "xywh_abs", "xyxy_abs"),
        default="cxcywh_norm",
        help="How to interpret detection bbox fields (default: cxcywh_norm).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip COCOeval and only validate/convert predictions (no pycocotools required).",
    )
    parser.add_argument(
        "-r",
        "--repair",
        action="store_true",
        help="Explicitly repair/clamp legacy prediction values and record every repair (default: strict rejection).",
    )
    parser.add_argument(
        "-n",
        "--max-images",
        type=int,
        default=None,
        help="Evaluate the first N dataset images; known predictions outside the subset are counted and excluded.",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Optional labels/<split>/classes.json path for category_id↔class_id normalization before eval.",
    )
    parser.add_argument(
        "--assume-class-id-is-category-id",
        action="store_true",
        help="Treat class_id in predictions as COCO category_id when --classes is provided.",
    )
    parser.add_argument("-o", "--output", default="reports/coco_eval.json", help="Where to write evaluation JSON.")
    return parser.parse_args(argv)


def _resolve_args(argv):
    args = _parse_args(argv)
    protocol = load_eval_protocol(args.protocol) if args.protocol else None
    if protocol:
        args = apply_eval_protocol_args(args, protocol)
    return args, protocol


def main(argv=None):
    args, protocol = _resolve_args(sys.argv[1:] if argv is None else argv)

    dataset_root = repo_root / args.dataset
    output_path = repo_root / args.output
    predictions_path = repo_root / args.predictions
    classes_path = (repo_root / args.classes) if args.classes else None
    try:
        result = evaluate_coco(
            dataset_root,
            predictions_path,
            split=args.split,
            bbox_format=args.bbox_format,
            max_images=args.max_images,
            dry_run=bool(args.dry_run),
            repair=bool(args.repair),
            classes=classes_path,
            assume_class_id_is_category_id=bool(args.assume_class_id_is_category_id),
        )
    except APIError as exc:
        failure = _failure_report(
            exc,
            dataset=str(args.dataset),
            predictions=str(args.predictions),
            split=args.split,
            bbox_format=args.bbox_format,
            max_images=args.max_images,
            dry_run=bool(args.dry_run),
            repair=bool(args.repair),
        )
        failure["protocol_id"] = args.protocol
        failure["protocol"] = protocol
        failure["normalization"] = {
            "classes": str(args.classes) if args.classes else None,
            "assume_class_id_is_category_id": bool(args.assume_class_id_is_category_id),
        }
        _write_json_atomic(output_path, failure)
        raise SystemExit(
            format_cli_error(
                code=exc.code,
                message=exc.message,
                hint=f"failure report written to {output_path}",
            )
        ) from exc

    payload = result.to_dict()
    payload["protocol_id"] = args.protocol
    payload["protocol"] = protocol
    payload["dataset"] = str(args.dataset)
    payload["predictions"] = str(args.predictions)
    payload["normalization"] = {
        "classes": str(args.classes) if args.classes else None,
        "assume_class_id_is_category_id": bool(args.assume_class_id_is_category_id),
    }
    _write_json_atomic(output_path, payload)
    print(output_path)


if __name__ == "__main__":
    main()
