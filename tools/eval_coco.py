import argparse
import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.coco_eval import build_coco_ground_truth, evaluate_coco_map, predictions_to_coco_detections
from yolozu.dataset import build_manifest
from yolozu.eval_protocol import apply_eval_protocol_args, load_eval_protocol
from yolozu.core.diagnostics import format_cli_error
from yolozu.predictions import load_predictions_entries
from yolozu.predictions_transform import load_classes_json, normalize_class_ids


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        choices=("yolo26", "nms_applied", "e2e_nms_free"),
        default=None,
        help="Apply canonical evaluation protocol presets (pins split/bbox_format).",
    )
    parser.add_argument("--dataset", default="data/coco128", help="YOLO-format dataset root (images/ + labels/).")
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split under images/ and labels/ (e.g. val2017, train2017). Default: auto (val2017 if present else train2017).",
    )
    parser.add_argument("--predictions", required=True, help="Predictions JSON path (YOLOZU format).")
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
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for number of images (for quick smoke runs).",
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
    parser.add_argument("--output", default="reports/coco_eval.json", help="Where to write evaluation JSON.")
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
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    split_effective = manifest["split"]
    if args.max_images is not None:
        records = records[: args.max_images]
    if not records:
        raise SystemExit(
            format_cli_error(
                code="E_DATASET_EMPTY",
                message=f"no dataset images resolved for split={split_effective!r} under {dataset_root}",
                hint="check --dataset/--split or remove overly restrictive --max-images",
            )
        )

    gt, index = build_coco_ground_truth(records)
    image_sizes = {img["id"]: (int(img["width"]), int(img["height"])) for img in gt["images"]}

    preds = load_predictions_entries(repo_root / args.predictions)
    if not preds:
        raise SystemExit(
            format_cli_error(
                code="E_PREDICTIONS_EMPTY",
                message=f"no prediction entries found in {args.predictions}",
                hint="provide at least one image entry in predictions interface contract format",
            )
        )
    normalization_warnings: list[str] = []
    if args.classes or args.assume_class_id_is_category_id:
        if not args.classes:
            raise SystemExit("--classes is required when --assume-class-id-is-category-id is enabled")
        classes = load_classes_json(repo_root / args.classes)
        transformed = normalize_class_ids(
            preds,
            classes_json=classes,
            assume_class_id_is_category_id=bool(args.assume_class_id_is_category_id),
        )
        preds = transformed.entries
        normalization_warnings = list(transformed.warnings)
    # Validate shape early; useful even in dry-run mode.
    # (Strict mode is optional; keep default permissive for external baselines.)
    from yolozu.predictions import validate_predictions_entries

    validation = validate_predictions_entries(preds, strict=False)
    dt = predictions_to_coco_detections(preds, coco_index=index, image_sizes=image_sizes, bbox_format=args.bbox_format)

    if args.dry_run:
        result = {
            "metrics": {
                "map50_95": None,
                "map50": None,
                "map75": None,
                "ar100": None,
            },
            "stats": [],
            "dry_run": True,
            "counts": {"images": len(records), "detections": len(dt)},
            "warnings": [*validation.warnings, *normalization_warnings],
        }
    else:
        result = evaluate_coco_map(gt, dt)
        result["warnings"] = [*validation.warnings, *normalization_warnings]

    payload = {
        "report_schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_id": args.protocol,
        "protocol": protocol,
        "dataset": str(args.dataset),
        "split": split_effective,
        "split_requested": args.split,
        "predictions": str(args.predictions),
        "bbox_format": args.bbox_format,
        "max_images": args.max_images,
        "normalization": {
            "classes": str(args.classes) if args.classes else None,
            "assume_class_id_is_category_id": bool(args.assume_class_id_is_category_id),
        },
        **result,
    }

    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(output_path)


if __name__ == "__main__":
    main()
