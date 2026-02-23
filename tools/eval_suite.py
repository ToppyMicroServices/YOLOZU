import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.coco_eval import build_coco_ground_truth, evaluate_coco_map, predictions_to_coco_detections
from yolozu.dataset import build_manifest
from yolozu.eval_protocol import apply_eval_protocol_args, eval_protocol_hash, load_eval_protocol
from yolozu.predictions import load_predictions_payload, validate_predictions_entries


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        choices=("yolo26", "nms_applied", "e2e_nms_free"),
        default=None,
        help="Apply canonical evaluation protocol presets (pins split/bbox_format).",
    )
    parser.add_argument(
        "--allow-protocol-mismatches",
        action="store_true",
        help="Do not fail the process on fixed-condition protocol mismatches (still recorded in output).",
    )
    parser.add_argument("--dataset", default="data/coco128", help="YOLO-format COCO root (images/ + labels/).")
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split under images/ and labels/ (e.g. val2017, train2017). Default: auto (val2017 if present else train2017).",
    )
    parser.add_argument(
        "--predictions-glob",
        required=True,
        help="Glob for predictions JSON files (e.g. 'reports/pred_yolo26*.json').",
    )
    parser.add_argument(
        "--bbox-format",
        choices=("cxcywh_norm", "cxcywh_abs", "xywh_abs", "xyxy_abs"),
        default="cxcywh_norm",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip COCOeval and only validate/convert predictions (no pycocotools required).",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument("--strict", action="store_true", help="Strict prediction schema validation.")
    parser.add_argument("--output", default="reports/eval_suite.json", help="Output JSON path.")
    return parser.parse_args(argv)


def _resolve_args(argv):
    args = _parse_args(argv)
    protocol = load_eval_protocol(args.protocol) if args.protocol else None
    if protocol:
        args = apply_eval_protocol_args(args, protocol)
    return args, protocol


def _now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normalize_image_size(value: Any) -> dict[str, int] | None:
    if isinstance(value, dict):
        w = _coerce_int(value.get("width"))
        h = _coerce_int(value.get("height"))
        if w is None or h is None:
            return None
        if w <= 0 or h <= 0:
            return None
        return {"width": int(w), "height": int(h)}
    if isinstance(value, (list, tuple)) and len(value) == 2:
        w = _coerce_int(value[0])
        h = _coerce_int(value[1])
        if w is None or h is None:
            return None
        if w <= 0 or h <= 0:
            return None
        return {"width": int(w), "height": int(h)}
    return None


def _resolve_meta_config(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(meta, dict):
        return None
    run = meta.get("run")
    if isinstance(run, dict):
        cfg = run.get("config_fingerprint")
        if isinstance(cfg, dict):
            return cfg
    return meta


def _extract_preprocess(entries: list[dict[str, Any]], meta_cfg: dict[str, Any] | None) -> dict[str, Any] | None:
    if entries:
        entry0 = entries[0]
        pp = entry0.get("preprocess")
        if isinstance(pp, dict):
            out: dict[str, Any] = {}
            method = pp.get("method")
            if isinstance(method, str) and method:
                out["method"] = method
            norm = pp.get("normalize")
            if isinstance(norm, str) and norm:
                out["normalize"] = norm
            input_color = pp.get("input_color")
            if isinstance(input_color, str) and input_color:
                out["input_color"] = input_color
            resize_interp = pp.get("resize_interp")
            if isinstance(resize_interp, str) and resize_interp:
                out["resize_interp"] = resize_interp
            letterbox_fill = pp.get("letterbox_fill")
            normalized_fill = _normalize_fill(letterbox_fill)
            if normalized_fill is not None:
                out["letterbox_fill"] = normalized_fill
            input_size = _normalize_image_size(pp.get("input_size"))
            if input_size is None:
                input_size = _normalize_image_size(entry0.get("image_size"))
            if input_size is not None:
                out["input_size"] = input_size
            if out:
                return out

    exporter = None
    if isinstance(meta_cfg, dict):
        v = meta_cfg.get("exporter")
        if isinstance(v, str) and v:
            exporter = v
    if exporter in {"onnxruntime", "tensorrt"}:
        return {
            "method": "letterbox",
            "input_color": "RGB",
            "normalize": "0_1",
            "resize_interp": "linear",
            "letterbox_fill": [114, 114, 114],
        }
    return None


def _extract_export_settings(
    entries: list[dict[str, Any]], meta: dict[str, Any] | None
) -> dict[str, Any] | None:
    meta_cfg = _resolve_meta_config(meta)

    imgsz = None
    if isinstance(meta_cfg, dict):
        imgsz = _coerce_int(meta_cfg.get("imgsz"))
    if imgsz is None and entries:
        entry0 = entries[0]
        size = _normalize_image_size(entry0.get("image_size"))
        if size is None and isinstance(entry0.get("preprocess"), dict):
            size = _normalize_image_size(entry0["preprocess"].get("input_size"))
        if size is not None and size["width"] == size["height"]:
            imgsz = int(size["width"])

    score_threshold = None
    iou_threshold = None
    max_detections = None
    if isinstance(meta_cfg, dict):
        score_threshold = _coerce_float(
            meta_cfg.get("score_threshold", meta_cfg.get("conf", meta_cfg.get("min_score")))
        )
        iou_threshold = _coerce_float(meta_cfg.get("iou_threshold", meta_cfg.get("nms_iou", meta_cfg.get("iou"))))
        max_detections = _coerce_int(
            meta_cfg.get("max_detections", meta_cfg.get("max_det", meta_cfg.get("topk")))
        )

    image_size = None
    if imgsz is not None:
        image_size = {"width": int(imgsz), "height": int(imgsz)}
    if entries:
        entry0 = entries[0]
        inferred = _normalize_image_size(entry0.get("image_size"))
        if inferred is None and isinstance(entry0.get("preprocess"), dict):
            inferred = _normalize_image_size(entry0["preprocess"].get("input_size"))
        if inferred is not None:
            image_size = inferred

    preprocess = _extract_preprocess(entries, meta_cfg)

    settings: dict[str, Any] = {
        "imgsz": imgsz,
        "image_size": image_size,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "max_detections": max_detections,
        "preprocess": preprocess,
    }
    if all(v is None for v in settings.values()):
        return None
    return settings


def _extract_predictions_meta_ref(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(meta, dict):
        return None
    meta_cfg = _resolve_meta_config(meta) or {}
    ref: dict[str, Any] = {}
    run = meta.get("run")
    if isinstance(run, dict):
        cfg_hash = run.get("config_hash")
        if isinstance(cfg_hash, str) and cfg_hash:
            ref["config_hash"] = cfg_hash

    for key in (
        "exporter",
        "generator",
        "protocol_id",
        "adapter",
        "config",
        "checkpoint",
        "onnx",
        "onnx_sha256",
        "engine",
        "engine_sha256",
    ):
        value = meta_cfg.get(key)
        if value is None:
            value = meta.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            ref[key] = value
    return ref or None


def _normalize_fill(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    out: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            return None
        out.append(int(item))
    return out


def _compare_fixed_conditions(
    export_settings: dict[str, Any] | None,
    protocol: dict[str, Any] | None,
    *,
    result_name: str,
    runtime_bbox_format: str,
) -> list[str]:
    if not isinstance(protocol, dict):
        return []
    fixed = protocol.get("fixed_conditions")
    if not isinstance(fixed, dict):
        return []

    mismatches: list[str] = []
    if not isinstance(export_settings, dict):
        return [f"{result_name}: missing export settings required for fixed-condition protocol validation"]

    def check_eq(key: str) -> None:
        expected = fixed.get(key)
        actual = export_settings.get(key)
        if actual is None:
            mismatches.append(f"{result_name}: missing export_settings.{key} (expected {expected})")
            return
        if actual != expected:
            mismatches.append(f"{result_name}: export_settings.{key}={actual} does not match protocol fixed_conditions.{key}={expected}")

    check_eq("imgsz")
    check_eq("score_threshold")
    check_eq("iou_threshold")
    check_eq("max_detections")

    expected_bbox = fixed.get("bbox_format")
    if expected_bbox is not None and runtime_bbox_format != expected_bbox:
        mismatches.append(
            f"{result_name}: runtime bbox_format={runtime_bbox_format} does not match protocol fixed_conditions.bbox_format={expected_bbox}"
        )

    expected_pp = fixed.get("preprocess")
    actual_pp = export_settings.get("preprocess")
    if isinstance(expected_pp, dict):
        if not isinstance(actual_pp, dict):
            mismatches.append(f"{result_name}: missing export_settings.preprocess")
        else:
            for key in ("method", "input_color", "normalize", "resize_interp"):
                exp_v = expected_pp.get(key)
                act_v = actual_pp.get(key)
                if act_v != exp_v:
                    mismatches.append(
                        f"{result_name}: export_settings.preprocess.{key}={act_v} does not match protocol preprocess.{key}={exp_v}"
                    )
            exp_fill = _normalize_fill(expected_pp.get("letterbox_fill"))
            act_fill = _normalize_fill(actual_pp.get("letterbox_fill"))
            if exp_fill is not None and act_fill != exp_fill:
                mismatches.append(
                    f"{result_name}: export_settings.preprocess.letterbox_fill={act_fill} does not match protocol preprocess.letterbox_fill={exp_fill}"
                )

    return mismatches


def main(argv=None):
    args, protocol = _resolve_args(sys.argv[1:] if argv is None else argv)

    dataset_root = repo_root / args.dataset
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    split_effective = manifest["split"]
    if args.max_images is not None:
        records = records[: args.max_images]

    gt, index = build_coco_ground_truth(records)
    image_sizes = {img["id"]: (int(img["width"]), int(img["height"])) for img in gt["images"]}

    if Path(args.predictions_glob).is_absolute():
        pred_paths = [Path(p) for p in sorted(glob.glob(args.predictions_glob))]
    else:
        pred_paths = sorted((repo_root / ".").glob(args.predictions_glob))
    if not pred_paths:
        raise SystemExit(f"no predictions matched: {args.predictions_glob}")

    results = []
    protocol_mismatches: list[str] = []
    for path in pred_paths:
        entries, meta = load_predictions_payload(path)
        validation = validate_predictions_entries(entries, strict=args.strict)
        dt = predictions_to_coco_detections(
            entries, coco_index=index, image_sizes=image_sizes, bbox_format=args.bbox_format
        )
        export_settings = _extract_export_settings(entries, meta)
        meta_ref = _extract_predictions_meta_ref(meta)
        fixed_condition_mismatches = _compare_fixed_conditions(
            export_settings,
            protocol,
            result_name=path.stem,
            runtime_bbox_format=args.bbox_format,
        )
        protocol_mismatches.extend(fixed_condition_mismatches)
        if args.dry_run:
            eval_result = {
                "metrics": {
                    "map50_95": None,
                    "map50": None,
                    "map75": None,
                    "ar100": None,
                },
                "stats": [],
                "dry_run": True,
                "counts": {"images": len(records), "detections": len(dt)},
            }
        else:
            eval_result = evaluate_coco_map(gt, dt)
        results.append(
            {
                "name": path.stem,
                "path": str(path),
                "warnings": validation.warnings,
                "export_settings": export_settings,
                "predictions_meta_ref": meta_ref,
                "protocol_check": {
                    "ok": len(fixed_condition_mismatches) == 0,
                    "mismatches": fixed_condition_mismatches,
                },
                **eval_result,
            }
        )

    if protocol and protocol_mismatches and not args.allow_protocol_mismatches:
        joined = "\n".join(protocol_mismatches)
        raise SystemExit(f"fixed-condition protocol validation failed:\n{joined}")

    protocol_version = None
    protocol_digest = None
    if protocol:
        protocol_version = protocol.get("protocol_schema_version")
        protocol_digest = eval_protocol_hash(protocol)

    payload = {
        "report_schema_version": 1,
        "timestamp": _now_utc(),
        "protocol_id": args.protocol,
        "protocol_schema_version": protocol_version,
        "protocol_hash": protocol_digest,
        "protocol": protocol,
        "dataset": str(args.dataset),
        "split": split_effective,
        "split_requested": args.split,
        "bbox_format": args.bbox_format,
        "images": len(records),
        "results": results,
    }

    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(output_path)


if __name__ == "__main__":
    main()
