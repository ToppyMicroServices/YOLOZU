import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.adapter import DummyAdapter, PrecomputedAdapter, RTDETRPoseAdapter
from yolozu.boxes import iou_xyxy_abs
from yolozu.cli_args import (
    require_float_in_range,
    require_non_negative_int,
    require_positive_int,
    resolve_input_path,
    resolve_output_path,
)
from yolozu.dataset import build_manifest
from yolozu.image_keys import add_image_aliases, lookup_image_alias
from yolozu.scenario_suite import build_report
from yolozu.coco_eval import build_coco_ground_truth, evaluate_coco_map, predictions_to_coco_detections


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--adapter",
        choices=("dummy", "precomputed", "rtdetr_pose"),
        default="dummy",
        help="Which adapter to run (default: dummy).",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="YOLO-format dataset root (defaults to data/coco128).",
    )
    p.add_argument(
        "--predictions",
        default=None,
        help="Predictions JSON path for --adapter precomputed.",
    )
    p.add_argument(
        "--config",
        default="rtdetr_pose/configs/base.json",
        help="Config path for rtdetr_pose adapter.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Device for rtdetr_pose adapter (default: cpu).",
    )
    p.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=None,
        help="Image size for rtdetr_pose adapter (one value or two values).",
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Score threshold for rtdetr_pose adapter (default: 0.3).",
    )
    p.add_argument(
        "--max-detections",
        type=int,
        default=50,
        help="Max detections per image for rtdetr_pose adapter (default: 50).",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint for rtdetr_pose adapter.",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for number of images (for quick smoke runs).",
    )
    p.add_argument(
        "--split",
        default=None,
        help="Dataset split under images/ and labels/ (e.g. val2017, train2017). Default: auto.",
    )
    p.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching detections to labels (default: 0.5).",
    )
    p.add_argument(
        "--output",
        default="reports/baseline.json",
        help="Where to write baseline JSON (default: reports/baseline.json).",
    )
    p.add_argument(
        "--coco",
        action="store_true",
        help="Also run COCO mAP evaluation (requires pycocotools).",
    )
    p.add_argument(
        "--no-scenarios",
        action="store_true",
        help="Disable scenario suite section in the report.",
    )
    p.add_argument(
        "--predictions-out",
        default=None,
        help="Optional path to write predictions JSON from adapter.",
    )
    p.add_argument(
        "--wrap",
        action="store_true",
        help="Wrap predictions output as {predictions: [...], meta: {...}} when --predictions-out is set.",
    )
    return p.parse_args(argv)


def _git_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return out or None
    except Exception:
        return None


def _sha256_json(obj) -> str:
    dumped = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(dumped).hexdigest()


def _bbox_xyxy_from_cxcywh_norm(cx, cy, w, h):
    x1 = float(cx) - float(w) / 2.0
    y1 = float(cy) - float(h) / 2.0
    x2 = float(cx) + float(w) / 2.0
    y2 = float(cy) + float(h) / 2.0
    return x1, y1, x2, y2


def _extract_det_bbox(det):
    bbox = det.get("bbox") if isinstance(det, dict) else None
    if not isinstance(bbox, dict):
        return None
    try:
        return _bbox_xyxy_from_cxcywh_norm(bbox["cx"], bbox["cy"], bbox["w"], bbox["h"])
    except Exception:
        return None


def _extract_label_bbox(label):
    try:
        return _bbox_xyxy_from_cxcywh_norm(label["cx"], label["cy"], label["w"], label["h"])
    except Exception:
        return None


def _index_predictions(predictions):
    index = {}
    for entry in predictions or []:
        if not isinstance(entry, dict):
            continue
        image = entry.get("image")
        if not image:
            continue
        dets = entry.get("detections") or []
        if not isinstance(dets, list):
            dets = []
        add_image_aliases(index, str(image), dets)
    return index


def _match_counts(records, predictions, iou_thr):
    pred_index = _index_predictions(predictions)
    total_gt = 0
    total_pred = 0
    matched = 0

    for record in records:
        labels = record.get("labels") or []
        dets = lookup_image_alias(pred_index, record.get("image")) or []

        total_gt += len(labels)
        total_pred += len(dets)

        if not labels or not dets:
            continue

        pairs = []
        for det_idx, det in enumerate(dets):
            det_bbox = _extract_det_bbox(det)
            if det_bbox is None:
                continue
            det_cls = det.get("class_id") if isinstance(det, dict) else None
            for gt_idx, gt in enumerate(labels):
                if det_cls is not None and det_cls != gt.get("class_id"):
                    continue
                gt_bbox = _extract_label_bbox(gt)
                if gt_bbox is None:
                    continue
                iou = iou_xyxy_abs(det_bbox, gt_bbox)
                if iou >= iou_thr:
                    pairs.append((iou, det_idx, gt_idx))

        pairs.sort(reverse=True, key=lambda item: item[0])
        used_det = set()
        used_gt = set()
        for _, det_idx, gt_idx in pairs:
            if det_idx in used_det or gt_idx in used_gt:
                continue
            used_det.add(det_idx)
            used_gt.add(gt_idx)
            matched += 1

    return matched, total_gt, total_pred


def _summary_metrics(records, predictions, iou_thr):
    matched, total_gt, total_pred = _match_counts(records, predictions, iou_thr)
    precision = matched / total_pred if total_pred else 0.0
    recall = matched / total_gt if total_gt else 0.0
    map_est = precision
    rejection_rate = max(0.0, 1.0 - precision)
    return {
        "matched": matched,
        "total_gt": total_gt,
        "total_pred": total_pred,
        "precision": precision,
        "recall": recall,
        "map": map_est,
        "rejection_rate": rejection_rate,
        "detections_per_image": (total_pred / max(1, len(records))),
    }


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        max_images = require_non_negative_int(args.max_images, flag_name="--max-images")
        max_detections = require_positive_int(args.max_detections, flag_name="--max-detections")
        iou_threshold = require_float_in_range(args.iou_threshold, flag_name="--iou-threshold", minimum=0.0, maximum=1.0)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    cwd = Path.cwd()
    dataset_root = (
        resolve_input_path(args.dataset, cwd=cwd, repo_root=repo_root)
        if args.dataset
        else resolve_input_path("data/coco128", cwd=cwd, repo_root=repo_root)
    )
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    if max_images is not None:
        records = records[: int(max_images)]

    precomputed_path: Path | None = None
    image_size = None
    config_path = resolve_input_path(args.config, cwd=cwd, repo_root=repo_root)
    checkpoint_path = (
        resolve_input_path(args.checkpoint, cwd=cwd, repo_root=repo_root) if args.checkpoint else None
    )
    if args.adapter == "dummy":
        adapter = DummyAdapter()
    elif args.adapter == "precomputed":
        if not args.predictions:
            raise SystemExit("--predictions is required for --adapter precomputed")
        precomputed_path = resolve_input_path(args.predictions, cwd=cwd, repo_root=repo_root)
        adapter = PrecomputedAdapter(predictions_path=str(precomputed_path))
    else:
        if args.image_size:
            if len(args.image_size) == 1:
                image_size = (args.image_size[0], args.image_size[0])
            elif len(args.image_size) == 2:
                image_size = (args.image_size[0], args.image_size[1])
            else:
                raise SystemExit("--image-size expects 1 or 2 integers")
        adapter = RTDETRPoseAdapter(
            config_path=str(config_path),
            checkpoint_path=(str(checkpoint_path) if checkpoint_path is not None else None),
            device=args.device,
            image_size=image_size or (320, 320),
            score_threshold=args.score_threshold,
            max_detections=int(max_detections or 1),
        )

    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start = time.perf_counter()
    predictions = adapter.predict(records)
    elapsed = time.perf_counter() - start
    fps = (len(records) / elapsed) if elapsed > 0 else float("inf")
    finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    summary = _summary_metrics(records, predictions, float(iou_threshold or 0.0))
    scenario_report = None
    if not bool(args.no_scenarios):
        scenario_report = build_report()
        for entry in scenario_report.get("scenarios", []):
            metrics = entry.get("metrics", {})
            metrics["fps"] = round(fps, 3)
            metrics["recall"] = round(summary["recall"], 4)
            metrics["map"] = round(summary["map"], 4)
            metrics["rejection_rate"] = round(summary["rejection_rate"], 4)
            entry["metrics"] = metrics

    coco_section: dict[str, object] | None = None
    if bool(args.coco):
        coco_section = {"enabled": True, "error": None, "metrics": None, "stats": None}
        try:
            gt, coco_index = build_coco_ground_truth(records)
            image_sizes = {img["id"]: (int(img["width"]), int(img["height"])) for img in gt.get("images", [])}
            dt = predictions_to_coco_detections(predictions, coco_index=coco_index, image_sizes=image_sizes)
            coco_eval = evaluate_coco_map(gt, dt)
            coco_section["metrics"] = coco_eval.get("metrics")
            coco_section["stats"] = coco_eval.get("stats")
        except Exception as exc:
            coco_section["error"] = str(exc)
    else:
        coco_section = {"enabled": False}

    output_path = resolve_output_path(args.output, cwd=cwd)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = {
        "schema_version": 1,
        "meta": {
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "argv": list(sys.argv if argv is None else ["run_baseline"] + list(argv)),
            "git_sha": _git_sha(repo_root),
            "dataset": str(dataset_root),
            "split": str(manifest.get("split")),
            "adapter": str(args.adapter),
            "predictions_in": str(precomputed_path) if precomputed_path is not None else None,
            "config": str(config_path),
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
            "device": str(args.device),
            "image_size": list(image_size or (320, 320)) if args.adapter == "rtdetr_pose" else None,
            "iou_threshold": float(iou_threshold or 0.0),
        },
        "speed": {
            "images": int(len(records)),
            "seconds": float(elapsed),
            "fps": float(round(fps, 3)),
        },
        "summary": summary,
        "scenario_suite": scenario_report,
        "coco": coco_section,
        "predictions": {
            "hash_sha256": _sha256_json(predictions),
            "images": int(len(predictions)),
        },
    }
    output_path.write_text(json.dumps(baseline, indent=2, sort_keys=True))

    if args.predictions_out:
        payload = predictions
        if args.wrap:
            payload = {
                "predictions": predictions,
                "meta": {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "adapter": args.adapter,
                    "images": len(records),
                },
            }
        pred_path = resolve_output_path(args.predictions_out, cwd=cwd)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(output_path)


if __name__ == "__main__":
    main()
