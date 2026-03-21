import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.adapter import DummyAdapter, RTDETRPoseAdapter
from yolozu.dataset import build_manifest
from yolozu.predictions_transform import apply_tta, summarize_task_coverage
from yolozu.tta.cli_options import (
    add_ttt_arguments,
    build_ttt_config_from_args,
    build_ttt_settings_from_args,
)
from yolozu.tta.integration import run_ttt
from yolozu.tta.presets import apply_ttt_preset_args


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        choices=("dummy", "rtdetr_pose"),
        default="dummy",
        help="Which adapter to run (default: dummy).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="YOLO-format dataset root (defaults to data/coco128).",
    )
    parser.add_argument(
        "--config",
        default="rtdetr_pose/configs/base.json",
        help="Config path for rtdetr_pose adapter.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for rtdetr_pose adapter (default: cpu).",
    )
    parser.add_argument(
        "--infer-batch-size",
        type=int,
        default=1,
        help="Inference batch size for rtdetr_pose adapter (default: 1).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=None,
        help="Image size for rtdetr_pose adapter (one value or two values).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Score threshold for rtdetr_pose adapter (default: 0.3).",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=50,
        help="Max detections per image for rtdetr_pose adapter (default: 50).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint for rtdetr_pose adapter.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=0,
        help="Enable LoRA by setting rank r>0 (default: 0 disables).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="LoRA alpha scaling (default: r).",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout on inputs (default: 0.0).",
    )
    parser.add_argument(
        "--lora-target",
        default="head",
        choices=("head", "all_linear", "all_conv1x1", "all_linear_conv1x1"),
        help="Where to apply LoRA (default: head).",
    )
    parser.add_argument(
        "--lora-freeze-base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze base weights and train LoRA params only (default: false).",
    )
    parser.add_argument(
        "--lora-train-bias",
        choices=("none", "all"),
        default="none",
        help="If LoRA is enabled, optionally train biases too (default: none).",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for rtdetr_pose inference (PyTorch 2.x).",
    )
    parser.add_argument(
        "--torch-compile-backend",
        default="inductor",
        help="torch.compile backend (default: inductor).",
    )
    parser.add_argument(
        "--torch-compile-mode",
        default="reduce-overhead",
        help="torch.compile mode (default: reduce-overhead).",
    )
    parser.add_argument(
        "--torch-amp",
        choices=("off", "fp16", "bf16"),
        default="off",
        help="Torch autocast dtype for inference (default: off).",
    )
    parser.add_argument(
        "--torch-channels-last",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use channels_last memory format for torch inference tensors (default: false).",
    )
    parser.add_argument(
        "--torch-inference-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torch.inference_mode instead of torch.no_grad (default: true).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for number of images (for quick smoke runs).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split under images/ and labels/ (e.g. val2017, train2017). Default: auto.",
    )
    parser.add_argument(
        "--output",
        default="reports/predictions.json",
        help="Where to write predictions JSON.",
    )
    parser.add_argument(
        "--domain-shift-recipe",
        default=None,
        help="Optional domain_shift_recipe.json path; copied into meta.export_settings.domain_shift_target.",
    )
    parser.add_argument(
        "--wrap",
        action="store_true",
        help="Wrap output as {predictions: [...], meta: {...}} (recommended).",
    )
    parser.add_argument("--tta", action="store_true", help="Enable TTA post-transform on predictions.")
    parser.add_argument(
        "--tta-mode",
        choices=("postprocess", "model"),
        default="postprocess",
        help="TTA mode: postprocess (default) or model-space branch merge.",
    )
    parser.add_argument("--tta-seed", type=int, default=None, help="Seed for TTA randomness.")
    parser.add_argument("--tta-flip-prob", type=float, default=0.5, help="Flip probability for TTA.")
    parser.add_argument("--tta-norm-only", action="store_true", help="Update only normalized bbox values for TTA.")
    parser.add_argument(
        "--tta-keypoint-swap-pairs",
        default=None,
        help="Optional keypoint swap pairs like '1:2,3:4' for hflip semantics.",
    )
    parser.add_argument(
        "--tta-model-merge-iou",
        type=float,
        default=0.55,
        help="IoU threshold to merge post-flip detections in --tta-mode model (default: 0.55).",
    )
    parser.add_argument(
        "--tta-flip-keypoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When TTA is enabled, horizontally flip keypoints x coordinates (default: true).",
    )
    parser.add_argument(
        "--tta-flip-pose-offsets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When TTA is enabled, horizontally flip pose offsets x component (default: true).",
    )
    parser.add_argument("--tta-log-out", default=None, help="Optional path to write TTA log JSON.")

    add_ttt_arguments(parser, include_enable_flag=True)
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_domain_shift_recipe(path_like):
    if not path_like:
        return None
    path = Path(str(path_like)).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise SystemExit(f"--domain-shift-recipe not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to parse --domain-shift-recipe JSON: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--domain-shift-recipe must be a JSON object")
    export_settings = payload.get("export_settings")
    if not isinstance(export_settings, dict):
        raise SystemExit("--domain-shift-recipe must contain export_settings object")
    domain_shift_target = export_settings.get("domain_shift_target")
    if not isinstance(domain_shift_target, dict):
        raise SystemExit("--domain-shift-recipe must contain export_settings.domain_shift_target object")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "domain_shift_target": domain_shift_target,
    }


def _summarize_tta(predictions, *, warnings):
    total = 0
    applied = 0
    for entry in predictions:
        mask = entry.get("tta_mask") if isinstance(entry, dict) else None
        if isinstance(mask, list):
            total += len(mask)
            applied += sum(1 for flag in mask if flag)
    ratio = float(applied) / float(total) if total else 0.0
    return {
        "detections": int(total),
        "applied": int(applied),
        "applied_ratio": float(ratio),
        "warnings": list(warnings),
    }


def _parse_swap_pairs(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    pairs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            a_text, b_text = item.split(":", 1)
        elif "-" in item:
            a_text, b_text = item.split("-", 1)
        else:
            raise SystemExit(
                "--tta-keypoint-swap-pairs must be comma-separated index pairs (example: 1:2,3:4)"
            )
        try:
            a = int(a_text.strip())
            b = int(b_text.strip())
        except ValueError as exc:
            raise SystemExit(
                "--tta-keypoint-swap-pairs must contain integer index pairs (example: 1:2,3:4)"
            ) from exc
        if a < 0 or b < 0:
            raise SystemExit("--tta-keypoint-swap-pairs indices must be >= 0")
        pairs.append((a, b))
    return pairs or None


def _xyxy_from_bbox_norm(bbox):
    if not isinstance(bbox, dict):
        return None
    try:
        cx = float(bbox["cx"])
        cy = float(bbox["cy"])
        w = float(bbox["w"])
        h = float(bbox["h"])
    except (KeyError, TypeError, ValueError):
        return None
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _bbox_iou_norm(a, b):
    aa = _xyxy_from_bbox_norm(a)
    bb = _xyxy_from_bbox_norm(b)
    if aa is None or bb is None:
        return 0.0
    ax1, ay1, ax2, ay2 = aa
    bx1, by1, bx2, by2 = bb
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _merge_det_pair(base_det, aug_det):
    merged = dict(base_det)
    try:
        s0 = float(base_det.get("score", 0.0))
    except (TypeError, ValueError):
        s0 = 0.0
    try:
        s1 = float(aug_det.get("score", 0.0))
    except (TypeError, ValueError):
        s1 = 0.0
    merged["score"] = float(max(0.0, min(1.0, 0.5 * (s0 + s1))))

    bbox0 = base_det.get("bbox")
    bbox1 = aug_det.get("bbox")
    if isinstance(bbox0, dict) and isinstance(bbox1, dict):
        merged["bbox"] = {
            "cx": 0.5 * float(bbox0.get("cx", 0.0)) + 0.5 * float(bbox1.get("cx", 0.0)),
            "cy": 0.5 * float(bbox0.get("cy", 0.0)) + 0.5 * float(bbox1.get("cy", 0.0)),
            "w": 0.5 * float(bbox0.get("w", 0.0)) + 0.5 * float(bbox1.get("w", 0.0)),
            "h": 0.5 * float(bbox0.get("h", 0.0)) + 0.5 * float(bbox1.get("h", 0.0)),
        }
    for key in ("keypoints", "offsets", "rot6d", "k_delta", "log_z", "sigma_z", "sigma_rot"):
        if key not in merged and key in aug_det:
            merged[key] = aug_det[key]
    return merged


def _merge_model_tta_branches(base_entries, aug_entries, *, iou_threshold, max_detections):
    warnings = []
    aug_by_image = {}
    for entry in aug_entries:
        if not isinstance(entry, dict):
            continue
        image = entry.get("image")
        if isinstance(image, str):
            aug_by_image[image] = entry

    merged_entries = []
    seen_images = set()
    for entry in base_entries:
        if not isinstance(entry, dict):
            continue
        image = entry.get("image")
        if not isinstance(image, str):
            continue
        seen_images.add(image)
        base_dets = list(entry.get("detections") or [])
        aug_entry = aug_by_image.get(image)
        aug_dets = list((aug_entry or {}).get("detections") or [])
        out_dets = [dict(det) for det in base_dets if isinstance(det, dict)]

        for aug_det in aug_dets:
            if not isinstance(aug_det, dict):
                continue
            best_idx = -1
            best_iou = 0.0
            aug_cls = aug_det.get("class_id")
            aug_bbox = aug_det.get("bbox")
            for idx, cand in enumerate(out_dets):
                if int(cand.get("class_id", -1)) != int(aug_cls if aug_cls is not None else -1):
                    continue
                iou = _bbox_iou_norm(cand.get("bbox"), aug_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= float(iou_threshold):
                out_dets[best_idx] = _merge_det_pair(out_dets[best_idx], aug_det)
            else:
                out_dets.append(dict(aug_det))

        out_dets.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        if int(max_detections) > 0:
            out_dets = out_dets[: int(max_detections)]
        new_entry = dict(entry)
        new_entry["detections"] = out_dets
        merged_entries.append(new_entry)

    for image, aug_entry in aug_by_image.items():
        if image in seen_images:
            continue
        warnings.append(f"tta_model_merge: image present only in augmented branch: {image}")
        merged_entries.append(dict(aug_entry))

    return merged_entries, warnings


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    domain_shift_recipe = _load_domain_shift_recipe(args.domain_shift_recipe)

    apply_ttt_preset_args(args)

    if int(args.infer_batch_size) <= 0:
        raise SystemExit("--infer-batch-size must be >= 1")

    if args.adapter == "dummy" and int(args.lora_r) > 0:
        raise SystemExit("--lora-* flags are only supported with --adapter rtdetr_pose")
    if float(args.tta_model_merge_iou) < 0.0 or float(args.tta_model_merge_iou) > 1.0:
        raise SystemExit("--tta-model-merge-iou must be within [0,1]")
    if args.adapter == "dummy":
        torch_opts_changed = bool(
            bool(args.torch_compile)
            or str(args.torch_compile_backend) != "inductor"
            or str(args.torch_compile_mode) != "reduce-overhead"
            or str(args.torch_amp) != "off"
            or bool(args.torch_channels_last)
            or not bool(args.torch_inference_mode)
        )
        if torch_opts_changed or int(args.infer_batch_size) != 1:
            raise SystemExit(
                "--torch-compile*/--torch-amp/--torch-channels-last/--[no-]torch-inference-mode "
                "and --infer-batch-size are only supported with --adapter rtdetr_pose"
            )

    dataset_root = Path(args.dataset) if args.dataset else (repo_root / "data" / "coco128")
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    if args.max_images is not None:
        records = records[: args.max_images]

    if args.adapter == "dummy":
        adapter = DummyAdapter()
    else:
        image_size = None
        if args.image_size:
            if len(args.image_size) == 1:
                image_size = (args.image_size[0], args.image_size[0])
            elif len(args.image_size) == 2:
                image_size = (args.image_size[0], args.image_size[1])
            else:
                raise SystemExit("--image-size expects 1 or 2 integers")
        adapter = RTDETRPoseAdapter(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device,
            image_size=image_size or (320, 320),
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
            infer_batch_size=int(args.infer_batch_size),
            lora_r=int(args.lora_r),
            lora_alpha=(float(args.lora_alpha) if args.lora_alpha is not None else None),
            lora_dropout=float(args.lora_dropout),
            lora_target=str(args.lora_target),
            lora_freeze_base=bool(args.lora_freeze_base),
            lora_train_bias=str(args.lora_train_bias),
            compile_model=bool(args.torch_compile),
            compile_backend=str(args.torch_compile_backend),
            compile_mode=str(args.torch_compile_mode),
            amp=str(args.torch_amp),
            channels_last=bool(args.torch_channels_last),
            use_inference_mode=bool(args.torch_inference_mode),
        )

    def _ttt_or_die(_records):
        try:
            return run_ttt(adapter, _records, config=ttt_config).to_dict()
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            extra = ""
            try:
                from yolozu.tta.ttt_mim import select_parameters

                if hasattr(adapter, "get_model"):
                    model = adapter.get_model()
                else:
                    model = None
                if model is not None:
                    params = select_parameters(
                        model,
                        update_filter=str(ttt_config.update_filter),
                        include=ttt_config.include,
                        exclude=ttt_config.exclude,
                    )
                    count = 0
                    seen = set()
                    for p in params:
                        pid = id(p)
                        if pid in seen:
                            continue
                        seen.add(pid)
                        count += int(p.numel())
                    extra = (
                        f" (method={ttt_config.method} update_filter={ttt_config.update_filter} "
                        f"selected_param_count={count} steps={ttt_config.steps} lr={ttt_config.lr})"
                    )
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                extra = ""
            raise SystemExit(f"TTT failed: {exc}{extra}")

    ttt_report = None
    if args.ttt:
        ttt_config = build_ttt_config_from_args(args)
        if str(args.ttt_reset) == "sample":
            try:
                import torch
            except ImportError as exc:  # pragma: no cover
                raise SystemExit(f"TTT failed: {exc}")
            try:
                from yolozu.tta.ttt_mim import select_parameters
            except ImportError as exc:  # pragma: no cover
                raise SystemExit(f"TTT failed: {exc}")

            model = adapter.get_model()
            params = select_parameters(
                model,
                update_filter=str(ttt_config.update_filter),
                include=ttt_config.include,
                exclude=ttt_config.exclude,
            )
            if not params:
                raise SystemExit("TTT failed: no parameters selected for TTT")
            with torch.no_grad():
                base_snapshot = [(p, p.detach().clone()) for p in params]
                base_buffers = []
                for name, buffer in model.named_buffers():
                    if buffer is None:
                        continue
                    name = str(name)
                    if not name.endswith(("running_mean", "running_var", "num_batches_tracked")):
                        continue
                    base_buffers.append((buffer, buffer.detach().clone()))

            def _restore_base():
                with torch.no_grad():
                    for p, value in base_snapshot:
                        p.copy_(value)
                    for buffer, value in base_buffers:
                        buffer.copy_(value)

            predictions = []
            per_sample: list[dict] = []
            max_keep = 20
            for idx, record in enumerate(records):
                _restore_base()
                try:
                    rep = _ttt_or_die([record])
                    pred = adapter.predict([record])
                finally:
                    _restore_base()
                predictions.extend(pred)
                if idx < max_keep:
                    per_sample.append({"index": int(idx), "image": record.get("image"), "report": rep})
            ttt_report = {
                "mode": "sample",
                "samples_total": int(len(records)),
                "samples_kept": int(len(per_sample)),
                "samples_truncated": bool(len(records) > max_keep),
                "per_sample": per_sample,
            }
        else:
            ttt_report = _ttt_or_die(records)
            predictions = adapter.predict(records)
    else:
        predictions = adapter.predict(records)

    tta_warnings = []
    tta_summary = None
    if args.tta:
        swap_pairs = _parse_swap_pairs(args.tta_keypoint_swap_pairs)
        tta_mode = str(args.tta_mode)
        if tta_mode == "model" and args.adapter == "rtdetr_pose":
            aug_records = []
            for record in records:
                if not isinstance(record, dict):
                    continue
                record_aug = dict(record)
                record_aug["_tta_hflip"] = True
                aug_records.append(record_aug)
            aug_predictions = adapter.predict(aug_records)
            tta_aug = apply_tta(
                aug_predictions,
                enabled=True,
                seed=args.tta_seed,
                flip_prob=1.0,
                norm_only=bool(args.tta_norm_only),
                flip_keypoints=bool(args.tta_flip_keypoints),
                flip_pose_offsets=bool(args.tta_flip_pose_offsets),
                keypoint_swap_pairs=swap_pairs,
            )
            predictions, merge_warnings = _merge_model_tta_branches(
                predictions,
                tta_aug.entries,
                iou_threshold=float(args.tta_model_merge_iou),
                max_detections=int(args.max_detections),
            )
            tta_warnings = [*tta_aug.warnings, *merge_warnings]
            tta_summary = _summarize_tta(predictions, warnings=tta_warnings)
            tta_summary["mode"] = "model"
            tta_summary["branches"] = {
                "base_images": int(len(records)),
                "aug_images": int(len(aug_records)),
            }
        else:
            if tta_mode == "model":
                tta_warnings.append(
                    "--tta-mode model is only available for --adapter rtdetr_pose; falling back to postprocess"
                )
            tta = apply_tta(
                predictions,
                enabled=True,
                seed=args.tta_seed,
                flip_prob=args.tta_flip_prob,
                norm_only=bool(args.tta_norm_only),
                flip_keypoints=bool(args.tta_flip_keypoints),
                flip_pose_offsets=bool(args.tta_flip_pose_offsets),
                keypoint_swap_pairs=swap_pairs,
            )
            predictions = tta.entries
            tta_warnings.extend(tta.warnings)
            tta_summary = _summarize_tta(predictions, warnings=tta_warnings)
            tta_summary["mode"] = "postprocess"

    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lora_report = None
    if hasattr(adapter, "get_lora_report"):
        try:
            lora_report = adapter.get_lora_report()
        except (AttributeError, RuntimeError):
            lora_report = None

    if args.wrap:
        ttt_meta = build_ttt_settings_from_args(args)
        ttt_meta["report"] = ttt_report
        task_coverage = summarize_task_coverage(predictions)
        payload = {
            "predictions": predictions,
            "meta": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "adapter": args.adapter,
                "config": args.config,
                "checkpoint": args.checkpoint,
                "images": len(records),
                "lora": {
                    "enabled": bool(int(args.lora_r) > 0),
                    "r": int(args.lora_r),
                    "alpha": (float(args.lora_alpha) if args.lora_alpha is not None else None),
                    "dropout": float(args.lora_dropout),
                    "target": str(args.lora_target),
                    "freeze_base": bool(args.lora_freeze_base),
                    "train_bias": str(args.lora_train_bias),
                    "report": lora_report,
                },
                "inference": {
                    "infer_batch_size": int(args.infer_batch_size),
                    "torch_compile": {
                        "enabled": bool(args.torch_compile),
                        "backend": (str(args.torch_compile_backend) if bool(args.torch_compile) else None),
                        "mode": (str(args.torch_compile_mode) if bool(args.torch_compile) else None),
                    },
                    "torch_amp": str(args.torch_amp),
                    "torch_channels_last": bool(args.torch_channels_last),
                    "torch_inference_mode": bool(args.torch_inference_mode),
                },
                "tta": {
                    "enabled": bool(args.tta),
                    "mode": str(args.tta_mode) if bool(args.tta) else "postprocess",
                    "seed": args.tta_seed,
                    "flip_prob": float(args.tta_flip_prob),
                    "norm_only": bool(args.tta_norm_only),
                    "keypoint_swap_pairs": (
                        list(_parse_swap_pairs(args.tta_keypoint_swap_pairs) or [])
                        if bool(args.tta)
                        else []
                    ),
                    "flip_keypoints": bool(args.tta_flip_keypoints),
                    "flip_pose_offsets": bool(args.tta_flip_pose_offsets),
                    "warnings": tta_warnings,
                    "summary": tta_summary,
                },
                "ttt": ttt_meta,
                "task_coverage": task_coverage,
            },
        }
        if isinstance(domain_shift_recipe, dict):
            payload["meta"]["export_settings"] = {
                "domain_shift_target": dict(domain_shift_recipe["domain_shift_target"]),
                "domain_shift_recipe": {
                    "path": str(domain_shift_recipe["path"]),
                    "sha256": str(domain_shift_recipe["sha256"]),
                },
            }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    else:
        output_path.write_text(json.dumps(predictions, indent=2, sort_keys=True))

    print(output_path)

    if args.tta_log_out and args.tta:
        log_path = Path(args.tta_log_out)
        if not log_path.is_absolute():
            log_path = repo_root / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output": str(output_path),
            "tta": {
                "enabled": bool(args.tta),
                "seed": args.tta_seed,
                "flip_prob": float(args.tta_flip_prob),
                "norm_only": bool(args.tta_norm_only),
                "flip_keypoints": bool(args.tta_flip_keypoints),
                "flip_pose_offsets": bool(args.tta_flip_pose_offsets),
                "summary": tta_summary,
            },
        }
        log_path.write_text(json.dumps(log_payload, indent=2, sort_keys=True))
        print(log_path)

    if args.ttt_log_out and args.ttt:
        log_path = Path(args.ttt_log_out)
        if not log_path.is_absolute():
            log_path = repo_root / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ttt_log = build_ttt_settings_from_args(args)
        ttt_log["report"] = ttt_report
        log_payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output": str(output_path),
            "ttt": ttt_log,
        }
        if isinstance(domain_shift_recipe, dict):
            log_payload["export_settings"] = {
                "domain_shift_target": dict(domain_shift_recipe["domain_shift_target"]),
                "domain_shift_recipe": {
                    "path": str(domain_shift_recipe["path"]),
                    "sha256": str(domain_shift_recipe["sha256"]),
                },
            }
        log_path.write_text(json.dumps(log_payload, indent=2, sort_keys=True))
        print(log_path)


if __name__ == "__main__":
    main()
