import argparse
import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.adapter import DummyAdapter, RTDETRPoseAdapter
from yolozu.dataset import build_manifest
from yolozu.predictions_transform import apply_tta
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
        "--wrap",
        action="store_true",
        help="Wrap output as {predictions: [...], meta: {...}} (recommended).",
    )
    parser.add_argument("--tta", action="store_true", help="Enable TTA post-transform on predictions.")
    parser.add_argument("--tta-seed", type=int, default=None, help="Seed for TTA randomness.")
    parser.add_argument("--tta-flip-prob", type=float, default=0.5, help="Flip probability for TTA.")
    parser.add_argument("--tta-norm-only", action="store_true", help="Update only normalized bbox values for TTA.")
    parser.add_argument("--tta-log-out", default=None, help="Optional path to write TTA log JSON.")

    add_ttt_arguments(parser, include_enable_flag=True)
    return parser.parse_args(argv)


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


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    apply_ttt_preset_args(args)

    if args.adapter == "dummy" and int(args.lora_r) > 0:
        raise SystemExit("--lora-* flags are only supported with --adapter rtdetr_pose")

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
            lora_r=int(args.lora_r),
            lora_alpha=(float(args.lora_alpha) if args.lora_alpha is not None else None),
            lora_dropout=float(args.lora_dropout),
            lora_target=str(args.lora_target),
            lora_freeze_base=bool(args.lora_freeze_base),
            lora_train_bias=str(args.lora_train_bias),
        )

    def _ttt_or_die(_records):
        try:
            return run_ttt(adapter, _records, config=ttt_config).to_dict()
        except Exception as exc:
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
            except Exception:
                extra = ""
            raise SystemExit(f"TTT failed: {exc}{extra}")

    ttt_report = None
    if args.ttt:
        ttt_config = build_ttt_config_from_args(args)
        if str(args.ttt_reset) == "sample":
            try:
                import torch
            except Exception as exc:  # pragma: no cover
                raise SystemExit(f"TTT failed: {exc}")
            try:
                from yolozu.tta.ttt_mim import select_parameters
            except Exception as exc:  # pragma: no cover
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
        tta = apply_tta(
            predictions,
            enabled=True,
            seed=args.tta_seed,
            flip_prob=args.tta_flip_prob,
            norm_only=bool(args.tta_norm_only),
        )
        predictions = tta.entries
        tta_warnings = tta.warnings
        tta_summary = _summarize_tta(predictions, warnings=tta_warnings)

    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lora_report = None
    if hasattr(adapter, "get_lora_report"):
        try:
            lora_report = adapter.get_lora_report()
        except Exception:
            lora_report = None

    if args.wrap:
        ttt_meta = build_ttt_settings_from_args(args)
        ttt_meta["report"] = ttt_report
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
                "tta": {
                    "enabled": bool(args.tta),
                    "seed": args.tta_seed,
                    "flip_prob": float(args.tta_flip_prob),
                    "norm_only": bool(args.tta_norm_only),
                    "warnings": tta_warnings,
                    "summary": tta_summary,
                },
                "ttt": ttt_meta,
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
        log_path.write_text(json.dumps(log_payload, indent=2, sort_keys=True))
        print(log_path)


if __name__ == "__main__":
    main()
