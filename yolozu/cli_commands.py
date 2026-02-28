"""YOLOZU command-line interface.

Provides the ``yolozu`` CLI with subcommands for training, evaluation,
export, doctor diagnostics, dataset migration, model fetching, and
demo pipelines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


from .cli_args import parse_image_size_arg, require_non_negative_int
from .config import simple_yaml_load

__all__: list[str] = []


def _load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
            return data or {}
        except Exception:
            return simple_yaml_load(text)
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        return json.loads(text)
    except Exception:
        return simple_yaml_load(text)


def _build_args_from_config(cfg: dict) -> list[str]:
    args: list[str] = []
    for key, value in cfg.items():
        if value is None:
            continue
        arg = f"--{str(key).replace('_', '-') }"
        if isinstance(value, bool):
            if value:
                args.append(arg)
            continue
        if isinstance(value, (list, tuple)):
            args.append(arg)
            args.extend([str(v) for v in value])
            continue
        args.append(arg)
        args.append(str(value))
    return args


def _resolve_auto_dataset_from_args(args: argparse.Namespace) -> str:
    if getattr(args, "instances", None) and getattr(args, "images_dir", None):
        return "coco-instances"
    if getattr(args, "data", None):
        return "ultralytics"
    raise SystemExit(
        "could not auto-detect dataset source; provide --data (ultralytics) or --instances + --images-dir (coco-instances)"
    )


def _detect_config_source_from_path(path_like: str | Path) -> str:
    p = Path(path_like).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        raise SystemExit(f"config not found for auto-detect: {p}")

    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()

    if suffix in (".yaml", ".yml", ".json"):
        try:
            cfg = _load_config(p)
        except Exception:
            cfg = {}
        if isinstance(cfg, dict):
            upper_keys = {str(k) for k in cfg.keys()}
            if {"MODEL", "SOLVER"} & upper_keys:
                return "detectron2"
            if any(k in cfg for k in ("imgsz", "batch", "epochs", "lr0", "weight_decay", "optimizer", "model")):
                return "ultralytics"
        if "solver:" in lower and "model:" in lower:
            return "detectron2"
        return "ultralytics"

    if suffix == ".py":
        if "yolox" in lower or "def get_exp" in lower or "class exp" in lower:
            return "yolox"
        if "detectron2" in lower:
            return "detectron2"
        if "mmengine" in lower or "train_dataloader" in lower or "optim_wrapper" in lower or "default_scope = 'mmdet'" in lower:
            return "mmdet"
        if "_base_" in lower:
            return "mmdet"
        raise SystemExit(f"could not auto-detect config source from Python file: {p}")

    raise SystemExit(f"could not auto-detect config source from file: {p}")


def _resolve_auto_config_from_args(args: argparse.Namespace) -> str:
    args_path = getattr(args, "args", None)
    cfg_path = getattr(args, "config", None) or getattr(args, "cfg", None)
    if args_path:
        return _detect_config_source_from_path(str(args_path))
    if cfg_path:
        return _detect_config_source_from_path(str(cfg_path))
    raise SystemExit("could not auto-detect config source; provide --args or --config/--cfg")


def _cmd_train(config_path: Path, extra_args: list[str] | None = None) -> int:
    try:
        from rtdetr_pose.train_minimal import main as train_main
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "yolozu train requires optional training deps. Install `yolozu[train]` (or `yolozu[full]`) "
            "to enable the RT-DETR pose trainer."
        ) from exc

    argv = ["--config", str(config_path)]
    if extra_args:
        argv.extend(list(extra_args))
    return int(train_main(argv))


def _cmd_train_import_preview(args: argparse.Namespace) -> int:
    from yolozu.imports import (
        import_detectron2_config,
        import_mmdet_config,
        import_ultralytics_config,
        import_yolox_config,
    )

    from_format = str(getattr(args, "import_from", "") or "").strip().lower()
    if not from_format:
        return 0

    if from_format == "auto":
        from_format = _resolve_auto_config_from_args(args)

    cfg_path = str(getattr(args, "cfg", "") or "").strip()
    if not cfg_path:
        raise SystemExit("--cfg is required when using train --import")

    doctor_args = argparse.Namespace(
        output="-",
        dataset_from=("ultralytics" if getattr(args, "data", None) else None),
        config_from=from_format,
        data=(str(args.data) if getattr(args, "data", None) else None),
        args=(cfg_path if from_format == "ultralytics" else None),
        task=None,
        split=None,
        max_images=200,
        instances=None,
        images_dir=None,
        include_crowd=False,
        config=(cfg_path if from_format in ("mmdet", "yolox", "detectron2") else None),
    )
    doctor_rc = int(_cmd_doctor_import(doctor_args))
    if doctor_rc != 0:
        raise SystemExit("train --import preview failed (doctor import reported errors)")

    output = str(getattr(args, "resolved_config_out", "reports/train_config_resolved_import.json") or "reports/train_config_resolved_import.json")
    force = bool(getattr(args, "force_import_overwrite", False))

    if from_format == "ultralytics":
        out = import_ultralytics_config(args_yaml=cfg_path, output=output, force=force)
    elif from_format == "mmdet":
        out = import_mmdet_config(config=cfg_path, output=output, force=force)
    elif from_format == "yolox":
        out = import_yolox_config(config=cfg_path, output=output, force=force)
    elif from_format == "detectron2":
        out = import_detectron2_config(config=cfg_path, output=output, force=force)
    else:
        raise SystemExit("unsupported --import value")

    print(str(out))
    return 0


def _cmd_test(config_path: Path, extra_args: list[str] | None = None) -> int:
    try:
        from yolozu.scenarios_cli import main as scenarios_main
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "yolozu test failed to import scenario runner."
        ) from exc

    cfg = _load_config(config_path)
    args = _build_args_from_config(cfg)
    if extra_args:
        args.extend(list(extra_args))
    scenarios_main(args)
    return 0


def _cmd_doctor(output: str) -> int:
    from yolozu.doctor import write_doctor_report

    return int(write_doctor_report(output=output))


def _cmd_list_models(args: argparse.Namespace) -> int:
    from yolozu.model_fetch import list_models

    specs = list_models(registry_path=getattr(args, "registry", None))
    if bool(getattr(args, "json", False)):
        payload = {
            "models": [
                {
                    "id": spec.model_id,
                    "family": spec.family,
                    "version": spec.version,
                    "license": spec.license,
                    "sha256": spec.expected_sha256,
                    "sha256_present": bool(spec.expected_sha256),
                    "source": spec.source_type,
                    "source_url": spec.source_url,
                }
                for spec in specs
            ]
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if not specs:
        print("no models found in registry")
        return 0
    for spec in specs:
        sha_status = "present" if spec.expected_sha256 else "missing"
        print(f"{spec.model_id}\t{spec.family}\t{spec.version}\t{spec.license}\tsha256:{sha_status}")
    return 0


def _cmd_fetch_model(args: argparse.Namespace) -> int:
    from yolozu.model_fetch import fetch_model

    try:
        model_path, meta_path = fetch_model(
            model_id=str(args.model_id),
            out_dir=str(args.out),
            cache_dir=getattr(args, "cache_dir", None),
            accept_license=bool(getattr(args, "accept_license", False)),
            allow_unsafe=bool(getattr(args, "allow_unsafe", False)),
            allow_non_apache=bool(getattr(args, "allow_non_apache", False)),
            retries=int(getattr(args, "retries", 3) or 3),
            timeout=float(getattr(args, "timeout", 60.0) or 60.0),
            registry_path=getattr(args, "registry", None),
            force=bool(getattr(args, "force", False)),
        )
    except PermissionError as exc:
        raise SystemExit(str(exc)) from exc
    except KeyError as exc:
        raise SystemExit(f"unknown model id: {exc.args[0]} (use `yolozu list models`)") from exc
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(str(model_path))
    print(str(meta_path))
    return 0


def _cmd_doctor_import(args: argparse.Namespace) -> int:
    import time

    from yolozu.coco_convert import build_category_map_from_coco
    from yolozu.dataset import build_manifest
    from yolozu.imports import (
        project_detectron2_config,
        project_mmdet_config,
        project_ultralytics_args,
        project_yolox_exp,
    )

    def _now_utc() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    report: dict[str, Any] = {
        "kind": "yolozu_doctor_import",
        "schema_version": 1,
        "timestamp": _now_utc(),
        "dataset": None,
        "config": None,
        "warnings": [],
        "errors": [],
    }

    dataset_from = getattr(args, "dataset_from", None)
    config_from = getattr(args, "config_from", None)
    if not dataset_from and not config_from:
        raise SystemExit("doctor import requires at least one of: --dataset-from, --config-from")

    if dataset_from and str(dataset_from).strip().lower() == "auto":
        dataset_from = _resolve_auto_dataset_from_args(args)
        report["warnings"].append(f"dataset source auto-detected: {dataset_from}")
    if config_from and str(config_from).strip().lower() == "auto":
        config_from = _resolve_auto_config_from_args(args)
        report["warnings"].append(f"config source auto-detected: {config_from}")

    if dataset_from:
        src = str(dataset_from)
        if src == "coco-instances":
            if not getattr(args, "instances", None) or not getattr(args, "images_dir", None):
                raise SystemExit("--instances and --images-dir are required for --dataset-from coco-instances")
            instances_path = Path(str(args.instances)).expanduser()
            if not instances_path.is_absolute():
                instances_path = Path.cwd() / instances_path
            images_dir = Path(str(args.images_dir)).expanduser()
            if not images_dir.is_absolute():
                images_dir = Path.cwd() / images_dir
            if not instances_path.exists():
                raise SystemExit(f"--instances not found: {instances_path}")
            if not images_dir.exists():
                raise SystemExit(f"--images-dir not found: {images_dir}")

            instances_doc = json.loads(instances_path.read_text(encoding="utf-8"))
            images = instances_doc.get("images") or []
            annotations = instances_doc.get("annotations") or []
            include_crowd = bool(getattr(args, "include_crowd", False))
            if not include_crowd and isinstance(annotations, list):
                annotations = [a for a in annotations if not (isinstance(a, dict) and int(a.get("iscrowd", 0) or 0) == 1)]

            cat_map = build_category_map_from_coco(instances_doc)
            categories = instances_doc.get("categories") or []
            category_ids: list[int] = []
            if isinstance(categories, list):
                for cat in categories:
                    if isinstance(cat, dict):
                        try:
                            category_ids.append(int(cat.get("id")))
                        except Exception:
                            continue
            has_category_id_zero = 0 in category_ids
            if has_category_id_zero:
                report["warnings"].append(
                    "category_id=0 detected in source categories; normalized mapping (classes.json) is required for apples-to-apples evaluation"
                )
            report["dataset"] = {
                "from": "coco-instances",
                "split": str(args.split) if getattr(args, "split", None) else None,
                "instances_json": str(instances_path),
                "images_dir": str(images_dir),
                "include_crowd": include_crowd,
                "counts": {
                    "images": int(len(images)) if isinstance(images, list) else None,
                    "annotations": int(len(annotations)) if isinstance(annotations, list) else None,
                    "classes": int(len(cat_map.class_names)),
                },
                "category_id_zero_present": bool(has_category_id_zero),
                "classes_preview": list(cat_map.class_names[:20]),
            }
        elif src == "ultralytics":
            data_yaml = getattr(args, "data", None)
            if not data_yaml:
                raise SystemExit("--data is required for --dataset-from ultralytics")
            label_format = None
            task = getattr(args, "task", None)
            if task and str(task).strip().lower() == "segment":
                label_format = "segment"
            manifest = build_manifest(
                str(data_yaml),
                split=str(args.split) if getattr(args, "split", None) else None,
                label_format=label_format,
            )
            records = list(manifest.get("images") or [])
            max_images = getattr(args, "max_images", None)
            if max_images is not None:
                records = records[: int(max_images)]
            label_count = 0
            max_class = -1
            for rec in records:
                for lab in rec.get("labels") or []:
                    label_count += 1
                    try:
                        max_class = max(max_class, int(lab.get("class_id", -1)))
                    except Exception:
                        continue
            report["dataset"] = {
                "from": "ultralytics",
                "data_yaml": str(data_yaml),
                "split": manifest.get("split"),
                "label_format": label_format,
                "counts": {
                    "images": int(len(records)),
                    "labels": int(label_count),
                    "classes_hint": int(max_class + 1) if max_class >= 0 else None,
                },
            }
        else:
            raise SystemExit(f"unsupported --dataset-from: {src}")

    if config_from:
        src = str(config_from)
        try:
            if src == "ultralytics":
                args_path = getattr(args, "args", None)
                if not args_path:
                    raise SystemExit("--args is required for --config-from ultralytics")
                p = Path(str(args_path)).expanduser()
                if not p.is_absolute():
                    p = Path.cwd() / p
                cfg = _load_config(p)
                train = project_ultralytics_args(cfg, source={"from": "ultralytics", "args_yaml": str(p)})
                report["config"] = {"from": "ultralytics", "train_config": train.to_dict()}
            elif src == "mmdet":
                cfg_path = getattr(args, "config", None)
                if not cfg_path:
                    raise SystemExit("--config is required for --config-from mmdet")
                train = project_mmdet_config(config=str(cfg_path))
                report["config"] = {"from": "mmdet", "train_config": train.to_dict()}
            elif src == "yolox":
                cfg_path = getattr(args, "config", None)
                if not cfg_path:
                    raise SystemExit("--config is required for --config-from yolox")
                train = project_yolox_exp(config=str(cfg_path))
                report["config"] = {"from": "yolox", "train_config": train.to_dict()}
            elif src == "detectron2":
                cfg_path = getattr(args, "config", None)
                if not cfg_path:
                    raise SystemExit("--config is required for --config-from detectron2")
                train = project_detectron2_config(config=str(cfg_path))
                report["config"] = {"from": "detectron2", "train_config": train.to_dict()}
            else:
                raise SystemExit(f"unsupported --config-from: {src}")
        except SystemExit:
            raise
        except Exception as exc:
            report["errors"].append(str(exc))

    output = str(getattr(args, "output", "-") or "-")
    if output == "-":
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if not report["errors"] else 2

    out_path = Path(output)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(out_path))
    return 0 if not report["errors"] else 2


def _cmd_export(args: argparse.Namespace) -> int:
    from yolozu.export import (
        DEFAULT_PREDICTIONS_PATH,
        export_dummy_predictions,
        export_labels_predictions,
        write_predictions_json,
    )

    backend = str(getattr(args, "backend", "dummy"))

    dataset = str(args.dataset)
    if not dataset:
        raise SystemExit("--dataset is required")

    try:
        if backend == "dummy":
            payload, _run = export_dummy_predictions(
                dataset_root=dataset,
                split=str(args.split) if args.split else None,
                max_images=int(args.max_images) if args.max_images is not None else None,
                score=float(args.score),
            )
        elif backend == "labels":
            payload, _run = export_labels_predictions(
                dataset_root=dataset,
                split=str(args.split) if args.split else None,
                max_images=int(args.max_images) if args.max_images is not None else None,
                score=float(args.score),
            )
        else:
            raise SystemExit(f"unsupported --backend: {backend}")
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    output = str(args.output) if args.output else DEFAULT_PREDICTIONS_PATH
    out_path = write_predictions_json(output=output, payload=payload, force=bool(args.force))
    print(str(out_path))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    if args.validate_command == "dataset":
        from yolozu.dataset import build_manifest
        from yolozu.dataset_validator import validate_dataset_records

        try:
            manifest = build_manifest(
                str(args.dataset),
                split=str(args.split) if args.split else None,
                label_format=str(getattr(args, "label_format", "")).strip() or None,
            )
        except Exception as exc:
            raise SystemExit(str(exc)) from exc
        records = manifest.get("images") or []
        if not isinstance(records, list):
            raise SystemExit("invalid dataset manifest (expected list under 'images')")
        if args.max_images is not None:
            records = records[: int(args.max_images)]

        res = validate_dataset_records(
            records,
            strict=bool(args.strict),
            mode=str(args.mode),
            check_images=not bool(args.no_check_images),
        )
        for w in res.warnings:
            print(w, file=sys.stderr)
        if res.errors:
            for e in res.errors:
                print(e, file=sys.stderr)
            return 1
        return 0

    path = Path(str(args.path))
    if not path.exists():
        raise SystemExit(f"file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"failed to parse json: {path} ({exc})") from exc

    if args.validate_command == "predictions":
        from yolozu.predictions import validate_predictions_payload

        try:
            res = validate_predictions_payload(payload, strict=bool(args.strict))
        except Exception as exc:
            raise SystemExit(str(exc)) from exc
        for w in res.warnings:
            print(w, file=sys.stderr)
        return 0

    if args.validate_command == "seg":
        from yolozu.segmentation_predictions import validate_segmentation_predictions_payload

        try:
            res = validate_segmentation_predictions_payload(payload)
        except Exception as exc:
            raise SystemExit(str(exc)) from exc
        for w in res.warnings:
            print(w, file=sys.stderr)
        return 0

    if args.validate_command == "instance-seg":
        from yolozu.instance_segmentation_predictions import (
            validate_instance_segmentation_predictions_payload,
        )

        try:
            res = validate_instance_segmentation_predictions_payload(payload)
        except Exception as exc:
            raise SystemExit(str(exc)) from exc
        for w in res.warnings:
            print(w, file=sys.stderr)
        return 0

    raise SystemExit("unknown validate command")


def _cmd_eval_instance_seg(args: argparse.Namespace) -> int:
    from yolozu.instance_segmentation_report import run_instance_segmentation_eval

    out_json, out_html = run_instance_segmentation_eval(
        dataset_root=str(args.dataset),
        split=str(args.split) if args.split else None,
        predictions=str(args.predictions),
        pred_root=str(args.pred_root) if args.pred_root else None,
        classes=str(args.classes) if args.classes else None,
        output=str(args.output),
        html=str(args.html) if args.html else None,
        title=str(args.title),
        overlays_dir=str(args.overlays_dir) if args.overlays_dir else None,
        max_overlays=int(args.max_overlays),
        overlay_sort=str(args.overlay_sort),
        overlay_max_size=int(args.overlay_max_size),
        overlay_alpha=float(args.overlay_alpha),
        min_score=float(args.min_score),
        max_images=int(args.max_images) if args.max_images is not None else None,
        diag_iou=float(args.diag_iou),
        per_image_limit=int(args.per_image_limit),
        allow_rgb_masks=bool(args.allow_rgb_masks),
    )
    print(str(out_json))
    if out_html is not None:
        print(str(out_html))
    return 0


def _cmd_onnxrt_export(args: argparse.Namespace) -> int:
    from yolozu.onnxrt_export import export_predictions_onnxrt, write_predictions_json

    try:
        payload = export_predictions_onnxrt(
            dataset_root=str(args.dataset),
            split=str(args.split) if args.split else None,
            max_images=int(args.max_images) if args.max_images is not None else None,
            onnx=str(args.onnx) if args.onnx else None,
            input_name=str(args.input_name),
            boxes_output=str(args.boxes_output),
            scores_output=str(args.scores_output),
            class_output=(str(args.class_output) if args.class_output else None),
            combined_output=(str(args.combined_output) if args.combined_output else None),
            combined_format=str(args.combined_format),
            raw_output=(str(args.raw_output) if args.raw_output else None),
            raw_format=str(args.raw_format),
            raw_postprocess=str(args.raw_postprocess),
            boxes_format=str(args.boxes_format),
            boxes_scale=str(args.boxes_scale),
            min_score=float(args.min_score),
            topk=int(args.topk),
            nms_iou=float(args.nms_iou),
            agnostic_nms=bool(args.agnostic_nms),
            imgsz=int(args.imgsz),
            dry_run=bool(args.dry_run),
            strict=bool(args.strict),
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    out_path = write_predictions_json(output=str(args.output), payload=payload, force=bool(args.force))
    print(str(out_path))
    return 0


def _cmd_onnxrt_quantize(args: argparse.Namespace) -> int:
    from yolozu.onnxrt_quantize import quantize_onnx_dynamic

    onnx_in = str(args.onnx)
    onnx_out = str(args.output)
    op_types = None
    if args.op_types:
        op_types = [t.strip() for t in str(args.op_types).split(",") if t.strip()]

    try:
        out_path = quantize_onnx_dynamic(
            onnx_in=onnx_in,
            onnx_out=onnx_out,
            weight_type=str(args.weight_type),
            per_channel=bool(args.per_channel),
            reduce_range=bool(args.reduce_range),
            op_types_to_quantize=op_types,
            use_external_data_format=bool(args.use_external_data_format),
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(str(out_path))
    return 0


def _cmd_predict_images(args: argparse.Namespace) -> int:
    from yolozu.predict_images import predict_images

    try:
        max_images = require_non_negative_int(args.max_images, flag_name="--max-images")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        out_json, out_html = predict_images(
            backend=str(args.backend),
            input_dir=str(args.input_dir),
            output=str(args.output),
            score=float(args.score),
            max_images=max_images,
            force=bool(args.force),
            glob_patterns=list(args.glob) if args.glob else None,
            overlays_dir=str(args.overlays_dir) if args.overlays_dir else None,
            html=str(args.html) if args.html else None,
            title=str(args.title),
            onnx=(str(args.onnx) if args.onnx else None),
            input_name=str(args.input_name),
            boxes_output=str(args.boxes_output),
            scores_output=str(args.scores_output),
            class_output=(str(args.class_output) if args.class_output else None),
            combined_output=(str(args.combined_output) if args.combined_output else None),
            combined_format=str(args.combined_format),
            raw_output=(str(args.raw_output) if args.raw_output else None),
            raw_format=str(args.raw_format),
            raw_postprocess=str(args.raw_postprocess),
            boxes_format=str(args.boxes_format),
            boxes_scale=str(args.boxes_scale),
            min_score=float(args.min_score),
            topk=int(args.topk),
            nms_iou=float(args.nms_iou),
            agnostic_nms=bool(args.agnostic_nms),
            imgsz=int(args.imgsz),
            dry_run=bool(args.dry_run),
            strict=bool(args.strict),
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(str(out_json))
    if out_html is not None:
        print(str(out_html))
    return 0


def _cmd_eval_coco(args: argparse.Namespace) -> int:
    import time

    from yolozu.coco_eval import build_coco_ground_truth, evaluate_coco_map, predictions_to_coco_detections
    from yolozu.dataset import build_manifest
    from yolozu.predictions import load_predictions_entries, validate_predictions_entries
    from yolozu.predictions_transform import load_classes_json, normalize_class_ids

    dataset_root = Path(str(args.dataset)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = Path.cwd() / dataset_root

    manifest = build_manifest(dataset_root, split=str(args.split) if args.split else None)
    records = list(manifest.get("images") or [])
    if args.max_images is not None:
        records = records[: int(args.max_images)]

    gt, index = build_coco_ground_truth(records)
    image_sizes = {image["id"]: (int(image["width"]), int(image["height"])) for image in gt["images"]}

    predictions_path = Path(str(args.predictions)).expanduser()
    if not predictions_path.is_absolute():
        predictions_path = Path.cwd() / predictions_path
    predictions = load_predictions_entries(predictions_path)
    normalization_warnings: list[str] = []
    if args.classes or args.assume_class_id_is_category_id:
        if not args.classes:
            raise SystemExit("--classes is required when --assume-class-id-is-category-id is enabled")
        classes = load_classes_json(Path(str(args.classes)).expanduser())
        transformed = normalize_class_ids(
            predictions,
            classes_json=classes,
            assume_class_id_is_category_id=bool(args.assume_class_id_is_category_id),
        )
        predictions = transformed.entries
        normalization_warnings = list(transformed.warnings)
    validation = validate_predictions_entries(predictions, strict=False)
    detections = predictions_to_coco_detections(
        predictions,
        coco_index=index,
        image_sizes=image_sizes,
        bbox_format=str(args.bbox_format),
    )

    if bool(args.dry_run):
        result: dict[str, object] = {
            "metrics": {"map50_95": None, "map50": None, "map75": None, "ar100": None},
            "stats": [],
            "dry_run": True,
            "counts": {"images": int(len(records)), "detections": int(len(detections))},
            "warnings": [*validation.warnings, *normalization_warnings],
        }
    else:
        result = evaluate_coco_map(gt, detections)
        result["warnings"] = [*validation.warnings, *normalization_warnings]

    payload: dict[str, object] = {
        "report_schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(dataset_root),
        "split": manifest.get("split"),
        "split_requested": str(args.split) if args.split else None,
        "predictions": str(predictions_path),
        "bbox_format": str(args.bbox_format),
        "max_images": int(args.max_images) if args.max_images is not None else None,
        "normalization": {
            "classes": str(args.classes) if args.classes else None,
            "assume_class_id_is_category_id": bool(args.assume_class_id_is_category_id),
        },
        **result,
    }

    output_path = Path(str(args.output)).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(output_path))
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    import time

    from yolozu.dataset import build_manifest
    from yolozu.export import write_predictions_json
    from yolozu.instance_segmentation_predictions import (
        normalize_instance_segmentation_predictions_payload,
        validate_instance_segmentation_predictions_entries,
    )
    from yolozu.long_tail_metrics import (
        build_fracal_stats,
        fracal_calibrate_instance_segmentation,
        fracal_calibrate_predictions,
        la_calibrate_instance_segmentation,
        la_calibrate_predictions,
        norcal_calibrate_instance_segmentation,
        norcal_calibrate_predictions,
        temperature_calibrate_instance_segmentation,
        temperature_calibrate_predictions,
    )
    from yolozu.predictions import normalize_predictions_payload, validate_predictions_entries

    method = str(getattr(args, "method", "fracal") or "fracal").strip().lower()
    if method not in ("fracal", "la", "norcal", "temperature"):
        raise SystemExit(f"unsupported calibration method: {method}")

    dataset_root = Path(str(args.dataset)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = Path.cwd() / dataset_root

    manifest = build_manifest(dataset_root, split=str(args.split) if args.split else None)
    records = list(manifest.get("images") or [])
    if args.max_images is not None:
        records = records[: int(args.max_images)]

    predictions_path = Path(str(args.predictions)).expanduser()
    if not predictions_path.is_absolute():
        predictions_path = Path.cwd() / predictions_path

    raw_data = json.loads(predictions_path.read_text(encoding="utf-8"))

    task = str(getattr(args, "task", "auto") or "auto").strip().lower()
    if task not in ("auto", "bbox", "seg", "pose"):
        raise SystemExit("--task must be one of: auto, bbox, seg, pose")

    if task == "auto":
        if isinstance(raw_data, list) and raw_data and isinstance(raw_data[0], dict) and "instances" in raw_data[0]:
            task = "seg"
        elif isinstance(raw_data, dict) and isinstance(raw_data.get("predictions"), list):
            preds = raw_data.get("predictions") or []
            first = preds[0] if preds else {}
            if isinstance(first, dict) and "instances" in first:
                task = "seg"
            elif isinstance(first, dict) and isinstance(first.get("detections"), list):
                first_det = first.get("detections", [None])[0] if first.get("detections") else None
                if isinstance(first_det, dict) and "keypoints" in first_det:
                    task = "pose"
                else:
                    task = "bbox"
            else:
                task = "bbox"
        else:
            task = "bbox"

    loaded_counts: dict[int, int] | None = None
    stats_source = "computed"
    stats_input_path = getattr(args, "stats_in", None)
    if stats_input_path:
        stats_path = Path(str(stats_input_path)).expanduser()
        if not stats_path.is_absolute():
            stats_path = Path.cwd() / stats_path
        if not stats_path.exists():
            raise SystemExit(f"stats file not found: {stats_path}")
        stats_doc = json.loads(stats_path.read_text(encoding="utf-8"))
        raw_counts = stats_doc.get("class_counts") if isinstance(stats_doc, dict) else None
        if not isinstance(raw_counts, dict):
            raise SystemExit("invalid stats file: expected object with class_counts")
        loaded_counts = {}
        for key, value in raw_counts.items():
            try:
                loaded_counts[int(key)] = int(value)
            except Exception:
                continue
        stats_source = str(stats_path)

    computed_stats = build_fracal_stats(
        records,
        task=task,
        allow_rgb_masks=bool(getattr(args, "allow_rgb_masks", False)),
        method=method,
    )
    if loaded_counts is None:
        loaded_counts = {int(k): int(v) for k, v in (computed_stats.get("class_counts") or {}).items()}

    if task == "seg":
        entries, wrapped_meta = normalize_instance_segmentation_predictions_payload(raw_data)
        validation = validate_instance_segmentation_predictions_entries(entries, where="predictions")
        if method == "fracal":
            calibrated_entries, calibration_report = fracal_calibrate_instance_segmentation(
                records,
                entries,
                alpha=float(args.alpha),
                strength=float(args.strength),
                min_score=(None if args.min_score is None else float(args.min_score)),
                max_score=(None if args.max_score is None else float(args.max_score)),
                class_counts=loaded_counts,
                allow_rgb_masks=bool(getattr(args, "allow_rgb_masks", False)),
            )
        elif method == "la":
            calibrated_entries, calibration_report = la_calibrate_instance_segmentation(
                records,
                entries,
                tau=float(args.tau),
                min_score=(None if args.min_score is None else float(args.min_score)),
                max_score=(None if args.max_score is None else float(args.max_score)),
                class_counts=loaded_counts,
                allow_rgb_masks=bool(getattr(args, "allow_rgb_masks", False)),
            )
        else:
            if method == "norcal":
                calibrated_entries, calibration_report = norcal_calibrate_instance_segmentation(
                    records,
                    entries,
                    gamma=float(args.gamma),
                    min_score=(None if args.min_score is None else float(args.min_score)),
                    max_score=(None if args.max_score is None else float(args.max_score)),
                    class_counts=loaded_counts,
                    allow_rgb_masks=bool(getattr(args, "allow_rgb_masks", False)),
                )
            else:
                calibrated_entries, calibration_report = temperature_calibrate_instance_segmentation(
                    records,
                    entries,
                    temperature=float(args.temperature),
                    min_score=(None if args.min_score is None else float(args.min_score)),
                    max_score=(None if args.max_score is None else float(args.max_score)),
                )
    else:
        entries, wrapped_meta = normalize_predictions_payload(raw_data)
        validation = validate_predictions_entries(entries, strict=False)
        if method == "fracal":
            calibrated_entries, calibration_report = fracal_calibrate_predictions(
                records,
                entries,
                alpha=float(args.alpha),
                strength=float(args.strength),
                min_score=(None if args.min_score is None else float(args.min_score)),
                max_score=(None if args.max_score is None else float(args.max_score)),
                class_counts=loaded_counts,
            )
        elif method == "la":
            calibrated_entries, calibration_report = la_calibrate_predictions(
                records,
                entries,
                tau=float(args.tau),
                min_score=(None if args.min_score is None else float(args.min_score)),
                max_score=(None if args.max_score is None else float(args.max_score)),
                class_counts=loaded_counts,
            )
        else:
            if method == "norcal":
                calibrated_entries, calibration_report = norcal_calibrate_predictions(
                    records,
                    entries,
                    gamma=float(args.gamma),
                    min_score=(None if args.min_score is None else float(args.min_score)),
                    max_score=(None if args.max_score is None else float(args.max_score)),
                    class_counts=loaded_counts,
                )
            else:
                temp_grid = None
                if getattr(args, "temperature_grid", None):
                    raw = str(args.temperature_grid)
                    temp_grid = []
                    for part in raw.split(","):
                        part = part.strip()
                        if not part:
                            continue
                        try:
                            temp_grid.append(float(part))
                        except Exception:
                            continue
                calibrated_entries, calibration_report = temperature_calibrate_predictions(
                    records,
                    entries,
                    temperature=float(args.temperature),
                    fit_temperature=bool(getattr(args, "fit_temperature", False)),
                    temperature_grid=temp_grid,
                    fit_iou=float(getattr(args, "fit_iou", 0.5)),
                    max_detections=int(getattr(args, "fit_max_detections", 100)),
                    min_score=(None if args.min_score is None else float(args.min_score)),
                    max_score=(None if args.max_score is None else float(args.max_score)),
                )
    calibration_report["task"] = task
    calibration_report["stats_source"] = stats_source

    out_meta: dict[str, Any] = dict(wrapped_meta or {})
    out_meta["posthoc_calibration"] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": method,
        "report": calibration_report,
    }

    payload = {
        "schema_version": 1,
        "predictions": calibrated_entries,
        "meta": out_meta,
    }

    if task == "seg":
        out_path = Path(str(args.output)).expanduser()
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not bool(args.force):
            raise SystemExit(f"output exists: {out_path} (use --force to overwrite)")
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        out_path = write_predictions_json(output=str(args.output), payload=payload, force=bool(args.force))

    stats_out = getattr(args, "stats_out", None)
    if stats_out:
        stats_payload = dict(computed_stats)
        stats_payload["task"] = task
        stats_payload["used_class_counts"] = {str(k): int(v) for k, v in sorted((loaded_counts or {}).items())}
        stats_out_path = Path(str(stats_out)).expanduser()
        if not stats_out_path.is_absolute():
            stats_out_path = Path.cwd() / stats_out_path
        stats_out_path.parent.mkdir(parents=True, exist_ok=True)
        if stats_out_path.exists() and not bool(args.force):
            raise SystemExit(f"output exists: {stats_out_path} (use --force to overwrite)")
        stats_out_path.write_text(json.dumps(stats_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    report_payload = {
        "report_schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(dataset_root),
        "split": manifest.get("split"),
        "predictions": str(predictions_path),
        "output": str(out_path),
        "method": method,
        "task": task,
        "stats_source": stats_source,
        "warnings": list(validation.warnings),
        "calibration": calibration_report,
    }
    report_path = Path(str(args.output_report)).expanduser()
    if not report_path.is_absolute():
        report_path = Path.cwd() / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists() and not bool(args.force):
        raise SystemExit(f"output exists: {report_path} (use --force to overwrite)")
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    print(str(out_path))
    return 0


def _cmd_eval_long_tail(args: argparse.Namespace) -> int:
    import time

    from yolozu.dataset import build_manifest
    from yolozu.long_tail_metrics import evaluate_long_tail_detection
    from yolozu.predictions import load_predictions_entries, validate_predictions_entries

    dataset_root = Path(str(args.dataset)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = Path.cwd() / dataset_root

    manifest = build_manifest(dataset_root, split=str(args.split) if args.split else None)
    records = list(manifest.get("images") or [])
    if args.max_images is not None:
        records = records[: int(args.max_images)]

    predictions_path = Path(str(args.predictions)).expanduser()
    if not predictions_path.is_absolute():
        predictions_path = Path.cwd() / predictions_path
    predictions = load_predictions_entries(predictions_path)
    validation = validate_predictions_entries(predictions, strict=False)

    metrics = evaluate_long_tail_detection(
        records,
        predictions,
        max_detections=int(args.max_detections),
        head_fraction=float(args.head_fraction),
        medium_fraction=float(args.medium_fraction),
        calibration_bins=int(args.calibration_bins),
        calibration_iou=float(args.calibration_iou),
    )

    payload: dict[str, Any] = {
        "report_schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(dataset_root),
        "split": manifest.get("split"),
        "split_requested": str(args.split) if args.split else None,
        "predictions": str(predictions_path),
        "max_images": int(args.max_images) if args.max_images is not None else None,
        "warnings": list(validation.warnings),
        **metrics,
    }

    output_path = Path(str(args.output)).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(output_path))
    return 0


def _cmd_long_tail_recipe(args: argparse.Namespace) -> int:
    import time

    from yolozu.dataset import build_manifest
    from yolozu.long_tail_recipe import build_long_tail_recipe

    dataset_root = Path(str(args.dataset)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = Path.cwd() / dataset_root

    manifest = build_manifest(dataset_root, split=str(args.split) if args.split else None)
    records = list(manifest.get("images") or [])
    if args.max_images is not None:
        records = records[: int(args.max_images)]

    recipe = build_long_tail_recipe(
        records,
        seed=int(args.seed),
        stage1_epochs=int(args.stage1_epochs),
        stage2_epochs=int(args.stage2_epochs),
        rebalance_sampler=str(args.rebalance_sampler),
        loss_plugin=str(args.loss_plugin),
        metric_plugin=str(args.metric_plugin),
        lr_scheduler=str(args.lr_scheduler),
        logit_adjustment_tau=float(args.logit_adjustment_tau),
        lort_tau=float(args.lort_tau),
        class_balanced_beta=float(args.class_balanced_beta),
        focal_gamma=float(args.focal_gamma),
        ldam_margin=float(args.ldam_margin),
    )

    payload: dict[str, Any] = {
        "report_schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(dataset_root),
        "split": manifest.get("split"),
        "split_requested": str(args.split) if args.split else None,
        "max_images": int(args.max_images) if args.max_images is not None else None,
        "recipe": recipe,
    }

    output_path = Path(str(args.output)).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not bool(args.force):
        raise SystemExit(f"output exists: {output_path} (use --force to overwrite)")
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(output_path))
    return 0


def _cmd_parity(args: argparse.Namespace) -> int:
    from yolozu.predictions_parity import compare_predictions

    try:
        max_images = require_non_negative_int(args.max_images, flag_name="--max-images")
        image_size = parse_image_size_arg(args.image_size, flag_name="--image-size")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    report = compare_predictions(
        reference=str(args.reference),
        candidate=str(args.candidate),
        image_size=image_size,
        max_images=max_images,
        iou_thresh=float(args.iou_thresh),
        score_atol=float(args.score_atol),
        bbox_atol=float(args.bbox_atol),
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if bool(report.get("ok")) else 2


def _cmd_resources(args: argparse.Namespace) -> int:
    from yolozu import resources

    if args.resources_command == "list":
        for p in resources.list_resource_paths():
            print(p)
        return 0

    if args.resources_command == "cat":
        text = resources.read_text(str(args.path))
        print(text, end="" if text.endswith("\n") else "\n")
        return 0

    if args.resources_command == "copy":
        out = resources.copy_to(str(args.path), output=str(args.output), force=bool(args.force))
        print(str(out))
        return 0

    raise SystemExit("unknown resources command")


def _cmd_migrate(args: argparse.Namespace) -> int:
    from yolozu.migrate import (
        migrate_coco_dataset_wrapper,
        migrate_coco_results_predictions,
        migrate_seg_dataset_descriptor,
        migrate_ultralytics_dataset_wrapper,
    )

    if args.migrate_command == "dataset":
        if str(args.from_format) == "ultralytics":
            out = migrate_ultralytics_dataset_wrapper(
                data_yaml=str(args.data) if args.data else None,
                args_yaml=str(args.args) if args.args else None,
                split=str(args.split) if args.split else None,
                task=str(args.task) if args.task else None,
                output=str(args.output),
                force=bool(args.force),
            )
        elif str(args.from_format) == "coco":
            if not args.coco_root:
                raise SystemExit("--coco-root is required for --from coco")
            split = str(args.split) if args.split else "val2017"
            out = migrate_coco_dataset_wrapper(
                coco_root=str(args.coco_root),
                split=split,
                instances_json=(str(args.instances_json) if args.instances_json else None),
                output=str(args.output),
                mode=str(args.mode),
                include_crowd=bool(args.include_crowd),
                force=bool(args.force),
            )
        else:
            raise SystemExit("unsupported --from for migrate dataset")
        print(str(out))
        return 0

    if args.migrate_command == "predictions":
        if str(args.from_format) != "coco-results":
            raise SystemExit("unsupported --from for migrate predictions")
        out = migrate_coco_results_predictions(
            results_json=str(args.results),
            instances_json=str(args.instances),
            output=str(args.output),
            score_threshold=float(args.score_threshold),
            force=bool(args.force),
        )
        print(str(out))
        return 0

    if args.migrate_command == "seg-dataset":
        out = migrate_seg_dataset_descriptor(
            from_format=str(args.from_format),
            root=str(args.root),
            split=str(args.split),
            output=str(args.output),
            path_type=str(args.path_type),
            mode=str(args.mode),
            force=bool(args.force),
            voc_year=str(args.year) if args.year else None,
            voc_masks_dirname=str(args.masks_dirname),
            cityscapes_label_type=str(args.label_type),
        )
        print(str(out))
        return 0

    raise SystemExit("unknown migrate command")


def _cmd_import(args: argparse.Namespace) -> int:
    from yolozu.imports import (
        import_coco_instances_dataset,
        import_detectron2_config,
        import_mmdet_config,
        import_ultralytics_config,
        import_yolox_config,
    )
    from yolozu.migrate import migrate_ultralytics_dataset_wrapper

    try:
        if args.import_command == "dataset":
            from_format = str(args.from_format)
            if from_format == "auto":
                from_format = _resolve_auto_dataset_from_args(args)

            if from_format == "ultralytics":
                out = migrate_ultralytics_dataset_wrapper(
                    data_yaml=str(args.data) if args.data else None,
                    args_yaml=str(args.args) if args.args else None,
                    split=str(args.split) if args.split else None,
                    task=str(args.task) if args.task else None,
                    output=str(args.output),
                    force=bool(args.force),
                )
                print(str(out))
                return 0

            if from_format == "coco-instances":
                if not args.instances or not args.images_dir:
                    raise SystemExit("--instances and --images-dir are required for --from coco-instances")
                out = import_coco_instances_dataset(
                    instances_json=str(args.instances),
                    images_dir=str(args.images_dir),
                    split=str(args.split) if args.split else "val2017",
                    output=str(args.output),
                    include_crowd=bool(args.include_crowd),
                    force=bool(args.force),
                )
                print(str(out))
                return 0

            raise SystemExit("unsupported --from for import dataset")

        if args.import_command == "config":
            from_format = str(args.from_format)
            if from_format == "auto":
                from_format = _resolve_auto_config_from_args(args)
            if from_format == "ultralytics":
                if not args.args:
                    raise SystemExit("--args is required for --from ultralytics")
                out = import_ultralytics_config(
                    args_yaml=str(args.args),
                    output=str(args.output),
                    force=bool(args.force),
                )
            elif from_format == "mmdet":
                if not args.config:
                    raise SystemExit("--config is required for --from mmdet")
                out = import_mmdet_config(
                    config=str(args.config),
                    output=str(args.output),
                    force=bool(args.force),
                )
            elif from_format == "yolox":
                if not args.config:
                    raise SystemExit("--config is required for --from yolox")
                out = import_yolox_config(
                    config=str(args.config),
                    output=str(args.output),
                    force=bool(args.force),
                )
            elif from_format == "detectron2":
                if not args.config:
                    raise SystemExit("--config is required for --from detectron2")
                out = import_detectron2_config(
                    config=str(args.config),
                    output=str(args.output),
                    force=bool(args.force),
                )
            else:
                raise SystemExit("unsupported --from for import config")
            print(str(out))
            return 0

        raise SystemExit("unknown import command")
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


__all__ = [name for name in globals().keys() if not name.startswith("__")]
