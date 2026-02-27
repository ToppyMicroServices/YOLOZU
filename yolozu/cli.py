from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from yolozu import __version__

from .cli_args import parse_image_size_arg, require_non_negative_int
from .config import simple_yaml_load


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yolozu",
        epilog=(
            "© 2026 ToppyMicroServices OÜ\n"
            "Legal address: Karamelli tn 2, 11317 Tallinn, Harju County, Estonia\n"
            "Registry code: 16551297\n"
            "Contact: develop@toppymicros.com"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Print environment diagnostics as JSON.")
    doctor.add_argument("--output", default="reports/doctor.json", help="Output JSON path (use - for stdout).")
    doctor_sub = doctor.add_subparsers(dest="doctor_command", required=False)
    doctor_imp = doctor_sub.add_parser("import", help="Summarize dataset/config import resolution (宣伝用).")
    doctor_imp.add_argument("--output", default="-", help="Output JSON path (use - for stdout).")
    doctor_imp.add_argument(
        "--dataset-from",
        choices=("auto", "ultralytics", "coco-instances"),
        default=None,
        help="Optional dataset import adapter to summarize.",
    )
    doctor_imp.add_argument(
        "--config-from",
        choices=("auto", "ultralytics", "mmdet", "yolox", "detectron2"),
        default=None,
        help="Optional config import adapter to summarize.",
    )
    doctor_imp.add_argument("--data", default=None, help="(dataset-from ultralytics) data.yaml path.")
    doctor_imp.add_argument("--args", default=None, help="(config-from ultralytics) args.yaml path.")
    doctor_imp.add_argument("--task", choices=("detect", "segment", "pose"), default=None, help="(dataset-from ultralytics) Task override.")
    doctor_imp.add_argument("--split", default=None, help="Split name (e.g. val2017/val/train).")
    doctor_imp.add_argument("--max-images", type=int, default=200, help="Cap number of samples loaded for summary (default: 200).")
    doctor_imp.add_argument("--instances", default=None, help="(dataset-from coco-instances) instances_*.json path.")
    doctor_imp.add_argument("--images-dir", default=None, help="(dataset-from coco-instances) images directory for this split.")
    doctor_imp.add_argument("--include-crowd", action="store_true", help="(dataset-from coco-instances) Include iscrowd annotations.")
    doctor_imp.add_argument("--config", default=None, help="(config-from mmdet/yolox/detectron2) config file path.")

    list_p = sub.add_parser("list", help="List registries and built-in catalogs.")
    list_sub = list_p.add_subparsers(dest="list_command", required=True)
    list_models = list_sub.add_parser("models", help="List fetchable model IDs.")
    list_models.add_argument("--registry", default=None, help="Optional registry JSON path (default: packaged model zoo).")
    list_models.add_argument("--json", action="store_true", help="Emit JSON.")

    fetch = sub.add_parser("fetch", help="Download a model artifact from the built-in (or custom) model registry.")
    fetch.add_argument("model_id", help="Model id from `yolozu list models`.")
    fetch.add_argument("--out", default="models", help="Output root directory (default: models).")
    fetch.add_argument("--cache-dir", default=None, help="Cache directory (default: ~/.cache/yolozu/models).")
    fetch.add_argument("--registry", default=None, help="Optional registry JSON path override.")
    fetch.add_argument("--accept-license", action="store_true", help="Required acknowledgment to download the selected model.")
    fetch.add_argument("--allow-unsafe", action="store_true", help="Allow fetching models without sha256 in the registry.")
    fetch.add_argument("--allow-non-apache", action="store_true", help="Allow fetching models with non-Apache-friendly licenses.")
    fetch.add_argument("--retries", type=int, default=3, help="Download retry count (default: 3).")
    fetch.add_argument("--timeout", type=float, default=60.0, help="Download timeout in seconds (default: 60).")
    fetch.add_argument("--force", action="store_true", help="Re-download to cache and overwrite output artifact.")

    export = sub.add_parser("export", help="Export predictions.json artifacts.")
    export.add_argument(
        "--backend",
        choices=("dummy", "labels"),
        default="dummy",
        help="Export backend (dummy=1 det/image; labels=use dataset labels).",
    )
    export.add_argument("--dataset", default="data/coco128", help="YOLO-format dataset root.")
    export.add_argument("--split", default=None, help="Dataset split under images/ and labels/ (default: auto).")
    export.add_argument("--max-images", type=int, default=50, help="Optional cap for number of images.")
    export.add_argument("--score", type=float, default=0.9, help="Score to assign to exported detections (default: 0.9).")
    export.add_argument(
        "--output",
        default=None,
        help="Predictions JSON output path (default: reports/predictions.json).",
    )
    export.add_argument("--force", action="store_true", help="Overwrite outputs if they exist.")

    predict = sub.add_parser("predict-images", help="Run folder inference and write predictions JSON + overlays + HTML.")
    predict.add_argument("--backend", choices=("dummy", "onnxrt"), default="dummy", help="Inference backend.")
    predict.add_argument("--input-dir", required=True, help="Input directory containing images.")
    predict.add_argument("--glob", action="append", default=None, help="Glob pattern(s) under input dir (repeatable).")
    predict.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")
    predict.add_argument("--score", type=float, default=0.9, help="Dummy score when --backend=dummy.")
    predict.add_argument("--output", default="reports/predict_images.json", help="Predictions JSON output path.")
    predict.add_argument("--force", action="store_true", help="Overwrite outputs if they exist.")
    predict.add_argument("--overlays-dir", default="reports/overlays", help="Overlay images output directory.")
    predict.add_argument("--html", default="reports/predict_images.html", help="Optional HTML report path.")
    predict.add_argument("--title", default="YOLOZU predict-images report", help="HTML title.")
    predict.add_argument("--onnx", default=None, help="Path to ONNX model (required for --backend onnxrt unless --dry-run).")
    predict.add_argument("--input-name", default="images", help="ONNX input name (default: images).")
    predict.add_argument("--boxes-output", default="boxes", help="Output name for boxes tensor (default: boxes).")
    predict.add_argument("--scores-output", default="scores", help="Output name for scores tensor (default: scores).")
    predict.add_argument("--class-output", default=None, help="Optional output name for class_id tensor.")
    predict.add_argument("--combined-output", default=None, help="Optional output name for [x1,y1,x2,y2,score,class_id].")
    predict.add_argument("--combined-format", choices=("xyxy_score_class",), default="xyxy_score_class")
    predict.add_argument("--raw-output", default=None, help="Optional output name for raw head output.")
    predict.add_argument("--raw-format", choices=("yolo_84",), default="yolo_84")
    predict.add_argument("--raw-postprocess", choices=("native", "ultralytics"), default="native")
    predict.add_argument("--boxes-format", choices=("xyxy",), default="xyxy")
    predict.add_argument("--boxes-scale", choices=("abs", "norm"), default="norm")
    predict.add_argument("--min-score", type=float, default=0.001, help="Score threshold (default: 0.001).")
    predict.add_argument("--topk", type=int, default=300, help="Top-K detections per image (default: 300).")
    predict.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU for raw output decode (default: 0.7).")
    predict.add_argument("--agnostic-nms", action="store_true", help="Class-agnostic NMS for raw output decode.")
    predict.add_argument("--imgsz", type=int, default=640, help="Input image size (square, default: 640).")
    predict.add_argument("--dry-run", action="store_true", help="Write schema-correct JSON without running inference.")
    predict.add_argument("--strict", action="store_true", help="Strict prediction schema validation before writing.")

    eval_coco = sub.add_parser("eval-coco", help="Evaluate detections with COCOeval (optional extra: yolozu[coco]).")
    eval_coco.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    eval_coco.add_argument("--split", default=None, help="Dataset split under images/ and labels/ (default: auto).")
    eval_coco.add_argument("--predictions", required=True, help="Predictions JSON path.")
    eval_coco.add_argument(
        "--bbox-format",
        choices=("cxcywh_norm", "cxcywh_abs", "xywh_abs", "xyxy_abs"),
        default="cxcywh_norm",
        help="How to interpret detection bbox fields (default: cxcywh_norm).",
    )
    eval_coco.add_argument("--dry-run", action="store_true", help="Skip COCOeval; only validate/convert predictions.")
    eval_coco.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")
    eval_coco.add_argument("--classes", default=None, help="Optional labels/<split>/classes.json for class-id normalization.")
    eval_coco.add_argument(
        "--assume-class-id-is-category-id",
        action="store_true",
        help="Treat class_id in predictions as COCO category_id when --classes is set.",
    )
    eval_coco.add_argument("--output", default="reports/coco_eval.json", help="Output report path.")

    calibrate = sub.add_parser("calibrate", help="Apply post-hoc FRACAL calibration to bbox or instance-seg predictions JSON.")
    calibrate.add_argument("--method", choices=("fracal", "la", "norcal", "temperature"), default="fracal", help="Calibration method (default: fracal).")
    calibrate.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    calibrate.add_argument("--split", default=None, help="Dataset split under images/ and labels/ (default: auto).")
    calibrate.add_argument("--task", choices=("auto", "bbox", "seg", "pose"), default="auto", help="Prediction task type (default: auto).")
    calibrate.add_argument("--predictions", required=True, help="Input predictions JSON path.")
    calibrate.add_argument("--output", default="reports/predictions_calibrated.json", help="Output calibrated predictions JSON path.")
    calibrate.add_argument("--output-report", default="reports/calibration_fracal_report.json", help="Output calibration report JSON path.")
    calibrate.add_argument("--stats-in", default=None, help="Optional precomputed FRACAL stats JSON path (class_counts).")
    calibrate.add_argument("--stats-out", default=None, help="Optional output path to write computed FRACAL stats JSON.")
    calibrate.add_argument("--max-images", type=int, default=None, help="Optional cap for calibration/eval subset size.")
    calibrate.add_argument("--alpha", type=float, default=0.5, help="FRACAL class-frequency exponent (default: 0.5).")
    calibrate.add_argument("--strength", type=float, default=1.0, help="Blend ratio [0,1] between original and FRACAL scores (default: 1.0).")
    calibrate.add_argument("--tau", type=float, default=1.0, help="(method=la) logit adjustment coefficient tau (default: 1.0).")
    calibrate.add_argument("--gamma", type=float, default=1.0, help="(method=norcal) frequency exponent gamma (default: 1.0).")
    calibrate.add_argument("--temperature", type=float, default=1.0, help="(method=temperature) global temperature T (>0, default: 1.0).")
    calibrate.add_argument("--fit-temperature", action="store_true", help="(method=temperature, bbox/pose) fit T on validation subset by minimizing binary NLL.")
    calibrate.add_argument("--temperature-grid", default="0.5,0.75,1.0,1.25,1.5,2.0", help="(method=temperature) candidate T values for --fit-temperature.")
    calibrate.add_argument("--fit-iou", type=float, default=0.5, help="(method=temperature) IoU threshold for positive matching in T fitting.")
    calibrate.add_argument("--fit-max-detections", type=int, default=100, help="(method=temperature) max detections/image used when fitting T.")
    calibrate.add_argument("--min-score", type=float, default=None, help="Optional post-clamp minimum score.")
    calibrate.add_argument("--max-score", type=float, default=None, help="Optional post-clamp maximum score.")
    calibrate.add_argument("--allow-rgb-masks", action="store_true", help="(task=seg) Treat RGB masks as valid by using channel-0 as foreground.")
    calibrate.add_argument("--force", action="store_true", help="Overwrite outputs if they exist.")

    eval_lt = sub.add_parser("eval-long-tail", help="Evaluate long-tail detection metrics in one standardized report.")
    eval_lt.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    eval_lt.add_argument("--split", default=None, help="Dataset split under images/ and labels/ (default: auto).")
    eval_lt.add_argument("--predictions", required=True, help="Predictions JSON path.")
    eval_lt.add_argument("--output", default="reports/long_tail_eval.json", help="Output long-tail report JSON path.")
    eval_lt.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")
    eval_lt.add_argument("--max-detections", type=int, default=100, help="Max detections per image for AR/calibration matching.")
    eval_lt.add_argument("--head-fraction", type=float, default=0.33, help="Top class fraction assigned to head bin.")
    eval_lt.add_argument("--medium-fraction", type=float, default=0.67, help="Top class fraction assigned up to medium bin.")
    eval_lt.add_argument("--calibration-bins", type=int, default=10, help="Bin count for calibration metrics (ECE/confidence bias).")
    eval_lt.add_argument("--calibration-iou", type=float, default=0.5, help="IoU threshold for calibration correctness matching.")

    lt_recipe = sub.add_parser("long-tail-recipe", help="Generate a decoupled long-tail training recipe with plugin-style rebalance config.")
    lt_recipe.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    lt_recipe.add_argument("--split", default=None, help="Dataset split under images/ and labels/ (default: auto).")
    lt_recipe.add_argument("--output", default="reports/long_tail_recipe.json", help="Output recipe JSON path.")
    lt_recipe.add_argument("--max-images", type=int, default=None, help="Optional cap for recipe stat scan.")
    lt_recipe.add_argument("--seed", type=int, default=0, help="Seed recorded in recipe for reproducibility.")
    lt_recipe.add_argument("--stage1-epochs", type=int, default=90, help="Representation learning stage epochs.")
    lt_recipe.add_argument("--stage2-epochs", type=int, default=30, help="Classifier re-training stage epochs.")
    lt_recipe.add_argument("--rebalance-sampler", choices=("none", "class_balanced"), default="class_balanced", help="Sampler plugin selection.")
    lt_recipe.add_argument("--loss-plugin", choices=("none", "focal", "ldam"), default="focal", help="Loss plugin selection.")
    lt_recipe.add_argument("--logit-adjustment-tau", type=float, default=1.0, help="Logit adjustment strength (0 disables).")
    lt_recipe.add_argument("--lort-tau", type=float, default=0.0, help="Frequency-free logits retargeting strength (0 disables).")
    lt_recipe.add_argument("--class-balanced-beta", type=float, default=0.999, help="Effective-number beta for class-balanced weights.")
    lt_recipe.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma (recipe parameter).")
    lt_recipe.add_argument("--ldam-margin", type=float, default=0.5, help="LDAM margin (recipe parameter).")
    lt_recipe.add_argument("--force", action="store_true", help="Overwrite output if it exists.")

    parity = sub.add_parser("parity", help="Compare two predictions JSON artifacts for backend parity.")
    parity.add_argument("--reference", required=True, help="Reference predictions JSON (e.g. PyTorch).")
    parity.add_argument("--candidate", required=True, help="Candidate predictions JSON (e.g. ONNXRuntime).")
    parity.add_argument("--iou-thresh", type=float, default=0.99, help="IoU threshold for a match.")
    parity.add_argument("--score-atol", type=float, default=1e-4, help="Absolute tolerance for score differences.")
    parity.add_argument("--bbox-atol", type=float, default=1e-4, help="Absolute tolerance for bbox differences.")
    parity.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")
    parity.add_argument("--image-size", default=None, help="Optional fixed image size (N or W,H) to skip image reads.")

    validate = sub.add_parser("validate", help="Validate artifacts (predictions JSON, instance-seg predictions).")
    validate_sub = validate.add_subparsers(dest="validate_command", required=True)
    val_pred = validate_sub.add_parser("predictions", help="Validate predictions JSON (detections+bbox schema).")
    val_pred.add_argument("path", type=str, help="Path to predictions JSON (list or wrapper).")
    val_pred.add_argument("--strict", action="store_true", help="Strict validation (types, required keys).")

    val_seg = validate_sub.add_parser("seg", help="Validate semantic segmentation predictions JSON (id->mask mapping).")
    val_seg.add_argument("path", type=str, help="Path to segmentation predictions JSON.")

    val_is = validate_sub.add_parser("instance-seg", help="Validate instance-segmentation predictions JSON (PNG masks).")
    val_is.add_argument("path", type=str, help="Path to instance-seg predictions JSON.")

    val_ds = validate_sub.add_parser("dataset", help="Validate a YOLO-format dataset layout + labels.")
    val_ds.add_argument("dataset", type=str, help="YOLO-format dataset root (contains images/ + labels/).")
    val_ds.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    val_ds.add_argument(
        "--label-format",
        choices=("detect", "segment"),
        default=None,
        help="How to parse label txt files (default: detect). Use segment for YOLO polygon labels.",
    )
    val_ds.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images checked.")
    val_ds.add_argument("--strict", action="store_true", help="Strict bbox checks (range + inside-image).")
    val_ds.add_argument("--mode", choices=("fail", "warn"), default="fail", help="fail=exit nonzero on errors; warn=always exit 0.")
    val_ds.add_argument("--no-check-images", action="store_true", help="Skip image existence/size checks.")

    eis = sub.add_parser(
        "eval-instance-seg",
        help="Evaluate instance segmentation predictions (mask mAP over PNG masks).",
    )
    eis.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    eis.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    eis.add_argument("--predictions", required=True, help="Instance segmentation predictions JSON.")
    eis.add_argument("--pred-root", default=None, help="Optional root to resolve relative prediction mask paths.")
    eis.add_argument("--classes", default=None, help="Optional classes.txt/classes.json for class_id→name.")
    eis.add_argument("--output", default="reports/instance_seg_eval.json", help="Output JSON report path.")
    eis.add_argument("--html", default=None, help="Optional HTML report path.")
    eis.add_argument("--title", default="YOLOZU instance segmentation eval report", help="HTML title.")
    eis.add_argument("--overlays-dir", default=None, help="Optional directory to write overlay images for HTML.")
    eis.add_argument("--max-overlays", type=int, default=0, help="Max overlays to render (default: 0).")
    eis.add_argument(
        "--overlay-sort",
        choices=("worst", "best", "first"),
        default="worst",
        help="How to select overlay samples (default: worst).",
    )
    eis.add_argument("--overlay-max-size", type=int, default=768, help="Max size (max(H,W)) for overlay images (default: 768).")
    eis.add_argument("--overlay-alpha", type=float, default=0.5, help="Mask overlay alpha (default: 0.5).")
    eis.add_argument("--min-score", type=float, default=0.0, help="Minimum score threshold for predictions (default: 0.0).")
    eis.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images to evaluate.")
    eis.add_argument("--diag-iou", type=float, default=0.5, help="IoU threshold used for per-image diagnostics/overlay selection (default: 0.5).")
    eis.add_argument("--per-image-limit", type=int, default=100, help="How many per-image rows to store in the report/meta and HTML (default: 100).")
    eis.add_argument(
        "--allow-rgb-masks",
        action="store_true",
        help="Allow 3-channel masks (uses channel 0; intended for grayscale stored as RGB).",
    )

    onnxrt = sub.add_parser("onnxrt", help="ONNXRuntime utilities (optional extra: yolozu[onnxrt]).")
    onnxrt_sub = onnxrt.add_subparsers(dest="onnxrt_command", required=True)
    onnxrt_export = onnxrt_sub.add_parser("export", help="Run ONNXRuntime inference and export predictions JSON.")
    onnxrt_export.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    onnxrt_export.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    onnxrt_export.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    onnxrt_export.add_argument("--onnx", default=None, help="Path to ONNX model (required unless --dry-run).")
    onnxrt_export.add_argument("--input-name", default="images", help="ONNX input name (default: images).")
    onnxrt_export.add_argument("--boxes-output", default="boxes", help="Output name for boxes tensor (default: boxes).")
    onnxrt_export.add_argument("--scores-output", default="scores", help="Output name for scores tensor (default: scores).")
    onnxrt_export.add_argument("--class-output", default=None, help="Optional output name for class_id tensor (default: none).")
    onnxrt_export.add_argument(
        "--combined-output",
        default=None,
        help="Optional single output name with (N,6) or (1,N,6) entries [x1,y1,x2,y2,score,class_id].",
    )
    onnxrt_export.add_argument(
        "--combined-format",
        choices=("xyxy_score_class",),
        default="xyxy_score_class",
        help="Layout for --combined-output (default: xyxy_score_class).",
    )
    onnxrt_export.add_argument(
        "--raw-output",
        default=None,
        help="Optional single output name with raw head output (e.g., 1x84x8400) to decode + NMS.",
    )
    onnxrt_export.add_argument(
        "--raw-format",
        choices=("yolo_84",),
        default="yolo_84",
        help="Layout for --raw-output (default: yolo_84).",
    )
    onnxrt_export.add_argument(
        "--raw-postprocess",
        choices=("native", "ultralytics"),
        default="native",
        help="Postprocess for --raw-output (default: native).",
    )
    onnxrt_export.add_argument(
        "--boxes-format",
        choices=("xyxy",),
        default="xyxy",
        help="Box layout produced by the model in input-image space (default: xyxy).",
    )
    onnxrt_export.add_argument(
        "--boxes-scale",
        choices=("abs", "norm"),
        default="norm",
        help="Whether boxes are in pixels (abs) or normalized [0,1] wrt input_size (default: norm).",
    )
    onnxrt_export.add_argument("--min-score", type=float, default=0.001, help="Score threshold (no NMS).")
    onnxrt_export.add_argument("--topk", type=int, default=300, help="Keep top-K detections per image (no NMS).")
    onnxrt_export.add_argument("--nms-iou", type=float, default=0.7, help="IoU threshold for NMS when decoding raw output.")
    onnxrt_export.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Use class-agnostic NMS when decoding raw output.",
    )
    onnxrt_export.add_argument("--imgsz", type=int, default=640, help="Input image size (square, default: 640).")
    onnxrt_export.add_argument("--output", default="reports/predictions_onnxrt.json", help="Where to write predictions JSON.")
    onnxrt_export.add_argument("--force", action="store_true", help="Overwrite outputs if they exist.")
    onnxrt_export.add_argument("--dry-run", action="store_true", help="Write schema-correct JSON without running inference.")
    onnxrt_export.add_argument("--strict", action="store_true", help="Strict prediction schema validation before writing.")

    onnxrt_quant = onnxrt_sub.add_parser("quantize", help="Quantize an ONNX model using ONNXRuntime (dynamic).")
    onnxrt_quant.add_argument("--onnx", required=True, help="Input ONNX model path.")
    onnxrt_quant.add_argument("--output", required=True, help="Output ONNX model path.")
    onnxrt_quant.add_argument(
        "--weight-type",
        choices=("qint8", "quint8"),
        default="qint8",
        help="Weight quantization type (default: qint8).",
    )
    onnxrt_quant.add_argument("--per-channel", action="store_true", help="Quantize weights per channel.")
    onnxrt_quant.add_argument("--reduce-range", action="store_true", help="Use 7-bit quantization for weights.")
    onnxrt_quant.add_argument("--op-types", default=None, help="Comma-separated operator types to quantize (default: all supported).")
    onnxrt_quant.add_argument("--use-external-data-format", action="store_true", help="Write weights as external data (>2GB models).")

    resources_p = sub.add_parser("resources", help="Access packaged configs/schemas/protocols.")
    resources_sub = resources_p.add_subparsers(dest="resources_command", required=True)
    resources_sub.add_parser("list", help="List packaged resource paths.")
    cat = resources_sub.add_parser("cat", help="Print a packaged resource to stdout.")
    cat.add_argument("path", type=str, help="Resource path under yolozu/data (e.g., schemas/predictions.schema.json).")
    copy = resources_sub.add_parser("copy", help="Copy a packaged resource to a file path.")
    copy.add_argument("path", type=str, help="Resource path under yolozu/data.")
    copy.add_argument("--output", required=True, help="Output file path.")
    copy.add_argument("--force", action="store_true", help="Overwrite output if it exists.")

    migrate = sub.add_parser("migrate", help="Migration helpers (dataset/config/predictions).")
    migrate_sub = migrate.add_subparsers(dest="migrate_command", required=True)

    mig_dataset = migrate_sub.add_parser("dataset", help="Generate dataset.json wrapper for external dataset layouts.")
    mig_dataset.add_argument(
        "--from",
        dest="from_format",
        choices=("ultralytics", "coco"),
        required=True,
        help="Source ecosystem.",
    )
    mig_dataset.add_argument("--data", default=None, help="(Ultralytics) data.yaml path (preferred).")
    mig_dataset.add_argument("--args", default=None, help="(Ultralytics) args.yaml (optional; used for task/data inference).")
    mig_dataset.add_argument(
        "--split",
        default=None,
        help="Split name (Ultralytics: select from data.yaml; COCO: instances_<split>.json, default: val2017).",
    )
    mig_dataset.add_argument(
        "--task",
        choices=("detect", "segment", "pose"),
        default=None,
        help="(Ultralytics) Override task inference (segment enables polygon label parsing).",
    )
    mig_dataset.add_argument("--coco-root", default=None, help="(COCO) Root containing images/ and annotations/.")
    mig_dataset.add_argument("--instances-json", default=None, help="(COCO) Override instances JSON path.")
    mig_dataset.add_argument(
        "--mode",
        choices=("manifest", "symlink", "copy"),
        default="manifest",
        help="(COCO) Image handling: manifest=do not copy; symlink/copy into output/images/<split>.",
    )
    mig_dataset.add_argument("--include-crowd", action="store_true", help="(COCO) Include iscrowd annotations.")
    mig_dataset.add_argument("--output", required=True, help="Output directory or dataset.json file path.")
    mig_dataset.add_argument("--force", action="store_true", help="Overwrite output if it exists.")

    mig_preds = migrate_sub.add_parser("predictions", help="Convert external prediction outputs into YOLOZU predictions.json.")
    mig_preds.add_argument(
        "--from",
        dest="from_format",
        choices=("coco-results",),
        required=True,
        help="Source prediction format.",
    )
    mig_preds.add_argument("--results", required=True, help="COCO results JSON path (list of detections).")
    mig_preds.add_argument("--instances", required=True, help="COCO instances JSON path (for image_id mapping + sizes).")
    mig_preds.add_argument("--output", required=True, help="Output predictions.json path.")
    mig_preds.add_argument("--score-threshold", type=float, default=0.0, help="Minimum score to keep (default: 0.0).")
    mig_preds.add_argument("--force", action="store_true", help="Overwrite output if it exists.")

    mig_seg = migrate_sub.add_parser("seg-dataset", help="Generate semantic segmentation dataset descriptor JSON.")
    mig_seg.add_argument(
        "--from",
        dest="from_format",
        choices=("voc", "cityscapes", "ade20k"),
        required=True,
        help="Source dataset type.",
    )
    mig_seg.add_argument("--root", required=True, help="Dataset root path.")
    mig_seg.add_argument("--split", default="val", help="Split name (train|val|test, dataset-specific aliases allowed).")
    mig_seg.add_argument("--output", required=True, help="Output descriptor JSON path.")
    mig_seg.add_argument("--path-type", choices=("absolute", "relative"), default="absolute", help="Emit absolute or relative paths.")
    mig_seg.add_argument("--mode", choices=("manifest", "symlink", "copy"), default="manifest", help="Descriptor mode hint.")
    mig_seg.add_argument("--force", action="store_true", help="Overwrite output if it exists.")
    mig_seg.add_argument("--year", default=None, help="(VOC) Optional year selector (e.g. 2012).")
    mig_seg.add_argument(
        "--masks-dirname",
        default="SegmentationClass",
        help="(VOC) Mask directory name under VOC year root (default: SegmentationClass).",
    )
    mig_seg.add_argument(
        "--label-type",
        choices=("labelTrainIds", "labelIds"),
        default="labelTrainIds",
        help="(Cityscapes) Mask suffix type (default: labelTrainIds).",
    )

    imp = sub.add_parser("import", help="Import adapters (read-only projection into canonical schema).")
    imp_sub = imp.add_subparsers(dest="import_command", required=True)

    imp_dataset = imp_sub.add_parser("dataset", help="Generate a read-only dataset wrapper for external layouts.")
    imp_dataset.add_argument(
        "--from",
        dest="from_format",
        choices=("auto", "ultralytics", "coco-instances"),
        required=True,
        help="Source ecosystem.",
    )
    imp_dataset.add_argument("--output", required=True, help="Output directory (wrapper) or dataset.json file path.")
    imp_dataset.add_argument("--force", action="store_true", help="Overwrite output if it exists.")
    imp_dataset.add_argument("--split", default=None, help="Split name (COCO default: val2017; Ultralytics: from data.yaml).")

    imp_dataset.add_argument("--data", default=None, help="(Ultralytics) data.yaml path (preferred).")
    imp_dataset.add_argument("--args", default=None, help="(Ultralytics) args.yaml (optional; used for task/data inference).")
    imp_dataset.add_argument("--task", choices=("detect", "segment", "pose"), default=None, help="(Ultralytics) Task override.")

    imp_dataset.add_argument("--instances", default=None, help="(COCO) instances_*.json path.")
    imp_dataset.add_argument("--images-dir", default=None, help="(COCO) Images directory for this split.")
    imp_dataset.add_argument("--include-crowd", action="store_true", help="(COCO) Include iscrowd annotations.")

    imp_cfg = imp_sub.add_parser("config", help="Project external configs into canonical TrainConfig (major keys only).")
    imp_cfg.add_argument(
        "--from",
        dest="from_format",
        choices=("auto", "ultralytics", "mmdet", "yolox", "detectron2"),
        required=True,
        help="Source ecosystem.",
    )
    imp_cfg.add_argument("--args", default=None, help="(Ultralytics) args.yaml path.")
    imp_cfg.add_argument("--config", default=None, help="(MMDet/YOLOX/Detectron2) config file path.")
    imp_cfg.add_argument("--output", required=True, help="Output path (file or directory).")
    imp_cfg.add_argument("--force", action="store_true", help="Overwrite output if it exists.")

    train_p = sub.add_parser("train", help="Train with RT-DETR pose scaffold (requires `yolozu[train]`).")
    train_p.add_argument("config", nargs="?", type=str, help="Path to train config YAML/JSON (train_setting.yaml).")
    train_p.add_argument(
        "--import",
        dest="import_from",
        choices=("auto", "ultralytics", "mmdet", "yolox", "detectron2"),
        default=None,
        help="Optional shorthand: resolve external config into canonical TrainConfig before training.",
    )
    train_p.add_argument("--data", default=None, help="(train --import ultralytics) data.yaml path for dataset preview.")
    train_p.add_argument("--cfg", default=None, help="(train --import) external framework config/args path.")
    train_p.add_argument(
        "--resolved-config-out",
        default="reports/train_config_resolved_import.json",
        help="Output path for canonical TrainConfig resolved by train --import.",
    )
    train_p.add_argument(
        "--force-import-overwrite",
        action="store_true",
        help="Overwrite --resolved-config-out if it already exists.",
    )
    train_p.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to the trainer (e.g. --run-contract --run-id exp01 --resume).",
    )

    test_p = sub.add_parser("test", help="Run scenario suite (dummy/precomputed adapters are CPU-only).")
    test_p.add_argument("config", type=str, help="Path to test config YAML/JSON (test_setting.yaml).")
    test_p.add_argument(
        "test_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to the scenario runner (e.g. --adapter rtdetr_pose --max-images 50).",
    )

    demo = sub.add_parser("demo", help="Run small self-contained demos (CPU-friendly).")
    demo.add_argument(
        "--coco-instances-json",
        default=None,
        help="(demo suite) COCO instances_*.json path to enable the polygon-mask instance-seg demo.",
    )
    demo.add_argument(
        "--coco-images-dir",
        default=None,
        help="(demo suite) COCO images dir (joined with image.file_name) for the polygon-mask instance-seg demo.",
    )
    demo_sub = demo.add_subparsers(dest="demo_command", required=False)

    demo_is = demo_sub.add_parser("instance-seg", help="Instance-seg eval demo (numpy + Pillow).")
    demo_is.add_argument("--run-dir", default=None, help="Run directory (default: demo_output/instance_seg/<utc>).")
    demo_is.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    demo_is.add_argument("--num-images", type=int, default=8, help="Number of images (default: 8).")
    demo_is.add_argument("--image-size", type=int, default=96, help="Square image size (default: 96).")
    demo_is.add_argument("--max-instances", type=int, default=2, help="Max instances per image (default: 2).")
    demo_is.add_argument(
        "--background",
        choices=("synthetic", "coco128", "coco-instances"),
        default="coco-instances",
        help="Background source: synthetic shapes, COCO128 (bbox-derived), or COCO instances polygons (default: coco-instances).",
    )
    demo_is.add_argument(
        "--coco-instances-json",
        default=None,
        help=(
            "(background=coco-instances) Path to COCO instances_*.json (polygon segmentations). "
            "If omitted, defaults to data/coco/annotations/instances_val2017.json."
        ),
    )
    demo_is.add_argument(
        "--coco-images-dir",
        default=None,
        help=(
            "(background=coco-instances) Root images dir for the COCO split (joined with image.file_name). "
            "If omitted, defaults to data/coco/images/val2017."
        ),
    )
    demo_is.add_argument(
        "--inference",
        choices=("none", "auto", "torchvision"),
        default=None,
        help=(
            "(background=coco-instances) Instance-seg inference backend. "
            "Default: auto when background=coco-instances, otherwise none."
        ),
    )
    demo_is.add_argument(
        "--device",
        default="auto",
        help="(inference) Torch device (cpu|cuda|mps|auto) (default: auto).",
    )
    demo_is.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="(inference) Score threshold for predicted instances (default: 0.5).",
    )

    demo_cl = demo_sub.add_parser("continual", help="Toy continual-learning demo (requires torch; CPU OK).")
    demo_cl.add_argument("--output", default=None, help="Output JSON path or dir (default: runs/yolozu_demos/continual/...).")
    demo_cl.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    demo_cl.add_argument("--device", default="cpu", help="Torch device (default: cpu).")
    demo_cl.add_argument(
        "--practical",
        action="store_true",
        help=(
            "Run a more practical vision demo (MNIST rotation shift + ResNet18 backbone) "
            "with CPU-friendly fast defaults. Equivalent to setting --problem mnist_rotate with smaller steps/sizes."
        ),
    )
    demo_cl.add_argument(
        "--fast",
        action="store_true",
        help="Reduce steps and sample counts to keep CPU runs short (applies to toy2d and mnist_rotate).",
    )
    demo_cl.add_argument(
        "--problem",
        default="toy2d",
        choices=("toy2d", "mnist_rotate"),
        help="Continual-learning problem: toy2d or MNIST rotation shift (default: toy2d).",
    )
    demo_cl.add_argument(
        "--data-dir",
        default=str(Path("data") / "torchvision"),
        help="(problem=mnist_rotate) Torchvision dataset root dir (default: data/torchvision).",
    )
    demo_cl.add_argument("--method", default="ewc_replay", choices=("naive", "ewc", "replay", "ewc_replay"))
    demo_cl.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=("naive", "ewc", "replay", "ewc_replay"),
        help="Run multiple methods and write a suite report.",
    )
    demo_cl.add_argument(
        "--compare",
        action="store_true",
        help="Convenience flag: run all methods (naive/ewc/replay/ewc_replay) and write a suite report.",
    )
    demo_cl.add_argument(
        "--markdown",
        action="store_true",
        help="Also write a markdown summary table next to the JSON output (suite or single).",
    )
    demo_cl.add_argument("--steps-a", type=int, default=200, help="Training steps on domain A (default: 200).")
    demo_cl.add_argument("--steps-b", type=int, default=200, help="Training steps on domain B (default: 200).")
    demo_cl.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    demo_cl.add_argument("--hidden", type=int, default=32, help="Hidden units (default: 32).")
    demo_cl.add_argument("--lr", type=float, default=1e-2, help="Learning rate (default: 1e-2).")
    demo_cl.add_argument("--corr", type=float, default=2.0, help="Spurious correlation magnitude (default: 2.0).")
    demo_cl.add_argument("--noise", type=float, default=0.6, help="Feature noise std (default: 0.6).")
    demo_cl.add_argument("--n-train", type=int, default=4096, help="Train samples per domain (default: 4096).")
    demo_cl.add_argument("--n-eval", type=int, default=1024, help="Eval samples per domain (default: 1024).")
    demo_cl.add_argument("--ewc-lambda", type=float, default=20.0, help="EWC penalty weight (default: 20.0).")
    demo_cl.add_argument("--fisher-batches", type=int, default=64, help="Batches for Fisher estimate (default: 64).")
    demo_cl.add_argument("--replay-capacity", type=int, default=512, help="Replay buffer capacity (default: 512).")
    demo_cl.add_argument("--replay-k", type=int, default=64, help="Replay samples per step (default: 64).")

    demo_kp = demo_sub.add_parser("keypoints", help="Keypoints inference demo (torchvision Keypoint R-CNN).")
    demo_kp.add_argument("--image", default=None, help="Input image path (default: a bundled smoke image if present).")
    demo_kp.add_argument("--run-dir", default=None, help="Run directory (default: demo_output/keypoints/<utc>).")
    demo_kp.add_argument("--device", default="auto", help="Torch device (cpu|cuda|mps|auto) (default: auto).")
    demo_kp.add_argument("--score-threshold", type=float, default=0.7, help="Min score to keep persons (default: 0.7).")
    demo_kp.add_argument("--max-persons", type=int, default=3, help="Max persons to render (default: 3).")

    demo_pose = demo_sub.add_parser("pose", help="6D pose demo (chessboard or ArUco + OpenCV solvePnP).")
    demo_pose.add_argument("--image", default=None, help="Input image path (default: a generated chessboard sample).")
    demo_pose.add_argument("--run-dir", default=None, help="Run directory (default: demo_output/pose/<utc>).")
    demo_pose.add_argument(
        "--backend",
        choices=("chessboard", "aruco", "densefusion"),
        default="chessboard",
        help="Pose backend (default: chessboard).",
    )
    demo_pose.add_argument("--pattern-cols", type=int, default=None, help="Chessboard inner corners (cols).")
    demo_pose.add_argument("--pattern-rows", type=int, default=None, help="Chessboard inner corners (rows).")
    demo_pose.add_argument("--square-size", type=float, default=0.04, help="Chessboard square size in meters (default: 0.04).")
    demo_pose.add_argument("--aruco-dict", default="DICT_4X4_50", help="ArUco dictionary name (default: DICT_4X4_50).")
    demo_pose.add_argument("--aruco-id", type=int, default=23, help="ArUco marker id (default: 23).")
    demo_pose.add_argument("--marker-length", type=float, default=0.05, help="ArUco marker length in meters (default: 0.05).")
    demo_pose.add_argument("--densefusion-root", default=None, help="DenseFusion repo root (default: demo_output/pose/_densefusion).")
    demo_pose.add_argument("--densefusion-object", default="ape", help="LineMOD object for DenseFusion demo (default: ape).")
    demo_pose.add_argument(
        "--densefusion-auto-download",
        action="store_true",
        help="Auto-download DenseFusion assets (large) (default: enabled).",
    )
    demo_pose.add_argument(
        "--no-densefusion-auto-download",
        dest="densefusion_auto_download",
        action="store_false",
        help="Disable auto-download for DenseFusion assets.",
    )
    demo_pose.set_defaults(densefusion_auto_download=True)
    demo_pose.add_argument("--densefusion-model", default=None, help="DenseFusion pose model checkpoint path.")
    demo_pose.add_argument("--densefusion-refine-model", default=None, help="DenseFusion refiner checkpoint path.")
    demo_pose.add_argument("--camera-fx", type=float, default=None, help="Camera fx (default: inferred from image).")
    demo_pose.add_argument("--camera-fy", type=float, default=None, help="Camera fy (default: inferred from image).")
    demo_pose.add_argument("--camera-cx", type=float, default=None, help="Camera cx (default: image center).")
    demo_pose.add_argument("--camera-cy", type=float, default=None, help="Camera cy (default: image center).")
    demo_pose.add_argument(
        "--sample-source",
        choices=("auto", "download", "synthetic"),
        default="auto",
        help="Sample source when --image is omitted (default: auto).",
    )

    demo_depth = demo_sub.add_parser(
        "depth", help="Monocular depth inference demo (Depth Anything / MiDaS / DPT; relative depth)."
    )
    demo_depth.add_argument("--image", default=None, help="Input image path (default: a bundled smoke image if present).")
    demo_depth.add_argument("--run-dir", default=None, help="Run directory (default: demo_output/depth/<utc>).")
    demo_depth.add_argument("--device", default="auto", help="Torch device (cpu|cuda|mps|auto) (default: auto).")
    demo_depth.add_argument(
        "--model",
        default="depth_anything",
        choices=("depth_anything", "midas_small", "dpt_hybrid", "dpt_large"),
        help="Depth model preset (default: depth_anything).",
    )
    demo_depth.add_argument(
        "--compare",
        action="store_true",
        help="Run 3-model comparison (midas_small, dpt_hybrid, depth_anything) and write suffixed outputs.",
    )
    demo_depth.add_argument(
        "--invert",
        action="store_true",
        help="Invert visualization (closer=brighter) (default: enabled).",
    )
    demo_depth.add_argument(
        "--no-invert",
        dest="invert",
        action="store_false",
        help="Disable inversion (farther=brighter).",
    )
    demo_depth.set_defaults(invert=True)

    demo_tr = demo_sub.add_parser("train", help="Training demo (MNIST fine-tune; requires torch+torchvision).")
    demo_tr.add_argument(
        "--output",
        default=None,
        help="Output JSON path or run directory (default: demo_output/train/<utc>/train_demo_report.json).",
    )
    demo_tr.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    demo_tr.add_argument("--device", default="cpu", help="Torch device (default: cpu).")
    demo_tr.add_argument(
        "--data-dir",
        default=str(Path("data") / "torchvision"),
        help="Torchvision dataset root dir (default: data/torchvision).",
    )
    demo_tr.add_argument("--epochs", type=int, default=1, help="Epochs (default: 1).")
    demo_tr.add_argument("--max-steps", type=int, default=80, help="Max train steps (default: 80).")
    demo_tr.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    demo_tr.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4).")

    args = parser.parse_args(argv)
    if args.command == "train":
        if getattr(args, "import_from", None):
            _cmd_train_import_preview(args)
            if not getattr(args, "config", None):
                return 0
        if not getattr(args, "config", None):
            raise SystemExit("train config is required unless using --import preview-only mode")
        config_path = Path(args.config)
        if not config_path.exists():
            raise SystemExit(f"config not found: {config_path}")
        return _cmd_train(config_path, extra_args=list(getattr(args, "train_args", []) or []))
    if args.command == "test":
        config_path = Path(args.config)
        if not config_path.exists():
            raise SystemExit(f"config not found: {config_path}")
        return _cmd_test(config_path, extra_args=list(getattr(args, "test_args", []) or []))
    if args.command == "doctor":
        if getattr(args, "doctor_command", None) == "import":
            return _cmd_doctor_import(args)
        return _cmd_doctor(str(args.output))
    if args.command == "list":
        if args.list_command == "models":
            return _cmd_list_models(args)
        raise SystemExit("unknown list command")
    if args.command == "fetch":
        return _cmd_fetch_model(args)
    if args.command == "export":
        return _cmd_export(args)
    if args.command == "predict-images":
        return _cmd_predict_images(args)
    if args.command == "eval-coco":
        return _cmd_eval_coco(args)
    if args.command == "calibrate":
        return _cmd_calibrate(args)
    if args.command == "eval-long-tail":
        return _cmd_eval_long_tail(args)
    if args.command == "long-tail-recipe":
        return _cmd_long_tail_recipe(args)
    if args.command == "parity":
        return _cmd_parity(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "eval-instance-seg":
        return _cmd_eval_instance_seg(args)
    if args.command == "onnxrt":
        if args.onnxrt_command == "export":
            return _cmd_onnxrt_export(args)
        if args.onnxrt_command == "quantize":
            return _cmd_onnxrt_quantize(args)
        raise SystemExit("unknown onnxrt command")
    if args.command == "resources":
        return _cmd_resources(args)
    if args.command == "migrate":
        return _cmd_migrate(args)
    if args.command == "import":
        return _cmd_import(args)
    if args.command == "demo":
        def _print_instance_seg_report(*, out_path: Path, label: str | None = None) -> None:
            try:
                payload = json.loads(Path(out_path).read_text(encoding="utf-8"))
                res = payload.get("result", {})
                counts = res.get("counts", {})
                meta = payload.get("meta", {})
                if label:
                    print(label)
                print(
                    "instance-seg demo: "
                    f"mAP50-95={res.get('map50_95'):.3f} mAP50={res.get('map50'):.3f} "
                    f"(images={counts.get('images')} gt={counts.get('gt_instances')} pred={counts.get('pred_instances')} classes={counts.get('classes')})"
                )
                if isinstance(meta, dict):
                    inf = meta.get("inference")
                    if isinstance(inf, dict):
                        used = bool(inf.get("used"))
                        backend = inf.get("backend")
                        weights = inf.get("weights")
                        dev = inf.get("device")
                        mode = inf.get("mode")
                        if used and backend:
                            print(f"inference: {backend} weights={weights} device={dev}")
                        elif mode:
                            print(f"inference: {mode} (used={used})")
                if isinstance(meta, dict) and meta.get("run_dir"):
                    print(f"output_dir: {meta.get('run_dir')}")
            except Exception:
                if label:
                    print(label)
            print(str(out_path))

        if args.demo_command is None:
            import time

            from yolozu.demos.instance_seg import run_instance_seg_demo

            suite_id = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
            suite_root = Path("demo_output") / "instance_seg" / f"suite_{suite_id}"
            ok = 0

            suite_instances = getattr(args, "coco_instances_json", None)
            suite_images = getattr(args, "coco_images_dir", None)
            if suite_instances or suite_images:
                if not (suite_instances and suite_images):
                    raise ValueError("demo suite requires both --coco-instances-json and --coco-images-dir")
                has_real_coco_instances = True
            else:
                default_instances = Path("data") / "coco" / "annotations" / "instances_val2017.json"
                default_images = Path("data") / "coco" / "images" / "val2017"
                has_real_coco_instances = default_instances.exists() and default_images.exists()

            if not has_real_coco_instances:
                # 1) Synthetic instance-seg
                syn_run_dir = suite_root / "synthetic"
                syn_run_dir.mkdir(parents=True, exist_ok=True)
                syn_cfg_path = syn_run_dir / "demo_config.json"
                syn_cfg_path.write_text(
                    json.dumps(
                        {
                            "kind": "demo_config",
                            "demo": "instance-seg",
                            "background": "synthetic",
                            "seed": 0,
                            "num_images": 4,
                            "image_size": 96,
                            "max_instances": 2,
                        },
                        indent=2,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                out_syn = run_instance_seg_demo(
                    run_dir=syn_run_dir,
                    seed=0,
                    num_images=4,
                    image_size=96,
                    max_instances=2,
                    background="synthetic",
                )
                print(f"config: {syn_cfg_path}")
                _print_instance_seg_report(out_path=Path(out_syn), label="== instance-seg (synthetic) ==")
                ok += 1

                # 2) COCO128-backed instance-seg (skip if dataset missing)
                try:
                    coco128_run_dir = suite_root / "coco128"
                    coco128_run_dir.mkdir(parents=True, exist_ok=True)
                    coco128_cfg_path = coco128_run_dir / "demo_config.json"
                    coco128_cfg_path.write_text(
                        json.dumps(
                            {
                                "kind": "demo_config",
                                "demo": "instance-seg",
                                "background": "coco128",
                                "seed": 0,
                                "num_images": 2,
                                "image_size": 96,
                                "max_instances": 2,
                            },
                            indent=2,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    out_coco = run_instance_seg_demo(
                        run_dir=coco128_run_dir,
                        seed=0,
                        num_images=2,
                        image_size=96,
                        max_instances=2,
                        background="coco128",
                    )
                    print(f"config: {coco128_cfg_path}")
                    _print_instance_seg_report(out_path=Path(out_coco), label="== instance-seg (coco128) ==")
                    ok += 1
                except FileNotFoundError as exc:
                    print("== instance-seg (coco128) ==")
                    print(f"skipped: {exc}")

            # 2b) COCO instances (polygon) instance-seg
            try:
                default_instances = Path("data") / "coco" / "annotations" / "instances_val2017.json"
                default_images = Path("data") / "coco" / "images" / "val2017"

                def _ensure_tiny_coco_instances_fixture(*, fixture_dir: Path) -> tuple[Path, Path]:
                    fixture_dir.mkdir(parents=True, exist_ok=True)
                    images_dir = fixture_dir / "images"
                    images_dir.mkdir(parents=True, exist_ok=True)
                    instances_path = fixture_dir / "instances_val.json"

                    # Keep this dependency-light (Pillow only), but generate something that
                    # looks more like a real segmentation dataset.
                    #
                    # If coco128 is available locally, prefer using real photos + bbox labels
                    # to create polygon masks that at least match *object locations*.
                    # (coco128 does not ship true segmentation polygons in this repo.)
                    import math
                    import random
                    import shutil

                    from PIL import Image, ImageDraw

                    rng = random.Random(0)
                    fallback_width, fallback_height = 320, 240
                    fallback_num_images = 6

                    categories = [
                        {"id": 1, "name": "person"},
                        {"id": 2, "name": "bicycle"},
                        {"id": 3, "name": "dog"},
                    ]

                    def _jitter_color(base: tuple[int, int, int], j: int = 25) -> tuple[int, int, int]:
                        return (
                            max(0, min(255, base[0] + rng.randint(-j, j))),
                            max(0, min(255, base[1] + rng.randint(-j, j))),
                            max(0, min(255, base[2] + rng.randint(-j, j))),
                        )

                    def _make_textured_bg(*, width: int, height: int) -> Image.Image:
                        # Procedural "photo-like" background: combine noise layers,
                        # colorize, blur, add gentle gradient + vignette.
                        from PIL import ImageChops, ImageEnhance, ImageFilter, ImageOps

                        # Base noise (low frequency)
                        n1 = Image.effect_noise((width, height), rng.uniform(35.0, 65.0)).convert("L")
                        n1 = n1.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.5, 3.5)))
                        # Detail noise (high frequency)
                        n2 = Image.effect_noise((width, height), rng.uniform(10.0, 25.0)).convert("L")
                        n2 = n2.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.1)))

                        mix = ImageChops.add(n1, n2, scale=2.0)

                        # Choose a palette that looks like real-world scenes.
                        palettes = [
                            ((35, 55, 90), (210, 205, 170)),  # blue -> tan
                            ((25, 70, 45), (200, 200, 205)),  # green -> gray
                            ((70, 50, 35), (215, 210, 195)),  # brown -> beige
                            ((40, 45, 55), (205, 210, 220)),  # slate -> light gray
                        ]
                        dark, light = rng.choice(palettes)
                        base = ImageOps.colorize(mix, black=dark, white=light).convert("RGB")

                        # Gentle global gradient (simulate lighting)
                        grad = Image.new("L", (width, height), 0)
                        gd = ImageDraw.Draw(grad)
                        top = rng.randint(40, 90)
                        bottom = rng.randint(170, 230)
                        for y in range(height):
                            v = int(top + (bottom - top) * (y / max(1, height - 1)))
                            gd.line([0, y, width, y], fill=v)
                        grad_rgb = ImageOps.colorize(grad, black=(0, 0, 0), white=(255, 255, 255)).convert("RGB")
                        base = ImageChops.multiply(base, grad_rgb)

                        # Mild sharpening/contrast like a camera pipeline
                        base = base.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))
                        base = ImageEnhance.Contrast(base).enhance(rng.uniform(1.05, 1.25))
                        base = ImageEnhance.Color(base).enhance(rng.uniform(1.05, 1.20))

                        # Vignette
                        vig = Image.new("L", (width, height), 255)
                        vd = ImageDraw.Draw(vig)
                        inset = rng.randint(10, 35)
                        vd.ellipse([inset, inset, width - inset, height - inset], fill=210)
                        vig = vig.filter(ImageFilter.GaussianBlur(radius=rng.uniform(10.0, 18.0)))
                        base = ImageChops.multiply(base, ImageOps.colorize(vig, black=(0, 0, 0), white=(255, 255, 255)).convert("RGB"))

                        return base

                    def _make_polygon(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
                        n = rng.randint(8, 14)
                        pts: list[tuple[float, float]] = []
                        phase = rng.random() * (2.0 * math.pi)
                        for k in range(n):
                            a = phase + (2.0 * math.pi * k / n)
                            rr = r * (0.65 + 0.55 * rng.random())
                            x = cx + rr * math.cos(a)
                            y = cy + rr * math.sin(a)
                            x = max(2.0, min(float(width) - 3.0, x))
                            y = max(2.0, min(float(height) - 3.0, y))
                            pts.append((x, y))
                        return pts

                    def _iter_coco128_pairs() -> list[tuple[Path, Path]]:
                        coco128_root = Path("data") / "coco128"
                        images_base = coco128_root / "images"
                        labels_base = coco128_root / "labels"
                        if not images_base.exists() or not labels_base.exists():
                            return []
                        pairs: list[tuple[Path, Path]] = []
                        for img in sorted(images_base.rglob("*")):
                            if not img.is_file() or img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"):
                                continue
                            rel = img.relative_to(images_base)
                            lab = labels_base / rel.with_suffix(".txt")
                            if lab.exists():
                                pairs.append((img, lab))
                        return pairs

                    def _yolo_bbox_to_xyxy(*, w: int, h: int, xc: float, yc: float, bw: float, bh: float) -> tuple[int, int, int, int]:
                        x0 = int((float(xc) - float(bw) / 2.0) * float(w))
                        y0 = int((float(yc) - float(bh) / 2.0) * float(h))
                        x1 = int((float(xc) + float(bw) / 2.0) * float(w))
                        y1 = int((float(yc) + float(bh) / 2.0) * float(h))
                        x0 = max(0, min(int(w) - 1, x0))
                        y0 = max(0, min(int(h) - 1, y0))
                        x1 = max(0, min(int(w), x1))
                        y1 = max(0, min(int(h), y1))
                        if x1 <= x0:
                            x1 = min(int(w), x0 + 1)
                        if y1 <= y0:
                            y1 = min(int(h), y0 + 1)
                        return x0, y0, x1, y1

                    def _bbox_to_polygon(*, x0: int, y0: int, x1: int, y1: int) -> list[float]:
                        # Create a jittered ellipse-like polygon inside bbox.
                        cx = (x0 + x1) / 2.0
                        cy = (y0 + y1) / 2.0
                        rx = max(2.0, (x1 - x0) / 2.0)
                        ry = max(2.0, (y1 - y0) / 2.0)
                        n = rng.randint(10, 18)
                        phase = rng.random() * (2.0 * math.pi)
                        pts: list[float] = []
                        for k in range(n):
                            a = phase + (2.0 * math.pi * k / n)
                            # Alternate radius a bit to avoid perfect ellipse.
                            s = 0.75 + 0.30 * rng.random()
                            if (k % 2) == 0:
                                s *= 0.92
                            x = cx + (rx * s) * math.cos(a)
                            y = cy + (ry * s) * math.sin(a)
                            x = max(float(x0), min(float(x1 - 1), x))
                            y = max(float(y0), min(float(y1 - 1), y))
                            pts.extend([float(x), float(y)])
                        return pts

                    images: list[dict[str, Any]] = []
                    annotations: list[dict[str, Any]] = []
                    ann_id = 1

                    coco128_pairs = _iter_coco128_pairs()
                    if coco128_pairs:
                        # Use real photos + their bbox labels (converted into polygons).
                        rng.shuffle(coco128_pairs)
                        selected = coco128_pairs[: min(6, len(coco128_pairs))]
                        seen_class_ids: set[int] = set()
                        for image_id, (src_img, src_lab) in enumerate(selected, start=1):
                            file_name = f"{image_id:012d}{src_img.suffix.lower()}"
                            dst_img = images_dir / file_name
                            shutil.copyfile(src_img, dst_img)
                            img = Image.open(dst_img).convert("RGB")
                            w, h = img.size

                            lines = [ln.strip() for ln in src_lab.read_text(encoding="utf-8").splitlines() if ln.strip()]
                            # Bound runtime.
                            lines = lines[:10]
                            for ln in lines:
                                parts = ln.split()
                                if len(parts) < 5:
                                    continue
                                class_id = int(float(parts[0]))
                                xc, yc, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                                x0, y0, x1, y1 = _yolo_bbox_to_xyxy(w=w, h=h, xc=xc, yc=yc, bw=bw, bh=bh)
                                if (x1 - x0) < 4 or (y1 - y0) < 4:
                                    continue

                                seg = [_bbox_to_polygon(x0=x0, y0=y0, x1=x1, y1=y1)]
                                # Map YOLO class_id to category_id space (just keep it as-is).
                                cat_id = int(class_id)
                                seen_class_ids.add(cat_id)

                                annotations.append(
                                    {
                                        "id": ann_id,
                                        "image_id": image_id,
                                        "category_id": cat_id,
                                        "iscrowd": 0,
                                        "segmentation": seg,
                                    }
                                )
                                ann_id += 1

                            images.append({"id": image_id, "file_name": file_name, "width": w, "height": h})

                        # Provide minimal categories list for the ids we used.
                        used_categories = [{"id": int(cid), "name": f"class_{int(cid)}"} for cid in sorted(seen_class_ids)]
                        if used_categories:
                            categories = used_categories
                    else:
                        # Fallback: fully synthetic photos + shapes.
                        width, height = fallback_width, fallback_height
                        num_images = fallback_num_images
                        for image_id in range(1, num_images + 1):
                            file_name = f"{image_id:012d}.jpg"
                            img = _make_textured_bg(width=width, height=height)
                            draw = ImageDraw.Draw(img)

                            num_inst = rng.randint(2, 5)
                            for _ in range(num_inst):
                                cat = rng.choice(categories)
                                cx = rng.uniform(60.0, float(width) - 60.0)
                                cy = rng.uniform(50.0, float(height) - 50.0)
                                r = rng.uniform(18.0, 55.0)

                                # Some instances are multi-polygons (COCO supports list of polygons).
                                segs: list[list[float]] = []
                                poly1 = _make_polygon(cx, cy, r)
                                segs.append([v for xy in poly1 for v in xy])
                                if rng.random() < 0.25:
                                    poly2 = _make_polygon(
                                        cx + rng.uniform(-25.0, 25.0),
                                        cy + rng.uniform(-20.0, 20.0),
                                        r * 0.55,
                                    )
                                    segs.append([v for xy in poly2 for v in xy])

                                fill = _jitter_color((60, 140, 220) if (int(cat["id"]) % 2) == 0 else (220, 120, 60), 20)
                                outline = (0, 0, 0)
                                for poly in segs:
                                    pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly) - 1, 2)]
                                    draw.polygon(pts, fill=fill, outline=outline)
                                    draw.line(pts + [pts[0]], fill=outline, width=2)

                                annotations.append(
                                    {
                                        "id": ann_id,
                                        "image_id": image_id,
                                        "category_id": int(cat["id"]),
                                        "iscrowd": 0,
                                        "segmentation": segs,
                                    }
                                )
                                ann_id += 1

                            img.save(images_dir / file_name)
                            images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})

                    coco = {"images": images, "annotations": annotations, "categories": categories}
                    instances_path.write_text(json.dumps(coco), encoding="utf-8")
                    return instances_path, images_dir

                suite_instances = getattr(args, "coco_instances_json", None)
                suite_images = getattr(args, "coco_images_dir", None)
                if suite_instances or suite_images:
                    instances_path = Path(str(suite_instances)) if suite_instances else None
                    images_path = Path(str(suite_images)) if suite_images else None
                    if instances_path is None or images_path is None:
                        raise ValueError("demo suite requires both --coco-instances-json and --coco-images-dir")
                    coco_source = "cli"
                else:
                    if default_instances.exists() and default_images.exists():
                        instances_path = default_instances
                        images_path = default_images
                        coco_source = "data/coco"
                    else:
                        instances_path, images_path = _ensure_tiny_coco_instances_fixture(
                            fixture_dir=Path("demo_output")
                            / "instance_seg"
                            / f"suite_{suite_id}"
                            / "_coco_instances_fixture"
                        )
                        coco_source = "fixture"

                ci_run_dir = suite_root / "coco_instances"
                ci_run_dir.mkdir(parents=True, exist_ok=True)
                ci_cfg_path = ci_run_dir / "demo_config.json"
                ci_cfg_path.write_text(
                    json.dumps(
                        {
                            "kind": "demo_config",
                            "demo": "instance-seg",
                            "background": "coco-instances",
                            "seed": 0,
                            "num_images": 2,
                            "image_size": 96,
                            "max_instances": 2,
                            "coco_instances_json": str(instances_path),
                            "coco_images_dir": str(images_path),
                            "coco_source": coco_source,
                            "inference": "auto",
                            "device": "cpu",
                            "score_threshold": 0.5,
                        },
                        indent=2,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )

                if instances_path.exists() and images_path.exists():
                    out_ci = run_instance_seg_demo(
                        run_dir=ci_run_dir,
                        seed=0,
                        num_images=2,
                        image_size=96,
                        max_instances=2,
                        background="coco-instances",
                        coco_instances_json=instances_path,
                        coco_images_dir=images_path,
                        inference="auto",
                        device="cpu",
                        score_threshold=0.5,
                    )
                    print(f"config: {ci_cfg_path}")
                    _print_instance_seg_report(out_path=Path(out_ci), label="== instance-seg (coco-instances) ==")
                    ok += 1
                else:
                    print("== instance-seg (coco-instances) ==")
                    print(f"skipped: not found: instances={instances_path} images_dir={images_path}")
            except Exception as exc:
                print("== instance-seg (coco-instances) ==")
                print(f"skipped: {exc}")

            try:
                suite_root.mkdir(parents=True, exist_ok=True)
                suite_cfg_path = suite_root / "suite_config.json"
                suite_cfg_path.write_text(
                    json.dumps(
                        {
                            "kind": "demo_suite_config",
                            "suite_id": suite_id,
                            "has_real_coco_instances": bool(has_real_coco_instances),
                            "cli_coco_instances_json": str(suite_instances) if suite_instances else None,
                            "cli_coco_images_dir": str(suite_images) if suite_images else None,
                        },
                        indent=2,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"suite_config: {suite_cfg_path}")
            except Exception:
                pass

            # 3) Continual demo (skip if torch missing)
            try:
                from yolozu.demos.continual import run_continual_demo

                out = run_continual_demo(
                    output=None,
                    seed=0,
                    device="cpu",
                    method="naive",
                    steps_a=50,
                    steps_b=50,
                    batch_size=64,
                    hidden=32,
                    lr=1e-2,
                    corr=2.0,
                    noise=0.6,
                    n_train=1024,
                    n_eval=256,
                    ewc_lambda=20.0,
                    fisher_batches=8,
                    replay_capacity=256,
                    replay_k=32,
                )
                if out is not None:
                    print("== continual ==")
                    print(f"output_dir: {Path(out).parent}")
                    print(str(out))
                    ok += 1
            except Exception as exc:
                print("== continual ==")
                print(f"skipped: {exc}")

            return 0 if ok > 0 else 1

        if args.demo_command == "instance-seg":
            from yolozu.demos.instance_seg import run_instance_seg_demo

            bg = str(getattr(args, "background", "synthetic"))
            resolved_instances = getattr(args, "coco_instances_json", None)
            resolved_images = getattr(args, "coco_images_dir", None)
            if str(bg).strip().lower() == "coco-instances":
                instances_was_default = resolved_instances is None
                images_was_default = resolved_images is None
                if resolved_instances is None:
                    resolved_instances = str(Path("data") / "coco" / "annotations" / "instances_val2017.json")
                if resolved_images is None:
                    resolved_images = str(Path("data") / "coco" / "images" / "val2017")

                # Keep the UX short: `yolozu demo instance-seg` should run even if
                # the user hasn't downloaded COCO instances yet.
                try:
                    instances_ok = Path(str(resolved_instances)).exists() if resolved_instances else False
                    images_ok = Path(str(resolved_images)).exists() if resolved_images else False
                except Exception:
                    instances_ok = False
                    images_ok = False

                if instances_was_default and images_was_default and (not (instances_ok and images_ok)):
                    print(
                        "note: COCO instances not found at the default paths; falling back to --background synthetic. "
                        "To enable coco-instances: python3 scripts/download_coco_instances_tiny.py"
                    )
                    bg = "synthetic"
                    resolved_instances = None
                    resolved_images = None

            resolved_inference = getattr(args, "inference", None)
            if resolved_inference is None:
                resolved_inference = "auto" if str(bg).strip().lower() == "coco-instances" else "none"

            if str(resolved_inference).strip().lower() != "none":
                try:
                    import torch  # noqa: F401
                    import torchvision  # noqa: F401
                except Exception as exc:
                    raise SystemExit(
                        "instance-seg inference requires torch+torchvision. "
                        "Install: python3 -m pip install -U 'yolozu[demo]' (pip) or python3 -m pip install -e '.[demo]' (repo checkout)"
                    ) from exc

            out = run_instance_seg_demo(
                run_dir=getattr(args, "run_dir", None),
                seed=int(getattr(args, "seed", 0)),
                num_images=int(getattr(args, "num_images", 8)),
                image_size=int(getattr(args, "image_size", 96)),
                max_instances=int(getattr(args, "max_instances", 2)),
                background=bg,
                coco_instances_json=resolved_instances,
                coco_images_dir=resolved_images,
                inference=str(resolved_inference),
                device=str(getattr(args, "device", "cpu")),
                score_threshold=float(getattr(args, "score_threshold", 0.5)),
            )
            try:
                cfg_path = Path(out).parent / "demo_config.json"
                cfg_path.write_text(
                    json.dumps(
                        {
                            "kind": "demo_config",
                            "demo": "instance-seg",
                            "background": bg,
                            "seed": int(getattr(args, "seed", 0)),
                            "num_images": int(getattr(args, "num_images", 8)),
                            "image_size": int(getattr(args, "image_size", 96)),
                            "max_instances": int(getattr(args, "max_instances", 2)),
                            "coco_instances_json": (
                                str(resolved_instances)
                                if resolved_instances
                                else None
                            ),
                            "coco_images_dir": (
                                str(resolved_images)
                                if resolved_images
                                else None
                            ),
                            "inference": str(resolved_inference),
                            "device": str(getattr(args, "device", "cpu")),
                            "score_threshold": float(getattr(args, "score_threshold", 0.5)),
                        },
                        indent=2,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"config: {cfg_path}")
            except Exception:
                pass
            _print_instance_seg_report(out_path=Path(out))
            return 0

        if args.demo_command == "continual":
            from yolozu.demos.continual import (
                format_continual_demo_suite_markdown,
                run_continual_demo,
                run_continual_demo_suite,
            )

            def _friendly_demo_deps_error(exc: Exception) -> SystemExit:
                msg = str(exc)
                if "requires torchvision" in msg:
                    return SystemExit(
                        "demo continual (--problem mnist_rotate) requires torchvision. "
                        "Install: python3 -m pip install -U 'yolozu[demo]' (pip) or python3 -m pip install -e '.[demo]' (repo checkout)"
                    )
                if "requires torch" in msg:
                    return SystemExit(
                        "demo continual requires torch. "
                        "Install: python3 -m pip install -U 'yolozu[demo]' (pip) or python3 -m pip install -e '.[demo]' (repo checkout)"
                    )
                return SystemExit(msg)

            methods = None
            if args.methods:
                methods = [str(m) for m in args.methods]
            elif args.compare:
                methods = ["naive", "ewc", "replay", "ewc_replay"]

            # Convenience presets to keep the command short.
            problem = str(getattr(args, "problem", "toy2d"))
            method_override = None
            steps_a = int(args.steps_a)
            steps_b = int(args.steps_b)
            n_train = int(args.n_train)
            n_eval = int(args.n_eval)
            fisher_batches = int(args.fisher_batches)
            replay_capacity = int(args.replay_capacity)
            replay_k = int(args.replay_k)

            if bool(getattr(args, "practical", False)):
                problem = "mnist_rotate"
                # Keep this simple and fast on CPU.
                method_override = "ewc"
                steps_a = min(steps_a, 60)
                steps_b = min(steps_b, 60)
                n_train = min(n_train, 2048)
                n_eval = min(n_eval, 512)
                fisher_batches = min(fisher_batches, 16)

            if bool(getattr(args, "fast", False)):
                steps_a = min(steps_a, 60)
                steps_b = min(steps_b, 60)
                n_train = min(n_train, 2048)
                n_eval = min(n_eval, 512)
                fisher_batches = min(fisher_batches, 16)
                replay_capacity = min(replay_capacity, 256)
                replay_k = min(replay_k, 32)

            if methods and len(methods) > 1:
                try:
                    out = run_continual_demo_suite(
                        methods=methods,
                        output=args.output,
                        seed=int(args.seed),
                        device=str(args.device),
                        problem=str(problem),
                        data_dir=str(getattr(args, "data_dir", str(Path("data") / "torchvision"))),
                        steps_a=int(steps_a),
                        steps_b=int(steps_b),
                        batch_size=int(args.batch_size),
                        hidden=int(args.hidden),
                        lr=float(args.lr),
                        corr=float(args.corr),
                        noise=float(args.noise),
                        n_train=int(n_train),
                        n_eval=int(n_eval),
                        ewc_lambda=float(args.ewc_lambda),
                        fisher_batches=int(fisher_batches),
                        replay_capacity=int(replay_capacity),
                        replay_k=int(replay_k),
                    )
                except RuntimeError as exc:
                    raise _friendly_demo_deps_error(exc)
                try:
                    payload = json.loads(Path(out).read_text(encoding="utf-8"))
                    md = format_continual_demo_suite_markdown(payload)
                    print(md, end="")
                    if args.markdown:
                        md_path = Path(out).with_suffix(".md")
                        md_path.write_text(md, encoding="utf-8")
                        print(str(md_path))
                except Exception:
                    pass
                print(str(out))
                return 0

            method = str(args.method)
            if methods and len(methods) == 1:
                method = str(methods[0])
            if method_override is not None:
                method = str(method_override)

            try:
                out = run_continual_demo(
                    output=args.output,
                    seed=int(args.seed),
                    device=str(args.device),
                    method=method,
                    problem=str(problem),
                    data_dir=str(getattr(args, "data_dir", str(Path("data") / "torchvision"))),
                    steps_a=int(steps_a),
                    steps_b=int(steps_b),
                    batch_size=int(args.batch_size),
                    hidden=int(args.hidden),
                    lr=float(args.lr),
                    corr=float(args.corr),
                    noise=float(args.noise),
                    n_train=int(n_train),
                    n_eval=int(n_eval),
                    ewc_lambda=float(args.ewc_lambda),
                    fisher_batches=int(fisher_batches),
                    replay_capacity=int(replay_capacity),
                    replay_k=int(replay_k),
                )
            except RuntimeError as exc:
                raise _friendly_demo_deps_error(exc)
            try:
                payload = json.loads(Path(out).read_text(encoding="utf-8"))
                metrics = payload.get("metrics", {})
                settings = payload.get("settings", {})
                a = metrics.get("after_task_a", {})
                b = metrics.get("after_task_b", {})
                forgetting = metrics.get("forgetting_acc_a")
                gain = metrics.get("gain_acc_b")
                prob = ""
                model = ""
                bb = ""
                if isinstance(settings, dict):
                    prob = str(settings.get("problem") or "")
                    model = str(settings.get("model") or "")
                    bb = str(settings.get("backbone") or "")
                msg = f"continual demo ({method}): "
                if prob or model:
                    msg += f"problem={prob} model={model} "
                if bb:
                    msg += f"backbone={bb} "
                msg += (
                    f"accA {a.get('acc_a'):.3f}→{b.get('acc_a'):.3f} "
                    f"accB {a.get('acc_b'):.3f}→{b.get('acc_b'):.3f} "
                    f"forget={forgetting:.3f} gain={gain:.3f} "
                    f"(output_dir={Path(out).parent})"
                )
                print(msg)
                if args.markdown:
                    md = format_continual_demo_suite_markdown({"runs": [{"method": method, "metrics": metrics}]})
                    md_path = Path(out).with_suffix(".md")
                    md_path.write_text(md, encoding="utf-8")
                    print(str(md_path))
            except Exception:
                pass
            print(str(out))
            return 0

        if args.demo_command == "keypoints":
            try:
                import torch  # noqa: F401
                import torchvision  # noqa: F401
            except Exception as exc:
                raise SystemExit(
                    "demo keypoints requires torch+torchvision. "
                    "Install: python3 -m pip install -U 'yolozu[demo]' (pip) or python3 -m pip install -e '.[demo]' (repo checkout)"
                ) from exc

            from yolozu.demos.keypoints import run_keypoints_demo

            out = run_keypoints_demo(
                image=getattr(args, "image", None),
                run_dir=getattr(args, "run_dir", None),
                device=str(getattr(args, "device", "auto")),
                score_threshold=float(getattr(args, "score_threshold", 0.7)),
                max_persons=int(getattr(args, "max_persons", 3)),
            )
            try:
                payload = json.loads(Path(out).read_text(encoding="utf-8"))
                res = payload.get("result", {})
                settings = payload.get("settings", {})
                n = res.get("num_persons")
                img = settings.get("image")
                run_dir = settings.get("run_dir")
                print(f"keypoints demo: persons={n} image={img} (output_dir={run_dir})")
                artifacts = res.get("artifacts", {}) if isinstance(res, dict) else {}
                overlay = artifacts.get("overlay") if isinstance(artifacts, dict) else None
                if overlay:
                    print(str(overlay))
            except Exception:
                pass
            print(str(out))
            return 0

        if args.demo_command == "pose":
            try:
                import cv2  # noqa: F401
                import numpy as np  # noqa: F401
            except Exception as exc:
                raise SystemExit(
                    "demo pose requires opencv-python and numpy (aruco backend needs opencv-contrib-python; "
                    "densefusion backend requires CUDA + DenseFusion assets). "
                    "Install: python3 -m pip install -U 'yolozu[demo]' (pip) or python3 -m pip install -e '.[demo]' (repo checkout)"
                ) from exc

            from yolozu.demos.pose6d import run_pose6d_demo

            out = run_pose6d_demo(
                image=getattr(args, "image", None),
                run_dir=getattr(args, "run_dir", None),
                backend=str(getattr(args, "backend", "chessboard")),
                densefusion_root=getattr(args, "densefusion_root", None),
                densefusion_object=str(getattr(args, "densefusion_object", "ape")),
                densefusion_auto_download=bool(getattr(args, "densefusion_auto_download", True)),
                densefusion_model=getattr(args, "densefusion_model", None),
                densefusion_refine_model=getattr(args, "densefusion_refine_model", None),
                pattern_cols=getattr(args, "pattern_cols", None),
                pattern_rows=getattr(args, "pattern_rows", None),
                square_size=float(getattr(args, "square_size", 0.04)),
                aruco_dict=str(getattr(args, "aruco_dict", "DICT_4X4_50")),
                aruco_id=int(getattr(args, "aruco_id", 23)),
                marker_length=float(getattr(args, "marker_length", 0.05)),
                camera_fx=getattr(args, "camera_fx", None),
                camera_fy=getattr(args, "camera_fy", None),
                camera_cx=getattr(args, "camera_cx", None),
                camera_cy=getattr(args, "camera_cy", None),
                sample_source=str(getattr(args, "sample_source", "auto")),
            )
            try:
                payload = json.loads(Path(out).read_text(encoding="utf-8"))
                res = payload.get("result", {})
                settings = payload.get("settings", {})
                run_dir = settings.get("run_dir")
                t_xyz = res.get("t_xyz")
                backend = settings.get("backend")
                print(f"pose demo ({backend}): t_xyz={t_xyz} (output_dir={run_dir})")
                artifacts = res.get("artifacts", {}) if isinstance(res, dict) else {}
                overlay = artifacts.get("overlay") if isinstance(artifacts, dict) else None
                if overlay:
                    print(str(overlay))
            except Exception:
                pass
            print(str(out))
            return 0

        if args.demo_command == "depth":
            try:
                import torch  # noqa: F401
            except Exception as exc:
                raise SystemExit(
                    "demo depth requires torch. "
                    "Install: python3 -m pip install -U 'yolozu[demo]' (pip) or python3 -m pip install -e '.[demo]' (repo checkout)"
                ) from exc

            from yolozu.demos.depth import run_depth_demo

            try:
                out = run_depth_demo(
                    image=getattr(args, "image", None),
                    run_dir=getattr(args, "run_dir", None),
                    device=str(getattr(args, "device", "auto")),
                    model=str(getattr(args, "model", "midas_small")),
                    invert=bool(getattr(args, "invert", True)),
                    compare=bool(getattr(args, "compare", False)),
                )
            except Exception as exc:
                msg = str(exc)
                if "intel-isl/MiDaS" in msg or "torch.hub" in msg:
                    raise SystemExit(
                        "demo depth uses MiDaS via torch.hub and downloads weights on first run; ensure network access. "
                        f"error: {exc}"
                    )
                raise

            try:
                payload = json.loads(Path(out).read_text(encoding="utf-8"))
                res = payload.get("result", {})
                settings = payload.get("settings", {})
                run_dir = settings.get("run_dir")

                models = res.get("models")
                if isinstance(models, list) and models:
                    names = ",".join([str(m.get("model")) for m in models])
                    print(f"depth demo: compare=[{names}] (output_dir={run_dir})")
                else:
                    d = (res.get("depth") or {})
                    print(
                        f"depth demo: model={settings.get('model')} depth_range=[{d.get('min'):.3g}, {d.get('max'):.3g}] (output_dir={run_dir})"
                    )
            except Exception:
                pass
            print(str(out))
            return 0

        if args.demo_command == "train":
            try:
                import torch  # noqa: F401
                import torchvision  # noqa: F401
            except Exception as exc:
                raise SystemExit(
                    "demo train requires torch+torchvision. "
                    "Install: python3 -m pip install -U 'yolozu[demo]' (pip) or python3 -m pip install -e '.[demo]' (repo checkout)"
                ) from exc

            from yolozu.demos.train import run_train_demo

            out = run_train_demo(
                output=getattr(args, "output", None),
                seed=int(getattr(args, "seed", 0)),
                device=str(getattr(args, "device", "cpu")),
                data_dir=str(getattr(args, "data_dir", str(Path("data") / "torchvision"))),
                epochs=int(getattr(args, "epochs", 1)),
                max_steps=int(getattr(args, "max_steps", 80)),
                batch_size=int(getattr(args, "batch_size", 64)),
                lr=float(getattr(args, "lr", 3e-4)),
            )
            try:
                payload = json.loads(Path(out).read_text(encoding="utf-8"))
                res = payload.get("result", {})
                train = res.get("train", {})
                val = res.get("val", {})
                settings = payload.get("settings", {})
                print(
                    "train demo: "
                    f"steps={train.get('steps')} loss_mean={train.get('loss_mean'):.3f} "
                    f"val_acc={val.get('acc'):.3f} (output_dir={settings.get('run_dir')})"
                )
            except Exception:
                pass
            print(str(out))
            return 0

    raise SystemExit("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
