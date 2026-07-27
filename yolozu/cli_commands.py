"""YOLOZU command-line interface.

Provides the ``yolozu`` CLI with subcommands for training, evaluation,
export, doctor diagnostics, dataset migration, model fetching, and
demo pipelines.
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


from yolozu.core.cli_args import parse_image_size_arg, require_non_negative_int
from yolozu.core.config import simple_yaml_load

__all__: list[str] = []


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_run_meta(cwd: Path) -> dict[str, Any]:
    from yolozu.core import doctor as doctor_mod

    info = doctor_mod._gather_git_info(cwd=cwd)
    return {"head": info.get("head"), "dirty": info.get("dirty")}


def _gpu_run_meta() -> dict[str, Any]:
    from yolozu.core import doctor as doctor_mod

    return doctor_mod._gather_gpu_info()


def _base_run_meta(*, seed: int | None, notes: str | None, config_fingerprint: dict[str, Any]) -> dict[str, Any]:
    from yolozu.inference.export_orchestrator import sha256_json

    cwd = Path.cwd()
    return {
        "timestamp": _now_utc(),
        "seed": seed,
        "notes": notes,
        "config_hash": sha256_json(config_fingerprint),
        "git": _git_run_meta(cwd),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "gpu": _gpu_run_meta(),
        "env": {
            "python_executable": sys.executable,
            "cwd": str(cwd),
        },
    }


def _subprocess_or_die(cmd: list[str]) -> str:
    module_command = len(cmd) >= 3 and cmd[1] == "-m"
    if len(cmd) >= 2:
        candidate = Path(str(cmd[1]))
        if candidate.suffix == ".py":
            script_path = candidate if candidate.is_absolute() else (_repo_root() / candidate)
            if not script_path.is_file():
                raise SystemExit(f"required script not found: {candidate}")
    proc = subprocess.run(
        cmd,
        cwd=str(Path.cwd() if module_command else _repo_root()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.stderr and proc.stderr.strip():
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def _repo_manifest_path() -> Path:
    return _repo_root() / "tools" / "manifest.json"


def _packaged_manifest_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "manifest" / "tools_manifest.json"


def _load_tool_manifest(*, prefer_repo: bool = True) -> tuple[dict[str, Any], Path, bool]:
    repo_manifest = _repo_manifest_path()
    if prefer_repo and repo_manifest.is_file():
        return json.loads(repo_manifest.read_text(encoding="utf-8")), repo_manifest, True
    packaged_manifest = _packaged_manifest_path()
    if packaged_manifest.is_file():
        return json.loads(packaged_manifest.read_text(encoding="utf-8")), packaged_manifest, False
    raise SystemExit("could not locate a tool manifest (repo or packaged copy)")


def _registry_payload(*, manifest: dict[str, Any], tool: dict[str, Any] | None = None) -> dict[str, Any]:
    if tool is not None:
        return {
            "kind": "yolozu_tool_spec",
            "schema_version": 1,
            "timestamp": _now_utc(),
            "repo": manifest.get("repo"),
            "contracts": manifest.get("contracts"),
            "tool": tool,
        }
    return {
        "kind": "yolozu_tool_registry",
        "schema_version": 1,
        "timestamp": _now_utc(),
        "repo": manifest.get("repo"),
        "contracts": manifest.get("contracts"),
        "tools": manifest.get("tools") or [],
    }


def _find_flag_value(argv: list[str], flag: str) -> str | None:
    for i in range(len(argv) - 1):
        if argv[i] == flag:
            return argv[i + 1]
    return None


def _find_flag_value_any(argv: list[str], flag: str) -> str | None:
    value = _find_flag_value(argv, flag)
    if value is not None:
        return value
    prefix = flag + "="
    for token in argv:
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _extract_forwarded_flags(argv: list[str]) -> set[str]:
    flags: set[str] = set()
    for token in argv:
        if token == "--":
            continue
        if token == "-h":
            flags.add("-h")
            continue
        if token.startswith("--"):
            flags.add(token.split("=", 1)[0])
    return flags


def _is_repo_relative_path_like(value: str) -> bool:
    if not isinstance(value, str) or not value or value.startswith("/"):
        return False
    parts = Path(value).parts
    return ".." not in parts


def _within(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _parse_contract_validator_cmd(template: str, *, path: str) -> list[str] | None:
    if not isinstance(template, str) or not template.strip():
        return None
    cleaned = " ".join(token for token in template.split() if not (token.startswith("[") and token.endswith("]")))
    if "<path>" not in cleaned:
        return None
    try:
        tokens = shlex.split(cleaned)
    except Exception:
        tokens = cleaned.split()
    return [path if token == "<path>" else token for token in tokens]


def _resolve_auto_dataset_from_args(args: argparse.Namespace) -> str:
    if getattr(args, "instances", None) and getattr(args, "images_dir", None):
        return "coco-instances"
    if getattr(args, "data", None):
        return "ultralytics"
    dataset_path = getattr(args, "dataset", None)
    if dataset_path:
        from yolozu.dataset import inspect_dataset_layout

        info = inspect_dataset_layout(str(dataset_path), split=str(args.split) if getattr(args, "split", None) else None)
        if info is None:
            raise SystemExit(f"could not auto-detect dataset source from path: {dataset_path}")
        fmt = str(info.get("format") or "")
        if fmt in ("coco_root", "coco_keypoints_root", "yolozu_coco_wrapper"):
            return "coco"
        if fmt in (
            "yolozu_segmentation_descriptor",
            "voc_segmentation_root",
            "cityscapes_segmentation_root",
            "ade20k_segmentation_root",
        ):
            return "segmentation"
        if fmt in ("ultralytics_data_yaml", "yolo_layout", "yolozu_wrapper"):
            return "ultralytics"
    raise SystemExit(
        "could not auto-detect dataset source; provide --dataset, --data (ultralytics), or --instances + --images-dir (coco-instances)"
    )


def _segmentation_output_path(output: str | Path) -> Path:
    out_path = Path(output)
    if out_path.suffix.lower() == ".json":
        return out_path
    return out_path / "dataset.json"


def _copy_descriptor_file(source: Path, output: str | Path, *, force: bool) -> Path:
    out_path = _segmentation_output_path(output)
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        raise FileExistsError(f"output already exists: {out_path} (use --force to overwrite)")
    out_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return out_path


def _write_segmentation_descriptor_from_layout(
    *,
    dataset_path: str | Path,
    layout_info: dict[str, Any],
    output: str | Path,
    split: str | None,
    force: bool,
) -> Path:
    from yolozu.migrate import migrate_seg_dataset_descriptor

    dataset_root = Path(dataset_path)
    if not dataset_root.is_absolute():
        dataset_root = (Path.cwd() / dataset_root).resolve()

    fmt = str(layout_info.get("format") or "")
    if fmt == "yolozu_segmentation_descriptor":
        descriptor = dataset_root if dataset_root.is_file() else dataset_root / "dataset.json"
        return _copy_descriptor_file(descriptor, output, force=force)

    if fmt == "voc_segmentation_root":
        from_format = "voc"
    elif fmt == "cityscapes_segmentation_root":
        from_format = "cityscapes"
    elif fmt == "ade20k_segmentation_root":
        from_format = "ade20k"
    else:
        raise ValueError(f"unsupported segmentation layout: {fmt}")

    return migrate_seg_dataset_descriptor(
        from_format=from_format,
        root=str(layout_info.get("root") or dataset_root),
        split=str(split or layout_info.get("split") or "val"),
        output=_segmentation_output_path(output),
        path_type="absolute",
        mode="manifest",
        force=force,
    )


def _summarize_segmentation_layout(*, dataset_path: str | Path, layout_info: dict[str, Any], split: str | None) -> dict[str, Any]:
    fmt = str(layout_info.get("format") or "")
    dataset_root = Path(dataset_path)
    if not dataset_root.is_absolute():
        dataset_root = (Path.cwd() / dataset_root).resolve()

    if fmt == "yolozu_segmentation_descriptor":
        from yolozu.segmentation_dataset import load_seg_dataset_descriptor

        descriptor_path = dataset_root if dataset_root.is_file() else dataset_root / "dataset.json"
        desc = load_seg_dataset_descriptor(descriptor_path)
        return {
            "from": "segmentation",
            "layout": layout_info,
            "task_family": "segmentation",
            "reference_trainer": _reference_trainer_readiness(task_family="segmentation", source_format=fmt),
            "dataset": desc.dataset,
            "split": desc.split,
            "counts": {"images": int(len(desc.samples)), "masks": int(sum(1 for sample in desc.samples if sample.mask is not None))},
            "classes_preview": list((desc.classes or [])[:20]),
            "ignore_index": int(desc.ignore_index),
        }

    if fmt == "voc_segmentation_root":
        from yolozu.datasets.pascal_voc import iter_pascal_voc_seg_samples

        samples = list(iter_pascal_voc_seg_samples(dataset_root, split=str(split or layout_info.get("split") or "val")))
        return {
            "from": "segmentation",
            "layout": layout_info,
            "task_family": "segmentation",
            "reference_trainer": _reference_trainer_readiness(task_family="segmentation", source_format=fmt),
            "dataset": str(layout_info.get("dataset_name") or "pascal_voc"),
            "split": str(split or layout_info.get("split") or "val"),
            "counts": {"images": int(len(samples)), "masks": int(sum(1 for sample in samples if sample.mask_path is not None))},
        }

    if fmt == "cityscapes_segmentation_root":
        from yolozu.datasets.cityscapes import iter_cityscapes_samples

        samples = list(iter_cityscapes_samples(dataset_root, split=str(split or layout_info.get("split") or "val")))
        return {
            "from": "segmentation",
            "layout": layout_info,
            "task_family": "segmentation",
            "reference_trainer": _reference_trainer_readiness(task_family="segmentation", source_format=fmt),
            "dataset": "cityscapes",
            "split": str(split or layout_info.get("split") or "val"),
            "counts": {"images": int(len(samples)), "masks": int(sum(1 for sample in samples if sample.mask_path is not None))},
        }

    if fmt == "ade20k_segmentation_root":
        from yolozu.datasets.ade20k import iter_ade20k_samples, load_ade20k_classes

        samples = list(iter_ade20k_samples(dataset_root, split=str(split or layout_info.get("split") or "val")))
        classes = load_ade20k_classes(dataset_root) or []
        return {
            "from": "segmentation",
            "layout": layout_info,
            "task_family": "segmentation",
            "reference_trainer": _reference_trainer_readiness(task_family="segmentation", source_format=fmt),
            "dataset": "ade20k",
            "split": str(split or layout_info.get("split") or "val"),
            "counts": {"images": int(len(samples)), "masks": int(sum(1 for sample in samples if sample.mask_path is not None))},
            "classes_preview": list(classes[:20]),
        }

    raise ValueError(f"unsupported segmentation layout: {fmt}")


def _reference_trainer_readiness(*, task_family: str, source_format: str, label_format: str | None = None) -> dict[str, Any]:
    task = str(task_family or "bbox").strip().lower()
    source = str(source_format or "").strip().lower()
    label = str(label_format or "").strip().lower()
    accepted_inputs = [
        "YOLO-style images/<split> + labels/<split>",
        "YOLO/Ultralytics data.yaml for detection labels",
        "dataset.json with images_dir and labels_dir pointing to YOLO label files",
        "records JSON using image/image_path and normalized bbox labels",
    ]

    if task in {"bbox", "detect", "detection"} and label not in {"segment", "seg", "polygon", "yolo-seg", "yolo_seg"}:
        direct = source in {"ultralytics", "ultralytics_data_yaml", "yolo", "yolozu_wrapper", "data_yaml", "yolo_layout"}
        return {
            "task_family": "bbox",
            "direct_train_ready": bool(direct),
            "train_ready_after_migration": True,
            "requires_normalization": not bool(direct),
            "accepted_inputs": accepted_inputs,
            "reason": (
                "already resolves to image files and YOLO detection labels"
                if direct
                else "native source must be migrated to YOLO labels or record JSON before reference training"
            ),
        }

    if task in {"keypoints", "pose"}:
        direct = source in {"ultralytics", "ultralytics_data_yaml", "yolo", "yolozu_wrapper", "data_yaml", "yolo_layout"}
        return {
            "task_family": "keypoints",
            "direct_train_ready": bool(direct),
            "train_ready_after_migration": True,
            "requires_normalization": not bool(direct),
            "accepted_inputs": accepted_inputs,
            "reason": (
                "already resolves to YOLO pose/keypoint labels"
                if direct
                else "COCO keypoints must be imported/exported to YOLO keypoint labels or record JSON before reference training"
            ),
        }

    if task in {"classification", "classify", "cls"}:
        return {
            "task_family": "classification",
            "direct_train_ready": False,
            "train_ready_after_migration": False,
            "requires_normalization": True,
            "accepted_inputs": [
                "classification folder layout such as train/<class>/*.jpg",
                "classification label manifest with image_path and class_id/class_name",
            ],
            "reason": "classification intake is recognized for preflight and external lanes, but the RT-DETR reference trainer does not consume classification-only records",
        }

    if task == "obb":
        return {
            "task_family": "obb",
            "direct_train_ready": False,
            "train_ready_after_migration": False,
            "requires_normalization": True,
            "accepted_inputs": [
                "YOLO OBB labels with class cx cy w h angle",
                "YOLO OBB labels with class x1 y1 x2 y2 x3 y3 x4 y4",
                "records JSON labels with bbox plus angle/obb",
            ],
            "reason": "OBB intake is recognized for validation and external lanes, but the RT-DETR reference trainer does not train rotated boxes directly",
        }

    if task == "depth":
        direct = source in {"ultralytics", "ultralytics_data_yaml", "yolo", "yolozu_wrapper", "data_yaml", "yolo_layout", "records"}
        return {
            "task_family": "depth",
            "direct_train_ready": bool(direct),
            "train_ready_after_migration": bool(direct),
            "requires_normalization": not bool(direct),
            "accepted_inputs": accepted_inputs
            + ["sidecar JSON with depth_path/depth/D_obj next to each label file or in records JSON"],
            "reason": (
                "bbox labels plus depth sidecars can be consumed by the reference trainer when depth fields are present"
                if direct
                else "depth training requires normalized bbox records plus depth sidecars"
            ),
        }

    if task in {"pose6d", "6dof", "pose_6d", "pose-6d"}:
        direct = source in {"ultralytics", "ultralytics_data_yaml", "yolo", "yolozu_wrapper", "data_yaml", "yolo_layout", "records"}
        return {
            "task_family": "pose6d",
            "direct_train_ready": bool(direct),
            "train_ready_after_migration": bool(direct),
            "requires_normalization": not bool(direct),
            "accepted_inputs": accepted_inputs
            + ["sidecar JSON with R_gt/t_gt or pose plus K_gt/intrinsics next to each label file or in records JSON"],
            "reason": (
                "bbox labels plus pose/intrinsics sidecars can be consumed by the reference trainer when pose fields are present"
                if direct
                else "pose6d training requires normalized bbox records plus pose/intrinsics sidecars"
            ),
        }

    return {
        "task_family": task or "unknown",
        "direct_train_ready": False,
        "train_ready_after_migration": False,
        "requires_normalization": True,
        "accepted_inputs": accepted_inputs,
        "reason": "reference trainer does not consume semantic-segmentation descriptors or masks as a direct training dataset",
    }


def _validate_segmentation_layout(
    *,
    dataset_path: str | Path,
    layout_info: dict[str, Any],
    max_images: int | None,
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    dataset_root = Path(dataset_path)
    if not dataset_root.is_absolute():
        dataset_root = (Path.cwd() / dataset_root).resolve()
    fmt = str(layout_info.get("format") or "")

    if fmt == "yolozu_segmentation_descriptor":
        from yolozu.segmentation_dataset import load_seg_dataset_descriptor, resolve_dataset_path

        descriptor_path = dataset_root if dataset_root.is_file() else dataset_root / "dataset.json"
        desc = load_seg_dataset_descriptor(descriptor_path)
        samples = list(desc.samples)
        if max_images is not None:
            samples = samples[: int(max_images)]
        if not samples:
            errors.append("dataset contains no segmentation samples")
        for idx, sample in enumerate(samples):
            image_path = resolve_dataset_path(sample.image, dataset_root=descriptor_path.parent, path_type=desc.path_type)
            if not image_path.exists():
                errors.append(f"samples[{idx}]: image file does not exist: {image_path}")
            if sample.mask is not None:
                mask_path = resolve_dataset_path(sample.mask, dataset_root=descriptor_path.parent, path_type=desc.path_type)
                if not mask_path.exists():
                    errors.append(f"samples[{idx}]: mask file does not exist: {mask_path}")
        return warnings, errors

    if fmt == "voc_segmentation_root":
        from yolozu.datasets.pascal_voc import iter_pascal_voc_seg_samples

        samples = list(iter_pascal_voc_seg_samples(dataset_root, split=str(layout_info.get("split") or "val")))
        if max_images is not None:
            samples = samples[: int(max_images)]
        if not samples:
            errors.append("dataset contains no segmentation samples")
        for idx, sample in enumerate(samples):
            if not sample.image_path.exists():
                errors.append(f"samples[{idx}]: image file does not exist: {sample.image_path}")
            if sample.mask_path is None or not sample.mask_path.exists():
                errors.append(f"samples[{idx}]: mask file does not exist: {sample.mask_path}")
        return warnings, errors

    if fmt == "cityscapes_segmentation_root":
        from yolozu.datasets.cityscapes import iter_cityscapes_samples

        samples = list(iter_cityscapes_samples(dataset_root, split=str(layout_info.get("split") or "val")))
        if max_images is not None:
            samples = samples[: int(max_images)]
        if not samples:
            errors.append("dataset contains no segmentation samples")
        for idx, sample in enumerate(samples):
            if not sample.image_path.exists():
                errors.append(f"samples[{idx}]: image file does not exist: {sample.image_path}")
            if sample.mask_path is None or not sample.mask_path.exists():
                errors.append(f"samples[{idx}]: mask file does not exist: {sample.mask_path}")
        return warnings, errors

    if fmt == "ade20k_segmentation_root":
        from yolozu.datasets.ade20k import iter_ade20k_samples

        samples = list(iter_ade20k_samples(dataset_root, split=str(layout_info.get("split") or "val")))
        if max_images is not None:
            samples = samples[: int(max_images)]
        if not samples:
            errors.append("dataset contains no segmentation samples")
        for idx, sample in enumerate(samples):
            if not sample.image_path.exists():
                errors.append(f"samples[{idx}]: image file does not exist: {sample.image_path}")
            if sample.mask_path is None or not sample.mask_path.exists():
                errors.append(f"samples[{idx}]: mask file does not exist: {sample.mask_path}")
        return warnings, errors

    raise ValueError(f"unsupported segmentation layout: {fmt}")


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


def _cmd_train_external(args: argparse.Namespace, extra_args: list[str] | None = None) -> int:
    backend = str(getattr(args, "external_backend", "") or "").strip().lower()
    backend_to_subcommand = {
        "yolox": "train-yolox",
        "detectron2": "train-detectron2",
        "mmdetection": "train-mmdetection",
        "mmpose": "train-mmpose",
        "mmseg": "train-mmseg",
        "tao": "train-tao",
        "ultralytics": "train-ultralytics",
        "hf-detr": "train-hf-detr",
    }
    if backend not in backend_to_subcommand:
        raise SystemExit("unsupported --external-backend value")

    config_value = str(getattr(args, "config", "") or "").strip()
    if backend == "yolox" and not config_value:
        raise SystemExit("train config/exp is required when using --external-backend yolox")
    if backend in {"detectron2", "mmdetection", "mmpose", "mmseg", "tao"} and not config_value:
        raise SystemExit(f"train config is required when using --external-backend {backend}")

    repo_root = Path(__file__).resolve().parents[1]
    helper = repo_root / "tools" / "support_external_training.py"
    if not helper.is_file():
        raise SystemExit(
            "train --external-backend requires a YOLOZU repo checkout with "
            "tools/support_external_training.py available"
        )

    cmd = [sys.executable, str(helper), backend_to_subcommand[backend]]
    if backend == "yolox":
        cmd.extend(["--exp", config_value])
    elif backend in {"detectron2", "mmdetection", "mmpose", "mmseg", "tao"}:
        cmd.extend(["--config", config_value])
    elif backend == "ultralytics" and config_value:
        cmd.extend(["--model", config_value])
    elif backend == "hf-detr" and config_value:
        cmd.extend(["--model-id", config_value])
    if extra_args:
        cmd.extend(list(extra_args))

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    return int(proc.returncode)


def _cmd_train_orchestrate(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    helper = repo_root / "tools" / "orchestrate_train.py"
    if not helper.is_file():
        raise SystemExit("missing tools/orchestrate_train.py")
    cmd = [sys.executable, str(helper), "--spec", str(args.spec), "--output", str(args.output)]
    if getattr(args, "registry_out", None):
        cmd.extend(["--registry-out", str(args.registry_out)])
    if bool(getattr(args, "execute", False)):
        cmd.append("--execute")
    if bool(getattr(args, "dry_run", False)):
        cmd.append("--dry-run")
    if bool(getattr(args, "stop_on_failure", False)):
        cmd.append("--stop-on-failure")
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    return int(proc.returncode)


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


def _cmd_doctor(output: str, *, explain: bool = False, proof: bool = False, proof_dir: str = "reports/doctor_proof") -> int:
    from yolozu.doctor import write_doctor_report

    return int(write_doctor_report(output=output, explain=explain, proof=proof, proof_dir=proof_dir))


def _cmd_registry_validate(_: argparse.Namespace) -> int:
    repo_root = _repo_root()
    manifest_path = repo_root / "tools" / "manifest.json"
    validator = repo_root / "tools" / "validate_tool_manifest.py"
    if not manifest_path.is_file() or not validator.is_file():
        raise SystemExit(
            "yolozu registry validate requires a repo checkout with tools/manifest.json "
            "and tools/validate_tool_manifest.py available"
        )
    proc = subprocess.run(
        [sys.executable, str(validator), "--manifest", str(manifest_path), "--require-declarative"],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    return int(proc.returncode)


def _cmd_registry_list(args: argparse.Namespace) -> int:
    manifest, _, _ = _load_tool_manifest()
    tools = list(manifest.get("tools") or [])
    tags = getattr(args, "tag", None) or []
    contracts = getattr(args, "contract", None) or []

    def _tool_matches(tool: dict[str, Any]) -> bool:
        if tags:
            tool_tags = set(tool.get("tags") or [])
            if not all(tag in tool_tags for tag in tags):
                return False
        if contracts:
            contract_spec = tool.get("contracts") or {}
            consumes = set(contract_spec.get("consumes") or [])
            produces = set(contract_spec.get("produces") or [])
            have = consumes | produces
            if not all(contract_id in have for contract_id in contracts):
                return False
        return True

    tools = [tool for tool in tools if isinstance(tool, dict) and _tool_matches(tool)]
    if bool(getattr(args, "json", False)):
        payload = _registry_payload(manifest=manifest)
        payload["tools"] = tools
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    for tool in tools:
        print(f"- {tool.get('id')}: {tool.get('summary')} ({tool.get('runner')} {tool.get('entrypoint')})")
    return 0


def _cmd_registry_show(args: argparse.Namespace) -> int:
    tool_id = str(getattr(args, "id"))
    manifest, _, _ = _load_tool_manifest()
    matches = [tool for tool in (manifest.get("tools") or []) if isinstance(tool, dict) and tool.get("id") == tool_id]
    if not matches:
        raise SystemExit(f"unknown tool id: {tool_id}")
    tool = matches[0]
    if bool(getattr(args, "json", False)):
        print(json.dumps(_registry_payload(manifest=manifest, tool=tool), indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    print(json.dumps(tool, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def _cmd_registry_run(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    manifest_path = repo_root / "tools" / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("yolozu registry run requires a repo checkout with tools/manifest.json available")

    manifest, _, _ = _load_tool_manifest(prefer_repo=True)
    tool_id = str(getattr(args, "id"))
    forwarded = getattr(args, "forward_args", None)
    forward_args: list[str] = [str(item) for item in forwarded] if isinstance(forwarded, list) else []
    if forward_args and forward_args[0] == "--":
        forward_args = forward_args[1:]

    matches = [tool for tool in (manifest.get("tools") or []) if isinstance(tool, dict) and tool.get("id") == tool_id]
    if not matches:
        raise SystemExit(f"unknown tool id: {tool_id}")
    tool = matches[0]

    requires = tool.get("requires") or {}
    platform_spec = tool.get("platform") or {}
    if bool(requires.get("network")) and not bool(getattr(args, "allow_network", False)):
        raise SystemExit("tool requires network access; rerun with --allow-network")
    if bool(platform_spec.get("gpu_required")) and not bool(getattr(args, "allow_gpu", False)):
        raise SystemExit("tool requires GPU; rerun with --allow-gpu")

    allowed_write_roots = list(getattr(args, "allow_write_root", None) or ["reports"])
    allow_unsafe_paths = bool(getattr(args, "allow_unsafe_paths", False))
    dry_run = bool(getattr(args, "dry_run", False))
    allow_undeclared_effects = bool(getattr(args, "allow_undeclared_effects", False))
    allow_unknown_flags_cli = bool(getattr(args, "allow_unknown_flags", False))

    declared_flags: set[str] = set()
    for item in (tool.get("inputs") or []):
        if isinstance(item, dict) and isinstance(item.get("flag"), str) and item.get("flag"):
            declared_flags.add(str(item["flag"]))

    effects = tool.get("effects")
    if effects is None:
        if not allow_undeclared_effects:
            raise SystemExit(
                "tool has no declarative effects metadata (tool.effects). "
                "Add effects to tools/manifest.json or rerun with --allow-undeclared-effects."
            )
        effects = {}
    if not isinstance(effects, dict):
        raise SystemExit("invalid tool.effects (expected object)")

    allow_unknown_flags = bool(effects.get("allow_unknown_flags", False) or allow_unknown_flags_cli)
    forwarded_flags = _extract_forwarded_flags(forward_args)
    always_ok = {"-h", "--help"}
    unknown = sorted(flag for flag in forwarded_flags if flag not in always_ok and flag not in declared_flags)
    if unknown and not allow_unknown_flags:
        raise SystemExit(
            "unknown forwarded flags (not declared in tools/manifest.json inputs):\n"
            + "\n".join(f"- {flag}" for flag in unknown)
            + "\nUse --allow-unknown-flags to bypass (not recommended for agents)."
        )

    runner = tool.get("runner")
    entrypoint = tool.get("entrypoint")
    if runner not in {"python3", "bash"}:
        raise SystemExit("unsupported runner")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise SystemExit("missing entrypoint")
    if entrypoint.startswith("/") or ".." in Path(entrypoint).parts:
        raise SystemExit("invalid entrypoint path")
    entry_path = repo_root / entrypoint
    if not entry_path.exists():
        raise SystemExit(f"entrypoint not found: {entrypoint}")

    cmd: list[str] = ["python3", str(entry_path)] if runner == "python3" else ["bash", str(entry_path)]
    cmd.extend(forward_args)

    roots: list[Path] = []
    for root in allowed_write_roots:
        if not isinstance(root, str) or not root:
            continue
        if root.startswith("/") or ".." in Path(root).parts:
            raise SystemExit(f"invalid --allow-write-root: {root}")
        roots.append(repo_root / root)

    def _check_write_path(src: str, value: str) -> None:
        if not isinstance(value, str) or not value or value.strip() == "-":
            return
        if (value.startswith("/") or ".." in Path(value).parts) and not allow_unsafe_paths:
            raise SystemExit(f"unsafe path blocked ({src}): {value} (use --allow-unsafe-paths to override)")
        if _is_repo_relative_path_like(value):
            resolved = repo_root / value
            if roots and not any(_within(root, resolved) for root in roots):
                roots_str = ", ".join(str(Path(root).relative_to(repo_root)) for root in roots)
                raise SystemExit(
                    f"write path blocked ({src}): {value} is outside allowed roots: {roots_str} "
                    "(use --allow-write-root to add a root)"
                )

    fixed = effects.get("fixed_writes")
    if fixed is not None:
        if not isinstance(fixed, list):
            raise SystemExit("invalid tool.effects.fixed_writes (expected list)")
        for index, item in enumerate(fixed):
            if not isinstance(item, dict):
                raise SystemExit(f"invalid tool.effects.fixed_writes[{index}] (expected object)")
            path = item.get("path")
            if isinstance(path, str) and path:
                _check_write_path(f"fixed_writes[{index}]", path)

    writes = effects.get("writes")
    if writes is not None:
        if not isinstance(writes, list):
            raise SystemExit("invalid tool.effects.writes (expected list)")
        for index, item in enumerate(writes):
            if not isinstance(item, dict):
                raise SystemExit(f"invalid tool.effects.writes[{index}] (expected object)")
            flag = item.get("flag")
            if not isinstance(flag, str) or not flag.startswith("--"):
                raise SystemExit(f"invalid tool.effects.writes[{index}].flag")
            value = _find_flag_value_any(forward_args, flag)
            if value is None:
                default_value: str | None = None
                for declared in (tool.get("inputs") or []):
                    if isinstance(declared, dict) and declared.get("flag") == flag:
                        default = declared.get("default")
                        if isinstance(default, str) and default:
                            default_value = default
                        break
                if default_value is None:
                    continue
                if "<" in default_value or ">" in default_value:
                    raise SystemExit(
                        f"write effect default contains placeholder; pass an explicit value for {flag}: {default_value}"
                    )
                value = default_value
            _check_write_path(flag, value)

    if dry_run:
        print("DRY_RUN:")
        print(" ".join(shlex.quote(item) for item in cmd))
        return 0

    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    contracts_registry = manifest.get("contracts") or {}
    produces = (tool.get("contracts") or {}).get("produces") or []
    contract_outputs = tool.get("contract_outputs") or {}
    for contract_id in produces:
        if not isinstance(contract_id, str) or not contract_id:
            continue
        spec = contracts_registry.get(contract_id) if isinstance(contracts_registry, dict) else None
        if not isinstance(spec, dict):
            continue
        validator_tpl = spec.get("validator")
        if not isinstance(validator_tpl, str) or not validator_tpl.strip():
            continue

        output_name = contract_outputs.get(contract_id) if isinstance(contract_outputs, dict) else None
        output_default: str | None = None
        if isinstance(output_name, str) and output_name:
            for output in (tool.get("outputs") or []):
                if isinstance(output, dict) and output.get("name") == output_name:
                    default = output.get("default")
                    if isinstance(default, str) and default:
                        output_default = default
                    break

        out_path = _find_flag_value_any(forward_args, "--output") or output_default
        if not out_path:
            continue
        validator_cmd = _parse_contract_validator_cmd(validator_tpl, path=out_path)
        if not validator_cmd:
            continue
        subprocess.run(validator_cmd, cwd=str(repo_root), check=True)

    return 0


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


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from yolozu.eval.benchmark_mode import run_benchmark_mode

    try:
        report, code = run_benchmark_mode(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    if bool(getattr(args, "verbose", False)):
        for item in report.get("results", []):
            print(f"{item.get('format')}: {item.get('status')} ({item.get('skip_reason') or item.get('latency_source')})")
    print(str(getattr(args, "output")))
    return int(code)


def _cmd_doctor_import(args: argparse.Namespace) -> int:
    import time

    from yolozu.coco_convert import build_category_map_from_coco
    from yolozu.dataset import build_manifest, inspect_dataset_layout
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
    dataset_arg = getattr(args, "dataset", None)
    if not dataset_from and not config_from:
        raise SystemExit("doctor import requires at least one of: --dataset-from, --config-from")

    layout_info = None
    if dataset_arg:
        layout_info = inspect_dataset_layout(str(dataset_arg), split=str(args.split) if getattr(args, "split", None) else None)
        if layout_info is None and dataset_from:
            raise SystemExit(f"could not inspect dataset layout: {dataset_arg}")

    if dataset_from and str(dataset_from).strip().lower() == "auto":
        if layout_info is not None:
            fmt = str(layout_info.get("format") or "")
            if fmt in ("coco_root", "coco_keypoints_root", "yolozu_coco_wrapper"):
                dataset_from = "coco"
            elif fmt in (
                "yolozu_segmentation_descriptor",
                "voc_segmentation_root",
                "cityscapes_segmentation_root",
                "ade20k_segmentation_root",
            ):
                dataset_from = "segmentation"
            else:
                dataset_from = "ultralytics"
            report["warnings"].append(f"dataset source auto-detected: {dataset_from} (layout: {fmt})")
        else:
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
                "task_family": "bbox",
                "reference_trainer": _reference_trainer_readiness(task_family="bbox", source_format="coco-instances"),
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
        elif src in {"coco", "coco-keypoints"}:
            dataset_path = getattr(args, "dataset", None)
            if not dataset_path:
                raise SystemExit(f"--dataset is required for --dataset-from {src}")
            info = layout_info or inspect_dataset_layout(str(dataset_path), split=str(args.split) if getattr(args, "split", None) else None)
            expected_formats = ("coco_keypoints_root",) if src == "coco-keypoints" else ("coco_root", "coco_keypoints_root")
            if info is None or str(info.get("format") or "") not in expected_formats:
                raise SystemExit(f"dataset is not a detectable {src} root: {dataset_path}")
            instances_path = Path(str(info.get("instances_json") or "")).expanduser()
            images_dir = Path(str(info.get("images_dir") or "")).expanduser()
            instances_doc = json.loads(instances_path.read_text(encoding="utf-8"))
            images = instances_doc.get("images") or []
            annotations = instances_doc.get("annotations") or []
            include_crowd = bool(getattr(args, "include_crowd", False))
            if not include_crowd and isinstance(annotations, list):
                annotations = [a for a in annotations if not (isinstance(a, dict) and int(a.get("iscrowd", 0) or 0) == 1)]
            cat_map = build_category_map_from_coco(instances_doc)
            task_family = str(info.get("task_family") or ("keypoints" if str(info.get("format") or "") == "coco_keypoints_root" else "bbox"))
            report["dataset"] = {
                "from": src,
                "layout": info,
                "task_family": task_family,
                "reference_trainer": _reference_trainer_readiness(
                    task_family=task_family,
                    source_format=str(info.get("format") or "coco"),
                ),
                "split": str(info.get("split") or ""),
                "instances_json": str(instances_path),
                "images_dir": str(images_dir),
                "include_crowd": include_crowd,
                "counts": {
                    "images": int(len(images)) if isinstance(images, list) else None,
                    "annotations": int(len(annotations)) if isinstance(annotations, list) else None,
                    "classes": int(len(cat_map.class_names)),
                },
                "classes_preview": list(cat_map.class_names[:20]),
            }
        elif src == "ultralytics":
            dataset_source = getattr(args, "data", None) or getattr(args, "dataset", None)
            if not dataset_source:
                raise SystemExit("--data or --dataset is required for --dataset-from ultralytics")
            label_format = None
            task = getattr(args, "task", None)
            if task and str(task).strip().lower() == "segment":
                label_format = "segment"
            manifest = build_manifest(
                str(dataset_source),
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
                "dataset": str(dataset_source),
                "split": manifest.get("split"),
                "label_format": label_format,
                "layout": layout_info,
                "task_family": str((layout_info or {}).get("task_family") or "bbox"),
                "reference_trainer": _reference_trainer_readiness(
                    task_family=str((layout_info or {}).get("task_family") or "bbox"),
                    source_format="ultralytics",
                    label_format=label_format,
                ),
                "counts": {
                    "images": int(len(records)),
                    "labels": int(label_count),
                    "classes_hint": int(max_class + 1) if max_class >= 0 else None,
                },
            }
        elif src == "segmentation":
            dataset_path = getattr(args, "dataset", None)
            if not dataset_path:
                raise SystemExit("--dataset is required for --dataset-from segmentation")
            info = layout_info or inspect_dataset_layout(str(dataset_path), split=str(args.split) if getattr(args, "split", None) else None)
            if info is None:
                raise SystemExit(f"dataset is not a detectable segmentation layout: {dataset_path}")
            report["dataset"] = _summarize_segmentation_layout(
                dataset_path=str(dataset_path),
                layout_info=info,
                split=str(args.split) if getattr(args, "split", None) else None,
            )
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


def _cmd_doctor_train_dataset(args: argparse.Namespace) -> int:
    import time

    from rtdetr_pose.dataset import build_manifest as build_rtdetr_manifest
    from rtdetr_pose.train_records import _load_records_json, normalize_training_records
    from yolozu.dataset import inspect_dataset_layout, load_coco_instances_dataset
    from yolozu.dataset_validator import validate_dataset_records

    def _now_utc() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _resolve_path(value: str | None) -> Path | None:
        if not value:
            return None
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    def _write_report(payload: dict[str, Any]) -> int:
        output = str(getattr(args, "output", "-") or "-")
        if output == "-":
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
            return 0 if not payload["errors"] else 2
        out_path = Path(output)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        print(str(out_path))
        return 0 if not payload["errors"] else 2

    dataset_from = str(getattr(args, "dataset_from", "auto") or "auto").strip().lower().replace("_", "-")
    if dataset_from == "coco-keypoint":
        dataset_from = "coco-keypoints"
    dataset_path = _resolve_path(getattr(args, "dataset", None))
    split = str(getattr(args, "split", "") or "")
    report: dict[str, Any] = {
        "kind": "yolozu_doctor_train_dataset",
        "schema_version": 1,
        "timestamp": _now_utc(),
        "dataset_from": dataset_from,
        "dataset": str(dataset_path) if dataset_path else None,
        "split": split or None,
        "layout": None,
        "reference_trainer": None,
        "records": None,
        "validation_records": None,
        "next_commands": [],
        "warnings": [],
        "errors": [],
    }

    def _candidate_exists(value: Any, *, base: Path) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        path = Path(value).expanduser()
        candidates = [path] if path.is_absolute() else [base / path, Path.cwd() / path]
        return any(candidate.exists() for candidate in candidates)

    def _summarize_records(records: list[dict[str, Any]], *, base: Path) -> dict[str, Any]:
        inspected = records[: int(getattr(args, "max_images", 200) or 200)]
        label_count = 0
        missing_image_path = 0
        missing_image_file = 0
        missing_depth_file = 0
        malformed_labels = 0
        classification_labels = 0
        obb_labels = 0
        keypoint_labels = 0
        depth_labels = 0
        pose_labels = 0
        pose_intrinsics = 0
        for record in inspected:
            image_path_value = record.get("image_path")
            if not image_path_value:
                missing_image_path += 1
            else:
                if not _candidate_exists(image_path_value, base=base):
                    missing_image_file += 1

            if record.get("class_id") is not None or record.get("class_name") is not None or record.get("label") is not None:
                classification_labels += 1
            depth_value = record.get("depth_path") or record.get("depth") or record.get("D_obj")
            if depth_value is not None:
                depth_labels += 1
                if isinstance(depth_value, str) and not _candidate_exists(depth_value, base=base):
                    missing_depth_file += 1
            if record.get("pose") is not None or record.get("R_gt") is not None or record.get("t_gt") is not None:
                pose_labels += 1
            if record.get("K_gt") is not None or record.get("intrinsics") is not None:
                pose_intrinsics += 1

            labels = record.get("labels") or []
            if not isinstance(labels, list):
                malformed_labels += 1
                continue
            for inst in labels:
                if not isinstance(inst, dict):
                    malformed_labels += 1
                    continue
                label_count += 1
                if not isinstance(inst.get("bbox"), dict):
                    malformed_labels += 1
                if inst.get("class_id") is not None or inst.get("class_name") is not None:
                    classification_labels += 1
                if any(k in inst for k in ("angle", "theta", "rotation", "obb", "rotated_bbox")):
                    obb_labels += 1
                if inst.get("keypoints") is not None:
                    keypoint_labels += 1
                if any(k in inst for k in ("depth", "depth_path", "z", "z_gt", "D_obj")):
                    depth_labels += 1
                    depth_value = inst.get("depth_path") or inst.get("depth") or inst.get("D_obj")
                    if isinstance(depth_value, str) and not _candidate_exists(depth_value, base=base):
                        missing_depth_file += 1
                if any(k in inst for k in ("pose6d", "rot6d", "R", "t", "translation")):
                    pose_labels += 1
        return {
            "count": int(len(records)),
            "inspected": int(len(inspected)),
            "labels": int(label_count),
            "missing_image_path": int(missing_image_path),
            "missing_image_file": int(missing_image_file),
            "missing_depth_file": int(missing_depth_file),
            "malformed_labels": int(malformed_labels),
            "classification_labels": int(classification_labels),
            "obb_labels": int(obb_labels),
            "keypoint_labels": int(keypoint_labels),
            "depth_labels": int(depth_labels),
            "pose_labels": int(pose_labels),
            "pose_intrinsics": int(pose_intrinsics),
        }

    def _strict_validation(records: list[dict[str, Any]], *, base: Path) -> dict[str, Any]:
        inspected = records[: int(getattr(args, "max_images", 200) or 200)]
        validation_records: list[dict[str, Any]] = []
        for record in inspected:
            prepared = dict(record)
            image_value = prepared.get("image") or prepared.get("image_path")
            if isinstance(image_value, str) and image_value.strip():
                image_path = Path(image_value).expanduser()
                if not image_path.is_absolute():
                    cwd_candidate = Path.cwd() / image_path
                    image_path = cwd_candidate if cwd_candidate.exists() else base / image_path
                prepared["image"] = str(image_path.resolve())
            validation_records.append(prepared)
        result = validate_dataset_records(
            validation_records,
            strict=True,
            mode="fail",
            check_images=True,
        )
        return {
            "policy": "strict",
            "scope": "all" if len(inspected) == len(records) else "first_n",
            "checked": int(len(inspected)),
            "total": int(len(records)),
            "bbox_edge_tolerance": 1e-6,
            "error_count": int(len(result.errors)),
            "warning_count": int(len(result.warnings)),
            "errors": list(result.errors),
            "warnings": list(result.warnings),
        }

    def _summarize_and_validate(records: list[dict[str, Any]], *, base: Path) -> dict[str, Any]:
        summary = _summarize_records(records, base=base)
        summary["validation"] = _strict_validation(records, base=base)
        return summary

    def _apply_task_readiness(readiness: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
        task = str(readiness.get("task_family") or "").strip().lower()
        validation = summary.get("validation") or {}
        base_ready = bool(summary.get("count")) and int(summary.get("missing_image_path") or 0) == 0
        base_ready = base_ready and int(summary.get("missing_image_file") or 0) == 0
        base_ready = base_ready and int(summary.get("malformed_labels") or 0) == 0
        base_ready = base_ready and int(validation.get("checked") or 0) > 0
        base_ready = base_ready and int(validation.get("error_count") or 0) == 0
        if task == "depth":
            ready = base_ready and int(summary.get("depth_labels") or 0) > 0 and int(summary.get("missing_depth_file") or 0) == 0
            readiness["direct_train_ready"] = bool(readiness.get("direct_train_ready")) and bool(ready)
            readiness["train_ready_after_migration"] = bool(readiness.get("train_ready_after_migration")) and bool(ready)
            if not ready:
                readiness["requires_normalization"] = True
                readiness["reason"] = "depth training requires bbox records plus existing depth sidecars"
        elif task == "pose6d":
            ready = base_ready and int(summary.get("pose_labels") or 0) > 0 and int(summary.get("pose_intrinsics") or 0) > 0
            readiness["direct_train_ready"] = bool(readiness.get("direct_train_ready")) and bool(ready)
            readiness["train_ready_after_migration"] = bool(readiness.get("train_ready_after_migration")) and bool(ready)
            if not ready:
                readiness["requires_normalization"] = True
                readiness["reason"] = "pose6d training requires bbox records plus pose and intrinsics sidecars"
        elif task == "obb":
            readiness["direct_train_ready"] = False
            readiness["train_ready_after_migration"] = False
            readiness["requires_normalization"] = True
        elif task == "classification":
            readiness["direct_train_ready"] = False
            readiness["train_ready_after_migration"] = False
            readiness["requires_normalization"] = True
        else:
            readiness["direct_train_ready"] = bool(readiness.get("direct_train_ready")) and bool(base_ready)
            readiness["train_ready_after_migration"] = bool(readiness.get("train_ready_after_migration")) and bool(base_ready)
        if not base_ready and task not in {"classification", "obb"}:
            readiness["requires_normalization"] = True
            readiness["reason"] = "strict dataset validation failed; inspect records.validation errors before training"
        return readiness

    def _record_summary(records_path: Path, *, label: str) -> dict[str, Any]:
        records = normalize_training_records(_load_records_json(records_path, label=label))
        summary = _summarize_and_validate(records, base=records_path.parent)
        summary["path"] = str(records_path)
        summary["direct_train_ready"] = (
            bool(records)
            and int(summary.get("missing_image_path") or 0) == 0
            and int(summary.get("missing_image_file") or 0) == 0
            and int(summary.get("missing_depth_file") or 0) == 0
            and int(summary.get("malformed_labels") or 0) == 0
            and int((summary.get("validation") or {}).get("error_count") or 0) == 0
        )
        return summary

    try:
        records_json = _resolve_path(getattr(args, "records_json", None))
        val_records_json = _resolve_path(getattr(args, "val_records_json", None))
        if records_json is not None:
            train_summary = _record_summary(records_json, label="records")
            report["records"] = train_summary
            if val_records_json is not None:
                report["validation_records"] = _record_summary(val_records_json, label="validation")
            records_task = dataset_from if dataset_from != "auto" else "records"
            if records_task == "records":
                readiness = {
                    "task_family": "records",
                    "direct_train_ready": bool(train_summary.get("direct_train_ready")),
                    "train_ready_after_migration": bool(train_summary.get("direct_train_ready")),
                    "requires_normalization": False,
                    "accepted_inputs": ["records JSON using image/image_path and normalized bbox labels"],
                    "reason": (
                        "records JSON can be consumed by the reference trainer"
                        if train_summary.get("direct_train_ready")
                        else "records JSON has missing image paths or malformed labels"
                    ),
                }
            else:
                readiness = _reference_trainer_readiness(task_family=records_task, source_format="records")
                readiness = _apply_task_readiness(readiness, train_summary)
            report["reference_trainer"] = readiness
            report["next_commands"].append(
                "python3 -m yolozu train <config> --records-json "
                f"{records_json}"
                + (f" --val-records-json {val_records_json}" if val_records_json else "")
            )
            if not (report.get("reference_trainer") or {}).get("direct_train_ready"):
                report["errors"].append("records JSON is not train-ready")
            return _write_report(report)

        instances_path = _resolve_path(getattr(args, "instances", None))
        images_dir = _resolve_path(getattr(args, "images_dir", None))
        if (instances_path is None) != (images_dir is None):
            report["errors"].append("--instances and --images-dir must be provided together")
            return _write_report(report)

        if instances_path is not None and images_dir is not None:
            if dataset_from not in {"auto", "coco", "coco-keypoints", "coco-instances"}:
                report["errors"].append("--instances/--images-dir require --from auto, coco, coco-keypoints, or coco-instances")
                return _write_report(report)
            if not instances_path.is_file():
                report["errors"].append(f"COCO annotations file does not exist: {instances_path}")
                return _write_report(report)
            if not images_dir.is_dir():
                report["errors"].append(f"COCO images directory does not exist: {images_dir}")
                return _write_report(report)

            instances_doc = json.loads(instances_path.read_text(encoding="utf-8"))
            if not isinstance(instances_doc, dict):
                raise ValueError("COCO annotations root must be an object")
            inferred_keypoints = any(
                isinstance(annotation, dict) and annotation.get("keypoints") is not None
                for annotation in (instances_doc.get("annotations") or [])
            )
            task_family = "keypoints" if dataset_from == "coco-keypoints" or (dataset_from == "auto" and inferred_keypoints) else "bbox"
            dataset_from = "coco-keypoints" if task_family == "keypoints" else "coco-instances"
            report["dataset_from"] = dataset_from
            report["layout"] = {
                "format": "coco_instances_explicit",
                "instances": str(instances_path),
                "images_dir": str(images_dir),
                "split": split or None,
                "task_family": task_family,
            }
            records = load_coco_instances_dataset(
                instances_doc,
                images_dir=images_dir,
                include_crowd=bool(getattr(args, "include_crowd", False)),
            )
            normalized = normalize_training_records(records)
            summary = _summarize_and_validate(normalized, base=instances_path.parent)
            report["records"] = summary
            readiness = _reference_trainer_readiness(
                task_family=task_family,
                source_format="coco-instances",
            )
            readiness = _apply_task_readiness(readiness, summary)
            report["reference_trainer"] = readiness
            validation = summary.get("validation") or {}
            if int(validation.get("error_count") or 0):
                report["errors"].append(
                    f"strict dataset validation failed with {int(validation.get('error_count') or 0)} error(s)"
                )
            import_command = (
                "python3 -m yolozu import dataset --from coco-instances "
                f"--instances {shlex.quote(str(instances_path))} "
                f"--images-dir {shlex.quote(str(images_dir))} "
                "--output reports/train_dataset_wrapper --force"
                + (f" --split {shlex.quote(split)}" if split else "")
            )
            if bool(readiness.get("train_ready_after_migration")):
                report["next_commands"].extend(
                    [
                        import_command,
                        "python3 -m yolozu train <config> --dataset-root reports/train_dataset_wrapper",
                    ]
                )
            else:
                report["next_commands"].append(
                    "Fix records.validation errors before importing or training this dataset."
                )
            return _write_report(report)

        if dataset_path is None:
            report["errors"].append("--dataset or --records-json is required")
            return _write_report(report)

        layout_info = inspect_dataset_layout(str(dataset_path), split=split or None)
        report["layout"] = layout_info
        if dataset_from == "auto":
            if layout_info is None:
                dataset_from = "unknown"
            else:
                fmt = str(layout_info.get("format") or "")
                if fmt == "coco_keypoints_root":
                    dataset_from = "coco-keypoints"
                elif fmt in ("coco_root", "yolozu_coco_wrapper"):
                    dataset_from = "coco"
                elif fmt in (
                    "yolozu_segmentation_descriptor",
                    "voc_segmentation_root",
                    "cityscapes_segmentation_root",
                    "ade20k_segmentation_root",
                ):
                    dataset_from = "segmentation"
                else:
                    dataset_from = "ultralytics"
            report["dataset_from"] = dataset_from

        if dataset_from == "coco-keypoints":
            task_family = "keypoints"
        elif dataset_from == "segmentation":
            task_family = "segmentation"
        elif dataset_from in {"classification", "obb", "depth", "pose6d"}:
            task_family = dataset_from
        else:
            task_family = str((layout_info or {}).get("task_family") or "bbox")

        source_format = str((layout_info or {}).get("format") or dataset_from)
        label_format = str((layout_info or {}).get("label_format") or "")
        readiness = _reference_trainer_readiness(
            task_family=task_family,
            source_format=source_format,
            label_format=(label_format or None),
        )

        scan_yolo_like = source_format in {"ultralytics", "ultralytics_data_yaml", "yolo", "yolozu_wrapper", "data_yaml", "yolo_layout"}
        if bool(readiness.get("direct_train_ready")) or (task_family == "obb" and scan_yolo_like):
            try:
                manifest = build_rtdetr_manifest(dataset_path, split=split or str((layout_info or {}).get("split") or "train"))
                records = list(manifest.get("images") or [])
                inspected = records[: int(getattr(args, "max_images", 200) or 200)]
                summary = _summarize_and_validate(normalize_training_records(records), base=dataset_path)
                summary["inspected"] = int(len(inspected))
                summary["keypoints_meta"] = manifest.get("keypoints_meta")
                report["records"] = summary
                readiness = _apply_task_readiness(readiness, summary)
                validation = summary.get("validation") or {}
                if int(validation.get("error_count") or 0):
                    report["errors"].append(
                        f"strict dataset validation failed with {int(validation.get('error_count') or 0)} error(s)"
                    )
                if not records:
                    readiness["direct_train_ready"] = False
                    readiness["requires_normalization"] = True
                    readiness["reason"] = "reference trainer resolved the layout but found no records"
                    report["errors"].append("no training records found")
            except Exception as exc:
                readiness["direct_train_ready"] = False
                readiness["requires_normalization"] = True
                readiness["reason"] = f"reference trainer could not resolve dataset: {exc}"
                report["errors"].append(str(exc))

        report["reference_trainer"] = readiness

        if bool(readiness.get("direct_train_ready")):
            report["next_commands"].append(f"python3 -m yolozu train <config> --dataset-root {dataset_path}" + (f" --split {split}" if split else ""))
        elif bool(readiness.get("train_ready_after_migration")):
            if dataset_from == "coco-keypoints":
                report["next_commands"].append(
                    f"python3 -m yolozu import dataset --from coco-keypoints --dataset {dataset_path} "
                    f"--output reports/train_dataset_wrapper --force"
                    + (f" --split {split}" if split else "")
                )
            elif dataset_from == "coco":
                report["next_commands"].append(
                    f"python3 -m yolozu migrate dataset --from coco --dataset {dataset_path} "
                    f"--output reports/train_dataset_wrapper --force"
                    + (f" --split {split}" if split else "")
                )
            else:
                report["next_commands"].append(
                    f"python3 -m yolozu import dataset --from auto --dataset {dataset_path} "
                    f"--output reports/train_dataset_wrapper --force"
                    + (f" --split {split}" if split else "")
                )
            report["next_commands"].append("python3 -m yolozu train <config> --dataset-root reports/train_dataset_wrapper")
        else:
            report["next_commands"].append(
                "Use an external training lane or convert the source into YOLO labels or records JSON before reference training."
            )

    except SystemExit as exc:
        report["errors"].append(str(exc))
    except Exception as exc:
        report["errors"].append(str(exc))

    return _write_report(report)


def _cmd_export(args: argparse.Namespace) -> int:
    from yolozu.inference.export_orchestrator import export_with_backend

    if not getattr(args, "dataset", None):
        args.dataset = str(Path.cwd() / "data" / "coco128")
    out_path = export_with_backend(
        args,
        subprocess_or_die=_subprocess_or_die,
        base_run_meta=_base_run_meta,
    )
    print(str(out_path))
    return 0


def _cmd_export_dataset(args: argparse.Namespace) -> int:
    from yolozu.datasets.exports import export_coco_dataset, export_kitti_dataset, export_segmentation_dataset, export_yolo_dataset

    target = str(getattr(args, "to_format", ""))
    dataset = str(getattr(args, "dataset", "") or "")
    out_dir = str(getattr(args, "out_dir", "") or "")
    split = str(args.split) if getattr(args, "split", None) else None
    image_mode = str(getattr(args, "image_mode", "copy") or "copy")
    force = bool(getattr(args, "force", False))

    if not dataset:
        raise SystemExit("--dataset is required")
    if not out_dir:
        raise SystemExit("--out-dir is required")

    try:
        if target == "yolo":
            out_root = export_yolo_dataset(dataset_root=dataset, split=split, out_dir=out_dir, image_mode=image_mode, force=force)
        elif target == "kitti":
            out_root = export_kitti_dataset(dataset_root=dataset, split=split, out_dir=out_dir, image_mode=image_mode, force=force)
        elif target == "coco":
            out_root = export_coco_dataset(dataset_root=dataset, split=split, out_dir=out_dir, image_mode=image_mode, force=force)
        elif target == "segmentation":
            out_root = export_segmentation_dataset(dataset_root=dataset, out_dir=out_dir, image_mode=image_mode, force=force)
        else:
            raise SystemExit(f"unsupported export-dataset target: {target}")
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(str(out_root))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    if args.validate_command == "dataset":
        from yolozu.dataset import build_manifest, inspect_dataset_layout
        from yolozu.dataset_validator import validate_dataset_records

        layout_info = inspect_dataset_layout(str(args.dataset), split=str(args.split) if args.split else None)
        if (
            layout_info is not None
            and str(layout_info.get("task_family") or "") == "segmentation"
            and str(layout_info.get("format") or "") != "yolozu_wrapper"
        ):
            warnings, errors = _validate_segmentation_layout(
                dataset_path=str(args.dataset),
                layout_info=layout_info,
                max_images=(int(args.max_images) if args.max_images is not None else None),
            )
            for warning in warnings:
                print(warning, file=sys.stderr)
            if errors:
                for error in errors:
                    print(error, file=sys.stderr)
                return 1
            return 0

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

    if args.validate_command == "predictions":
        from yolozu.predictions import validate_predictions_path
        from yolozu.predictions.validation_result import (
            _validate_predictions_path_result,
        )

        if bool(getattr(args, "json", False)):
            result, exit_code = validate_predictions_path(
                str(args.path),
                strict=bool(args.strict),
            )
        else:
            result, exit_code = _validate_predictions_path_result(
                str(args.path),
                strict=bool(args.strict),
                max_warnings=None,
            )
        if bool(getattr(args, "json", False)):
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return exit_code
        if not result["ok"]:
            errors = list(result.get("errors") or [])
            message = (
                str(errors[0].get("message"))
                if errors
                else "validation failed"
            )
            raise SystemExit(message)
        for warning in result["warnings"]:
            print(warning, file=sys.stderr)
        return 0

    path = Path(str(args.path))
    if not path.exists():
        raise SystemExit(f"file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"failed to parse json: {path} ({exc})") from exc

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
    from pathlib import Path

    from yolozu.predict_images import predict_images_with_namespace

    try:
        out_json, out_html = predict_images_with_namespace(
            args,
            subprocess_or_die=_subprocess_or_die,
            base_run_meta=_base_run_meta,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(str(out_json))
    print(f"predictions_json: {out_json}")
    if out_html is not None:
        print(str(out_html))
        print(f"html_report: {out_html}")
    overlays_arg = getattr(args, "overlays_dir", None)
    if overlays_arg:
        overlays_dir = Path(str(overlays_arg)).expanduser()
        if not overlays_dir.is_absolute():
            overlays_dir = Path.cwd() / overlays_dir
        print(f"overlays_dir: {overlays_dir}")
        try:
            overlays = sorted(overlays_dir.glob("*.png"))
        except Exception:
            overlays = []
        if overlays:
            print(f"first_overlay: {overlays[0]}")
    return 0


def _cmd_eval_coco(args: argparse.Namespace) -> int:
    from yolozu.api import APIError, _failure_report, _write_json_atomic, evaluate_coco
    from yolozu.core.diagnostics import format_cli_error

    dataset_root = Path(str(args.dataset)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = Path.cwd() / dataset_root

    predictions_path = Path(str(args.predictions)).expanduser()
    if not predictions_path.is_absolute():
        predictions_path = Path.cwd() / predictions_path

    output_path = Path(str(args.output)).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    classes_path: Path | None = None
    if args.classes:
        classes_path = Path(str(args.classes)).expanduser()
        if not classes_path.is_absolute():
            classes_path = Path.cwd() / classes_path

    try:
        result = evaluate_coco(
            dataset_root,
            predictions_path,
            split=str(args.split) if args.split else None,
            bbox_format=str(args.bbox_format),
            max_images=int(args.max_images) if args.max_images is not None else None,
            dry_run=bool(args.dry_run),
            repair=bool(args.repair),
            classes=classes_path,
            assume_class_id_is_category_id=bool(args.assume_class_id_is_category_id),
        )
    except APIError as exc:
        failure = _failure_report(
            exc,
            dataset=str(dataset_root),
            predictions=str(predictions_path),
            split=str(args.split) if args.split else None,
            bbox_format=str(args.bbox_format),
            max_images=int(args.max_images) if args.max_images is not None else None,
            dry_run=bool(args.dry_run),
            repair=bool(args.repair),
        )
        failure["normalization"] = {
            "classes": str(classes_path) if classes_path else None,
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

    _write_json_atomic(output_path, result.to_dict())
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
        bbox_format=str(args.bbox_format),
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
    from yolozu.dataset import inspect_dataset_layout
    from yolozu.migrate import (
        migrate_coco_dataset_wrapper,
        migrate_coco_results_predictions,
        migrate_seg_dataset_descriptor,
        migrate_ultralytics_dataset_wrapper,
        write_dataset_wrapper,
    )

    if args.migrate_command == "dataset":
        from_format = str(args.from_format)
        if from_format == "auto":
            from_format = _resolve_auto_dataset_from_args(args)

        if from_format == "ultralytics":
            dataset_path = getattr(args, "dataset", None)
            if dataset_path:
                info = inspect_dataset_layout(str(dataset_path), split=str(args.split) if args.split else None)
                if info is None:
                    raise SystemExit(f"could not inspect dataset layout: {dataset_path}")
                fmt = str(info.get("format") or "")
                if fmt in ("yolo_layout", "yolozu_wrapper", "yolozu_coco_wrapper"):
                    out = write_dataset_wrapper(
                        str(args.output),
                        images_dir=str(info.get("images_dir")),
                        labels_dir=str(info.get("labels_dir")),
                        split=str(info.get("split") or args.split or "val"),
                        label_format=(str(info.get("label_format")) if info.get("label_format") else None),
                        source={"from": fmt, "dataset": str(dataset_path)},
                        force=bool(args.force),
                    )
                else:
                    data_yaml = str(info.get("data_yaml") or dataset_path)
                    out = migrate_ultralytics_dataset_wrapper(
                        data_yaml=data_yaml,
                        args_yaml=str(args.args) if args.args else None,
                        split=str(args.split) if args.split else None,
                        task=str(args.task) if args.task else None,
                        output=str(args.output),
                        force=bool(args.force),
                    )
            else:
                out = migrate_ultralytics_dataset_wrapper(
                    data_yaml=str(args.data) if args.data else None,
                    args_yaml=str(args.args) if args.args else None,
                    split=str(args.split) if args.split else None,
                    task=str(args.task) if args.task else None,
                    output=str(args.output),
                    force=bool(args.force),
                )
        elif from_format in {"coco", "coco-keypoints"}:
            coco_root = getattr(args, "coco_root", None) or getattr(args, "dataset", None)
            if not coco_root:
                raise SystemExit(f"--coco-root or --dataset is required for --from {from_format}")
            info = inspect_dataset_layout(str(coco_root), split=str(args.split) if args.split else None)
            if info is not None and str(info.get("format") or "") in ("coco_root", "coco_keypoints_root"):
                effective_split = str(info.get("split") or args.split or "val2017")
            else:
                effective_split = str(args.split) if args.split else "val2017"
            if from_format == "coco-keypoints" and (info is None or str(info.get("format") or "") != "coco_keypoints_root"):
                raise SystemExit(f"dataset is not a detectable COCO keypoints root: {coco_root}")
            if info is not None and str(info.get("format") or "") == "coco_keypoints_root":
                from yolozu.imports import import_coco_keypoints_dataset

                out = import_coco_keypoints_dataset(
                    annotations_json=str(info.get("instances_json")),
                    images_dir=str(info.get("images_dir")),
                    split=effective_split,
                    output=str(args.output),
                    include_crowd=bool(args.include_crowd),
                    force=bool(args.force),
                )
                print(str(out))
                return 0
            out = migrate_coco_dataset_wrapper(
                coco_root=str(coco_root),
                split=effective_split,
                instances_json=(str(args.instances_json) if args.instances_json else None),
                output=str(args.output),
                mode=str(args.mode),
                include_crowd=bool(args.include_crowd),
                force=bool(args.force),
            )
        elif from_format == "segmentation":
            dataset_path = getattr(args, "dataset", None)
            if not dataset_path:
                raise SystemExit("--dataset is required for --from segmentation")
            info = inspect_dataset_layout(str(dataset_path), split=str(args.split) if args.split else None)
            if info is None:
                raise SystemExit(f"dataset is not a detectable segmentation layout: {dataset_path}")
            out = _write_segmentation_descriptor_from_layout(
                dataset_path=str(dataset_path),
                layout_info=info,
                output=str(args.output),
                split=str(args.split) if args.split else None,
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


def _cmd_predictions(args: argparse.Namespace) -> int:
    from yolozu.migrate import migrate_predictions_entries_schema

    if args.predictions_command == "migrate":
        out = migrate_predictions_entries_schema(
            input_path=str(args.input),
            output=str(args.output),
            from_version=str(args.from_version),
            to_version=str(args.to_version),
            strict_source=bool(args.strict_source),
            force=bool(args.force),
        )
        print(str(out))
        return 0

    raise SystemExit("unknown predictions command")


def _cmd_import(args: argparse.Namespace) -> int:
    from yolozu.dataset import inspect_dataset_layout
    from yolozu.imports import (
        import_coco_instances_dataset,
        import_coco_keypoints_dataset,
        import_detectron2_config,
        import_mmdet_config,
        import_ultralytics_config,
        import_yolox_config,
    )
    from yolozu.migrate import migrate_ultralytics_dataset_wrapper, write_dataset_wrapper

    try:
        if args.import_command == "dataset":
            from_format = str(args.from_format)
            if from_format == "auto":
                from_format = _resolve_auto_dataset_from_args(args)

            if from_format == "ultralytics":
                dataset_path = getattr(args, "dataset", None)
                layout_info = None
                if dataset_path:
                    layout_info = inspect_dataset_layout(str(dataset_path), split=str(args.split) if args.split else None)
                    if layout_info is None:
                        raise SystemExit(f"could not inspect dataset layout: {dataset_path}")
                    fmt = str(layout_info.get("format") or "")
                    if fmt in ("yolo_layout", "yolozu_wrapper"):
                        out = write_dataset_wrapper(
                            str(args.output),
                            images_dir=str(layout_info.get("images_dir")),
                            labels_dir=str(layout_info.get("labels_dir")),
                            split=str(layout_info.get("split") or args.split or "val"),
                            label_format=(str(layout_info.get("label_format")) if layout_info.get("label_format") else None),
                            source={"from": fmt, "dataset": str(dataset_path)},
                            force=bool(args.force),
                        )
                        print(str(out))
                        return 0
                    data_yaml = str(layout_info.get("data_yaml") or dataset_path)
                else:
                    data_yaml = str(args.data) if args.data else None
                out = migrate_ultralytics_dataset_wrapper(
                    data_yaml=data_yaml,
                    args_yaml=str(args.args) if args.args else None,
                    split=str(args.split) if args.split else None,
                    task=str(args.task) if args.task else None,
                    output=str(args.output),
                    force=bool(args.force),
                )
                print(str(out))
                return 0

            if from_format in {"coco", "coco-keypoints"}:
                dataset_path = getattr(args, "dataset", None)
                if not dataset_path:
                    raise SystemExit(f"--dataset is required for --from {from_format}")
                info = inspect_dataset_layout(str(dataset_path), split=str(args.split) if args.split else None)
                expected_formats = ("coco_keypoints_root",) if from_format == "coco-keypoints" else ("coco_root", "coco_keypoints_root")
                if info is None or str(info.get("format") or "") not in expected_formats:
                    raise SystemExit(f"dataset is not a detectable {from_format} root: {dataset_path}")
                if str(info.get("format") or "") == "coco_keypoints_root":
                    out = import_coco_keypoints_dataset(
                        annotations_json=str(info.get("instances_json")),
                        images_dir=str(info.get("images_dir")),
                        split=str(info.get("split") or args.split or "val2017"),
                        output=str(args.output),
                        include_crowd=bool(args.include_crowd),
                        force=bool(args.force),
                    )
                else:
                    out = import_coco_instances_dataset(
                        instances_json=str(info.get("instances_json")),
                        images_dir=str(info.get("images_dir")),
                        split=str(info.get("split") or args.split or "val2017"),
                        output=str(args.output),
                        include_crowd=bool(args.include_crowd),
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

            if from_format == "segmentation":
                dataset_path = getattr(args, "dataset", None)
                if not dataset_path:
                    raise SystemExit("--dataset is required for --from segmentation")
                info = inspect_dataset_layout(str(dataset_path), split=str(args.split) if args.split else None)
                if info is None:
                    raise SystemExit(f"dataset is not a detectable segmentation layout: {dataset_path}")
                out = _write_segmentation_descriptor_from_layout(
                    dataset_path=str(dataset_path),
                    layout_info=info,
                    output=str(args.output),
                    split=str(args.split) if args.split else None,
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
