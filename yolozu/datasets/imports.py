"""Config and dataset import adapters.

Projects training configurations and COCO datasets from Ultralytics,
MMDet, Detectron2, and YOLOX formats into YOLOZU's canonical
representation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "project_ultralytics_args",
    "project_mmdet_config",
    "project_detectron2_config",
    "project_yolox_exp",
    "import_coco_instances_dataset",
    "import_ultralytics_config",
    "import_mmdet_config",
    "import_detectron2_config",
    "import_yolox_config",
]

from yolozu.core.canonical import TrainConfig
from yolozu.core.config import simple_yaml_load
from .coco_convert import build_category_map_from_coco


def _extract_keypoint_schema_from_coco(instances_doc: dict[str, Any]) -> dict[str, Any]:
    categories = instances_doc.get("categories") or []
    if not isinstance(categories, list):
        return {}

    for category in categories:
        if not isinstance(category, dict):
            continue

        raw_names = category.get("keypoints") or []
        if not isinstance(raw_names, list) or not raw_names:
            continue

        keypoint_names: list[str] = []
        for name in raw_names:
            text = str(name).strip()
            if text:
                keypoint_names.append(text)
        if not keypoint_names:
            continue

        k_count = int(len(keypoint_names))
        raw_skeleton = category.get("skeleton") or []
        skeleton: list[list[int]] = []
        seen: set[tuple[int, int]] = set()
        if isinstance(raw_skeleton, list):
            for edge in raw_skeleton:
                if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                    continue
                try:
                    a = int(edge[0])
                    b = int(edge[1])
                except Exception:
                    continue
                if a <= 0 or b <= 0 or a == b:
                    continue
                if a > k_count or b > k_count:
                    continue
                key = (a, b) if a <= b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                skeleton.append([a, b])

        out: dict[str, Any] = {
            "keypoint_names": keypoint_names,
            "num_keypoints": int(k_count),
        }
        if skeleton:
            out["skeleton"] = skeleton
        category_id = category.get("id")
        if category_id is not None:
            try:
                out["keypoint_category_id"] = int(category_id)
            except (TypeError, ValueError):
                pass
        return out

    return {}


def _load_config(path: Path) -> dict[str, Any]:
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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_train_config(
    output: str | Path,
    *,
    train: TrainConfig,
    force: bool = False,
    default_name: str = "train_config_import.json",
) -> Path:
    out_path = Path(output)
    if out_path.suffix.lower() != ".json":
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / default_name
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        raise FileExistsError(f"output already exists: {out_path} (use --force to overwrite)")

    _write_json(out_path, train.to_dict())
    return out_path


def _as_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _get_path(cfg: Any, path: str, default: Any = None) -> Any:
    cur = cfg
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def project_ultralytics_args(cfg: dict[str, Any], *, source: dict[str, Any]) -> TrainConfig:
    imgsz = cfg.get("imgsz")
    batch = cfg.get("batch")
    epochs = cfg.get("epochs")
    lr0 = cfg.get("lr0")
    wd = cfg.get("weight_decay")
    optimizer = cfg.get("optimizer")
    seed = cfg.get("seed")
    device = cfg.get("device")
    model = cfg.get("model")

    preprocess: dict[str, Any] = {}
    for key in ("imgsz", "rect", "single_cls", "multi_scale", "cache"):
        if key in cfg:
            preprocess[key] = cfg.get(key)

    return TrainConfig(
        model=(str(model) if isinstance(model, str) and model.strip() else None),
        imgsz=(imgsz if isinstance(imgsz, (int, list)) else None),
        batch=(_as_int(batch)),
        epochs=(_as_int(epochs)),
        optimizer=(str(optimizer) if isinstance(optimizer, str) and optimizer.strip() else None),
        lr=(_as_float(lr0)),
        weight_decay=(_as_float(wd)),
        seed=(_as_int(seed)),
        device=(str(device) if isinstance(device, str) and device.strip() else None),
        preprocess=preprocess or None,
        source=dict(source),
    )


def _require_module(name: str, *, pip_hint: str) -> Any:
    try:
        import importlib

        return importlib.import_module(name)
    except Exception as exc:
        raise RuntimeError(f"{name} is required for this import mode. Install it (e.g. `{pip_hint}`).") from exc


def project_mmdet_config(*, config: str | Path) -> TrainConfig:
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"--config not found: {config_path}")

    mmengine_config = _require_module("mmengine.config", pip_hint="pip install mmengine").Config
    cfg_obj = mmengine_config.fromfile(str(config_path))
    cfg = cfg_obj.to_dict() if hasattr(cfg_obj, "to_dict") else dict(cfg_obj)

    # Major key projections (best-effort).
    batch = _as_int(_get_path(cfg, "train_dataloader.batch_size"))
    if batch is None:
        batch = _as_int(_get_path(cfg, "data.samples_per_gpu"))

    max_epochs = _as_int(_get_path(cfg, "train_cfg.max_epochs")) or _as_int(_get_path(cfg, "max_epochs"))
    max_iters = _as_int(_get_path(cfg, "train_cfg.max_iters")) or _as_int(_get_path(cfg, "max_iters"))

    optim = _get_path(cfg, "optim_wrapper.optimizer") or _get_path(cfg, "optimizer") or {}
    optimizer = str(optim.get("type") or "") if isinstance(optim, dict) else None
    lr = _as_float(optim.get("lr")) if isinstance(optim, dict) else None
    wd = _as_float(optim.get("weight_decay")) if isinstance(optim, dict) else None

    # Try to infer resize target from pipeline (common: dict(type="Resize", scale=(w,h))).
    imgsz = None
    pipeline = _get_path(cfg, "train_pipeline")
    if isinstance(pipeline, list):
        for step in pipeline:
            if not isinstance(step, dict):
                continue
            if str(step.get("type") or "").lower() != "resize":
                continue
            scale = step.get("scale") or step.get("img_scale")
            if isinstance(scale, (list, tuple)) and len(scale) == 2:
                w, h = scale
                if _as_int(w) is not None and _as_int(h) is not None:
                    imgsz = [int(w), int(h)]
                    break

    return TrainConfig(
        imgsz=imgsz,
        batch=batch,
        epochs=max_epochs,
        steps=max_iters,
        optimizer=(optimizer.strip() if isinstance(optimizer, str) and optimizer.strip() else None),
        lr=lr,
        weight_decay=wd,
        source={"from": "mmdet", "config": str(config_path)},
    )


def project_detectron2_config(*, config: str | Path) -> TrainConfig:
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"--config not found: {config_path}")

    d2_config = _require_module("detectron2.config", pip_hint="pip install detectron2")
    cfg = d2_config.get_cfg()
    cfg.merge_from_file(str(config_path))

    def _safe(getter: Callable[[], Any]) -> Any:
        try:
            return getter()
        except Exception:
            return None

    batch = _safe(lambda: int(cfg.SOLVER.IMS_PER_BATCH))
    steps = _safe(lambda: int(cfg.SOLVER.MAX_ITER))
    lr = _safe(lambda: float(cfg.SOLVER.BASE_LR))
    wd = _safe(lambda: float(cfg.SOLVER.WEIGHT_DECAY))

    # INPUT.MIN_SIZE_TRAIN is often a list of candidates. Keep the first for a single canonical value.
    imgsz = None
    min_sizes = _safe(lambda: list(cfg.INPUT.MIN_SIZE_TRAIN))
    if isinstance(min_sizes, list) and min_sizes:
        if _as_int(min_sizes[0]) is not None:
            imgsz = int(min_sizes[0])

    preprocess = {
        "min_size_train": min_sizes,
        "max_size_train": _safe(lambda: int(cfg.INPUT.MAX_SIZE_TRAIN)),
        "format": _safe(lambda: str(cfg.INPUT.FORMAT)),
    }

    return TrainConfig(
        imgsz=imgsz,
        batch=batch if isinstance(batch, int) else None,
        steps=steps if isinstance(steps, int) else None,
        optimizer=_safe(lambda: str(getattr(cfg.SOLVER, "OPTIMIZER", ""))) or None,
        lr=lr if isinstance(lr, float) else None,
        weight_decay=wd if isinstance(wd, float) else None,
        preprocess=preprocess,
        source={"from": "detectron2", "config": str(config_path)},
    )


def project_yolox_exp(*, config: str | Path) -> TrainConfig:
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"--config not found: {config_path}")

    # YOLOX exp files are Python and often import yolox.*. Treat as optional.
    try:
        import runpy

        ns = runpy.run_path(str(config_path))
    except Exception as exc:
        raise RuntimeError(f"failed to execute YOLOX exp file: {config_path} ({exc})") from exc

    exp = ns.get("exp")
    if exp is None:
        exp_cls = ns.get("Exp")
        if isinstance(exp_cls, type):
            try:
                exp = exp_cls()
            except Exception:
                exp = None
    if exp is None:
        get_exp = ns.get("get_exp")
        if callable(get_exp):
            try:
                exp = get_exp()
            except Exception:
                exp = None
    if exp is None:
        raise RuntimeError("could not find YOLOX exp object (expected `exp`, `Exp`, or `get_exp()` in the file)")

    input_size = getattr(exp, "input_size", None)
    imgsz = None
    if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
        h, w = input_size
        if _as_int(w) is not None and _as_int(h) is not None:
            imgsz = [int(w), int(h)]

    batch = _as_int(getattr(exp, "batch_size", None))
    epochs = _as_int(getattr(exp, "max_epoch", None))

    basic_lr_per_img = _as_float(getattr(exp, "basic_lr_per_img", None))
    lr = None
    if basic_lr_per_img is not None and batch is not None:
        lr = float(basic_lr_per_img) * float(batch)
    weight_decay = _as_float(getattr(exp, "weight_decay", None))

    return TrainConfig(
        imgsz=imgsz,
        batch=batch,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        source={"from": "yolox", "config": str(config_path), "note": "executed exp python"},
    )


def import_coco_instances_dataset(
    *,
    instances_json: str | Path,
    images_dir: str | Path,
    split: str,
    output: str | Path,
    include_crowd: bool = False,
    force: bool = False,
) -> Path:
    """Create a read-only dataset wrapper for COCO instances JSON."""

    instances_path = Path(instances_json)
    if not instances_path.is_absolute():
        instances_path = (Path.cwd() / instances_path).resolve()

    images_dir_path = Path(images_dir)
    if not images_dir_path.is_absolute():
        images_dir_path = (Path.cwd() / images_dir_path).resolve()

    out_path = Path(output)
    if out_path.suffix.lower() != ".json":
        out_root = out_path
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / "dataset.json"
    else:
        out_root = out_path.parent
        out_root.mkdir(parents=True, exist_ok=True)

    if not instances_path.exists():
        raise FileNotFoundError(f"--instances not found: {instances_path}")
    if not images_dir_path.exists():
        raise FileNotFoundError(f"--images-dir not found: {images_dir_path}")
    if out_path.exists() and not force:
        raise FileExistsError(f"output already exists: {out_path} (use --force to overwrite)")

    instances_doc = json.loads(instances_path.read_text(encoding="utf-8"))
    cat_map = build_category_map_from_coco(instances_doc)
    keypoint_schema = _extract_keypoint_schema_from_coco(instances_doc)

    # Persist a stable classes.json mapping under labels/<split>/ for downstream tools.
    labels_dir = out_root / "labels" / str(split)
    labels_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "category_id_to_class_id": cat_map.category_id_to_class_id,
        "class_id_to_category_id": cat_map.class_id_to_category_id,
        "class_names": cat_map.class_names,
    }
    if keypoint_schema:
        mapping.update(keypoint_schema)
    (labels_dir / "classes.json").write_text(json.dumps(mapping, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (labels_dir / "classes.txt").write_text("\n".join(cat_map.class_names) + "\n", encoding="utf-8")

    payload: dict[str, Any] = {
        "format": "coco_instances",
        "instances_json": str(instances_path),
        "images_dir": str(images_dir_path),
        "split": str(split),
        "include_crowd": bool(include_crowd),
        "source": {"from": "coco-instances"},
    }
    if keypoint_schema:
        payload.update(keypoint_schema)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def import_ultralytics_config(
    *,
    args_yaml: str | Path,
    output: str | Path,
    force: bool = False,
) -> Path:
    args_path = Path(args_yaml)
    if not args_path.is_absolute():
        args_path = (Path.cwd() / args_path).resolve()
    out_path = Path(output)
    if not args_path.exists():
        raise FileNotFoundError(f"--args not found: {args_path}")

    if out_path.suffix.lower() != ".json":
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / "train_config_import.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        raise FileExistsError(f"output already exists: {out_path} (use --force to overwrite)")

    cfg = _load_config(args_path)
    train = project_ultralytics_args(cfg, source={"from": "ultralytics", "args_yaml": str(args_path)})
    return _write_train_config(out_path, train=train, force=force)


def import_mmdet_config(
    *,
    config: str | Path,
    output: str | Path,
    force: bool = False,
) -> Path:
    train = project_mmdet_config(config=config)
    return _write_train_config(output, train=train, force=force)


def import_detectron2_config(
    *,
    config: str | Path,
    output: str | Path,
    force: bool = False,
) -> Path:
    train = project_detectron2_config(config=config)
    return _write_train_config(output, train=train, force=force)


def import_yolox_config(
    *,
    config: str | Path,
    output: str | Path,
    force: bool = False,
) -> Path:
    train = project_yolox_exp(config=config)
    return _write_train_config(output, train=train, force=force)
