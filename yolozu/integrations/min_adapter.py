"""Minimal shared adapters for Ultralytics/DETR integration flows.

This module provides a small common surface for:
- dataset conversion (COCO/Ultralytics/internal -> YOLOZU internal wrapper),
- train template helpers,
- ONNX export template helpers,
- prediction canonicalization to the YOLOZU predictions interface contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yolozu.datasets.dataset import build_manifest
from yolozu.datasets.imports import import_coco_instances_dataset
from yolozu.datasets.migrate import migrate_coco_dataset_wrapper, migrate_ultralytics_dataset_wrapper
from yolozu.predictions import canonicalize_predictions, normalize_predictions_payload
from yolozu.predictions.predictions_transform import load_classes_json, normalize_class_ids

__all__ = [
    "DatasetResolution",
    "resolve_internal_dataset",
    "write_ultralytics_data_yaml",
    "canonicalize_predictions_file",
    "layered_support_matrix",
]


_LAYER_CONTRACT: dict[str, str] = {
    "trainer_runner": "Framework training/runtime entrypoints (trainer orchestration).",
    "repo_impl": "Repository-level integration wrappers (Ultralytics/HF/OpenMMLab-derived entry).",
    "export_deploy": "Export/deploy surface (ONNX + optional TensorRT/OpenVINO/Triton handoff).",
}


def layered_support_matrix() -> dict[str, Any]:
    return {
        "layers": dict(_LAYER_CONTRACT),
        "providers": {
            "ultralytics_yolo": {
                "trainer_runner": "ultralytics.YOLO.train",
                "repo_impl": "tools/support_ultralytics_detr.py train-ultralytics",
                "export_deploy": "tools/support_ultralytics_detr.py export-onnx --provider ultralytics",
            },
            "hf_detr_rtdetr": {
                "trainer_runner": "transformers/accelerate entry (template + external script bridge)",
                "repo_impl": "tools/support_ultralytics_detr.py train-hf-detr",
                "export_deploy": "tools/support_ultralytics_detr.py export-onnx --provider hf_detr",
            },
            "onnx_deploy": {
                "trainer_runner": "N/A (export/deploy layer)",
                "repo_impl": "tools/support_ultralytics_detr.py export-onnx",
                "export_deploy": "ONNX + optional TensorRT engine build command template",
            },
        },
    }


@dataclass(frozen=True)
class DatasetResolution:
    source_format: str
    dataset_root: Path
    split: str
    dataset_wrapper: Path | None
    notes: list[str]


def _as_abs(path_like: str | Path) -> Path:
    p = Path(path_like).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def _detect_source_format(
    *,
    source_format: str,
    dataset: Path | None,
    instances_json: Path | None,
    images_dir: Path | None,
) -> str:
    fmt = str(source_format).strip().lower()
    if fmt and fmt != "auto":
        return fmt

    if dataset is not None and dataset.is_file() and dataset.suffix.lower() in (".yaml", ".yml"):
        return "ultralytics"
    if dataset is not None and dataset.is_dir():
        if (dataset / "images").exists() and (dataset / "labels").exists():
            return "internal"
        if (dataset / "annotations").exists() and (dataset / "images").exists():
            return "coco"
    if instances_json is not None and images_dir is not None:
        return "coco_instances"
    raise ValueError("could not auto-detect dataset source; provide --from")


def _infer_split(dataset_root: Path, split: str | None) -> str:
    if split:
        return str(split)
    labels_root = dataset_root / "labels"
    if labels_root.exists():
        if (labels_root / "train").exists():
            return "train"
        if (labels_root / "val").exists():
            return "val"
        for child in sorted(labels_root.iterdir()):
            if child.is_dir():
                return str(child.name)
    return "val"


def resolve_internal_dataset(
    *,
    source_format: str = "auto",
    dataset: str | Path | None = None,
    split: str | None = None,
    output: str | Path | None = None,
    instances_json: str | Path | None = None,
    images_dir: str | Path | None = None,
    force: bool = False,
) -> DatasetResolution:
    """Resolve any supported source dataset into a YOLOZU internal dataset root.

    Returned `dataset_root` always points to a directory accepted by
    `yolozu.dataset.build_manifest(...)`.
    """

    notes: list[str] = []
    dataset_path = _as_abs(dataset) if dataset is not None else None
    instances_path = _as_abs(instances_json) if instances_json is not None else None
    images_dir_path = _as_abs(images_dir) if images_dir is not None else None

    fmt = _detect_source_format(
        source_format=source_format,
        dataset=dataset_path,
        instances_json=instances_path,
        images_dir=images_dir_path,
    )
    out_root = _as_abs(output) if output is not None else (Path.cwd() / "runs" / "integration_dataset_cache")
    out_root.mkdir(parents=True, exist_ok=True)

    wrapper: Path | None = None
    effective_root: Path
    effective_split: str

    if fmt == "internal":
        if dataset_path is None:
            raise ValueError("--dataset is required for --from internal")
        effective_root = dataset_path
        effective_split = _infer_split(effective_root, split)
        notes.append("dataset source=internal (no conversion)")
    elif fmt == "ultralytics":
        if dataset_path is None:
            raise ValueError("--dataset is required for --from ultralytics")
        wrapper_root = out_root / "ultralytics_wrapper"
        wrapper = migrate_ultralytics_dataset_wrapper(
            data_yaml=dataset_path,
            args_yaml=None,
            split=split,
            task=None,
            output=wrapper_root,
            force=bool(force),
        )
        effective_root = wrapper.parent
        effective_split = str(build_manifest(effective_root, split=split).get("split") or (split or "val"))
        notes.append("converted Ultralytics data.yaml -> YOLOZU dataset wrapper")
    elif fmt == "coco":
        if dataset_path is None:
            raise ValueError("--dataset is required for --from coco")
        coco_split = str(split or "val2017")
        wrapper_root = out_root / "coco_wrapper"
        wrapper = migrate_coco_dataset_wrapper(
            coco_root=dataset_path,
            split=coco_split,
            output=wrapper_root,
            instances_json=None,
            mode="manifest",
            include_crowd=False,
            force=bool(force),
        )
        effective_root = wrapper.parent
        effective_split = str(build_manifest(effective_root, split=coco_split).get("split") or coco_split)
        notes.append("converted COCO root -> YOLOZU dataset wrapper")
    elif fmt == "coco_instances":
        if instances_path is None or images_dir_path is None:
            raise ValueError("--instances-json and --images-dir are required for --from coco_instances")
        coco_split = str(split or "val")
        wrapper_root = out_root / "coco_instances_wrapper"
        wrapper = import_coco_instances_dataset(
            instances_json=instances_path,
            images_dir=images_dir_path,
            split=coco_split,
            output=wrapper_root,
            include_crowd=False,
            force=bool(force),
        )
        effective_root = wrapper.parent
        effective_split = str(build_manifest(effective_root, split=coco_split).get("split") or coco_split)
        notes.append("converted COCO instances/images pair -> YOLOZU dataset wrapper")
    else:
        raise ValueError(f"unsupported --from value: {fmt}")

    manifest = build_manifest(effective_root, split=effective_split)
    records = manifest.get("images") or []
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError(f"resolved dataset has no images: {effective_root} (split={effective_split})")

    return DatasetResolution(
        source_format=fmt,
        dataset_root=effective_root,
        split=str(manifest.get("split") or effective_split),
        dataset_wrapper=wrapper,
        notes=notes,
    )


def _collect_class_names(dataset_root: Path, split: str) -> list[str]:
    labels_dir = dataset_root / "labels" / split
    classes_json = labels_dir / "classes.json"
    if classes_json.exists():
        try:
            payload = json.loads(classes_json.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                names = [str(x) for x in payload]
                if names:
                    return names
            if isinstance(payload, dict):
                names = payload.get("class_names")
                if isinstance(names, list) and names:
                    return [str(x) for x in names]
                names_map = payload.get("names")
                if isinstance(names_map, dict):
                    pairs: list[tuple[int, str]] = []
                    for key, value in names_map.items():
                        try:
                            pairs.append((int(key), str(value)))
                        except (TypeError, ValueError):
                            continue
                    if pairs:
                        return [v for _, v in sorted(pairs, key=lambda x: x[0])]
        except (OSError, json.JSONDecodeError):
            # Fall back to classes.txt or discovered class IDs when metadata is absent.
            pass

    classes_txt = labels_dir / "classes.txt"
    if classes_txt.exists():
        names = [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
        if names:
            return names

    max_class = -1
    for txt in sorted(labels_dir.glob("*.txt")):
        for line in txt.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cid = int(float(parts[0]))
            except (TypeError, ValueError):
                continue
            max_class = max(max_class, cid)
    if max_class < 0:
        return ["class_0"]
    return [f"class_{i}" for i in range(max_class + 1)]


def write_ultralytics_data_yaml(*, dataset_root: str | Path, split: str, output: str | Path) -> Path:
    """Write an Ultralytics-compatible data.yaml (JSON-encoded YAML)."""

    root = _as_abs(dataset_root)
    out = _as_abs(output)
    train_split = str(split)
    val_split = "val" if (root / "images" / "val").exists() else train_split
    names = _collect_class_names(root, train_split)
    payload = {
        "path": str(root),
        "train": f"images/{train_split}",
        "val": f"images/{val_split}",
        "names": {int(i): str(name) for i, name in enumerate(names)},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML and avoids hard dependency on PyYAML.
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def canonicalize_predictions_file(
    *,
    input_path: str | Path,
    output_path: str | Path,
    strict: bool = False,
    classes_json: str | Path | None = None,
    assume_class_id_is_category_id: bool = False,
) -> dict[str, Any]:
    """Normalize + canonicalize prediction JSON into YOLOZU entry contract."""

    in_path = _as_abs(input_path)
    out_path = _as_abs(output_path)
    raw = json.loads(in_path.read_text(encoding="utf-8"))
    entries, meta = normalize_predictions_payload(raw)

    transform_warnings: list[str] = []
    if classes_json is not None:
        mapping = load_classes_json(_as_abs(classes_json))
        transformed = normalize_class_ids(
            entries,
            classes_json=mapping,
            assume_class_id_is_category_id=bool(assume_class_id_is_category_id),
        )
        entries = transformed.entries
        transform_warnings.extend(transformed.warnings)

    canonical = canonicalize_predictions(
        entries,
        strict=bool(strict),
        policy=("error" if strict else "clamp"),
        unknown_keys=("error" if strict else "warn"),
    )

    payload: Any
    if isinstance(raw, dict) and "predictions" in raw:
        wrapped = dict(raw)
        wrapped["predictions"] = canonical.entries
        if isinstance(meta, dict):
            wrapped["meta"] = dict(meta)
            wrapped["meta"]["canonicalized"] = True
            wrapped["meta"]["canonicalized_at"] = True
            wrapped["meta"]["canonicalization_warnings"] = canonical.warnings
        payload = wrapped
    else:
        payload = canonical.entries

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "input": str(in_path),
        "output": str(out_path),
        "entries": len(canonical.entries),
        "transform_warnings": transform_warnings,
        "canonicalization_warnings": canonical.warnings,
    }
