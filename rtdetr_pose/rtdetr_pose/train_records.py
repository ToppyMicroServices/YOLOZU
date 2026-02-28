"""Record loading and dataset checks for train_minimal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def load_train_records(
    *,
    args: Any,
    dataset_root: Path,
    workspace_root: Path,
    build_manifest_fn: Callable[..., dict[str, Any]],
    extract_manifest_keypoints_meta_fn: Callable[[dict[str, Any] | None], tuple[list[str], list[list[int]]]],
) -> tuple[list[dict[str, Any]], list[str], list[list[int]]]:
    records: list[dict[str, Any]]
    keypoint_names: list[str] = []
    keypoint_skeleton: list[list[int]] = []

    if getattr(args, "records_json", None):
        records_path = Path(str(args.records_json))
        if not records_path.is_absolute():
            records_path = (workspace_root / records_path).resolve()
            if not records_path.exists():
                records_path = (workspace_root.parent / Path(str(args.records_json))).resolve()
        if not records_path.exists():
            raise SystemExit(f"records json not found: {records_path}")
        loaded = json.loads(records_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "images" in loaded:
            loaded = loaded.get("images")
        if not isinstance(loaded, list):
            raise SystemExit(f"records json must be a list or {{images:[...]}}: {records_path}")
        records = [r for r in loaded if isinstance(r, dict)]
    else:
        manifest = build_manifest_fn(dataset_root, split=args.split)
        records = list(manifest.get("images") or [])
        keypoint_names, keypoint_skeleton = extract_manifest_keypoints_meta_fn(manifest)

    if getattr(args, "extra_records_json", None):
        extra_path = Path(str(args.extra_records_json))
        if not extra_path.is_absolute():
            extra_path = (workspace_root / extra_path).resolve()
            if not extra_path.exists():
                extra_path = (workspace_root.parent / Path(str(args.extra_records_json))).resolve()
        if not extra_path.exists():
            raise SystemExit(f"extra records json not found: {extra_path}")
        loaded = json.loads(extra_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "images" in loaded:
            loaded = loaded.get("images")
        if not isinstance(loaded, list):
            raise SystemExit(f"extra records json must be a list or {{images:[...]}}: {extra_path}")
        extra = [r for r in loaded if isinstance(r, dict)]
        if extra:
            records = list(records) + extra

    if not records:
        raise SystemExit(
            f"No records found under {dataset_root}. "
            "Fetch coco128 first: bash tools/fetch_coco128.sh"
        )

    return records, keypoint_names, keypoint_skeleton


def enforce_strict_task_data(*, args: Any, records: list[dict[str, Any]], workspace_root: Path) -> None:
    if not bool(getattr(args, "strict_task_data", False)):
        return

    issues: list[str] = []
    if not bool(getattr(args, "real_images", False)):
        issues.append("strict-task-data requires --real-images to avoid synthetic image fallback")

    records_with_labels = 0
    records_with_keypoints = 0
    records_with_depth = 0
    records_with_pose = 0
    records_with_intrinsics = 0
    missing_image_path = 0

    for record in records:
        labels = record.get("labels")
        if isinstance(labels, list) and labels:
            records_with_labels += 1
            for inst in labels:
                if not isinstance(inst, dict):
                    continue
                kps = inst.get("keypoints")
                if isinstance(kps, list) and kps:
                    records_with_keypoints += 1
                    break

        depth_value = record.get("depth_path")
        if depth_value is None:
            depth_value = record.get("depth")
        if depth_value is not None:
            records_with_depth += 1

        pose_value = record.get("pose")
        if (
            pose_value is not None
            or record.get("R_gt") is not None
            or record.get("t_gt") is not None
        ):
            records_with_pose += 1
        if record.get("K_gt") is not None or record.get("intrinsics") is not None:
            records_with_intrinsics += 1

        image_path = str(record.get("image_path", "") or "").strip()
        if not image_path:
            missing_image_path += 1
        else:
            image_path_obj = Path(image_path)
            if not image_path_obj.is_absolute():
                image_path_obj = (workspace_root / image_path_obj).resolve()
            if not image_path_obj.exists():
                missing_image_path += 1

    if records_with_labels <= 0:
        issues.append("no records contain labels[] (bbox supervision missing)")
    if missing_image_path > 0:
        issues.append(f"{missing_image_path} records are missing a valid image_path")

    if int(getattr(args, "num_keypoints", 0) or 0) > 0 and records_with_keypoints <= 0:
        issues.append("--num-keypoints > 0 but no label instances contain keypoints")
    if str(getattr(args, "depth_mode", "none") or "none").strip().lower() != "none" and records_with_depth <= 0:
        issues.append("--depth-mode is enabled but no records contain depth/depth_path")
    pose_requested = bool(
        bool(getattr(args, "use_matcher", False))
        and not bool(getattr(args, "synthetic_pose", False))
        and (
            float(getattr(args, "cost_rot", 0.0) or 0.0) > 0.0
            or float(getattr(args, "cost_t", 0.0) or 0.0) > 0.0
        )
    )
    if pose_requested:
        if records_with_pose <= 0:
            issues.append("pose costs requested but no records contain pose fields (pose/R_gt/t_gt)")
        if records_with_intrinsics <= 0:
            issues.append("pose costs requested but no records contain intrinsics (K_gt/intrinsics)")

    if issues:
        message = "strict-task-data checks failed:\n  - " + "\n  - ".join(issues)
        raise SystemExit(message)


def collect_record_stats(records: list[dict[str, Any]]) -> dict[str, int]:
    stats = {
        "mask": 0,
        "depth": 0,
        "pose": 0,
        "intrinsics": 0,
        "cad_points": 0,
    }
    for rec in records:
        if rec.get("mask_path") is not None:
            stats["mask"] += 1
        if rec.get("depth_path") is not None:
            stats["depth"] += 1
        if rec.get("R_gt") is not None or rec.get("t_gt") is not None or rec.get("pose") is not None:
            stats["pose"] += 1
        if rec.get("K_gt") is not None or rec.get("intrinsics") is not None:
            stats["intrinsics"] += 1
        if rec.get("cad_points") is not None:
            stats["cad_points"] += 1
    return stats


def log_dataset_stats_and_update_run_record(
    *,
    args: Any,
    is_main: bool,
    records: list[dict[str, Any]],
    run_record: dict[str, Any],
) -> None:
    if not is_main:
        return
    stats = collect_record_stats(records)
    print("dataset_stats " + " ".join(f"{key}={value}" for key, value in sorted(stats.items())))
    depth_mode = str(getattr(args, "depth_mode", "none") or "none").strip().lower()
    depth_unit = str(getattr(args, "depth_unit", "unspecified") or "unspecified").strip().lower()
    depth_used = bool(depth_mode != "none" and int(stats.get("depth", 0)) > 0)
    run_record["depth_used"] = bool(depth_used)
    run_record["depth_unit"] = depth_unit
    run_record["depth_scale"] = float(getattr(args, "depth_scale", 1.0) or 1.0)
    run_record["depth_mode"] = depth_mode


def maybe_write_run_meta_and_fracal_stats(
    *,
    args: Any,
    is_main: bool,
    run_record: dict[str, Any],
    records: list[dict[str, Any]],
    dataset_root: Path,
    build_fracal_stats_fn: Callable[..., dict[str, Any]],
) -> None:
    if not is_main:
        return

    if getattr(args, "run_meta_out", None):
        out_path = Path(str(args.run_meta_out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(run_record, indent=2, sort_keys=True), encoding="utf-8")

    if getattr(args, "fracal_stats_out", None):
        stats_path = Path(str(args.fracal_stats_out))
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        fracal_stats = build_fracal_stats_fn(
            records,
            task=str(getattr(args, "fracal_stats_task", "bbox") or "bbox"),
            allow_rgb_masks=bool(getattr(args, "fracal_allow_rgb_masks", False)),
        )
        fracal_stats["source"] = {
            "kind": "train_records",
            "dataset_root": str(dataset_root),
            "split": (None if args.records_json else str(getattr(args, "split", "") or "")),
            "records_json": (str(args.records_json) if getattr(args, "records_json", None) else None),
        }
        stats_path.write_text(json.dumps(fracal_stats, indent=2, sort_keys=True), encoding="utf-8")
        print(
            "fracal_stats "
            f"task={str(fracal_stats.get('task', 'bbox'))} classes={int(fracal_stats.get('summary', {}).get('classes', 0))} "
            f"instances_total={int(fracal_stats.get('summary', {}).get('instances_total', 0))} "
            f"out={stats_path}"
        )


def resolve_val_records(
    *,
    args: Any,
    dataset_root: Path,
    build_manifest_fn: Callable[..., dict[str, Any]],
    flatten_records_for_map_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    val_split = str(args.val_split) if getattr(args, "val_split", None) else None
    if val_split is None:
        candidate = dataset_root / "images" / "val2017"
        if candidate.exists():
            val_split = "val2017"

    val_records: list[dict[str, Any]] = []
    if val_split:
        try:
            val_manifest = build_manifest_fn(dataset_root, split=val_split)
            val_records = list(val_manifest.get("images") or [])
        except Exception:
            val_records = []

    if val_records and int(getattr(args, "val_max_images", 0) or 0) > 0:
        val_records = list(val_records)[: int(args.val_max_images)]
    val_records_map = flatten_records_for_map_fn(val_records)
    return val_records, val_records_map
