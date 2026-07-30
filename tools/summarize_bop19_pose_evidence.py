#!/usr/bin/env python3
"""Summarize official BOP19 and task-native T-LESS pose evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine official BOP19 scores with BOP-toolkit ADD/ADD-S errors "
            "and matched rotation/translation metrics for at least three seeds."
        )
    )
    parser.add_argument("--bop-root", required=True, help="T-LESS root containing test_primesense/.")
    parser.add_argument("--targets", required=True, help="Official test_targets_bop19.json.")
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="BOP19 result CSV. Repeat for each seed (at least three).",
    )
    parser.add_argument(
        "--official-eval-root",
        required=True,
        help="Root produced by the official eval_bop19_pose.py.",
    )
    parser.add_argument(
        "--native-errors-root",
        required=True,
        help="Root produced by official eval_calc_errors.py for add, adi, and ad with n_top=-1.",
    )
    parser.add_argument("--output", required=True, help="Fresh qualification summary JSON.")
    parser.add_argument("--role", choices=("primary", "independent"), default="primary")
    parser.add_argument(
        "--source-summary",
        default=None,
        help="Primary summary required for --role independent.",
    )
    parser.add_argument(
        "--toolkit-commit",
        required=True,
        help="Pinned official bop_toolkit git commit used for evaluation.",
    )
    return parser.parse_args(argv)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else None


def _rotation_error_deg(estimate: list[float], ground_truth: list[float]) -> float:
    relative = [
        sum(estimate[row * 3 + index] * ground_truth[column * 3 + index] for index in range(3))
        for row in range(3)
        for column in range(3)
    ]
    cosine = max(-1.0, min(1.0, (relative[0] + relative[4] + relative[8] - 1.0) * 0.5))
    return float(math.degrees(math.acos(cosine)))


def _translation_error_mm(estimate: list[float], ground_truth: list[float]) -> float:
    return float(math.sqrt(sum((estimate[index] - ground_truth[index]) ** 2 for index in range(3))))


def _load_results(path: Path) -> dict[tuple[int, int, int, int], dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["scene_id"]), int(row["im_id"]), int(row["obj_id"]))
            grouped.setdefault(key, []).append(
                {
                    "score": float(row["score"]),
                    "R": [float(value) for value in row["R"].split()],
                    "t": [float(value) for value in row["t"].split()],
                }
            )
    indexed: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for key, rows in grouped.items():
        for estimate_id, row in enumerate(sorted(rows, key=lambda item: item["score"], reverse=True)):
            indexed[(*key, estimate_id)] = row
    return indexed


def _error_index(root: Path, result_name: str, metric: str) -> dict[tuple[int, int, int, int], dict[str, float]]:
    directory = root / result_name / f"error={metric}_ntop=-1"
    paths = sorted(directory.glob("errors_*.json"))
    if not paths:
        raise FileNotFoundError(f"official {metric} errors not found: {directory}")
    index: dict[tuple[int, int, int, int], dict[str, float]] = {}
    for path in paths:
        payload = _json(path)
        if not isinstance(payload, list):
            raise ValueError(f"expected error list: {path}")
        for row in payload:
            values = {
                str(gt_id): float(errors[0])
                for gt_id, errors in row["errors"].items()
                if isinstance(errors, list) and errors
            }
            key = (int(row["scene_id"]), int(row["im_id"]), int(row["obj_id"]), int(row["est_id"]))
            index[key] = values
    return index


def _load_scene_gt(bop_root: Path, scene_id: int) -> dict[str, Any]:
    path = bop_root / "test_primesense" / f"{scene_id:06d}" / "scene_gt.json"
    payload = _json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected scene_gt object: {path}")
    return payload


def _matched_metrics(
    *,
    bop_root: Path,
    targets: list[dict[str, Any]],
    results: dict[tuple[int, int, int, int], dict[str, Any]],
    ad_errors: dict[tuple[int, int, int, int], dict[str, float]],
    add_errors: dict[tuple[int, int, int, int], dict[str, float]],
    adi_errors: dict[tuple[int, int, int, int], dict[str, float]],
    model_info: dict[str, Any],
) -> dict[str, Any]:
    rotations: list[float] = []
    translations: list[float] = []
    add_values: list[float] = []
    adi_values: list[float] = []
    ad_values: list[float] = []
    successes = 0
    matched = 0
    target_instances = 0
    scenes: dict[int, dict[str, Any]] = {}

    for target in targets:
        scene_id = int(target["scene_id"])
        image_id = int(target["im_id"])
        object_id = int(target["obj_id"])
        count = max(1, int(target.get("inst_count", 1)))
        target_instances += count
        if scene_id not in scenes:
            scenes[scene_id] = _load_scene_gt(bop_root, scene_id)
        gt_rows = scenes[scene_id][str(image_id)]
        valid_gt = {
            str(index): row
            for index, row in enumerate(gt_rows)
            if int(row["obj_id"]) == object_id
        }
        unused = set(valid_gt)
        diameter = float(model_info[str(object_id)]["diameter"])
        for estimate_id in range(count):
            key = (scene_id, image_id, object_id, estimate_id)
            estimate = results.get(key)
            candidate_errors = ad_errors.get(key, {})
            candidates = [
                (float(error), gt_id)
                for gt_id, error in candidate_errors.items()
                if gt_id in unused and math.isfinite(float(error))
            ]
            if estimate is None or not candidates:
                continue
            ad_error, gt_id = min(candidates)
            unused.remove(gt_id)
            ground_truth = valid_gt[gt_id]
            matched += 1
            rotations.append(_rotation_error_deg(estimate["R"], ground_truth["cam_R_m2c"]))
            translations.append(_translation_error_mm(estimate["t"], ground_truth["cam_t_m2c"]))
            ad_values.append(ad_error)
            add_values.append(float(add_errors.get(key, {}).get(gt_id, math.inf)))
            adi_values.append(float(adi_errors.get(key, {}).get(gt_id, math.inf)))
            successes += int(ad_error <= 0.1 * diameter)

    return {
        "target_instances": target_instances,
        "matched_instances": matched,
        "unmatched_instances": target_instances - matched,
        "rotation_error_deg_mean": _mean(rotations),
        "translation_error_mm_mean": _mean(translations),
        "add_error_mm_mean": _mean(add_values),
        "add_s_error_mm_mean": _mean(adi_values),
        "symmetry_aware_ad_error_mm_mean": _mean(ad_values),
        "pose_success_0.1d_count": successes,
        "pose_success_0.1d_rate": float(successes / target_instances) if target_instances else None,
    }


def _seed_from_name(name: str) -> int:
    import re

    match = re.search(r"(?:^|[-_])s(?:eed)?(\d+)(?:[-_]|$)", name)
    if not match:
        raise ValueError(f"result filename must contain a seed token such as -s11_: {name}")
    return int(match.group(1))


def _same_metrics(source: dict[str, Any], current: dict[str, Any]) -> bool:
    source_runs = {int(row["seed"]): row for row in source.get("runs", [])}
    current_runs = {int(row["seed"]): row for row in current.get("runs", [])}
    if set(source_runs) != set(current_runs):
        return False
    keys = (
        "bop19_average_recall",
        "bop19_average_recall_vsd",
        "bop19_average_recall_mssd",
        "bop19_average_recall_mspd",
        "rotation_error_deg_mean",
        "translation_error_mm_mean",
        "add_error_mm_mean",
        "add_s_error_mm_mean",
        "symmetry_aware_ad_error_mm_mean",
        "pose_success_0.1d_rate",
    )
    for seed in source_runs:
        source_values = {**source_runs[seed]["official_bop19"], **source_runs[seed]["task_native"]}
        current_values = {**current_runs[seed]["official_bop19"], **current_runs[seed]["task_native"]}
        for key in keys:
            left, right = source_values.get(key), current_values.get(key)
            if left is None or right is None:
                if left is not right:
                    return False
            elif not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9):
                return False
    return True


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    bop_root = Path(args.bop_root).expanduser().resolve()
    targets_path = Path(args.targets).expanduser().resolve()
    official_root = Path(args.official_eval_root).expanduser().resolve()
    native_root = Path(args.native_errors_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    result_paths = [Path(value).expanduser().resolve() for value in args.result]
    if len(result_paths) < 3:
        raise SystemExit("--result must be repeated for at least three seeds")
    for path in (bop_root, targets_path, official_root, native_root, *result_paths):
        if not path.exists():
            raise SystemExit(f"input not found: {path}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace existing output: {output}")
    if args.role == "independent" and not args.source_summary:
        raise SystemExit("--source-summary is required when --role independent")

    targets = _json(targets_path)
    model_info_path = bop_root / "models_eval" / "models_info.json"
    model_info = _json(model_info_path)
    runs = []
    for result_path in result_paths:
        result_name = result_path.stem
        official_path = official_root / result_name / "scores_bop19.json"
        report_path = result_path.with_suffix(result_path.suffix + ".report.json")
        if not official_path.is_file() or not report_path.is_file():
            raise SystemExit(f"official score or exporter report missing for: {result_path}")
        indexed_results = _load_results(result_path)
        native = _matched_metrics(
            bop_root=bop_root,
            targets=targets,
            results=indexed_results,
            ad_errors=_error_index(native_root, result_name, "ad"),
            add_errors=_error_index(native_root, result_name, "add"),
            adi_errors=_error_index(native_root, result_name, "adi"),
            model_info=model_info,
        )
        runs.append(
            {
                "seed": _seed_from_name(result_name),
                "result_csv": str(result_path),
                "result_csv_sha256": _sha256(result_path),
                "export_report": str(report_path),
                "export_report_sha256": _sha256(report_path),
                "official_scores": str(official_path),
                "official_scores_sha256": _sha256(official_path),
                "official_bop19": _json(official_path),
                "task_native": native,
            }
        )
    runs.sort(key=lambda row: int(row["seed"]))

    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "bop19_tless_pose_qualification",
        "role": args.role,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "bop_toolkit_commit": args.toolkit_commit,
        },
        "dataset": {
            "name": "BOP T-LESS",
            "license": "CC-BY-4.0",
            "targets": str(targets_path),
            "targets_sha256": _sha256(targets_path),
            "target_entries": len(targets),
            "target_instances": sum(max(1, int(row.get("inst_count", 1))) for row in targets),
            "models_info": str(model_info_path),
            "models_info_sha256": _sha256(model_info_path),
        },
        "protocol": {
            "official_evaluator": "bop_toolkit scripts/eval_bop19_pose.py",
            "official_metrics": ["VSD", "MSSD", "MSPD"],
            "additional_official_error_calculations": ["ADD", "ADI", "AD"],
            "matching": "BOP score-ordered estimates matched to unused GT by minimum symmetry-aware AD",
            "test_ground_truth_used_for_inference": False,
            "translation_unit": "millimetre",
            "pose_success_threshold": "symmetry-aware AD <= 0.1 * model diameter",
        },
        "runs": runs,
        "decision": {
            "status": "hold",
            "efficacy": "not_established",
            "reasons": [
                "official and task-native measurements are recorded without converting null to zero",
                "the bounded local training budget is not a competitive BOP benchmark claim",
            ],
        },
    }
    if args.role == "independent":
        source_path = Path(args.source_summary).expanduser().resolve()
        if not source_path.is_file():
            raise SystemExit(f"source summary not found: {source_path}")
        source = _json(source_path)
        reproduced = _same_metrics(source, summary)
        summary["reproduction"] = {
            "source_summary": str(source_path),
            "source_summary_sha256": _sha256(source_path),
            "same_seed_metrics_within_1e-9": reproduced,
        }
        if not reproduced:
            summary["decision"]["reasons"].insert(0, "independent metrics did not reproduce")
        else:
            summary["decision"]["reasons"].insert(0, "independent metrics reproduced")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
