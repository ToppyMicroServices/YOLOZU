#!/usr/bin/env python3
"""Summarize baseline/trained BOP T-LESS runs without promoting diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate BOP T-LESS baseline/trained run_metadata.json files into "
            "a task-native qualification summary."
        )
    )
    parser.add_argument("--run-base", required=True, help="Directory containing per-run subdirectories.")
    parser.add_argument("--dataset", required=True, help="Converted YOLOZU BOP dataset root.")
    parser.add_argument("--download-manifest", required=True, help="BOP download_manifest.json.")
    parser.add_argument("--output", required=True, help="Output qualification_summary.json.")
    parser.add_argument(
        "--role",
        choices=("primary", "independent"),
        default="primary",
        help="Reproduction role (default: primary).",
    )
    parser.add_argument(
        "--source-summary",
        default=None,
        help="Primary summary to compare semantically when --role independent.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def _delta(before: Any, after: Any) -> Any:
    if isinstance(before, dict) and isinstance(after, dict):
        return {
            key: _delta(before.get(key), after.get(key))
            for key in sorted(set(before) | set(after))
        }
    if (
        isinstance(before, (int, float))
        and not isinstance(before, bool)
        and isinstance(after, (int, float))
        and not isinstance(after, bool)
    ):
        return float(after) - float(before)
    return None


def _semantic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "seed": row.get("seed"),
            "trained_epochs": row.get("trained_epochs"),
            "before": row.get("before"),
            "after": row.get("after"),
            "delta": row.get("delta"),
            "thresholds": row.get("thresholds"),
        }
        for row in rows
    ]


def main() -> int:
    args = _parse_args()
    run_base = Path(args.run_base).resolve()
    dataset = Path(args.dataset).resolve()
    download_manifest = Path(args.download_manifest).resolve()
    output = Path(args.output).resolve()

    if not run_base.is_dir():
        raise SystemExit(f"run base not found: {run_base}")
    if not dataset.is_dir():
        raise SystemExit(f"dataset not found: {dataset}")
    if not download_manifest.is_file():
        raise SystemExit(f"download manifest not found: {download_manifest}")

    metadata_paths = sorted(run_base.glob("*/run_metadata.json"))
    if not metadata_paths:
        raise SystemExit(f"no run_metadata.json files found under: {run_base}")

    grouped: dict[int, dict[str, list[tuple[Path, dict[str, Any]]]]] = {}
    for path in metadata_paths:
        metadata = _load_object(path)
        seed = int(metadata.get("seed"))
        kind = str(metadata.get("run_kind"))
        if kind not in {"baseline", "trained"}:
            continue
        grouped.setdefault(seed, {"baseline": [], "trained": []})[kind].append((path, metadata))

    rows: list[dict[str, Any]] = []
    for seed in sorted(grouped):
        baseline_runs = grouped[seed]["baseline"]
        trained_runs = grouped[seed]["trained"]
        if len(baseline_runs) != 1:
            raise SystemExit(f"seed {seed}: expected exactly one baseline, found {len(baseline_runs)}")
        if not trained_runs:
            raise SystemExit(f"seed {seed}: no trained run found")
        baseline_path, baseline = baseline_runs[0]
        trained_path, trained = max(trained_runs, key=lambda item: int(item[1].get("epochs") or 0))
        before = {
            "bbox": baseline.get("bbox_metrics") or baseline.get("metrics") or {},
            "task_native": baseline.get("task_native_metrics") or {},
        }
        after = {
            "bbox": trained.get("bbox_metrics") or trained.get("metrics") or {},
            "task_native": trained.get("task_native_metrics") or {},
        }
        rows.append(
            {
                "seed": seed,
                "trained_epochs": int(trained.get("epochs") or 0),
                "before": before,
                "after": after,
                "delta": _delta(before, after),
                "thresholds": trained.get("thresholds") or baseline.get("thresholds") or {},
                "baseline": {
                    "metadata": str(baseline_path),
                    "metadata_sha256": _sha256(baseline_path),
                    "checkpoint_sha256": baseline.get("checkpoint_sha256"),
                    "runtime_seconds": baseline.get("runtime_seconds"),
                    "peak_rss_bytes": (baseline.get("train_resource") or {}).get("peak_rss_bytes"),
                },
                "trained": {
                    "metadata": str(trained_path),
                    "metadata_sha256": _sha256(trained_path),
                    "checkpoint_sha256": trained.get("checkpoint_sha256"),
                    "runtime_seconds": trained.get("runtime_seconds"),
                    "peak_rss_bytes": (trained.get("train_resource") or {}).get("peak_rss_bytes"),
                },
            }
        )

    prepare_summary_path = dataset / "prepare_summary.json"
    prepare_summary = _load_object(prepare_summary_path) if prepare_summary_path.is_file() else {}
    strict_ground_truth = bool((prepare_summary.get("checks") or {}).get("strict_ground_truth"))

    comparison: dict[str, Any] | None = None
    if args.role == "independent":
        if not args.source_summary:
            raise SystemExit("--source-summary is required when --role independent")
        source_path = Path(args.source_summary).resolve()
        source = _load_object(source_path)
        source_semantics = {
            "dataset_archives": source.get("dataset_archives"),
            "ground_truth": source.get("ground_truth"),
            "rows": _semantic_rows(source.get("rows") or []),
        }
        rerun_semantics = {
            "dataset_archives": _load_object(download_manifest).get("archives"),
            "ground_truth": {
                "strict": strict_ground_truth,
                "provenance": prepare_summary.get("label_provenance") or {},
                "annotation_counts": prepare_summary.get("annotation_counts") or {},
            },
            "rows": _semantic_rows(rows),
        }
        comparison = {
            "source_summary": str(source_path),
            "source_summary_sha256": _sha256(source_path),
            "semantic_match": source_semantics == rerun_semantics,
        }

    download = _load_object(download_manifest)
    qualification_reasons = []
    if len(rows) < 3:
        qualification_reasons.append("fewer_than_three_seeds")
    if not strict_ground_truth:
        qualification_reasons.append("strict_ground_truth_not_confirmed")
    if args.role == "independent" and not bool((comparison or {}).get("semantic_match")):
        qualification_reasons.append("independent_semantic_mismatch")
    qualification_reasons.extend(
        [
            "diagnostic_frame_holdout_is_not_official_bop_test",
            "no_predefined_promotion_threshold_pass",
        ]
    )

    first_metadata = _load_object(metadata_paths[0])
    payload = {
        "schema_version": 1,
        "kind": "bop_tless_multitask_qualification",
        "role": args.role,
        "protocol": {
            "dataset": "BOP T-LESS train_primesense",
            "scope": "object 6DoF pose; human 3D skeleton pose unsupported",
            "partition": "frame_id modulo 5, remainder 0 held out",
            "status": "diagnostic_frame_holdout_not_official_bop_test",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "dataset": str(dataset),
        "dataset_prepare_summary": str(prepare_summary_path),
        "dataset_prepare_summary_sha256": (
            _sha256(prepare_summary_path) if prepare_summary_path.is_file() else None
        ),
        "dataset_download_manifest": str(download_manifest),
        "dataset_download_manifest_sha256": _sha256(download_manifest),
        "dataset_archives": download.get("archives"),
        "dataset_license": download.get("license"),
        "model_implementation_license": first_metadata.get("model_implementation_license"),
        "config": first_metadata.get("config"),
        "config_sha256": first_metadata.get("config_sha256"),
        "ground_truth": {
            "strict": strict_ground_truth,
            "provenance": prepare_summary.get("label_provenance") or {},
            "annotation_counts": prepare_summary.get("annotation_counts") or {},
        },
        "rows": rows,
        "seed_count": len(rows),
        "comparison": comparison,
        "decision": "hold",
        "efficacy": "not_established",
        "promotion_eligible": False,
        "qualification_reasons": qualification_reasons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
