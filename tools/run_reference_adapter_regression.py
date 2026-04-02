#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.adapter import RTDETRPoseAdapter
from yolozu.core.cli_args import (
    require_float_in_range,
    require_non_negative_int,
    require_positive_int,
    resolve_input_path,
    resolve_output_path,
)
from yolozu.dataset import build_manifest
from yolozu.predictions import validate_predictions_entries

DEFAULT_BASELINE = "baselines/reference_adapter/rtdetr_pose_smoke_val.json"
DEFAULT_DATASET = "data/smoke"
DEFAULT_SPLIT = "val"
DEFAULT_BASELINE_ROOT = "baselines/reference_adapter"
DEFAULT_PROFILE = "micro"

GATE_SCHEMA = "schema_drift"
GATE_CONSISTENCY = "consistency_drift"
GATE_METRIC = "metric_drift"
GATE_SPEED = "speed_drift"
RUNTIME_LOCK_KEYS = ("torch", "onnxruntime")
ERR_IO = "E_IO"
ERR_DECODE = "E_DECODE"
ERR_PREPROC = "E_PREPROC"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run RT-DETR reference-adapter regression on a fixed real-image dataset with "
            "separated contract/behavior gates and reproducibility metadata."
        )
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="YOLO-format dataset root.")
    p.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split (default: val).")
    p.add_argument("--max-images", type=int, default=2, help="Max images to evaluate (default: 2).")
    p.add_argument(
        "--profile",
        choices=("micro", "full", "custom"),
        default=DEFAULT_PROFILE,
        help="Regression profile label used for matrix baselines/reporting (default: micro).",
    )
    p.add_argument("--baseline", default=DEFAULT_BASELINE, help="Baseline JSON path.")
    p.add_argument(
        "--baseline-layout",
        choices=("flat", "matrix"),
        default="flat",
        help="Baseline path layout: flat (use --baseline) or matrix (derive from backend/device/version).",
    )
    p.add_argument(
        "--baseline-root",
        default=DEFAULT_BASELINE_ROOT,
        help="Root directory for matrix baseline layout (default: baselines/reference_adapter).",
    )
    p.add_argument(
        "--adapter-id",
        default="rtdetr_pose",
        help="Adapter id for matrix baseline layout (default: rtdetr_pose).",
    )
    p.add_argument(
        "--backend-id",
        default="torch",
        help="Backend id for matrix baseline layout (default: torch).",
    )
    p.add_argument(
        "--matrix-device",
        default=None,
        help="Optional device id override for matrix baseline layout (default: --device).",
    )
    p.add_argument(
        "--baseline-version",
        default="v1",
        help="Baseline version segment used by matrix baseline layout (default: v1).",
    )
    p.add_argument(
        "--output",
        default="reports/reference_adapter_regression.json",
        help="Regression report output JSON path.",
    )
    p.add_argument(
        "--diff-summary-out",
        default=None,
        help="Optional diff_summary.json output path (default: <output stem>.diff_summary.json).",
    )
    p.add_argument(
        "--topk-examples-dir",
        default=None,
        help="Optional top-k overlay directory (default: <output stem>_topk_examples).",
    )
    p.add_argument(
        "--topk-examples",
        type=int,
        default=3,
        help="Number of counterexample overlays to emit when regression fails (default: 3, 0 disables).",
    )
    p.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write/update baseline JSON from current run instead of comparing.",
    )
    p.add_argument(
        "--runtime-lock",
        default="requirements-locks/requirements-ci.lock",
        help="Pinned runtime lock file used for CI/runtime reproducibility checks.",
    )
    p.add_argument(
        "--enforce-runtime-lock",
        action="store_true",
        help="Fail if run-time torch/onnxruntime versions differ from --runtime-lock pins.",
    )
    p.add_argument(
        "--enforce-weights-hash",
        action="store_true",
        help="Fail consistency gate if baseline/current weights_hash differ (even checkpoint-free).",
    )
    p.add_argument(
        "--expected-dataset-hash",
        default=None,
        help="Optional expected dataset SHA256. Mismatch fails consistency gate.",
    )
    p.add_argument(
        "--expected-weights-hash",
        default=None,
        help="Optional expected weights SHA256. Mismatch fails consistency gate.",
    )
    p.add_argument(
        "--expected-checkpoint-hash",
        default=None,
        help="Optional expected checkpoint SHA256. Mismatch fails consistency gate.",
    )
    p.add_argument(
        "--capture-provenance",
        choices=("full", "minimal", "off"),
        default="full",
        help="Capture SBOM/environment provenance snapshot in baseline_meta (default: full).",
    )

    p.add_argument("--config", default="rtdetr_pose/configs/base.json", help="RT-DETR config path.")
    p.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    p.add_argument("--device", default="cpu", help="Device for adapter inference (default: cpu).")
    p.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=[160],
        help="Image size for adapter (one value or two values).",
    )
    p.add_argument("--score-threshold", type=float, default=0.05, help="Adapter score threshold.")
    p.add_argument("--max-detections", type=int, default=20, help="Max detections per image.")
    p.add_argument(
        "--init-seed",
        type=int,
        default=2026,
        help="Deterministic model-init seed for reference baseline (default: 2026).",
    )
    p.add_argument(
        "--repro-policy",
        choices=("strict", "relaxed", "off"),
        default="relaxed",
        help="Reproducibility policy: strict (deterministic), relaxed (seed-only), off (speed).",
    )

    p.add_argument(
        "--schema-gate-mode",
        choices=("hard", "off"),
        default="hard",
        help="Schema gate mode (default: hard).",
    )
    p.add_argument(
        "--consistency-gate-mode",
        choices=("hard", "off"),
        default="hard",
        help="Consistency gate mode (default: hard).",
    )
    p.add_argument(
        "--score-gate-mode",
        choices=("warn", "hard", "off"),
        default="warn",
        help="Behavior score gate mode (default: warn).",
    )
    p.add_argument(
        "--perf-gate-mode",
        choices=("warn", "hard", "off"),
        default="warn",
        help="Behavior performance gate mode (default: warn).",
    )

    p.add_argument(
        "--canonical-decimals",
        type=int,
        default=6,
        help="Decimal places used by canonicalization (default: 6).",
    )

    p.add_argument(
        "--metric-total-detections-abs",
        type=float,
        default=0.0,
        help="Allowed absolute drift for total detections.",
    )
    p.add_argument(
        "--metric-score-sum-abs",
        type=float,
        default=0.01,
        help="Allowed absolute drift for score_sum.",
    )
    p.add_argument(
        "--metric-score-mean-abs",
        type=float,
        default=0.001,
        help="Allowed absolute drift for score_mean.",
    )
    p.add_argument(
        "--metric-bbox-checksum-abs",
        type=float,
        default=0.1,
        help="Allowed absolute drift for bbox_checksum.",
    )
    p.add_argument(
        "--metric-map50-abs",
        type=float,
        default=0.03,
        help="Allowed absolute drift for robust metric map50.",
    )
    p.add_argument(
        "--metric-map50-95-abs",
        type=float,
        default=0.03,
        help="Allowed absolute drift for robust metric map50_95.",
    )
    p.add_argument(
        "--metric-worst-k-map50-abs",
        type=float,
        default=0.05,
        help="Allowed absolute drift for robust metric worst_k_map50.",
    )
    p.add_argument(
        "--metric-median-class-map50-abs",
        type=float,
        default=0.05,
        help="Allowed absolute drift for robust metric median_class_map50.",
    )
    p.add_argument(
        "--metric-recall-at-k-abs",
        type=float,
        default=0.05,
        help="Allowed absolute drift for robust metric recall_at_k.",
    )
    p.add_argument(
        "--metric-iou-p10-abs",
        type=float,
        default=0.1,
        help="Allowed absolute drift for robust metric iou_p10.",
    )
    p.add_argument(
        "--metric-iou-p50-abs",
        type=float,
        default=0.1,
        help="Allowed absolute drift for robust metric iou_p50.",
    )
    p.add_argument(
        "--metric-missing-count-abs",
        type=float,
        default=2.0,
        help="Allowed absolute drift for robust metric missing_count.",
    )
    p.add_argument(
        "--metric-extra-count-abs",
        type=float,
        default=2.0,
        help="Allowed absolute drift for robust metric extra_count.",
    )
    p.add_argument(
        "--metric-class-mismatch-abs",
        type=float,
        default=2.0,
        help="Allowed absolute drift for robust metric class_mismatch_count.",
    )
    p.add_argument(
        "--metric-worst-k",
        type=int,
        default=3,
        help="Worst-k class mAP aggregation size for robust metrics (default: 3).",
    )
    p.add_argument(
        "--metric-recall-k",
        type=int,
        default=20,
        help="K for recall@K robust metric (default: 20).",
    )
    p.add_argument(
        "--peer-report",
        default=None,
        help="Optional peer backend regression report for backend parity drift checks.",
    )
    p.add_argument(
        "--backend-parity-mode",
        choices=("off", "warn", "hard"),
        default="off",
        help="Backend parity drift policy against --peer-report (default: off).",
    )
    p.add_argument(
        "--backend-parity-map50-abs",
        type=float,
        default=0.03,
        help="Allowed absolute map50 delta vs peer backend report.",
    )
    p.add_argument(
        "--backend-parity-map50-95-abs",
        type=float,
        default=0.03,
        help="Allowed absolute map50_95 delta vs peer backend report.",
    )
    p.add_argument(
        "--min-fps-ratio",
        type=float,
        default=0.25,
        help="Perf gate lower bound as ratio against baseline fps (default: 0.25).",
    )
    p.add_argument(
        "--absolute-floor-fps",
        type=float,
        default=0.2,
        help="Absolute minimum fps floor for perf gate (default: 0.2).",
    )
    return p.parse_args(argv)



from yolozu.inference.reference_regression_gates import (
    _append_gate_failure,
    _build_baseline_payload,
    _collect_run_meta,
    _compare_against_baseline,
    _configure_repro_policy,
    _new_gate,
)
from yolozu.inference.reference_regression_io import (
    _build_contract,
    _build_diff_summary_payload,
    _build_robust_metrics,
    _build_summary,
    _canonicalize_predictions,
    _collect_provenance,
    _dataset_fingerprint,
    _default_diff_summary_path,
    _default_topk_examples_dir,
    _ensure_repo_write_target,
    _gate_policy_from_args,
    _image_size,
    _now_utc,
    _parse_runtime_lock,
    _preflight_records,
    _protocol_spec,
    _repo_relative_display,
    _resolve_baseline_path,
    _sha256_file,
    _sha256_json,
    _thresholds_from_args,
    _validate_reference_entry_metadata,
    _write_topk_examples,
)

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        max_images = require_non_negative_int(args.max_images, flag_name="--max-images")
        max_detections = require_positive_int(args.max_detections, flag_name="--max-detections")
        score_threshold = require_float_in_range(
            args.score_threshold,
            flag_name="--score-threshold",
            minimum=0.0,
            maximum=1.0,
        )
        init_seed = require_non_negative_int(args.init_seed, flag_name="--init-seed")
        canonical_decimals = require_non_negative_int(
            args.canonical_decimals,
            flag_name="--canonical-decimals",
        )
        metric_worst_k = require_positive_int(args.metric_worst_k, flag_name="--metric-worst-k")
        metric_recall_k = require_positive_int(args.metric_recall_k, flag_name="--metric-recall-k")
        topk_examples = require_non_negative_int(args.topk_examples, flag_name="--topk-examples")
        require_float_in_range(
            args.min_fps_ratio,
            flag_name="--min-fps-ratio",
            minimum=0.0,
            maximum=1.0,
        )
        require_float_in_range(
            args.absolute_floor_fps,
            flag_name="--absolute-floor-fps",
            minimum=0.0,
            maximum=100000.0,
        )
        for flag, value in (
            ("--metric-total-detections-abs", args.metric_total_detections_abs),
            ("--metric-score-sum-abs", args.metric_score_sum_abs),
            ("--metric-score-mean-abs", args.metric_score_mean_abs),
            ("--metric-bbox-checksum-abs", args.metric_bbox_checksum_abs),
            ("--metric-map50-abs", args.metric_map50_abs),
            ("--metric-map50-95-abs", args.metric_map50_95_abs),
            ("--metric-worst-k-map50-abs", args.metric_worst_k_map50_abs),
            ("--metric-median-class-map50-abs", args.metric_median_class_map50_abs),
            ("--metric-recall-at-k-abs", args.metric_recall_at_k_abs),
            ("--metric-iou-p10-abs", args.metric_iou_p10_abs),
            ("--metric-iou-p50-abs", args.metric_iou_p50_abs),
            ("--metric-missing-count-abs", args.metric_missing_count_abs),
            ("--metric-extra-count-abs", args.metric_extra_count_abs),
            ("--metric-class-mismatch-abs", args.metric_class_mismatch_abs),
            ("--backend-parity-map50-abs", args.backend_parity_map50_abs),
            ("--backend-parity-map50-95-abs", args.backend_parity_map50_95_abs),
        ):
            require_float_in_range(value, flag_name=flag, minimum=0.0, maximum=100000.0)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    cwd = Path.cwd()
    dataset_root = resolve_input_path(args.dataset, cwd=cwd, repo_root=repo_root)
    config_path = resolve_input_path(args.config, cwd=cwd, repo_root=repo_root)
    checkpoint_path = (
        resolve_input_path(args.checkpoint, cwd=cwd, repo_root=repo_root)
        if args.checkpoint
        else None
    )
    baseline_path = _resolve_baseline_path(args=args, cwd=cwd)
    output_path = resolve_output_path(args.output, cwd=cwd)
    diff_summary_path = (
        resolve_output_path(args.diff_summary_out, cwd=cwd)
        if args.diff_summary_out
        else _default_diff_summary_path(output_path)
    )
    topk_examples_dir = (
        resolve_output_path(args.topk_examples_dir, cwd=cwd)
        if args.topk_examples_dir
        else _default_topk_examples_dir(output_path)
    )
    runtime_lock_path = resolve_input_path(args.runtime_lock, cwd=cwd, repo_root=repo_root)
    peer_report_path = (
        resolve_input_path(args.peer_report, cwd=cwd, repo_root=repo_root)
        if args.peer_report
        else None
    )
    _ensure_repo_write_target(baseline_path, flag_name="--baseline")
    _ensure_repo_write_target(output_path, flag_name="--output")
    _ensure_repo_write_target(diff_summary_path, flag_name="--diff-summary-out")
    _ensure_repo_write_target(topk_examples_dir, flag_name="--topk-examples-dir")
    if args.enforce_runtime_lock and not runtime_lock_path.exists():
        raise SystemExit(f"--runtime-lock not found: {runtime_lock_path}")
    if peer_report_path is not None and not peer_report_path.exists():
        raise SystemExit(f"--peer-report not found: {peer_report_path}")

    runtime_lock_meta = {
        "path": _repo_relative_display(runtime_lock_path),
        "sha256": (_sha256_file(runtime_lock_path) if runtime_lock_path.exists() else None),
        "versions": _parse_runtime_lock(runtime_lock_path),
    }

    image_size = _image_size(list(args.image_size))
    manifest = build_manifest(dataset_root, split=args.split)
    records = list(manifest.get("images") or [])
    if max_images is not None:
        records = records[: int(max_images)]
    if not records:
        raise SystemExit("no records to evaluate; check --dataset/--split/--max-images")

    record_preflight, preflight_errors = _preflight_records(
        records,
        dataset_root=dataset_root,
        image_size=image_size,
    )
    if preflight_errors:
        raise SystemExit("record preflight failed:\n- " + "\n- ".join(preflight_errors))

    repro_details = _configure_repro_policy(
        policy=str(args.repro_policy),
        seed=(None if str(args.repro_policy) == "off" else int(init_seed)),
    )

    adapter = RTDETRPoseAdapter(
        config_path=str(config_path),
        checkpoint_path=(str(checkpoint_path) if checkpoint_path is not None else None),
        device=str(args.device),
        image_size=image_size,
        score_threshold=float(score_threshold),
        max_detections=int(max_detections),
        init_seed=(None if str(args.repro_policy) == "off" else int(init_seed)),
        repro_policy=str(args.repro_policy),
    )

    started_utc = _now_utc()
    start = time.perf_counter()
    try:
        predictions_raw = adapter.predict(records)
    except Exception as exc:
        raise SystemExit(f"{ERR_PREPROC}: adapter.predict failed: {exc}") from exc
    elapsed = time.perf_counter() - start
    finished_utc = _now_utc()

    predictions, canonicalization = _canonicalize_predictions(
        predictions_raw,
        dataset_root=dataset_root,
        decimals=int(canonical_decimals),
    )

    schema_errors = list(canonicalization.get("schema_errors") or [])
    schema_warnings = list(canonicalization.get("warnings") or [])
    try:
        validate_result = validate_predictions_entries(predictions, strict=True)
        schema_warnings.extend(list(validate_result.warnings))
    except ValueError as exc:
        schema_errors.append(str(exc))

    summary = _build_summary(predictions)
    robust_metrics = _build_robust_metrics(
        records=records,
        predictions=predictions,
        dataset_root=dataset_root,
        worst_k=int(metric_worst_k),
        recall_k=int(metric_recall_k),
    )
    contract, contract_errors = _build_contract(
        records,
        predictions,
        dataset_root=dataset_root,
    )

    consistency_errors = list(canonicalization.get("consistency_errors") or [])
    consistency_errors.extend(
        _validate_reference_entry_metadata(
            predictions,
            record_preflight=record_preflight,
        )
    )

    speed = {
        "images": int(len(records)),
        "seconds": float(round(elapsed, 6)),
        "fps": float(round((len(records) / elapsed) if elapsed > 0 else 0.0, 6)),
    }
    predictions_sha256 = _sha256_json(predictions)

    dataset_fingerprint = _dataset_fingerprint(
        records,
        dataset_root=dataset_root,
        split=str(manifest.get("split")),
        max_images=int(len(records)),
    )
    provenance = _collect_provenance(
        capture_mode=str(args.capture_provenance),
        runtime_lock=runtime_lock_meta,
    )

    run_meta = _collect_run_meta(
        adapter=adapter,
        args=args,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        dataset_fingerprint=dataset_fingerprint,
        repro_details=repro_details,
        canonicalization=canonicalization,
        runtime_lock=runtime_lock_meta,
        provenance=provenance,
    )
    run_meta["record_io_boundary"] = {
        "checked_records": int(len(records)),
        "record_preflight_count": int(len(record_preflight)),
        "error_codes": [ERR_IO, ERR_DECODE, ERR_PREPROC],
        "input_requirements": {
            "image_exists": True,
            "decode_success": True,
            "exif_orientation_normalized": True,
            "color_order": "RGB",
            "dtype": "float32",
            "model_input_size": [int(image_size[0]), int(image_size[1])],
        },
    }
    if bool(args.enforce_runtime_lock):
        pinned_versions = dict(runtime_lock_meta.get("versions") or {})
        for key in RUNTIME_LOCK_KEYS:
            expected = pinned_versions.get(key)
            if expected is None:
                consistency_errors.append(f"runtime lock missing pin for package '{key}'")
                continue
            actual = (run_meta.get("versions") or {}).get(key)
            if actual is None:
                consistency_errors.append(f"runtime package '{key}' not installed (expected {expected})")
                continue
            if str(actual) != str(expected):
                consistency_errors.append(f"runtime lock mismatch for {key}: expected={expected} actual={actual}")
    if args.expected_dataset_hash:
        expected_dataset_hash = str(args.expected_dataset_hash).strip()
        current_dataset_hash = str(run_meta.get("dataset_hash") or "")
        if expected_dataset_hash != current_dataset_hash:
            consistency_errors.append(
                f"dataset_hash mismatch: expected={expected_dataset_hash} current={current_dataset_hash}"
            )
    if args.expected_weights_hash:
        expected_weights_hash = str(args.expected_weights_hash).strip()
        current_weights_hash = str(run_meta.get("weights_hash") or "")
        if expected_weights_hash != current_weights_hash:
            consistency_errors.append(
                f"weights_hash mismatch: expected={expected_weights_hash} current={current_weights_hash}"
            )
    if args.expected_checkpoint_hash:
        expected_checkpoint_hash = str(args.expected_checkpoint_hash).strip()
        current_checkpoint_hash = str(run_meta.get("checkpoint_hash") or "")
        if expected_checkpoint_hash != current_checkpoint_hash:
            consistency_errors.append(
                f"checkpoint_hash mismatch: expected={expected_checkpoint_hash} current={current_checkpoint_hash}"
            )

    run_context = {
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "dataset": _repo_relative_display(dataset_root),
        "split": str(manifest.get("split")),
        "adapter": str(args.adapter_id),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "device": str(args.device),
        "profile": str(args.profile),
        "baseline_layout": str(args.baseline_layout),
        "baseline_path": str(baseline_path),
        "peer_report": (str(peer_report_path) if peer_report_path is not None else None),
        "diff_summary_path": str(diff_summary_path),
        "topk_examples_dir": str(topk_examples_dir),
        "topk_examples": int(topk_examples),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "score_threshold": float(score_threshold),
        "max_detections": int(max_detections),
        "init_seed": (None if str(args.repro_policy) == "off" else int(init_seed)),
        "repro_policy": str(args.repro_policy),
        "records": int(len(records)),
    }

    gate_policy = _gate_policy_from_args(args)
    protocol = _protocol_spec(canonical_decimals=int(canonical_decimals))

    hard_failures: list[str] = []
    soft_failures: list[str] = []
    failure_records: list[dict[str, Any]] = []
    gates: dict[str, Any]
    baseline_payload: dict[str, Any]
    peer_robust_metrics: dict[str, Any] | None = None
    if peer_report_path is not None:
        peer_payload = json.loads(peer_report_path.read_text(encoding="utf-8"))
        peer_robust_metrics = (
            peer_payload.get("robust_metrics")
            or ((peer_payload.get("baseline") or {}).get("robust_metrics"))
            or ((peer_payload.get("summary") or {}).get("robust_metrics"))
        )

    if args.write_baseline:
        baseline_payload = _build_baseline_payload(
            args=args,
            baseline_path=baseline_path,
            dataset_root=dataset_root,
            split=str(manifest.get("split")),
            summary=summary,
            robust_metrics=robust_metrics,
            speed=speed,
            contract=contract,
            predictions_sha256=predictions_sha256,
            run_meta=run_meta,
            protocol=protocol,
            gate_policy=gate_policy,
            canonicalization=canonicalization,
        )
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(baseline_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        gates = {
            GATE_SCHEMA: _new_gate(mode=gate_policy[GATE_SCHEMA], category="contract"),
            GATE_CONSISTENCY: _new_gate(mode=gate_policy[GATE_CONSISTENCY], category="contract"),
            GATE_METRIC: _new_gate(mode="off", category="behavior"),
            GATE_SPEED: _new_gate(mode="off", category="behavior"),
        }

        if gate_policy[GATE_SCHEMA] != "off":
            gates[GATE_SCHEMA]["details"] = {
                "warnings": schema_warnings,
                "errors": schema_errors,
                "mode": "baseline_write",
            }
            for msg in schema_warnings:
                gates[GATE_SCHEMA]["ok"] = False
                _append_gate_failure(
                    gate_key=GATE_SCHEMA,
                    message=f"schema warning: {msg}",
                    gate_policy=gate_policy,
                    hard_failures=hard_failures,
                    soft_failures=soft_failures,
                    failure_records=failure_records,
                )
            for msg in schema_errors:
                gates[GATE_SCHEMA]["ok"] = False
                _append_gate_failure(
                    gate_key=GATE_SCHEMA,
                    message=f"schema error: {msg}",
                    gate_policy=gate_policy,
                    hard_failures=hard_failures,
                    soft_failures=soft_failures,
                    failure_records=failure_records,
                )
        else:
            gates[GATE_SCHEMA]["details"] = {"skipped": True, "mode": "baseline_write"}

        baseline_write_consistency = [*contract_errors, *consistency_errors]
        if gate_policy[GATE_CONSISTENCY] != "off":
            gates[GATE_CONSISTENCY]["details"] = {
                "errors": baseline_write_consistency,
                "mode": "baseline_write",
            }
            if baseline_write_consistency:
                gates[GATE_CONSISTENCY]["ok"] = False
                for msg in baseline_write_consistency:
                    _append_gate_failure(
                        gate_key=GATE_CONSISTENCY,
                        message=msg,
                        gate_policy=gate_policy,
                        hard_failures=hard_failures,
                        soft_failures=soft_failures,
                        failure_records=failure_records,
                    )
        else:
            gates[GATE_CONSISTENCY]["details"] = {"skipped": True, "mode": "baseline_write"}

        gates[GATE_METRIC]["details"] = {"mode": "baseline_write", "skipped": True}
        gates[GATE_SPEED]["details"] = {"mode": "baseline_write", "skipped": True}
    else:
        if not baseline_path.exists():
            raise SystemExit(
                f"baseline not found: {baseline_path} (run with --write-baseline to create it)"
            )
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        gates, hard_failures, soft_failures = _compare_against_baseline(
            baseline_payload=baseline_payload,
            summary=summary,
            robust_metrics=robust_metrics,
            speed=speed,
            contract=contract,
            run_meta=run_meta,
            schema_warnings=schema_warnings,
            schema_errors=schema_errors,
            consistency_errors=consistency_errors,
            contract_errors=contract_errors,
            gate_policy=gate_policy,
            predictions=predictions,
            enforce_runtime_lock=bool(args.enforce_runtime_lock),
            enforce_weights_hash=bool(args.enforce_weights_hash),
            peer_robust_metrics=peer_robust_metrics,
            backend_parity=(_thresholds_from_args(args).get("backend_parity") or {}),
            failure_records=failure_records,
        )

    report = {
        "schema_version": 2,
        "ok": len(hard_failures) == 0,
        "run": run_context,
        "run_meta": run_meta,
        "baseline_meta": (baseline_payload.get("baseline_meta") if not args.write_baseline else run_meta),
        "protocol": protocol,
        "gate_policy": gate_policy,
        "summary": summary,
        "robust_metrics": robust_metrics,
        "thresholds": _thresholds_from_args(args),
        "speed": speed,
        "contract": contract,
        "canonicalization": canonicalization,
        "predictions_sha256": predictions_sha256,
        "gates": gates,
        "failures": hard_failures,
        "hard_failures": hard_failures,
        "soft_failures": soft_failures,
        "failure_records": failure_records,
        "warnings": soft_failures,
        "baseline_path": str(baseline_path),
        "peer_report_path": (str(peer_report_path) if peer_report_path is not None else None),
    }

    if (not args.write_baseline) and hard_failures:
        diff_summary_payload = _build_diff_summary_payload(report)
        topk_written: list[str] = []
        try:
            topk_written = _write_topk_examples(
                out_dir=topk_examples_dir,
                report=report,
                predictions=predictions,
                records=records,
                dataset_root=dataset_root,
                topk=int(topk_examples),
            )
        except Exception as exc:
            diff_summary_payload["topk_examples_error"] = str(exc)
        if topk_written:
            diff_summary_payload["topk_examples"] = topk_written
            report["topk_examples_dir"] = _repo_relative_display(topk_examples_dir)
            report["topk_examples"] = topk_written
        diff_summary_path.parent.mkdir(parents=True, exist_ok=True)
        diff_summary_path.write_text(json.dumps(diff_summary_payload, indent=2, sort_keys=True), encoding="utf-8")
        report["diff_summary_path"] = _repo_relative_display(diff_summary_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path)

    if soft_failures:
        print("reference adapter regression soft warnings:", file=sys.stderr)
        for item in soft_failures:
            print(f"- {item}", file=sys.stderr)

    if hard_failures:
        raise SystemExit("reference adapter regression failed:\n- " + "\n- ".join(hard_failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
