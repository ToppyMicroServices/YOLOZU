from __future__ import annotations

import argparse
import hashlib
import os
import platform
from pathlib import Path
from typing import Any

from yolozu.adapter import RTDETRPoseAdapter

from .reference_regression_io import (
    ERR_DECODE,
    ERR_IO,
    ERR_PREPROC,
    GATE_CONSISTENCY,
    GATE_METRIC,
    GATE_SCHEMA,
    GATE_SPEED,
    _git_sha,
    _image_size,
    _now_utc,
    _repo_relative_display,
    _safe_package_version,
    _sha256_file,
    _thresholds_from_args,
)
def _configure_repro_policy(*, policy: str, seed: int | None) -> dict[str, Any]:
    details: dict[str, Any] = {
        "policy": str(policy),
        "seed": (int(seed) if seed is not None else None),
        "actions": [],
        "determinism_knobs": {
            "image_decode_library": "Pillow",
            "exif_orientation": "normalized",
            "color_order": "RGB",
            "preprocess_dtype": "float32",
            "resize_algorithm": "bilinear",
            "input_resolution_policy": "fixed_resize",
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
    }

    try:
        import torch
    except ImportError:
        details["actions"].append("torch_unavailable")
        return details

    if policy in ("strict", "relaxed") and seed is not None:
        torch.manual_seed(int(seed))
        details["actions"].append("torch.manual_seed")
        if policy == "strict":
            os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
            details["actions"].append("env:PYTHONHASHSEED")
        if bool(getattr(torch.cuda, "is_available", lambda: False)()):
            try:
                torch.cuda.manual_seed_all(int(seed))
                details["actions"].append("torch.cuda.manual_seed_all")
            except (AttributeError, RuntimeError):
                details["actions"].append("torch.cuda.manual_seed_all_failed")

    if policy == "strict":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        details["actions"].append("env:CUBLAS_WORKSPACE_CONFIG")
        try:
            torch.use_deterministic_algorithms(True)
            details["actions"].append("torch.use_deterministic_algorithms(True)")
        except (AttributeError, RuntimeError):
            details["actions"].append("torch.use_deterministic_algorithms_failed")
        if hasattr(torch.backends, "cudnn"):
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                details["actions"].append("torch.backends.cudnn.deterministic=True")
                details["actions"].append("torch.backends.cudnn.benchmark=False")
            except (AttributeError, RuntimeError):
                details["actions"].append("cudnn_flags_failed")

    if policy == "off":
        details["actions"].append("repro_disabled")

    try:
        details["deterministic_algorithms_enabled"] = bool(torch.are_deterministic_algorithms_enabled())
    except (AttributeError, RuntimeError):
        details["deterministic_algorithms_enabled"] = None
    details["determinism_knobs"]["pythonhashseed"] = os.environ.get("PYTHONHASHSEED")
    details["determinism_knobs"]["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    return details


def _hash_model_state_dict(model: Any, torch_module: Any) -> str | None:
    try:
        state = model.state_dict()
    except (AttributeError, RuntimeError):
        return None

    digest = hashlib.sha256()
    for key in sorted(state.keys()):
        tensor = state[key]
        if not hasattr(tensor, "detach"):
            continue
        t = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(getattr(t, "shape", ()))).encode("utf-8"))
        digest.update(str(getattr(t, "dtype", "unknown")).encode("utf-8"))
        try:
            digest.update(t.numpy().tobytes(order="C"))
        except (RuntimeError, TypeError, ValueError):
            try:
                digest.update(bytes(t.view(torch_module.uint8).tolist()))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return None
    return digest.hexdigest()


def _collect_run_meta(
    *,
    adapter: RTDETRPoseAdapter,
    args: argparse.Namespace,
    config_path: Path,
    checkpoint_path: Path | None,
    dataset_fingerprint: dict[str, Any],
    repro_details: dict[str, Any],
    canonicalization: dict[str, Any],
    runtime_lock: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    model_dtype: str | None = None
    model_state_hash: str | None = None
    model_hash_source: str | None = None
    backend = str(args.backend_id)

    try:
        model = adapter.get_model()
        torch_module = (adapter._backend or {}).get("torch") if hasattr(adapter, "_backend") else None
        if model is not None:
            try:
                param = next(model.parameters())
                model_dtype = str(getattr(param, "dtype", None))
            except (AttributeError, StopIteration, TypeError):
                model_dtype = None
        if model is not None and torch_module is not None:
            model_state_hash = _hash_model_state_dict(model, torch_module)
            model_hash_source = "model_state_dict"
    except (AttributeError, RuntimeError, TypeError):
        model_dtype = None

    checkpoint_hash = _sha256_file(checkpoint_path) if checkpoint_path is not None and checkpoint_path.exists() else None
    config_hash = _sha256_file(config_path) if config_path.exists() else None

    weights_hash = checkpoint_hash if checkpoint_hash else model_state_hash
    weights_source = "checkpoint" if checkpoint_hash else model_hash_source

    versions = {
        "python": platform.python_version(),
        "torch": _safe_package_version("torch"),
        "onnxruntime": _safe_package_version("onnxruntime"),
        "ultralytics": _safe_package_version("ultralytics"),
        "numpy": _safe_package_version("numpy"),
        "pillow": _safe_package_version("Pillow"),
        "yolozu": _safe_package_version("yolozu"),
    }

    return {
        "generated_utc": _now_utc(),
        "git_sha": _git_sha(),
        "versions": versions,
        "backend": backend,
        "device": str(args.device),
        "dtype": model_dtype,
        "repro_policy": str(args.repro_policy),
        "seed": (None if str(args.repro_policy) == "off" else int(args.init_seed)),
        "repro_actions": list(repro_details.get("actions") or []),
        "deterministic_algorithms_enabled": repro_details.get("deterministic_algorithms_enabled"),
        "determinism_knobs": dict(repro_details.get("determinism_knobs") or {}),
        "weights_hash": weights_hash,
        "weights_source": weights_source,
        "checkpoint_hash": checkpoint_hash,
        "config_hash": config_hash,
        "dataset_hash": dataset_fingerprint.get("hash"),
        "dataset_count": dataset_fingerprint.get("count"),
        "dataset_missing": list(dataset_fingerprint.get("missing") or []),
        "canonical_decimals": int(canonicalization.get("canonical_decimals", 6)),
        "bbox_format": str(canonicalization.get("bbox_format", "cxcywh_norm")),
        "runtime_lock_path": runtime_lock.get("path"),
        "runtime_lock_sha256": runtime_lock.get("sha256"),
        "runtime_lock_versions": dict(runtime_lock.get("versions") or {}),
        "profile": str(args.profile),
        "baseline_layout": str(args.baseline_layout),
        "provenance": provenance,
    }


def _build_baseline_payload(
    *,
    args: argparse.Namespace,
    baseline_path: Path,
    dataset_root: Path,
    split: str,
    summary: dict[str, Any],
    robust_metrics: dict[str, Any],
    speed: dict[str, Any],
    contract: dict[str, Any],
    predictions_sha256: str,
    run_meta: dict[str, Any],
    protocol: dict[str, Any],
    gate_policy: dict[str, str],
    canonicalization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "reference_adapter": str(args.adapter_id),
        "generated_utc": _now_utc(),
        "profile": str(args.profile),
        "baseline_layout": str(args.baseline_layout),
        "baseline_path": _repo_relative_display(baseline_path),
        "dataset": {
            "path": _repo_relative_display(dataset_root),
            "split": str(split),
            "max_images": int(args.max_images),
            "hash": run_meta.get("dataset_hash"),
        },
        "adapter": {
            "config": str(args.config),
            "checkpoint": (str(args.checkpoint) if args.checkpoint else None),
            "device": str(args.device),
            "image_size": [int(v) for v in _image_size(list(args.image_size))],
            "score_threshold": float(args.score_threshold),
            "max_detections": int(args.max_detections),
            "init_seed": (None if str(args.repro_policy) == "off" else int(args.init_seed)),
            "repro_policy": str(args.repro_policy),
        },
        "thresholds": _thresholds_from_args(args),
        "gate_policy": gate_policy,
        "protocol": protocol,
        "canonicalization": {
            "canonical_decimals": int(canonicalization.get("canonical_decimals", 6)),
            "bbox_format": str(canonicalization.get("bbox_format", "cxcywh_norm")),
            "stable_sort": list(canonicalization.get("stable_sort") or []),
        },
        "baseline_meta": run_meta,
        "baseline": {
            "summary": summary,
            "robust_metrics": robust_metrics,
            "speed": speed,
            "contract": contract,
            "predictions_sha256": predictions_sha256,
        },
    }


def _new_gate(*, mode: str, category: str) -> dict[str, Any]:
    return {
        "mode": str(mode),
        "category": str(category),
        "ok": True,
        "details": {},
    }


def _failure_code(gate_key: str, message: str) -> str:
    text = str(message).lower()
    if gate_key == GATE_SCHEMA:
        if "unknown keys" in text:
            return "E_SCHEMA_UNKNOWN_KEYS"
        if "missing" in text:
            return "E_SCHEMA_MISSING_FIELD"
        return "E_SCHEMA_VALIDATION"
    if gate_key == GATE_CONSISTENCY:
        if f"{ERR_IO.lower()}:" in text or text.startswith(ERR_IO.lower()):
            return ERR_IO
        if f"{ERR_DECODE.lower()}:" in text or text.startswith(ERR_DECODE.lower()):
            return ERR_DECODE
        if f"{ERR_PREPROC.lower()}:" in text or text.startswith(ERR_PREPROC.lower()):
            return ERR_PREPROC
        if "runtime lock" in text:
            return "E_CANON_RUNTIME_LOCK"
        if "weights_hash mismatch" in text:
            return "E_CANON_WEIGHTS_HASH"
        if "image order mismatch" in text:
            return "E_CANON_IMAGE_ORDER"
        if "duplicate" in text:
            return "E_CANON_DUPLICATE_IMAGE"
        return "E_CANON_CONSISTENCY"
    if gate_key == GATE_METRIC:
        if "parity" in text:
            return "E_SCORE_PARITY"
        return "E_SCORE_DRIFT"
    if gate_key == GATE_SPEED:
        return "E_PERF_DRIFT"
    return "E_UNKNOWN"


def _append_gate_failure(
    *,
    gate_key: str,
    message: str,
    gate_policy: dict[str, str],
    hard_failures: list[str],
    soft_failures: list[str],
    failure_records: list[dict[str, Any]],
    mode_override: str | None = None,
) -> None:
    mode = str(mode_override or gate_policy.get(gate_key, "hard"))
    code = _failure_code(gate_key, message)
    line = f"[{gate_key}][{code}] {message}"
    if mode == "hard":
        hard_failures.append(line)
    elif mode == "warn":
        soft_failures.append(line)
    failure_records.append(
        {
            "gate": gate_key,
            "mode": mode,
            "code": code,
            "message": str(message),
        }
    )


def _compare_against_baseline(
    *,
    baseline_payload: dict[str, Any],
    summary: dict[str, Any],
    robust_metrics: dict[str, Any],
    speed: dict[str, Any],
    contract: dict[str, Any],
    run_meta: dict[str, Any],
    schema_warnings: list[str],
    schema_errors: list[str],
    consistency_errors: list[str],
    contract_errors: list[str],
    gate_policy: dict[str, str],
    predictions: list[dict[str, Any]],
    enforce_runtime_lock: bool,
    enforce_weights_hash: bool,
    peer_robust_metrics: dict[str, Any] | None,
    backend_parity: dict[str, Any],
    failure_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str], list[str]]:
    hard_failures: list[str] = []
    soft_failures: list[str] = []

    gates: dict[str, Any] = {
        GATE_SCHEMA: _new_gate(mode=gate_policy[GATE_SCHEMA], category="contract"),
        GATE_CONSISTENCY: _new_gate(mode=gate_policy[GATE_CONSISTENCY], category="contract"),
        GATE_METRIC: _new_gate(mode=gate_policy[GATE_METRIC], category="behavior"),
        GATE_SPEED: _new_gate(mode=gate_policy[GATE_SPEED], category="behavior"),
    }

    baseline = baseline_payload.get("baseline") or {}
    thresholds = baseline_payload.get("thresholds") or {}
    baseline_summary = baseline.get("summary") or {}
    baseline_robust = baseline.get("robust_metrics") or {}
    baseline_speed = baseline.get("speed") or {}
    baseline_contract = baseline.get("contract") or {}
    baseline_meta = baseline_payload.get("baseline_meta") or {}
    current_backend = str(run_meta.get("backend") or baseline_meta.get("backend") or "unknown")
    metric_by_backend = thresholds.get("metric_by_backend") or {}
    metric_thr = metric_by_backend.get(current_backend) or thresholds.get("metric") or {}
    parity_by_backend = thresholds.get("backend_parity_by_backend") or {}
    backend_parity_cfg = parity_by_backend.get(current_backend) or thresholds.get("backend_parity") or backend_parity or {}
    speed_thr = thresholds.get("speed") or {}

    schema_gate = gates[GATE_SCHEMA]
    schema_gate["details"] = {
        "warnings": list(schema_warnings),
        "errors": list(schema_errors),
    }
    if str(schema_gate["mode"]) != "off":
        for msg in schema_warnings:
            schema_gate["ok"] = False
            _append_gate_failure(
                gate_key=GATE_SCHEMA,
                message=f"schema warning: {msg}",
                gate_policy=gate_policy,
                hard_failures=hard_failures,
                soft_failures=soft_failures,
                failure_records=failure_records,
            )
        for msg in schema_errors:
            schema_gate["ok"] = False
            _append_gate_failure(
                gate_key=GATE_SCHEMA,
                message=f"schema error: {msg}",
                gate_policy=gate_policy,
                hard_failures=hard_failures,
                soft_failures=soft_failures,
                failure_records=failure_records,
            )
    else:
        schema_gate["details"]["skipped"] = True

    consistency_gate = gates[GATE_CONSISTENCY]
    consistency_mismatches: list[str] = []
    consistency_warnings: list[str] = []
    if str(consistency_gate["mode"]) != "off":
        for key in (
            "record_images_sha256",
            "prediction_images_sha256",
            "record_images",
            "prediction_images",
        ):
            if key in baseline_contract and baseline_contract.get(key) != contract.get(key):
                consistency_mismatches.append(
                    f"{key} mismatch: baseline={baseline_contract.get(key)} current={contract.get(key)}"
                )

        for key in (
            "dataset_hash",
            "config_hash",
            "checkpoint_hash",
            "repro_policy",
            "canonical_decimals",
            "backend",
        ):
            ref = baseline_meta.get(key)
            cur = run_meta.get(key)
            if ref is None:
                continue
            if ref != cur:
                consistency_mismatches.append(f"{key} mismatch: baseline={ref} current={cur}")

        ref_lock_sha = baseline_meta.get("runtime_lock_sha256")
        cur_lock_sha = run_meta.get("runtime_lock_sha256")
        ref_lock_versions = dict(baseline_meta.get("runtime_lock_versions") or {})
        cur_lock_versions = dict(run_meta.get("runtime_lock_versions") or {})

        if bool(enforce_runtime_lock):
            if not ref_lock_sha:
                consistency_mismatches.append("runtime lock baseline missing runtime_lock_sha256")
            elif not cur_lock_sha:
                consistency_mismatches.append("runtime lock current run missing runtime_lock_sha256")
            elif ref_lock_sha != cur_lock_sha:
                consistency_mismatches.append(
                    f"runtime lock sha mismatch: baseline={ref_lock_sha} current={cur_lock_sha}"
                )

            if not ref_lock_versions:
                consistency_mismatches.append("runtime lock baseline missing runtime_lock_versions")
            elif not cur_lock_versions:
                consistency_mismatches.append("runtime lock current run missing runtime_lock_versions")
            elif ref_lock_versions != cur_lock_versions:
                consistency_mismatches.append(
                    f"runtime lock versions mismatch: baseline={ref_lock_versions} current={cur_lock_versions}"
                )
        else:
            if ref_lock_sha is not None and ref_lock_sha != cur_lock_sha:
                consistency_mismatches.append(
                    f"runtime_lock_sha256 mismatch: baseline={ref_lock_sha} current={cur_lock_sha}"
                )
            if ref_lock_versions and ref_lock_versions != cur_lock_versions:
                consistency_mismatches.append(
                    f"runtime_lock_versions mismatch: baseline={ref_lock_versions} current={cur_lock_versions}"
                )

        ref_weights_hash = baseline_meta.get("weights_hash")
        cur_weights_hash = run_meta.get("weights_hash")
        ref_checkpoint_hash = baseline_meta.get("checkpoint_hash")
        cur_checkpoint_hash = run_meta.get("checkpoint_hash")
        if bool(enforce_weights_hash):
            if not ref_weights_hash:
                consistency_mismatches.append("weights_hash missing in baseline_meta")
            elif not cur_weights_hash:
                consistency_mismatches.append("weights_hash missing in run_meta")
            elif ref_weights_hash != cur_weights_hash:
                consistency_mismatches.append(
                    f"weights_hash mismatch: baseline={ref_weights_hash} current={cur_weights_hash}"
                )
        elif ref_weights_hash is not None:
            # Default mode: checkpoint-backed comparisons are hard, checkpoint-free are warnings.
            if ref_checkpoint_hash is not None and cur_checkpoint_hash is not None:
                if ref_weights_hash != cur_weights_hash:
                    consistency_mismatches.append(
                        f"weights_hash mismatch: baseline={ref_weights_hash} current={cur_weights_hash}"
                    )
            elif ref_weights_hash != cur_weights_hash:
                consistency_warnings.append(
                    "weights_hash differs in checkpoint-free comparison; skipped hard consistency check"
                )

        consistency_mismatches.extend(contract_errors)
        consistency_mismatches.extend(consistency_errors)

        if consistency_mismatches:
            consistency_gate["ok"] = False
            for msg in consistency_mismatches:
                _append_gate_failure(
                    gate_key=GATE_CONSISTENCY,
                    message=msg,
                    gate_policy=gate_policy,
                    hard_failures=hard_failures,
                    soft_failures=soft_failures,
                    failure_records=failure_records,
                )
    else:
        consistency_gate["details"]["skipped"] = True
    if consistency_mismatches:
        first_image = None
        for key in ("prediction_images", "record_images"):
            images = contract.get(key)
            if isinstance(images, list) and images:
                first_image = str(images[0])
                break
        excerpt: list[dict[str, Any]] = []
        if predictions:
            first = predictions[0]
            for det in list(first.get("detections") or [])[:2]:
                excerpt.append(
                    {
                        "class_id": det.get("class_id"),
                        "score": det.get("score"),
                        "bbox": det.get("bbox"),
                    }
                )
            if first_image is None:
                first_image = str(first.get("image"))
        consistency_gate["details"]["first_counterexample"] = {
            "image": first_image,
            "detections_excerpt": excerpt,
            "mismatch": consistency_mismatches[0],
        }
    consistency_gate["details"]["mismatches"] = consistency_mismatches
    consistency_gate["details"]["warnings"] = consistency_warnings

    metric_gate = gates[GATE_METRIC]
    if str(metric_gate["mode"]) == "off":
        metric_gate["details"] = {"skipped": True}
    else:
        metric_checks = {
            "total_detections": float(metric_thr.get("total_detections_abs", 0.0)),
            "score_sum": float(metric_thr.get("score_sum_abs", 0.0)),
            "score_mean": float(metric_thr.get("score_mean_abs", 0.0)),
            "bbox_checksum": float(metric_thr.get("bbox_checksum_abs", 0.0)),
        }
        metric_deltas: dict[str, Any] = {}
        for key, tol in metric_checks.items():
            cur = float(summary.get(key, 0.0))
            ref = float(baseline_summary.get(key, 0.0))
            delta = abs(cur - ref)
            ok = bool(delta <= tol)
            metric_deltas[key] = {
                "baseline": ref,
                "current": cur,
                "abs_delta": delta,
                "allowed_abs": tol,
                "ok": ok,
            }
            if not ok:
                metric_gate["ok"] = False
                _append_gate_failure(
                    gate_key=GATE_METRIC,
                    message=f"{key} abs_delta={delta:.6f} exceeds tolerance={tol:.6f}",
                    gate_policy=gate_policy,
                    hard_failures=hard_failures,
                    soft_failures=soft_failures,
                    failure_records=failure_records,
                )
        robust_checks = {
            "map50": float(metric_thr.get("map50_abs", 0.0)),
            "map50_95": float(metric_thr.get("map50_95_abs", 0.0)),
            "worst_k_map50": float(metric_thr.get("worst_k_map50_abs", 0.0)),
            "median_class_map50": float(metric_thr.get("median_class_map50_abs", 0.0)),
            "recall_at_k": float(metric_thr.get("recall_at_k_abs", 0.0)),
            "iou_p10": float(metric_thr.get("iou_p10_abs", 0.0)),
            "iou_p50": float(metric_thr.get("iou_p50_abs", 0.0)),
            "missing_count": float(metric_thr.get("missing_count_abs", 0.0)),
            "extra_count": float(metric_thr.get("extra_count_abs", 0.0)),
            "class_mismatch_count": float(metric_thr.get("class_mismatch_count_abs", 0.0)),
        }
        robust_deltas: dict[str, Any] = {}
        missing_baseline_keys: list[str] = []
        for key, tol in robust_checks.items():
            if key not in baseline_robust:
                missing_baseline_keys.append(key)
                continue
            cur = float(robust_metrics.get(key, 0.0))
            ref = float(baseline_robust.get(key, 0.0))
            delta = abs(cur - ref)
            ok = bool(delta <= tol)
            robust_deltas[key] = {
                "baseline": ref,
                "current": cur,
                "abs_delta": delta,
                "allowed_abs": tol,
                "ok": ok,
            }
            if not ok:
                metric_gate["ok"] = False
                _append_gate_failure(
                    gate_key=GATE_METRIC,
                    message=f"{key} abs_delta={delta:.6f} exceeds tolerance={tol:.6f}",
                    gate_policy=gate_policy,
                    hard_failures=hard_failures,
                    soft_failures=soft_failures,
                    failure_records=failure_records,
                )

        parity_details: dict[str, Any] = {"mode": str(backend_parity_cfg.get("mode", "off")), "backend": current_backend}
        parity_mode = str(backend_parity_cfg.get("mode", "off"))
        if parity_mode == "off" or not peer_robust_metrics:
            parity_details["skipped"] = True
        else:
            checks = {
                "map50": float(backend_parity_cfg.get("map50_abs", 0.0)),
                "map50_95": float(backend_parity_cfg.get("map50_95_abs", 0.0)),
            }
            rows: dict[str, Any] = {}
            parity_ok = True
            for key, tol in checks.items():
                if key not in peer_robust_metrics:
                    rows[key] = {"missing_peer_metric": True, "allowed_abs": tol}
                    continue
                cur = float(robust_metrics.get(key, 0.0))
                peer = float(peer_robust_metrics.get(key, 0.0))
                delta = abs(cur - peer)
                ok = bool(delta <= tol)
                rows[key] = {
                    "current": cur,
                    "peer": peer,
                    "abs_delta": delta,
                    "allowed_abs": tol,
                    "ok": ok,
                }
                if not ok:
                    parity_ok = False
                    metric_gate["ok"] = False
                    _append_gate_failure(
                        gate_key=GATE_METRIC,
                        message=f"parity {key} abs_delta={delta:.6f} exceeds tolerance={tol:.6f}",
                        gate_policy=gate_policy,
                        hard_failures=hard_failures,
                        soft_failures=soft_failures,
                        failure_records=failure_records,
                        mode_override=parity_mode,
                    )
            parity_details["metrics"] = rows
            parity_details["ok"] = parity_ok

        class_hist_cur = summary.get("class_hist") or {}
        class_hist_ref = baseline_summary.get("class_hist") or {}
        class_delta_rows: list[dict[str, Any]] = []
        for class_id in sorted(set(class_hist_cur.keys()) | set(class_hist_ref.keys()), key=lambda x: int(str(x))):
            cur_v = int(class_hist_cur.get(class_id, 0))
            ref_v = int(class_hist_ref.get(class_id, 0))
            delta = cur_v - ref_v
            if delta != 0:
                class_delta_rows.append(
                    {
                        "class_id": int(class_id),
                        "baseline": ref_v,
                        "current": cur_v,
                        "delta": delta,
                        "abs_delta": abs(delta),
                    }
                )
        class_delta_rows.sort(key=lambda row: int(row["abs_delta"]), reverse=True)

        failing_metrics = [name for name, row in metric_deltas.items() if not bool(row.get("ok"))]
        failing_robust_metrics = [name for name, row in robust_deltas.items() if not bool(row.get("ok"))]
        metric_gate["details"] = {
            "metrics": metric_deltas,
            "robust_metrics": {
                "current": robust_metrics,
                "baseline": baseline_robust,
                "deltas": robust_deltas,
                "missing_baseline_metric_keys": missing_baseline_keys,
                "failed_metric_names": failing_robust_metrics,
            },
            "failed_metric_names": failing_metrics,
            "class_hist_topk": class_delta_rows[:5],
            "backend_parity": parity_details,
        }

    speed_gate = gates[GATE_SPEED]
    if str(speed_gate["mode"]) == "off":
        speed_gate["details"] = {"skipped": True}
    else:
        baseline_fps = float(baseline_speed.get("fps", 0.0))
        current_fps = float(speed.get("fps", 0.0))
        min_ratio = float(speed_thr.get("min_fps_ratio", 0.0))
        floor = float(speed_thr.get("absolute_floor_fps", 0.0))
        required_fps = max(floor, baseline_fps * min_ratio)
        speed_ok = bool(current_fps >= required_fps)
        speed_gate["ok"] = speed_ok
        ratio_vs_baseline = (current_fps / baseline_fps) if baseline_fps > 0 else None
        speed_gate["details"] = {
            "baseline_fps": baseline_fps,
            "current_fps": current_fps,
            "ratio_vs_baseline": ratio_vs_baseline,
            "required_min_fps": required_fps,
            "min_fps_ratio": min_ratio,
            "absolute_floor_fps": floor,
            "measurement": {
                "mode": "single_shot",
                "images": int(speed.get("images", 0)),
                "seconds": float(speed.get("seconds", 0.0)),
                "percentiles_fps": {
                    "p50": current_fps,
                    "p95": current_fps,
                },
            },
            "ok": speed_ok,
        }
        if not speed_ok:
            _append_gate_failure(
                gate_key=GATE_SPEED,
                message=(
                    "current_fps="
                    f"{current_fps:.3f} below required_min_fps={required_fps:.3f}"
                ),
                gate_policy=gate_policy,
                hard_failures=hard_failures,
                soft_failures=soft_failures,
                failure_records=failure_records,
            )

    return gates, hard_failures, soft_failures
