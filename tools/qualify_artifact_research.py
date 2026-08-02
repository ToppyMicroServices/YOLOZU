#!/usr/bin/env python3
"""Build measured, reproducible evidence for artifact-only research lanes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.predictions import canonicalize_predictions, validate_predictions_payload


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run three deterministic offline-distillation and Hessian repetitions, "
            "evaluate each with COCO metrics, and write one evidence bundle."
        )
    )
    parser.add_argument(
        "--student",
        default="reports/predictions_rtdetr_pose_baseline.json",
        help="Stable student predictions inside this repository.",
    )
    parser.add_argument(
        "--teacher",
        default="data/smoke/predictions/predictions_dummy.json",
        help="Teacher predictions inside this repository.",
    )
    parser.add_argument("--dataset", default="data/coco128", help="Evaluation dataset inside this repository.")
    parser.add_argument("--split", default="train2017", help="Dataset split.")
    parser.add_argument(
        "--distill-config",
        default="configs/examples/distill_predictions.yaml",
        help="Distillation config inside this repository.",
    )
    parser.add_argument(
        "--hessian-config",
        default="configs/runtime/hessian_refine_example.yaml",
        help="Hessian config inside this repository.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/artifact_research_qualification",
        help="Fresh output directory. Existing directories are refused.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Deterministic repeated inputs (default: 3).")
    return parser.parse_args(argv)


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _repo_input(value: str, *, kind: str) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{kind} must stay inside the repository: {resolved}") from exc
    if kind == "dataset":
        if not resolved.is_dir():
            raise FileNotFoundError(f"{kind} directory not found: {resolved}")
    elif not resolved.is_file():
        raise FileNotFoundError(f"{kind} file not found: {resolved}")
    return resolved


def _fresh_output(value: str, *, create: bool = True) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to replace existing output directory: {resolved}")
    if create:
        resolved.mkdir(parents=True)
    return resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_ref(path: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _bundle_ref(path: Path, *, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _canonical_json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
        payload = payload["predictions"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def _normalize_baseline(source: Path, output: Path) -> int:
    raw = json.loads(source.read_text(encoding="utf-8"))
    entries = raw.get("predictions") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ValueError(f"unsupported student predictions payload: {source}")
    canonical = canonicalize_predictions(entries, strict=False, policy="clamp").entries
    validate_predictions_payload(canonical, strict=False)
    output.write_text(json.dumps(canonical, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(canonical)


def _metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics")
    if payload.get("ok") is not True or not isinstance(metrics, dict):
        raise ValueError(f"stable evaluator did not succeed: {path}")
    return {str(key): float(value) for key, value in metrics.items()}


def _research_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    report = payload.get("research_report")
    if not isinstance(report, dict):
        raise ValueError(f"missing research_report: {path}")
    return report


def _hessian_stop_reasons(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    for image in payload.get("images", []):
        for detection in image.get("detections", []):
            offsets = detection.get("offsets")
            if isinstance(offsets, dict):
                counts[str(offsets.get("stop_reason", "unknown"))] += 1
            else:
                counts["not_attempted"] += 1
    return dict(sorted(counts.items()))


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_source() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout.splitlines()
    return {"commit": commit, "tracked_changes_present": bool(dirty), "tracked_changes": dirty}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.repeats < 3:
        raise SystemExit("--repeats must be >= 3 for qualification")

    try:
        _fresh_output(args.output_dir, create=False)
        student = _repo_input(args.student, kind="student")
        teacher = _repo_input(args.teacher, kind="teacher")
        dataset = _repo_input(args.dataset, kind="dataset")
        distill_config = _repo_input(args.distill_config, kind="distill config")
        hessian_config = _repo_input(args.hessian_config, kind="hessian config")
        output_dir = _fresh_output(args.output_dir)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    baseline_predictions = output_dir / "baseline_predictions.json"
    baseline_images = _normalize_baseline(student, baseline_predictions)
    baseline_eval = output_dir / "baseline_eval.json"
    eval_base = [
        sys.executable,
        str(repo_root / "tools/eval_coco.py"),
        "--dataset",
        str(dataset),
        "--split",
        args.split,
    ]
    _run([*eval_base, "--predictions", str(baseline_predictions), "--output", str(baseline_eval)])

    distill_runs: list[dict[str, Any]] = []
    hessian_runs: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        distill_predictions = output_dir / f"distill_repeat_{repeat}.json"
        distill_report = output_dir / f"distill_repeat_{repeat}_report.json"
        distill_eval = output_dir / f"distill_repeat_{repeat}_eval.json"
        _run(
            [
                sys.executable,
                str(repo_root / "tools/distill_predictions.py"),
                "--student",
                str(student),
                "--teacher",
                str(teacher),
                "--dataset",
                str(dataset),
                "--split",
                args.split,
                "--config",
                str(distill_config),
                "--output",
                str(distill_predictions),
                "--output-report",
                str(distill_report),
            ]
        )
        _run([*eval_base, "--predictions", str(distill_predictions), "--output", str(distill_eval)])
        distill_boundary = _research_report(distill_report)
        distill_runs.append(
            {
                "repeat": repeat,
                "prediction_artifact": _bundle_ref(distill_predictions, output_dir=output_dir),
                "prediction_sha256": _sha256(distill_predictions),
                "prediction_canonical_sha256": _canonical_json_sha256(distill_predictions),
                "research_report": _bundle_ref(distill_report, output_dir=output_dir),
                "research_report_sha256": _sha256(distill_report),
                "latency_overhead": distill_boundary["latency_overhead"],
                "stable_eval": _bundle_ref(distill_eval, output_dir=output_dir),
                "stable_eval_sha256": _sha256(distill_eval),
                "metrics": _metrics(distill_eval),
            }
        )

        hessian_predictions = output_dir / f"hessian_repeat_{repeat}.json"
        hessian_report = output_dir / f"hessian_repeat_{repeat}_report.json"
        hessian_eval = output_dir / f"hessian_repeat_{repeat}_eval.json"
        _run(
            [
                sys.executable,
                str(repo_root / "tools/refine_predictions_hessian.py"),
                "--predictions",
                str(baseline_predictions),
                "--output",
                str(hessian_predictions),
                "--dataset",
                str(dataset),
                "--split",
                args.split,
                "--config",
                str(hessian_config),
                "--enable",
                "--refine-offsets",
                "--log-output",
                str(hessian_report),
            ]
        )
        _run([*eval_base, "--predictions", str(hessian_predictions), "--output", str(hessian_eval)])
        hessian_boundary = _research_report(hessian_report)
        hessian_runs.append(
            {
                "repeat": repeat,
                "prediction_artifact": _bundle_ref(hessian_predictions, output_dir=output_dir),
                "prediction_sha256": _sha256(hessian_predictions),
                "prediction_canonical_sha256": _canonical_json_sha256(hessian_predictions),
                "research_report": _bundle_ref(hessian_report, output_dir=output_dir),
                "research_report_sha256": _sha256(hessian_report),
                "latency_overhead": hessian_boundary["latency_overhead"],
                "stable_eval": _bundle_ref(hessian_eval, output_dir=output_dir),
                "stable_eval_sha256": _sha256(hessian_eval),
                "metrics": _metrics(hessian_eval),
                "stop_reasons": _hessian_stop_reasons(hessian_report),
            }
        )

    baseline_metrics = _metrics(baseline_eval)
    distill_hashes = {run["prediction_canonical_sha256"] for run in distill_runs}
    hessian_hashes = {run["prediction_canonical_sha256"] for run in hessian_runs}
    distill_metrics = distill_runs[0]["metrics"]
    hessian_metrics = hessian_runs[0]["metrics"]
    teacher_raw = json.loads(teacher.read_text(encoding="utf-8"))
    teacher_entries = teacher_raw.get("predictions") if isinstance(teacher_raw, dict) else teacher_raw

    summary = {
        "schema_version": 1,
        "kind": "artifact_research_qualification",
        "timestamp": _now_utc(),
        "source": _git_source(),
        "inputs": {
            "student": _repo_ref(student),
            "student_sha256": _sha256(student),
            "teacher": _repo_ref(teacher),
            "teacher_sha256": _sha256(teacher),
            "teacher_prediction_images": len(teacher_entries) if isinstance(teacher_entries, list) else None,
            "dataset": _repo_ref(dataset),
            "dataset_tree_sha256": _tree_sha256(dataset),
            "split": args.split,
            "distill_config": _repo_ref(distill_config),
            "distill_config_sha256": _sha256(distill_config),
            "hessian_config": _repo_ref(hessian_config),
            "hessian_config_sha256": _sha256(hessian_config),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "pycocotools": _package_version("pycocotools"),
        },
        "stable_baseline": {
            "source_artifact": _repo_ref(student),
            "normalized_artifact": _bundle_ref(baseline_predictions, output_dir=output_dir),
            "normalized_sha256": _sha256(baseline_predictions),
            "prediction_images": baseline_images,
            "eval_artifact": _bundle_ref(baseline_eval, output_dir=output_dir),
            "eval_sha256": _sha256(baseline_eval),
            "metrics": baseline_metrics,
        },
        "distillation": {
            "repetitions": distill_runs,
            "deterministic_prediction_output": len(distill_hashes) == 1,
            "metric_delta": {
                key: distill_metrics[key] - baseline_metrics[key]
                for key in sorted(set(baseline_metrics) & set(distill_metrics))
            },
            "promotion_gate": {
                "decision": "hold",
                "reason": (
                    "The checked-in teacher is a ten-image smoke fixture, not an independently "
                    "inferred teacher over the full evaluation set; positive metrics are interface "
                    "evidence, not model-efficacy evidence."
                ),
            },
            "rollback": "Keep the stable baseline artifact; transformed outputs are separate files.",
        },
        "hessian": {
            "repetitions": hessian_runs,
            "deterministic_prediction_output": len(hessian_hashes) == 1,
            "metric_delta": {
                key: hessian_metrics[key] - baseline_metrics[key]
                for key in sorted(set(baseline_metrics) & set(hessian_metrics))
            },
            "promotion_gate": {
                "decision": "hold",
                "reason": (
                    "COCO128 has no per-instance depth/mask auxiliary signal for this offset "
                    "objective, so the run is a measured no-signal negative control."
                ),
            },
            "rollback": "Keep the stable baseline artifact; refined outputs are separate files.",
        },
        "qualification": {
            "repeats_required": 3,
            "repeats_completed": args.repeats,
            "stable_evaluator": "tools/eval_coco.py with pycocotools",
            "all_prediction_outputs_deterministic": len(distill_hashes) == 1 and len(hessian_hashes) == 1,
            "production_promotion": False,
        },
    }
    summary_path = output_dir / "qualification_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
