#!/usr/bin/env python3
"""Run a bounded naive-vs-SDFT continual-learning qualification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a repository-tracked three-seed naive-vs-SDFT sequence with "
            "real COCOeval, promotion decisions, provenance, and a checksum bundle."
        )
    )
    parser.add_argument(
        "--spec",
        default="configs/continual/sdft_coco128_blur_qualification.json",
        help="Repository-tracked qualification spec.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/sdft_continual_qualification",
        help="Fresh output directory. Existing paths are refused.",
    )
    parser.add_argument(
        "--archive",
        default=None,
        help="Bundle path (default: <output-dir>.tgz). Existing paths are refused.",
    )
    parser.add_argument(
        "--role",
        choices=("primary", "independent"),
        default="primary",
        help="Reproduction role (default: primary).",
    )
    parser.add_argument(
        "--source-summary",
        default=None,
        help="Primary qualification_summary.json required for --role independent.",
    )
    return parser.parse_args(argv)


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _repo_path(value: str, *, kind: str, directory: bool = False) -> Path:
    raw = Path(value).expanduser()
    path = raw.resolve() if raw.is_absolute() else (repo_root / raw).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{kind} must stay inside the repository: {path}") from exc
    if directory:
        if not path.is_dir():
            raise FileNotFoundError(f"{kind} directory not found: {path}")
    elif not path.is_file():
        raise FileNotFoundError(f"{kind} file not found: {path}")
    return path


def _fresh_dir(value: str) -> Path:
    raw = Path(value).expanduser()
    path = raw.resolve() if raw.is_absolute() else (repo_root / raw).resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing output path: {path}")
    path.mkdir(parents=True)
    return path


def _fresh_archive(value: str | None, *, output_dir: Path) -> Path:
    if value:
        raw = Path(value).expanduser()
        path = raw.resolve() if raw.is_absolute() else (repo_root / raw).resolve()
    else:
        path = Path(f"{output_dir}.tgz")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing archive path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _state_dict_sha256(path: Path) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for checkpoint qualification") from exc
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model") if isinstance(payload, dict) and isinstance(payload.get("model"), dict) else payload
    if not isinstance(state, dict):
        raise ValueError(f"unsupported checkpoint payload: {path}")
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
        if hasattr(value, "detach"):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _run(command: list[str], *, label: str) -> float:
    print(f"[sdft-qualification] {label}", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return float(elapsed)


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


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _initial_checkpoint(
    *,
    output: Path,
    model_config: Path,
    dataset: Path,
    split: str,
    train: dict[str, Any],
    initial_training: dict[str, Any],
    seed: int,
) -> tuple[Path, float]:
    initial_epochs = int(initial_training.get("epochs", 0))
    initial_max_steps = int(initial_training.get("max_steps", 1))
    initial_lr = float(initial_training.get("lr", train["lr"]))
    command = [
        sys.executable,
        str(repo_root / "rtdetr_pose/tools/train_minimal.py"),
        "--config",
        str(model_config),
        "--dataset-root",
        str(dataset),
        "--split",
        split,
        "--run-dir",
        str(output),
        "--device",
        str(train["device"]),
        "--amp",
        str(train["amp"]),
        "--image-size",
        str(int(train["image_size"])),
        "--epochs",
        str(initial_epochs),
        "--max-steps",
        str(initial_max_steps),
        "--batch-size",
        str(int(train["batch_size"])),
        "--lr",
        str(initial_lr),
        "--num-workers",
        str(int(train["num_workers"])),
        "--seed",
        str(int(seed)),
        "--real-images",
        "--deterministic",
        "--use-matcher",
        "--strict-task-data",
        "--no-export-onnx",
    ]
    seconds = _run(command, label=f"seed {seed}: build initial checkpoint")
    checkpoint = output / "checkpoint.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"initial checkpoint missing: {checkpoint}")
    return checkpoint, seconds


def _continual_config(
    *,
    model_config: Path,
    source: Path,
    target: Path,
    split: str,
    train: dict[str, Any],
    sdft: dict[str, Any],
    seed: int,
    method: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_config": str(model_config),
        "train": dict(train),
        "continual": {
            "seed": int(seed),
            "replay_size": 0,
            "replay_strategy": "reservoir",
            "distill": {
                "enabled": method == "sdft",
                "keys": str(sdft["keys"]),
                "weight": float(sdft["weight"]),
                "temperature": float(sdft["temperature"]),
                "kl": str(sdft["kl"]),
            },
        },
        "tasks": [
            {
                "name": "source",
                "dataset_root": str(source),
                "train_split": split,
                "val_split": split,
            },
            {
                "name": "gaussian_blur_s3",
                "dataset_root": str(target),
                "train_split": split,
                "val_split": split,
            },
        ],
    }


def _run_method(
    *,
    output_dir: Path,
    model_config: Path,
    source: Path,
    target: Path,
    split: str,
    train: dict[str, Any],
    sdft: dict[str, Any],
    evaluation: dict[str, Any],
    gates: dict[str, Any],
    initial_checkpoint: Path,
    seed: int,
    method: str,
) -> dict[str, Any]:
    method_dir = output_dir / f"seed_{seed}" / method
    method_dir.mkdir(parents=True)
    config_path = method_dir / "continual_config.json"
    _write_json(
        config_path,
        _continual_config(
            model_config=model_config,
            source=source,
            target=target,
            split=split,
            train=train,
            sdft=sdft,
            seed=seed,
            method=method,
        ),
    )
    run_dir = method_dir / "run"
    train_seconds = _run(
        [
            sys.executable,
            str(repo_root / "rtdetr_pose/tools/train_continual.py"),
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--initial-checkpoint",
            str(initial_checkpoint),
            "--replay-size",
            "0",
        ],
        label=f"seed {seed} {method}: train source then target",
    )
    run_json = run_dir / "continual_run.json"
    eval_seconds = _run(
        [
            sys.executable,
            str(repo_root / "tools/eval_continual.py"),
            "--run-json",
            str(run_json),
            "--device",
            str(train["device"]),
            "--image-size",
            str(int(train["image_size"])),
            "--max-images",
            str(int(evaluation["max_images"])),
            "--metric",
            str(evaluation["backend"]),
            "--metric-key",
            str(evaluation["metric_key"]),
            "--force",
        ],
        label=f"seed {seed} {method}: real COCOeval matrix",
    )
    eval_json = run_dir / "continual_eval.json"
    decision_json = run_dir / "continual_promotion_decision.json"
    decision_seconds = _run(
        [
            sys.executable,
            str(repo_root / "tools/continual_decide.py"),
            "--eval-json",
            str(eval_json),
            "--run-json",
            str(run_json),
            "--max-forgetting",
            str(float(gates["max_forgetting"])),
            "--min-new-task-score",
            str(float(gates["min_new_task_score"])),
            "--min-old-task-final",
            str(float(gates["min_old_task_final"])),
            "--output",
            str(decision_json),
        ],
        label=f"seed {seed} {method}: promotion gate",
    )

    run = _json(run_json)
    evaluation_payload = _json(eval_json)
    decision = _json(decision_json)
    tasks = run.get("tasks") if isinstance(run.get("tasks"), list) else []
    if len(tasks) != 2:
        raise ValueError(f"expected two task stages: {run_json}")
    checkpoints = []
    for task in tasks:
        checkpoint = Path(str(task["checkpoint"]))
        checkpoints.append(
            {
                "path": checkpoint.relative_to(output_dir).as_posix(),
                "sha256": _sha256(checkpoint),
                "state_dict_sha256": _state_dict_sha256(checkpoint),
                "teacher_checkpoint": task.get("teacher_checkpoint"),
                "teacher_checkpoint_sha256": task.get("teacher_checkpoint_sha256"),
                "train_records_sha256": task.get("train_records_sha256"),
                "val_records_sha256": task.get("val_records_sha256"),
                "train_seconds": task.get("train_seconds"),
                "child_peak_rss": task.get("child_peak_rss"),
            }
        )
    return {
        "method": method,
        "seed": seed,
        "config": config_path.relative_to(output_dir).as_posix(),
        "config_sha256": _sha256(config_path),
        "run_json": run_json.relative_to(output_dir).as_posix(),
        "run_json_sha256": _sha256(run_json),
        "eval_json": eval_json.relative_to(output_dir).as_posix(),
        "eval_json_sha256": _sha256(eval_json),
        "decision_json": decision_json.relative_to(output_dir).as_posix(),
        "decision_json_sha256": _sha256(decision_json),
        "summary": evaluation_payload.get("summary"),
        "matrix_values": evaluation_payload.get("matrix_values"),
        "baseline_values": evaluation_payload.get("baseline_values"),
        "promotion_decision": decision.get("decision"),
        "checkpoints": checkpoints,
        "wall_seconds": {
            "train": train_seconds,
            "evaluation": eval_seconds,
            "decision": decision_seconds,
        },
    }


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _comparison(naive: dict[str, Any], sdft: dict[str, Any]) -> dict[str, Any]:
    naive_summary = naive.get("summary") if isinstance(naive.get("summary"), dict) else {}
    sdft_summary = sdft.get("summary") if isinstance(sdft.get("summary"), dict) else {}
    naive_matrix = naive.get("matrix_values")
    sdft_matrix = sdft.get("matrix_values")

    def delta(key: str, *, reverse: bool = False) -> float | None:
        left = _number(naive_summary.get(key))
        right = _number(sdft_summary.get(key))
        if left is None or right is None:
            return None
        return float(left - right) if reverse else float(right - left)

    old_delta = None
    new_delta = None
    if (
        isinstance(naive_matrix, list)
        and isinstance(sdft_matrix, list)
        and len(naive_matrix) >= 2
        and len(sdft_matrix) >= 2
        and isinstance(naive_matrix[-1], list)
        and isinstance(sdft_matrix[-1], list)
    ):
        naive_old = _number(naive_matrix[-1][0])
        sdft_old = _number(sdft_matrix[-1][0])
        naive_new = _number(naive_matrix[-1][-1])
        sdft_new = _number(sdft_matrix[-1][-1])
        old_delta = None if naive_old is None or sdft_old is None else float(sdft_old - naive_old)
        new_delta = None if naive_new is None or sdft_new is None else float(sdft_new - naive_new)

    naive_task0 = naive["checkpoints"][0]
    sdft_task0 = sdft["checkpoints"][0]
    return {
        "seed": naive["seed"],
        "task0_state_identical": naive_task0["state_dict_sha256"] == sdft_task0["state_dict_sha256"],
        "task0_train_order_identical": naive_task0["train_records_sha256"] == sdft_task0["train_records_sha256"],
        "task1_train_order_identical": (
            naive["checkpoints"][1]["train_records_sha256"] == sdft["checkpoints"][1]["train_records_sha256"]
        ),
        "final_old_task_delta_sdft_minus_naive": old_delta,
        "final_new_task_delta_sdft_minus_naive": new_delta,
        "avg_acc_delta_sdft_minus_naive": delta("avg_acc"),
        "forgetting_reduction_naive_minus_sdft": delta("forgetting", reverse=True),
        "bwt_delta_sdft_minus_naive": delta("bwt"),
        "fwt_delta_sdft_minus_naive": delta("fwt"),
    }


def _efficacy_assessment(
    *,
    comparisons: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    gates: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(gates, dict):
        return {
            "configured": False,
            "passed": False,
            "seed_results": [],
            "reason": "efficacy_gates_not_configured",
        }

    min_source = float(gates["min_source_score"])
    min_target = float(gates["min_target_score"])
    min_old_delta = float(gates["min_old_task_delta_sdft_minus_naive"])
    min_new_delta = float(gates["min_new_task_delta_sdft_minus_naive"])
    strict_old = bool(gates.get("require_strict_old_task_improvement", False))
    sdft_by_seed = {
        int(run["seed"]): run
        for run in runs
        if str(run.get("method")) == "sdft"
    }
    seed_results: list[dict[str, Any]] = []
    for comparison in comparisons:
        seed = int(comparison["seed"])
        run = sdft_by_seed.get(seed) or {}
        matrix = run.get("matrix_values")
        final_row = matrix[-1] if isinstance(matrix, list) and matrix and isinstance(matrix[-1], list) else []
        source_score = _number(final_row[0]) if final_row else None
        target_score = _number(final_row[-1]) if final_row else None
        old_delta = _number(comparison.get("final_old_task_delta_sdft_minus_naive"))
        new_delta = _number(comparison.get("final_new_task_delta_sdft_minus_naive"))
        checks = {
            "source_score_nonzero": source_score is not None and source_score >= min_source,
            "target_score_nonzero": target_score is not None and target_score >= min_target,
            "old_task_delta": (
                old_delta is not None
                and (old_delta > min_old_delta if strict_old else old_delta >= min_old_delta)
            ),
            "new_task_delta": new_delta is not None and new_delta >= min_new_delta,
        }
        seed_results.append(
            {
                "seed": seed,
                "source_score": source_score,
                "target_score": target_score,
                "old_task_delta_sdft_minus_naive": old_delta,
                "new_task_delta_sdft_minus_naive": new_delta,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return {
        "configured": True,
        "passed": len(seed_results) >= 3 and all(item["passed"] for item in seed_results),
        "gates": dict(gates),
        "seed_results": seed_results,
    }


def _independent_reproduction(
    *,
    source_summary: dict[str, Any],
    source_summary_path: Path,
    current_summary: dict[str, Any],
) -> dict[str, Any]:
    source_protocol = source_summary.get("protocol") if isinstance(source_summary.get("protocol"), dict) else {}
    current_protocol = current_summary.get("protocol") if isinstance(current_summary.get("protocol"), dict) else {}
    source_assessment = (
        source_summary.get("efficacy_assessment")
        if isinstance(source_summary.get("efficacy_assessment"), dict)
        else {}
    )
    current_assessment = (
        current_summary.get("efficacy_assessment")
        if isinstance(current_summary.get("efficacy_assessment"), dict)
        else {}
    )

    def directions(assessment: dict[str, Any]) -> list[dict[str, Any]]:
        rows = assessment.get("seed_results") if isinstance(assessment.get("seed_results"), list) else []
        return [
            {
                "seed": item.get("seed"),
                "old_positive": _number(item.get("old_task_delta_sdft_minus_naive")) is not None
                and float(item["old_task_delta_sdft_minus_naive"]) > 0.0,
                "new_non_degrading": _number(item.get("new_task_delta_sdft_minus_naive")) is not None
                and float(item["new_task_delta_sdft_minus_naive"])
                >= float((assessment.get("gates") or {}).get("min_new_task_delta_sdft_minus_naive", 0.0)),
                "passed": bool(item.get("passed")),
            }
            for item in rows
            if isinstance(item, dict)
        ]

    protocol_match = {
        key: source_protocol.get(key) == current_protocol.get(key)
        for key in ("seeds", "methods", "train", "initial_training", "evaluation", "promotion_gates")
    }
    direction_match = directions(source_assessment) == directions(current_assessment)
    reproduced = bool(all(protocol_match.values()) and direction_match)
    supported = bool(
        source_assessment.get("passed")
        and current_assessment.get("passed")
        and reproduced
    )
    return {
        "source_summary": str(source_summary_path),
        "source_summary_sha256": _sha256(source_summary_path),
        "protocol_match": protocol_match,
        "direction_and_gate_outcome_match": direction_match,
        "reproduced": reproduced,
        "efficacy_supported": supported,
    }


def _report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# SDFT Continual Qualification",
        "",
        f"- source commit: `{summary['source']['commit']}`",
        f"- metric backend: `{summary['protocol']['evaluation']['backend']}`",
        f"- evaluated images per task: {summary['protocol']['evaluation']['max_images']}",
        f"- aggregate decision: **{summary['decision']['status']}**",
        f"- efficacy: **{summary['decision']['efficacy']}**",
        "",
        "| Seed | Method | Avg acc | Forgetting | BWT | FWT | Decision |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for run in summary["runs"]:
        metrics = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        lines.append(
            "| {seed} | {method} | {avg} | {forget} | {bwt} | {fwt} | {decision} |".format(
                seed=run["seed"],
                method=run["method"],
                avg=metrics.get("avg_acc"),
                forget=metrics.get("forgetting"),
                bwt=metrics.get("bwt"),
                fwt=metrics.get("fwt"),
                decision=run.get("promotion_decision"),
            )
        )
    lines.extend(
        [
            "",
            "The run uses the real `pycocotools` COCOeval path. The deterministic",
            "Gaussian-blur domain is a bounded diagnostic, not an independent benchmark.",
            "The aggregate status remains `hold` until an independent reproduction and",
            "a positive retention/adaptation trade-off are both available.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_checksums(output_dir: Path) -> Path:
    path = output_dir / "checksums.sha256"
    lines = []
    for item in sorted(candidate for candidate in output_dir.rglob("*") if candidate.is_file() and candidate != path):
        lines.append(f"{_sha256(item)}  {item.relative_to(output_dir).as_posix()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _archive(output_dir: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as bundle:
        for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
            bundle.add(path, arcname=f"{output_dir.name}/{path.relative_to(output_dir).as_posix()}", recursive=False)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        spec_path = _repo_path(args.spec, kind="spec")
        spec = _json(spec_path)
        source_cfg = spec.get("source_dataset")
        if not isinstance(source_cfg, dict):
            raise ValueError("spec.source_dataset must be an object")
        source = _repo_path(str(source_cfg["root"]), kind="source dataset", directory=True)
        model_config = _repo_path(str(spec["model_config"]), kind="model config")
        output_dir = _fresh_dir(args.output_dir)
        archive_path = _fresh_archive(args.archive, output_dir=output_dir)
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    seeds = spec.get("seeds")
    methods = spec.get("methods")
    if not isinstance(seeds, list) or len(seeds) < 3:
        raise SystemExit("spec.seeds must contain at least three seeds")
    if methods != ["naive", "sdft"]:
        raise SystemExit("spec.methods must be exactly ['naive', 'sdft']")
    source_summary_path: Path | None = None
    source_summary: dict[str, Any] | None = None
    if args.role == "independent":
        if not args.source_summary:
            raise SystemExit("--source-summary is required when --role independent")
        source_summary_path = Path(str(args.source_summary)).expanduser().resolve()
        if not source_summary_path.is_file():
            raise SystemExit(f"source summary not found: {source_summary_path}")
        source_summary = _json(source_summary_path)
    train = spec["train"]
    initial_training = spec.get("initial_training")
    if initial_training is None:
        initial_training = {"epochs": 0, "max_steps": 1, "lr": train["lr"]}
    if not isinstance(initial_training, dict):
        raise SystemExit("spec.initial_training must be an object when provided")
    target_cfg = spec["target_domain"]
    evaluation = spec["evaluation"]
    gates = spec["promotion_gates"]
    split = str(source_cfg["split"])

    target = output_dir / "datasets" / "gaussian_blur_s3"
    _run(
        [
            sys.executable,
            str(repo_root / "scripts/prepare_ttt_domain_shift_target.py"),
            "--dataset-root",
            str(source),
            "--split",
            split,
            "--out",
            str(target),
            "--corruption",
            str(target_cfg["corruption"]),
            "--severity",
            str(int(target_cfg["severity"])),
            "--seed",
            str(int(target_cfg["transform_seed"])),
            "--force",
        ],
        label="prepare deterministic target domain",
    )

    runs: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    initial_records: list[dict[str, Any]] = []
    for raw_seed in seeds:
        seed = int(raw_seed)
        initial_dir = output_dir / f"seed_{seed}" / "initial"
        initial_dir.mkdir(parents=True)
        checkpoint, seconds = _initial_checkpoint(
            output=initial_dir,
            model_config=model_config,
            dataset=source,
            split=split,
            train=train,
            initial_training=initial_training,
            seed=seed,
        )
        initial_records.append(
            {
                "seed": seed,
                "checkpoint": checkpoint.relative_to(output_dir).as_posix(),
                "sha256": _sha256(checkpoint),
                "state_dict_sha256": _state_dict_sha256(checkpoint),
                "wall_seconds": seconds,
            }
        )
        seed_runs: dict[str, dict[str, Any]] = {}
        for method in methods:
            result = _run_method(
                output_dir=output_dir,
                model_config=model_config,
                source=source,
                target=target,
                split=split,
                train=train,
                sdft=spec["sdft"],
                evaluation=evaluation,
                gates=gates,
                initial_checkpoint=checkpoint,
                seed=seed,
                method=str(method),
            )
            runs.append(result)
            seed_runs[str(method)] = result
        comparisons.append(_comparison(seed_runs["naive"], seed_runs["sdft"]))

    fairness_ok = all(
        item["task0_state_identical"]
        and item["task0_train_order_identical"]
        and item["task1_train_order_identical"]
        for item in comparisons
    )
    all_decisions = [str(run.get("promotion_decision")) for run in runs]
    efficacy_assessment = _efficacy_assessment(
        comparisons=comparisons,
        runs=runs,
        gates=spec.get("efficacy_gates"),
    )
    summary = {
        "schema_version": 1,
        "kind": "sdft_continual_qualification",
        "role": str(args.role),
        "timestamp": _now_utc(),
        "source": _git_source(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "pycocotools": _package_version("pycocotools"),
        },
        "inputs": {
            "spec": spec_path.relative_to(repo_root).as_posix(),
            "spec_sha256": _sha256(spec_path),
            "model_config": model_config.relative_to(repo_root).as_posix(),
            "model_config_sha256": _sha256(model_config),
            "source_dataset": source.relative_to(repo_root).as_posix(),
            "source_dataset_tree_sha256": _tree_sha256(source),
            "target_dataset": target.relative_to(output_dir).as_posix(),
            "target_dataset_tree_sha256": _tree_sha256(target),
            "domain_shift_recipe_sha256": _sha256(target / "domain_shift_recipe.json"),
        },
        "protocol": {
            "seeds": [int(seed) for seed in seeds],
            "methods": list(methods),
            "train": train,
            "initial_training": initial_training,
            "evaluation": evaluation,
            "promotion_gates": gates,
            "same_data_order_and_budget": fairness_ok,
        },
        "initial_checkpoints": initial_records,
        "runs": runs,
        "comparisons": comparisons,
        "efficacy_assessment": efficacy_assessment,
        "decision": {
            "status": "hold",
            "efficacy": "not_established",
            "per_run_decisions": all_decisions,
            "reasons": [
                "independent reproduction is not present",
                "the repository-local blur sequence is a bounded diagnostic rather than an external benchmark",
                "promotion requires a positive retention/adaptation trade-off across all preregistered seeds",
            ],
        },
    }
    if args.role == "primary" and bool(efficacy_assessment.get("passed")):
        summary["decision"] = {
            "status": "review",
            "efficacy": "not_established",
            "per_run_decisions": all_decisions,
            "reasons": [
                "the preregistered non-zero retention/adaptation gates passed",
                "independent reproduction is required before efficacy is supported",
                "the repository-local blur sequence remains a bounded Research diagnostic",
            ],
        }
    if args.role == "independent":
        assert source_summary is not None
        assert source_summary_path is not None
        reproduction = _independent_reproduction(
            source_summary=source_summary,
            source_summary_path=source_summary_path,
            current_summary=summary,
        )
        summary["reproduction"] = reproduction
        if reproduction["efficacy_supported"]:
            summary["decision"] = {
                "status": "review",
                "efficacy": "supported",
                "per_run_decisions": all_decisions,
                "reasons": [
                    "the preregistered non-zero retention/adaptation gates passed in both runs",
                    "the independent run reproduced the per-seed direction and gate outcomes",
                    "the repository-local blur sequence remains Research and is not an external benchmark",
                ],
            }
        elif not reproduction["reproduced"]:
            summary["decision"]["reasons"].insert(
                0,
                "independent run did not reproduce all protocol, direction, and gate outcomes",
            )
        else:
            summary["decision"]["reasons"] = [
                "the independent run reproduced the protocol and gate outcomes",
                "one or more preregistered efficacy gates failed in both runs",
                "the repository-local blur sequence remains a bounded Research diagnostic",
            ]
    summary_path = output_dir / "qualification_summary.json"
    _write_json(summary_path, summary)
    (output_dir / "qualification_report.md").write_text(_report_markdown(summary), encoding="utf-8")
    checksums = _write_checksums(output_dir)
    _archive(output_dir, archive_path)

    result = {
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "checksums": str(checksums),
        "checksums_sha256": _sha256(checksums),
        "archive": str(archive_path),
        "archive_sha256": _sha256(archive_path),
        "decision": summary["decision"]["status"],
        "efficacy": summary["decision"]["efficacy"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
