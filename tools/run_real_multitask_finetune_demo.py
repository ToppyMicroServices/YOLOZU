#!/usr/bin/env python3
"""Run real-image multitask finetune smoke demo.

Tasks covered:
- bbox
- segmentation (mask-supervised metadata + bbox objective scaffold)
- keypoints
- depth
- pose6d
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run real-image few-shot multitask finetune demo.")
    p.add_argument("--dataset-root", default="data/real_multitask_fewshot", help="Prepared dataset root.")
    p.add_argument("--prepare", action="store_true", help="Run dataset preparation before training.")
    p.add_argument(
        "--download-if-missing",
        action="store_true",
        help=(
            "When --prepare is set, request tiny COCO download if inputs are missing. "
            "Requires --allow-auto-download and --accept-dataset-license."
        ),
    )
    p.add_argument(
        "--allow-auto-download",
        action="store_true",
        help="Forward --allow-auto-download to dataset preparation tool.",
    )
    p.add_argument(
        "--accept-dataset-license",
        action="store_true",
        help="Forward --accept-dataset-license to dataset preparation tool.",
    )
    p.add_argument(
        "--download-num-images",
        type=int,
        default=None,
        help="When --prepare is set, override tiny COCO download count.",
    )
    p.add_argument(
        "--prepare-args",
        type=str,
        default="",
        help="Extra args forwarded to tools/prepare_real_multitask_fewshot.py (quoted).",
    )
    p.add_argument("--out", default="reports/real_multitask_finetune_demo", help="Output directory.")
    p.add_argument("--device", default="cpu", help="Torch device for train_minimal (default: cpu).")
    p.add_argument("--epochs", type=int, default=1, help="Epochs per task (default: 1).")
    p.add_argument("--max-steps", type=int, default=2, help="Max micro-steps per epoch (default: 2).")
    p.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2).")
    p.add_argument("--image-size", type=int, default=128, help="Image size (default: 128).")
    p.add_argument("--num-keypoints", type=int, default=4, help="Keypoints count for keypoints task (default: 4).")
    p.add_argument("--python", default=sys.executable, help="Python executable for subprocess calls.")
    p.add_argument("--backbone-name", default="cspdarknet_s", help="Backbone override for demo runs.")
    p.add_argument("--backbone-norm", default="bn", help="Backbone norm override for demo runs.")
    p.add_argument(
        "--strict-provenance",
        action="store_true",
        help="Fail if dataset provenance indicates model-inference-generated labels.",
    )
    p.add_argument(
        "--strict-realism",
        action="store_true",
        help="Fail if dataset provenance indicates heuristic/scaffold labels.",
    )
    p.add_argument(
        "--backbone-args",
        default='{"width_mult": 0.5, "depth_mult": 0.34}',
        help='JSON args for backbone override (default: {"width_mult":0.5,"depth_mult":0.34}).',
    )
    p.add_argument("--skip-imbalance", action="store_true", help="Disable class-balanced sampler on bbox stage.")
    p.add_argument("--force", action="store_true", help="Overwrite output directory.")
    return p


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, cwd=(str(cwd) if cwd else None), check=False)


def _load_last_val_map(path: Path) -> float | None:
    if not path.exists():
        return None
    last = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        metrics = rec.get("metrics") if isinstance(rec, dict) else None
        if isinstance(metrics, dict) and "map50_95" in metrics:
            try:
                last = float(metrics["map50_95"])
            except Exception:
                pass
    return last


def _collect_artifact_presence(run_dir: Path) -> dict[str, Any]:
    expected = {
        "stdout_log": run_dir / "stdout.log",
        "stderr_log": run_dir / "stderr.log",
        "val_metrics_jsonl": run_dir / "val_metrics.jsonl",
        "checkpoint_bundle": run_dir / "checkpoint_bundle.pt",
    }
    present = {k: bool(p.exists()) for k, p in expected.items()}
    missing = [k for k, ok in present.items() if not ok]
    return {
        "expected": {k: str(v) for k, v in expected.items()},
        "present": present,
        "missing": missing,
        "complete": bool(not missing),
    }


def _load_prepare_summary(dataset_root: Path) -> dict[str, Any] | None:
    path = dataset_root / "prepare_summary.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _validate_prepare_summary(*, summary: dict[str, Any] | None, strict_provenance: bool, strict_realism: bool) -> list[str]:
    warnings: list[str] = []
    if not isinstance(summary, dict):
        warnings.append("prepare_summary_missing_or_invalid")
        return warnings
    provenance = summary.get("label_provenance")
    if not isinstance(provenance, dict):
        warnings.append("prepare_summary.label_provenance_missing")
        return warnings
    if bool(strict_provenance) and bool(provenance.get("model_inference_used")):
        raise SystemExit("strict provenance violation: model_inference_used must be false")
    heuristic_fields = []
    for key, value in provenance.items():
        if isinstance(value, str) and ("heuristic" in value or "derived" in value):
            heuristic_fields.append(str(key))
    if heuristic_fields:
        warnings.append("heuristic_or_scaffold_labels:" + ",".join(sorted(heuristic_fields)))
    if bool(strict_realism) and heuristic_fields:
        raise SystemExit(
            "strict realism violation: heuristic/scaffold labels are present. "
            "Use manually annotated labels for keypoints/depth/pose."
        )
    checks = summary.get("checks")
    if isinstance(checks, dict) and not bool(checks.get("segmentation_masks_non_empty", True)):
        warnings.append("segmentation_masks_non_empty_check_failed")
    return warnings


def _train_task(
    *,
    python: str,
    dataset_root: Path,
    out_dir: Path,
    task_name: str,
    device: str,
    epochs: int,
    max_steps: int,
    batch_size: int,
    image_size: int,
    backbone_name: str,
    backbone_norm: str,
    backbone_args: str,
    resume_from: Path | None,
    extra_args: list[str],
) -> dict[str, Any]:
    run_dir = out_dir / task_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python,
        "rtdetr_pose/tools/train_minimal.py",
        "--dataset-root",
        str(dataset_root),
        "--split",
        "train",
        "--val-split",
        "val",
        "--real-images",
        "--strict-task-data",
        "--epochs",
        str(int(epochs)),
        "--max-steps",
        str(int(max_steps)),
        "--batch-size",
        str(int(batch_size)),
        "--image-size",
        str(int(image_size)),
        "--device",
        str(device),
        "--run-dir",
        str(run_dir),
        "--val-every",
        "1",
        "--val-metrics-jsonl",
        str(run_dir / "val_metrics.jsonl"),
        "--no-export-onnx",
        "--use-matcher",
        "--model-config",
        "rtdetr_pose/configs/base.json",
        "--backbone-name",
        str(backbone_name),
        "--backbone-norm",
        str(backbone_norm),
        "--backbone-args",
        str(backbone_args),
    ]
    if resume_from is not None and resume_from.exists():
        cmd.extend(["--resume-from", str(resume_from)])
    cmd.extend(list(extra_args))

    proc = _run(cmd)
    (run_dir / "stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (run_dir / "stderr.log").write_text(proc.stderr or "", encoding="utf-8")

    val_map = _load_last_val_map(run_dir / "val_metrics.jsonl")
    checkpoint = run_dir / "checkpoint_bundle.pt"
    artifacts = _collect_artifact_presence(run_dir)

    return {
        "task": task_name,
        "command": cmd,
        "returncode": int(proc.returncode),
        "val_map50_95": val_map,
        "run_dir": str(run_dir),
        "checkpoint_bundle": str(checkpoint),
        "ok": bool(proc.returncode == 0),
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    out_dir = Path(str(args.out)).expanduser()
    if out_dir.exists() and not bool(args.force):
        raise SystemExit(f"output already exists: {out_dir} (use --force)")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(str(args.dataset_root)).expanduser()
    prep_result = None
    if bool(args.prepare):
        cmd = [args.python, "tools/prepare_real_multitask_fewshot.py", "--out", str(dataset_root), "--force"]
        if bool(args.download_if_missing):
            cmd.append("--download-if-missing")
        if bool(args.allow_auto_download):
            cmd.append("--allow-auto-download")
        if bool(args.accept_dataset_license):
            cmd.append("--accept-dataset-license")
        if args.download_num_images is not None:
            cmd.extend(["--download-num-images", str(int(args.download_num_images))])
        if bool(args.strict_provenance):
            cmd.append("--strict-provenance")
        if bool(args.strict_realism):
            cmd.append("--strict-realism")
        extra = [x for x in str(args.prepare_args or "").split(" ") if x.strip()]
        cmd.extend(extra)
        proc = _run(cmd)
        (out_dir / "prepare.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (out_dir / "prepare.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
        prep_result = {"command": cmd, "returncode": int(proc.returncode)}
        if proc.returncode != 0:
            report = {
                "ok": False,
                "prepare": prep_result,
                "error": "dataset preparation failed",
            }
            report_path = out_dir / "multitask_finetune_demo_report.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            print(str(report_path))
            return 2

    if not dataset_root.exists():
        raise SystemExit(f"dataset root not found: {dataset_root}")
    prepare_summary = _load_prepare_summary(dataset_root)
    provenance_warnings = _validate_prepare_summary(
        summary=prepare_summary,
        strict_provenance=bool(args.strict_provenance),
        strict_realism=bool(args.strict_realism),
    )

    base_bbox_args = []
    if not bool(args.skip_imbalance):
        base_bbox_args.extend(["--imbalance-strategy", "class_balanced", "--imbalance-gamma", "1.0"])

    task_matrix: list[tuple[str, list[str]]] = [
        ("bbox", base_bbox_args),
        (
            "segmentation",
            [
                "--fracal-stats-task",
                "seg",
                "--fracal-stats-out",
                str(out_dir / "segmentation" / "fracal_seg.json"),
            ],
        ),
        ("keypoints", ["--num-keypoints", str(int(args.num_keypoints))]),
        (
            "depth",
            [
                "--depth-mode",
                "sidecar",
                "--depth-unit",
                "relative",
                "--cost-z",
                "1.0",
            ],
        ),
        (
            "pose6d",
            [
                "--depth-mode",
                "sidecar",
                "--depth-unit",
                "relative",
                "--cost-z",
                "1.0",
                "--cost-rot",
                "1.0",
                "--cost-t",
                "0.5",
            ],
        ),
    ]

    results: list[dict[str, Any]] = []
    previous_ckpt: Path | None = None
    epoch_budget = int(args.epochs)
    for task_name, extra in task_matrix:
        res = _train_task(
            python=str(args.python),
            dataset_root=dataset_root,
            out_dir=out_dir,
            task_name=task_name,
            device=str(args.device),
            epochs=int(epoch_budget),
            max_steps=int(args.max_steps),
            batch_size=int(args.batch_size),
            image_size=int(args.image_size),
            backbone_name=str(args.backbone_name),
            backbone_norm=str(args.backbone_norm),
            backbone_args=str(args.backbone_args),
            resume_from=previous_ckpt,
            extra_args=extra,
        )
        results.append(res)

        ckpt = Path(str(res.get("checkpoint_bundle") or ""))
        previous_ckpt = ckpt if ckpt.exists() else previous_ckpt

        if not bool(res.get("ok", False)):
            break
        epoch_budget += int(args.epochs)

    ok_all = bool(results) and all(bool(r.get("ok", False)) for r in results)
    artifact_complete = all(bool((r.get("artifacts") or {}).get("complete", False)) for r in results)
    ok = bool(ok_all and artifact_complete)
    report = {
        "ok": ok,
        "dataset_root": str(dataset_root),
        "prepare": prep_result,
        "prepare_summary": prepare_summary,
        "settings": {
            "device": str(args.device),
            "epochs": int(args.epochs),
            "max_steps": int(args.max_steps),
            "batch_size": int(args.batch_size),
            "image_size": int(args.image_size),
            "strict_provenance": bool(args.strict_provenance),
            "strict_realism": bool(args.strict_realism),
            "backbone_name": str(args.backbone_name),
            "backbone_norm": str(args.backbone_norm),
            "backbone_args": str(args.backbone_args),
        },
        "tasks": results,
        "evidence": {
            "tasks_total": int(len(results)),
            "tasks_ok": int(sum(1 for r in results if bool(r.get("ok", False)))),
            "artifacts_complete": bool(artifact_complete),
            "warnings": provenance_warnings,
        },
        "notes": {
            "segmentation": "segmentation stage uses real mask metadata plus bbox objective in current train_minimal scaffold",
            "data_policy": "real source images are used; pseudo keypoints/depth/pose labels are annotation-derived heuristics (no model-inference-generated labels)",
        },
    }

    report_path = out_dir / "multitask_finetune_demo_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(str(report_path))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
