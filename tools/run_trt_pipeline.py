import argparse
import json
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]

_BUCKETS = ("yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run YOLO26 TensorRT pipeline end-to-end (Runpod/Linux).")
    p.add_argument("--dataset", required=True, help="YOLO-format COCO root (images/ + labels/). Can be absolute.")
    p.add_argument("--buckets", default=",".join(_BUCKETS), help="Comma-separated bucket list.")
    p.add_argument(
        "--onnx-template",
        required=True,
        help="ONNX path template with {bucket} (and optional {precision}).",
    )
    p.add_argument(
        "--engine-template",
        default="engines/{bucket}_{precision}.plan",
        help="TensorRT engine output template (default: engines/{bucket}_{precision}.plan).",
    )
    p.add_argument(
        "--engine-meta-template",
        default="reports/trt_engine_{bucket}_{precision}.json",
        help="Engine build metadata template (default: reports/trt_engine_{bucket}_{precision}.json).",
    )
    p.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp16", help="Engine precision.")
    p.add_argument("--input-name", default="images", help="Input binding name (default: images).")
    p.add_argument("--shape", default="1x3x640x640", help="Input shape (default: 1x3x640x640).")
    p.add_argument("--min-shape", default=None, help="Min shape override for engine build (default: --shape).")
    p.add_argument("--opt-shape", default=None, help="Opt shape override for engine build (default: --shape).")
    p.add_argument("--max-shape", default=None, help="Max shape override for engine build (default: --shape).")
    p.add_argument("--workspace-mib", type=int, default=4096, help="trtexec workspace in MiB (default: 4096).")
    p.add_argument("--timing-cache", default="engines/timing.cache", help="Timing cache path.")
    p.add_argument("--trtexec", default="trtexec", help="Path to trtexec binary.")

    # INT8 knobs (optional; FP16 is the default path).
    p.add_argument(
        "--calib-cache-template",
        default=None,
        help="INT8 calibration cache template with {bucket} (required for int8 unless --dry-run).",
    )
    p.add_argument("--calib-dataset", default=None, help="Optional dataset root to generate calibration image list.")
    p.add_argument("--calib-split", default=None, help="Optional dataset split for calibration image list.")
    p.add_argument("--calib-images", type=int, default=500, help="Calibration images count (default: 500).")
    p.add_argument(
        "--calib-list-template",
        default="reports/calib_images_{bucket}.txt",
        help="Calibration image list path template (default: reports/calib_images_{bucket}.txt).",
    )

    # Export settings (combined-output is the common Ultralytics path).
    p.add_argument("--combined-output", default="output0", help="Combined output name (default: output0).")
    p.add_argument("--boxes-scale", choices=("abs", "norm"), default="abs", help="Combined boxes scale.")
    p.add_argument("--min-score", type=float, default=0.0, help="Score threshold (no NMS).")
    p.add_argument("--topk", type=int, default=300, help="Top-K per image (no NMS).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for export/eval/parity.")

    p.add_argument(
        "--pred-onnxrt-template",
        default="reports/pred_onnxrt_{bucket}.json",
        help="ONNXRuntime predictions output template.",
    )
    p.add_argument(
        "--pred-trt-template",
        default="reports/pred_trt_{bucket}.json",
        help="TensorRT predictions output template.",
    )
    p.add_argument(
        "--parity-report-template",
        default="reports/parity_{bucket}.json",
        help="Parity report output template.",
    )
    p.add_argument("--image-size", default="640", help="Fixed image size for parity (default: 640).")
    p.add_argument("--parity-iou", type=float, default=0.99, help="Parity IoU threshold.")
    p.add_argument("--parity-score-atol", type=float, default=1e-4, help="Parity score atol.")
    p.add_argument("--parity-bbox-atol", type=float, default=1e-4, help="Parity bbox atol.")

    p.add_argument("--eval-suite-output", default="reports/eval_suite_trt.json", help="Eval suite output JSON path.")

    p.add_argument(
        "--latency-template",
        default="reports/latency_{bucket}.json",
        help="Per-bucket TensorRT latency output template.",
    )
    p.add_argument("--latency-iterations", type=int, default=200, help="Latency iterations per bucket.")
    p.add_argument("--latency-warmup", type=int, default=20, help="Latency warmup per bucket.")

    p.add_argument(
        "--benchmark-config",
        default="reports/benchmark_latency_config.json",
        help="Benchmark harness config JSON path (generated).",
    )
    p.add_argument("--benchmark-output", default="reports/benchmark_latency.json", help="Latency report output JSON.")
    p.add_argument("--benchmark-history", default=None, help="Optional JSONL history path for latency reports.")

    p.add_argument("--run-id", default=None, help="Optional run id for run record (default: UTC timestamp).")
    p.add_argument("--run-dir", default=None, help="Optional run folder to copy key artifacts into.")
    p.add_argument("--notes", default=None, help="Short notes to include in run records.")

    p.add_argument("--skip-build", action="store_true", help="Skip TensorRT engine build.")
    p.add_argument("--skip-onnxrt", action="store_true", help="Skip ONNXRuntime export.")
    p.add_argument("--skip-trt", action="store_true", help="Skip TensorRT export.")
    p.add_argument("--skip-parity", action="store_true", help="Skip parity checks.")
    p.add_argument("--skip-eval", action="store_true", help="Skip eval_suite.")
    p.add_argument("--skip-latency", action="store_true", help="Skip per-engine latency measurement.")
    p.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark_latency summary.")

    p.add_argument("--force", action="store_true", help="Recompute outputs even if files exist.")
    p.add_argument("--dry-run", action="store_true", help="Run sub-tools in --dry-run mode (no GPU required).")
    return p.parse_args(argv)


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_head() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip() or None
    except Exception:
        return None


def _bucket_list(value: str) -> list[str]:
    if not value:
        return list(_BUCKETS)
    return [b.strip() for b in value.split(",") if b.strip()]


def _format_path(template: str, *, bucket: str, precision: str, run_id: str) -> Path:
    rendered = template.format(bucket=bucket, precision=precision, run_id=run_id)
    path = Path(rendered)
    if path.is_absolute():
        return path
    return repo_root / path


def _partial_template(template: str, *, precision: str, run_id: str) -> str:
    """Fill known placeholders while keeping {bucket} for downstream format()."""

    return template.format(bucket="{bucket}", precision=precision, run_id=run_id)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _run(
    cmd: list[str],
    *,
    capture_stdout: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(repo_root),
        check=True,
        stdout=subprocess.PIPE if capture_stdout else None,
        stderr=subprocess.PIPE if capture_stdout else None,
        text=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    python = sys.executable
    buckets = _bucket_list(str(args.buckets))
    run_id = args.run_id or _now_utc().replace(":", "-")
    precision = str(args.precision)

    commands: list[str] = []
    artifacts: dict[str, Any] = {"buckets": {}}

    dataset = str(args.dataset)
    split_note = None if not args.calib_split else str(args.calib_split)

    for bucket in buckets:
        bucket_art: dict[str, Any] = {}

        onnx_path = _format_path(str(args.onnx_template), bucket=bucket, precision=precision, run_id=run_id)
        engine_path = _format_path(str(args.engine_template), bucket=bucket, precision=precision, run_id=run_id)
        engine_meta = _format_path(str(args.engine_meta_template), bucket=bucket, precision=precision, run_id=run_id)
        pred_onnxrt = _format_path(str(args.pred_onnxrt_template), bucket=bucket, precision=precision, run_id=run_id)
        pred_trt = _format_path(str(args.pred_trt_template), bucket=bucket, precision=precision, run_id=run_id)
        parity_out = _format_path(str(args.parity_report_template), bucket=bucket, precision=precision, run_id=run_id)
        latency_out = _format_path(str(args.latency_template), bucket=bucket, precision=precision, run_id=run_id)

        bucket_art.update(
            {
                "onnx": str(onnx_path),
                "engine": str(engine_path),
                "engine_meta": str(engine_meta),
                "pred_onnxrt": str(pred_onnxrt),
                "pred_trt": str(pred_trt),
                "parity_report": str(parity_out),
                "latency": str(latency_out),
            }
        )

        if not args.skip_build:
            if precision == "int8" and not args.dry_run and not args.calib_cache_template:
                raise SystemExit("--calib-cache-template is required for --precision int8")
            if args.force or not engine_path.exists() or not engine_meta.exists():
                _ensure_parent(engine_path)
                _ensure_parent(engine_meta)
                build_cmd = [
                    python,
                    "tools/build_trt_engine.py",
                    "--onnx",
                    str(onnx_path),
                    "--engine",
                    str(engine_path),
                    "--precision",
                    precision,
                    "--input-name",
                    str(args.input_name),
                    "--min-shape",
                    str(args.min_shape or args.shape),
                    "--opt-shape",
                    str(args.opt_shape or args.shape),
                    "--max-shape",
                    str(args.max_shape or args.shape),
                    "--workspace",
                    str(int(args.workspace_mib)),
                    "--timing-cache",
                    str(args.timing_cache),
                    "--trtexec",
                    str(args.trtexec),
                    "--meta-output",
                    str(engine_meta),
                ]
                if precision == "int8" and args.calib_cache_template:
                    calib_cache = _format_path(
                        str(args.calib_cache_template), bucket=bucket, precision=precision, run_id=run_id
                    )
                    build_cmd.extend(["--calib-cache", str(calib_cache)])
                    if args.calib_dataset:
                        calib_list = _format_path(
                            str(args.calib_list_template), bucket=bucket, precision=precision, run_id=run_id
                        )
                        _ensure_parent(calib_list)
                        build_cmd.extend(
                            [
                                "--calib-dataset",
                                str(args.calib_dataset),
                                "--calib-images",
                                str(int(args.calib_images)),
                                "--calib-list-output",
                                str(calib_list),
                            ]
                        )
                        if split_note:
                            build_cmd.extend(["--calib-split", split_note])
                if args.dry_run:
                    build_cmd.append("--dry-run")
                commands.append(shlex.join(build_cmd))
                _run(build_cmd)

        if not args.skip_onnxrt:
            if args.force or not pred_onnxrt.exists():
                _ensure_parent(pred_onnxrt)
                cmd = [
                    python,
                    "tools/export_predictions_onnxrt.py",
                    "--dataset",
                    dataset,
                    "--onnx",
                    str(onnx_path),
                    "--input-name",
                    str(args.input_name),
                    "--combined-output",
                    str(args.combined_output),
                    "--combined-format",
                    "xyxy_score_class",
                    "--boxes-scale",
                    str(args.boxes_scale),
                    "--min-score",
                    str(float(args.min_score)),
                    "--topk",
                    str(int(args.topk)),
                    "--wrap",
                    "--output",
                    str(pred_onnxrt),
                ]
                if args.max_images is not None:
                    cmd.extend(["--max-images", str(int(args.max_images))])
                if args.dry_run:
                    cmd.append("--dry-run")
                commands.append(shlex.join(cmd))
                _run(cmd)

        if not args.skip_trt:
            if args.force or not pred_trt.exists():
                _ensure_parent(pred_trt)
                cmd = [
                    python,
                    "tools/export_predictions_trt.py",
                    "--dataset",
                    dataset,
                    "--engine",
                    str(engine_path),
                    "--input-name",
                    str(args.input_name),
                    "--combined-output",
                    str(args.combined_output),
                    "--combined-format",
                    "xyxy_score_class",
                    "--boxes-scale",
                    str(args.boxes_scale),
                    "--min-score",
                    str(float(args.min_score)),
                    "--topk",
                    str(int(args.topk)),
                    "--wrap",
                    "--output",
                    str(pred_trt),
                ]
                if args.max_images is not None:
                    cmd.extend(["--max-images", str(int(args.max_images))])
                if args.dry_run:
                    cmd.append("--dry-run")
                commands.append(shlex.join(cmd))
                _run(cmd)

        if not args.skip_parity and not args.skip_onnxrt and not args.skip_trt:
            if args.force or not parity_out.exists():
                _ensure_parent(parity_out)
                cmd = [
                    python,
                    "tools/check_predictions_parity.py",
                    "--reference",
                    str(pred_onnxrt),
                    "--candidate",
                    str(pred_trt),
                    "--image-size",
                    str(args.image_size),
                    "--iou-thresh",
                    str(float(args.parity_iou)),
                    "--score-atol",
                    str(float(args.parity_score_atol)),
                    "--bbox-atol",
                    str(float(args.parity_bbox_atol)),
                ]
                if args.max_images is not None:
                    cmd.extend(["--max-images", str(int(args.max_images))])
                commands.append(shlex.join(cmd))
                proc = _run(cmd, capture_stdout=True)
                parity_out.write_text(proc.stdout or "")

        if not args.skip_latency and not args.skip_trt:
            if args.force or not latency_out.exists():
                _ensure_parent(latency_out)
                cmd = [
                    python,
                    "tools/measure_trt_latency.py",
                    "--engine",
                    str(engine_path),
                    "--input-name",
                    str(args.input_name),
                    "--shape",
                    str(args.shape),
                    "--iterations",
                    str(int(args.latency_iterations)),
                    "--warmup",
                    str(int(args.latency_warmup)),
                    "--output",
                    str(latency_out),
                ]
                if args.notes:
                    cmd.extend(["--notes", str(args.notes)])
                if args.dry_run:
                    cmd.append("--dry-run")
                commands.append(shlex.join(cmd))
                _run(cmd)

        artifacts["buckets"][bucket] = bucket_art

    # Eval suite (TRT predictions)
    eval_suite_out = _format_path(str(args.eval_suite_output), bucket=buckets[0], precision=precision, run_id=run_id)
    if not args.skip_eval and not args.skip_trt:
        if args.force or not eval_suite_out.exists():
            _ensure_parent(eval_suite_out)
            glob_str = str(
                _format_path(str(args.pred_trt_template), bucket="yolo26*", precision=precision, run_id=run_id)
            )
            cmd = [
                python,
                "tools/eval_suite.py",
                "--protocol",
                "yolo26",
                "--dataset",
                dataset,
                "--predictions-glob",
                glob_str,
                "--output",
                str(eval_suite_out),
            ]
            if args.max_images is not None:
                cmd.extend(["--max-images", str(int(args.max_images))])
            if args.dry_run:
                cmd.append("--dry-run")
                cmd.append("--allow-protocol-mismatches")
            commands.append(shlex.join(cmd))
            _run(cmd)

    # Benchmark summary (consume per-bucket latency files)
    benchmark_config = _format_path(str(args.benchmark_config), bucket=buckets[0], precision=precision, run_id=run_id)
    benchmark_out = _format_path(str(args.benchmark_output), bucket=buckets[0], precision=precision, run_id=run_id)
    benchmark_history = (
        None
        if not args.benchmark_history
        else _format_path(str(args.benchmark_history), bucket=buckets[0], precision=precision, run_id=run_id)
    )

    if not args.skip_benchmark and not args.skip_latency and not args.skip_trt:
        if args.force or not benchmark_out.exists():
            _ensure_parent(benchmark_config)
            _ensure_parent(benchmark_out)
            if benchmark_history is not None:
                _ensure_parent(benchmark_history)

            engine_template_bench = _partial_template(str(args.engine_template), precision=precision, run_id=run_id)
            model_template_bench = _partial_template(str(args.onnx_template), precision=precision, run_id=run_id)

            cfg = {
                "output": str(benchmark_out),
                "history": None if benchmark_history is None else str(benchmark_history),
                "notes": args.notes,
                "engine_template": engine_template_bench,
                "model_template": model_template_bench,
                "buckets": [{"name": b, "metrics_path": str(artifacts["buckets"][b]["latency"])} for b in buckets],
            }
            benchmark_config.write_text(json.dumps(cfg, indent=2, sort_keys=True))

            cmd = [
                python,
                "tools/benchmark_latency.py",
                "--config",
                str(benchmark_config),
                "--engine-template",
                engine_template_bench,
                "--model-template",
                model_template_bench,
            ]
            if args.notes:
                cmd.extend(["--notes", str(args.notes)])
            commands.append(shlex.join(cmd))
            _run(cmd)

    # Optional run record / artifact copy
    run_dir = None
    if args.run_dir:
        run_dir = Path(str(args.run_dir).format(run_id=run_id))
        if not run_dir.is_absolute():
            run_dir = repo_root / run_dir
    else:
        run_dir = repo_root / "runs" / "trt_runs" / run_id

    run_dir.mkdir(parents=True, exist_ok=True)

    def _copy_if_exists(path: Path) -> None:
        if not path.exists():
            return
        dst = run_dir / path.name
        if dst.exists() and not args.force:
            return
        shutil.copy2(path, dst)

    for bucket in buckets:
        _copy_if_exists(Path(artifacts["buckets"][bucket]["engine_meta"]))
        _copy_if_exists(Path(artifacts["buckets"][bucket]["latency"]))
        _copy_if_exists(Path(artifacts["buckets"][bucket]["parity_report"]))
    _copy_if_exists(eval_suite_out)
    _copy_if_exists(benchmark_out)
    if benchmark_history is not None:
        _copy_if_exists(benchmark_history)

    run_payload: dict[str, Any] = {
        "timestamp": _now_utc(),
        "run_id": run_id,
        "notes": args.notes,
        "dry_run": bool(args.dry_run),
        "dataset": dataset,
        "buckets": buckets,
        "precision": precision,
        "onnx_template": str(args.onnx_template),
        "engine_template": str(args.engine_template),
        "git_head": _git_head(),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "commands": commands,
        "artifacts": {
            "run_dir": str(run_dir),
            "eval_suite": str(eval_suite_out),
            "benchmark": str(benchmark_out),
            "benchmark_history": None if benchmark_history is None else str(benchmark_history),
            "benchmark_config": str(benchmark_config),
            "buckets": artifacts["buckets"],
        },
    }

    (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2, sort_keys=True))
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
