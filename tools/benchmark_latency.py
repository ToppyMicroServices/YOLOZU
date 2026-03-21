import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.benchmark import measure_latency
from yolozu.metrics_report import append_jsonl, build_report, write_json


_BUCKETS = ("yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Optional JSON config with bucket settings.")
    p.add_argument("--output", default="reports/benchmark_latency.json", help="Where to write the report JSON.")
    p.add_argument("--history", default=None, help="Optional JSONL path to append the report for comparisons.")
    p.add_argument("--buckets", default=",".join(_BUCKETS), help="Comma-separated bucket list.")
    p.add_argument("--iterations", type=int, default=200, help="Benchmark iterations per bucket.")
    p.add_argument("--warmup", type=int, default=20, help="Warmup iterations per bucket.")
    p.add_argument("--sleep-s", type=float, default=0.0, help="Optional sleep per iteration (fallback step).")
    p.add_argument("--engine-template", default=None, help="Engine path template, supports {bucket}.")
    p.add_argument("--model-template", default=None, help="Model path template, supports {bucket}.")
    p.add_argument("--notes", default=None, help="Short notes to include in report meta.")
    p.add_argument("--notes-file", default=None, help="Path to notes text/markdown to embed in meta.")
    p.add_argument("--run-id", default=None, help="Optional run id for comparison; default is UTC timestamp.")
    p.add_argument("--target-fps", type=float, default=None, help="Optional minimum FPS target.")
    p.add_argument("--target-latency-ms", type=float, default=None, help="Optional maximum mean latency target (ms).")
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
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, PermissionError):
        return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_path(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = repo_root / path
    return str(path)


def _bucket_list(value: str) -> list[str]:
    if not value:
        return list(_BUCKETS)
    return [b.strip() for b in value.split(",") if b.strip()]


def _bucket_from_entry(entry: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(entry, str):
        return entry, {}
    name = str(entry.get("name")) if isinstance(entry, dict) else str(entry)
    return name, entry if isinstance(entry, dict) else {}


def _extract_metrics(entry: dict[str, Any]) -> dict[str, Any] | None:
    if "metrics" in entry and isinstance(entry.get("metrics"), dict):
        return dict(entry["metrics"])
    metrics_path = entry.get("metrics_path") or entry.get("metricsPath")
    if metrics_path:
        path = Path(str(metrics_path))
        if not path.is_absolute():
            path = repo_root / path
        payload = _load_json(path)
        if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
            return dict(payload["metrics"])
        if isinstance(payload, dict):
            return dict(payload)
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    config: dict[str, Any] = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = repo_root / config_path
        config = _load_json(config_path)

    buckets_cfg = config.get("buckets") if isinstance(config.get("buckets"), list) else None
    if not buckets_cfg:
        buckets_cfg = _bucket_list(args.buckets)

    output_path = config.get("output") or args.output
    history_path = config.get("history") or args.history

    notes_text = None
    if args.notes_file:
        notes_path = Path(args.notes_file)
        if not notes_path.is_absolute():
            notes_path = repo_root / notes_path
        notes_text = notes_path.read_text()

    run_id = args.run_id or config.get("run_id") or _now_utc().replace(":", "-")
    target_fps = args.target_fps if args.target_fps is not None else config.get("target_fps")
    target_latency_ms = (
        args.target_latency_ms if args.target_latency_ms is not None else config.get("target_latency_ms")
    )

    bucket_results: list[dict[str, Any]] = []
    for entry in buckets_cfg:
        bucket_name, entry_cfg = _bucket_from_entry(entry)
        if not bucket_name:
            continue

        engine_path = entry_cfg.get("engine") or entry_cfg.get("engine_path")
        if engine_path is None and args.engine_template:
            engine_path = args.engine_template.format(bucket=bucket_name)

        model_path = entry_cfg.get("model") or entry_cfg.get("model_path")
        if model_path is None and args.model_template:
            model_path = args.model_template.format(bucket=bucket_name)

        metrics = _extract_metrics(entry_cfg)
        if metrics is None:
            iterations = int(entry_cfg.get("iterations", args.iterations))
            warmup = int(entry_cfg.get("warmup", args.warmup))
            sleep_s = float(entry_cfg.get("sleep_s", args.sleep_s))
            metrics = measure_latency(iterations=iterations, warmup=warmup, sleep_s=sleep_s)

        bucket_results.append(
            {
                "name": bucket_name,
                "engine": _resolve_path(engine_path),
                "model": _resolve_path(model_path),
                "metrics": metrics,
                "metrics_path": entry_cfg.get("metrics_path"),
            }
        )

    fps_values = [float(b["metrics"].get("fps", 0.0)) for b in bucket_results if b.get("metrics")]
    latency_values = [
        float((b["metrics"].get("latency_ms") or {}).get("mean", 0.0))
        for b in bucket_results
        if b.get("metrics")
    ]
    summary = {
        "fps_mean": round(sum(fps_values) / len(fps_values), 3) if fps_values else 0.0,
        "latency_ms_mean": round(sum(latency_values) / len(latency_values), 3) if latency_values else 0.0,
    }
    ok = True
    if target_fps is not None:
        ok = ok and summary["fps_mean"] >= float(target_fps)
    if target_latency_ms is not None:
        ok = ok and summary["latency_ms_mean"] <= float(target_latency_ms)
    summary["target_fps"] = None if target_fps is None else float(target_fps)
    summary["target_latency_ms"] = None if target_latency_ms is None else float(target_latency_ms)
    summary["ok"] = bool(ok)

    meta = {
        "run_id": run_id,
        "git_head": _git_head(),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "notes": args.notes or config.get("notes"),
        "notes_text": notes_text,
        "engine_template": args.engine_template or config.get("engine_template"),
        "model_template": args.model_template or config.get("model_template"),
        "targets": {
            "fps_min": None if target_fps is None else float(target_fps),
            "latency_ms_max": None if target_latency_ms is None else float(target_latency_ms),
        },
    }

    report = build_report(metrics={"summary": summary, "buckets": bucket_results}, meta=meta)
    output_abs = Path(output_path)
    if not output_abs.is_absolute():
        output_abs = repo_root / output_abs
    write_json(output_abs, report)
    if history_path:
        history_abs = Path(history_path)
        if not history_abs.is_absolute():
            history_abs = repo_root / history_abs
        append_jsonl(history_abs, report)
    print(output_abs)
    if (target_fps is not None or target_latency_ms is not None) and not ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
