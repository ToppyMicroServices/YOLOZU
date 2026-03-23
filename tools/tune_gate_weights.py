import argparse
import json
import platform
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.cli_args import (
    require_non_negative_float,
    require_non_negative_int,
    require_positive_int,
    resolve_input_path,
    resolve_output_path,
)
from yolozu.dataset import build_manifest
from yolozu.image_keys import image_key_aliases
from yolozu.metrics_report import build_report, write_json
from yolozu.predictions import load_predictions_entries
from yolozu.predictions_transform import fuse_detection_scores
from yolozu.simple_map import evaluate_map


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


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list):
        out = []
        for item in value:
            try:
                out.append(float(item))
            except (TypeError, ValueError, OverflowError):
                continue
        return out
    if isinstance(value, str):
        raw = [p.strip() for p in value.split(",")]
        out = []
        for p in raw:
            if not p:
                continue
            out.append(float(p))
        return out
    raise ValueError(f"expected list or csv string, got {type(value).__name__}")


def _allowed_image_keys(records: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for r in records:
        image = str(r.get("image", ""))
        if not image:
            continue
        keys.update(image_key_aliases(image))
    return keys


def _filter_entries(entries: list[dict[str, Any]], allowed_keys: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in entries:
        image = str(e.get("image", ""))
        if not image:
            continue
        aliases = image_key_aliases(image)
        if any(alias in allowed_keys for alias in aliases):
            out.append(e)
    return out


def _iou_thresholds(metric: str) -> list[float]:
    metric = str(metric)
    if metric == "map50":
        return [0.5]
    # simple_map proxy: approximate COCO-style averaging
    return [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50..0.95


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune inference-time gate weights (CPU-only, simple mAP proxy).")
    p.add_argument("--config", default=None, help="Optional JSON config file.")
    p.add_argument("--dataset", default=None, help="YOLO-format dataset root (images/ + labels/).")
    p.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    p.add_argument("--predictions", default=None, help="Predictions JSON (YOLOZU schema).")
    p.add_argument("--output-report", default=None, help="Where to write the tuning report JSON.")
    p.add_argument("--output-predictions", default=None, help="Optional path to write tuned predictions JSON.")
    p.add_argument(
        "--metric",
        choices=("map50", "map50_95"),
        default=None,
        help="Optimization metric (default: map50_95).",
    )
    p.add_argument("--det-score-key", default=None, help="Detection score key (default: score).")
    p.add_argument("--template-score-key", default=None, help="Template score key (default: score_tmp_sym).")
    p.add_argument("--sigma-z-key", default=None, help="Uncertainty key for depth (default: sigma_z).")
    p.add_argument("--sigma-rot-key", default=None, help="Uncertainty key for rotation (default: sigma_rot).")
    p.add_argument("--preserve-det-score-key", default=None, help="Where to store original det score (default: score_det).")
    p.add_argument("--grid-det", default=None, help="Grid values for w_det (csv or JSON list).")
    p.add_argument("--grid-tmp", default=None, help="Grid values for w_tmp (csv or JSON list).")
    p.add_argument("--grid-unc", default=None, help="Grid values for w_unc (csv or JSON list).")
    p.add_argument("--grid-tau", default=None, help="Optional grid values for template gate tau.")
    p.add_argument(
        "--template-gate",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable template gate (default: false).",
    )
    p.add_argument("--tau", type=float, default=None, help="Template gate tau when not tuning grid-tau (default: 0.0).")
    p.add_argument("--min-score", type=float, default=None, help="Optional min fused score threshold.")
    p.add_argument("--topk-per-image", type=int, default=None, help="Optional top-K per image after fusion.")
    p.add_argument(
        "--wrap-output",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="If set, wrap tuned predictions as {predictions:[...], meta:{...}}.",
    )
    return p.parse_args(argv)


def _load_config(path: str | None) -> tuple[dict[str, Any], Path | None]:
    if not path:
        return {}, None
    cfg_path = resolve_input_path(path, cwd=Path.cwd(), repo_root=repo_root)
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}, cfg_path.parent
    return data, cfg_path.parent


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    cfg, cfg_dir = _load_config(args.config)
    cwd = Path.cwd()

    # Defaults (overridden by config, then CLI).
    defaults: dict[str, Any] = {
        "output_report": "reports/gate_tuning_report.json",
        "metric": "map50_95",
        "det_score_key": "score",
        "template_score_key": "score_tmp_sym",
        "sigma_z_key": "sigma_z",
        "sigma_rot_key": "sigma_rot",
        "preserve_det_score_key": "score_det",
        "grid_det": [1.0],
        "grid_tmp": [0.0, 0.5, 1.0],
        "grid_unc": [0.0, 0.5, 1.0],
        "template_gate": False,
        "tau": 0.0,
        "wrap_output": False,
    }

    # Nested config conveniences.
    if isinstance(cfg.get("keys"), dict):
        keys = cfg["keys"]
        cfg = dict(cfg)
        cfg.setdefault("det_score_key", keys.get("det_score"))
        cfg.setdefault("template_score_key", keys.get("template_score"))
        cfg.setdefault("sigma_z_key", keys.get("sigma_z"))
        cfg.setdefault("sigma_rot_key", keys.get("sigma_rot"))
        cfg.setdefault("preserve_det_score_key", keys.get("preserve_det_score"))
    if isinstance(cfg.get("grid"), dict):
        grid = cfg["grid"]
        cfg = dict(cfg)
        cfg.setdefault("grid_det", grid.get("det"))
        cfg.setdefault("grid_tmp", grid.get("tmp"))
        cfg.setdefault("grid_unc", grid.get("unc"))
        cfg.setdefault("grid_tau", grid.get("tau"))
    if isinstance(cfg.get("template_gate"), dict):
        tg = cfg["template_gate"]
        cfg = dict(cfg)
        if "enabled" in tg:
            cfg.setdefault("template_gate", tg.get("enabled"))
        if "tau" in tg:
            cfg.setdefault("tau", tg.get("tau"))

    def pick_with_source(name: str, cli_value: Any) -> tuple[Any, str]:
        if cli_value is not None:
            return cli_value, "cli"
        if name in cfg and cfg[name] is not None:
            return cfg[name], "config"
        return defaults.get(name), "default"

    def pick(name: str, cli_value: Any) -> Any:
        return pick_with_source(name, cli_value)[0]

    def resolve_input_from_source(value: Any, *, source: str) -> Path:
        if source == "config" and cfg_dir is not None:
            return resolve_input_path(value, cwd=cwd, repo_root=repo_root, config_dir=cfg_dir)
        return resolve_input_path(value, cwd=cwd, repo_root=repo_root)

    def resolve_output_from_source(value: Any, *, source: str) -> Path:
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path
        if source == "config" and cfg_dir is not None:
            return Path(cfg_dir) / path
        return resolve_output_path(path, cwd=cwd)

    dataset_raw, dataset_source = pick_with_source("dataset", args.dataset)
    if not dataset_raw:
        raise SystemExit("--dataset is required (or set it in --config)")
    predictions_raw, predictions_source = pick_with_source("predictions", args.predictions)
    if not predictions_raw:
        raise SystemExit("--predictions is required (or set it in --config)")
    dataset = resolve_input_from_source(dataset_raw, source=dataset_source)
    predictions_path = resolve_input_from_source(predictions_raw, source=predictions_source)

    metric = pick("metric", args.metric)
    det_score_key = pick("det_score_key", args.det_score_key)
    template_score_key = pick("template_score_key", args.template_score_key)
    sigma_z_key = pick("sigma_z_key", args.sigma_z_key)
    sigma_rot_key = pick("sigma_rot_key", args.sigma_rot_key)
    preserve_det_score_key = pick("preserve_det_score_key", args.preserve_det_score_key)

    template_gate = pick("template_gate", args.template_gate)
    tau = float(pick("tau", args.tau))

    grid_det = _as_float_list(pick("grid_det", args.grid_det)) or [1.0]
    grid_tmp = _as_float_list(pick("grid_tmp", args.grid_tmp)) or [0.0]
    grid_unc = _as_float_list(pick("grid_unc", args.grid_unc)) or [0.0]
    grid_tau = _as_float_list(pick("grid_tau", args.grid_tau))
    if grid_tau is None:
        grid_tau = [tau]

    max_images = pick("max_images", args.max_images)
    min_score = pick("min_score", args.min_score)
    topk_per_image = pick("topk_per_image", args.topk_per_image)
    wrap_output = bool(pick("wrap_output", args.wrap_output))

    try:
        max_images = require_non_negative_int(max_images, flag_name="--max-images")
        min_score = require_non_negative_float(min_score, flag_name="--min-score")
        topk_per_image = require_positive_int(topk_per_image, flag_name="--topk-per-image")
        tau = require_non_negative_float(tau, flag_name="--tau")
        for idx, tau_value in enumerate(grid_tau):
            require_non_negative_float(float(tau_value), flag_name=f"--grid-tau[{idx}]")
        if str(metric) not in ("map50", "map50_95"):
            raise ValueError("--metric must be map50 or map50_95")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    manifest = build_manifest(dataset, split=pick("split", args.split))
    records = manifest["images"]
    if max_images is not None:
        records = records[: int(max_images)]

    allowed_keys = _allowed_image_keys(records)
    base_entries = load_predictions_entries(predictions_path)
    base_entries = _filter_entries(base_entries, allowed_keys)

    thresholds = _iou_thresholds(str(metric))
    baseline = evaluate_map(records, base_entries, iou_thresholds=thresholds)

    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_value = float("-inf")

    for w_det, w_tmp, w_unc, tau_i in product(grid_det, grid_tmp, grid_unc, grid_tau):
        fused = fuse_detection_scores(
            base_entries,
            weights={"det": float(w_det), "tmp": float(w_tmp), "unc": float(w_unc)},
            det_score_key=str(det_score_key),
            template_score_key=str(template_score_key),
            sigma_z_key=str(sigma_z_key),
            sigma_rot_key=str(sigma_rot_key),
            preserve_det_score_key=None if preserve_det_score_key in (None, "") else str(preserve_det_score_key),
            template_gate_enabled=bool(template_gate),
            template_gate_tau=float(tau_i),
            min_score=None if min_score is None else float(min_score),
            topk_per_image=None if topk_per_image is None else int(topk_per_image),
        )

        scores = evaluate_map(records, fused.entries, iou_thresholds=thresholds)
        row = {
            "det": float(w_det),
            "tmp": float(w_tmp),
            "unc": float(w_unc),
            "tau": float(tau_i),
            "map50": float(scores.map50),
            "map50_95": float(scores.map50_95),
            "warnings": fused.warnings[:50],
        }
        results.append(row)

        value = float(scores.map50_95 if str(metric) == "map50_95" else scores.map50)
        if value > best_value:
            best_value = value
            best = row

    results.sort(key=lambda r: float(r.get(str(metric), 0.0)), reverse=True)

    best_weights = None
    if best is not None:
        best_weights = {"det": best["det"], "tmp": best["tmp"], "unc": best["unc"]}

    metrics = {
        "baseline": {"map50": float(baseline.map50), "map50_95": float(baseline.map50_95)},
        "best": best,
        "grid_size": len(results),
    }
    meta = {
        "run_id": _now_utc().replace(":", "-"),
        "git_head": _git_head(),
        "config": str(resolve_input_path(args.config, cwd=cwd, repo_root=repo_root)) if args.config else None,
        "dataset": str(dataset),
        "split": manifest.get("split"),
        "max_images": None if max_images is None else int(max_images),
        "predictions": str(predictions_path),
        "path_resolution": {
            "cli_relative": "cwd",
            "config_relative": "config_file_dir",
            "repo_fallback": True,
        },
        "metric": str(metric),
        "iou_thresholds": thresholds,
        "keys": {
            "det_score": str(det_score_key),
            "template_score": str(template_score_key),
            "sigma_z": str(sigma_z_key),
            "sigma_rot": str(sigma_rot_key),
            "preserve_det_score": preserve_det_score_key,
        },
        "template_gate": {"enabled": bool(template_gate), "tau_grid": grid_tau, "tau": float(tau)},
        "weights_grid": {"det": grid_det, "tmp": grid_tmp, "unc": grid_unc},
        "min_score": min_score,
        "topk_per_image": topk_per_image,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": sys.version,
    }
    report = build_report(metrics={"tuning": metrics, "results": results}, meta=meta)
    output_report_value, output_report_source = pick_with_source("output_report", args.output_report)
    report_path = resolve_output_from_source(output_report_value, source=output_report_source)
    write_json(report_path, report)
    print(report_path)

    output_predictions_value, output_predictions_source = pick_with_source("output_predictions", args.output_predictions)
    output_predictions = (
        resolve_output_from_source(output_predictions_value, source=output_predictions_source)
        if output_predictions_value
        else None
    )
    if output_predictions and best_weights is not None:
        tuned = fuse_detection_scores(
            base_entries,
            weights=best_weights,
            det_score_key=str(det_score_key),
            template_score_key=str(template_score_key),
            sigma_z_key=str(sigma_z_key),
            sigma_rot_key=str(sigma_rot_key),
            preserve_det_score_key=None if preserve_det_score_key in (None, "") else str(preserve_det_score_key),
            template_gate_enabled=bool(template_gate),
            template_gate_tau=float(best.get("tau", tau) if best else tau),
            min_score=None if min_score is None else float(min_score),
            topk_per_image=None if topk_per_image is None else int(topk_per_image),
        )

        payload: Any = tuned.entries
        if wrap_output:
            payload = {"predictions": tuned.entries, "meta": {"best": best, "best_weights": best_weights, "source": meta}}
        out_path = output_predictions
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
