import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.map_targets import load_map_targets_doc, load_targets_map


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="reports/eval_suite.json", help="Suite output JSON from tools/eval_suite.py")
    parser.add_argument("--targets", default="baselines/yolo26_targets.json", help="Targets JSON path")
    parser.add_argument("--key", default="map50_95", help="Metric key to compare (default: map50_95)")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Require model >= target + min_delta")
    return parser.parse_args(argv)


def _validate_suite_contract(suite: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(suite, dict):
        return ["suite contract: root must be an object"]

    for key in ("timestamp", "dataset", "bbox_format", "images", "results"):
        if key not in suite:
            errors.append(f"suite contract: missing top-level key `{key}`")

    if "images" in suite and not isinstance(suite.get("images"), int):
        errors.append("suite contract: `images` must be an integer")

    results = suite.get("results")
    if results is None:
        return errors
    if not isinstance(results, list):
        errors.append("suite contract: `results` must be an array")
        return errors

    required_metric_keys = ("map50_95", "map50", "map75", "ar100")
    for idx, result in enumerate(results):
        path = f"results[{idx}]"
        if not isinstance(result, dict):
            errors.append(f"suite contract: `{path}` must be an object")
            continue
        for key in ("name", "path", "metrics"):
            if key not in result:
                errors.append(f"suite contract: missing key `{path}.{key}`")

        metrics = result.get("metrics")
        if metrics is None:
            continue
        if not isinstance(metrics, dict):
            errors.append(f"suite contract: `{path}.metrics` must be an object")
            continue
        for metric_key in required_metric_keys:
            if metric_key not in metrics:
                errors.append(f"suite contract: missing key `{path}.metrics.{metric_key}`")

    return errors


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    suite = json.loads((repo_root / args.suite).read_text())
    contract_errors = _validate_suite_contract(suite)
    targets_doc = load_map_targets_doc(repo_root / args.targets)
    targets = load_targets_map(repo_root / args.targets)
    doc_key = targets_doc.get("metric_key")

    failures = list(contract_errors)
    if doc_key and args.key != doc_key:
        failures.append(f"metric key mismatch: --key={args.key} but targets.metric_key={doc_key}")

    rows = []
    for result in suite.get("results", []):
        name = result.get("name", "")
        metrics = (result.get("metrics") or {})
        value = metrics.get(args.key)

        target = None
        for k, v in targets.items():
            if k in name:
                target = v
                break

        rows.append({"name": name, "value": value, "target": target})

        if target is None:
            failures.append(f"{name}: target missing (set baselines/yolo26_targets.json)")
            continue
        if value is None:
            failures.append(f"{name}: metric missing ({args.key})")
            continue
        if float(value) < float(target) + float(args.min_delta):
            failures.append(f"{name}: {args.key}={value:.4f} < target={target:.4f} (+{args.min_delta})")

    report = {
        "suite": args.suite,
        "targets": args.targets,
        "key": args.key,
        "min_delta": args.min_delta,
        "rows": rows,
        "ok": len(failures) == 0,
        "failures": failures,
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
