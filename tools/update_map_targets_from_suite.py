import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


_BUCKET_RE = re.compile(r"(yolo26[nsmlx])")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update baselines/yolo26_targets.json from an eval_suite report.")
    p.add_argument("--suite", required=True, help="Path to reports/eval_suite.json (or similar).")
    p.add_argument("--targets", default="baselines/yolo26_targets.json", help="Targets JSON path.")
    p.add_argument("--key", default=None, help="Metric key under result.metrics (default: from targets.metric_key).")
    p.add_argument("--dry-run", action="store_true", help="Print updated JSON but do not write to disk.")
    return p.parse_args(argv)


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_head(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, PermissionError):
        return None
    return out if out else None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_bucket(value: str) -> str | None:
    m = _BUCKET_RE.search(value)
    if not m:
        return None
    return m.group(1)


def _extract_metrics_map(suite: dict[str, Any], *, key: str) -> dict[str, float]:
    results = suite.get("results")
    if not isinstance(results, list):
        raise ValueError("suite.results must be a list")

    out: dict[str, float] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        name = str(result.get("name") or "")
        path = str(result.get("path") or "")
        bucket = _infer_bucket(name) or _infer_bucket(path)
        if not bucket:
            continue

        metrics = result.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        out[bucket] = float(value)

    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = Path(__file__).resolve().parents[1]

    suite_path = Path(args.suite)
    if not suite_path.is_absolute():
        suite_path = repo_root / suite_path
    targets_path = Path(args.targets)
    if not targets_path.is_absolute():
        targets_path = repo_root / targets_path

    suite = _load_json(suite_path)
    if not isinstance(suite, dict):
        raise ValueError("suite JSON must be an object")

    targets_doc = _load_json(targets_path)
    if not isinstance(targets_doc, dict):
        raise ValueError("targets JSON must be an object")

    key = str(args.key or targets_doc.get("metric_key") or "map50_95")
    metric_map = _extract_metrics_map(suite, key=key)
    if not metric_map:
        raise SystemExit(f"no {key} metrics matched yolo26 buckets in suite: {suite_path}")

    targets = targets_doc.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("targets.targets must be an object")

    changed = 0
    for bucket, value in metric_map.items():
        if bucket not in targets:
            continue
        prev = targets.get(bucket)
        if prev is None or float(prev) != float(value):
            targets[bucket] = float(value)
            changed += 1

    provenance = targets_doc.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
        targets_doc["provenance"] = provenance
    provenance.update(
        {
            "updated_at": _now_utc(),
            "metric_key": key,
            "suite_path": str(suite_path),
            "suite_timestamp": suite.get("timestamp"),
            "suite_protocol_id": suite.get("protocol_id"),
            "git_head": _git_head(repo_root),
        }
    )

    out_text = json.dumps(targets_doc, indent=2, sort_keys=True) + "\n"
    if args.dry_run:
        print(out_text, end="")
        print(f"dry-run: would update {changed} buckets", file=sys.stderr)
        return 0

    targets_path.write_text(out_text, encoding="utf-8")
    print(targets_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
