import argparse
import glob
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.predictions import load_predictions_entries, validate_predictions_entries


_BUCKETS = ("yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x")


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="YOLO-format COCO root (images/ + labels/).")
    p.add_argument("--predictions-glob", default="reports/pred_yolo26*.json", help="Glob for yolo26 bucket JSONs.")
    p.add_argument(
        "--protocol",
        choices=("yolo26", "none"),
        default="yolo26",
        help="Evaluation protocol preset to apply (default: yolo26). Use 'none' for smoke runs.",
    )
    p.add_argument("--split", default=None, help="Override dataset split when --protocol none (e.g., train2017).")
    p.add_argument("--output-suite", default="reports/eval_suite.json", help="Where to write suite JSON.")
    p.add_argument("--archive-root", default="baselines/yolo26_runs", help="Where to archive suite + metadata.")
    p.add_argument(
        "--run-id",
        default=None,
        help="Archive folder name (default: UTC timestamp).",
    )
    p.add_argument("--notes", default=None, help="Short notes to include in run metadata JSON.")
    p.add_argument("--notes-file", default=None, help="Path to a notes text/markdown file to embed in metadata.")
    p.add_argument(
        "--copy-predictions",
        action="store_true",
        help="Also copy prediction JSONs into the archive folder (can be large).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip COCOeval; only validate/convert predictions (no pycocotools required).",
    )
    p.add_argument("--strict", action="store_true", help="Use strict prediction schema validation.")
    return p.parse_args(argv)


def _now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _load_tool_module(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bucket_from_path(path: Path) -> str | None:
    name = path.stem.lower()
    for b in _BUCKETS:
        if b in name:
            return b
    return None


def _collect_bucket_files(paths: list[Path], *, glob_str: str) -> dict[str, Path]:
    by_bucket: dict[str, list[Path]] = {b: [] for b in _BUCKETS}
    for p in paths:
        bucket = _bucket_from_path(p)
        if bucket is None:
            continue
        by_bucket[bucket].append(p)

    missing = [b for b, ps in by_bucket.items() if not ps]
    if missing:
        raise SystemExit(f"missing YOLO26 buckets: {', '.join(missing)} (glob: {glob_str})")
    dup = {b: ps for b, ps in by_bucket.items() if len(ps) > 1}
    if dup:
        msg = "; ".join(f"{b}: {[p.name for p in ps]}" for b, ps in dup.items())
        raise SystemExit(f"multiple files matched per bucket (make glob more specific): {msg}")

    return {b: ps[0] for b, ps in by_bucket.items()}


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if Path(args.predictions_glob).is_absolute():
        pred_paths = [Path(p) for p in sorted(glob.glob(args.predictions_glob))]
    else:
        pred_paths = sorted((repo_root / ".").glob(args.predictions_glob))
    if not pred_paths:
        raise SystemExit(f"no predictions matched: {args.predictions_glob}")

    bucket_files = _collect_bucket_files(pred_paths, glob_str=args.predictions_glob)
    pred_paths = [bucket_files[b] for b in _BUCKETS]

    commands: list[str] = []
    predictions_meta = []

    for path in pred_paths:
        bucket = _bucket_from_path(path)
        entries = load_predictions_entries(path)
        validation = validate_predictions_entries(entries, strict=args.strict)
        commands.append(f"python3 tools/validate_predictions.py {'--strict ' if args.strict else ''}{path.as_posix()}")
        predictions_meta.append(
            {
                "bucket": bucket,
                "path": str(path),
                "entries": len(entries),
                "warnings": validation.warnings,
                "sha256": _file_sha256(path),
            }
        )

    suite_cmd = [
        "python3",
        "tools/eval_suite.py",
        "--dataset",
        args.dataset,
        "--predictions-glob",
        args.predictions_glob,
        "--output",
        args.output_suite,
    ]
    if args.protocol != "none":
        suite_cmd.extend(["--protocol", args.protocol])
    elif args.split:
        suite_cmd.extend(["--split", args.split])
    if args.dry_run:
        suite_cmd.append("--dry-run")
        if args.protocol != "none":
            suite_cmd.append("--allow-protocol-mismatches")
    if args.strict:
        suite_cmd.append("--strict")

    commands.append(" ".join(suite_cmd))

    eval_suite = _load_tool_module(repo_root / "tools" / "eval_suite.py", "eval_suite")
    eval_suite.main(suite_cmd[2:])  # pass argv after 'python3 tools/eval_suite.py'

    suite_path = repo_root / args.output_suite
    if not suite_path.exists():
        raise SystemExit(f"eval_suite did not write: {suite_path}")

    run_id = args.run_id or _now_utc().replace(":", "-")
    archive_dir = repo_root / args.archive_root / run_id
    archive_dir.mkdir(parents=True, exist_ok=False)

    suite_archive_path = archive_dir / "eval_suite.json"
    shutil.copy2(suite_path, suite_archive_path)

    notes_text = None
    if args.notes_file:
        notes_path = Path(args.notes_file)
        if not notes_path.is_absolute():
            notes_path = repo_root / notes_path
        notes_text = notes_path.read_text()

    run_payload = {
        "timestamp": _now_utc(),
        "protocol_id": None if args.protocol == "none" else args.protocol,
        "dataset": args.dataset,
        "split": args.split,
        "predictions_glob": args.predictions_glob,
        "bucket_files": {b: str(p) for b, p in bucket_files.items()},
        "output_suite": args.output_suite,
        "dry_run": bool(args.dry_run),
        "strict": bool(args.strict),
        "run_id": run_id,
        "archive_dir": str(archive_dir),
        "git_head": _git_head(),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "notes": args.notes,
        "notes_text": notes_text,
        "commands": commands,
        "predictions": predictions_meta,
        "artifacts": {"eval_suite_json": str(suite_archive_path)},
    }
    (archive_dir / "run.json").write_text(json.dumps(run_payload, indent=2, sort_keys=True))

    if args.copy_predictions:
        preds_dir = archive_dir / "predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)
        for path in pred_paths:
            shutil.copy2(path, preds_dir / path.name)

    print(archive_dir)


if __name__ == "__main__":
    main()
