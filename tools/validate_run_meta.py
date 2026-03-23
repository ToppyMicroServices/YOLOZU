import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.run_record import validate_run_record_contract  # noqa: E402


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("run_meta", help="Path to runs/<run_id>/reports/run_meta.json")
    p.add_argument(
        "--allow-missing-git-sha",
        action="store_true",
        help="Allow missing git SHA (for detached artifact-only environments).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    path = Path(args.run_meta)
    if not path.exists():
        raise SystemExit(f"file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise SystemExit(f"failed to parse json: {path} ({exc})") from exc

    try:
        validate_run_record_contract(payload, require_git_sha=not bool(args.allow_missing_git_sha))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"OK: {path}")


if __name__ == "__main__":
    main()
