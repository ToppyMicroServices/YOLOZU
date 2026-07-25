import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.predictions import validate_predictions_path  # noqa: E402


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="Path to predictions JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail closed on interface-contract violations. Without this flag, "
            "compatibility repair is enabled and reported in warnings."
        ),
    )
    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Emit a stable machine-readable validation result.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    result, exit_code = validate_predictions_path(
        args.predictions,
        strict=bool(args.strict),
        max_warnings=100 if args.json else None,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return exit_code

    if not result["ok"]:
        errors = list(result.get("errors") or [])
        message = str(errors[0].get("message")) if errors else "validation failed"
        raise SystemExit(message)
    for w in result["warnings"]:
        print(f"WARN: {w}")

    print(f"OK: {result['entry_count']} image entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
