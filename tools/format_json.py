import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretty-format JSON file for easier debugging.")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", default=None, help="Output JSON file path (default: overwrite input)")
    parser.add_argument("--indent", type=int, default=2, help="Indent width (default: 2)")
    parser.add_argument("--sort-keys", action="store_true", help="Sort keys alphabetically")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=args.indent, sort_keys=args.sort_keys, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(out_path)


if __name__ == "__main__":
    main()
