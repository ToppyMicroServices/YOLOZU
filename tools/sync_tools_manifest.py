import json
import argparse
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync tools/manifest.json to packaged copy yolozu/data/manifest/tools_manifest.json."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _parse_args(sys.argv[1:] if argv is None else argv)
    src = repo_root / "tools" / "manifest.json"
    dst = repo_root / "yolozu" / "data" / "manifest" / "tools_manifest.json"

    if not src.exists():
        print(f"error: source manifest not found: {src}", file=sys.stderr)
        return 2

    obj = _read_json(src)
    _write_json(dst, obj)
    print(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
