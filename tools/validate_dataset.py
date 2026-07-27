import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest  # noqa: E402
from yolozu.dataset_validator import validate_dataset_records  # noqa: E402


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None, help="YOLO-format dataset root (defaults to data/coco128).")
    p.add_argument("--split", default=None, help="Dataset split under images/ and labels/.")
    p.add_argument(
        "--mode",
        choices=("fail", "warn"),
        default="fail",
        help="Treat record validation failures as errors or warnings; an empty dataset always fails.",
    )
    p.add_argument("--non-strict", action="store_true", help="Relax bbox range checks (still checks types/structure).")
    p.add_argument("--no-image-check", action="store_true", help="Skip checking that image files exist and can be read.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    dataset_root = Path(args.dataset) if args.dataset else (repo_root / "data" / "coco128")
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]

    result = validate_dataset_records(
        records,
        strict=not bool(args.non_strict),
        mode=str(args.mode),
        check_images=not bool(args.no_image_check),
    )

    for w in result.warnings:
        print(f"WARN: {w}")
    for e in result.errors:
        print(f"ERROR: {e}")

    if result.errors:
        raise SystemExit(1)

    print(f"OK: {len(records)} records")


if __name__ == "__main__":
    main()
