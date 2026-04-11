#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.datasets.segmentation_dataset import load_seg_dataset_descriptor  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Package a directory of class-id masks into the YOLOZU segmentation predictions interface contract."
    )
    p.add_argument("--dataset-json", required=True, help="Segmentation dataset descriptor with samples[].")
    p.add_argument("--masks-dir", required=True, help="Directory containing predicted masks named by sample id.")
    p.add_argument("--output", required=True, help="Segmentation predictions JSON output path.")
    p.add_argument("--suffix", default=".png", help="Mask filename suffix inside --masks-dir (default: .png).")
    p.add_argument(
        "--relative-to-output",
        action="store_true",
        help="Write relative mask paths from --output parent instead of absolute paths.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite --output if it already exists.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    out_path = Path(str(args.output)).resolve()
    if out_path.exists() and not bool(args.force):
        raise SystemExit(f"--output already exists: {out_path} (use --force to overwrite)")

    desc = load_seg_dataset_descriptor(args.dataset_json)
    masks_dir = Path(str(args.masks_dir)).resolve()
    if not masks_dir.is_dir():
        raise SystemExit(f"--masks-dir not found: {masks_dir}")

    entries: list[dict[str, str]] = []
    for sample in desc.samples:
        mask_path = masks_dir / f"{sample.sample_id}{args.suffix}"
        if not mask_path.is_file():
            raise SystemExit(f"missing predicted mask for sample_id={sample.sample_id}: {mask_path}")
        mask_value = str(mask_path)
        if bool(args.relative_to_output):
            mask_value = str(mask_path.relative_to(out_path.parent))
        entries.append({"id": str(sample.sample_id), "mask": mask_value})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
