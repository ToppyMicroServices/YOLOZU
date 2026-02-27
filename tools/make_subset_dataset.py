#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.run_record import build_run_record  # noqa: E402


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pick_split(dataset_root: Path, requested: str | None) -> str:
    if requested:
        return str(requested)
    images_dir = dataset_root / "images"
    for candidate in ("val2017", "train2017"):
        if (images_dir / candidate).is_dir():
            return candidate
    if images_dir.is_dir():
        splits = sorted([p.name for p in images_dir.iterdir() if p.is_dir()])
        if splits:
            return splits[0]
    raise SystemExit(f"could not infer split under: {images_dir} (pass --split)")


def _iter_images(images_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.gif"):
        paths.extend(images_dir.glob(ext))
    return sorted(paths)


def _hash_key(*, seed: int, name: str) -> str:
    return hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()


def _select(paths: list[Path], *, n: int | None, strategy: str, seed: int) -> list[Path]:
    if n is None:
        return list(paths)
    n = max(0, int(n))
    if n == 0:
        return []
    if strategy == "first":
        return list(paths[:n])
    if strategy == "hash":
        ranked = [(_hash_key(seed=int(seed), name=p.name), p.name, p) for p in paths]
        ranked.sort(key=lambda t: (t[0], t[1]))
        return [p for _, _, p in ranked[:n]]
    raise SystemExit(f"unknown strategy: {strategy}")


def _link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(str(src), str(dst))
    except Exception:
        shutil.copy2(src, dst)


def _sha256_lines(lines: list[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a deterministic subset YOLO dataset (symlink/copy images+labels).")
    p.add_argument("--dataset", required=True, help="Source YOLO-format dataset root.")
    p.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p.add_argument("--n", type=int, default=50, help="Number of images to include (default: 50). Use 0 for empty.")
    p.add_argument("--seed", type=int, default=0, help="Selection seed (default: 0).")
    p.add_argument(
        "--strategy",
        choices=("hash", "first"),
        default="hash",
        help="Selection strategy (default: hash).",
    )
    p.add_argument(
        "--out",
        default="reports/subset_dataset",
        help="Output dataset root (default: reports/subset_dataset).",
    )
    p.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks.")
    p.add_argument("--overwrite", action="store_true", help="Delete output directory if it exists.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    dataset_root = Path(args.dataset)
    if not dataset_root.is_absolute():
        dataset_root = (repo_root / dataset_root).resolve()

    split = _pick_split(dataset_root, args.split)
    images_src = dataset_root / "images" / split
    labels_src = dataset_root / "labels" / split

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = (repo_root / out_root).resolve()
    if out_root.exists() and bool(args.overwrite):
        shutil.rmtree(out_root)

    images_out = out_root / "images" / split
    labels_out = out_root / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    all_images = _iter_images(images_src)
    selected = _select(all_images, n=args.n, strategy=str(args.strategy), seed=int(args.seed))
    names = [p.name for p in selected]

    for src in selected:
        _link_or_copy(src, images_out / src.name, copy=bool(args.copy))

        stem = src.stem
        label_txt = labels_src / f"{stem}.txt"
        if label_txt.exists():
            _link_or_copy(label_txt, labels_out / label_txt.name, copy=bool(args.copy))

        meta_json = labels_src / f"{stem}.json"
        if meta_json.exists():
            _link_or_copy(meta_json, labels_out / meta_json.name, copy=bool(args.copy))

    subset_txt = out_root / "subset_images.txt"
    subset_txt.write_text("".join([f"{n}\n" for n in names]), encoding="utf-8")
    subset_sha = _sha256_lines(names)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": _now_utc(),
        "source": {"dataset": str(dataset_root), "split": str(split)},
        "output": {"dataset": str(out_root), "split": str(split)},
        "selection": {"strategy": str(args.strategy), "n": int(args.n), "seed": int(args.seed)},
        "images": names,
        "images_sha256": subset_sha,
        "run": build_run_record(
            repo_root=repo_root,
            argv=(sys.argv[1:] if argv is None else list(argv)),
            args={"dataset": str(args.dataset), "split": args.split, "n": int(args.n), "seed": int(args.seed), "strategy": str(args.strategy)},
            dataset_root=str(dataset_root),
            extra={"subset_images_txt": str(subset_txt), "subset_sha256": subset_sha},
        ),
    }

    subset_json = out_root / "subset.json"
    subset_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

