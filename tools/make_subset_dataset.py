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

_OWNERSHIP_FILE = ".yolozu_subset_output.json"
_OWNERSHIP_KIND = "yolozu_subset_output"
_PATH_SIDECAR_KEYS = (
    "mask_path",
    "M_path",
    "depth_path",
    "D_obj_path",
    "cad_path",
    "cad_points_path",
)


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
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_lines(lines: list[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _read_owned_output(path: Path) -> bool:
    marker = path / _OWNERSHIP_FILE
    if marker.is_symlink() or not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("kind") == _OWNERSHIP_KIND


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    resolved = path.resolve(strict=False)
    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        repo_root.resolve(),
        repo_root.parent.resolve(),
    }
    if resolved in protected:
        raise SystemExit(f"refusing protected output directory: {resolved}")
    if path.is_symlink():
        raise SystemExit(f"refusing symlink output directory: {path}")
    if path.exists() and not path.is_dir():
        raise SystemExit(f"output exists and is not a directory: {path}")
    if path.exists():
        entries = list(path.iterdir())
        if entries and not overwrite:
            raise SystemExit(f"output directory is not empty: {path} (pass --overwrite only for YOLOZU-owned output)")
        if entries and not _read_owned_output(path):
            raise SystemExit(f"refusing to delete unowned output directory: {path}")
        if overwrite:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / _OWNERSHIP_FILE).write_text(
        json.dumps({"kind": _OWNERSHIP_KIND, "schema_version": 1}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_relative_sidecar_path(
    value: str,
    *,
    dataset_root: Path,
    split: str,
    stem: str,
    key: str,
    index: int | None,
) -> tuple[Path, Path]:
    source = Path(value).expanduser()
    if not source.is_absolute():
        source = dataset_root / source
    source = source.resolve()
    if not source.is_file():
        raise SystemExit(f"referenced sidecar does not exist: {source}")

    try:
        relative = source.relative_to(dataset_root.resolve())
    except ValueError:
        suffix = source.suffix
        filename = f"{key}{suffix}" if index is None else f"{key}_{index:04d}{suffix}"
        relative = Path("sidecars") / split / stem / filename
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(f"unsafe sidecar path: {relative}")
    return source, relative


def _materialize_metadata(
    *,
    source: Path,
    destination: Path,
    dataset_root: Path,
    out_root: Path,
    split: str,
    stem: str,
    copy: bool,
) -> list[Path]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"label metadata must be a JSON object: {source}")
    materialized: list[Path] = []
    for key in _PATH_SIDECAR_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        values = list(value) if isinstance(value, list) and all(isinstance(item, str) for item in value) else [value]
        if not all(isinstance(item, str) for item in values):
            continue
        rewritten: list[str] = []
        for index, item in enumerate(values):
            sidecar_source, relative = _safe_relative_sidecar_path(
                item,
                dataset_root=dataset_root,
                split=split,
                stem=stem,
                key=key,
                index=(index if isinstance(value, list) else None),
            )
            sidecar_destination = out_root / relative
            _link_or_copy(sidecar_source, sidecar_destination, copy=copy)
            materialized.append(sidecar_destination)
            rewritten.append(relative.as_posix())
        payload[key] = rewritten if isinstance(value, list) else rewritten[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    materialized.append(destination)
    return materialized


def _source_metadata_hashes(dataset_root: Path) -> dict[str, str]:
    names = (
        "LICENSE",
        "LICENSE.txt",
        "README",
        "README.md",
        "README.txt",
        "PROVENANCE.md",
        "dataset.json",
        "prepare_summary.json",
    )
    return {
        name: _sha256_file(dataset_root / name)
        for name in names
        if (dataset_root / name).is_file()
    }


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
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output only when its YOLOZU ownership marker is present.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
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
    dataset_resolved = dataset_root.resolve()
    output_resolved = out_root.resolve(strict=False)
    try:
        dataset_resolved.relative_to(output_resolved)
        overlaps = True
    except ValueError:
        try:
            output_resolved.relative_to(dataset_resolved)
            overlaps = True
        except ValueError:
            overlaps = False
    if overlaps:
        raise SystemExit(f"source and output directories must not overlap: {dataset_resolved} vs {output_resolved}")
    _prepare_output(out_root, overwrite=bool(args.overwrite))

    images_out = out_root / "images" / split
    labels_out = out_root / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    all_images = _iter_images(images_src)
    selected = _select(all_images, n=args.n, strategy=str(args.strategy), seed=int(args.seed))
    names = [p.name for p in selected]

    materialized_paths: list[Path] = [out_root / _OWNERSHIP_FILE]
    for src in selected:
        image_destination = images_out / src.name
        _link_or_copy(src, image_destination, copy=bool(args.copy))
        materialized_paths.append(image_destination)

        stem = src.stem
        label_txt = labels_src / f"{stem}.txt"
        if label_txt.exists():
            label_destination = labels_out / label_txt.name
            _link_or_copy(label_txt, label_destination, copy=bool(args.copy))
            materialized_paths.append(label_destination)

        meta_json = labels_src / f"{stem}.json"
        if meta_json.exists():
            materialized_paths.extend(
                _materialize_metadata(
                    source=meta_json,
                    destination=labels_out / meta_json.name,
                    dataset_root=dataset_root,
                    out_root=out_root,
                    split=split,
                    stem=stem,
                    copy=bool(args.copy),
                )
            )

    classes_json = labels_src / "classes.json"
    if classes_json.is_file():
        classes_destination = labels_out / "classes.json"
        _link_or_copy(classes_json, classes_destination, copy=bool(args.copy))
        materialized_paths.append(classes_destination)

    subset_txt = out_root / "subset_images.txt"
    subset_txt.write_text("".join([f"{n}\n" for n in names]), encoding="utf-8")
    materialized_paths.append(subset_txt)
    subset_sha = _sha256_lines(names)
    file_hashes = {
        str(path.relative_to(out_root)): _sha256_file(path.resolve())
        for path in sorted(set(materialized_paths))
        if path.is_file()
    }

    payload: dict[str, Any] = {
        "kind": "yolozu_subset_dataset",
        "schema_version": 1,
        "timestamp_utc": _now_utc(),
        "source": {"dataset": str(dataset_root), "split": str(split)},
        "output": {"dataset": str(out_root), "split": str(split)},
        "selection": {"strategy": str(args.strategy), "n": int(args.n), "seed": int(args.seed)},
        "images": names,
        "images_sha256": subset_sha,
        "artifacts": {
            "files": int(len(file_hashes)),
            "bytes": int(sum(path.resolve().stat().st_size for path in set(materialized_paths) if path.is_file())),
            "sha256": file_hashes,
        },
        "source_metadata_sha256": _source_metadata_hashes(dataset_root),
        "runtime_seconds": float(time.perf_counter() - started),
        "run": build_run_record(
            repo_root=repo_root,
            argv=(sys.argv[1:] if argv is None else list(argv)),
            args={"dataset": str(args.dataset), "split": args.split, "n": int(args.n), "seed": int(args.seed), "strategy": str(args.strategy)},
            dataset_root=str(dataset_root),
            extra={"subset_images_txt": str(subset_txt), "subset_sha256": subset_sha},
        ),
    }

    subset_json = out_root / "subset.json"
    subset_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
