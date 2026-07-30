import argparse
import hashlib
import json
import stat
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


HF_DATASETS_BASE = "https://huggingface.co/datasets"


DATASET_DEFAULT_ARCHIVES: dict[str, list[str]] = {
    # Common BOP dataset repos on HF use <dataset>_base.zip plus split-specific archives.
    # We keep this list minimal and allow overriding via --archives.
    "tless": ["tless_base.zip", "tless_models.zip", "tless_train_primesense.zip"],
    "lm": ["lm_base.zip", "lm_train_pbr.zip"],
}

DATASET_LICENSES: dict[str, dict[str, str]] = {
    "tless": {
        "spdx": "CC-BY-4.0",
        "source": "https://bop.felk.cvut.cz/datasets/",
        "dataset_card": "https://huggingface.co/datasets/bop-benchmark/tless",
    }
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, out_path: Path, *, force: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_symlink():
        raise ValueError(f"refusing symlink archive cache path: {out_path}")
    if out_path.exists() and not out_path.is_file():
        raise ValueError(f"refusing non-file archive cache path: {out_path}")
    if out_path.exists() and not force:
        return
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    if tmp.is_symlink():
        raise ValueError(f"refusing symlink partial archive path: {tmp}")
    if tmp.exists() and force:
        tmp.unlink()
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(out_path)


def _validate_archive_name(name: str) -> str:
    candidate = str(name).strip()
    path = PurePosixPath(candidate)
    if (
        not candidate
        or "\\" in candidate
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name != candidate
        or not candidate.lower().endswith(".zip")
    ):
        raise ValueError(f"archive must be a plain .zip filename: {name!r}")
    return candidate


def _validate_zip_members(zf: zipfile.ZipFile, *, out_dir: Path) -> None:
    root = out_dir.resolve()
    for member in zf.infolist():
        name = member.filename
        path = PurePosixPath(name)
        if not name or "\\" in name or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe zip member path: {name!r}")
        mode = (member.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            raise ValueError(f"zip symlink is not allowed: {name!r}")
        target = (root / Path(*path.parts)).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"zip member escapes output directory: {name!r}") from exc


def _extract_zip(zip_path: Path, out_dir: Path, *, force: bool) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = out_dir / f".extracted_{zip_path.name}.sha256"
    digest = _sha256(zip_path)
    if stamp.exists() and stamp.read_text(encoding="utf-8").strip() == digest and not force:
        return "complete_cached"
    with zipfile.ZipFile(zip_path, "r") as zf:
        _validate_zip_members(zf, out_dir=out_dir)
        try:
            zf.extractall(out_dir)
        except OSError as exc:
            # Some RunPod images have tight quotas on the root overlay FS; extraction can fail even
            # when targeting /workspace. Allow opting into partial extracts for smoke tests.
            allow_partial = bool(getattr(_extract_zip, "_allow_partial", False))
            if allow_partial and getattr(exc, "errno", None) in (122,):
                print(f"warning: partial extract due to quota: {exc}", file=sys.stderr)
                return "partial_quota"
            raise
    stamp.write_text(digest + "\n", encoding="utf-8")
    return "complete_extracted"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download BOP dataset archives from the bop-benchmark HuggingFace repo.")
    p.add_argument("--dataset", required=True, help="Dataset id (e.g., tless, lm).")
    p.add_argument(
        "--archives",
        default=None,
        help="Comma-separated archive filenames to download (overrides defaults).",
    )
    p.add_argument("--out", required=True, help="Output directory (will contain extracted dataset folder).")
    p.add_argument("--cache", default=None, help="Optional cache directory for zips (default: <out>/zips).")
    p.add_argument("--force", action="store_true", help="Re-download and re-extract even if present.")
    p.add_argument(
        "--allow-partial-extract",
        action="store_true",
        help="If extraction fails with a disk quota error, keep partial extraction and continue (for smoke tests).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    dataset = str(args.dataset).strip().lower()
    out_dir = Path(str(args.out)).expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    if args.cache:
        cache_dir = Path(str(args.cache)).expanduser()
        if not cache_dir.is_absolute():
            cache_dir = (Path.cwd() / cache_dir).resolve()
    else:
        cache_dir = out_dir / "zips"

    if args.archives:
        raw_archives = [a.strip() for a in str(args.archives).split(",") if a.strip()]
    else:
        raw_archives = list(DATASET_DEFAULT_ARCHIVES.get(dataset, []))
        if not raw_archives:
            raise SystemExit(
                f"no default archives known for dataset={dataset!r}. "
                "Pass --archives (comma-separated), e.g. --archives tless_base.zip,tless_train_primesense.zip"
            )
    try:
        archives = [_validate_archive_name(name) for name in raw_archives]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    archive_records = []
    for name in archives:
        url = f"{HF_DATASETS_BASE}/bop-benchmark/{dataset}/resolve/main/{name}"
        zip_path = cache_dir / name
        print(f"download: {url}")
        _download(url, zip_path, force=bool(args.force))
        print(f"extract: {zip_path} -> {out_dir}")
        setattr(_extract_zip, "_allow_partial", bool(args.allow_partial_extract))
        extraction_status = _extract_zip(zip_path, out_dir, force=bool(args.force))
        archive_records.append(
            {
                "name": name,
                "url": url,
                "cache_path": str(zip_path),
                "bytes": zip_path.stat().st_size,
                "sha256": _sha256(zip_path),
                "extraction_status": extraction_status,
            }
        )

    manifest = {
        "schema_version": 1,
        "kind": "bop_download_manifest",
        "complete": all(record["extraction_status"] != "partial_quota" for record in archive_records),
        "dataset": dataset,
        "source": f"{HF_DATASETS_BASE}/bop-benchmark/{dataset}",
        "license": DATASET_LICENSES.get(dataset),
        "archives": archive_records,
    }
    (out_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Official BOP archives are not consistent about their top-level folder.
    # T-LESS base metadata extracts under <out>/tless, while models and
    # train_primesense extract directly under <out>. Prefer the directory that
    # actually contains a split/model folder instead of assuming <out>/<id>.
    nested = out_dir / dataset
    direct_markers = [
        out_dir / "models",
        out_dir / "models_eval",
        out_dir / "models_cad",
        *sorted(out_dir.glob("train*")),
        *sorted(out_dir.glob("test*")),
        *sorted(out_dir.glob("val*")),
    ]
    nested_markers = [
        nested / "models",
        nested / "models_eval",
        nested / "models_cad",
        *sorted(nested.glob("train*")),
        *sorted(nested.glob("test*")),
        *sorted(nested.glob("val*")),
    ]
    if any(path.exists() for path in direct_markers):
        print(out_dir)
    elif any(path.exists() for path in nested_markers):
        print(nested)
    else:
        print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
