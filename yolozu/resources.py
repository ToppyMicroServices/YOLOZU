"""Shipped data-resource accessors.

Provides ``importlib.resources``-based helpers to read bundled files
under ``yolozu/data/``.
"""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

__all__ = ["list_resource_paths", "read_text", "as_file_path", "copy_to"]


def _files():
    from importlib.resources import files

    return files("yolozu.data")


def list_resource_paths() -> list[str]:
    """Return all shipped resource paths under yolozu/data (relative, posix)."""

    root = _files()

    out: list[str] = []

    def _walk(node, prefix: str) -> None:
        for child in node.iterdir():
            name = child.name
            if name in {"__pycache__", ".DS_Store"}:
                continue
            if name.endswith(".pyc"):
                continue
            if name == "__init__.py":
                continue
            rel = f"{prefix}{name}" if prefix else name
            if child.is_dir():
                _walk(child, rel + "/")
            else:
                out.append(rel)

    _walk(root, "")
    return sorted(out)


def read_text(rel_path: str, *, encoding: str = "utf-8") -> str:
    node = _files().joinpath(rel_path)
    return node.read_text(encoding=encoding)


@contextmanager
def as_file_path(rel_path: str) -> Iterator[Path]:
    """Materialize a resource as a filesystem path (temp dir if needed)."""

    from importlib.resources import as_file

    node = _files().joinpath(rel_path)
    with as_file(node) as p:
        yield Path(p)


def copy_to(rel_path: str, *, output: str | Path, force: bool = False) -> Path:
    out_path = Path(output)
    if out_path.exists() and not force:
        raise FileExistsError(f"output exists: {out_path} (use --force to overwrite)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with as_file_path(rel_path) as src:
        shutil.copyfile(str(src), str(out_path))
    return out_path
