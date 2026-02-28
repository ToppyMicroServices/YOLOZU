"""Image-path key normalisation and alias lookups.

Provides helpers to build alias dicts (basename, stem, relative path)
so that prediction and ground-truth entries keyed by different path
formats can be matched reliably.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, TypeVar

__all__ = [
    "image_key_aliases",
    "image_basename",
    "add_image_aliases",
    "lookup_image_alias",
    "require_image_key",
]

T = TypeVar("T")


def image_key_aliases(value: Any) -> tuple[str, ...]:
    if isinstance(value, Path):
        raw = str(value)
    elif isinstance(value, str):
        raw = value
    else:
        return ()

    key = raw.strip()
    if not key:
        return ()

    out: list[str] = [key]
    key_unix = key.replace("\\", "/")
    if key_unix not in out:
        out.append(key_unix)
    base = key_unix.rsplit("/", 1)[-1]
    if base and base not in out:
        out.append(base)
    return tuple(out)


def image_basename(value: Any) -> str:
    aliases = image_key_aliases(value)
    if not aliases:
        return ""
    return aliases[-1]


def add_image_aliases(
    index: dict[str, T],
    image: Any,
    value: T,
    *,
    overwrite_primary: bool = True,
) -> bool:
    aliases = image_key_aliases(image)
    if not aliases:
        return False

    primary = aliases[0]
    if overwrite_primary or primary not in index:
        index[primary] = value
    for alias in aliases[1:]:
        index.setdefault(alias, value)
    return True


def lookup_image_alias(index: Mapping[str, T], image: Any) -> T | None:
    for key in image_key_aliases(image):
        value = index.get(key)
        if value is not None:
            return value
    return None


def require_image_key(value: Any, *, where: str) -> str:
    aliases = image_key_aliases(value)
    if not aliases:
        raise ValueError(f"{where} must be a non-empty string path")
    return aliases[0]
