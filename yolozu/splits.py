"""Deterministic train/val splitting.

Uses SHA-256 hashing of file paths for reproducible,
deterministic train/val assignment.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

__all__ = ["Split", "deterministic_split_paths"]


@dataclass(frozen=True)
class Split:
    train: list[str]
    val: list[str]


def deterministic_split_paths(
    paths: Iterable[str | Path],
    *,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Split:
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError("val_fraction must be in (0,1)")

    train: list[str] = []
    val: list[str] = []
    for p in paths:
        key = str(p)
        digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
        # Use first 4 bytes as uniform bucket in [0,1).
        bucket = int.from_bytes(digest[:4], "big") / 2**32
        if bucket < float(val_fraction):
            val.append(key)
        else:
            train.append(key)
    return Split(train=train, val=val)

