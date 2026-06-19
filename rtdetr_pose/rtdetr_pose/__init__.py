"""RT-DETR 6DoF pose reference-trainer package."""

from __future__ import annotations

from pathlib import Path

_outer_package = Path(__file__).resolve().parents[1]
if str(_outer_package) not in __path__:  # type: ignore[name-defined]
    __path__.append(str(_outer_package))  # type: ignore[name-defined]

__all__ = [
    "config",
    "dataset",
    "geometry",
    "losses",
    "metrics",
    "model",
    "validator",
]
__version__ = "0.1.0"
