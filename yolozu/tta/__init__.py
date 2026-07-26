"""Test-time adaptation (TTA/TTT) utilities.

Keep this module import lightweight so CLI/help/doctor paths do not eagerly
pull optional torch-backed runners into process startup.
"""

from __future__ import annotations

from typing import Any

from .base import TTARunner, TTAConfig, apply_tta_transform
from .config import TTTConfig
from .method_profiles import TTT_METHOD_PROFILES, get_ttt_method_profile

__all__ = [
    "TTARunner",
    "TTAConfig",
    "apply_tta_transform",
    "TTTConfig",
    "TTT_METHOD_PROFILES",
    "get_ttt_method_profile",
]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in {"TTARunner", "TTAConfig", "apply_tta_transform"}:
        from .base import TTARunner, TTAConfig, apply_tta_transform

        exports = {
            "TTARunner": TTARunner,
            "TTAConfig": TTAConfig,
            "apply_tta_transform": apply_tta_transform,
        }
        value = exports[name]
        globals()[name] = value
        return value

    if name == "TTTConfig":
        from .config import TTTConfig

        globals()[name] = TTTConfig
        return TTTConfig

    if name in {"TentConfig", "TentRunner"}:
        from .tent import TentConfig, TentRunner

        exports = {
            "TentConfig": TentConfig,
            "TentRunner": TentRunner,
        }
        value = exports[name]
        globals()[name] = value
        return value

    if name in {"TTTReport", "run_ttt"}:
        from .integration import TTTReport, run_ttt

        exports = {
            "TTTReport": TTTReport,
            "run_ttt": run_ttt,
        }
        value = exports[name]
        globals()[name] = value
        return value

    raise AttributeError(name)


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(set(globals().keys()) | set(__all__))
