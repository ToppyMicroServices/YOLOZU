"""Limit Vispy's DPI probe to a deterministic value on display-less macOS.

The official BOP renderer uses an EGL surfaceless context, but Vispy still
queries Quartz for a physical display size while creating its hidden canvas.
Quartz returns a zero millimetre size in a headless session. DPI does not enter
the BOP rendering or pose-error calculations, so use a fixed canvas DPI only
when that physical display size is unavailable.
"""

from __future__ import annotations

import sys


if sys.platform == "darwin":
    try:
        import vispy.app.canvas as _canvas
        from vispy.util.dpi import _quartz

        _original_get_dpi = _quartz.get_dpi

        def _headless_safe_get_dpi(raise_error: bool = True) -> float:
            try:
                value = float(_original_get_dpi(raise_error=raise_error))
            except (ZeroDivisionError, ValueError):
                value = 0.0
            return value if value > 0.0 else 96.0

        _quartz.get_dpi = _headless_safe_get_dpi
        _canvas.get_dpi = _headless_safe_get_dpi
    except (ImportError, RuntimeError):
        pass
