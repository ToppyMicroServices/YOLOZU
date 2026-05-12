"""Small stderr progress helpers for user-facing CLI loops."""

from __future__ import annotations

import os
import sys
import time
from typing import TextIO

__all__ = ["ProgressBar", "progress_enabled"]


def progress_enabled(explicit: bool | None = None, *, stream: TextIO | None = None) -> bool:
    """Resolve progress visibility.

    ``explicit`` comes from CLI flags. Without it, ``YOLOZU_PROGRESS`` can force
    progress on/off; otherwise progress is shown only on an interactive stderr.
    """

    if explicit is not None:
        return bool(explicit)
    env = os.environ.get("YOLOZU_PROGRESS")
    if env is not None:
        value = env.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    target = sys.stderr if stream is None else stream
    return bool(getattr(target, "isatty", lambda: False)())


class ProgressBar:
    """Dependency-free progress bar.

    Interactive terminals get an updating bar. Forced non-TTY progress prints
    one compact line per update so logs remain readable.
    """

    def __init__(
        self,
        *,
        label: str,
        total: int,
        unit: str = "item",
        enabled: bool | None = None,
        stream: TextIO | None = None,
    ) -> None:
        self.label = str(label)
        self.total = max(0, int(total))
        self.unit = str(unit)
        self.stream = sys.stderr if stream is None else stream
        self.enabled = progress_enabled(enabled, stream=self.stream)
        self.current = 0
        self._started = time.monotonic()
        self._last_len = 0
        self._interactive = bool(getattr(self.stream, "isatty", lambda: False)())

    def update(self, current: int, message: str | None = None) -> None:
        self.current = max(0, min(int(current), self.total if self.total else int(current)))
        self._render(message)

    def step(self, message: str | None = None, *, advance: int = 1) -> None:
        self.update(self.current + int(advance), message)

    def close(self, message: str = "done") -> None:
        if not self.enabled:
            return
        if self.total and self.current < self.total:
            self.current = self.total
        self._render(message)
        if self._interactive:
            self.stream.write("\n")
            self.stream.flush()

    def _render(self, message: str | None) -> None:
        if not self.enabled:
            return
        elapsed = max(0.0, time.monotonic() - self._started)
        suffix = f" {message}" if message else ""
        if self.total > 0:
            ratio = min(1.0, float(self.current) / float(self.total))
            width = 18
            filled = int(round(ratio * width))
            bar = "#" * filled + "-" * (width - filled)
            line = f"{self.label} [{bar}] {self.current}/{self.total} {self.unit}{suffix} ({elapsed:.1f}s)"
        else:
            line = f"{self.label} {self.current} {self.unit}{suffix} ({elapsed:.1f}s)"
        if self._interactive:
            pad = max(0, self._last_len - len(line))
            self.stream.write("\r" + line + (" " * pad))
            self._last_len = len(line)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()
