"""Compatibility wrapper for the YOLOZU CLI entrypoint."""

from __future__ import annotations

from . import cli_commands as _commands
from . import cli_entry as _entry

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    return int(_entry.main(argv))


def __getattr__(name: str):  # pragma: no cover
    if hasattr(_entry, name):
        return getattr(_entry, name)
    return getattr(_commands, name)


if __name__ == "__main__":
    raise SystemExit(main())
