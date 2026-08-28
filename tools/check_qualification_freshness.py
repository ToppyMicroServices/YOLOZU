#!/usr/bin/env python3
"""Repository wrapper for the canonical qualification freshness CLI."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolozu.cli_entry import main


if __name__ == "__main__":
    raise SystemExit(main(["check-qualification-freshness", *sys.argv[1:]]))
