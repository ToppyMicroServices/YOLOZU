#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.integrations.mcp_cli import main as _package_main

def _main() -> int:
    return _package_main()


if __name__ == "__main__":
    raise SystemExit(_main())
