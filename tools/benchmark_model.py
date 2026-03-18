#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.eval.benchmark_mode import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
