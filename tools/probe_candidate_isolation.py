#!/usr/bin/env python3
"""Print the fail-closed candidate-isolation capability observation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.adaptive.isolation_policy import probe_candidate_isolation


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report the code-owned candidate-isolation decision and bounded host "
            "backend presence observations without executing candidate code."
        )
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _parse_args(sys.argv[1:] if argv is None else argv)
    print(
        json.dumps(
            probe_candidate_isolation(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
