#!/usr/bin/env python3
"""Experimental part-level inspection comparison; no inference or activation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.eval.part_inspection import compare_inspection_files  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare fixed predictions before/after part-level acceptance rules (Experimental, offline)."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Existing prediction JSON (read-only; maximum 64 MiB).",
    )
    parser.add_argument(
        "--mapping",
        required=True,
        help="Part labels, splits, expected views and image mapping JSON (maximum 64 MiB).",
    )
    parser.add_argument(
        "--rules",
        required=True,
        help="Versioned region rules and calibration JSON (maximum 64 MiB).",
    )
    parser.add_argument(
        "--protocol",
        required=True,
        help="Pinned model identities, class map and evaluation settings JSON (maximum 64 MiB).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New report JSON path; existing files are never overwritten.",
    )
    args = parser.parse_args(argv)
    temporary = None
    try:
        report = compare_inspection_files(
            predictions=args.predictions,
            mapping=args.mapping,
            rules=args.rules,
            protocol=args.protocol,
        )
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Publish complete bytes atomically, without replacing inputs, existing
        # reports, symlinks or hard links. No background workers or shared state.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=".part-inspection-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.link(temporary, output)
        print(output)
        return 0
    except (ValueError, TypeError, OSError, RecursionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
