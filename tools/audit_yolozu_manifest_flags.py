#!/usr/bin/env python3
"""Audit runtime yolozu flags against tools/manifest.json inputs."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOLOZU = ROOT / "tools" / "yolozu.py"
MANIFEST = ROOT / "tools" / "manifest.json"
PYTHON = sys.executable
FLAG_PATTERN = re.compile(r"--[a-zA-Z0-9][a-zA-Z0-9_-]*")

SUBCOMMANDS = [
    "doctor",
    "sweep",
    "continual-train",
    "continual-eval",
    "export",
    "predict-images",
    "eval-keypoints",
    "eval-instance-seg",
    "calibrate",
    "eval-long-tail",
    "long-tail-recipe",
    "registry",
]
REGISTRY_SUBCOMMANDS = ["list", "show", "validate", "run"]


def extract_flags(args: list[str]) -> set[str]:
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    flags = {f for f in FLAG_PATTERN.findall(text) if f != "--help" and not f.endswith("-")}
    return flags


def main() -> int:
    runtime_flags: set[str] = set()

    for sub in SUBCOMMANDS:
        runtime_flags |= extract_flags([PYTHON, str(YOLOZU), sub, "--help"])
        if sub == "registry":
            for reg_sub in REGISTRY_SUBCOMMANDS:
                runtime_flags |= extract_flags([PYTHON, str(YOLOZU), "registry", reg_sub, "--help"])

    manifest = json.loads(MANIFEST.read_text())
    yolozu_entry = next(t for t in manifest["tools"] if t["id"] == "yolozu")
    manifest_flags = {
        i["flag"]
        for i in yolozu_entry.get("inputs", [])
        if isinstance(i, dict) and i.get("flag", "").startswith("--") and i.get("flag") != "--help"
    }

    missing = sorted(runtime_flags - manifest_flags)
    extra = sorted(manifest_flags - runtime_flags)

    print(f"runtime_flags={len(runtime_flags)}")
    print(f"manifest_flags={len(manifest_flags)}")
    print(f"missing_in_manifest={len(missing)}")
    for flag in missing:
        print(flag)
    print(f"extra_in_manifest={len(extra)}")
    for flag in extra:
        print(flag)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
