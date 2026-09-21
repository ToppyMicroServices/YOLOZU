#!/usr/bin/env python3
"""Prepare a local plugin copy for an explicitly selected, already installed runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_NAME = "yolozu-image-service"
SOURCE = REPO_ROOT / "plugins" / PLUGIN_NAME
FILES = (
    ".codex-plugin/plugin.json", ".mcp.json",
    "skills/yolozu-image-service/SKILL.md",
    "scripts/image_service_client.py",
)
EXPECTED_TOOLS = sorted((
    "image_service_capabilities", "put_image_asset", "submit_image_job",
    "get_image_job", "cancel_image_job",
))
RUNTIME_CHECK = (
    "import asyncio,json; from yolozu.integrations.mcp_server import service_app; "
    "print(json.dumps(sorted(t.name for t in asyncio.run(service_app.list_tools()))))"
)


def prepare(output: Path, python: Path) -> dict:
    # Do not resolve the interpreter symlink: doing so loses virtualenv identity.
    python = Path(os.path.abspath(python.expanduser()))
    output = output.expanduser().absolute()
    if output.name != PLUGIN_NAME:
        raise ValueError(f"--output must end with {PLUGIN_NAME}")
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists; choose a new destination (no overwrite)")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("--python must name an executable Python interpreter")
    if output.resolve().is_relative_to(SOURCE.resolve()):
        raise ValueError("output must be outside the source plugin")
    # No package installation, credential forwarding, or network request.
    env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR") if key in os.environ}
    checked = subprocess.run(
        [str(python), "-I", "-c", RUNTIME_CHECK], cwd=SOURCE,
        env=env, capture_output=True, text=True, timeout=30, check=False,
    )
    if checked.returncode != 0:
        raise ValueError("selected Python cannot import the bounded yolozu[mcp] service; install the reviewed runtime first")
    if json.loads(checked.stdout) != EXPECTED_TOOLS:
        raise ValueError("selected runtime does not expose exactly the five image-service tools")
    for relative in FILES:
        path = SOURCE / relative
        if not path.is_file() or any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("source plugin must contain regular files without symlink components")
    config = json.loads((SOURCE / ".mcp.json").read_text(encoding="utf-8"))
    config["mcpServers"][PLUGIN_NAME]["command"] = str(python)
    # Build only the reviewed file set; never copy images, weights, or local caches.
    output.mkdir(parents=True, exist_ok=False)
    for relative in FILES:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / relative, target)
    shutil.copyfile(REPO_ROOT / "LICENSE", output / "LICENSE")
    (output / ".mcp.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "plugin": str(output), "python": str(python), "tools": EXPECTED_TOOLS,
            "installed_in_host": False, "models_qualified": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory ending in yolozu-image-service; never overwritten.")
    parser.add_argument("--python", type=Path, required=True, help="Trusted Python executable with this revision of yolozu[mcp] already installed.")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.output, args.python)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
