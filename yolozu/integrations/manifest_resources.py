from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any


def workspace_root(path: str | Path | None = None) -> Path:
    """Return the explicit workspace root, or the caller's current directory."""
    root = Path.cwd() if path is None else Path(path).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def resolve_workspace_path(
    path: str | Path,
    *,
    root: str | Path | None = None,
) -> Path:
    """Resolve one caller path while keeping it inside a trusted workspace."""
    raw = str(path)
    if raw.startswith("~"):
        raise ValueError(f"home-dir paths are not allowed: {raw}")
    candidate_path = Path(raw)
    if ".." in candidate_path.parts:
        raise ValueError(f"path traversal is not allowed: {raw}")
    trusted_root = workspace_root(root)
    candidate = (
        candidate_path
        if candidate_path.is_absolute()
        else trusted_root / candidate_path
    )
    resolved = candidate.resolve()
    try:
        resolved.relative_to(trusted_root)
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {raw}") from exc
    return resolved


def load_tool_manifest(manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Load an override manifest or the copy packaged with ``yolozu``."""
    if manifest_path is None:
        text = (
            files("yolozu.data")
            .joinpath("manifest")
            .joinpath("tools_manifest.json")
            .read_text(encoding="utf-8")
        )
    else:
        path = resolve_workspace_path(manifest_path)
        text = path.read_text(encoding="utf-8")

    doc = json.loads(text)
    if not isinstance(doc, dict):
        raise ValueError("manifest must be a JSON object")
    return doc


def packaged_manifest_bytes() -> bytes:
    return (
        files("yolozu.data")
        .joinpath("manifest")
        .joinpath("tools_manifest.json")
        .read_bytes()
    )


def load_packaged_tool_reference() -> dict[str, Any]:
    """Load the generated MCP reference shipped inside the wheel."""
    text = (
        files("yolozu.data")
        .joinpath("integrations")
        .joinpath("mcp_actions_tool_reference.json")
        .read_text(encoding="utf-8")
    )
    doc = json.loads(text)
    if not isinstance(doc, dict):
        raise ValueError("packaged MCP reference must be a JSON object")
    return doc
