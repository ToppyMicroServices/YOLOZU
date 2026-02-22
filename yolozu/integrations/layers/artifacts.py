from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_sha() -> str | None:
    try:
        root = _repo_root()
        out = subprocess.check_output(
            ["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return None


def _manifest_hash() -> str | None:
    path = _repo_root() / "tools" / "manifest.json"
    if not path.exists():
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def collect_artifact_metadata() -> dict[str, Any]:
    return {
        "git_sha": _git_sha(),
        "manifest_sha256": _manifest_hash(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    runs_root = _repo_root() / "runs"
    if not runs_root.exists():
        return []
    candidates = [p for p in runs_root.iterdir() if p.is_dir()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict[str, Any]] = []
    for run_dir in candidates[: max(limit, 0)]:
        out.append({
            "run_id": run_dir.name,
            "path": str(run_dir),
            "mtime": run_dir.stat().st_mtime,
        })
    return out


def describe_run(run_id: str) -> dict[str, Any] | None:
    runs_root = _repo_root() / "runs"
    run_dir = runs_root / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        return None

    files: list[str] = []
    for path in run_dir.rglob("*"):
        if path.is_file():
            files.append(str(path.relative_to(_repo_root())))
    files.sort()
    return {
        "run_id": run_id,
        "path": str(run_dir),
        "files": files,
        "meta": collect_artifact_metadata(),
    }
