from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .core import fail_response, ok_response

_FALLBACK_ALLOWED_TOP_LEVEL = {
    "doctor",
    "validate",
    "eval-coco",
    "eval-instance-seg",
    "eval-long-tail",
    "predict-images",
    "parity",
    "calibrate",
    "test",
    "migrate",
    "train",
    "export",
}

_MAX_STDOUT_CHARS = 200_000
_MAX_STDERR_CHARS = 100_000
_DEFAULT_TIMEOUT_SEC = 600


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _allowed_from_manifest() -> set[str]:
    path = repo_root() / "tools" / "manifest.json"
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return set()
    tools = payload.get("tools") if isinstance(payload, dict) else None
    if not isinstance(tools, list):
        return set()

    allowed: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        examples = tool.get("examples")
        if not isinstance(examples, list):
            continue
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            cmd = ex.get("command")
            if not isinstance(cmd, str):
                continue
            parts = cmd.strip().split()
            if len(parts) >= 4 and parts[0].startswith("python") and parts[1] == "-m" and parts[2] in {"yolozu", "yolozu.cli"}:
                allowed.add(parts[3])
            elif len(parts) >= 2 and parts[0] == "yolozu":
                allowed.add(parts[1])
    return allowed


def _build_allowed_top_level() -> set[str]:
    manifest_allowed = _allowed_from_manifest()
    if not manifest_allowed:
        return set(_FALLBACK_ALLOWED_TOP_LEVEL)
    return manifest_allowed | _FALLBACK_ALLOWED_TOP_LEVEL


_ALLOWED_TOP_LEVEL = _build_allowed_top_level()


def _looks_like_path_token(token: str) -> bool:
    return "/" in token or token.startswith(".") or token.endswith((".json", ".yaml", ".yml", ".onnx", ".pt", ".engine"))


def _guard_path_token(token: str) -> None:
    if not _looks_like_path_token(token):
        return
    if token.startswith("~"):
        raise ValueError(f"home-dir paths are not allowed: {token}")
    p = Path(token)
    if ".." in p.parts:
        raise ValueError(f"path traversal is not allowed: {token}")
    if p.is_absolute():
        raise ValueError(f"absolute paths are not allowed: {token}")


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n...[truncated]", True


def run_cli_tool(name: str, args: list[str], *, artifacts: dict[str, str] | None = None) -> dict[str, Any]:
    if not args:
        return fail_response(name, message="empty command")
    if args[0] not in _ALLOWED_TOP_LEVEL:
        return fail_response(name, message=f"command not allowed: {args[0]}")

    try:
        for token in args:
            _guard_path_token(token)
    except ValueError as exc:
        return fail_response(name, message=str(exc), exc=exc)

    cmd = [sys.executable, "-m", "yolozu", *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root()),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_DEFAULT_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode("utf-8", errors="replace") if exc.stdout else "")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode("utf-8", errors="replace") if exc.stderr else "")
        stdout, stdout_truncated = _truncate_text(stdout, _MAX_STDOUT_CHARS)
        stderr, stderr_truncated = _truncate_text(stderr, _MAX_STDERR_CHARS)
        payload = fail_response(name, message=f"cli timeout after {_DEFAULT_TIMEOUT_SEC}s", exit_code=124)
        payload["command"] = cmd
        payload["stdout"] = stdout
        payload["stderr"] = stderr
        payload["artifacts"] = {}
        payload["limits"] = {
            "timeout_sec": _DEFAULT_TIMEOUT_SEC,
            "stdout_max_chars": _MAX_STDOUT_CHARS,
            "stderr_max_chars": _MAX_STDERR_CHARS,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }
        return payload

    stdout, stdout_truncated = _truncate_text(proc.stdout, _MAX_STDOUT_CHARS)
    stderr, stderr_truncated = _truncate_text(proc.stderr, _MAX_STDERR_CHARS)

    if proc.returncode != 0:
        payload = fail_response(
            name,
            message=f"cli failed with exit code {proc.returncode}",
            exit_code=proc.returncode,
        )
        payload["command"] = cmd
        payload["stdout"] = stdout
        payload["stderr"] = stderr
        payload["artifacts"] = {}
        payload["limits"] = {
            "timeout_sec": _DEFAULT_TIMEOUT_SEC,
            "stdout_max_chars": _MAX_STDOUT_CHARS,
            "stderr_max_chars": _MAX_STDERR_CHARS,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }
        return payload

    payload = ok_response(
        name,
        exit_code=proc.returncode,
        data={
            "command": cmd,
            "stdout": stdout,
            "stderr": stderr,
            "artifacts": {},
            "limits": {
                "timeout_sec": _DEFAULT_TIMEOUT_SEC,
                "stdout_max_chars": _MAX_STDOUT_CHARS,
                "stderr_max_chars": _MAX_STDERR_CHARS,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            },
        },
    )
    for key, raw_path in (artifacts or {}).items():
        _guard_path_token(raw_path)
        payload["artifacts"][key] = raw_path
    return payload


def run_cli_tool_redacted(name: str, args: list[str], *, artifacts: dict[str, str] | None = None) -> dict[str, Any]:
    """Run a yolozu CLI subcommand without returning stdout/stderr.

    This is intended for public-facing REST surfaces where raw process output may
    contain stack traces or sensitive environment details.
    """
    if not args:
        return fail_response(name, message="empty command")
    if args[0] not in _ALLOWED_TOP_LEVEL:
        return fail_response(name, message=f"command not allowed: {args[0]}")

    try:
        for token in args:
            _guard_path_token(token)
    except ValueError:
        return fail_response(name, message="invalid command arguments")

    cmd = [sys.executable, "-m", "yolozu", *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root()),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_DEFAULT_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired:
        payload = fail_response(name, message=f"cli timeout after {_DEFAULT_TIMEOUT_SEC}s", exit_code=124)
        payload["command"] = cmd
        payload["artifacts"] = {}
        payload["limits"] = {
            "timeout_sec": _DEFAULT_TIMEOUT_SEC,
            "stdio_redacted": True,
        }
        return payload
    except (FileNotFoundError, OSError, PermissionError, subprocess.SubprocessError):
        payload = fail_response(name, message="cli execution failed", exit_code=1)
        payload["command"] = cmd
        payload["artifacts"] = {}
        payload["limits"] = {
            "timeout_sec": _DEFAULT_TIMEOUT_SEC,
            "stdio_redacted": True,
        }
        return payload

    if proc.returncode != 0:
        payload = fail_response(
            name,
            message=f"cli failed with exit code {proc.returncode}",
            exit_code=proc.returncode,
        )
        payload["command"] = cmd
        payload["artifacts"] = {}
        payload["limits"] = {
            "timeout_sec": _DEFAULT_TIMEOUT_SEC,
            "stdio_redacted": True,
        }
        return payload

    payload = ok_response(
        name,
        exit_code=proc.returncode,
        data={
            "command": cmd,
            "artifacts": {},
            "limits": {
                "timeout_sec": _DEFAULT_TIMEOUT_SEC,
                "stdio_redacted": True,
            },
        },
    )
    for key, raw_path in (artifacts or {}).items():
        _guard_path_token(raw_path)
        payload["artifacts"][key] = raw_path
    return payload
