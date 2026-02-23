from __future__ import annotations

from typing import Any


def _short_summary(ok: bool, name: str, exit_code: int | None = None) -> str:
    suffix = "" if exit_code is None else f" (exit={exit_code})"
    return f"{name}: {'ok' if ok else 'failed'}{suffix}"


def ok_response(name: str, *, data: dict[str, Any] | None = None, exit_code: int = 0) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "tool": name,
        "summary": _short_summary(True, name, exit_code),
        "exit_code": exit_code,
    }
    if data:
        payload.update(data)
    return payload


def fail_response(name: str, *, message: str, exit_code: int = 1, exc: Exception | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "tool": name,
        "summary": _short_summary(False, name, exit_code),
        "exit_code": exit_code,
        "error": message,
    }
    if exc is not None:
        payload["error_type"] = exc.__class__.__name__
    return payload
