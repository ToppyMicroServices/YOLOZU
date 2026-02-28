from __future__ import annotations


def format_cli_error(*, code: str, message: str, hint: str | None = None) -> str:
    base = f"[{code}] {message}"
    if hint:
        return f"{base}\nHint: {hint}"
    return base
