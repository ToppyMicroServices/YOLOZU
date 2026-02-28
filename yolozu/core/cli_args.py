"""CLI argument parsing and path resolution helpers.

Used internally by ``yolozu.cli`` to validate flags before they reach the
core library functions.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "parse_image_size_arg",
    "require_non_negative_int",
    "require_positive_int",
    "require_non_negative_float",
    "require_float_in_range",
    "resolve_input_path",
    "resolve_output_path",
]


def parse_image_size_arg(value: str | None, *, flag_name: str = "--image-size") -> tuple[int, int] | None:
    """Parse ``'N'`` or ``'W,H'`` (or ``'WxH'``) into an ``(width, height)`` pair."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    parts = [part.strip() for part in raw.lower().replace("x", ",").split(",") if part.strip()]
    if len(parts) == 1:
        size = int(parts[0])
        if size <= 0:
            raise ValueError(f"{flag_name} expects positive integers")
        return size, size
    if len(parts) == 2:
        width = int(parts[0])
        height = int(parts[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"{flag_name} expects positive integers")
        return width, height
    raise ValueError(f"{flag_name} expects 'N' or 'W,H'")


def require_non_negative_int(value: int | None, *, flag_name: str) -> int | None:
    if value is None:
        return None
    number = int(value)
    if number < 0:
        raise ValueError(f"{flag_name} must be >= 0")
    return number


def require_positive_int(value: int | None, *, flag_name: str) -> int | None:
    if value is None:
        return None
    number = int(value)
    if number <= 0:
        raise ValueError(f"{flag_name} must be > 0")
    return number


def require_non_negative_float(value: float | None, *, flag_name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{flag_name} must be >= 0")
    return number


def require_float_in_range(
    value: float | None,
    *,
    flag_name: str,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    number = float(value)
    if number < float(minimum) or number > float(maximum):
        raise ValueError(f"{flag_name} must be in [{minimum}, {maximum}]")
    return number


def resolve_input_path(
    value: str | Path,
    *,
    cwd: Path | None = None,
    repo_root: Path | None = None,
    config_dir: Path | None = None,
) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() if cwd is None else Path(cwd)
    candidates: list[Path] = []
    if config_dir is not None:
        candidates.append(Path(config_dir) / path)
    candidates.append(cwd_path / path)
    if repo_root is not None:
        repo_candidate = Path(repo_root) / path
        if repo_candidate not in candidates:
            candidates.append(repo_candidate)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0] if candidates else path


def resolve_output_path(value: str | Path, *, cwd: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() if cwd is None else Path(cwd)
    return cwd_path / path
