"""Atomic replacement helper for bounded append-only control streams."""

from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path
from typing import Callable

__all__ = [
    "atomic_replace_control_stream",
    "read_control_stream_bytes",
    "resolve_confined_regular_file",
    "resolve_workspace_root",
]


FaultHook = Callable[[str], None]


def resolve_workspace_root(path: str | Path) -> Path:
    """Resolve one existing non-symlink workspace directory."""

    lexical = Path(os.path.abspath(Path(path)))
    if lexical.is_symlink():
        raise ValueError("workspace root cannot be a symlink")
    resolved = lexical.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("workspace root must be a directory")
    return resolved


def resolve_confined_regular_file(
    path: str | Path,
    *,
    workspace: Path,
    label: str,
) -> Path:
    """Resolve one singly linked regular file without crossing workspace."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the workspace") from exc
    current = workspace
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component")
    resolved = lexical.resolve(strict=True)
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside the workspace") from exc
    info = os.stat(resolved, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"{label} must be one singly linked regular file")
    return resolved


def read_control_stream_bytes(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    """Read a bounded singly linked regular file with identity revalidation."""
    before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"{label} must be one singly linked regular file")
    if before.st_size > maximum_bytes:
        raise ValueError(f"{label} exceeds its byte limit")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise ValueError(f"{label} changed while opening")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError(f"{label} exceeds its byte limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if total != after.st_size or identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def atomic_replace_control_stream(
    *,
    path: Path,
    observed_bytes: bytes,
    replacement_bytes: bytes,
    maximum_bytes: int,
    label: str,
    fault_hook: FaultHook | None = None,
) -> None:
    """Publish exact replacement bytes after revalidating the observed stream.

    The caller owns semantic append-only validation. This helper owns the
    no-follow, compare-before-commit, same-directory replacement boundary.
    """

    if len(replacement_bytes) > maximum_bytes:
        raise ValueError(f"{label} exceeds its byte limit")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if os.name != "posix" or not nofollow or not directory_flag:
        raise ValueError("atomic control-stream replacement requires POSIX no-follow primitives")
    parent_fd = os.open(path.parent, os.O_RDONLY | directory_flag | nofollow)
    temporary = f".{path.name}.stage.{secrets.token_hex(16)}"
    descriptor: int | None = None
    published = False
    try:
        try:
            before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            before = None
        if before is not None and (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
        ):
            raise ValueError(f"{label} must be one singly linked regular file")
        current = (
            b""
            if before is None
            else read_control_stream_bytes(
                path,
                maximum_bytes=maximum_bytes,
                label=label,
            )
        )
        if current != observed_bytes:
            raise ValueError(f"{label} changed after dry-run validation")
        if fault_hook is not None:
            fault_hook("before_stage_open")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
            0o600,
            dir_fd=parent_fd,
        )
        written = 0
        while written < len(replacement_bytes):
            count = os.write(descriptor, replacement_bytes[written:])
            if count <= 0:
                raise OSError("short control-stream write")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if fault_hook is not None:
            fault_hook("before_replace")
        try:
            latest = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            latest = None
        if (before is None) != (latest is None):
            raise ValueError(f"{label} changed before commit")
        if before is not None and latest is not None and (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            latest.st_dev,
            latest.st_ino,
            latest.st_size,
            latest.st_mtime_ns,
        ):
            raise ValueError(f"{label} identity changed before commit")
        os.replace(temporary, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        published = True
        os.fsync(parent_fd)
        if fault_hook is not None:
            fault_hook("after_replace")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if not published:
            try:
                info = os.stat(temporary, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                info = None
            if info is not None and stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                os.unlink(temporary, dir_fd=parent_fd)
        os.close(parent_fd)
