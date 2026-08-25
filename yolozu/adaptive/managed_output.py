"""Bounded, fail-closed transactions for repository-owned output trees.

The implementation intentionally supports only POSIX directory-descriptor
operations.  It never falls back to path-based mutation on platforms that do
not provide the required ``dir_fd`` and no-follow primitives.
"""

from __future__ import annotations

import errno
import hashlib
import os
import secrets
import stat
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Iterator, Literal, Mapping

from .canonical import canonical_json_v1
from .control_records import load_bounded_json_bytes

__all__ = [
    "ManagedOutputCapabilities",
    "ManagedOutputError",
    "ManagedOutputLimits",
    "ManagedOutputTransaction",
    "RecoveryResult",
    "recover_managed_output",
]


_CHECKSUMS_NAME = "checksums.json"
_MANIFEST_SCHEMA_VERSION = 1
_MARKER_SCHEMA_VERSION = 1
_READ_CHUNK_BYTES = 1024 * 1024
_NAME_BYTES_MAX = 255
_RELATIVE_PATH_BYTES_MAX = 4096
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)

FaultHook = Callable[[str], None]
RecoveryResult = Literal["committed", "rolled_back", "no_recovery_needed"]


class ManagedOutputError(ValueError):
    """One stable fail-closed managed-output failure."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class ManagedOutputLimits:
    """Inclusive limits for a complete managed tree, including its manifest."""

    max_files: int
    max_file_bytes: int
    max_total_bytes: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_files", self.max_files),
            ("max_file_bytes", self.max_file_bytes),
            ("max_total_bytes", self.max_total_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True)
class ManagedOutputCapabilities:
    """Observed capability scope; this is not a universal durability claim."""

    platform: str
    directory_relative_no_follow: bool
    same_filesystem_rename_probed: bool
    same_filesystem_atomic_visibility: bool
    directory_fsync_supported: bool
    power_loss_durability: Literal["best_effort", "unsupported"]


@dataclass(frozen=True)
class _Identity:
    device: int
    inode: int
    mode_type: int
    link_count: int


@dataclass(frozen=True)
class _ManifestEntry:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class _ValidatedTree:
    identity: _Identity
    manifest_identity: _Identity
    manifest_sha256: str
    entries: tuple[_ManifestEntry, ...]
    file_identities: Mapping[str, _Identity]
    directory_identities: Mapping[str, _Identity]


def _fail(code: str, detail: str) -> ManagedOutputError:
    return ManagedOutputError(code, detail)


def _identity(info: os.stat_result) -> _Identity:
    return _Identity(
        device=info.st_dev,
        inode=info.st_ino,
        mode_type=stat.S_IFMT(info.st_mode),
        link_count=info.st_nlink,
    )


def _same_identity(left: _Identity, right: _Identity) -> bool:
    if (
        left.device != right.device
        or left.inode != right.inode
        or left.mode_type != right.mode_type
    ):
        return False
    # Directory link counts can change when children are added or removed.
    # Regular-file link counts are security-relevant and must stay exactly one.
    return left.mode_type == stat.S_IFDIR or left.link_count == right.link_count


def _require_posix_primitives() -> None:
    if os.name != "posix" or not _NOFOLLOW or not _DIRECTORY:
        raise _fail(
            "platform_unsupported",
            "secure directory-relative no-follow mutation is unavailable",
        )
    required = (os.open, os.stat, os.mkdir, os.rename, os.unlink, os.rmdir)
    if any(function not in os.supports_dir_fd for function in required):
        raise _fail(
            "platform_unsupported",
            "required dir_fd operations are unavailable",
        )


def _step(hook: FaultHook | None, name: str) -> None:
    if hook is not None:
        hook(name)


def _validate_component(component: str, *, field: str) -> str:
    if (
        not component
        or component in {".", ".."}
        or "/" in component
        or "\x00" in component
        or len(component.encode("utf-8")) > _NAME_BYTES_MAX
    ):
        raise _fail("path_invalid", f"{field} contains an invalid component")
    return component


def _relative_parts(value: str | PurePosixPath, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (str, PurePosixPath)):
        raise TypeError(f"{field} must be a relative POSIX path")
    raw = str(value)
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or raw.endswith("/")
        or "\\" in raw
        or len(raw.encode("utf-8")) > _RELATIVE_PATH_BYTES_MAX
    ):
        raise _fail("path_invalid", f"{field} must be a bounded relative POSIX path")
    parts = tuple(path.parts)
    if not parts:
        raise _fail("path_invalid", f"{field} is empty")
    for component in parts:
        _validate_component(component, field=field)
    return parts


def _open_directory_at(parent_fd: int, name: str) -> int:
    try:
        return os.open(
            name,
            os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise _fail("unsafe_directory", f"{name!r} is not a safe directory") from exc


@contextmanager
def _opened_directory_at(parent_fd: int, name: str) -> Iterator[int]:
    descriptor = _open_directory_at(parent_fd, name)
    try:
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _opened_root_directory(path: Path) -> Iterator[int]:
    descriptor = os.open(
        path,
        os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
    )
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def _lstat_at(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _assert_identity_at(
    parent_fd: int,
    name: str,
    expected: _Identity,
    *,
    regular: bool | None = None,
) -> os.stat_result:
    info = _lstat_at(parent_fd, name)
    if info is None or not _same_identity(_identity(info), expected):
        raise _fail("identity_changed", f"{name!r} changed after validation")
    if regular is True and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
        raise _fail("unsafe_file", f"{name!r} is not a singly linked regular file")
    if regular is False and not stat.S_ISDIR(info.st_mode):
        raise _fail("unsafe_directory", f"{name!r} is not a directory")
    return info


class _ParentGuard:
    def __init__(self, root: Path, parent_parts: tuple[str, ...]) -> None:
        _require_posix_primitives()
        root_path = root.absolute()
        try:
            root_info = os.stat(root_path, follow_symlinks=False)
        except OSError as exc:
            raise _fail("root_invalid", "approved root does not exist") from exc
        if not stat.S_ISDIR(root_info.st_mode):
            raise _fail("root_invalid", "approved root must be a non-symlink directory")
        descriptor_stack = ExitStack()
        try:
            try:
                # ExitStack owns this descriptor until _ParentGuard.close().
                # codeql[py/file-not-closed]
                root_fd = descriptor_stack.enter_context(
                    _opened_root_directory(root_path)
                )
            except OSError as exc:
                raise _fail("root_invalid", "approved root could not be pinned") from exc
            identities: list[_Identity] = [_identity(os.fstat(root_fd))]
            if not _same_identity(identities[0], _identity(root_info)):
                raise _fail("root_changed", "approved root changed while it was opened")
            current_fd = root_fd
            for component in parent_parts:
                # ExitStack owns every pinned parent descriptor until close().
                # codeql[py/file-not-closed]
                next_fd = descriptor_stack.enter_context(
                    _opened_directory_at(current_fd, component)
                )
                identities.append(_identity(os.fstat(next_fd)))
                current_fd = next_fd
        except Exception:
            descriptor_stack.close()
            raise
        self.root_path = root_path
        self.parent_parts = parent_parts
        self._descriptor_stack = descriptor_stack.pop_all()
        self._identities = tuple(identities)
        self.root_fd = root_fd
        self.parent_fd = current_fd

    @property
    def parent_identity(self) -> _Identity:
        return self._identities[-1]

    def revalidate(self) -> None:
        try:
            root_info = os.stat(self.root_path, follow_symlinks=False)
        except OSError as exc:
            raise _fail("parent_changed", "approved root is no longer addressable") from exc
        if not _same_identity(_identity(root_info), self._identities[0]):
            raise _fail("parent_changed", "approved root identity changed")
        current_fd = self.root_fd
        with ExitStack() as temporary:
            for index, component in enumerate(self.parent_parts, start=1):
                # The surrounding ExitStack closes this temporary descriptor.
                # codeql[py/file-not-closed]
                next_fd = temporary.enter_context(
                    _opened_directory_at(current_fd, component)
                )
                if not _same_identity(
                    _identity(os.fstat(next_fd)), self._identities[index]
                ):
                    raise _fail("parent_changed", "destination parent chain changed")
                current_fd = next_fd
            if not _same_identity(
                _identity(os.fstat(self.parent_fd)), self.parent_identity
            ):
                raise _fail("parent_changed", "pinned destination parent changed")

    def close(self) -> None:
        descriptor_stack = self._descriptor_stack
        self._descriptor_stack = ExitStack()
        descriptor_stack.close()


def _read_regular_at(
    parent_fd: int,
    name: str,
    *,
    maximum_bytes: int,
    expected_identity: _Identity | None = None,
) -> tuple[bytes, _Identity]:
    before = _lstat_at(parent_fd, name)
    if before is None or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise _fail("unsafe_file", f"{name!r} is not a singly linked regular file")
    before_identity = _identity(before)
    if expected_identity is not None and not _same_identity(
        before_identity, expected_identity
    ):
        raise _fail("identity_changed", f"{name!r} changed after validation")
    if before.st_size > maximum_bytes:
        raise _fail("file_limit_exceeded", f"{name!r} exceeds its byte limit")
    descriptor = os.open(
        name,
        os.O_RDONLY | _NOFOLLOW | _CLOEXEC,
        dir_fd=parent_fd,
    )
    try:
        opened = os.fstat(descriptor)
        if not _same_identity(_identity(opened), before_identity):
            raise _fail("identity_changed", f"{name!r} changed while it was opened")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, maximum_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise _fail("file_limit_exceeded", f"{name!r} exceeds its byte limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if not _same_identity(_identity(after), before_identity) or after.st_size != total:
            raise _fail("identity_changed", f"{name!r} changed while it was read")
    finally:
        os.close(descriptor)
    _assert_identity_at(parent_fd, name, before_identity, regular=True)
    return b"".join(chunks), before_identity


def _open_relative_parent(base_fd: int, parts: tuple[str, ...]) -> tuple[int, list[int]]:
    current_fd = base_fd
    opened: list[int] = []
    for component in parts[:-1]:
        current_fd = _open_directory_at(current_fd, component)
        opened.append(current_fd)
    return current_fd, opened


def _close_all(descriptors: Iterable[int]) -> None:
    for descriptor in reversed(tuple(descriptors)):
        try:
            os.close(descriptor)
        except OSError:
            # Cleanup can revisit a descriptor after an earlier failure closed it.
            pass


def _sha256_regular_at(
    tree_fd: int,
    parts: tuple[str, ...],
    *,
    maximum_bytes: int,
) -> tuple[int, str, _Identity]:
    parent_fd, opened = _open_relative_parent(tree_fd, parts)
    try:
        data, identity = _read_regular_at(
            parent_fd,
            parts[-1],
            maximum_bytes=maximum_bytes,
        )
        return len(data), hashlib.sha256(data).hexdigest(), identity
    finally:
        _close_all(opened)


def _parse_manifest(data: bytes, *, limits: ManagedOutputLimits) -> tuple[_ManifestEntry, ...]:
    value = load_bounded_json_bytes(data, label="managed checksums manifest")
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "files",
        "expected_paths",
        "file_count",
        "total_bytes",
    }:
        raise _fail("manifest_invalid", "manifest fields do not match v1")
    if value["schema_version"] != _MANIFEST_SCHEMA_VERSION:
        raise _fail("manifest_invalid", "unsupported manifest schema version")
    raw_files = value["files"]
    raw_paths = value["expected_paths"]
    if not isinstance(raw_files, list) or not isinstance(raw_paths, list):
        raise _fail("manifest_invalid", "files and expected_paths must be arrays")
    if len(raw_files) + 1 > limits.max_files:
        raise _fail("file_count_limit_exceeded", "managed tree exceeds file limit")
    entries: list[_ManifestEntry] = []
    for raw in raw_files:
        if not isinstance(raw, dict) or set(raw) != {"path", "size_bytes", "sha256"}:
            raise _fail("manifest_invalid", "manifest entry fields do not match v1")
        path = raw["path"]
        parts = _relative_parts(path, field="manifest path")
        if "/".join(parts) == _CHECKSUMS_NAME:
            raise _fail("manifest_self_entry", "checksums.json must not list itself")
        size = raw["size_bytes"]
        digest = raw["sha256"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or size > limits.max_file_bytes
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise _fail("manifest_invalid", f"invalid metadata for {path!r}")
        entries.append(_ManifestEntry(path="/".join(parts), size_bytes=size, sha256=digest))
    paths = [entry.path for entry in entries]
    expected_order = sorted(paths, key=lambda item: item.encode("utf-8"))
    if paths != expected_order or len(set(paths)) != len(paths):
        raise _fail("manifest_invalid", "manifest paths must be unique and byte-sorted")
    if raw_paths != paths:
        raise _fail("manifest_invalid", "expected_paths must exactly match files")
    if value["file_count"] != len(entries):
        raise _fail("manifest_invalid", "file_count does not match files")
    total = sum(entry.size_bytes for entry in entries)
    if value["total_bytes"] != total:
        raise _fail("manifest_invalid", "total_bytes does not match files")
    if total + len(data) > limits.max_total_bytes:
        raise _fail("total_limit_exceeded", "managed tree exceeds total byte limit")
    return tuple(entries)


def _walk_tree(
    directory_fd: int,
    *,
    prefix: tuple[str, ...] = (),
) -> tuple[set[str], dict[str, _Identity]]:
    files: set[str] = set()
    directories: dict[str, _Identity] = {}
    for name in os.listdir(directory_fd):
        _validate_component(name, field="managed tree entry")
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        parts = prefix + (name,)
        relative = "/".join(parts)
        if stat.S_ISREG(info.st_mode):
            if info.st_nlink != 1:
                raise _fail("unsafe_file", f"{relative!r} is hardlinked")
            files.add(relative)
        elif stat.S_ISDIR(info.st_mode):
            child_fd = _open_directory_at(directory_fd, name)
            try:
                directories[relative] = _identity(os.fstat(child_fd))
                child_files, child_directories = _walk_tree(child_fd, prefix=parts)
                files.update(child_files)
                directories.update(child_directories)
            finally:
                os.close(child_fd)
        else:
            raise _fail("unsafe_tree_entry", f"{relative!r} is not regular or directory")
    return files, directories


def _validate_tree_fd(
    tree_fd: int,
    *,
    limits: ManagedOutputLimits,
    parent_device: int,
    expected_manifest_sha256: str | None = None,
) -> _ValidatedTree:
    tree_info = os.fstat(tree_fd)
    if not stat.S_ISDIR(tree_info.st_mode):
        raise _fail("unsafe_tree", "managed output is not a directory")
    tree_identity = _identity(tree_info)
    if tree_identity.device != parent_device:
        raise _fail("cross_filesystem", "managed output is not on the parent filesystem")
    manifest, manifest_identity = _read_regular_at(
        tree_fd,
        _CHECKSUMS_NAME,
        maximum_bytes=min(limits.max_file_bytes, 4 * 1024 * 1024),
    )
    manifest_sha = hashlib.sha256(manifest).hexdigest()
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise _fail("manifest_changed", "manifest digest does not match the recovery marker")
    entries = _parse_manifest(manifest, limits=limits)
    actual_files, directory_identities = _walk_tree(tree_fd)
    expected_files = {entry.path for entry in entries} | {_CHECKSUMS_NAME}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise _fail("tree_membership_mismatch", f"missing={missing}, extra={extra}")
    file_identities: dict[str, _Identity] = {}
    for entry in entries:
        size, digest, identity = _sha256_regular_at(
            tree_fd,
            _relative_parts(entry.path, field="manifest path"),
            maximum_bytes=limits.max_file_bytes,
        )
        if size != entry.size_bytes or digest != entry.sha256:
            raise _fail("checksum_mismatch", f"{entry.path!r} does not match manifest")
        file_identities[entry.path] = identity
    _assert_identity_at(tree_fd, _CHECKSUMS_NAME, manifest_identity, regular=True)
    return _ValidatedTree(
        identity=tree_identity,
        manifest_identity=manifest_identity,
        manifest_sha256=manifest_sha,
        entries=entries,
        file_identities=file_identities,
        directory_identities=directory_identities,
    )


def _validate_named_tree(
    parent_fd: int,
    name: str,
    *,
    limits: ManagedOutputLimits,
    expected_manifest_sha256: str | None = None,
) -> _ValidatedTree:
    descriptor = _open_directory_at(parent_fd, name)
    try:
        result = _validate_tree_fd(
            descriptor,
            limits=limits,
            parent_device=os.fstat(parent_fd).st_dev,
            expected_manifest_sha256=expected_manifest_sha256,
        )
    finally:
        os.close(descriptor)
    _assert_identity_at(parent_fd, name, result.identity, regular=False)
    return result


def _cleanup_tree(
    guard: _ParentGuard,
    name: str,
    tree: _ValidatedTree,
    *,
    hook: FaultHook | None,
) -> None:
    guard.revalidate()
    _assert_identity_at(guard.parent_fd, name, tree.identity, regular=False)
    tree_fd = _open_directory_at(guard.parent_fd, name)
    try:
        for entry in tree.entries:
            parts = _relative_parts(entry.path, field="manifest path")
            parent_fd, opened = _open_relative_parent(tree_fd, parts)
            try:
                guard.revalidate()
                _assert_identity_at(
                    parent_fd,
                    parts[-1],
                    tree.file_identities[entry.path],
                    regular=True,
                )
                _step(hook, f"before_cleanup_file:{entry.path}")
                guard.revalidate()
                _assert_identity_at(
                    parent_fd,
                    parts[-1],
                    tree.file_identities[entry.path],
                    regular=True,
                )
                os.unlink(parts[-1], dir_fd=parent_fd)
                _step(hook, f"after_cleanup_file:{entry.path}")
            finally:
                _close_all(opened)
        guard.revalidate()
        _assert_identity_at(
            tree_fd,
            _CHECKSUMS_NAME,
            tree.manifest_identity,
            regular=True,
        )
        _step(hook, "before_cleanup_manifest")
        guard.revalidate()
        _assert_identity_at(
            tree_fd,
            _CHECKSUMS_NAME,
            tree.manifest_identity,
            regular=True,
        )
        os.unlink(_CHECKSUMS_NAME, dir_fd=tree_fd)
        _step(hook, "after_cleanup_manifest")
        for relative in sorted(
            tree.directory_identities,
            key=lambda item: (item.count("/"), item.encode("utf-8")),
            reverse=True,
        ):
            parts = _relative_parts(relative, field="managed directory")
            parent_fd, opened = _open_relative_parent(tree_fd, parts)
            try:
                guard.revalidate()
                _assert_identity_at(
                    parent_fd,
                    parts[-1],
                    tree.directory_identities[relative],
                    regular=False,
                )
                _step(hook, f"before_cleanup_directory:{relative}")
                guard.revalidate()
                _assert_identity_at(
                    parent_fd,
                    parts[-1],
                    tree.directory_identities[relative],
                    regular=False,
                )
                os.rmdir(parts[-1], dir_fd=parent_fd)
                _step(hook, f"after_cleanup_directory:{relative}")
            finally:
                _close_all(opened)
    finally:
        os.close(tree_fd)
    guard.revalidate()
    _assert_identity_at(guard.parent_fd, name, tree.identity, regular=False)
    _step(hook, f"before_cleanup_tree:{name}")
    guard.revalidate()
    _assert_identity_at(guard.parent_fd, name, tree.identity, regular=False)
    os.rmdir(name, dir_fd=guard.parent_fd)
    _step(hook, f"after_cleanup_tree:{name}")


def _directory_fsync(fd: int, *, hook: FaultHook | None, label: str) -> bool:
    _step(hook, f"before_fsync_directory:{label}")
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno in {errno.EINVAL, errno.ENOTSUP, errno.EBADF}:
            return False
        raise
    finally:
        _step(hook, f"after_fsync_directory:{label}")
    return True


def _write_exclusive_at(
    parent_fd: int,
    name: str,
    data: bytes,
    *,
    hook: FaultHook | None,
    label: str,
) -> _Identity:
    _step(hook, f"before_open:{label}")
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC,
            0o600,
            dir_fd=parent_fd,
        )
    except FileExistsError as exc:
        raise _fail("name_collision", f"exclusive {label} name collided") from exc
    created_identity = _identity(os.fstat(descriptor))
    completed = False
    try:
        written = 0
        while written < len(data):
            _step(hook, f"before_write:{label}")
            count = os.write(descriptor, data[written:])
            if count <= 0:
                raise OSError("short write")
            written += count
            _step(hook, f"after_write:{label}")
        _step(hook, f"before_fsync_file:{label}")
        os.fsync(descriptor)
        _step(hook, f"after_fsync_file:{label}")
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != len(data):
            raise _fail("write_invalid", f"{label} did not produce one exact regular file")
        identity = _identity(info)
        completed = True
    finally:
        os.close(descriptor)
        if not completed:
            current = _lstat_at(parent_fd, name)
            if (
                current is not None
                and stat.S_ISREG(current.st_mode)
                and current.st_nlink == 1
                and _same_identity(_identity(current), created_identity)
            ):
                os.unlink(name, dir_fd=parent_fd)
    _assert_identity_at(parent_fd, name, identity, regular=True)
    return identity


def _marker_name(destination_name: str) -> str:
    return f".{destination_name}.yolozu-output-transaction.json"


def _probe_same_directory_rename(
    guard: _ParentGuard,
    *,
    hook: FaultHook | None,
) -> bool:
    token = secrets.token_hex(12)
    source = f".{token}.rename-probe-source"
    target = f".{token}.rename-probe-target"
    source_identity: _Identity | None = None
    try:
        source_identity = _write_exclusive_at(
            guard.parent_fd,
            source,
            b"probe",
            hook=hook,
            label="rename_probe",
        )
        guard.revalidate()
        _assert_identity_at(guard.parent_fd, source, source_identity, regular=True)
        if _lstat_at(guard.parent_fd, target) is not None:
            raise _fail("name_collision", "rename probe target collided")
        os.rename(source, target, src_dir_fd=guard.parent_fd, dst_dir_fd=guard.parent_fd)
        _assert_identity_at(guard.parent_fd, target, source_identity, regular=True)
        os.unlink(target, dir_fd=guard.parent_fd)
        return True
    finally:
        for name in (source, target):
            info = _lstat_at(guard.parent_fd, name)
            if info is not None and stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                if source_identity is None or _same_identity(
                    _identity(info), source_identity
                ):
                    os.unlink(name, dir_fd=guard.parent_fd)


class ManagedOutputTransaction:
    """Write and atomically publish one exact, bounded managed directory tree."""

    def __init__(
        self,
        *,
        root: Path,
        destination: str | PurePosixPath,
        declared_paths: Iterable[str | PurePosixPath],
        limits: ManagedOutputLimits,
        force: bool = False,
        fault_hook: FaultHook | None = None,
    ) -> None:
        destination_parts = _relative_parts(destination, field="destination")
        destination_name = destination_parts[-1]
        if destination_name.startswith("."):
            raise _fail("path_invalid", "destination basename must not be hidden")
        normalized: list[str] = []
        for item in declared_paths:
            parts = _relative_parts(item, field="declared output path")
            relative = "/".join(parts)
            if relative == _CHECKSUMS_NAME:
                raise _fail("manifest_self_entry", "checksums.json is implicit")
            normalized.append(relative)
        if not normalized or len(set(normalized)) != len(normalized):
            raise _fail("declared_paths_invalid", "declared paths must be nonempty and unique")
        normalized.sort(key=lambda item: item.encode("utf-8"))
        if len(normalized) + 1 > limits.max_files:
            raise _fail("file_count_limit_exceeded", "declared tree exceeds file limit")
        self.root = Path(root)
        self.destination_parts = destination_parts
        self.destination_name = destination_name
        self.declared_paths = tuple(normalized)
        self.limits = limits
        self.force = bool(force)
        self.fault_hook = fault_hook
        self._stage_name = f".{destination_name}.stage.{secrets.token_hex(16)}"
        self._backup_name = f".{destination_name}.backup.{secrets.token_hex(16)}"
        self._marker_name = _marker_name(destination_name)
        for derived_name in (
            self._stage_name,
            self._backup_name,
            self._marker_name,
        ):
            _validate_component(derived_name, field="transaction sibling name")
        self._guard = _ParentGuard(self.root, destination_parts[:-1])
        self._stage_fd: int | None = None
        self._stage_identity: _Identity | None = None
        self._written: dict[str, _ManifestEntry] = {}
        self._written_identities: dict[str, _Identity] = {}
        self._created_directories: dict[str, _Identity] = {}
        self._manifest_identity: _Identity | None = None
        self._marker_identity: _Identity | None = None
        self._old_tree: _ValidatedTree | None = None
        self._finished = False
        self._published = False
        try:
            self._begin()
        except Exception:
            try:
                self.abort()
            finally:
                self.close()
            raise

    @property
    def capabilities(self) -> ManagedOutputCapabilities:
        return self._capabilities

    @property
    def published(self) -> bool:
        return self._published

    def _begin(self) -> None:
        guard = self._guard
        guard.revalidate()
        if _lstat_at(guard.parent_fd, self._marker_name) is not None:
            raise _fail(
                "recovery_required",
                "an existing recovery marker must be handled before writing",
            )
        for name in (self._stage_name, self._backup_name):
            if _lstat_at(guard.parent_fd, name) is not None:
                raise _fail("name_collision", "exclusive sibling name collided")
        destination_info = _lstat_at(guard.parent_fd, self.destination_name)
        if destination_info is not None:
            if not self.force:
                raise _fail("destination_exists", "existing destination requires force")
            self._old_tree = _validate_named_tree(
                guard.parent_fd,
                self.destination_name,
                limits=self.limits,
            )
        rename_probed = _probe_same_directory_rename(guard, hook=self.fault_hook)
        fsync_supported = _directory_fsync(
            guard.parent_fd,
            hook=self.fault_hook,
            label="capability_probe",
        )
        self._capabilities = ManagedOutputCapabilities(
            platform=os.name,
            directory_relative_no_follow=True,
            same_filesystem_rename_probed=rename_probed,
            same_filesystem_atomic_visibility=rename_probed,
            directory_fsync_supported=fsync_supported,
            power_loss_durability="best_effort" if fsync_supported else "unsupported",
        )
        _step(self.fault_hook, "before_create_stage")
        try:
            os.mkdir(self._stage_name, mode=0o700, dir_fd=guard.parent_fd)
        except FileExistsError as exc:
            raise _fail("name_collision", "exclusive stage name collided") from exc
        self._stage_fd = _open_directory_at(guard.parent_fd, self._stage_name)
        self._stage_identity = _identity(os.fstat(self._stage_fd))
        _step(self.fault_hook, "after_create_stage")
        if self._stage_identity.device != guard.parent_identity.device:
            raise _fail("cross_filesystem", "stage is not on the destination filesystem")
        _assert_identity_at(
            guard.parent_fd,
            self._stage_name,
            self._stage_identity,
            regular=False,
        )

    def __enter__(self) -> ManagedOutputTransaction:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._finished and not self._published:
            self.abort()
        self.close()

    def _stage_parent(self, relative: str) -> tuple[int, list[int]]:
        if self._stage_fd is None:
            raise _fail("transaction_closed", "stage is unavailable")
        parts = _relative_parts(relative, field="output path")
        current_fd = self._stage_fd
        opened: list[int] = []
        prefix: list[str] = []
        for component in parts[:-1]:
            prefix.append(component)
            directory = "/".join(prefix)
            info = _lstat_at(current_fd, component)
            if info is None:
                _step(self.fault_hook, f"before_create_directory:{directory}")
                os.mkdir(component, mode=0o700, dir_fd=current_fd)
            next_fd = _open_directory_at(current_fd, component)
            identity = _identity(os.fstat(next_fd))
            known = self._created_directories.get(directory)
            if known is not None and not _same_identity(known, identity):
                os.close(next_fd)
                _close_all(opened)
                raise _fail("identity_changed", f"staged directory {directory!r} changed")
            self._created_directories[directory] = identity
            if info is None:
                _step(self.fault_hook, f"after_create_directory:{directory}")
            opened.append(next_fd)
            current_fd = next_fd
        return current_fd, opened

    def write_bytes(self, relative_path: str | PurePosixPath, data: bytes) -> None:
        if self._finished or self._published:
            raise _fail("transaction_finished", "transaction no longer accepts writes")
        parts = _relative_parts(relative_path, field="output path")
        relative = "/".join(parts)
        if relative not in self.declared_paths:
            raise _fail("undeclared_output", f"{relative!r} was not declared")
        if relative in self._written:
            raise _fail("duplicate_output", f"{relative!r} was already written")
        if not isinstance(data, bytes):
            raise TypeError("managed output data must be bytes")
        if len(data) > self.limits.max_file_bytes:
            raise _fail("file_limit_exceeded", f"{relative!r} exceeds file byte limit")
        if sum(item.size_bytes for item in self._written.values()) + len(data) > self.limits.max_total_bytes:
            raise _fail("total_limit_exceeded", "managed output exceeds total byte limit")
        parent_fd, opened = self._stage_parent(relative)
        try:
            identity = _write_exclusive_at(
                parent_fd,
                parts[-1],
                data,
                hook=self.fault_hook,
                label=f"output:{relative}",
            )
        finally:
            _close_all(opened)
        self._written[relative] = _ManifestEntry(
            path=relative,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        self._written_identities[relative] = identity

    def _manifest_bytes(self) -> bytes:
        missing = sorted(set(self.declared_paths) - set(self._written))
        if missing:
            raise _fail("incomplete_output", f"declared outputs were not written: {missing}")
        entries = [self._written[path] for path in self.declared_paths]
        value = {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "files": [
                {
                    "path": entry.path,
                    "size_bytes": entry.size_bytes,
                    "sha256": entry.sha256,
                }
                for entry in entries
            ],
            "expected_paths": list(self.declared_paths),
            "file_count": len(entries),
            "total_bytes": sum(entry.size_bytes for entry in entries),
        }
        data = canonical_json_v1(value)
        if len(data) > min(self.limits.max_file_bytes, 4 * 1024 * 1024):
            raise _fail("file_limit_exceeded", "checksums manifest exceeds byte limit")
        if value["total_bytes"] + len(data) > self.limits.max_total_bytes:
            raise _fail("total_limit_exceeded", "managed tree including manifest exceeds limit")
        return data

    def commit(self) -> ManagedOutputCapabilities:
        if self._finished or self._published:
            raise _fail("transaction_finished", "transaction was already finished")
        if self._stage_fd is None or self._stage_identity is None:
            raise _fail("transaction_closed", "stage is unavailable")
        manifest = self._manifest_bytes()
        self._manifest_identity = _write_exclusive_at(
            self._stage_fd,
            _CHECKSUMS_NAME,
            manifest,
            hook=self.fault_hook,
            label="checksums_manifest",
        )
        _directory_fsync(self._stage_fd, hook=self.fault_hook, label="stage")
        stage_tree = _validate_tree_fd(
            self._stage_fd,
            limits=self.limits,
            parent_device=self._guard.parent_identity.device,
        )
        marker = canonical_json_v1(
            {
                "schema_version": _MARKER_SCHEMA_VERSION,
                "phase": "prepared",
                "destination_name": self.destination_name,
                "stage_name": self._stage_name,
                "backup_name": self._backup_name,
                "force": self.force,
                "new_manifest_sha256": stage_tree.manifest_sha256,
                "old_manifest_sha256": (
                    self._old_tree.manifest_sha256 if self._old_tree is not None else None
                ),
            }
        )
        self._guard.revalidate()
        self._marker_identity = _write_exclusive_at(
            self._guard.parent_fd,
            self._marker_name,
            marker,
            hook=self.fault_hook,
            label="recovery_marker",
        )
        _directory_fsync(
            self._guard.parent_fd,
            hook=self.fault_hook,
            label="marker_visible",
        )
        old_moved = False
        try:
            self._guard.revalidate()
            _assert_identity_at(
                self._guard.parent_fd,
                self._stage_name,
                stage_tree.identity,
                regular=False,
            )
            if self._old_tree is not None:
                _assert_identity_at(
                    self._guard.parent_fd,
                    self.destination_name,
                    self._old_tree.identity,
                    regular=False,
                )
                if _lstat_at(self._guard.parent_fd, self._backup_name) is not None:
                    raise _fail("name_collision", "backup name is no longer exclusive")
                _step(self.fault_hook, "before_rename_old_to_backup")
                self._guard.revalidate()
                _assert_identity_at(
                    self._guard.parent_fd,
                    self.destination_name,
                    self._old_tree.identity,
                    regular=False,
                )
                if _lstat_at(self._guard.parent_fd, self._backup_name) is not None:
                    raise _fail("name_collision", "backup name is no longer exclusive")
                os.rename(
                    self.destination_name,
                    self._backup_name,
                    src_dir_fd=self._guard.parent_fd,
                    dst_dir_fd=self._guard.parent_fd,
                )
                old_moved = True
                _step(self.fault_hook, "after_rename_old_to_backup")
                _assert_identity_at(
                    self._guard.parent_fd,
                    self._backup_name,
                    self._old_tree.identity,
                    regular=False,
                )
            elif _lstat_at(self._guard.parent_fd, self.destination_name) is not None:
                raise _fail("destination_changed", "destination appeared before commit")
            self._guard.revalidate()
            _assert_identity_at(
                self._guard.parent_fd,
                self._stage_name,
                stage_tree.identity,
                regular=False,
            )
            _step(self.fault_hook, "before_rename_stage_to_destination")
            self._guard.revalidate()
            _assert_identity_at(
                self._guard.parent_fd,
                self._stage_name,
                stage_tree.identity,
                regular=False,
            )
            if _lstat_at(self._guard.parent_fd, self.destination_name) is not None:
                raise _fail("destination_changed", "destination changed before visibility")
            os.rename(
                self._stage_name,
                self.destination_name,
                src_dir_fd=self._guard.parent_fd,
                dst_dir_fd=self._guard.parent_fd,
            )
            self._published = True
            _step(self.fault_hook, "after_rename_stage_to_destination")
        except Exception:
            if old_moved and not self._published and self._old_tree is not None:
                self._guard.revalidate()
                if _lstat_at(self._guard.parent_fd, self.destination_name) is None:
                    _assert_identity_at(
                        self._guard.parent_fd,
                        self._backup_name,
                        self._old_tree.identity,
                        regular=False,
                    )
                    os.rename(
                        self._backup_name,
                        self.destination_name,
                        src_dir_fd=self._guard.parent_fd,
                        dst_dir_fd=self._guard.parent_fd,
                    )
            raise
        _directory_fsync(
            self._guard.parent_fd,
            hook=self.fault_hook,
            label="destination_visible",
        )
        if self._old_tree is not None:
            _cleanup_tree(
                self._guard,
                self._backup_name,
                self._old_tree,
                hook=self.fault_hook,
            )
        self._guard.revalidate()
        if self._marker_identity is None:
            raise _fail("marker_invalid", "recovery marker identity is unavailable")
        _assert_identity_at(
            self._guard.parent_fd,
            self._marker_name,
            self._marker_identity,
            regular=True,
        )
        _step(self.fault_hook, "before_unlink_marker")
        self._guard.revalidate()
        _assert_identity_at(
            self._guard.parent_fd,
            self._marker_name,
            self._marker_identity,
            regular=True,
        )
        os.unlink(self._marker_name, dir_fd=self._guard.parent_fd)
        _step(self.fault_hook, "after_unlink_marker")
        _directory_fsync(
            self._guard.parent_fd,
            hook=self.fault_hook,
            label="transaction_complete",
        )
        self._finished = True
        return self._capabilities

    def abort(self) -> None:
        if self._finished or self._published or self._stage_fd is None or self._stage_identity is None:
            return
        if self._manifest_identity is not None:
            tree = _validate_tree_fd(
                self._stage_fd,
                limits=self.limits,
                parent_device=self._guard.parent_identity.device,
            )
            os.close(self._stage_fd)
            self._stage_fd = None
            _cleanup_tree(
                self._guard,
                self._stage_name,
                tree,
                hook=self.fault_hook,
            )
        else:
            for relative in sorted(
                self._written,
                key=lambda item: (item.count("/"), item.encode("utf-8")),
                reverse=True,
            ):
                parts = _relative_parts(relative, field="output path")
                parent_fd, opened = _open_relative_parent(self._stage_fd, parts)
                try:
                    self._guard.revalidate()
                    _assert_identity_at(
                        parent_fd,
                        parts[-1],
                        self._written_identities[relative],
                        regular=True,
                    )
                    os.unlink(parts[-1], dir_fd=parent_fd)
                finally:
                    _close_all(opened)
            for relative in sorted(
                self._created_directories,
                key=lambda item: (item.count("/"), item.encode("utf-8")),
                reverse=True,
            ):
                parts = _relative_parts(relative, field="output directory")
                parent_fd, opened = _open_relative_parent(self._stage_fd, parts)
                try:
                    self._guard.revalidate()
                    _assert_identity_at(
                        parent_fd,
                        parts[-1],
                        self._created_directories[relative],
                        regular=False,
                    )
                    os.rmdir(parts[-1], dir_fd=parent_fd)
                finally:
                    _close_all(opened)
            os.close(self._stage_fd)
            self._stage_fd = None
            self._guard.revalidate()
            _assert_identity_at(
                self._guard.parent_fd,
                self._stage_name,
                self._stage_identity,
                regular=False,
            )
            os.rmdir(self._stage_name, dir_fd=self._guard.parent_fd)
        if self._marker_identity is not None:
            self._guard.revalidate()
            _assert_identity_at(
                self._guard.parent_fd,
                self._marker_name,
                self._marker_identity,
                regular=True,
            )
            os.unlink(self._marker_name, dir_fd=self._guard.parent_fd)
        self._finished = True

    def close(self) -> None:
        if self._stage_fd is not None:
            try:
                os.close(self._stage_fd)
            except OSError:
                # Closing an already-closed staged descriptor is harmless cleanup.
                pass
            self._stage_fd = None
        if hasattr(self, "_guard"):
            self._guard.close()


def _read_marker(
    guard: _ParentGuard,
    destination_name: str,
) -> tuple[dict[str, object], _Identity]:
    marker_name = _marker_name(destination_name)
    data, identity = _read_regular_at(
        guard.parent_fd,
        marker_name,
        maximum_bytes=4 * 1024 * 1024,
    )
    try:
        value = load_bounded_json_bytes(data, label="managed output recovery marker")
    except (TypeError, ValueError) as exc:
        raise _fail(
            "manual_recovery_required", "recovery marker JSON is invalid"
        ) from exc
    expected = {
        "schema_version",
        "phase",
        "destination_name",
        "stage_name",
        "backup_name",
        "force",
        "new_manifest_sha256",
        "old_manifest_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise _fail("manual_recovery_required", "recovery marker fields are invalid")
    if value["schema_version"] != _MARKER_SCHEMA_VERSION or value["phase"] != "prepared":
        raise _fail("manual_recovery_required", "recovery marker phase is unknown")
    if value["destination_name"] != destination_name or not isinstance(value["force"], bool):
        raise _fail("manual_recovery_required", "recovery marker destination is invalid")
    for key, fragment in (("stage_name", ".stage."), ("backup_name", ".backup.")):
        name = value[key]
        if (
            not isinstance(name, str)
            or not name.startswith(f".{destination_name}{fragment}")
            or len(name) != len(f".{destination_name}{fragment}") + 32
            or any(
                character not in "0123456789abcdef"
                for character in name[-32:]
            )
        ):
            raise _fail("manual_recovery_required", f"recovery {key} is invalid")
        _validate_component(name, field=key)
    for key in ("new_manifest_sha256", "old_manifest_sha256"):
        digest = value[key]
        if digest is None and key == "old_manifest_sha256":
            continue
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise _fail("manual_recovery_required", f"recovery {key} is invalid")
    if bool(value["force"]) != (value["old_manifest_sha256"] is not None):
        raise _fail("manual_recovery_required", "force and old manifest state disagree")
    return value, identity


def _tree_state(
    parent_fd: int,
    name: str,
    *,
    limits: ManagedOutputLimits,
    expected_digest: str,
) -> _ValidatedTree | None:
    info = _lstat_at(parent_fd, name)
    if info is None:
        return None
    if not stat.S_ISDIR(info.st_mode):
        raise _fail("manual_recovery_required", f"{name!r} is not a directory")
    try:
        return _validate_named_tree(
            parent_fd,
            name,
            limits=limits,
            expected_manifest_sha256=expected_digest,
        )
    except ManagedOutputError as exc:
        raise _fail("manual_recovery_required", f"{name!r}: {exc}") from exc


def recover_managed_output(
    *,
    root: Path,
    destination: str | PurePosixPath,
    limits: ManagedOutputLimits,
    fault_hook: FaultHook | None = None,
) -> RecoveryResult:
    """Recover one exact known marker state or retain everything as ambiguous."""

    destination_parts = _relative_parts(destination, field="destination")
    destination_name = destination_parts[-1]
    _validate_component(
        _marker_name(destination_name), field="transaction sibling name"
    )
    guard = _ParentGuard(Path(root), destination_parts[:-1])
    try:
        marker_name = _marker_name(destination_name)
        if _lstat_at(guard.parent_fd, marker_name) is None:
            return "no_recovery_needed"
        marker, marker_identity = _read_marker(guard, destination_name)
        stage_name = str(marker["stage_name"])
        backup_name = str(marker["backup_name"])
        new_digest = str(marker["new_manifest_sha256"])
        old_digest_value = marker["old_manifest_sha256"]
        old_digest = str(old_digest_value) if old_digest_value is not None else None

        destination_info = _lstat_at(guard.parent_fd, destination_name)
        destination_new: _ValidatedTree | None = None
        destination_old: _ValidatedTree | None = None
        if destination_info is not None:
            try:
                destination_new = _tree_state(
                    guard.parent_fd,
                    destination_name,
                    limits=limits,
                    expected_digest=new_digest,
                )
            except ManagedOutputError:
                if old_digest is None:
                    raise
                destination_old = _tree_state(
                    guard.parent_fd,
                    destination_name,
                    limits=limits,
                    expected_digest=old_digest,
                )
        stage = _tree_state(
            guard.parent_fd,
            stage_name,
            limits=limits,
            expected_digest=new_digest,
        )
        backup = (
            _tree_state(
                guard.parent_fd,
                backup_name,
                limits=limits,
                expected_digest=old_digest,
            )
            if old_digest is not None
            else None
        )
        if old_digest is None and _lstat_at(guard.parent_fd, backup_name) is not None:
            raise _fail("manual_recovery_required", "unexpected backup is retained")

        result: RecoveryResult
        if destination_new is not None and stage is None:
            if backup is not None:
                _cleanup_tree(guard, backup_name, backup, hook=fault_hook)
            result = "committed"
        elif destination_old is not None and stage is not None and backup is None:
            _cleanup_tree(guard, stage_name, stage, hook=fault_hook)
            result = "rolled_back"
        elif destination_info is None and backup is not None and stage is not None:
            guard.revalidate()
            _assert_identity_at(guard.parent_fd, backup_name, backup.identity, regular=False)
            _step(fault_hook, "before_recovery_restore_backup")
            guard.revalidate()
            _assert_identity_at(guard.parent_fd, backup_name, backup.identity, regular=False)
            os.rename(
                backup_name,
                destination_name,
                src_dir_fd=guard.parent_fd,
                dst_dir_fd=guard.parent_fd,
            )
            _cleanup_tree(guard, stage_name, stage, hook=fault_hook)
            result = "rolled_back"
        elif destination_info is None and backup is not None and stage is None:
            guard.revalidate()
            _assert_identity_at(guard.parent_fd, backup_name, backup.identity, regular=False)
            _step(fault_hook, "before_recovery_restore_backup")
            guard.revalidate()
            _assert_identity_at(guard.parent_fd, backup_name, backup.identity, regular=False)
            os.rename(
                backup_name,
                destination_name,
                src_dir_fd=guard.parent_fd,
                dst_dir_fd=guard.parent_fd,
            )
            result = "rolled_back"
        elif destination_info is None and backup is None and stage is not None and old_digest is None:
            _cleanup_tree(guard, stage_name, stage, hook=fault_hook)
            result = "rolled_back"
        else:
            raise _fail(
                "manual_recovery_required",
                "filesystem state does not match one deterministic recovery case",
            )
        guard.revalidate()
        _assert_identity_at(guard.parent_fd, marker_name, marker_identity, regular=True)
        _step(fault_hook, "before_recovery_unlink_marker")
        guard.revalidate()
        _assert_identity_at(guard.parent_fd, marker_name, marker_identity, regular=True)
        os.unlink(marker_name, dir_fd=guard.parent_fd)
        _directory_fsync(guard.parent_fd, hook=fault_hook, label="recovery_complete")
        return result
    finally:
        guard.close()
