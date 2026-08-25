"""Read-only, no-follow artifact resolution for immutable bundle assets."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bundles import (
    MAX_ARTIFACT_BYTES,
    MAX_ARTIFACT_SET_BYTES,
    AlgorithmBundleSpec,
)
from .canonical import canonical_sha256_v1

__all__ = [
    "ArtifactResolver",
    "VerifiedArtifact",
    "VerifiedArtifactSet",
]


RESOLVER_VERSION = "artifact-resolver-v1"


@dataclass(frozen=True)
class VerifiedArtifact:
    """One verified asset; the path is sensitive process-local state."""

    artifact_id: str
    order: int
    role: str
    cache_key: str
    size_bytes: int
    sha256: str
    local_path: Path


@dataclass(frozen=True)
class VerifiedArtifactSet:
    bundle_spec_digest: str
    artifact_set_digest: str
    artifact_resolver_state_digest: str
    artifacts: tuple[VerifiedArtifact, ...]


class ArtifactResolver:
    """Verify bundle cache keys below one pinned local store root."""

    def __init__(
        self,
        *,
        workspace: Path,
        artifact_root: Path | None = None,
    ) -> None:
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            raise RuntimeError("artifact resolver requires no-follow directory opens")
        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("workspace must be a directory")
        if artifact_root is None:
            selected = Path.home() / ".cache" / "yolozu" / "models"
            self.store_kind = "default_yolozu_model_cache"
        else:
            selected = Path(artifact_root)
            self.store_kind = "workspace_artifact_root"
        if selected.is_symlink():
            raise ValueError("artifact root must not be a symlink")
        self.root = selected.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("artifact root must be a directory")
        if artifact_root is not None:
            try:
                self.root.relative_to(self.workspace)
            except ValueError as exc:
                raise ValueError("explicit artifact root must stay inside workspace") from exc
        self._root_fd = os.open(
            self.root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        root_stat = os.fstat(self._root_fd)
        if not stat.S_ISDIR(root_stat.st_mode):
            self.close()
            raise ValueError("artifact root is not a directory")
        current_uid = getattr(os, "getuid", lambda: root_stat.st_uid)()
        if hasattr(root_stat, "st_uid") and root_stat.st_uid != current_uid:
            self.close()
            raise ValueError("artifact root is not owned by the current user")
        self._root_identity = (
            int(root_stat.st_dev),
            int(root_stat.st_ino),
            int(getattr(root_stat, "st_uid", 0)),
        )

    def __enter__(self) -> ArtifactResolver:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        root_fd = getattr(self, "_root_fd", -1)
        if root_fd >= 0:
            os.close(root_fd)
            self._root_fd = -1

    def _require_open(self) -> int:
        if self._root_fd < 0:
            raise RuntimeError("artifact resolver is closed")
        current = os.fstat(self._root_fd)
        identity = (
            int(current.st_dev),
            int(current.st_ino),
            int(getattr(current, "st_uid", 0)),
        )
        if identity != self._root_identity or not stat.S_ISDIR(current.st_mode):
            raise ValueError("artifact root identity changed")
        return self._root_fd

    def _open_cache_key(self, cache_key: str) -> tuple[int, os.stat_result]:
        parts = cache_key.split("/")
        parent_fd = os.dup(self._require_open())
        try:
            for component in parts[:-1]:
                child_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
                child_stat = os.fstat(child_fd)
                if not stat.S_ISDIR(child_stat.st_mode):
                    os.close(child_fd)
                    raise ValueError("cache-key parent is not a directory")
                os.close(parent_fd)
                parent_fd = child_fd
            file_fd = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise ValueError(f"cache key cannot be opened safely: {cache_key}") from exc
        finally:
            os.close(parent_fd)
        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode):
            os.close(file_fd)
            raise ValueError("resolved artifact is not a regular file")
        root_uid = self._root_identity[2]
        if hasattr(file_stat, "st_uid") and file_stat.st_uid != root_uid:
            os.close(file_fd)
            raise ValueError("resolved artifact ownership differs from the store root")
        return file_fd, file_stat

    def verify(self, bundle: AlgorithmBundleSpec) -> VerifiedArtifactSet:
        """Verify every required member before returning local observations."""

        record = bundle.to_dict()
        resolved: list[VerifiedArtifact] = []
        opened: list[tuple[dict[str, Any], int, os.stat_result]] = []
        identities: set[tuple[int, int]] = set()
        actual_total = 0
        try:
            # Preflight every required member before reading any artifact bytes.
            for artifact in record["artifacts"]:
                file_fd, before = self._open_cache_key(artifact["cache_key"])
                opened.append((artifact, file_fd, before))
                if before.st_size != artifact["expected_size_bytes"]:
                    raise ValueError(
                        f"artifact {artifact['artifact_id']}: stat size mismatch"
                    )
                if before.st_size < 1 or before.st_size > MAX_ARTIFACT_BYTES:
                    raise ValueError(
                        f"artifact {artifact['artifact_id']}: stat size exceeds cap"
                    )
                identity = (int(before.st_dev), int(before.st_ino))
                if identity in identities:
                    raise ValueError("two cache keys resolve to the same regular file")
                identities.add(identity)
                actual_total += int(before.st_size)
                if actual_total > MAX_ARTIFACT_SET_BYTES:
                    raise ValueError("resolved artifact set exceeds 64 GiB")

            for artifact, file_fd, before in opened:
                digest = hashlib.sha256()
                observed = 0
                while True:
                    chunk = os.read(file_fd, 1024 * 1024)
                    if not chunk:
                        break
                    observed += len(chunk)
                    if observed > artifact["expected_size_bytes"]:
                        raise ValueError(
                            f"artifact {artifact['artifact_id']}: grew while hashing"
                        )
                    digest.update(chunk)
                after = os.fstat(file_fd)
                after_identity = (
                    int(after.st_dev),
                    int(after.st_ino),
                    int(after.st_size),
                    int(after.st_mtime_ns),
                    int(after.st_ctime_ns),
                    int(after.st_mode),
                    int(getattr(after, "st_uid", 0)),
                )
                before_identity = (
                    int(before.st_dev),
                    int(before.st_ino),
                    int(before.st_size),
                    int(before.st_mtime_ns),
                    int(before.st_ctime_ns),
                    int(before.st_mode),
                    int(getattr(before, "st_uid", 0)),
                )
                if after_identity != before_identity:
                    raise ValueError(
                        f"artifact {artifact['artifact_id']}: identity changed while hashing"
                    )
                observed_digest = digest.hexdigest()
                if observed != artifact["expected_size_bytes"]:
                    raise ValueError(
                        f"artifact {artifact['artifact_id']}: byte count mismatch"
                    )
                if observed_digest != artifact["sha256"]:
                    raise ValueError(
                        f"artifact {artifact['artifact_id']}: SHA-256 mismatch"
                    )
                resolved.append(
                    VerifiedArtifact(
                        artifact_id=artifact["artifact_id"],
                        order=artifact["order"],
                        role=artifact["role"],
                        cache_key=artifact["cache_key"],
                        size_bytes=observed,
                        sha256=observed_digest,
                        local_path=self.root.joinpath(*artifact["cache_key"].split("/")),
                    )
                )
        finally:
            for _artifact, file_fd, _before in opened:
                os.close(file_fd)
        state_digest = canonical_sha256_v1(
            {
                "resolver_version": RESOLVER_VERSION,
                "store_kind": self.store_kind,
                "root_identity": {
                    "device": self._root_identity[0],
                    "inode": self._root_identity[1],
                    "owner": self._root_identity[2],
                },
                "cache_keys": [artifact["cache_key"] for artifact in record["artifacts"]],
            }
        )
        return VerifiedArtifactSet(
            bundle_spec_digest=bundle.spec_digest,
            artifact_set_digest=bundle.artifact_set_digest,
            artifact_resolver_state_digest=state_digest,
            artifacts=tuple(resolved),
        )
