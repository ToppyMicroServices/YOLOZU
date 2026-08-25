"""Read-only, no-follow artifact resolution for immutable bundle assets."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .bundles import (
    MAX_ARTIFACT_BYTES,
    MAX_ARTIFACT_SET_BYTES,
    AlgorithmBundleSpec,
)
from .canonical import canonical_sha256_v1

__all__ = [
    "ArtifactResolver",
    "PinnedVerifiedArtifactSet",
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


@dataclass(frozen=True)
class _PinnedArtifact:
    artifact_id: str
    order: int
    role: str
    cache_key: str
    size_bytes: int
    sha256: str
    descriptor: int
    identity: tuple[int, int, int, int, int, int, int]


class PinnedVerifiedArtifactSet:
    """Verified artifact descriptors retained for one runner lifetime.

    Runner code receives only bounded reads against these descriptors. It is
    never given a caller-controlled cache path and therefore cannot silently
    reopen a different file after preflight.
    """

    def __init__(
        self,
        *,
        bundle_spec_digest: str,
        artifact_set_digest: str,
        artifact_resolver_state_digest: str,
        artifacts: tuple[_PinnedArtifact, ...],
        revalidate_entry: Callable[[str], tuple[int, int, int, int, int, int, int]],
    ) -> None:
        self.bundle_spec_digest = bundle_spec_digest
        self.artifact_set_digest = artifact_set_digest
        self.artifact_resolver_state_digest = artifact_resolver_state_digest
        self._artifacts = artifacts
        self._by_id = {item.artifact_id: item for item in artifacts}
        self._revalidate_entry = revalidate_entry
        self._closed = False

    def __enter__(self) -> "PinnedVerifiedArtifactSet":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for artifact in self._artifacts:
            os.close(artifact.descriptor)

    def _require(self, artifact_id: str) -> _PinnedArtifact:
        if self._closed:
            raise RuntimeError("pinned artifact set is closed")
        try:
            artifact = self._by_id[artifact_id]
        except KeyError as exc:
            raise ValueError("unknown pinned artifact id") from exc
        current = _file_identity(os.fstat(artifact.descriptor))
        entry = self._revalidate_entry(artifact.cache_key)
        if current != artifact.identity or entry != artifact.identity:
            raise ValueError(f"artifact {artifact_id}: identity changed after verification")
        return artifact

    def artifact_ids(self) -> tuple[str, ...]:
        if self._closed:
            raise RuntimeError("pinned artifact set is closed")
        return tuple(item.artifact_id for item in self._artifacts)

    def artifact_size_bytes(self, artifact_id: str) -> int:
        return self._require(artifact_id).size_bytes

    def read_artifact_chunk(
        self,
        artifact_id: str,
        *,
        offset_bytes: int,
        maximum_bytes: int,
    ) -> bytes:
        artifact = self._require(artifact_id)
        if (
            isinstance(offset_bytes, bool)
            or not isinstance(offset_bytes, int)
            or offset_bytes < 0
            or offset_bytes > artifact.size_bytes
        ):
            raise ValueError("offset_bytes is outside the pinned artifact")
        if (
            isinstance(maximum_bytes, bool)
            or not isinstance(maximum_bytes, int)
            or not 1 <= maximum_bytes <= 16 * 1024 * 1024
        ):
            raise ValueError("maximum_bytes must be in 1..16777216")
        data = os.pread(
            artifact.descriptor,
            min(maximum_bytes, artifact.size_bytes - offset_bytes),
            offset_bytes,
        )
        self._require(artifact_id)
        return data

    def iter_local_observations(self) -> Iterator[dict[str, Any]]:
        """Yield privacy-safe verified observations without paths."""

        for item in self._artifacts:
            self._require(item.artifact_id)
            yield {
                "artifact_id": item.artifact_id,
                "order": item.order,
                "role": item.role,
                "cache_key": item.cache_key,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
        int(info.st_mode),
        int(getattr(info, "st_uid", 0)),
    )


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

    def _revalidate_cache_key(
        self, cache_key: str
    ) -> tuple[int, int, int, int, int, int, int]:
        descriptor, info = self._open_cache_key(cache_key)
        try:
            return _file_identity(info)
        finally:
            os.close(descriptor)

    def pin(self, bundle: AlgorithmBundleSpec) -> PinnedVerifiedArtifactSet:
        """Verify every member and retain the same no-follow descriptors."""

        record = bundle.to_dict()
        pinned: list[_PinnedArtifact] = []
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
                after_identity = _file_identity(after)
                before_identity = _file_identity(before)
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
                pinned.append(
                    _PinnedArtifact(
                        artifact_id=artifact["artifact_id"],
                        order=artifact["order"],
                        role=artifact["role"],
                        cache_key=artifact["cache_key"],
                        size_bytes=observed,
                        sha256=observed_digest,
                        descriptor=file_fd,
                        identity=after_identity,
                    )
                )
            opened.clear()
        except Exception:
            for _artifact, file_fd, _before in opened:
                os.close(file_fd)
            raise
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
        return PinnedVerifiedArtifactSet(
            bundle_spec_digest=bundle.spec_digest,
            artifact_set_digest=bundle.artifact_set_digest,
            artifact_resolver_state_digest=state_digest,
            artifacts=tuple(pinned),
            revalidate_entry=self._revalidate_cache_key,
        )

    def verify(self, bundle: AlgorithmBundleSpec) -> VerifiedArtifactSet:
        """Verify members and return the existing path-bearing local view."""

        with self.pin(bundle) as pinned:
            artifacts = tuple(
                VerifiedArtifact(
                    artifact_id=item["artifact_id"],
                    order=item["order"],
                    role=item["role"],
                    cache_key=item["cache_key"],
                    size_bytes=item["size_bytes"],
                    sha256=item["sha256"],
                    local_path=self.root.joinpath(*item["cache_key"].split("/")),
                )
                for item in pinned.iter_local_observations()
            )
            return VerifiedArtifactSet(
                bundle_spec_digest=pinned.bundle_spec_digest,
                artifact_set_digest=pinned.artifact_set_digest,
                artifact_resolver_state_digest=pinned.artifact_resolver_state_digest,
                artifacts=artifacts,
            )
