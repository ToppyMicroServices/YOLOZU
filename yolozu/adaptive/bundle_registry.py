"""Validated loading for the adaptive algorithm-bundle SSOT.

The packaged registry is metadata only. Loading never imports a model runtime,
fetches an artifact, instantiates a runner, or makes a bundle selectable.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from .bundles import (
    CODE_OWNED_RUNNER_IDS,
    AlgorithmBundleRegistry,
    AlgorithmBundleSpec,
    BundleLifecycleProjection,
    SupportProfileProjection,
    project_bundle_lifecycle,
    validate_algorithm_bundle_registry,
)
from .contracts import EnvironmentProfile
from .control_records import (
    MAX_CONTROL_RECORD_BYTES,
    MAX_CONTROL_STREAM_BYTES,
    load_bounded_json_bytes,
    load_bounded_jsonl_bytes,
)

__all__ = [
    "AlgorithmRunner",
    "CODE_OWNED_RUNNER_IDS",
    "LoadedAlgorithmBundleRegistry",
    "PinnedArtifactSet",
    "PinnedInput",
    "RunnerProbeResult",
    "load_algorithm_bundle_registry",
]


_REGISTRY_BASENAME = "bundle_specs.json"
_LIFECYCLE_BASENAME = "bundle_lifecycle.jsonl"
_MAX_LIFECYCLE_RECORDS = 4096


@runtime_checkable
class PinnedInput(Protocol):
    """Runner-visible input handle with no caller-controlled path surface."""

    @property
    def input_index(self) -> int:
        raise NotImplementedError

    @property
    def source_size_bytes(self) -> int:
        raise NotImplementedError

    def read_source_bytes(self) -> bytes:
        """Return bytes from the already pinned and identity-checked source."""


@runtime_checkable
class PinnedArtifactSet(Protocol):
    """Runner-visible immutable artifact handles bound to one bundle spec."""

    @property
    def bundle_spec_digest(self) -> str:
        raise NotImplementedError

    @property
    def artifact_set_digest(self) -> str:
        raise NotImplementedError

    def artifact_ids(self) -> tuple[str, ...]:
        """Return ordered code-owned artifact identifiers."""

    def artifact_size_bytes(self, artifact_id: str) -> int:
        """Return the validated size for one pinned artifact."""

    def read_artifact_chunk(
        self,
        artifact_id: str,
        *,
        offset_bytes: int,
        maximum_bytes: int,
    ) -> bytes:
        """Read a bounded chunk without exposing or reopening a caller path."""


@dataclass(frozen=True)
class RunnerProbeResult:
    """Bounded runner availability result; failure is never inferred support."""

    status: Literal["supported", "unsupported", "failed"]
    reason_code: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"supported", "unsupported", "failed"}:
            raise ValueError("runner probe status is invalid")
        if self.status == "supported" and self.reason_code is not None:
            raise ValueError("supported runner probe forbids a reason code")
        if self.status != "supported":
            reason = self.reason_code
            if (
                not isinstance(reason, str)
                or not reason
                or len(reason.encode("utf-8")) > 128
                or not all(
                    character.isascii() and (character.isalnum() or character in "_-")
                    for character in reason
                )
            ):
                raise ValueError(
                    "non-supported runner probe requires a bounded reason code"
                )


@runtime_checkable
class AlgorithmRunner(Protocol):
    """Typed lifecycle implemented only by audited code-owned runner factories."""

    @property
    def runner_id(self) -> str:
        raise NotImplementedError

    @property
    def runner_version(self) -> str:
        raise NotImplementedError

    def probe(
        self,
        *,
        bundle: AlgorithmBundleSpec,
        environment: EnvironmentProfile,
    ) -> RunnerProbeResult:
        """Check exact runtime/provider availability without loading artifacts."""

    def load(
        self,
        *,
        bundle: AlgorithmBundleSpec,
        artifacts: PinnedArtifactSet,
    ) -> None:
        """Load only the validated immutable bundle and pinned artifact set."""

    def warmup(self, *, input_item: PinnedInput) -> None:
        """Warm the already loaded runner using one pinned input."""

    def predict(
        self,
        *,
        input_item: PinnedInput,
        requested_labels: tuple[str, ...],
    ) -> tuple[Mapping[str, Any], ...]:
        """Return typed record candidates for strict postprocessing/validation."""

    def close(self) -> None:
        """Release the loaded runner and its bounded child resources."""


@dataclass(frozen=True)
class LoadedAlgorithmBundleRegistry:
    """Validated registry/lifecycle view with explicit source trust."""

    registry: AlgorithmBundleRegistry
    bundles: tuple[AlgorithmBundleSpec, ...]
    lifecycle: BundleLifecycleProjection
    registry_trust_domain: Literal["yolozu_managed", "operator_asserted"]
    lifecycle_trust_domain: Literal["yolozu_managed", "operator_asserted"]
    source_kind: Literal["packaged_ssot", "workspace_custom"]

    @property
    def selection_trust_reason_codes(self) -> tuple[str, ...]:
        """Return the fail-closed v1 reasons for a non-managed source."""

        reasons: list[str] = []
        if self.registry_trust_domain != "yolozu_managed":
            reasons.append("registry_untrusted")
        if self.lifecycle_trust_domain != "yolozu_managed":
            reasons.append("lifecycle_untrusted")
        return tuple(reasons)

    def by_spec_digest(self) -> dict[str, AlgorithmBundleSpec]:
        return {bundle.spec_digest: bundle for bundle in self.bundles}

    def is_lifecycle_eligible(self, *, family_id: str, channel: str) -> bool:
        """Apply source trust before the validated lifecycle/license gate."""

        return (
            not self.selection_trust_reason_codes
            and self.lifecycle.is_lifecycle_eligible(
                family_id=family_id,
                channel=channel,
            )
        )


def _bundle_identity(bundle: AlgorithmBundleSpec) -> tuple[bytes, bytes, bytes, bytes]:
    record = bundle.to_dict()
    return (
        record["family_id"].encode("utf-8"),
        record["bundle_id"].encode("utf-8"),
        record["bundle_version"].encode("utf-8"),
        bundle.spec_digest.encode("ascii"),
    )


def _workspace_directory(path: Path, *, workspace_root: Path) -> Path:
    workspace_input = Path(workspace_root)
    workspace_lexical = Path(os.path.abspath(workspace_input))
    if workspace_lexical.is_symlink():
        raise ValueError("workspace_root: symlinks are invalid")
    try:
        workspace = workspace_lexical.resolve(strict=True)
    except OSError as exc:
        raise ValueError("workspace_root: unavailable") from exc
    if not workspace.is_dir():
        raise ValueError("workspace_root: expected directory")

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace_lexical / candidate
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(workspace_lexical)
    except ValueError as exc:
        raise ValueError("custom registry root must stay inside workspace") from exc
    current = workspace_lexical
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("custom registry root contains a symlink component")
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(workspace)
    except (OSError, ValueError) as exc:
        raise ValueError(
            "custom registry root is unavailable or outside workspace"
        ) from exc
    if not resolved.is_dir():
        raise ValueError("custom registry root must be a directory")
    return resolved


def _read_child(root_fd: int, basename: str, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(basename, flags, dir_fd=root_fd)
    except OSError as exc:
        raise ValueError(f"custom registry file is unavailable: {basename}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"custom registry file is not regular: {basename}")
        if before.st_size > maximum_bytes:
            raise ValueError(f"custom registry file exceeds byte cap: {basename}")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1_048_576, maximum_bytes + 1 - observed))
            if not chunk:
                break
            observed += len(chunk)
            if observed > maximum_bytes:
                raise ValueError(f"custom registry file exceeds byte cap: {basename}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        before_identity = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_ctime_ns),
            int(before.st_mode),
        )
        after_identity = (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_ctime_ns),
            int(after.st_mode),
        )
        if before_identity != after_identity or observed != after.st_size:
            raise ValueError(f"custom registry file changed while reading: {basename}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_custom_pair(root: Path) -> tuple[bytes, bytes]:
    if not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("custom registry loading requires directory-relative opens")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        root_fd = os.open(root, flags)
    except OSError as exc:
        raise ValueError("custom registry root cannot be opened safely") from exc
    try:
        registry_bytes = _read_child(
            root_fd,
            _REGISTRY_BASENAME,
            maximum_bytes=MAX_CONTROL_RECORD_BYTES,
        )
        remaining = MAX_CONTROL_STREAM_BYTES - len(registry_bytes)
        lifecycle_bytes = _read_child(
            root_fd,
            _LIFECYCLE_BASENAME,
            maximum_bytes=remaining,
        )
        return registry_bytes, lifecycle_bytes
    finally:
        os.close(root_fd)


def _read_packaged_pair() -> tuple[bytes, bytes]:
    root = resources.files("yolozu.data").joinpath("adaptive_routing")
    registry_bytes = root.joinpath(_REGISTRY_BASENAME).read_bytes()
    lifecycle_bytes = root.joinpath(_LIFECYCLE_BASENAME).read_bytes()
    if len(registry_bytes) > MAX_CONTROL_RECORD_BYTES:
        raise ValueError("packaged bundle registry exceeds 4 MiB")
    if len(registry_bytes) + len(lifecycle_bytes) > MAX_CONTROL_STREAM_BYTES:
        raise ValueError("packaged registry input exceeds 128 MiB")
    return registry_bytes, lifecycle_bytes


def load_algorithm_bundle_registry(
    *,
    workspace_root: Path | None = None,
    custom_registry_root: Path | None = None,
    support_profiles: SupportProfileProjection | None = None,
) -> LoadedAlgorithmBundleRegistry:
    """Load the exact packaged SSOT or one explicit workspace-confined custom pair.

    A custom pair is always operator-asserted catalog input. Its JSON claims do
    not upgrade trust and it cannot satisfy the v1 selection trust gate.
    """

    if custom_registry_root is None:
        if workspace_root is not None:
            raise ValueError("workspace_root is valid only with custom_registry_root")
        registry_bytes, lifecycle_bytes = _read_packaged_pair()
        trust: Literal["yolozu_managed", "operator_asserted"] = "yolozu_managed"
        source_kind: Literal["packaged_ssot", "workspace_custom"] = "packaged_ssot"
    else:
        if workspace_root is None:
            raise ValueError("custom_registry_root requires workspace_root")
        root = _workspace_directory(
            Path(custom_registry_root),
            workspace_root=Path(workspace_root),
        )
        registry_bytes, lifecycle_bytes = _read_custom_pair(root)
        trust = "operator_asserted"
        source_kind = "workspace_custom"

    if len(registry_bytes) + len(lifecycle_bytes) > MAX_CONTROL_STREAM_BYTES:
        raise ValueError("registry input exceeds 128 MiB")
    if support_profiles is not None:
        if not isinstance(support_profiles, SupportProfileProjection):
            raise TypeError("support_profiles must be a validated projection")
        if trust == "yolozu_managed" and any(
            record.source_trust_domain != "yolozu_managed"
            for record in support_profiles.record_by_digest.values()
        ):
            raise ValueError("managed lifecycle requires managed support-profile trust")
    registry_payload = load_bounded_json_bytes(
        registry_bytes,
        label="algorithm bundle registry",
    )
    registry = validate_algorithm_bundle_registry(registry_payload)
    lifecycle_payload = load_bounded_jsonl_bytes(
        lifecycle_bytes,
        label="bundle lifecycle",
        max_records=_MAX_LIFECYCLE_RECORDS,
    )
    lifecycle = project_bundle_lifecycle(
        registry,
        lifecycle_payload,
        source_trust_domain=trust,
        support_profiles=support_profiles,
    )
    bundles = tuple(sorted(registry.bundles, key=_bundle_identity))
    return LoadedAlgorithmBundleRegistry(
        registry=registry,
        bundles=bundles,
        lifecycle=lifecycle,
        registry_trust_domain=trust,
        lifecycle_trust_domain=trust,
        source_kind=source_kind,
    )
