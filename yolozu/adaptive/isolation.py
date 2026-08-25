"""Code-owned isolation interfaces for adaptive image-pipeline execution."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .artifact_resolver import PinnedVerifiedArtifactSet
from .bundle_registry import RunnerProbeResult
from .bundles import CODE_OWNED_RUNNER_IDS, AlgorithmBundleSpec
from .contracts import EnvironmentProfile
from .inventory import PinnedDecodedInputSet

__all__ = ["IsolatedRunnerCapability", "IsolatedRunnerService"]


@dataclass(frozen=True)
class IsolatedRunnerCapability:
    """Code-owned live isolation capability bound to one policy digest."""

    runner_id: str
    policy_digest: str
    status: str
    backend_id: str | None = None
    backend_version: str | None = None
    image_present: bool | None = None

    def __post_init__(self) -> None:
        if self.status not in {"available", "unavailable"}:
            raise ValueError("isolated runner capability status is invalid")
        if self.runner_id not in CODE_OWNED_RUNNER_IDS:
            raise ValueError("isolated runner capability runner is not code-owned")
        if (
            not isinstance(self.policy_digest, str)
            or len(self.policy_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.policy_digest
            )
        ):
            raise ValueError("isolated runner capability policy digest is invalid")
        if self.status == "available":
            if (
                not isinstance(self.backend_id, str)
                or not self.backend_id
                or not isinstance(self.backend_version, str)
                or not self.backend_version
                or type(self.image_present) is not bool
            ):
                raise ValueError("available isolation requires complete live capability")
        elif any(
            item is not None
            for item in (self.backend_id, self.backend_version, self.image_present)
        ):
            raise ValueError("unavailable isolation forbids live capability claims")


class _RunnerSession(Protocol):
    runner_id: str
    runner_version: str

    @abstractmethod
    def probe(self, timeout_seconds: int) -> RunnerProbeResult:
        raise NotImplementedError

    @abstractmethod
    def load(self, timeout_seconds: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self, index: int, timeout_seconds: int
    ) -> tuple[Mapping[str, Any], ...]:
        raise NotImplementedError

    @abstractmethod
    def close(self, timeout_seconds: int) -> None:
        raise NotImplementedError


class IsolatedRunnerService(Protocol):
    """Code-owned isolation extension; P0 registers no implementation."""

    @property
    @abstractmethod
    def capability(self) -> IsolatedRunnerCapability:
        raise NotImplementedError

    @abstractmethod
    def open_session(
        self,
        *,
        bundle: AlgorithmBundleSpec,
        environment: EnvironmentProfile,
        artifacts: PinnedVerifiedArtifactSet,
        inputs: PinnedDecodedInputSet,
        labels: tuple[str, ...],
        outer_deadline_ns: int,
    ) -> _RunnerSession:
        raise NotImplementedError


# Repository-owned adapter modules may populate this map. No environment
# variable, registry field, MCP argument, entry point, or import string can do so.
_CODE_OWNED_ISOLATED_SERVICES: dict[str, IsolatedRunnerService] = {}


def _code_owned_isolated_services() -> dict[str, IsolatedRunnerService]:
    """Return the internal live registry without exposing a registration API."""

    return _CODE_OWNED_ISOLATED_SERVICES
