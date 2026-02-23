from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from yolozu.contracts.synthgen import normalize_synthgen_sample


@dataclass(frozen=True)
class SynthGenStreamPolicy:
    timeout_seconds: float = 3.0
    reconnect_backoff_seconds: float = 0.5
    max_retries: int = 5
    drop_on_timeout: bool = True
    drop_on_validation_error: bool = True


class SynthGenStreamDataset:
    """Iterable intake adapter for online SynthGen workers/queues."""

    def __init__(
        self,
        source: Any,
        *,
        schema_id: str | None = None,
        policy: SynthGenStreamPolicy | None = None,
    ) -> None:
        self.source = source
        self.schema_id = str(schema_id) if schema_id else None
        self.policy = policy or SynthGenStreamPolicy()

    def _next_raw(self, iterator: Iterator[Any] | None) -> tuple[Any, Iterator[Any] | None]:
        src = self.source
        if hasattr(src, "get") and callable(getattr(src, "get")):
            timeout = max(float(self.policy.timeout_seconds), 0.0)
            return src.get(timeout=timeout), iterator
        if callable(src):
            return src(), iterator
        if iterator is None:
            iterator = iter(src)  # type: ignore[arg-type]
        return next(iterator), iterator

    def _try_reconnect(self) -> None:
        if hasattr(self.source, "reconnect") and callable(getattr(self.source, "reconnect")):
            self.source.reconnect()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        retries = 0
        iterator: Iterator[Any] | None = None
        while True:
            try:
                raw, iterator = self._next_raw(iterator)
            except StopIteration:
                return
            except queue.Empty:
                if self.policy.drop_on_timeout:
                    continue
                raise TimeoutError(f"synthgen stream timed out after {self.policy.timeout_seconds}s")
            except Exception:
                retries += 1
                if retries > int(self.policy.max_retries):
                    raise
                self._try_reconnect()
                time.sleep(max(float(self.policy.reconnect_backoff_seconds), 0.0))
                continue

            retries = 0
            if raw is None:
                continue
            if not isinstance(raw, dict):
                if self.policy.drop_on_validation_error:
                    continue
                raise ValueError("stream record must be dict")
            try:
                sample = normalize_synthgen_sample(raw)
            except Exception:
                if self.policy.drop_on_validation_error:
                    continue
                raise

            if self.schema_id and str(sample.get("schema_id")) != self.schema_id:
                continue
            yield sample


class SynthGenQueueSource:
    """Small queue wrapper with explicit reconnect hook for stream datasets."""

    def __init__(self, q: "queue.Queue[Any]") -> None:
        self._queue = q

    def get(self, timeout: float) -> Any:
        return self._queue.get(timeout=timeout)

    def reconnect(self) -> None:
        return None


def iter_synthgen_stream(
    source: Any,
    *,
    schema_id: str | None = None,
    timeout_seconds: float = 3.0,
    reconnect_backoff_seconds: float = 0.5,
    max_retries: int = 5,
) -> Iterable[dict[str, Any]]:
    policy = SynthGenStreamPolicy(
        timeout_seconds=float(timeout_seconds),
        reconnect_backoff_seconds=float(reconnect_backoff_seconds),
        max_retries=int(max_retries),
        drop_on_timeout=True,
        drop_on_validation_error=True,
    )
    return SynthGenStreamDataset(source, schema_id=schema_id, policy=policy)
