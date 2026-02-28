"""Reservoir-sampling replay buffer for continual learning.

Stores arbitrary JSON records with per-task capacity capping,
using reservoir sampling for unbiased retention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Iterable

__all__ = ["ReplayBuffer"]


@dataclass
class ReplayBuffer:
    """A tiny replay buffer with reservoir sampling.

    The buffer stores arbitrary JSON-serializable records (typically dataset records).

    - capacity=0 disables storage (memoryless mode).
    - Reservoir sampling keeps a uniform sample over all items seen so far.
    """

    capacity: int
    seed: int = 0
    items: list[dict[str, Any]] = field(default_factory=list)
    seen: int = 0

    def __post_init__(self) -> None:
        if int(self.capacity) < 0:
            raise ValueError("capacity must be >= 0")
        self.capacity = int(self.capacity)
        self._rng = random.Random(int(self.seed))

    def __len__(self) -> int:
        return int(len(self.items))

    def add(self, record: dict[str, Any]) -> None:
        if self.capacity <= 0:
            self.seen += 1
            return

        i = int(self.seen)
        self.seen += 1

        if len(self.items) < self.capacity:
            self.items.append(record)
            return

        # Reservoir: replace existing item with decreasing probability.
        j = self._rng.randint(0, i)
        if j < self.capacity:
            self.items[j] = record

    def add_with_info(self, record: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
        """Add a record and return (inserted, replaced_record).

        This is useful for callers that need to attach sidecar data for items
        that actually make it into the buffer (e.g. DER++ teacher outputs).
        """

        if self.capacity <= 0:
            self.seen += 1
            return False, None

        i = int(self.seen)
        self.seen += 1

        if len(self.items) < self.capacity:
            self.items.append(record)
            return True, None

        j = self._rng.randint(0, i)
        if j < self.capacity:
            replaced = self.items[j]
            self.items[j] = record
            return True, replaced

        return False, None

    def add_many(self, records: Iterable[dict[str, Any]]) -> None:
        for rec in records:
            if isinstance(rec, dict):
                self.add(rec)

    def sample(
        self,
        k: int | None = None,
        *,
        task_key: str | None = None,
        per_task_cap: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.items:
            return []

        items = self.items
        if per_task_cap is not None:
            if not task_key:
                raise ValueError("task_key is required when per_task_cap is set")
            cap = int(per_task_cap)
            if cap <= 0:
                return []

            by_task: dict[Any, list[dict[str, Any]]] = {}
            for rec in items:
                task_id: Any = rec.get(str(task_key))
                try:
                    hash(task_id)
                except Exception:
                    task_id = repr(task_id)
                by_task.setdefault(task_id, []).append(rec)

            capped: list[dict[str, Any]] = []
            for group in by_task.values():
                if len(group) <= cap:
                    capped.extend(group)
                else:
                    capped.extend(self._rng.sample(group, cap))
            items = capped

        if k is None:
            return list(items)
        k_i = int(k)
        if k_i <= 0:
            return []
        if k_i >= len(items):
            return list(items)
        return list(self._rng.sample(items, k_i))

    def summary(self) -> dict[str, Any]:
        def _safe_image_key(rec: dict[str, Any]) -> str | None:
            for key in ("image_path", "image"):
                value = rec.get(key)
                if isinstance(value, str) and value:
                    return value
            return None

        images = []
        for rec in self.items:
            p = _safe_image_key(rec)
            if p:
                images.append(p)

        return {
            "capacity": int(self.capacity),
            "seed": int(self.seed),
            "seen": int(self.seen),
            "size": int(len(self.items)),
            "images": images,
        }
