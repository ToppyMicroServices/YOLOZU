"""Class-imbalance utilities for train_minimal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch.utils.data import WeightedRandomSampler
except ImportError:  # pragma: no cover
    torch = None
    WeightedRandomSampler = None


@dataclass
class RebalanceReport:
    strategy: str
    classes_with_labels: int
    instances_total: int
    records_total: int
    records_with_labels: int
    min_weight: float
    max_weight: float
    mean_weight: float


def _record_label_ids(record: dict[str, Any]) -> list[int]:
    labels = record.get("labels")
    if not isinstance(labels, list):
        return []

    out: list[int] = []
    for item in labels:
        if not isinstance(item, dict):
            continue
        try:
            out.append(int(item.get("class_id", -1)))
        except Exception:
            continue
    return [cid for cid in out if cid >= 0]


def collect_class_counts(records: list[dict[str, Any]], *, num_classes: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    limit = max(0, int(num_classes))
    for record in records:
        for class_id in _record_label_ids(record):
            if class_id >= limit:
                continue
            counts[class_id] = int(counts.get(class_id, 0) + 1)
    return counts


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(v)))


def _build_class_weights(
    counts: dict[int, int],
    *,
    gamma: float,
    min_weight: float,
    max_weight: float,
) -> dict[int, float]:
    if not counts:
        return {}
    gamma_v = max(0.0, float(gamma))
    # Inverse-frequency weights, normalized by max-count class.
    max_count = max(int(v) for v in counts.values())
    base = max(1.0, float(max_count))

    out: dict[int, float] = {}
    for class_id, count in counts.items():
        freq = max(1.0, float(count))
        raw = (base / freq) ** gamma_v
        out[int(class_id)] = _clamp(raw, float(min_weight), float(max_weight))
    return out


def build_record_weights(
    records: list[dict[str, Any]],
    class_weights: dict[int, float],
    *,
    aggregate: str,
    default_weight: float,
) -> list[float]:
    agg = str(aggregate or "max").strip().lower()
    if agg not in ("max", "mean"):
        raise ValueError("aggregate must be one of: max, mean")

    out: list[float] = []
    fallback = float(default_weight)

    for record in records:
        cids = _record_label_ids(record)
        if not cids:
            out.append(fallback)
            continue
        vals = [float(class_weights.get(cid, fallback)) for cid in cids]
        if not vals:
            out.append(fallback)
            continue
        if agg == "mean":
            out.append(float(sum(vals) / max(1, len(vals))))
        else:
            out.append(float(max(vals)))
    return out


def build_weighted_sampler(
    records: list[dict[str, Any]],
    *,
    num_classes: int,
    strategy: str,
    gamma: float,
    min_weight: float,
    max_weight: float,
    aggregate: str,
    seed: int,
) -> tuple["WeightedRandomSampler | None", RebalanceReport | None]:
    if torch is None or WeightedRandomSampler is None:  # pragma: no cover
        raise RuntimeError("torch is required for imbalance rebalancing")

    chosen = str(strategy or "none").strip().lower()
    if chosen == "none":
        return None, None
    if chosen != "class_balanced":
        raise ValueError(f"unsupported imbalance strategy: {strategy}")

    if not records:
        raise ValueError("cannot build imbalance sampler: records is empty")

    class_counts = collect_class_counts(records, num_classes=int(num_classes))
    if not class_counts:
        raise ValueError(
            "--imbalance-strategy=class_balanced requires class labels, but no class_id entries were found"
        )

    class_weights = _build_class_weights(
        class_counts,
        gamma=float(gamma),
        min_weight=float(min_weight),
        max_weight=float(max_weight),
    )
    record_weights = build_record_weights(
        records,
        class_weights,
        aggregate=str(aggregate),
        default_weight=float(min_weight),
    )

    if not record_weights:
        raise ValueError("failed to build record weights")

    weights_t = torch.tensor(record_weights, dtype=torch.double)
    if int(weights_t.numel()) != int(len(records)):
        raise ValueError("record weight count mismatch")
    if not bool(torch.isfinite(weights_t).all().item()):
        raise ValueError("non-finite weights detected for imbalance sampler")
    if bool((weights_t <= 0).all().item()):
        raise ValueError("all record weights are <=0; cannot sample")

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        weights_t,
        num_samples=int(len(record_weights)),
        replacement=True,
        generator=gen,
    )

    records_with_labels = 0
    for record in records:
        if _record_label_ids(record):
            records_with_labels += 1

    report = RebalanceReport(
        strategy="class_balanced",
        classes_with_labels=int(len(class_counts)),
        instances_total=int(sum(int(v) for v in class_counts.values())),
        records_total=int(len(records)),
        records_with_labels=int(records_with_labels),
        min_weight=float(weights_t.min().item()),
        max_weight=float(weights_t.max().item()),
        mean_weight=float(weights_t.mean().item()),
    )
    return sampler, report
