"""Semantic-segmentation parity utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yolozu.eval.segmentation_eval import load_mask_array
from yolozu.predictions.segmentation_predictions import build_id_to_mask, load_segmentation_predictions_entries


def _resolve_mask_path(mask_value: str, *, base: Path) -> Path:
    p = Path(mask_value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def compare_segmentation_predictions(
    *,
    reference: str | Path,
    candidate: str | Path,
    mismatch_atol: float = 0.0,
    max_samples: int | None = None,
) -> dict[str, Any]:
    ref_path = Path(reference)
    cand_path = Path(candidate)
    ref_entries, _ = load_segmentation_predictions_entries(ref_path)
    cand_entries, _ = load_segmentation_predictions_entries(cand_path)
    ref_masks = build_id_to_mask(ref_entries)
    cand_masks = build_id_to_mask(cand_entries)

    sample_ids = sorted(set(ref_masks) | set(cand_masks))
    if max_samples is not None:
        sample_ids = sample_ids[: max(int(max_samples), 0)]

    results: list[dict[str, Any]] = []
    ok = True

    for sample_id in sample_ids:
        item: dict[str, Any] = {"id": sample_id, "ok": True}
        ref_mask = ref_masks.get(sample_id)
        cand_mask = cand_masks.get(sample_id)
        if ref_mask is None:
            item.update({"ok": False, "reason": "missing_reference"})
            ok = False
            results.append(item)
            continue
        if cand_mask is None:
            item.update({"ok": False, "reason": "missing_candidate"})
            ok = False
            results.append(item)
            continue

        ref_arr = load_mask_array(_resolve_mask_path(ref_mask, base=ref_path.parent))
        cand_arr = load_mask_array(_resolve_mask_path(cand_mask, base=cand_path.parent))
        if ref_arr.shape != cand_arr.shape:
            item.update(
                {
                    "ok": False,
                    "reason": "shape_mismatch",
                    "reference_shape": list(ref_arr.shape),
                    "candidate_shape": list(cand_arr.shape),
                }
            )
            ok = False
            results.append(item)
            continue

        mismatched = int((ref_arr != cand_arr).sum())
        total = int(ref_arr.size)
        mismatch_rate = float(mismatched) / float(total) if total > 0 else 0.0
        item.update(
            {
                "pixels_total": total,
                "pixels_mismatched": mismatched,
                "mismatch_rate": mismatch_rate,
                "ok": bool(mismatch_rate <= float(mismatch_atol)),
            }
        )
        if not item["ok"]:
            ok = False
        results.append(item)

    return {
        "schema_version": 1,
        "kind": "segmentation_parity_report",
        "ok": bool(ok),
        "images": int(len(results)),
        "thresholds": {"mismatch_atol": float(mismatch_atol)},
        "results": results,
    }
