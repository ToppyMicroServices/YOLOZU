#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.run_record import build_run_record  # noqa: E402


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"expected top-level object: {path}")
    return data


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _extract_eval_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    details = summary.get("details") if isinstance(summary.get("details"), dict) else {}
    matrix_values = payload.get("matrix_values") if isinstance(payload.get("matrix_values"), list) else []
    tasks = payload.get("tasks") if isinstance(payload.get("tasks"), list) else []

    avg_acc = _as_float(summary.get("avg_acc"))
    forgetting = _as_float(summary.get("forgetting"))

    final_scores = details.get("final") if isinstance(details.get("final"), list) else []
    new_task_score = None
    if final_scores:
        new_task_score = _as_float(final_scores[-1])
    elif matrix_values and isinstance(matrix_values[-1], list) and matrix_values[-1]:
        new_task_score = _as_float(matrix_values[-1][-1])

    old_task_final_min = None
    if final_scores and len(final_scores) >= 2:
        vals = [_as_float(v) for v in final_scores[:-1]]
        vals = [float(v) for v in vals if v is not None]
        if vals:
            old_task_final_min = min(vals)

    task_names = []
    for item in tasks:
        if isinstance(item, dict):
            task_names.append(str(item.get("name") or ""))

    return {
        "avg_acc": avg_acc,
        "forgetting": forgetting,
        "new_task_score": new_task_score,
        "old_task_final_min": old_task_final_min,
        "task_names": task_names,
        "n_tasks": len(task_names),
    }


def _extract_curation_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}

    reviewed_labels = _as_int(counts.get("reviewed_labels"))
    if reviewed_labels is None:
        reviewed_labels = _as_int(counts.get("human_reviewed_labels"))

    pseudo_labels = _as_int(counts.get("pseudo_labels"))
    pseudo_high_conf = _as_int(counts.get("pseudo_labels_high_confidence"))
    if pseudo_high_conf is None:
        pseudo_high_conf = _as_int(counts.get("high_confidence_pseudo_labels"))

    candidate_images = _as_int(counts.get("candidate_images"))
    samples_total = _as_int(counts.get("samples_total"))
    if samples_total is None:
        samples_total = _as_int(counts.get("total_samples"))

    candidate_share = None
    if candidate_images is not None and samples_total and samples_total > 0:
        candidate_share = float(candidate_images) / float(samples_total)
    elif quality.get("candidate_share") is not None:
        candidate_share = _as_float(quality.get("candidate_share"))

    total_curated = 0
    if reviewed_labels is not None:
        total_curated += int(reviewed_labels)
    if pseudo_high_conf is not None:
        total_curated += int(pseudo_high_conf)

    return {
        "reviewed_labels": reviewed_labels,
        "pseudo_labels": pseudo_labels,
        "pseudo_labels_high_confidence": pseudo_high_conf,
        "candidate_images": candidate_images,
        "samples_total": samples_total,
        "candidate_share": candidate_share,
        "total_curated_examples": total_curated,
    }


def _gate(
    *,
    name: str,
    ok: bool,
    observed: Any,
    threshold: Any,
    comparator: str,
    severity: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "name": str(name),
        "ok": bool(ok),
        "observed": observed,
        "threshold": threshold,
        "comparator": str(comparator),
        "severity": str(severity),
        "detail": str(detail),
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Decide whether a continual-learning candidate should be promoted, reviewed, or held."
    )
    p.add_argument("--eval-json", required=True, help="Path to continual_eval.json produced by tools/eval_continual.py.")
    p.add_argument(
        "--curation-json",
        default=None,
        help="Optional curation summary JSON with counts.reviewed_labels / counts.pseudo_labels_high_confidence.",
    )
    p.add_argument(
        "--run-json",
        default=None,
        help="Optional continual_run.json path for provenance (recommended but not required).",
    )
    p.add_argument("--max-forgetting", type=float, default=0.05, help="Hard gate: forgetting must be <= this value.")
    p.add_argument("--min-avg-acc", type=float, default=None, help="Optional hard gate: avg_acc must be >= this value.")
    p.add_argument(
        "--min-new-task-score",
        type=float,
        default=None,
        help="Optional hard gate: final score on the newest task must be >= this value.",
    )
    p.add_argument(
        "--min-old-task-final",
        type=float,
        default=None,
        help="Optional hard gate: minimum final score across previous tasks must be >= this value.",
    )
    p.add_argument(
        "--min-reviewed-labels",
        type=int,
        default=0,
        help="Soft gate: reviewed label count should be >= this value when curation JSON is provided.",
    )
    p.add_argument(
        "--min-highconf-pseudo-labels",
        type=int,
        default=0,
        help="Soft gate: high-confidence pseudo-label count should be >= this value when curation JSON is provided.",
    )
    p.add_argument(
        "--min-total-curated-examples",
        type=int,
        default=0,
        help="Soft gate: reviewed_labels + pseudo_labels_high_confidence should be >= this value when curation JSON is provided.",
    )
    p.add_argument(
        "--max-candidate-share",
        type=float,
        default=None,
        help="Soft gate: candidate_images / samples_total should be <= this value when curation JSON provides both counts.",
    )
    p.add_argument(
        "--ttt-active",
        action="store_true",
        help="Mark that TTT was active in the serving path; by default this forces review rather than automatic promotion.",
    )
    p.add_argument(
        "--allow-ttt-active-promotion",
        action="store_true",
        help="Allow automatic promote even when --ttt-active is set.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: sibling of --eval-json named continual_promotion_decision.json).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    eval_json_path = _resolve(args.eval_json)
    if eval_json_path is None or not eval_json_path.exists():
        raise SystemExit(f"eval json not found: {args.eval_json}")
    eval_payload = _load_json(eval_json_path)

    curation_payload: dict[str, Any] | None = None
    curation_json_path = _resolve(args.curation_json)
    if curation_json_path is not None:
        if not curation_json_path.exists():
            raise SystemExit(f"curation json not found: {args.curation_json}")
        curation_payload = _load_json(curation_json_path)

    run_json_path = _resolve(args.run_json)
    if run_json_path is not None and not run_json_path.exists():
        raise SystemExit(f"run json not found: {args.run_json}")

    observed = _extract_eval_metrics(eval_payload)
    curation = _extract_curation_metrics(curation_payload or {})

    hard_gates: list[dict[str, Any]] = []
    soft_gates: list[dict[str, Any]] = []

    forgetting = observed.get("forgetting")
    hard_gates.append(
        _gate(
            name="max_forgetting",
            ok=(forgetting is not None and float(forgetting) <= float(args.max_forgetting)),
            observed=forgetting,
            threshold=float(args.max_forgetting),
            comparator="<=",
            severity="hard",
            detail="Keep catastrophic forgetting below the allowed ceiling.",
        )
    )

    if args.min_avg_acc is not None:
        avg_acc = observed.get("avg_acc")
        hard_gates.append(
            _gate(
                name="min_avg_acc",
                ok=(avg_acc is not None and float(avg_acc) >= float(args.min_avg_acc)),
                observed=avg_acc,
                threshold=float(args.min_avg_acc),
                comparator=">=",
                severity="hard",
                detail="Require minimum average continual-learning accuracy before promotion.",
            )
        )

    if args.min_new_task_score is not None:
        new_task_score = observed.get("new_task_score")
        hard_gates.append(
            _gate(
                name="min_new_task_score",
                ok=(new_task_score is not None and float(new_task_score) >= float(args.min_new_task_score)),
                observed=new_task_score,
                threshold=float(args.min_new_task_score),
                comparator=">=",
                severity="hard",
                detail="Require the newest task to reach the configured minimum score.",
            )
        )

    if args.min_old_task_final is not None:
        old_task_final_min = observed.get("old_task_final_min")
        hard_gates.append(
            _gate(
                name="min_old_task_final",
                ok=(old_task_final_min is not None and float(old_task_final_min) >= float(args.min_old_task_final)),
                observed=old_task_final_min,
                threshold=float(args.min_old_task_final),
                comparator=">=",
                severity="hard",
                detail="Require old tasks to stay above the configured minimum retained score.",
            )
        )

    if curation_payload is not None:
        soft_gates.append(
            _gate(
                name="min_reviewed_labels",
                ok=int(curation.get("reviewed_labels") or 0) >= int(args.min_reviewed_labels),
                observed=int(curation.get("reviewed_labels") or 0),
                threshold=int(args.min_reviewed_labels),
                comparator=">=",
                severity="soft",
                detail="Prefer a minimum amount of human-reviewed supervision before promotion.",
            )
        )
        soft_gates.append(
            _gate(
                name="min_highconf_pseudo_labels",
                ok=int(curation.get("pseudo_labels_high_confidence") or 0) >= int(args.min_highconf_pseudo_labels),
                observed=int(curation.get("pseudo_labels_high_confidence") or 0),
                threshold=int(args.min_highconf_pseudo_labels),
                comparator=">=",
                severity="soft",
                detail="Prefer a minimum amount of trusted pseudo-labeled data before promotion.",
            )
        )
        soft_gates.append(
            _gate(
                name="min_total_curated_examples",
                ok=int(curation.get("total_curated_examples") or 0) >= int(args.min_total_curated_examples),
                observed=int(curation.get("total_curated_examples") or 0),
                threshold=int(args.min_total_curated_examples),
                comparator=">=",
                severity="soft",
                detail="Prefer a minimum combined amount of reviewed and trusted pseudo-labeled data.",
            )
        )
        if args.max_candidate_share is not None:
            candidate_share = curation.get("candidate_share")
            soft_gates.append(
                _gate(
                    name="max_candidate_share",
                    ok=(candidate_share is not None and float(candidate_share) <= float(args.max_candidate_share)),
                    observed=candidate_share,
                    threshold=float(args.max_candidate_share),
                    comparator="<=",
                    severity="soft",
                    detail="Prefer candidate backlog to remain bounded relative to total observed samples.",
                )
            )

    if bool(args.ttt_active):
        soft_gates.append(
            _gate(
                name="ttt_active_requires_review",
                ok=bool(args.allow_ttt_active_promotion),
                observed=True,
                threshold=bool(args.allow_ttt_active_promotion),
                comparator="requires opt-in",
                severity="soft",
                detail="When TTT is active, require explicit review unless override is provided.",
            )
        )

    hard_failed = [gate for gate in hard_gates if not bool(gate.get("ok"))]
    soft_failed = [gate for gate in soft_gates if not bool(gate.get("ok"))]

    if hard_failed:
        decision = "hold"
        recommended_next = "keep_current_production_model_and_tune_data_replay_distillation_or_schedule"
    elif soft_failed:
        decision = "review"
        if any(g.get("name") == "ttt_active_requires_review" for g in soft_failed):
            recommended_next = "review_ttt_scope_separately_before_checkpoint_promotion"
        else:
            recommended_next = "review_curation_quality_and_operator_signoff_before_promotion"
    else:
        decision = "promote"
        recommended_next = "promote_candidate_checkpoint"

    out_path = _resolve(args.output)
    if out_path is None:
        out_path = eval_json_path.parent / "continual_promotion_decision.json"

    payload = {
        "schema_version": 1,
        "kind": "continual_promotion_decision",
        "timestamp_utc": _now_utc(),
        "inputs": {
            "eval_json": str(eval_json_path),
            "curation_json": (str(curation_json_path) if curation_json_path is not None else None),
            "run_json": (str(run_json_path) if run_json_path is not None else None),
        },
        "observed": {
            **observed,
            "ttt_active": bool(args.ttt_active),
            "curation": curation,
        },
        "thresholds": {
            "max_forgetting": float(args.max_forgetting),
            "min_avg_acc": args.min_avg_acc,
            "min_new_task_score": args.min_new_task_score,
            "min_old_task_final": args.min_old_task_final,
            "min_reviewed_labels": int(args.min_reviewed_labels),
            "min_highconf_pseudo_labels": int(args.min_highconf_pseudo_labels),
            "min_total_curated_examples": int(args.min_total_curated_examples),
            "max_candidate_share": args.max_candidate_share,
            "ttt_active_requires_review": not bool(args.allow_ttt_active_promotion),
        },
        "hard_gates": hard_gates,
        "soft_gates": soft_gates,
        "decision": decision,
        "recommended_next": recommended_next,
        "run_record": build_run_record(
            repo_root=repo_root,
            argv=(sys.argv[1:] if argv is None else argv),
            args=vars(args),
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
