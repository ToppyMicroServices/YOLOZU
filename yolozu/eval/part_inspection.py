"""Experimental, offline part-level comparison over immutable predictions.

This module never runs a model, reads images, or makes a production acceptance
decision. Input identities and calibration are operator assertions, not proofs.
"""

from __future__ import annotations

import hashlib
import copy
import json
import math
import os
import stat
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = 64 * 1024 * 1024
DECISIONS = ("OK", "NOK", "ABSTAIN")


def _object(
    value: Any, where: str, required: set[str], optional: set[str] | None = None
) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected an object")
    missing = required - value.keys()
    extra = value.keys() - required - (optional or set())
    if missing or extra:
        raise ValueError(
            f"{where}: missing fields {sorted(missing)}; unknown fields {sorted(extra)}"
        )
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{where}: expected a nonempty string without edge whitespace")
    return value


def _number(
    value: Any, where: str, minimum: float = 0, maximum: float | None = None
) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{where}: expected a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{where}: number is too large") from exc
    if (
        not math.isfinite(number)
        or number < minimum
        or (maximum is not None and number > maximum)
    ):
        raise ValueError(f"{where}: number outside the allowed range")
    return number


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{where}: expected an integer >= {minimum}")
    return value


def _list(value: Any, where: str, *, nonempty: bool = False) -> list:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{where}: expected {'a nonempty' if nonempty else 'a'} list")
    return value


def _version(value: Any, where: str) -> None:
    if type(value) is not int or value != 1:
        raise ValueError(f"{where}: only schema_version 1 is supported")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _validate_protocol(protocol: Any) -> None:
    _object(
        protocol,
        "protocol",
        {
            "schema_version",
            "protocol_id",
            "split",
            "models",
            "class_map",
            "score_threshold",
            "sampling",
            "evidence_kind",
        },
    )
    _version(protocol["schema_version"], "protocol")
    for key in ("protocol_id", "split"):
        _text(protocol[key], f"protocol.{key}")
    _number(protocol["score_threshold"], "score_threshold", maximum=1)
    if protocol["sampling"] not in ("independent_parts", "dependent_parts"):
        raise ValueError("sampling: declare independent_parts or dependent_parts")
    if protocol["evidence_kind"] not in ("synthetic", "operator_supplied"):
        raise ValueError("evidence_kind: declare synthetic or operator_supplied")
    classes = protocol["class_map"]
    if not isinstance(classes, dict) or not classes:
        raise ValueError("class_map: expected a nonempty object")
    for key, label in classes.items():
        if (
            not isinstance(key, str)
            or not key.isascii()
            or not key.isdecimal()
            or str(int(key)) != key
        ):
            raise ValueError(
                "class_map: keys must be canonical nonnegative integer strings"
            )
        _text(label, "class_map label")
    if not isinstance(protocol["models"], dict) or not protocol["models"]:
        raise ValueError("models: expected region-keyed model identities")
    for region, model in protocol["models"].items():
        _text(region, "model region")
        _object(model, "model", {"model_id", "model_sha256", "inference_settings"})
        _text(model["model_id"], "model_id")
        digest = model["model_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("model_sha256: expected a lowercase SHA-256 digest")
        if not isinstance(model["inference_settings"], dict):
            raise ValueError("inference_settings: expected an object")


def _validate_rules(rules: Any, protocol: dict) -> dict[tuple[str, int], list[dict]]:
    _object(
        rules,
        "rules",
        {
            "schema_version",
            "ruleset_id",
            "version",
            "precedence",
            "regions",
            "rules",
            "calibrations",
        },
    )
    _version(rules["schema_version"], "rules")
    _text(rules["ruleset_id"], "ruleset_id")
    _text(rules["version"], "rules version")
    if rules["precedence"] != "first_match":
        raise ValueError(
            "precedence: only first_match (lowest priority number) is supported"
        )
    regions = rules["regions"]
    if not isinstance(regions, dict) or set(regions) != set(protocol["models"]):
        raise ValueError("regions must exactly match the protocol model regions")
    for name, region in regions.items():
        _text(name, "region_id")
        _object(region, "region", {"adapter", "defect_class_ids", "bad_view_threshold"})
        if region["adapter"] not in ("detections", "classification"):
            raise ValueError("region adapter must be detections or classification")
        _integer(region["bad_view_threshold"], "bad_view_threshold", 1)
        ids = _list(region["defect_class_ids"], "defect_class_ids", nonempty=True)
        for cid in ids:
            _integer(cid, "defect_class_id")
            if str(cid) not in protocol["class_map"]:
                raise ValueError("defect_class_id missing from class_map")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate defect_class_id")
    calibrations = rules["calibrations"]
    if not isinstance(calibrations, dict):
        raise ValueError("calibrations: expected an object")
    for name, calibration in calibrations.items():
        _text(name, "calibration_id")
        _object(calibration, "calibration", {"unit", "x", "y", "source"})
        if calibration["unit"] != "mm_per_pixel":
            raise ValueError("calibration unit must be mm_per_pixel")
        for axis in ("x", "y"):
            if _number(calibration[axis], f"calibration.{axis}") <= 0:
                raise ValueError("calibration scale must be positive")
        _text(calibration["source"], "calibration source")
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    seen_ids: set[str] = set()
    for rule in _list(rules["rules"], "rules.rules"):
        _object(
            rule,
            "rule",
            {"rule_id", "region_id", "class_id", "priority", "min_count", "source"},
            {"unit", "min_width", "min_height"},
        )
        rid = _text(rule["rule_id"], "rule_id")
        if rid in seen_ids:
            raise ValueError("duplicate rule_id")
        seen_ids.add(rid)
        region_id = _text(rule["region_id"], "rule region_id")
        cid = _integer(rule["class_id"], "rule class_id")
        if (
            region_id not in regions
            or cid not in regions[region_id]["defect_class_ids"]
        ):
            raise ValueError(
                "rule must target a declared defect class in a known region"
            )
        _integer(rule["priority"], "priority")
        _integer(rule["min_count"], "min_count", 1)
        _text(rule["source"], "rule source")
        dimensions = "min_width" in rule or "min_height" in rule
        if dimensions:
            if regions[region_id]["adapter"] != "detections" or rule.get(
                "unit"
            ) not in ("px", "mm"):
                raise ValueError(
                    "dimensions require the detections adapter and px or mm units"
                )
            for key in ("min_width", "min_height"):
                _number(rule.get(key, 0), key)
        elif "unit" in rule:
            raise ValueError("unit without a dimension threshold")
        if regions[region_id]["adapter"] == "classification" and rule["min_count"] != 1:
            raise ValueError(
                "classification has one result per view; min_count must be 1"
            )
        key = (region_id, cid)
        if any(r["priority"] == rule["priority"] for r in grouped[key]):
            raise ValueError("overlapping rules need distinct priorities")
        grouped[key].append(rule)
    return {
        key: sorted(value, key=lambda r: r["priority"])
        for key, value in grouped.items()
    }


def _validate_mapping(
    mapping: Any, rules: dict, protocol: dict, grouped: dict
) -> tuple[dict, dict]:
    _object(mapping, "mapping", {"schema_version", "label_source", "parts", "views"})
    _version(mapping["schema_version"], "mapping")
    _text(mapping["label_source"], "label_source")
    parts: dict[str, dict] = {}
    for part in _list(mapping["parts"], "parts", nonempty=True):
        _object(part, "part", {"part_id", "label", "split", "expected_views"})
        pid = _text(part["part_id"], "part_id")
        if pid in parts:
            raise ValueError("duplicate part_id (including cross-split reuse)")
        _text(part["split"], "split")
        if part["label"] not in ("OK", "NOK"):
            raise ValueError("part label must be independently assigned OK or NOK")
        expected: dict[str, str] = {}
        for view in _list(part["expected_views"], "expected_views", nonempty=True):
            _object(view, "expected view", {"view_id", "region_id"})
            vid = _text(view["view_id"], "view_id")
            region = _text(view["region_id"], "region_id")
            if vid in expected or region not in rules["regions"]:
                raise ValueError("duplicate expected view_id or unknown region")
            expected[vid] = region
        counts = Counter(expected.values())
        if set(counts) != set(rules["regions"]):
            raise ValueError("each part must declare views for every region")
        for region, config in rules["regions"].items():
            if config["bad_view_threshold"] > counts[region]:
                raise ValueError(
                    "bad_view_threshold exceeds the expected region view count"
                )
        parts[pid] = {**part, "expected": expected}
    if not any(p["split"] == protocol["split"] for p in parts.values()):
        raise ValueError("evaluation split has no parts")
    images: dict[str, dict] = {}
    seen: set[tuple[str, str]] = set()
    dimensional: dict[str, list[dict]] = defaultdict(list)
    for (region, _), matches in grouped.items():
        rule = matches[0]
        if "min_width" in rule or "min_height" in rule:
            dimensional[region].append(rule)
    for view in _list(mapping["views"], "views"):
        _object(
            view,
            "mapped view",
            {"image", "part_id", "view_id", "region_id"},
            {"width_px", "height_px", "calibration_id"},
        )
        for key in ("image", "part_id", "view_id", "region_id"):
            _text(view[key], key)
        pid, vid, region = view["part_id"], view["view_id"], view["region_id"]
        if pid not in parts or parts[pid]["expected"].get(vid) != region:
            raise ValueError("view mapping disagrees with expected part/view/region")
        if view["image"] in images or (pid, vid) in seen:
            raise ValueError("duplicate image or part/view mapping")
        seen.add((pid, vid))
        if rules["regions"][region]["adapter"] == "detections":
            for key in ("width_px", "height_px"):
                _integer(view.get(key), key, 1)
                _number(view[key], key, maximum=2**31 - 1)
        elif "width_px" in view or "height_px" in view:
            raise ValueError("classification views do not use image geometry")
        calibration_id = view.get("calibration_id")
        if calibration_id is not None:
            _text(calibration_id, "calibration_id")
            if calibration_id not in rules["calibrations"]:
                raise ValueError("unknown calibration_id")
        if any(rule["unit"] == "mm" for rule in dimensional[region]):
            if calibration_id is None:
                raise ValueError("missing calibration for a dimensional mm rule")
        for rule in dimensional[region]:
            _normalized_limits(rule, view, rules)
        images[view["image"]] = view
    return parts, images


def _normalized_limits(rule: dict, view: dict, rules: dict) -> tuple[float, float]:
    sx = sy = 1.0
    if rule["unit"] == "mm":
        calibration = rules["calibrations"][view["calibration_id"]]
        sx, sy = calibration["x"], calibration["y"]
    scales = (view["width_px"] * sx, view["height_px"] * sy)
    if not all(math.isfinite(scale) and scale > 0 for scale in scales):
        raise ValueError("calibrated dimensions overflow or underflow")
    limits = tuple(
        rule.get(key, 0) / scale
        for key, scale in zip(("min_width", "min_height"), scales)
    )
    if any(
        not math.isfinite(limit) or (limit == 0 and rule.get(key, 0) > 0)
        for key, limit in zip(("min_width", "min_height"), limits)
    ):
        raise ValueError("normalized dimension limits overflow or underflow")
    return limits


def _validate_predictions(
    predictions: Any, images: dict, rules: dict, protocol: dict
) -> dict:
    # Keep the existing Stable schema untouched. A top-1 classification adapter
    # is explicitly separate and never fabricates localization boxes.
    if isinstance(predictions, dict):
        if "schema_version" in predictions and (
            type(predictions["schema_version"]) is not int
            or predictions["schema_version"] not in (1, 2)
        ):
            raise ValueError("unsupported predictions schema_version")
        if "predictions" not in predictions:
            raise ValueError("predictions must be a list or a wrapped predictions list")
        predictions = predictions["predictions"]
    index = {}
    for entry in _list(predictions, "predictions"):
        if not isinstance(entry, dict):
            raise ValueError("prediction entry must be an object")
        image = _text(entry.get("image"), "prediction image")
        if image in index or image not in images:
            raise ValueError("duplicate prediction image or image without a mapping")
        if "schema_version" in entry and (
            type(entry["schema_version"]) is not int
            or entry["schema_version"] not in (1, 2)
        ):
            raise ValueError("unsupported prediction entry schema_version")
        adapter = rules["regions"][images[image]["region_id"]]["adapter"]
        if adapter == "detections":
            if "classification" in entry:
                raise ValueError("classification supplied to detections adapter")
            items = _list(
                entry.get("detections"),
                "detections (explicit empty list required for an empty view)",
            )
        else:
            if "detections" in entry:
                raise ValueError("detections supplied to classification adapter")
            items = [
                _object(
                    entry.get("classification"), "classification", {"class_id", "score"}
                )
            ]
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("prediction must be an object")
            cid = _integer(item.get("class_id"), "prediction class_id")
            if str(cid) not in protocol["class_map"]:
                raise ValueError("prediction class_id missing from class_map")
            _number(item.get("score"), "prediction score", maximum=1)
            if adapter == "detections":
                box = _object(item.get("bbox"), "bbox", {"cx", "cy", "w", "h"})
                for key in ("cx", "cy", "w", "h"):
                    value = _number(box[key], f"bbox.{key}", maximum=1)
                    if key in ("w", "h") and value == 0:
                        raise ValueError("bbox dimensions must be positive")
        index[image] = items
    return index


def _view_decision(
    view: dict, items: list, rules: dict, protocol: dict, grouped: dict
) -> dict:
    region = view["region_id"]
    candidates: dict[int, list] = defaultdict(list)
    for item in items:
        if (
            item["class_id"] in rules["regions"][region]["defect_class_ids"]
            and item["score"] >= protocol["score_threshold"]
        ):
            candidates[item["class_id"]].append(item)
    traces = []
    for cid, found in sorted(candidates.items()):
        matches = grouped.get((region, cid), [])
        rule = matches[0] if matches else None
        eligible = len(found)
        if rule is not None and ("min_width" in rule or "min_height" in rule):
            eligible = 0
            # Compare in normalized space so rounding the predicted size to
            # pixels does not promote a just-below-threshold detection.
            min_w, min_h = _normalized_limits(rule, view, rules)
            for item in found:
                if item["bbox"]["w"] >= min_w and item["bbox"]["h"] >= min_h:
                    eligible += 1
        retained = eligible >= (rule["min_count"] if rule else 1)
        traces.append(
            {
                "class_id": cid,
                "candidate_count": len(found),
                "eligible_count": eligible,
                "retained": retained,
                "rule_id": rule["rule_id"] if rule else None,
                "shadowed_rule_ids": [r["rule_id"] for r in matches[1:]],
                "reason": "rule_applied" if rule else "unmatched_defect_retained",
            }
        )
    return {
        "view_id": view["view_id"],
        "region_id": region,
        "baseline_bad": bool(candidates),
        "rules_bad": any(t["retained"] for t in traces),
        "classes": traces,
    }


def _rate(numerator: int, denominator: int, *, independent: bool) -> dict:
    interval = None
    if denominator and independent:
        z = 1.959963984540054
        p = numerator / denominator
        scale = 1 + z * z / denominator
        center = (p + z * z / (2 * denominator)) / scale
        half = (
            z
            * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator**2))
            / scale
        )
        interval = [max(0.0, center - half), min(1.0, center + half)]
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
        "wilson_95": interval,
    }


def _summary(parts: list[dict], arm: str, *, independent: bool) -> dict:
    counts = {
        key: 0 for key in ("TP", "FP", "FN", "TN", "abstained_OK", "abstained_NOK")
    }
    support = Counter(p["label"] for p in parts)
    for part in parts:
        decision, label = part[arm], part["label"]
        key = (
            f"abstained_{label}"
            if decision == "ABSTAIN"
            else {
                ("NOK", "NOK"): "TP",
                ("OK", "NOK"): "FP",
                ("NOK", "OK"): "FN",
                ("OK", "OK"): "TN",
            }[(label, decision)]
        )
        counts[key] += 1
    tp, fp, fn, tn = (counts[k] for k in ("TP", "FP", "FN", "TN"))
    ratios = {
        "nok_recall": (tp, tp + fn),
        "false_reject_rate": (fp, fp + tn),
        "decision_coverage": (tp + fp + fn + tn, len(parts)),
        "nok_detection_yield": (tp, support["NOK"]),
        "correct_accept_yield": (tn, support["OK"]),
    }
    return {
        "counts": counts,
        "support": {"OK": support["OK"], "NOK": support["NOK"], "total": len(parts)},
        "metrics": {k: _rate(*v, independent=independent) for k, v in ratios.items()},
    }


def compare_inspection(
    predictions: Any, mapping: dict, rules: dict, protocol: dict
) -> dict:
    """Compare two arms without mutating inputs. Raises ValueError on bad input."""
    # Also reject non-JSON/nonfinite values in otherwise unused metadata.
    hashes = {
        k: _digest(v)
        for k, v in {
            "predictions": predictions,
            "mapping": mapping,
            "rules": rules,
            "protocol": protocol,
        }.items()
    }
    _validate_protocol(protocol)
    grouped = _validate_rules(rules, protocol)
    parts, images = _validate_mapping(mapping, rules, protocol, grouped)
    index = _validate_predictions(predictions, images, rules, protocol)
    by_part: dict[str, dict[str, dict]] = defaultdict(dict)
    for image, view in images.items():
        if image in index:
            by_part[view["part_id"]][view["view_id"]] = _view_decision(
                view, index[image], rules, protocol, grouped
            )
    results = []
    for pid, part in sorted(parts.items()):
        if part["split"] != protocol["split"]:
            continue
        observed = by_part[pid]
        missing = sorted(set(part["expected"]) - observed.keys())
        traces = [observed[v] for v in sorted(observed)]
        bad = Counter(t["region_id"] for t in traces if t["rules_bad"])
        baseline = "NOK" if any(t["baseline_bad"] for t in traces) else "OK"
        ruled = (
            "NOK"
            if any(
                bad[r] >= config["bad_view_threshold"]
                for r, config in rules["regions"].items()
            )
            else "OK"
        )
        results.append(
            {
                "part_id": pid,
                "label": part["label"],
                "baseline": "ABSTAIN" if missing else baseline,
                "rules": "ABSTAIN" if missing else ruled,
                "missing_view_ids": missing,
                "reason": "incomplete_views" if missing else "complete_views",
                "views": traces,
                "rules_bad_view_counts": {r: bad[r] for r in sorted(rules["regions"])},
            }
        )
    independent = protocol["sampling"] == "independent_parts"
    common = [
        p for p in results if p["baseline"] != "ABSTAIN" and p["rules"] != "ABSTAIN"
    ]
    transitions = {
        f"{label}:{before}->{after}": 0
        for label in ("OK", "NOK")
        for before in DECISIONS
        for after in DECISIONS
    }
    for part in results:
        transitions[f"{part['label']}:{part['baseline']}->{part['rules']}"] += 1
    paired = {
        "common_decided_support": len(common),
        "transitions": transitions,
        "repaired_false_rejects": transitions["OK:NOK->OK"],
        "introduced_false_accepts": transitions["NOK:NOK->OK"],
        "repaired_false_accepts": transitions["NOK:OK->NOK"],
        "introduced_false_rejects": transitions["OK:OK->NOK"],
        **{
            arm: _summary(common, arm, independent=independent)
            for arm in ("baseline", "rules")
        },
    }
    return {
        "schema_version": 1,
        "kind": "yolozu_part_inspection_comparison",
        "maturity": "experimental",
        "outcome": "comparison_only",
        "production_qualified": False,
        "evidence_kind": protocol["evidence_kind"],
        "protocol_id": protocol["protocol_id"],
        "ruleset_id": rules["ruleset_id"],
        "rules_version": rules["version"],
        "sampling": {
            "unit": "physical_part",
            "assumption": protocol["sampling"],
            "interval_method": "Wilson score, two-sided 95%" if independent else None,
            "note": "Intervals assume independent parts; correlated lots need a separate analysis. No interval is a zero-miss guarantee.",
        },
        "provenance": {
            "engine_version": 1,
            "canonical_input_sha256": hashes,
            "comparison_sha256": _digest({"engine_version": 1, "inputs": hashes}),
            "models": copy.deepcopy(protocol["models"]),
            "class_map_sha256": _digest(protocol["class_map"]),
            "cohort_labels_sha256": _digest(
                [{k: p[k] for k in ("part_id", "label")} for p in results]
            ),
        },
        "arms": {
            arm: _summary(results, arm, independent=independent)
            for arm in ("baseline", "rules")
        },
        "paired": paired,
        "parts": results,
        "image_metrics": {
            "status": "not_computed",
            "note": "Run existing image-level evaluators separately; part labels cannot establish localization quality.",
        },
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def compare_inspection_files(
    *,
    predictions: str | Path,
    mapping: str | Path,
    rules: str | Path,
    protocol: str | Path,
) -> dict:
    """Read bounded JSON snapshots once; record digests of exactly those bytes."""
    payloads, hashes = {}, {}
    for key, path in {
        "predictions": predictions,
        "mapping": mapping,
        "rules": rules,
        "protocol": protocol,
    }.items():
        # Nonblocking open plus fstat rejects FIFOs/devices before any read.
        # This CLI accepts local regular files, not streaming inputs.
        fd = os.open(Path(path), os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{key}: expected a regular file")
            if before.st_size > MAX_INPUT_BYTES:
                raise ValueError(f"{key}: input exceeds {MAX_INPUT_BYTES} bytes")
            raw = handle.read(before.st_size + 1)
            after = os.fstat(handle.fileno())
        if (
            len(raw) != before.st_size
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise ValueError(
                f"{key}: input changed while reading; freeze inputs before evaluation"
            )
        payloads[key] = json.loads(raw, object_pairs_hook=_unique_object)
        hashes[key] = hashlib.sha256(raw).hexdigest()
    report = compare_inspection(**payloads)
    report["provenance"]["input_sha256"] = hashes
    report["provenance"]["comparison_sha256"] = _digest(
        {"engine_version": 1, "inputs": hashes}
    )
    return report
