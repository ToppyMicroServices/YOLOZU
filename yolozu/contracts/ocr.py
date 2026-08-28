"""Strict, model-free OCR result interface contract v1."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Sequence

from yolozu.adaptive.canonical import (
    canonical_decimal_v1,
    canonical_json_v1,
    canonical_sha256_v1,
)

__all__ = [
    "OCRBundleInterface",
    "OCRContractError",
    "OCRResult",
    "map_ocr_runner_result",
    "privacy_safe_ocr_summary",
    "validate_ocr_bundle_interface",
    "validate_ocr_input_media",
    "validate_ocr_result",
]


CONTENT_TRUST = "untrusted_input_derived"
COMPONENT_MODE = "detect_and_recognize"
MAX_TEXT_CODEPOINTS = 4_096
MAX_TEXT_BYTES = 16_384
MAX_ITEMS_PER_IMAGE = 1_000
MAX_TEXT_BYTES_PER_IMAGE = 1_048_576
MAX_ITEMS_PER_JOB = 100_000
MAX_TEXT_BYTES_PER_JOB = 67_108_864
MAX_GEOMETRY_BYTES_PER_ITEM = 512
MAX_GEOMETRY_BYTES_PER_JOB = 67_108_864

_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z", re.ASCII)
_CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")
_COMPONENT_ROLES = ("detector", "recognizer")
_COMPONENT_FAILURES = frozenset({"timeout", "crash", "missing", "invalid_output"})
_SUPPORTED_MEDIA = frozenset({"image/jpeg", "image/png", "image/webp"})


class OCRContractError(ValueError):
    """Stable fail-closed OCR interface-contract violation."""


@dataclass(frozen=True)
class OCRBundleInterface:
    _record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._record)

    @property
    def component_by_role(self) -> dict[str, dict[str, str]]:
        return {
            str(component["role"]): copy.deepcopy(component)
            for component in self._record["ocr_components"]
        }


@dataclass(frozen=True)
class OCRResult:
    _record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._record)

    def canonical_bytes(self) -> bytes:
        return canonical_json_v1(self._record)


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OCRContractError(f"{field}: expected object")
    return dict(value)


def _check_keys(
    value: Mapping[str, Any],
    *,
    field: str,
    allowed: frozenset[str],
    required: frozenset[str],
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise OCRContractError(f"{field}: unknown keys")
    missing = sorted(required - set(value))
    if missing:
        raise OCRContractError(f"{field}: missing required keys")


def _exact_int(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OCRContractError(f"{field}: expected integer")
    if value < minimum or value > maximum:
        raise OCRContractError(f"{field}: out of range")
    return value


def _id(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise OCRContractError(f"{field}: invalid ID")
    return value


def _id_array(value: Any, *, field: str, minimum: int, maximum: int) -> list[str]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise OCRContractError(f"{field}: invalid item count")
    checked = [_id(item, field=f"{field}[]") for item in value]
    if len(set(checked)) != len(checked):
        raise OCRContractError(f"{field}: duplicate ID")
    return checked


def _text(value: Any, *, field: str) -> tuple[str, int]:
    if not isinstance(value, str):
        raise OCRContractError(f"{field}: expected string")
    if "\x00" in value:
        raise OCRContractError(f"{field}: NUL is invalid")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise OCRContractError(f"{field}: invalid Unicode") from exc
    if len(value) > MAX_TEXT_CODEPOINTS:
        raise OCRContractError(f"{field}: code-point limit exceeded")
    if len(encoded) > MAX_TEXT_BYTES:
        raise OCRContractError(f"{field}: UTF-8 byte limit exceeded")
    return value, len(encoded)


def _confidence(value: Any, *, field: str) -> str:
    try:
        token = canonical_decimal_v1(value, field=field, nonnegative=True)
    except ValueError as exc:
        raise OCRContractError(str(exc)) from exc
    if Decimal(token) > 1:
        raise OCRContractError(f"{field}: expected 0..1")
    return token


def _coordinate(value: Any, *, field: str, maximum: int) -> str:
    try:
        token = canonical_decimal_v1(value, field=field, nonnegative=True)
    except ValueError as exc:
        raise OCRContractError(str(exc)) from exc
    if Decimal(token) > maximum:
        raise OCRContractError(f"{field}: outside decoded image")
    return token


def _validate_geometry(
    value: Any, *, width: int, height: int, field: str
) -> dict[str, Any]:
    geometry = _mapping(value, field=field)
    _check_keys(
        geometry,
        field=field,
        allowed=frozenset({"corners"}),
        required=frozenset({"corners"}),
    )
    corners = geometry["corners"]
    if not isinstance(corners, list) or len(corners) != 4:
        raise OCRContractError(f"{field}.corners: expected exactly four corners")

    checked: list[dict[str, str]] = []
    points: list[tuple[Decimal, Decimal]] = []
    for index, (raw, expected_name) in enumerate(zip(corners, _CORNER_NAMES)):
        corner = _mapping(raw, field=f"{field}.corners[{index}]")
        _check_keys(
            corner,
            field=f"{field}.corners[{index}]",
            allowed=frozenset({"corner", "x", "y"}),
            required=frozenset({"corner", "x", "y"}),
        )
        if corner["corner"] != expected_name:
            raise OCRContractError(f"{field}.corners: wrong semantic corner order")
        x = _coordinate(corner["x"], field=f"{field}.{expected_name}.x", maximum=width)
        y = _coordinate(corner["y"], field=f"{field}.{expected_name}.y", maximum=height)
        checked.append({"corner": expected_name, "x": x, "y": y})
        points.append((Decimal(x), Decimal(y)))

    if len(set(points)) != 4:
        raise OCRContractError(f"{field}: corners must be distinct")

    crosses: list[Decimal] = []
    for index in range(4):
        a = points[index]
        b = points[(index + 1) % 4]
        c = points[(index + 2) % 4]
        crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if any(cross <= 0 for cross in crosses):
        raise OCRContractError(
            f"{field}: expected strict clockwise image-coordinate convexity"
        )

    twice_area = sum(
        points[index][0] * points[(index + 1) % 4][1]
        - points[index][1] * points[(index + 1) % 4][0]
        for index in range(4)
    )
    if twice_area <= 0:
        raise OCRContractError(
            f"{field}: expected positive image-coordinate shoelace area"
        )

    sums = [x + y for x, y in points]
    differences = [x - y for x, y in points]
    semantic_extrema = (
        sums.index(min(sums)),
        differences.index(max(differences)),
        sums.index(max(sums)),
        differences.index(min(differences)),
    )
    if semantic_extrema != (0, 1, 2, 3):
        raise OCRContractError(f"{field}: corner values do not match semantic names")

    result = {"corners": checked}
    if len(canonical_json_v1(result)) > MAX_GEOMETRY_BYTES_PER_ITEM:
        raise OCRContractError(f"{field}: canonical geometry byte limit exceeded")
    return result


def validate_ocr_bundle_interface(payload: Mapping[str, Any]) -> OCRBundleInterface:
    record = _mapping(payload, field="ocr_bundle_interface")
    _check_keys(
        record,
        field="ocr_bundle_interface",
        allowed=frozenset(
            {
                "schema_version",
                "bundle_id",
                "component_mode",
                "language_ids",
                "script_ids",
                "ocr_components",
                "bundle_interface_digest",
            }
        ),
        required=frozenset(
            {
                "schema_version",
                "bundle_id",
                "component_mode",
                "language_ids",
                "script_ids",
                "ocr_components",
                "bundle_interface_digest",
            }
        ),
    )
    if record["schema_version"] != 1:
        raise OCRContractError("ocr_bundle_interface.schema_version: expected 1")
    checked: dict[str, Any] = {
        "schema_version": 1,
        "bundle_id": _id(record["bundle_id"], field="bundle_id"),
        "component_mode": record["component_mode"],
        "language_ids": _id_array(
            record["language_ids"], field="language_ids", minimum=1, maximum=256
        ),
        "script_ids": _id_array(
            record["script_ids"], field="script_ids", minimum=1, maximum=128
        ),
    }
    if checked["component_mode"] != COMPONENT_MODE:
        raise OCRContractError("component_mode: unsupported value")
    components = record["ocr_components"]
    if not isinstance(components, list) or len(components) != 2:
        raise OCRContractError("ocr_components: expected exactly two components")
    checked_components: list[dict[str, str]] = []
    component_ids: set[str] = set()
    model_ids: set[str] = set()
    for index, expected_role in enumerate(_COMPONENT_ROLES):
        component = _mapping(components[index], field=f"ocr_components[{index}]")
        _check_keys(
            component,
            field=f"ocr_components[{index}]",
            allowed=frozenset({"role", "component_id", "model_id"}),
            required=frozenset({"role", "component_id", "model_id"}),
        )
        if component["role"] != expected_role:
            raise OCRContractError(
                "ocr_components: roles must be detector then recognizer"
            )
        component_id = _id(component["component_id"], field="component_id")
        model_id = _id(component["model_id"], field="model_id")
        if (
            component_id in component_ids
            or model_id in model_ids
            or component_id == model_id
        ):
            raise OCRContractError(
                "ocr_components: component/model IDs must be distinct"
            )
        component_ids.add(component_id)
        model_ids.add(model_id)
        checked_components.append(
            {"role": expected_role, "component_id": component_id, "model_id": model_id}
        )
    checked["ocr_components"] = checked_components
    digest = record["bundle_interface_digest"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise OCRContractError("bundle_interface_digest: expected lowercase SHA-256")
    checked["bundle_interface_digest"] = digest
    if (
        canonical_sha256_v1(checked, own_digest_field="bundle_interface_digest")
        != digest
    ):
        raise OCRContractError("bundle_interface_digest: mismatch")
    return OCRBundleInterface(checked)


def validate_ocr_input_media(
    *, mime_type: str, frame_count: int, animated: bool, multipage: bool
) -> None:
    if mime_type not in _SUPPORTED_MEDIA:
        raise OCRContractError("ocr_input: unsupported media type")
    if isinstance(animated, bool) is False or isinstance(multipage, bool) is False:
        raise OCRContractError("ocr_input: animation flags must be boolean")
    _exact_int(
        frame_count, field="ocr_input.frame_count", minimum=1, maximum=2_147_483_647
    )
    if frame_count != 1 or animated or multipage:
        raise OCRContractError("ocr_input: only one decoded still image is supported")


def _map_index(value: Any, *, values: Sequence[str], field: str) -> dict[str, str]:
    if value == "unknown":
        return {"status": "unknown"}
    index = _exact_int(value, field=field, minimum=0, maximum=len(values) - 1)
    return {"status": "known", "id": values[index]}


def _validate_optional_fields(
    region: Mapping[str, Any], *, item: dict[str, Any]
) -> None:
    reading_order = region.get("reading_order", "unknown")
    if reading_order != "unknown":
        reading_order = _exact_int(
            reading_order, field="reading_order", minimum=0, maximum=99_999
        )
    item["reading_order"] = reading_order
    orientation = region.get("orientation", "unknown")
    if orientation not in {"unknown", "0", "90", "180", "270"}:
        raise OCRContractError("orientation: unsupported value")
    item["orientation"] = orientation


def map_ocr_runner_result(
    bundle_payload: Mapping[str, Any],
    runner_payload: Mapping[str, Any],
    *,
    image_bounds: Sequence[tuple[int, int]],
    logical_page_references: Sequence[int | None] | None = None,
) -> OCRResult:
    """Validate untrusted runner output, then stamp pinned provenance atomically."""

    bundle = validate_ocr_bundle_interface(bundle_payload)
    runner = _mapping(runner_payload, field="ocr_runner_result")
    _check_keys(
        runner,
        field="ocr_runner_result",
        allowed=frozenset(
            {
                "schema_version",
                "component_mode",
                "detector_status",
                "recognizer_status",
                "regions",
            }
        ),
        required=frozenset(
            {
                "schema_version",
                "component_mode",
                "detector_status",
                "recognizer_status",
                "regions",
            }
        ),
    )
    if runner["schema_version"] != 1 or runner["component_mode"] != COMPONENT_MODE:
        raise OCRContractError("ocr_runner_result: unsupported version or mode")
    for role in _COMPONENT_ROLES:
        status = runner[f"{role}_status"]
        if status != "succeeded":
            if status not in _COMPONENT_FAILURES:
                raise OCRContractError(f"ocr_component_{role}_invalid_output")
            raise OCRContractError(f"ocr_component_{role}_{status}")
    regions = runner["regions"]
    if not isinstance(regions, list):
        raise OCRContractError("regions: expected array")
    if len(regions) > MAX_ITEMS_PER_JOB:
        raise OCRContractError("regions: job item limit exceeded")
    if not image_bounds:
        raise OCRContractError("image_bounds: at least one decoded image is required")
    if logical_page_references is None:
        expected_pages: list[int | None] = [None] * len(image_bounds)
    else:
        if len(logical_page_references) != len(image_bounds):
            raise OCRContractError("logical_page_references: image-count mismatch")
        expected_pages = []
        for index, page in enumerate(logical_page_references):
            if page is not None:
                page = _exact_int(
                    page,
                    field=f"logical_page_references[{index}]",
                    minimum=1,
                    maximum=2_147_483_647,
                )
            expected_pages.append(page)

    bundle_record = bundle.to_dict()
    language_ids = bundle_record["language_ids"]
    script_ids = bundle_record["script_ids"]
    components = bundle.component_by_role
    image_counts = [0] * len(image_bounds)
    image_text_bytes = [0] * len(image_bounds)
    seen_pages: dict[int, int | None] = {}
    items: list[dict[str, Any]] = []
    total_text_bytes = 0
    total_geometry_bytes = 0
    known_language_count = 0
    known_script_count = 0

    allowed_region_keys = frozenset(
        {
            "recognized_text",
            "quadrilateral",
            "detection_confidence",
            "recognition_confidence",
            "language_index",
            "script_index",
            "input_image_index",
            "logical_page_reference",
            "reading_order",
            "orientation",
        }
    )
    required_region_keys = frozenset(
        {
            "recognized_text",
            "quadrilateral",
            "detection_confidence",
            "recognition_confidence",
            "language_index",
            "script_index",
            "input_image_index",
        }
    )
    for region_index, raw_region in enumerate(regions):
        region = _mapping(raw_region, field=f"regions[{region_index}]")
        _check_keys(
            region,
            field=f"regions[{region_index}]",
            allowed=allowed_region_keys,
            required=required_region_keys,
        )
        image_index = _exact_int(
            region["input_image_index"],
            field="input_image_index",
            minimum=0,
            maximum=len(image_bounds) - 1,
        )
        width, height = image_bounds[image_index]
        width = _exact_int(width, field="image_width", minimum=1, maximum=2_147_483_647)
        height = _exact_int(
            height, field="image_height", minimum=1, maximum=2_147_483_647
        )
        page = region.get("logical_page_reference")
        if page is not None:
            page = _exact_int(
                page, field="logical_page_reference", minimum=1, maximum=2_147_483_647
            )
        if image_index in seen_pages and seen_pages[image_index] != page:
            raise OCRContractError(
                "logical_page_reference: inconsistent for decoded input"
            )
        seen_pages[image_index] = page
        if page != expected_pages[image_index]:
            raise OCRContractError("logical_page_reference: input mapping mismatch")
        recognized_text, text_bytes = _text(
            region["recognized_text"], field="recognized_text"
        )
        geometry = _validate_geometry(
            region["quadrilateral"], width=width, height=height, field="quadrilateral"
        )
        geometry_bytes = len(canonical_json_v1(geometry))
        image_counts[image_index] += 1
        image_text_bytes[image_index] += text_bytes
        total_text_bytes += text_bytes
        total_geometry_bytes += geometry_bytes
        if image_counts[image_index] > MAX_ITEMS_PER_IMAGE:
            raise OCRContractError("regions: per-image item limit exceeded")
        if image_text_bytes[image_index] > MAX_TEXT_BYTES_PER_IMAGE:
            raise OCRContractError("recognized_text: per-image byte limit exceeded")
        if total_text_bytes > MAX_TEXT_BYTES_PER_JOB:
            raise OCRContractError("recognized_text: job byte limit exceeded")
        if total_geometry_bytes > MAX_GEOMETRY_BYTES_PER_JOB:
            raise OCRContractError("quadrilateral: job byte limit exceeded")
        language = _map_index(
            region["language_index"], values=language_ids, field="language_index"
        )
        script = _map_index(
            region["script_index"], values=script_ids, field="script_index"
        )
        known_language_count += language["status"] == "known"
        known_script_count += script["status"] == "known"
        item: dict[str, Any] = {
            "input_image_index": image_index,
            "logical_page_reference": page,
            "recognized_text": recognized_text,
            "content_trust": CONTENT_TRUST,
            "quadrilateral": geometry,
            "detection_confidence": _confidence(
                region["detection_confidence"], field="detection_confidence"
            ),
            "recognition_confidence": _confidence(
                region["recognition_confidence"], field="recognition_confidence"
            ),
            "detected_language": language,
            "detected_script": script,
            "components": copy.deepcopy(components),
        }
        _validate_optional_fields(region, item=item)
        items.append(item)

    summary = {
        "image_count": len(image_bounds),
        "item_count": len(items),
        "images_with_text_count": sum(count > 0 for count in image_counts),
        "known_language_item_count": known_language_count,
        "known_script_item_count": known_script_count,
        "recognized_text_included": False,
    }
    result = {
        "schema_version": 1,
        "kind": "ocr_result",
        "component_mode": COMPONENT_MODE,
        "content_trust": CONTENT_TRUST,
        "bundle_interface_digest": bundle_record["bundle_interface_digest"],
        "image_count": len(image_bounds),
        "component_outcomes": {"detector": "succeeded", "recognizer": "succeeded"},
        "items": items,
        "privacy_safe_summary": summary,
    }
    return validate_ocr_result(
        result, bundle_payload=bundle_record, image_bounds=image_bounds
    )


def validate_ocr_result(
    payload: Mapping[str, Any],
    *,
    bundle_payload: Mapping[str, Any],
    image_bounds: Sequence[tuple[int, int]],
) -> OCRResult:
    """Revalidate one successful core-produced result without runner trust claims."""

    bundle = validate_ocr_bundle_interface(bundle_payload)
    record = _mapping(payload, field="ocr_result")
    _check_keys(
        record,
        field="ocr_result",
        allowed=frozenset(
            {
                "schema_version",
                "kind",
                "component_mode",
                "content_trust",
                "bundle_interface_digest",
                "image_count",
                "component_outcomes",
                "items",
                "privacy_safe_summary",
            }
        ),
        required=frozenset(
            {
                "schema_version",
                "kind",
                "component_mode",
                "content_trust",
                "bundle_interface_digest",
                "image_count",
                "component_outcomes",
                "items",
                "privacy_safe_summary",
            }
        ),
    )
    if record["schema_version"] != 1 or record["kind"] != "ocr_result":
        raise OCRContractError("ocr_result: unsupported version or kind")
    if (
        record["component_mode"] != COMPONENT_MODE
        or record["content_trust"] != CONTENT_TRUST
    ):
        raise OCRContractError("ocr_result: invalid fixed trust or component mode")
    if record["bundle_interface_digest"] != bundle.to_dict()["bundle_interface_digest"]:
        raise OCRContractError("ocr_result: bundle interface mismatch")
    if record["image_count"] != len(image_bounds):
        raise OCRContractError("ocr_result: image count mismatch")
    if record["component_outcomes"] != {
        "detector": "succeeded",
        "recognizer": "succeeded",
    }:
        raise OCRContractError("ocr_result: incomplete components cannot publish")
    items = record["items"]
    if not isinstance(items, list) or len(items) > MAX_ITEMS_PER_JOB:
        raise OCRContractError("ocr_result.items: invalid item count")
    expected_components = bundle.component_by_role
    image_counts = [0] * len(image_bounds)
    image_text_bytes = [0] * len(image_bounds)
    total_text_bytes = 0
    total_geometry_bytes = 0
    known_languages = 0
    known_scripts = 0
    checked_items: list[dict[str, Any]] = []
    for index, raw_item in enumerate(items):
        item = _mapping(raw_item, field=f"ocr_result.items[{index}]")
        required = frozenset(
            {
                "input_image_index",
                "logical_page_reference",
                "recognized_text",
                "content_trust",
                "quadrilateral",
                "detection_confidence",
                "recognition_confidence",
                "detected_language",
                "detected_script",
                "components",
                "reading_order",
                "orientation",
            }
        )
        _check_keys(
            item,
            field=f"ocr_result.items[{index}]",
            allowed=required,
            required=required,
        )
        image_index = _exact_int(
            item["input_image_index"],
            field="input_image_index",
            minimum=0,
            maximum=len(image_bounds) - 1,
        )
        page = item["logical_page_reference"]
        if page is not None:
            _exact_int(
                page, field="logical_page_reference", minimum=1, maximum=2_147_483_647
            )
        text, text_bytes = _text(item["recognized_text"], field="recognized_text")
        width, height = image_bounds[image_index]
        geometry = _validate_geometry(
            item["quadrilateral"], width=width, height=height, field="quadrilateral"
        )
        if item["content_trust"] != CONTENT_TRUST:
            raise OCRContractError("content_trust: fixed value required")
        language = _mapping(item["detected_language"], field="detected_language")
        script = _mapping(item["detected_script"], field="detected_script")
        for detected, allowed, field in (
            (language, bundle.to_dict()["language_ids"], "detected_language"),
            (script, bundle.to_dict()["script_ids"], "detected_script"),
        ):
            if detected == {"status": "unknown"}:
                continue
            if (
                set(detected) != {"status", "id"}
                or detected.get("status") != "known"
                or detected.get("id") not in allowed
            ):
                raise OCRContractError(f"{field}: invalid core mapping")
        if item["components"] != expected_components:
            raise OCRContractError("components: expected exact core-stamped provenance")
        normalized = copy.deepcopy(item)
        normalized["recognized_text"] = text
        normalized["quadrilateral"] = geometry
        normalized["detection_confidence"] = _confidence(
            item["detection_confidence"], field="detection_confidence"
        )
        normalized["recognition_confidence"] = _confidence(
            item["recognition_confidence"], field="recognition_confidence"
        )
        _validate_optional_fields(item, item=normalized)
        checked_items.append(normalized)
        image_counts[image_index] += 1
        image_text_bytes[image_index] += text_bytes
        total_text_bytes += text_bytes
        total_geometry_bytes += len(canonical_json_v1(geometry))
        known_languages += language.get("status") == "known"
        known_scripts += script.get("status") == "known"
    if any(count > MAX_ITEMS_PER_IMAGE for count in image_counts):
        raise OCRContractError("ocr_result.items: per-image item limit exceeded")
    if any(count > MAX_TEXT_BYTES_PER_IMAGE for count in image_text_bytes):
        raise OCRContractError("recognized_text: per-image byte limit exceeded")
    if (
        total_text_bytes > MAX_TEXT_BYTES_PER_JOB
        or total_geometry_bytes > MAX_GEOMETRY_BYTES_PER_JOB
    ):
        raise OCRContractError("ocr_result: aggregate byte limit exceeded")
    expected_summary = {
        "image_count": len(image_bounds),
        "item_count": len(checked_items),
        "images_with_text_count": sum(count > 0 for count in image_counts),
        "known_language_item_count": known_languages,
        "known_script_item_count": known_scripts,
        "recognized_text_included": False,
    }
    if record["privacy_safe_summary"] != expected_summary:
        raise OCRContractError("privacy_safe_summary: mismatch")
    checked = copy.deepcopy(record)
    checked["items"] = checked_items
    checked["privacy_safe_summary"] = expected_summary
    return OCRResult(checked)


def privacy_safe_ocr_summary(result: OCRResult) -> dict[str, Any]:
    """Return the bounded aggregate summary; recognized content is excluded."""

    return copy.deepcopy(result.to_dict()["privacy_safe_summary"])
