"""Canonical serialization primitives for adaptive-routing records."""

from __future__ import annotations

import hashlib
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

__all__ = [
    "CANONICAL_DECIMAL_V1_PATTERN",
    "canonical_decimal_v1",
    "canonical_json_v1",
    "canonical_sha256_v1",
]


CANONICAL_DECIMAL_V1_PATTERN = re.compile(
    r"-?(?:0|[1-9][0-9]{0,17})(?:\.[0-9]{0,8}[1-9])?\Z"
)

_SHORT_ESCAPES = {
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def canonical_decimal_v1(
    value: Any,
    *,
    field: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> str:
    """Validate and return one CanonicalDecimalV1 token.

    Binary floats are intentionally rejected. Callers must provide the exact
    decimal token that will be compared and persisted.
    """

    if not isinstance(value, str) or not CANONICAL_DECIMAL_V1_PATTERN.fullmatch(value):
        raise ValueError(f"{field}: expected CanonicalDecimalV1 string")
    if len(value.encode("ascii")) > 29:
        raise ValueError(f"{field}: CanonicalDecimalV1 token exceeds 29 bytes")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:  # pragma: no cover - regex is the primary gate
        raise ValueError(f"{field}: invalid decimal") from exc
    if number.is_zero() and value != "0":
        raise ValueError(f"{field}: numeric zero is encoded exactly as '0'")
    if positive and number <= 0:
        raise ValueError(f"{field}: expected a positive decimal")
    if nonnegative and number < 0:
        raise ValueError(f"{field}: expected a nonnegative decimal")
    return value


def _escape_string(value: str) -> bytes:
    pieces: list[str] = ['"']
    for character in value:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise ValueError("canonical_json_v1: surrogate code points are invalid")
        if character == '"':
            pieces.append('\\"')
        elif character == "\\":
            pieces.append("\\\\")
        elif character in _SHORT_ESCAPES:
            pieces.append(_SHORT_ESCAPES[character])
        elif codepoint <= 0x1F:
            pieces.append(f"\\u00{codepoint:02x}")
        else:
            pieces.append(character)
    pieces.append('"')
    return "".join(pieces).encode("utf-8")


def _serialize(value: Any) -> bytes:
    if value is None:
        return b"null"
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if isinstance(value, int):
        return str(value).encode("ascii")
    if isinstance(value, float):
        raise ValueError("canonical_json_v1: binary floats are invalid")
    if isinstance(value, str):
        return _escape_string(value)
    if isinstance(value, Mapping):
        pairs: list[tuple[bytes, bytes]] = []
        seen: set[str] = set()
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("canonical_json_v1: object keys must be strings")
            if key in seen:
                raise ValueError(f"canonical_json_v1: duplicate object key {key!r}")
            seen.add(key)
            key_bytes = key.encode("utf-8")
            pairs.append((key_bytes, _escape_string(key) + b":" + _serialize(item)))
        pairs.sort(key=lambda pair: pair[0])
        return b"{" + b",".join(serialized for _, serialized in pairs) + b"}"
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        return b"[" + b",".join(_serialize(item) for item in value) + b"]"
    raise ValueError(
        "canonical_json_v1: expected null, bool, int, string, array, or object; "
        f"got {type(value).__name__}"
    )


def canonical_json_v1(value: Any) -> bytes:
    """Return the exact UTF-8 canonical JSON representation for *value*."""

    return _serialize(value)


def canonical_sha256_v1(
    value: Mapping[str, Any] | Sequence[Any],
    *,
    own_digest_field: str | None = None,
) -> str:
    """Hash a canonical record, optionally omitting its own digest field."""

    digest_value: Any = value
    if own_digest_field is not None:
        if not isinstance(value, Mapping):
            raise ValueError("own_digest_field is valid only for an object record")
        digest_value = {key: item for key, item in value.items() if key != own_digest_field}
    return hashlib.sha256(canonical_json_v1(digest_value)).hexdigest()
