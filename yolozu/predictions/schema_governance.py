"""Prediction payload schema-version governance.

Validates ``schema_version`` lifecycle and compatibility rules for
wrapped prediction payloads.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "MIN_SUPPORTED_SCHEMA_VERSION",
    "validate_payload_schema_version",
]

CURRENT_SCHEMA_VERSION = 1
MIN_SUPPORTED_SCHEMA_VERSION = 1


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_payload_schema_version(payload: Any, *, artifact: str) -> list[str]:
    """Validate schema_version lifecycle and compatibility rules.

    Compatibility rules:
    - Wrapped payload (object with 'predictions') may omit schema_version for legacy compatibility.
    - When present, schema_version must be an integer in [MIN_SUPPORTED_SCHEMA_VERSION, CURRENT_SCHEMA_VERSION].
    - Future schema versions are rejected until the validator is upgraded.
    """

    warnings: list[str] = []
    if not isinstance(payload, dict) or "predictions" not in payload:
        return warnings

    if "schema_version" not in payload:
        warnings.append(
            f"{artifact}: schema_version missing in wrapped payload; treating as legacy compatibility mode (v{CURRENT_SCHEMA_VERSION})"
        )
        return warnings

    version = payload.get("schema_version")
    if not _is_int(version):
        raise ValueError(f"{artifact}: schema_version must be an integer")

    if version < MIN_SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"{artifact}: schema_version {version} is older than minimum supported {MIN_SUPPORTED_SCHEMA_VERSION}"
        )
    if version > CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"{artifact}: schema_version {version} is newer than supported {CURRENT_SCHEMA_VERSION}; upgrade YOLOZU"
        )

    return warnings
