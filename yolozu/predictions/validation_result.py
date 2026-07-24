from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .predictions import validate_predictions_payload

_MAX_WARNINGS = 100


def _entry_count(payload: Any) -> int | None:
    if isinstance(payload, dict) and "predictions" in payload:
        entries = payload.get("predictions")
        return len(entries) if isinstance(entries, list) else None
    if isinstance(payload, (dict, list)):
        return len(payload)
    return None


def validate_predictions_path(
    path: str | Path,
    *,
    strict: bool = False,
    max_warnings: int | None = _MAX_WARNINGS,
) -> tuple[dict[str, Any], int]:
    """Validate one predictions file and return a stable JSON-ready result."""
    raw_path = str(path)
    result: dict[str, Any] = {
        "schema_version": 1,
        "tool": "validate_predictions",
        "ok": False,
        "path": raw_path,
        "strict": bool(strict),
        "entry_count": None,
        "warnings": [],
        "errors": [],
        "limits": {
            "warnings_max": max_warnings,
            "warnings_truncated": 0,
        },
    }
    resolved = Path(path).expanduser()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        result["entry_count"] = _entry_count(payload)
        validation = validate_predictions_payload(payload, strict=bool(strict))
    except FileNotFoundError as exc:
        result["errors"] = [
            {"code": "file_not_found", "message": str(exc)}
        ]
        return result, 1
    except json.JSONDecodeError as exc:
        result["errors"] = [
            {"code": "invalid_json", "message": str(exc)}
        ]
        return result, 1
    except (OSError, UnicodeError) as exc:
        result["errors"] = [
            {"code": "read_error", "message": str(exc)}
        ]
        return result, 1
    except (TypeError, ValueError) as exc:
        result["errors"] = [
            {"code": "invalid_predictions", "message": str(exc)}
        ]
        return result, 1

    result["ok"] = True
    warnings = list(validation.warnings)
    if max_warnings is None:
        result["warnings"] = warnings
    else:
        limit = max(0, int(max_warnings))
        result["warnings"] = warnings[:limit]
        result["limits"]["warnings_truncated"] = max(
            0,
            len(warnings) - limit,
        )
    return result, 0
