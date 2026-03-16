from __future__ import annotations

import json
from typing import Any

try:
    import atheris  # type: ignore
except ImportError:  # pragma: no cover
    atheris = None

from yolozu.predictions import canonicalize_predictions, normalize_predictions_payload, validate_predictions_payload


def consume_input(data: bytes) -> None:
    try:
        text = data.decode("utf-8", errors="ignore")
        if not text.strip():
            return
        payload: Any = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return

    try:
        entries, _meta = normalize_predictions_payload(payload)
        canonicalize_predictions(entries, strict=False, policy="clamp")
        validate_predictions_payload(payload, strict=False)
    except (TypeError, ValueError):
        return


def main() -> None:
    if atheris is None:  # pragma: no cover
        raise SystemExit("atheris is required to run fuzz targets")
    atheris.Setup([], consume_input)
    atheris.Fuzz()


if __name__ == "__main__":  # pragma: no cover
    main()
