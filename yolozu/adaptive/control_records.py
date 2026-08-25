"""Bounded JSON and JSONL parsing for adaptive-routing control records."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

__all__ = [
    "MAX_CONTROL_DEPTH",
    "MAX_CONTROL_INTEGER_BYTES",
    "MAX_CONTROL_KEY_BYTES",
    "MAX_CONTROL_NODES",
    "MAX_CONTROL_RECORD_BYTES",
    "MAX_CONTROL_STRING_BYTES",
    "MAX_CONTROL_STREAM_BYTES",
    "load_bounded_json",
    "load_bounded_json_bytes",
    "load_bounded_jsonl",
    "load_bounded_jsonl_bytes",
]


MAX_CONTROL_RECORD_BYTES = 4 * 1024 * 1024
MAX_CONTROL_STREAM_BYTES = 128 * 1024 * 1024
MAX_CONTROL_DEPTH = 64
MAX_CONTROL_NODES = 100_000
MAX_CONTROL_KEY_BYTES = 256
MAX_CONTROL_STRING_BYTES = 1024 * 1024
MAX_CONTROL_INTEGER_BYTES = 128

_NUMBER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)")


class _BoundedParser:
    """Small JSON parser that enforces bounds while constructing the value."""

    def __init__(self, data: bytes, *, label: str) -> None:
        if len(data) > MAX_CONTROL_RECORD_BYTES:
            raise ValueError(f"{label}: control record exceeds 4 MiB")
        try:
            self.text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{label}: expected UTF-8") from exc
        self.label = label
        self.position = 0
        self.nodes = 0

    def parse(self) -> Any:
        self._space()
        value = self._value(depth=0)
        self._space()
        if self.position != len(self.text):
            raise ValueError(f"{self.label}: trailing JSON data")
        return value

    def _space(self) -> None:
        while self.position < len(self.text) and self.text[self.position] in " \t\r\n":
            self.position += 1

    def _count(self) -> None:
        self.nodes += 1
        if self.nodes > MAX_CONTROL_NODES:
            raise ValueError(f"{self.label}: control record exceeds node limit")

    def _value(self, *, depth: int) -> Any:
        if depth > MAX_CONTROL_DEPTH:
            raise ValueError(f"{self.label}: control record exceeds depth limit")
        if self.position >= len(self.text):
            raise ValueError(f"{self.label}: unexpected end of JSON")
        token = self.text[self.position]
        if token == "{":
            if depth >= MAX_CONTROL_DEPTH:
                raise ValueError(f"{self.label}: control record exceeds depth limit")
            return self._object(depth=depth + 1)
        if token == "[":
            if depth >= MAX_CONTROL_DEPTH:
                raise ValueError(f"{self.label}: control record exceeds depth limit")
            return self._array(depth=depth + 1)
        if token == '"':
            return self._string(maximum=MAX_CONTROL_STRING_BYTES, field="string")
        if self.text.startswith("true", self.position):
            self.position += 4
            return True
        if self.text.startswith("false", self.position):
            self.position += 5
            return False
        if self.text.startswith("null", self.position):
            self.position += 4
            return None
        return self._integer()

    def _string(self, *, maximum: int, field: str) -> str:
        start = self.position
        self.position += 1
        pieces: list[str] = []
        byte_count = 0

        def append(piece: str) -> None:
            nonlocal byte_count
            byte_count += len(piece.encode("utf-8"))
            if byte_count > maximum:
                raise ValueError(f"{self.label}: {field} exceeds byte limit")
            pieces.append(piece)

        while self.position < len(self.text):
            character = self.text[self.position]
            if character == '"':
                self.position += 1
                return "".join(pieces)
            if ord(character) <= 0x1F:
                raise ValueError(f"{self.label}: unescaped control character in string")
            if character != "\\":
                if 0xD800 <= ord(character) <= 0xDFFF:
                    raise ValueError(f"{self.label}: surrogate code point is invalid")
                append(character)
                self.position += 1
                continue
            self.position += 1
            if self.position >= len(self.text):
                break
            escape = self.text[self.position]
            self.position += 1
            simple = {
                '"': '"',
                "\\": "\\",
                "/": "/",
                "b": "\b",
                "f": "\f",
                "n": "\n",
                "r": "\r",
                "t": "\t",
            }
            if escape in simple:
                append(simple[escape])
                continue
            if escape != "u" or self.position + 4 > len(self.text):
                break
            raw = self.text[self.position : self.position + 4]
            if not all(character in "0123456789abcdefABCDEF" for character in raw):
                break
            self.position += 4
            codepoint = int(raw, 16)
            if 0xD800 <= codepoint <= 0xDBFF:
                if not self.text.startswith("\\u", self.position):
                    raise ValueError(f"{self.label}: unpaired surrogate escape")
                self.position += 2
                low_raw = self.text[self.position : self.position + 4]
                if len(low_raw) != 4 or not all(
                    character in "0123456789abcdefABCDEF" for character in low_raw
                ):
                    raise ValueError(f"{self.label}: invalid surrogate escape")
                self.position += 4
                low = int(low_raw, 16)
                if not 0xDC00 <= low <= 0xDFFF:
                    raise ValueError(f"{self.label}: invalid surrogate pair")
                codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)
            elif 0xDC00 <= codepoint <= 0xDFFF:
                raise ValueError(f"{self.label}: unpaired surrogate escape")
            append(chr(codepoint))
        self.position = start
        raise ValueError(f"{self.label}: malformed JSON string")

    def _integer(self) -> int:
        match = _NUMBER_RE.match(self.text, self.position)
        if match is None:
            raise ValueError(f"{self.label}: unsupported JSON token")
        end = match.end()
        if end < len(self.text) and self.text[end] in ".eE":
            raise ValueError(f"{self.label}: binary/fractional JSON numbers are invalid")
        token = match.group(0)
        if len(token.encode("ascii")) > MAX_CONTROL_INTEGER_BYTES:
            raise ValueError(f"{self.label}: integer token exceeds byte limit")
        if token == "-0":
            raise ValueError(f"{self.label}: negative zero is non-canonical")
        self.position = end
        return int(token)

    def _object(self, *, depth: int) -> dict[str, Any]:
        self.position += 1
        self._space()
        result: dict[str, Any] = {}
        if self.position < len(self.text) and self.text[self.position] == "}":
            self.position += 1
            return result
        while True:
            if self.position >= len(self.text) or self.text[self.position] != '"':
                raise ValueError(f"{self.label}: object key must be a string")
            key = self._string(maximum=MAX_CONTROL_KEY_BYTES, field="object key")
            if key in result:
                raise ValueError(f"{self.label}: duplicate object key {key!r}")
            self._space()
            if self.position >= len(self.text) or self.text[self.position] != ":":
                raise ValueError(f"{self.label}: missing object colon")
            self.position += 1
            self._space()
            self._count()
            result[key] = self._value(depth=depth)
            self._space()
            if self.position >= len(self.text):
                raise ValueError(f"{self.label}: unterminated object")
            delimiter = self.text[self.position]
            self.position += 1
            if delimiter == "}":
                return result
            if delimiter != ",":
                raise ValueError(f"{self.label}: invalid object delimiter")
            self._space()

    def _array(self, *, depth: int) -> list[Any]:
        self.position += 1
        self._space()
        result: list[Any] = []
        if self.position < len(self.text) and self.text[self.position] == "]":
            self.position += 1
            return result
        while True:
            self._count()
            result.append(self._value(depth=depth))
            self._space()
            if self.position >= len(self.text):
                raise ValueError(f"{self.label}: unterminated array")
            delimiter = self.text[self.position]
            self.position += 1
            if delimiter == "]":
                return result
            if delimiter != ",":
                raise ValueError(f"{self.label}: invalid array delimiter")
            self._space()


def load_bounded_json_bytes(data: bytes, *, label: str = "control JSON") -> Any:
    """Parse one bounded integer/string JSON control record."""

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    return _BoundedParser(data, label=label).parse()


def load_bounded_json(path: Path, *, label: str = "control JSON") -> Any:
    """Read and parse one bounded JSON file without following a file symlink."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label}: expected a regular non-symlink file")
    size = candidate.stat().st_size
    if size > MAX_CONTROL_RECORD_BYTES:
        raise ValueError(f"{label}: control record exceeds 4 MiB")
    return load_bounded_json_bytes(candidate.read_bytes(), label=label)


def load_bounded_jsonl_bytes(
    data: bytes,
    *,
    label: str = "control JSONL",
    max_records: int,
) -> list[Any]:
    """Parse a bounded JSONL control stream and reject blank lines."""

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if len(data) > MAX_CONTROL_STREAM_BYTES:
        raise ValueError(f"{label}: stream exceeds 128 MiB")
    if not data:
        return []
    lines = data.splitlines()
    if len(lines) > max_records:
        raise ValueError(f"{label}: record count exceeds {max_records}")
    records: list[Any] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{label}:{line_number}: blank line is invalid")
        records.append(
            load_bounded_json_bytes(line, label=f"{label}:{line_number}")
        )
    return records


def load_bounded_jsonl(
    path: Path,
    *,
    label: str = "control JSONL",
    max_records: int,
) -> list[Any]:
    """Read and parse one bounded JSONL stream."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label}: expected a regular non-symlink file")
    size = candidate.stat().st_size
    if size > MAX_CONTROL_STREAM_BYTES:
        raise ValueError(f"{label}: stream exceeds 128 MiB")
    return load_bounded_jsonl_bytes(
        candidate.read_bytes(), label=label, max_records=max_records
    )
