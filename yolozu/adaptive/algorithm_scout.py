"""Read-only monitored-source inbox for Experimental algorithm intake."""

from __future__ import annotations

import hashlib
import errno
import json
import multiprocessing
import os
import re
import resource
import shutil
import signal
import socket
import stat
import struct
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

from .canonical import canonical_json_v1
from .control_records import load_bounded_json, load_bounded_json_bytes
from .managed_output import (
    ManagedOutputLimits,
    ManagedOutputTransaction,
    validate_managed_output_destination,
)
from .safe_https import (
    FetchedDocument,
    HttpsLocation,
    SafeHttpsError,
    SafeHttpsTransport,
    TransportLimits,
)

__all__ = [
    "AlgorithmScoutError",
    "AlgorithmScoutSource",
    "DocumentParserLimits",
    "ScoutPlan",
    "build_scout_plan",
    "collect_algorithm_candidates",
    "load_algorithm_scout_sources",
]


CANONICAL_SOURCES_PATH = PurePosixPath("docs/algorithm_intake/sources.json")
REPORT_KIND = "yolozu_algorithm_scout_report"
REPORT_SCHEMA_VERSION = 1
COLLECTION_TIMEOUT_SECONDS = 12 * 60
WORKFLOW_RESERVE_SECONDS = 3 * 60
TOTAL_DECODED_BYTES = 512 * 1024 * 1024
MAX_SOURCES = 128
MAX_PRIOR_REPORTS = 104
MAX_CANDIDATES = 10_000
MAX_HISTORY_ITEMS = 104
REPORT_MAX_BYTES = 4 * 1024 * 1024
_SOURCE_ID_RE = re.compile(r"\A[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?\Z")
_VERSION_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._+:/-]{0,127}\Z")
_DATE_RE = re.compile(r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_UTC_RE = re.compile(r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_UNKNOWN = "unknown"


class AlgorithmScoutError(ValueError):
    """One bounded, non-sensitive scout failure."""

    def __init__(self, code: str, detail: str) -> None:
        safe = detail.encode("utf-8", "replace")[:512].decode("utf-8", "ignore")
        super().__init__(f"{code}: {safe}")
        self.code = code


def _fail(code: str, detail: str) -> AlgorithmScoutError:
    return AlgorithmScoutError(code, detail)


def _bounded_text(value: object, *, field: str, maximum: int, unknown: bool = False) -> str:
    if value is None and unknown:
        return _UNKNOWN
    if not isinstance(value, str):
        raise _fail("source_config_invalid", f"{field} must be a string")
    if not value or len(value.encode("utf-8")) > maximum or any(ord(char) < 0x20 for char in value):
        raise _fail("source_config_invalid", f"{field} is empty or exceeds its bound")
    return value


def _bounded_string_list(
    value: object,
    *,
    field: str,
    maximum_items: int,
    maximum_item_bytes: int,
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum_items:
        raise _fail("source_config_invalid", f"{field} must be a bounded array")
    normalized = tuple(
        _bounded_text(item, field=field, maximum=maximum_item_bytes) for item in value
    )
    if len(set(normalized)) != len(normalized):
        raise _fail("source_config_invalid", f"{field} contains duplicates")
    return normalized


def _validate_calendar_date(value: str) -> str:
    if not isinstance(value, str) or _DATE_RE.fullmatch(value) is None:
        raise _fail("collection_date_invalid", "collection date must be exact YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise _fail("collection_date_invalid", "collection date is not Gregorian") from exc
    if parsed.isoformat() != value:
        raise _fail("collection_date_invalid", "collection date spelling is not canonical")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_utc(value: str) -> str:
    if _UTC_RE.fullmatch(value) is None:
        raise _fail("timestamp_invalid", "timestamp must be exact RFC3339 UTC seconds")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise _fail("timestamp_invalid", "timestamp is not Gregorian UTC") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise _fail("timestamp_invalid", "timestamp spelling is not canonical")
    return value


def _parse_utc(value: str) -> datetime:
    _validate_utc(value)
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


@dataclass(frozen=True)
class AlgorithmScoutSource:
    source_id: str
    enabled: bool
    location: HttpsLocation
    redirects: tuple[HttpsLocation, ...]
    project_identity: str
    model_identity: str
    tasks: tuple[str, ...]
    local_availability: str
    hosted_availability: str
    license_status: str
    license_expression: str
    weight_status: str
    runtime_hints: tuple[str, ...]

    @property
    def allowlist(self) -> tuple[HttpsLocation, ...]:
        return (self.location, *self.redirects)


@dataclass(frozen=True)
class ScoutPlan:
    sources_path: str
    output_dir: str
    collection_date: str
    trigger: str
    enabled_sources: tuple[AlgorithmScoutSource, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "yolozu_algorithm_scout_plan",
            "schema_version": 1,
            "maturity": "experimental",
            "mode": "plan",
            "collection_date": self.collection_date,
            "trigger": self.trigger,
            "sources_path": self.sources_path,
            "output_dir": self.output_dir,
            "enabled_source_count": len(self.enabled_sources),
            "sources": [
                {
                    "source_id": source.source_id,
                    "location": source.location.to_mapping(),
                    "redirect_allowlist": [item.to_mapping() for item in source.redirects],
                }
                for source in self.enabled_sources
            ],
            "network_used": False,
            "writes_performed": False,
            "selectability": "inbox_only",
            "limits": {
                "collection_timeout_seconds": COLLECTION_TIMEOUT_SECONDS,
                "workflow_finalization_reserve_seconds": WORKFLOW_RESERVE_SECONDS,
                "decoded_total_bytes": TOTAL_DECODED_BYTES,
            },
        }


@dataclass(frozen=True)
class DocumentParserLimits:
    html_nodes: int = 200_000
    html_tokens: int = 400_000
    html_depth: int = 128
    html_retained_text_bytes: int = 4 * 1024 * 1024
    pdf_pages: int = 64
    pdf_characters: int = 1_000_000
    pdf_wall_seconds: int = 10
    pdf_cpu_seconds: int = 8
    pdf_rss_bytes: int = 512 * 1024 * 1024
    pdf_pids: int = 4
    pdf_temp_bytes: int = 64 * 1024 * 1024
    pdf_ipc_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        maxima = DocumentParserLimits.__dataclass_fields__
        defaults = {
            name: field.default for name, field in maxima.items()
        }
        for name, maximum in defaults.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > maximum:
                raise ValueError(f"{name} must be a positive integer no larger than {maximum}")


def _source_from_record(value: object) -> AlgorithmScoutSource:
    required = {
        "source_id",
        "enabled",
        "location",
        "redirect_allowlist",
        "identity",
        "tasks",
        "availability",
        "license",
        "weights",
        "runtime_hints",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise _fail("source_config_invalid", "source record fields are invalid")
    source_id = _bounded_text(value["source_id"], field="source_id", maximum=64)
    if _SOURCE_ID_RE.fullmatch(source_id) is None:
        raise _fail("source_config_invalid", "source_id is not canonical")
    if type(value["enabled"]) is not bool:
        raise _fail("source_config_invalid", "enabled must be boolean")
    location = HttpsLocation.from_mapping(value["location"])
    raw_redirects = value["redirect_allowlist"]
    if not isinstance(raw_redirects, list) or len(raw_redirects) > 3:
        raise _fail("source_config_invalid", "redirect_allowlist exceeds three locations")
    redirects = tuple(HttpsLocation.from_mapping(item) for item in raw_redirects)
    if len(set(redirects)) != len(redirects) or location in redirects:
        raise _fail("source_config_invalid", "redirect_allowlist contains duplicates")
    identity = value["identity"]
    if not isinstance(identity, dict) or set(identity) != {"project", "model"}:
        raise _fail("source_config_invalid", "identity fields are invalid")
    availability = value["availability"]
    if not isinstance(availability, dict) or set(availability) != {"local", "hosted"}:
        raise _fail("source_config_invalid", "availability fields are invalid")
    license_record = value["license"]
    if not isinstance(license_record, dict) or set(license_record) != {"status", "expression"}:
        raise _fail("source_config_invalid", "license fields are invalid")
    weights = value["weights"]
    if not isinstance(weights, dict) or set(weights) != {"status"}:
        raise _fail("source_config_invalid", "weight fields are invalid")
    status_values = {"available", "unavailable", "unknown"}
    license_values = {"approved", "rejected", "review_required", "unknown"}
    local = _bounded_text(availability["local"], field="availability.local", maximum=32)
    hosted = _bounded_text(availability["hosted"], field="availability.hosted", maximum=32)
    license_status = _bounded_text(license_record["status"], field="license.status", maximum=32)
    weight_status = _bounded_text(weights["status"], field="weights.status", maximum=32)
    if local not in status_values or hosted not in status_values or weight_status not in status_values:
        raise _fail("source_config_invalid", "availability/weight status is invalid")
    if license_status not in license_values:
        raise _fail("source_config_invalid", "license status is invalid")
    return AlgorithmScoutSource(
        source_id=source_id,
        enabled=bool(value["enabled"]),
        location=location,
        redirects=redirects,
        project_identity=_bounded_text(identity["project"], field="identity.project", maximum=128),
        model_identity=_bounded_text(identity["model"], field="identity.model", maximum=128),
        tasks=_bounded_string_list(value["tasks"], field="tasks", maximum_items=32, maximum_item_bytes=64),
        local_availability=local,
        hosted_availability=hosted,
        license_status=license_status,
        license_expression=_bounded_text(
            license_record["expression"], field="license.expression", maximum=128
        ),
        weight_status=weight_status,
        runtime_hints=_bounded_string_list(
            value["runtime_hints"],
            field="runtime_hints",
            maximum_items=32,
            maximum_item_bytes=128,
        ),
    )


def load_algorithm_scout_sources(
    sources_path: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[AlgorithmScoutSource, ...]:
    root = Path(repository_root).resolve(strict=True)
    canonical = root.joinpath(*CANONICAL_SOURCES_PATH.parts)
    supplied = Path(sources_path)
    if supplied.is_absolute():
        candidate = supplied
    else:
        candidate = root / supplied
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise _fail("sources_invalid", "canonical source file is unavailable") from exc
    if resolved != canonical.resolve(strict=True):
        raise _fail("sources_invalid", "only the canonical source file is accepted")
    relative = canonical.relative_to(root)
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise _fail("sources_invalid", "canonical source path cannot contain symlinks")
    payload = load_bounded_json(canonical, label="algorithm scout sources")
    if not isinstance(payload, dict) or set(payload) != {"kind", "schema_version", "sources"}:
        raise _fail("source_config_invalid", "canonical source document fields are invalid")
    if payload["kind"] != "yolozu_algorithm_scout_sources" or payload["schema_version"] != 1:
        raise _fail("source_config_invalid", "canonical source document identity is invalid")
    raw_sources = payload["sources"]
    if not isinstance(raw_sources, list) or not 1 <= len(raw_sources) <= MAX_SOURCES:
        raise _fail("source_config_invalid", "canonical source count is invalid")
    sources = tuple(_source_from_record(item) for item in raw_sources)
    if len({item.source_id for item in sources}) != len(sources):
        raise _fail("source_config_invalid", "source IDs must be unique")
    locations = [location for source in sources for location in source.allowlist]
    if len(set(locations)) != len(locations):
        raise _fail("source_config_invalid", "allowlisted locations must be globally unique")
    if not any(source.enabled for source in sources):
        raise _fail("source_config_invalid", "at least one source must be enabled")
    return sources


def _relative_output_dir(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail("output_dir_invalid", "output directory is required")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value.endswith("/")
        or "\\" in value
        or len(value.encode("utf-8")) > 4096
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(part.startswith(".") for part in path.parts)
    ):
        raise _fail("output_dir_invalid", "output directory must be a visible workspace-relative path")
    return "/".join(path.parts)


def _workspace(value: str | Path) -> Path:
    candidate = Path(value)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise _fail("workspace_invalid", "workspace is unavailable") from exc
    if candidate.is_symlink() or not resolved.is_dir():
        raise _fail("workspace_invalid", "workspace must be a non-symlink directory")
    return resolved


def build_scout_plan(
    *,
    sources_path: str | Path,
    output_dir: str,
    collection_date: str,
    trigger: str,
    workspace_root: str | Path = ".",
    repository_root: str | Path | None = None,
) -> ScoutPlan:
    _workspace(workspace_root)
    repo = Path(repository_root).resolve(strict=True) if repository_root is not None else Path(__file__).resolve().parents[2]
    sources = load_algorithm_scout_sources(sources_path, repository_root=repo)
    if trigger not in {"schedule", "workflow_dispatch"}:
        raise _fail("trigger_invalid", "trigger must be schedule or workflow_dispatch")
    return ScoutPlan(
        sources_path=CANONICAL_SOURCES_PATH.as_posix(),
        output_dir=_relative_output_dir(output_dir),
        collection_date=_validate_calendar_date(collection_date),
        trigger=trigger,
        enabled_sources=tuple(source for source in sources if source.enabled),
    )


class _BoundedHtmlParser(HTMLParser):
    _VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"})

    def __init__(self, limits: DocumentParserLimits) -> None:
        super().__init__(convert_charrefs=True)
        self.limits = limits
        self.nodes = 0
        self.tokens = 0
        self.depth = 0
        self.max_depth_observed = 0
        self.text_bytes = 0
        self.in_title = False
        self.title_parts: list[str] = []

    def _token(self) -> None:
        self.tokens += 1
        if self.tokens > self.limits.html_tokens:
            raise _fail("html_token_limit", "HTML token count exceeds its cap")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        self._token()
        self.nodes += 1
        if self.nodes > self.limits.html_nodes:
            raise _fail("html_node_limit", "HTML node count exceeds its cap")
        lowered = tag.lower()
        if lowered in {"script", "object", "embed", "iframe"}:
            raise _fail("html_active_content", "active or embedded HTML content is forbidden")
        if lowered not in self._VOID:
            self.depth += 1
            self.max_depth_observed = max(self.max_depth_observed, self.depth)
            if self.depth > self.limits.html_depth:
                raise _fail("html_depth_limit", "HTML nesting depth exceeds its cap")
        if lowered == "title":
            self.in_title = True

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in self._VOID:
            self.depth = max(0, self.depth - 1)

    def handle_endtag(self, tag: str) -> None:
        self._token()
        if tag.lower() == "title":
            self.in_title = False
        if tag.lower() not in self._VOID:
            self.depth = max(0, self.depth - 1)

    def handle_data(self, data: str) -> None:
        self._token()
        if data:
            self.nodes += 1
            if self.nodes > self.limits.html_nodes:
                raise _fail("html_node_limit", "HTML node count exceeds its cap")
        self.text_bytes += len(data.encode("utf-8"))
        if self.text_bytes > self.limits.html_retained_text_bytes:
            raise _fail("html_text_limit", "HTML retained text exceeds its cap")
        if self.in_title and sum(len(part.encode("utf-8")) for part in self.title_parts) < 512:
            self.title_parts.append(data)

    def handle_decl(self, decl: str) -> None:
        self._token()
        lowered = decl.lower()
        if (
            not lowered.startswith("doctype html")
            or "[" in lowered
            or " system " in lowered
            or " public " in lowered
        ):
            raise _fail("html_declaration_forbidden", "HTML DTD/entity declarations are forbidden")

    def unknown_decl(self, data: str) -> None:
        del data
        raise _fail("html_declaration_forbidden", "HTML declarations are forbidden")

    def handle_comment(self, data: str) -> None:
        del data
        self._token()

    def summary(self) -> dict[str, Any]:
        title = " ".join("".join(self.title_parts).split())
        return {
            "document_title": title[:512] if title else _UNKNOWN,
            "html_nodes": self.nodes,
            "html_tokens": self.tokens,
            "html_max_depth_observed": self.max_depth_observed,
            "html_text_bytes": self.text_bytes,
        }


def _parse_html(body: bytes, limits: DocumentParserLimits) -> dict[str, Any]:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _fail("document_utf8_invalid", "HTML must be UTF-8") from exc
    lowered = text.lower()
    if any(marker in lowered for marker in ("<!entity", "<!element", "<!attlist", "<!notation")):
        raise _fail("html_declaration_forbidden", "HTML DTD/entity declarations are forbidden")
    parser = _BoundedHtmlParser(limits)
    for offset in range(0, len(text), 64 * 1024):
        parser.feed(text[offset : offset + 64 * 1024])
    parser.close()
    return parser.summary()


def _disable_child_network() -> None:
    def denied(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise PermissionError("network disabled in PDF parser")

    socket.socket = denied  # type: ignore[assignment]
    socket.create_connection = denied  # type: ignore[assignment]
    socket.getaddrinfo = denied  # type: ignore[assignment]


def _disable_child_process_creation() -> None:
    def denied(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise PermissionError("process creation disabled in PDF parser")

    for name in (
        "fork",
        "forkpty",
        "posix_spawn",
        "posix_spawnp",
        "system",
    ):
        if hasattr(os, name):
            setattr(os, name, denied)


def _apply_pdf_resource_limits(limits: DocumentParserLimits) -> None:
    for kind, soft, hard in (
        (resource.RLIMIT_CPU, limits.pdf_cpu_seconds, limits.pdf_cpu_seconds),
        (resource.RLIMIT_AS, limits.pdf_rss_bytes, limits.pdf_rss_bytes),
        (resource.RLIMIT_FSIZE, limits.pdf_temp_bytes, limits.pdf_temp_bytes),
        (resource.RLIMIT_NPROC, limits.pdf_pids, limits.pdf_pids),
    ):
        try:
            resource.setrlimit(kind, (soft, hard))
        except (OSError, ValueError):
            # The parent still enforces wall/RSS and the worker never executes
            # untrusted code or exposes a process-creation path.
            continue


def _process_rss_bytes(pid: int) -> int | None:
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{pid}/statm").read_text(encoding="ascii").split()
            return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            return None
    if sys.platform == "darwin":
        try:
            import ctypes

            library = ctypes.CDLL("/usr/lib/libproc.dylib")
            buffer = ctypes.create_string_buffer(256)
            size = library.proc_pidinfo(pid, 4, 0, buffer, len(buffer))
            if size < 16:
                return None
            _virtual, resident = struct.unpack_from("QQ", buffer.raw)
            return int(resident)
        except (OSError, ValueError):
            return None
    return None


def _pdf_worker(
    connection: Any,
    body: bytes,
    temp_dir: str,
    limits: DocumentParserLimits,
) -> None:
    try:
        os.setsid()
        os.environ.clear()
        os.chdir(temp_dir)
        null_input = os.open(os.devnull, os.O_RDONLY)
        null_output = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null_input, 0)
            os.dup2(null_output, 1)
            os.dup2(null_output, 2)
        finally:
            os.close(null_input)
            os.close(null_output)
        _disable_child_network()
        _disable_child_process_creation()
        _apply_pdf_resource_limits(limits)
        if not body.startswith(b"%PDF-"):
            raise _fail("pdf_magic_mismatch", "PDF magic does not match content type")
        if b"%%EOF" not in body[-2048:]:
            raise _fail("pdf_malformed", "PDF end marker is missing")
        if b"/EmbeddedFile" in body or b"/EmbeddedFiles" in body:
            raise _fail("pdf_embedded_file", "PDF embedded files are forbidden")
        pages = len(re.findall(rb"/Type\s*/Page\b", body))
        if pages > limits.pdf_pages:
            raise _fail("pdf_page_limit", "PDF page count exceeds its cap")
        characters = 0
        for match in re.finditer(rb"\(([^()]{0,1048576})\)\s*T[Jj]", body):
            characters += len(match.group(1).decode("latin-1"))
            if characters > limits.pdf_characters:
                raise _fail("pdf_character_limit", "PDF extracted character count exceeds its cap")
        payload = {
            "ok": True,
            "summary": {
                "pdf_pages": pages,
                "pdf_extracted_characters": characters,
                "embedded_files": False,
            },
        }
    except Exception as exc:
        code = exc.code if isinstance(exc, AlgorithmScoutError) else "pdf_parser_failed"
        payload = {"ok": False, "code": code}
    try:
        encoded = canonical_json_v1(payload)
        if len(encoded) <= limits.pdf_ipc_bytes:
            connection.send_bytes(encoded)
    finally:
        connection.close()


def _terminate_process_group(process: multiprocessing.Process) -> None:
    if not process.is_alive():
        process.join(timeout=0.2)
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    process.join(timeout=1)
    if process.is_alive():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.join(timeout=1)


def _parse_pdf(
    body: bytes,
    limits: DocumentParserLimits,
    *,
    _worker: Callable[..., None] = _pdf_worker,
    _temp_parent: str | None = None,
) -> dict[str, Any]:
    if os.name != "posix":
        raise _fail("pdf_platform_unsupported", "PDF parser isolation requires POSIX")
    temp_dir = tempfile.mkdtemp(prefix="yolozu-scout-pdf-", dir=_temp_parent)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(child, body, temp_dir, limits), daemon=False)
    wait_started = time.monotonic()
    process.start()
    child.close()
    try:
        while not parent.poll(0.05):
            rss = _process_rss_bytes(process.pid)
            if rss is not None and rss > limits.pdf_rss_bytes:
                _terminate_process_group(process)
                raise _fail("pdf_rss_limit", "PDF parser exceeded its RSS cap")
            if time.monotonic() - wait_started >= limits.pdf_wall_seconds:
                _terminate_process_group(process)
                raise _fail("pdf_wall_limit", "PDF parser exceeded its wall-clock cap")
        try:
            encoded = parent.recv_bytes(limits.pdf_ipc_bytes)
        except OSError as exc:
            raise _fail("pdf_ipc_limit", "PDF parser exceeded its IPC cap") from exc
        except EOFError as exc:
            process.join(timeout=0.2)
            if process.exitcode in {
                -getattr(signal, "SIGXCPU", signal.SIGKILL),
                -signal.SIGKILL,
            }:
                raise _fail("pdf_cpu_limit", "PDF parser exceeded its CPU cap") from exc
            raise _fail("pdf_parser_failed", "PDF parser ended without a bounded result") from exc
        process.join(timeout=1)
        if process.is_alive():
            _terminate_process_group(process)
            raise _fail("pdf_reap_failed", "PDF parser did not exit after returning")
        payload = load_bounded_json_bytes(encoded, label="PDF parser result")
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            code = payload.get("code") if isinstance(payload, dict) else "pdf_parser_failed"
            raise _fail(str(code), "PDF parser rejected the document")
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            raise _fail("pdf_parser_failed", "PDF parser result is invalid")
        return summary
    finally:
        parent.close()
        _terminate_process_group(process)
        shutil.rmtree(temp_dir, ignore_errors=False)


def _validate_document_magic(document: FetchedDocument) -> None:
    body = document.body
    stripped = body.lstrip()
    lowered = stripped[:64].lower()
    if body.startswith((b"PK\x03\x04", b"\x1f\x8b", b"Rar!", b"7z\xbc\xaf\x27\x1c")):
        raise _fail("archive_forbidden", "archive content is forbidden")
    if body.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a")):
        raise _fail("media_forbidden", "media content is forbidden")
    content_type = document.content_type
    if content_type == "application/pdf":
        if not body.startswith(b"%PDF-"):
            raise _fail("content_magic_mismatch", "PDF magic does not match content type")
    elif content_type == "application/json":
        if not stripped.startswith((b"{", b"[")):
            raise _fail("content_magic_mismatch", "JSON magic does not match content type")
    elif content_type in {"text/html", "application/xhtml+xml"}:
        if not lowered.startswith((b"<!doctype html", b"<html", b"<head", b"<body")):
            raise _fail("content_magic_mismatch", "HTML magic does not match content type")
    elif content_type == "text/plain":
        if b"\x00" in body:
            raise _fail("content_magic_mismatch", "plain text contains binary data")
    else:
        raise _fail("content_type_invalid", "document type is not allowed")


def _candidate_text(value: object, *, maximum: int, unknown: bool = True) -> str:
    if value is None and unknown:
        return _UNKNOWN
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        return _UNKNOWN if unknown else ""
    if any(ord(character) < 0x20 for character in value):
        return _UNKNOWN if unknown else ""
    return value


def _candidate_from_source(
    source: AlgorithmScoutSource,
    *,
    collected_at: str,
    collection_date: str,
    content_sha256: str,
    untrusted: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    item = untrusted or {}
    version = _candidate_text(
        item.get("version", item.get("tag_name")), maximum=128
    )
    revision = _candidate_text(item.get("revision"), maximum=128)
    if version != _UNKNOWN and _VERSION_RE.fullmatch(version) is None:
        version = _UNKNOWN
    if revision != _UNKNOWN and _VERSION_RE.fullmatch(revision) is None:
        revision = _UNKNOWN
    raw_release_date = item.get("release_date", item.get("published_at"))
    if isinstance(raw_release_date, str) and len(raw_release_date) >= 10:
        raw_release_date = raw_release_date[:10]
    release_date = _candidate_text(raw_release_date, maximum=10)
    if release_date != _UNKNOWN:
        try:
            release_date = _validate_calendar_date(release_date)
        except AlgorithmScoutError:
            release_date = _UNKNOWN
    project = _candidate_text(item.get("project"), maximum=128)
    model = _candidate_text(item.get("model"), maximum=128)
    if project == _UNKNOWN:
        project = source.project_identity
    if model == _UNKNOWN:
        model = source.model_identity
    identity = version if version != _UNKNOWN else revision
    if identity == _UNKNOWN:
        identity = _UNKNOWN
    key_record = {
        "source_url": source.location.to_url(),
        "version_or_revision": identity,
    }
    candidate_key = hashlib.sha256(canonical_json_v1(key_record)).hexdigest()
    return {
        "candidate_key": candidate_key,
        "source_id": source.source_id,
        "source_url": source.location.to_url(),
        "identity": {"project": project, "model": model},
        "version": version,
        "revision": revision,
        "release_date": release_date,
        "tasks": list(source.tasks),
        "availability": {
            "local": source.local_availability,
            "hosted": source.hosted_availability,
        },
        "license": {
            "status": source.license_status,
            "expression": source.license_expression,
        },
        "weights": {"status": source.weight_status},
        "runtime_hints": list(source.runtime_hints),
        "collection_status": "collected",
        "collected_at": collected_at,
        "timezone": "UTC",
        "first_seen_date": collection_date,
        "last_seen_date": collection_date,
        "history": [
            {
                "collection_date": collection_date,
                "collected_at": collected_at,
                "content_sha256": content_sha256,
            }
        ],
    }


def _parse_document(
    source: AlgorithmScoutSource,
    document: FetchedDocument,
    *,
    collected_at: str,
    collection_date: str,
    limits: DocumentParserLimits,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_document_magic(document)
    digest = hashlib.sha256(document.body).hexdigest()
    if document.content_type in {"text/html", "application/xhtml+xml"}:
        summary = _parse_html(document.body, limits)
        candidates = [
            _candidate_from_source(
                source,
                collected_at=collected_at,
                collection_date=collection_date,
                content_sha256=digest,
            )
        ]
    elif document.content_type == "application/json":
        payload = load_bounded_json_bytes(document.body, label="untrusted scout JSON")
        raw_candidates: Iterable[Mapping[str, Any]]
        if isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
            raw_candidates = [item for item in payload["candidates"] if isinstance(item, dict)][:MAX_CANDIDATES]
        elif isinstance(payload, dict):
            raw_candidates = [payload]
        elif isinstance(payload, list):
            raw_candidates = [item for item in payload if isinstance(item, dict)][:MAX_CANDIDATES]
        else:
            raw_candidates = [{}]
        candidates = [
            _candidate_from_source(
                source,
                collected_at=collected_at,
                collection_date=collection_date,
                content_sha256=digest,
                untrusted=item,
            )
            for item in raw_candidates
        ]
        summary = {"json_nodes_bound": 100_000, "json_depth_bound": 64}
    elif document.content_type == "application/pdf":
        summary = _parse_pdf(document.body, limits)
        candidates = [
            _candidate_from_source(
                source,
                collected_at=collected_at,
                collection_date=collection_date,
                content_sha256=digest,
            )
        ]
    else:
        try:
            document.body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _fail("document_utf8_invalid", "plain text must be UTF-8") from exc
        summary = {"text_bytes": len(document.body)}
        candidates = [
            _candidate_from_source(
                source,
                collected_at=collected_at,
                collection_date=collection_date,
                content_sha256=digest,
            )
        ]
    return candidates, {
        "content_sha256": digest,
        "content_type": document.content_type,
        "transferred_bytes": document.transferred_bytes,
        "decoded_bytes": document.decoded_bytes,
        "redirect_count": document.redirect_count,
        "parser_summary": summary,
    }


def _read_prior_candidates(workspace: Path, output_dir: str) -> list[dict[str, Any]]:
    root = workspace.joinpath(*PurePosixPath(output_dir).parts)
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise _fail("output_dir_invalid", "existing output directory is unsafe")
    dated = sorted(
        (
            item
            for item in root.iterdir()
            if item.is_dir() and not item.is_symlink() and _DATE_RE.fullmatch(item.name)
        ),
        key=lambda item: item.name,
    )[-MAX_PRIOR_REPORTS:]
    candidates: list[dict[str, Any]] = []
    for directory in dated:
        try:
            validate_managed_output_destination(
                root=workspace,
                destination=f"{output_dir}/{directory.name}",
                limits=ManagedOutputLimits(
                    max_files=2,
                    max_file_bytes=REPORT_MAX_BYTES,
                    max_total_bytes=REPORT_MAX_BYTES + 4096,
                ),
                force=True,
            )
        except ValueError as exc:
            raise _fail(
                "prior_report_invalid",
                "prior scout managed output failed integrity validation",
            ) from exc
        report_path = directory / "algorithm_scout_report.json"
        if report_path.is_symlink() or not report_path.is_file():
            continue
        payload = load_bounded_json(report_path, label="prior algorithm scout report")
        if (
            not isinstance(payload, dict)
            or payload.get("kind") != REPORT_KIND
            or payload.get("schema_version") != REPORT_SCHEMA_VERSION
            or not isinstance(payload.get("candidates"), list)
        ):
            raise _fail("prior_report_invalid", "prior scout report is invalid")
        for item in payload["candidates"]:
            candidates.append(_validate_prior_candidate(item))
            if len(candidates) > MAX_CANDIDATES:
                raise _fail("history_limit", "prior candidate history exceeds its cap")
    return candidates


def _prior_text(value: object, *, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > maximum
        or any(ord(character) < 0x20 for character in value)
    ):
        raise _fail("prior_report_invalid", f"prior {field} is invalid")
    return value


def _validate_prior_candidate(value: object) -> dict[str, Any]:
    fields = {
        "candidate_key",
        "source_id",
        "source_url",
        "identity",
        "version",
        "revision",
        "release_date",
        "tasks",
        "availability",
        "license",
        "weights",
        "runtime_hints",
        "collection_status",
        "collected_at",
        "timezone",
        "first_seen_date",
        "last_seen_date",
        "history",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise _fail("prior_report_invalid", "prior candidate fields are invalid")
    candidate_key = _prior_text(value["candidate_key"], field="candidate key", maximum=64)
    if len(candidate_key) != 64 or any(character not in "0123456789abcdef" for character in candidate_key):
        raise _fail("prior_report_invalid", "prior candidate key is invalid")
    source_id = _prior_text(value["source_id"], field="source ID", maximum=64)
    if _SOURCE_ID_RE.fullmatch(source_id) is None:
        raise _fail("prior_report_invalid", "prior source ID is invalid")
    source_url = _prior_text(value["source_url"], field="source URL", maximum=4096)
    try:
        source_location = HttpsLocation.from_redirect(
            source_url,
            base=HttpsLocation(host="invalid.example", path="/"),
        )
    except SafeHttpsError as exc:
        raise _fail("prior_report_invalid", "prior source URL is invalid") from exc
    if source_location.to_url() != source_url:
        raise _fail("prior_report_invalid", "prior source URL is not canonical")
    identity = value["identity"]
    if not isinstance(identity, dict) or set(identity) != {"project", "model"}:
        raise _fail("prior_report_invalid", "prior identity is invalid")
    normalized_identity = {
        "project": _prior_text(identity["project"], field="project identity", maximum=128),
        "model": _prior_text(identity["model"], field="model identity", maximum=128),
    }
    version = _prior_text(value["version"], field="version", maximum=128)
    revision = _prior_text(value["revision"], field="revision", maximum=128)
    for label, item in (("version", version), ("revision", revision)):
        if item != _UNKNOWN and _VERSION_RE.fullmatch(item) is None:
            raise _fail("prior_report_invalid", f"prior {label} is invalid")
    release_date = _prior_text(value["release_date"], field="release date", maximum=10)
    if release_date != _UNKNOWN:
        _validate_calendar_date(release_date)
    try:
        tasks = _bounded_string_list(
            value["tasks"], field="prior tasks", maximum_items=32, maximum_item_bytes=64
        )
        runtime_hints = _bounded_string_list(
            value["runtime_hints"],
            field="prior runtime hints",
            maximum_items=32,
            maximum_item_bytes=128,
        )
    except AlgorithmScoutError as exc:
        raise _fail("prior_report_invalid", "prior candidate arrays are invalid") from exc
    availability = value["availability"]
    if not isinstance(availability, dict) or set(availability) != {"local", "hosted"}:
        raise _fail("prior_report_invalid", "prior availability is invalid")
    allowed_status = {"available", "unavailable", "unknown"}
    normalized_availability = {
        name: _prior_text(availability[name], field=f"availability {name}", maximum=32)
        for name in ("local", "hosted")
    }
    if any(item not in allowed_status for item in normalized_availability.values()):
        raise _fail("prior_report_invalid", "prior availability status is invalid")
    license_record = value["license"]
    if not isinstance(license_record, dict) or set(license_record) != {"status", "expression"}:
        raise _fail("prior_report_invalid", "prior license is invalid")
    normalized_license = {
        "status": _prior_text(license_record["status"], field="license status", maximum=32),
        "expression": _prior_text(
            license_record["expression"], field="license expression", maximum=128
        ),
    }
    if normalized_license["status"] not in {"approved", "rejected", "review_required", "unknown"}:
        raise _fail("prior_report_invalid", "prior license status is invalid")
    weights = value["weights"]
    if not isinstance(weights, dict) or set(weights) != {"status"}:
        raise _fail("prior_report_invalid", "prior weight status is invalid")
    weight_status = _prior_text(weights["status"], field="weight status", maximum=32)
    if weight_status not in allowed_status:
        raise _fail("prior_report_invalid", "prior weight status is invalid")
    collection_status = _prior_text(
        value["collection_status"], field="collection status", maximum=16
    )
    if collection_status not in {"collected", "historical"}:
        raise _fail("prior_report_invalid", "prior collection status is invalid")
    collected_at = _validate_utc(
        _prior_text(value["collected_at"], field="collected timestamp", maximum=20)
    )
    if value["timezone"] != "UTC":
        raise _fail("prior_report_invalid", "prior timezone is invalid")
    first_seen = _validate_calendar_date(
        _prior_text(value["first_seen_date"], field="first seen date", maximum=10)
    )
    last_seen = _validate_calendar_date(
        _prior_text(value["last_seen_date"], field="last seen date", maximum=10)
    )
    if date.fromisoformat(last_seen) < date.fromisoformat(first_seen):
        raise _fail("prior_report_invalid", "prior candidate dates are reversed")
    raw_history = value["history"]
    if not isinstance(raw_history, list) or len(raw_history) > MAX_HISTORY_ITEMS:
        raise _fail("prior_report_invalid", "prior candidate history is invalid")
    history: list[dict[str, str]] = []
    for item in raw_history:
        if not isinstance(item, dict) or set(item) != {"collection_date", "collected_at", "content_sha256"}:
            raise _fail("prior_report_invalid", "prior history item is invalid")
        digest = _prior_text(item["content_sha256"], field="content digest", maximum=64)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise _fail("prior_report_invalid", "prior content digest is invalid")
        history.append(
            {
                "collection_date": _validate_calendar_date(
                    _prior_text(item["collection_date"], field="history date", maximum=10)
                ),
                "collected_at": _validate_utc(
                    _prior_text(item["collected_at"], field="history timestamp", maximum=20)
                ),
                "content_sha256": digest,
            }
        )
    version_or_revision = version if version != _UNKNOWN else revision
    expected_key = hashlib.sha256(
        canonical_json_v1(
            {"source_url": source_url, "version_or_revision": version_or_revision}
        )
    ).hexdigest()
    if candidate_key != expected_key:
        raise _fail("prior_report_invalid", "prior candidate key does not match identity")
    return {
        "candidate_key": candidate_key,
        "source_id": source_id,
        "source_url": source_url,
        "identity": normalized_identity,
        "version": version,
        "revision": revision,
        "release_date": release_date,
        "tasks": list(tasks),
        "availability": normalized_availability,
        "license": normalized_license,
        "weights": {"status": weight_status},
        "runtime_hints": list(runtime_hints),
        "collection_status": collection_status,
        "collected_at": collected_at,
        "timezone": "UTC",
        "first_seen_date": first_seen,
        "last_seen_date": last_seen,
        "history": history,
    }


def _merge_candidates(
    prior: Iterable[Mapping[str, Any]],
    current: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    prior_items = list(prior)
    current_items = list(current)
    current_keys = {
        str(item.get("candidate_key"))
        for item in current_items
        if isinstance(item.get("candidate_key"), str)
    }
    merged: dict[str, dict[str, Any]] = {}
    for raw in [*prior_items, *current_items]:
        key = raw.get("candidate_key")
        if not isinstance(key, str) or len(key) != 64:
            continue
        candidate = json.loads(json.dumps(raw, ensure_ascii=False))
        existing = merged.get(key)
        if existing is None:
            history = candidate.get("history")
            candidate["history"] = history[-MAX_HISTORY_ITEMS:] if isinstance(history, list) else []
            merged[key] = candidate
            continue
        histories = [
            item
            for item in [*(existing.get("history") or []), *(candidate.get("history") or [])]
            if isinstance(item, dict)
        ]
        unique = {
            (
                str(item.get("collection_date")),
                str(item.get("collected_at")),
                str(item.get("content_sha256")),
            ): item
            for item in histories
        }
        candidate["history"] = [
            unique[key]
            for key in sorted(unique, key=lambda value: tuple(part.encode("utf-8") for part in value))
        ][-MAX_HISTORY_ITEMS:]
        first_dates = [
            value
            for value in (existing.get("first_seen_date"), candidate.get("first_seen_date"))
            if isinstance(value, str) and _DATE_RE.fullmatch(value)
        ]
        last_dates = [
            value
            for value in (existing.get("last_seen_date"), candidate.get("last_seen_date"))
            if isinstance(value, str) and _DATE_RE.fullmatch(value)
        ]
        candidate["first_seen_date"] = min(first_dates) if first_dates else candidate.get("first_seen_date")
        candidate["last_seen_date"] = max(last_dates) if last_dates else candidate.get("last_seen_date")
        merged[key] = candidate
    if len(merged) > MAX_CANDIDATES:
        raise _fail("candidate_limit", "candidate inbox exceeds its cap")
    for key, candidate in merged.items():
        candidate["collection_status"] = (
            "collected" if key in current_keys else "historical"
        )
    return [merged[key] for key in sorted(merged, key=lambda item: item.encode("ascii"))]


def _ensure_directory_chain(workspace: Path, output_dir: str) -> None:
    _open_output_chain(workspace, output_dir, create=True)


def _validate_existing_output_chain(workspace: Path, output_dir: str) -> None:
    _open_output_chain(workspace, output_dir, create=False)


def _open_output_chain(workspace: Path, output_dir: str, *, create: bool) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if os.name != "posix" or not nofollow or not directory:
        raise _fail("platform_unsupported", "workspace output requires POSIX no-follow directories")
    descriptor = os.open(workspace, os.O_RDONLY | directory | nofollow | cloexec)
    try:
        for component in PurePosixPath(output_dir).parts:
            try:
                child = os.open(
                    component,
                    os.O_RDONLY | directory | nofollow | cloexec,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    return
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(
                    component,
                    os.O_RDONLY | directory | nofollow | cloexec,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise _fail(
                        "output_dir_invalid",
                        "output path contains an unsafe component",
                    ) from exc
                raise
            info = os.fstat(child)
            if not stat.S_ISDIR(info.st_mode):
                os.close(child)
                raise _fail("output_dir_invalid", "output path component is not a directory")
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)


def collect_algorithm_candidates(
    plan: ScoutPlan,
    *,
    workspace_root: str | Path = ".",
    transport_factory: Callable[..., SafeHttpsTransport] = SafeHttpsTransport,
    parser_limits: DocumentParserLimits | None = None,
    now_utc: Callable[[], str] = _utc_now,
    monotonic: Callable[[], float] = time.monotonic,
) -> tuple[dict[str, Any], int, Path]:
    workspace = _workspace(workspace_root)
    limits = parser_limits or DocumentParserLimits()
    started_at = _validate_utc(now_utc())
    deadline = monotonic() + COLLECTION_TIMEOUT_SECONDS
    _validate_existing_output_chain(workspace, plan.output_dir)
    prior = _read_prior_candidates(workspace, plan.output_dir)
    locations = tuple(
        location for source in plan.enabled_sources for location in source.allowlist
    )
    transport = transport_factory(allowlist=locations, limits=TransportLimits())
    source_results: list[dict[str, Any]] = []
    current_candidates: list[dict[str, Any]] = []
    decoded_total = 0
    failures = 0
    for source in plan.enabled_sources:
        collected_at = _validate_utc(now_utc())
        base = {
            "source_id": source.source_id,
            "source_url": source.location.to_url(),
            "collected_at": collected_at,
            "timezone": "UTC",
        }
        if monotonic() >= deadline:
            failures += 1
            source_results.append({**base, "collection_status": "missed", "failure_code": "collection_deadline"})
            continue
        try:
            document = transport.fetch(source.location, collection_deadline=deadline)
            decoded_total += document.decoded_bytes
            if decoded_total > TOTAL_DECODED_BYTES:
                raise _fail("decoded_total_limit", "collection decoded-byte total exceeds its cap")
            candidates, provenance = _parse_document(
                source,
                document,
                collected_at=collected_at,
                collection_date=plan.collection_date,
                limits=limits,
            )
            current_candidates.extend(candidates)
            if len(current_candidates) > MAX_CANDIDATES:
                raise _fail("candidate_limit", "candidate inbox exceeds its cap")
            source_results.append(
                {
                    **base,
                    "collection_status": "collected",
                    "candidate_count": len(candidates),
                    "provenance": provenance,
                }
            )
        except (AlgorithmScoutError, SafeHttpsError, OSError, ValueError) as exc:
            failures += 1
            code = getattr(exc, "code", "collection_failed")
            source_results.append({**base, "collection_status": "failed", "failure_code": str(code)[:64]})
    candidates = _merge_candidates(prior, current_candidates)
    completed_at = _validate_utc(now_utc())
    if _parse_utc(completed_at) < _parse_utc(started_at):
        raise _fail("timestamp_reversed", "report completion precedes collection start")
    report = {
        "kind": REPORT_KIND,
        "schema_version": REPORT_SCHEMA_VERSION,
        "maturity": "experimental",
        "selectability": "inbox_only",
        "collection_date": plan.collection_date,
        "trigger": plan.trigger,
        "started_at": started_at,
        "completed_at": completed_at,
        "timezone": "UTC",
        "source_policy": {
            "canonical_sources": CANONICAL_SOURCES_PATH.as_posix(),
            "scope": "explicit_official_allowlist_only",
            "coverage": "monitored_sources_not_latest_world",
            "redirect_limit": 3,
            "network": "https_443_collect_only",
        },
        "retention": {
            "raw_documents": "not_retained",
            "candidate_metadata": "repository_policy",
            "history_reports_scanned_max": MAX_PRIOR_REPORTS,
        },
        "automation_caveat": "Discovery metadata is an untrusted inbox signal, not qualification, support, recommendation, adoption, or promotion evidence.",
        "sources": source_results,
        "candidates": candidates,
        "summary": {
            "enabled_sources": len(plan.enabled_sources),
            "collected_sources": sum(item["collection_status"] == "collected" for item in source_results),
            "failed_or_missed_sources": failures,
            "candidate_count": len(candidates),
            "current_candidate_count": sum(
                item["collection_status"] == "collected" for item in candidates
            ),
            "historical_candidate_count": sum(
                item["collection_status"] == "historical" for item in candidates
            ),
            "decoded_total_bytes": decoded_total,
            "network_used": True,
            "writes_performed": True,
        },
    }
    encoded = canonical_json_v1(report)
    if len(encoded) > REPORT_MAX_BYTES:
        raise _fail("report_limit", "final report exceeds its byte cap")
    _ensure_directory_chain(workspace, plan.output_dir)
    destination = f"{plan.output_dir}/{plan.collection_date}"
    with ManagedOutputTransaction(
        root=workspace,
        destination=destination,
        declared_paths=("algorithm_scout_report.json",),
        limits=ManagedOutputLimits(
            max_files=2,
            max_file_bytes=REPORT_MAX_BYTES,
            max_total_bytes=REPORT_MAX_BYTES + 4096,
        ),
        force=True,
    ) as transaction:
        transaction.write_bytes("algorithm_scout_report.json", encoded)
        transaction.commit()
    return report, 3 if failures else 0, workspace / destination / "algorithm_scout_report.json"
