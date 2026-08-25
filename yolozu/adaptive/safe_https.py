"""Bounded HTTPS-only transport for code-owned adaptive intake services.

The transport accepts structured, pre-allowlisted locations.  It does not
accept caller headers, cookies, credentials, query strings, or fragments.
Fetched bytes remain an untrusted in-memory document for a caller-owned
bounded parser; this module never executes or persists them.
"""

from __future__ import annotations

import ipaddress
import socket
import ssl
import time
import zlib
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Protocol
from urllib.parse import urljoin, urlsplit

__all__ = [
    "FetchedDocument",
    "HttpsLocation",
    "SafeHttpsError",
    "SafeHttpsTransport",
    "TransportLimits",
]


_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_ALLOWED_CONTENT_TYPES = {
    "application/json",
    "application/pdf",
    "application/xhtml+xml",
    "text/html",
    "text/plain",
}
_DENIED_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in (
        "192.0.2.0/24",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "2001:db8::/32",
    )
)


class SafeHttpsError(ValueError):
    """A bounded failure that never includes a rejected raw URL."""

    def __init__(self, code: str, detail: str) -> None:
        safe_detail = detail.encode("utf-8", "replace")[:512].decode(
            "utf-8", "ignore"
        )
        super().__init__(f"{code}: {safe_detail}")
        self.code = code


def _fail(code: str, detail: str) -> SafeHttpsError:
    return SafeHttpsError(code, detail)


@dataclass(frozen=True, order=True)
class HttpsLocation:
    """One credential-free canonical HTTPS location."""

    host: str
    path: str
    scheme: str = "https"

    def __post_init__(self) -> None:
        if self.scheme != "https":
            raise _fail("url_scheme_invalid", "only https is allowed")
        host = self.host.rstrip(".").lower()
        if not host or len(host.encode("ascii", "ignore")) != len(host):
            raise _fail("url_host_invalid", "host must be bounded ASCII DNS text")
        if len(host) > 253 or any(
            not label
            or len(label) > 63
            or label[0] == "-"
            or label[-1] == "-"
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in label)
            for label in host.split(".")
        ):
            raise _fail("url_host_invalid", "host is not a canonical DNS name")
        try:
            ipaddress.ip_address(host.strip("[]"))
        except ValueError:
            pass
        else:
            raise _fail("url_ip_literal", "IP literals are forbidden")
        if (
            not self.path.startswith("/")
            or "\\" in self.path
            or "?" in self.path
            or "#" in self.path
            or "\x00" in self.path
            or not self.path.isascii()
            or any(ord(character) <= 0x20 or ord(character) == 0x7F for character in self.path)
            or len(self.path.encode("utf-8")) > 4096
            or any(segment in {".", ".."} for segment in self.path.split("/"))
        ):
            raise _fail("url_path_invalid", "path is not canonical and bounded")
        for index, character in enumerate(self.path):
            if character == "%" and (
                index + 2 >= len(self.path)
                or any(
                    item not in "0123456789abcdefABCDEF"
                    for item in self.path[index + 1 : index + 3]
                )
            ):
                raise _fail("url_path_invalid", "path percent encoding is invalid")
        object.__setattr__(self, "host", host)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "HttpsLocation":
        if not isinstance(value, Mapping) or set(value) != {"scheme", "host", "path"}:
            raise _fail(
                "url_record_invalid",
                "location must contain only scheme, host, and path",
            )
        if not all(isinstance(value[key], str) for key in value):
            raise _fail("url_record_invalid", "location fields must be strings")
        return cls(
            scheme=str(value["scheme"]),
            host=str(value["host"]),
            path=str(value["path"]),
        )

    @classmethod
    def from_redirect(cls, raw: str, *, base: "HttpsLocation") -> "HttpsLocation":
        if not isinstance(raw, str) or len(raw.encode("utf-8", "replace")) > 4096:
            raise _fail("redirect_invalid", "redirect location is invalid")
        combined = urlsplit(urljoin(base.to_url(), raw))
        if (
            combined.scheme != "https"
            or combined.username is not None
            or combined.password is not None
            or combined.query
            or combined.fragment
        ):
            raise _fail(
                "redirect_not_allowed",
                "redirect must remain credential-free and query-free HTTPS",
            )
        try:
            port = combined.port
        except ValueError as exc:
            raise _fail("redirect_not_allowed", "redirect port is invalid") from exc
        if port not in (None, 443):
            raise _fail("redirect_not_allowed", "redirect port must be 443")
        if combined.hostname is None:
            raise _fail("redirect_not_allowed", "redirect host is missing")
        return cls(host=combined.hostname, path=combined.path or "/")

    def to_url(self) -> str:
        return f"https://{self.host}{self.path}"

    def to_mapping(self) -> dict[str, str]:
        return {"scheme": "https", "host": self.host, "path": self.path}


@dataclass(frozen=True)
class TransportLimits:
    connect_timeout_seconds: int = 5
    read_timeout_seconds: int = 15
    per_source_timeout_seconds: int = 30
    max_redirects: int = 3
    max_header_bytes: int = 64 * 1024
    max_transferred_bytes: int = 16 * 1024 * 1024
    max_decoded_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        bounds = {
            "connect_timeout_seconds": (self.connect_timeout_seconds, 1, 5),
            "read_timeout_seconds": (self.read_timeout_seconds, 1, 15),
            "per_source_timeout_seconds": (self.per_source_timeout_seconds, 1, 30),
            "max_redirects": (self.max_redirects, 0, 3),
            "max_header_bytes": (self.max_header_bytes, 1024, 64 * 1024),
            "max_transferred_bytes": (
                self.max_transferred_bytes,
                1024,
                16 * 1024 * 1024,
            ),
            "max_decoded_bytes": (
                self.max_decoded_bytes,
                1024,
                32 * 1024 * 1024,
            ),
        }
        for name, (value, minimum, maximum) in bounds.items():
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be in {minimum}..{maximum}")


@dataclass(frozen=True)
class FetchedDocument:
    source: HttpsLocation
    final_location: HttpsLocation
    content_type: str
    body: bytes
    transferred_bytes: int
    decoded_bytes: int
    redirect_count: int


class _SocketLike(Protocol):
    def sendall(self, data: bytes) -> None: ...
    def recv(self, size: int) -> bytes: ...
    def settimeout(self, value: float) -> None: ...
    def getpeername(self) -> tuple[object, ...]: ...
    def close(self) -> None: ...


Resolver = Callable[[str], tuple[str, ...]]
Dialer = Callable[[str, str, float, float], _SocketLike]


def _is_public_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if (
        not address.is_global
        or address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        return False
    return not any(address in network for network in _DENIED_NETWORKS)


def _system_resolver(host: str) -> tuple[str, ...]:
    try:
        records = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise _fail("dns_failed", "allowlisted host resolution failed") from exc
    addresses = tuple(sorted({str(record[4][0]) for record in records}))
    if not addresses:
        raise _fail("dns_failed", "allowlisted host resolved to no addresses")
    return addresses


def _tls_dial(ip: str, host: str, connect_timeout: float, read_timeout: float) -> _SocketLike:
    raw = socket.create_connection((ip, 443), timeout=connect_timeout)
    try:
        context = ssl.create_default_context()
        wrapped = context.wrap_socket(raw, server_hostname=host)
        wrapped.settimeout(read_timeout)
        return wrapped
    except Exception:
        raw.close()
        raise


class _Reader:
    def __init__(self, connection: _SocketLike, initial: bytes, deadline: float, read_timeout: int) -> None:
        self.connection = connection
        self.buffer = bytearray(initial)
        self.deadline = deadline
        self.read_timeout = read_timeout

    def _recv(self) -> bytes:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise _fail("source_timeout", "per-source collection deadline expired")
        self.connection.settimeout(min(float(self.read_timeout), remaining))
        try:
            return self.connection.recv(64 * 1024)
        except (OSError, TimeoutError) as exc:
            raise _fail("read_failed", "bounded HTTPS response read failed") from exc

    def until(self, delimiter: bytes, maximum: int) -> bytes:
        while True:
            index = self.buffer.find(delimiter)
            if index >= 0:
                end = index + len(delimiter)
                if end > maximum:
                    raise _fail("response_limit", "response framing exceeds its byte cap")
                result = bytes(self.buffer[:end])
                del self.buffer[:end]
                return result
            if len(self.buffer) >= maximum:
                raise _fail("response_limit", "response framing exceeds its byte cap")
            chunk = self._recv()
            if not chunk:
                raise _fail("response_invalid", "response ended before framing completed")
            self.buffer.extend(chunk)

    def exact(self, count: int) -> bytes:
        while len(self.buffer) < count:
            chunk = self._recv()
            if not chunk:
                raise _fail("response_invalid", "response body ended early")
            self.buffer.extend(chunk)
        result = bytes(self.buffer[:count])
        del self.buffer[:count]
        return result

    def to_eof(self, maximum: int) -> bytes:
        result = bytearray(self.buffer)
        self.buffer.clear()
        if len(result) > maximum:
            raise _fail("transfer_limit", "transferred body exceeds its byte cap")
        while True:
            chunk = self._recv()
            if not chunk:
                return bytes(result)
            result.extend(chunk)
            if len(result) > maximum:
                raise _fail("transfer_limit", "transferred body exceeds its byte cap")


def _parse_headers(data: bytes) -> tuple[int, dict[str, list[str]]]:
    try:
        text = data.decode("iso-8859-1")
    except UnicodeDecodeError as exc:
        raise _fail("response_invalid", "response headers are invalid") from exc
    lines = text[:-4].split("\r\n")
    if not lines or any(line.startswith((" ", "\t")) for line in lines[1:]):
        raise _fail("response_invalid", "response header folding is forbidden")
    status_parts = lines[0].split(" ", 2)
    if len(status_parts) < 2 or not status_parts[0].startswith("HTTP/1."):
        raise _fail("response_invalid", "response status line is invalid")
    try:
        status = int(status_parts[1])
    except ValueError as exc:
        raise _fail("response_invalid", "response status is invalid") from exc
    headers: dict[str, list[str]] = {}
    for line in lines[1:]:
        if not line:
            continue
        if ":" not in line:
            raise _fail("response_invalid", "response header is malformed")
        name, value = line.split(":", 1)
        lowered = name.strip().lower()
        if not lowered or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in lowered):
            raise _fail("response_invalid", "response header name is invalid")
        headers.setdefault(lowered, []).append(value.strip())
    return status, headers


def _single_header(headers: Mapping[str, list[str]], name: str) -> str | None:
    values = headers.get(name, [])
    if len(values) > 1:
        raise _fail("response_invalid", f"duplicate {name} header is forbidden")
    return values[0] if values else None


def _read_body(reader: _Reader, headers: Mapping[str, list[str]], maximum: int) -> bytes:
    transfer = _single_header(headers, "transfer-encoding")
    length = _single_header(headers, "content-length")
    if transfer is not None and length is not None:
        raise _fail("response_invalid", "ambiguous response body framing")
    if transfer is not None:
        if transfer.lower() != "chunked":
            raise _fail("response_invalid", "unsupported transfer encoding")
        body = bytearray()
        chunks = 0
        while True:
            chunks += 1
            if chunks > 200_000:
                raise _fail("response_limit", "chunk count exceeds its cap")
            line = reader.until(b"\r\n", 1024)
            token = line[:-2]
            if not token or b";" in token or any(character not in b"0123456789abcdefABCDEF" for character in token):
                raise _fail("response_invalid", "chunk framing is invalid")
            size = int(token, 16)
            if size == 0:
                trailer = reader.until(b"\r\n", 1024)
                if trailer != b"\r\n":
                    raise _fail("response_invalid", "response trailers are forbidden")
                return bytes(body)
            if len(body) + size > maximum:
                raise _fail("transfer_limit", "transferred body exceeds its byte cap")
            body.extend(reader.exact(size))
            if reader.exact(2) != b"\r\n":
                raise _fail("response_invalid", "chunk terminator is invalid")
    if length is not None:
        if not length.isascii() or not length.isdigit():
            raise _fail("response_invalid", "content length is invalid")
        count = int(length)
        if count > maximum:
            raise _fail("transfer_limit", "transferred body exceeds its byte cap")
        return reader.exact(count)
    return reader.to_eof(maximum)


def _decode_body(body: bytes, encoding: str | None, maximum: int) -> bytes:
    if encoding is None or encoding.lower() == "identity":
        if len(body) > maximum:
            raise _fail("decoded_limit", "decoded body exceeds its byte cap")
        return body
    normalized = encoding.lower()
    if normalized == "gzip":
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    elif normalized == "deflate":
        decoder = zlib.decompressobj()
    else:
        raise _fail("content_encoding_invalid", "unsupported content encoding")
    try:
        decoded = decoder.decompress(body, maximum + 1)
        if len(decoded) > maximum or decoder.unconsumed_tail:
            raise _fail("decoded_limit", "decoded body exceeds its byte cap")
        decoded += decoder.flush(maximum + 1 - len(decoded))
    except zlib.error as exc:
        raise _fail("content_encoding_invalid", "compressed response is invalid") from exc
    if len(decoded) > maximum:
        raise _fail("decoded_limit", "decoded body exceeds its byte cap")
    if not decoder.eof or decoder.unused_data:
        raise _fail("content_encoding_invalid", "compressed response has invalid trailing data")
    return decoded


class SafeHttpsTransport:
    """Fetch only vetted addresses for an exact structured allowlist."""

    def __init__(
        self,
        *,
        allowlist: Iterable[HttpsLocation],
        limits: TransportLimits | None = None,
        resolver: Resolver | None = None,
        dialer: Dialer | None = None,
    ) -> None:
        allowed = frozenset(allowlist)
        if not allowed:
            raise ValueError("allowlist must not be empty")
        self.allowlist = allowed
        self.limits = limits or TransportLimits()
        self._resolver = resolver or _system_resolver
        self._dialer = dialer or _tls_dial

    def _resolve(self, host: str) -> tuple[str, ...]:
        addresses = tuple(self._resolver(host))
        if not addresses or any(not _is_public_address(item) for item in addresses):
            raise _fail("address_not_public", "host resolved to a forbidden address class")
        return tuple(sorted(set(addresses)))

    def fetch(
        self,
        location: HttpsLocation,
        *,
        collection_deadline: float | None = None,
    ) -> FetchedDocument:
        if location not in self.allowlist:
            raise _fail("url_not_allowlisted", "requested location is not allowlisted")
        deadline = time.monotonic() + self.limits.per_source_timeout_seconds
        if collection_deadline is not None:
            deadline = min(deadline, collection_deadline)
        source = location
        current = location
        seen: set[HttpsLocation] = set()
        redirects = 0
        while True:
            if current in seen:
                raise _fail("redirect_loop", "redirect loop detected")
            seen.add(current)
            if time.monotonic() >= deadline:
                raise _fail("source_timeout", "per-source collection deadline expired")
            addresses = self._resolve(current.host)
            connection: _SocketLike | None = None
            last_error: Exception | None = None
            for address in addresses:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    connection = self._dialer(
                        address,
                        current.host,
                        min(float(self.limits.connect_timeout_seconds), remaining),
                        min(float(self.limits.read_timeout_seconds), remaining),
                    )
                    peer = str(connection.getpeername()[0])
                    if peer not in addresses or not _is_public_address(peer):
                        connection.close()
                        connection = None
                        raise _fail("peer_changed", "TLS peer address was not the vetted DNS result")
                    break
                except SafeHttpsError:
                    raise
                except (OSError, ssl.SSLError, TimeoutError) as exc:
                    last_error = exc
                    if connection is not None:
                        connection.close()
                        connection = None
            if connection is None:
                raise _fail("connect_failed", "all vetted HTTPS addresses failed") from last_error
            try:
                request = (
                    f"GET {current.path} HTTP/1.1\r\n"
                    f"Host: {current.host}\r\n"
                    "User-Agent: YOLOZU-Algorithm-Scout/1\r\n"
                    "Accept: application/json, application/pdf, text/html, text/plain\r\n"
                    "Accept-Encoding: gzip, deflate\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("ascii")
                connection.sendall(request)
                reader = _Reader(connection, b"", deadline, self.limits.read_timeout_seconds)
                header_bytes = reader.until(b"\r\n\r\n", self.limits.max_header_bytes)
                status, headers = _parse_headers(header_bytes)
                if status in _REDIRECT_STATUSES:
                    raw_location = _single_header(headers, "location")
                    if raw_location is None:
                        raise _fail("redirect_invalid", "redirect location is missing")
                    redirects += 1
                    if redirects > self.limits.max_redirects:
                        raise _fail("redirect_limit", "redirect count exceeds its cap")
                    target = HttpsLocation.from_redirect(raw_location, base=current)
                    if target not in self.allowlist:
                        raise _fail("redirect_not_allowlisted", "redirect target is not allowlisted")
                    current = target
                    continue
                if status != 200:
                    raise _fail("http_status", "allowlisted source returned a non-success status")
                raw_content_type = _single_header(headers, "content-type")
                if raw_content_type is None:
                    raise _fail("content_type_invalid", "content type is missing")
                content_type = raw_content_type.split(";", 1)[0].strip().lower()
                if content_type not in _ALLOWED_CONTENT_TYPES:
                    raise _fail("content_type_invalid", "content type is not an allowed document type")
                transferred = _read_body(reader, headers, self.limits.max_transferred_bytes)
                decoded = _decode_body(
                    transferred,
                    _single_header(headers, "content-encoding"),
                    self.limits.max_decoded_bytes,
                )
                return FetchedDocument(
                    source=source,
                    final_location=current,
                    content_type=content_type,
                    body=decoded,
                    transferred_bytes=len(transferred),
                    decoded_bytes=len(decoded),
                    redirect_count=redirects,
                )
            finally:
                connection.close()
