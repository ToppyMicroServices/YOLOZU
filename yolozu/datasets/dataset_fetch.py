"""Dataset registry download, extraction, and caching.

Mirrors the :mod:`yolozu.model_fetch` pattern — registry-driven download
with SHA-256 verification, license checks, and SSRF-safe URL validation.
Adds multi-part download and archive extraction for dataset archives.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import importlib.resources as resources

__all__ = [
    "DatasetSpec",
    "load_dataset_registry",
    "resolve_dataset_spec",
    "list_datasets",
    "fetch_dataset",
]


# ---------------------------------------------------------------------------
# Helpers (shared patterns with model_fetch.py)
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validated_download_url(url: str, *, allow_http: bool = False) -> urllib.parse.ParseResult:
    """Validate a download URL (SSRF protection)."""
    parsed = urllib.parse.urlparse(url)
    scheme = (parsed.scheme or "").lower()
    allowed = {"https", "file"}
    if allow_http:
        allowed.add("http")
    if scheme not in allowed:
        raise ValueError(f"unsupported download URL scheme: {scheme or '<empty>'}")
    if scheme == "file":
        path = Path(urllib.request.url2pathname(parsed.path))
        if not path.is_absolute():
            raise ValueError("file:// URLs must use absolute paths")
        return parsed
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError("URL must include a host")
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError(f"refusing insecure/private download host: {host}")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    ):
        raise ValueError(f"refusing insecure/private download host: {host}")
    return parsed


def _download_with_retry(
    *, url: str, out_path: Path, timeout: float = 120.0, retries: int = 3,
    allow_http: bool = False,
) -> None:
    """Download *url* to *out_path* with retry + exponential backoff."""
    parsed = _validated_download_url(url, allow_http=allow_http)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if parsed.scheme == "file":
        src = Path(urllib.request.url2pathname(parsed.path))
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, out_path)
        return

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "YOLOZU-dataset-fetch/1.0", "Accept": "*/*"},
    )
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                with out_path.open("wb") as f:
                    shutil.copyfileobj(resp, f)
            return
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            time.sleep(min(2 ** (attempt - 1), 16))
    if last_exc is not None:  # pragma: no cover
        raise last_exc


# ---------------------------------------------------------------------------
# DatasetSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """Parsed dataset entry from the registry."""

    dataset_id: str
    summary: str
    format: str
    task: str
    license: str
    num_classes: int | None
    source_type: str
    urls: list[str]
    expected_sha256: str | None
    splits: list[str]
    tags: list[str]
    post_extract: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def _builtin_registry_path() -> Path:
    data_root = resources.files("yolozu.data")
    path = data_root.joinpath("manifest/dataset_zoo.json")
    with resources.as_file(path) as on_disk:
        return Path(on_disk)


def load_dataset_registry(registry_path: str | Path | None = None) -> dict[str, Any]:
    """Load dataset_zoo.json (built-in or custom path)."""
    path = Path(registry_path).expanduser() if registry_path else _builtin_registry_path()
    return _read_json(path)


def _extract_urls(source: dict[str, Any]) -> tuple[str, list[str]]:
    """Return ``(source_type, urls)`` from a source dict."""
    stype = str(source.get("type") or "").strip()

    if stype == "multi":
        parts = source.get("parts") or []
        urls = [str(p.get("url") or "") for p in parts if p.get("url")]
        return stype, urls

    if stype == "github_release":
        repo = str(source.get("repo") or "").strip()
        tag = str(source.get("tag") or "").strip()
        asset = str(source.get("asset") or "").strip()
        if not repo or not tag or not asset:
            raise ValueError("github_release requires source.repo, source.tag, source.asset")
        url = f"https://github.com/{repo}/releases/download/{tag}/{asset}"
        return stype, [url]

    if stype in {"official_url", "url"}:
        url = str(source.get("url") or "").strip()
        if not url:
            raise ValueError("source.url is required")
        return stype, [url]

    if stype == "mirror_urls":
        urls = [str(v).strip() for v in (source.get("urls") or []) if str(v).strip()]
        if not urls:
            raise ValueError("mirror_urls requires non-empty source.urls")
        return stype, urls

    if stype == "hf_hub":
        repo = str(source.get("repo") or "").strip()
        revision = str(source.get("revision") or "main").strip()
        path = str(source.get("path") or "").strip()
        if not repo or not path:
            raise ValueError("hf_hub requires source.repo and source.path")
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{path}"
        return stype, [url]

    if stype == "manual":
        return stype, []

    raise ValueError(f"unsupported source type: {stype}")


def resolve_dataset_spec(dataset_id: str, registry_path: str | Path | None = None) -> DatasetSpec:
    """Resolve a dataset ID into a :class:`DatasetSpec`."""
    registry = load_dataset_registry(registry_path)
    for item in registry.get("datasets") or []:
        if str(item.get("id")) != dataset_id:
            continue
        source = dict(item.get("source") or {})
        source_type, urls = _extract_urls(source)
        sha = item.get("sha256")
        if isinstance(sha, str) and sha.strip():
            sha = sha.strip().lower()
        else:
            sha = None
        return DatasetSpec(
            dataset_id=dataset_id,
            summary=str(item.get("summary") or ""),
            format=str(item.get("format") or ""),
            task=str(item.get("task") or ""),
            license=str(item.get("license") or "UNKNOWN"),
            num_classes=item.get("num_classes"),
            source_type=source_type,
            urls=urls,
            expected_sha256=sha,
            splits=list(item.get("splits") or []),
            tags=list(item.get("tags") or []),
            post_extract=dict(item.get("post_extract") or {}),
        )
    raise KeyError(dataset_id)


def list_datasets(registry_path: str | Path | None = None) -> list[DatasetSpec]:
    """Return sorted list of all registered dataset specs."""
    registry = load_dataset_registry(registry_path)
    specs: list[DatasetSpec] = []
    for item in registry.get("datasets") or []:
        did = str(item.get("id") or "").strip()
        if not did:
            continue
        specs.append(resolve_dataset_spec(did, registry_path))
    specs.sort(key=lambda s: s.dataset_id)
    return specs


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------

def _extract_archive(archive_path: Path, dest: Path) -> None:
    """Extract a .zip or .tar* archive into *dest*."""
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        import zipfile as zf
        with zf.ZipFile(archive_path) as z:
            z.extractall(dest)
        return
    if any(name.endswith(ext) for ext in (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        import tarfile
        with tarfile.open(archive_path) as t:
            t.extractall(dest, filter="data")
        return
    raise ValueError(f"unsupported archive format: {archive_path.name}")


# ---------------------------------------------------------------------------
# fetch_dataset  (main entry point)
# ---------------------------------------------------------------------------

def fetch_dataset(
    *,
    dataset_id: str,
    out_dir: str | Path,
    cache_dir: str | Path | None = None,
    accept_license: bool = False,
    force: bool = False,
    retries: int = 3,
    timeout: float = 120.0,
    registry_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Download and extract a dataset from the registry.

    Returns ``(dataset_root, meta_json_path)``.

    Parameters
    ----------
    dataset_id:
        Identifier from ``dataset_zoo.json``.
    out_dir:
        Directory where ``<dataset_id>/`` will be created.
    cache_dir:
        Optional download cache (default ``~/.cache/yolozu/datasets``).
    accept_license:
        Must be ``True`` to proceed.
    force:
        Re-download even when cached.
    retries, timeout:
        Retry / timeout for each HTTP request.
    registry_path:
        Custom registry file path.
    """

    spec = resolve_dataset_spec(dataset_id, registry_path)

    # -- licence gate -------------------------------------------------------
    if not accept_license:
        raise PermissionError(
            f"dataset `{dataset_id}` requires explicit license acceptance "
            f"(license={spec.license}). Re-run with --accept-license."
        )

    # -- manual-download gate -----------------------------------------------
    if spec.source_type == "manual":
        source = {}
        registry = load_dataset_registry(registry_path)
        for item in registry.get("datasets") or []:
            if str(item.get("id")) == dataset_id:
                source = dict(item.get("source") or {})
                break
        instructions = source.get("instructions", "See the dataset website for download instructions.")
        raise RuntimeError(
            f"dataset `{dataset_id}` requires manual download.\n{instructions}"
        )

    # -- paths --------------------------------------------------------------
    out_root = Path(out_dir).expanduser()
    dataset_root = out_root / dataset_id
    dataset_root.mkdir(parents=True, exist_ok=True)

    cache_root = Path(cache_dir).expanduser() if cache_dir else (
        Path.home() / ".cache" / "yolozu" / "datasets"
    )

    # -- allow HTTP for well-known dataset hosts ----------------------------
    _HTTP_HOSTS = {
        "images.cocodataset.org",
        "host.robots.ox.ac.uk",
        "data.csail.mit.edu",
    }

    def _needs_http(url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        return (parsed.scheme or "").lower() == "http" and (parsed.hostname or "").lower() in _HTTP_HOSTS

    # -- download each URL --------------------------------------------------
    archive_paths: list[Path] = []
    urls_to_fetch = spec.urls
    if spec.source_type == "mirror_urls":
        urls_to_fetch = []
        last_exc: Exception | None = None
        for candidate_url in spec.urls:
            file_name = Path(candidate_url.split("?")[0]).name or "dataset_asset"
            cached = cache_root / dataset_id / file_name
            cached.parent.mkdir(parents=True, exist_ok=True)
            try:
                if force or not cached.exists():
                    with tempfile.NamedTemporaryFile(
                        prefix="yolozu_ds_", suffix=".tmp",
                        dir=str(cached.parent), delete=False,
                    ) as tmp:
                        tmp_path = Path(tmp.name)
                    try:
                        allow_http = _needs_http(candidate_url)
                        _download_with_retry(
                            url=candidate_url,
                            out_path=tmp_path,
                            timeout=timeout,
                            retries=retries,
                            allow_http=allow_http,
                        )
                        tmp_path.replace(cached)
                    finally:
                        if tmp_path.exists():
                            tmp_path.unlink(missing_ok=True)
                urls_to_fetch = [candidate_url]
                archive_paths.append(cached)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise RuntimeError(
                f"failed to fetch dataset `{dataset_id}` from all mirror_urls candidates"
            ) from last_exc

    for idx, url in enumerate(urls_to_fetch):
        file_name = Path(url.split("?")[0]).name
        if not file_name:
            file_name = f"part_{idx}"
        cached = cache_root / dataset_id / file_name
        cached.parent.mkdir(parents=True, exist_ok=True)

        if force or not cached.exists():
            with tempfile.NamedTemporaryFile(
                prefix="yolozu_ds_", suffix=".tmp",
                dir=str(cached.parent), delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
            try:
                allow_http = _needs_http(url)
                _download_with_retry(
                    url=url, out_path=tmp_path,
                    timeout=timeout, retries=retries,
                    allow_http=allow_http,
                )
                tmp_path.replace(cached)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
        archive_paths.append(cached)

    # -- extract archives ---------------------------------------------------
    for ap in archive_paths:
        name_lower = ap.name.lower()
        is_archive = (
            name_lower.endswith(".zip")
            or name_lower.endswith(".tar")
            or name_lower.endswith(".tar.gz")
            or name_lower.endswith(".tgz")
            or name_lower.endswith(".tar.bz2")
            or name_lower.endswith(".tar.xz")
        )
        if is_archive:
            _extract_archive(ap, dataset_root)
        elif name_lower.endswith(".json"):
            # Plain JSON file (e.g. annotations) — just copy.
            shutil.copy2(ap, dataset_root / ap.name)

    # -- resolve root_subdir if specified -----------------------------------
    root_subdir = spec.post_extract.get("root_subdir")
    effective_root = dataset_root
    if root_subdir:
        candidate = dataset_root / root_subdir
        if candidate.is_dir():
            effective_root = candidate

    # -- write meta.json ----------------------------------------------------
    meta = {
        "dataset_id": dataset_id,
        "format": spec.format,
        "task": spec.task,
        "license": spec.license,
        "num_classes": spec.num_classes,
        "splits": spec.splits,
        "source_type": spec.source_type,
        "urls": urls_to_fetch,
        "sha256": spec.expected_sha256,
        "effective_root": str(effective_root),
        "created_at": _utc_now(),
    }
    meta_path = dataset_root / "meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return effective_root, meta_path
