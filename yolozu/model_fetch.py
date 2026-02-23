from __future__ import annotations

import hashlib
import ipaddress
import json
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import importlib.resources as resources


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


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    summary: str
    family: str
    source_type: str
    source_url: str
    version: str
    license: str
    expected_sha256: str | None
    file_name: str


_APACHE_FRIENDLY_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "0bsd",
    "unlicense",
    "cc0-1.0",
}


def _is_apache_friendly_license(license_id: str) -> bool:
    normalized = (license_id or "").strip().lower()
    if not normalized:
        return False
    if normalized in _APACHE_FRIENDLY_LICENSES:
        return True
    # Conservative heuristic for non-standard strings: treat any GPL-family token as non-Apache-friendly.
    if "agpl" in normalized or "gpl" in normalized or "lgpl" in normalized:
        return False
    return False


def _source_url(source: dict[str, Any]) -> str:
    source_type = str(source.get("type") or "").strip()
    if source_type in {"official_url", "url"}:
        url = str(source.get("url") or "").strip()
        if not url:
            raise ValueError("source.url is required for official_url")
        return url
    if source_type == "github_release":
        repo = str(source.get("repo") or "").strip()
        tag = str(source.get("tag") or "").strip()
        asset = str(source.get("asset") or "").strip()
        if not repo or not tag or not asset:
            raise ValueError("github_release requires source.repo, source.tag, source.asset")
        return f"https://github.com/{repo}/releases/download/{tag}/{asset}"
    if source_type == "hf_hub":
        repo = str(source.get("repo") or "").strip()
        revision = str(source.get("revision") or "main").strip()
        path = str(source.get("path") or "").strip()
        if not repo or not path:
            raise ValueError("hf_hub requires source.repo and source.path")
        return f"https://huggingface.co/{repo}/resolve/{revision}/{path}"
    raise ValueError(f"unsupported source.type: {source_type}")


def _source_asset_name(source: dict[str, Any], url: str) -> str:
    source_type = str(source.get("type") or "").strip()
    if source_type == "github_release":
        return str(source.get("asset"))
    if source_type == "hf_hub":
        return Path(str(source.get("path"))).name
    explicit = str(source.get("file_name") or "").strip()
    if explicit:
        return explicit
    return Path(url.split("?")[0]).name


def _validated_download_url(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"https", "file"}:
        raise ValueError(f"unsupported download URL scheme: {scheme or '<empty>'}")
    if scheme == "file":
        path = Path(urllib.request.url2pathname(parsed.path))
        if not path.is_absolute():
            raise ValueError("file:// model URLs must use absolute paths")
        return parsed

    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError("https model URL must include host")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        raise ValueError(f"refusing insecure/private download host: {host}")
    return parsed


def _builtin_registry_path() -> Path:
    data_root = resources.files("yolozu.data")
    path = data_root.joinpath("manifest/model_zoo.json")
    with resources.as_file(path) as on_disk:
        return Path(on_disk)


def load_registry(registry_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(registry_path).expanduser() if registry_path else _builtin_registry_path()
    return _read_json(path)


def resolve_model_spec(model_id: str, registry_path: str | Path | None = None) -> ModelSpec:
    registry = load_registry(registry_path)
    models = registry.get("models") or []
    for item in models:
        if str(item.get("id")) != model_id:
            continue
        source = dict(item.get("source") or {})
        source_type = str(source.get("type") or "")
        url = _source_url(source)
        file_name = _source_asset_name(source, url)
        expected_sha = item.get("sha256")
        if isinstance(expected_sha, str) and expected_sha.strip():
            expected_sha = expected_sha.strip().lower()
        else:
            expected_sha = None
        return ModelSpec(
            model_id=model_id,
            summary=str(item.get("summary") or ""),
            family=str(item.get("family") or "generic"),
            source_type=source_type,
            source_url=url,
            version=str(item.get("version") or "unknown"),
            license=str(item.get("license") or "UNKNOWN"),
            expected_sha256=expected_sha,
            file_name=file_name,
        )
    raise KeyError(model_id)


def list_models(registry_path: str | Path | None = None) -> list[ModelSpec]:
    registry = load_registry(registry_path)
    models = registry.get("models") or []
    specs: list[ModelSpec] = []
    for item in models:
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        specs.append(resolve_model_spec(model_id, registry_path))
    specs.sort(key=lambda spec: spec.model_id)
    return specs


def _download(url: str, out_path: Path) -> None:
    return _download_with_retry(url=url, out_path=out_path, timeout=60.0, retries=3)


def _download_with_retry(*, url: str, out_path: Path, timeout: float, retries: int) -> None:
    parsed = _validated_download_url(url)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if parsed.scheme == "file":
        src_path = Path(urllib.request.url2pathname(parsed.path))
        if not src_path.exists():
            raise FileNotFoundError(src_path)
        shutil.copy2(src_path, out_path)
        return

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "YOLOZU-model-fetch/1.0",
            "Accept": "*/*",
        },
    )
    if retries <= 0:
        raise ValueError("--retries must be >= 1")
    if timeout <= 0:
        raise ValueError("--timeout must be > 0")

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with out_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            return
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            if attempt >= retries:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))
    if last_exc is not None:  # pragma: no cover
        raise last_exc


def fetch_model(
    *,
    model_id: str,
    out_dir: str | Path,
    cache_dir: str | Path | None,
    accept_license: bool,
    allow_unsafe: bool = False,
    allow_non_apache: bool = False,
    retries: int = 3,
    timeout: float = 60.0,
    registry_path: str | Path | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    spec = resolve_model_spec(model_id, registry_path)
    apache_friendly = _is_apache_friendly_license(spec.license)
    if not accept_license:
        extra = " --allow-non-apache" if not apache_friendly else ""
        raise PermissionError(
            f"model `{model_id}` requires explicit license acceptance "
            f"(license={spec.license}). Re-run with --accept-license{extra}."
        )
    if not apache_friendly and not allow_non_apache:
        raise PermissionError(
            f"model `{model_id}` is not Apache-friendly (license={spec.license}). "
            f"Use a separate baseline environment and import only predictions.json when possible. "
            f"To proceed anyway, re-run with --allow-non-apache."
        )
    if spec.expected_sha256 is None and not allow_unsafe:
        raise PermissionError(
            f"model `{model_id}` is missing sha256 in the registry (unsafe to fetch reproducibly). "
            f"Update the registry to include sha256, or re-run with --allow-unsafe."
        )

    out_root = Path(out_dir).expanduser()
    model_root = out_root / model_id
    model_root.mkdir(parents=True, exist_ok=True)
    output_path = model_root / spec.file_name

    cache_root = Path(cache_dir).expanduser() if cache_dir else (Path.home() / ".cache" / "yolozu" / "models")
    cached_path = cache_root / model_id / spec.version / spec.file_name
    cached_path.parent.mkdir(parents=True, exist_ok=True)

    should_download = force or (not cached_path.exists())
    if should_download:
        with tempfile.NamedTemporaryFile(prefix="yolozu_fetch_", suffix=".tmp", dir=str(cached_path.parent), delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _download_with_retry(url=spec.source_url, out_path=tmp_path, timeout=float(timeout), retries=int(retries))
            downloaded_sha = _sha256(tmp_path)
            if spec.expected_sha256 and downloaded_sha != spec.expected_sha256:
                raise RuntimeError(
                    f"sha256 mismatch for {model_id}: expected {spec.expected_sha256}, got {downloaded_sha}"
                )
            tmp_path.replace(cached_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    cached_sha = _sha256(cached_path)
    if spec.expected_sha256 and cached_sha != spec.expected_sha256:
        raise RuntimeError(
            f"cached sha256 mismatch for {model_id}: expected {spec.expected_sha256}, got {cached_sha}"
        )

    if force or (not output_path.exists()) or (_sha256(output_path) != cached_sha):
        shutil.copy2(cached_path, output_path)

    meta = {
        "model_id": model_id,
        "source": spec.source_type,
        "source_url": spec.source_url,
        "version": spec.version,
        "license": spec.license,
        "sha256": cached_sha,
        "created_at": _utc_now(),
        "cached_path": str(cached_path),
        "file_name": spec.file_name,
    }
    meta_path = model_root / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path, meta_path
