"""Prepare one exact Torchvision checkpoint for the adaptive image runner.

This module is intentionally local-only. It does not download model bytes and
does not treat the YOLOZU Apache-2.0 license as a license for the checkpoint.
The caller must provide the exact upstream checkpoint and acknowledge the
upstream pretrained-model notice before conversion.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "MASKRCNN_ARTIFACT_CACHE_KEY",
    "MASKRCNN_CHECKPOINT_SHA256",
    "MASKRCNN_CHECKPOINT_SIZE_BYTES",
    "MASKRCNN_CHECKPOINT_URL",
    "MASKRCNN_SAFETENSORS_SHA256",
    "MASKRCNN_SAFETENSORS_SIZE_BYTES",
    "MASKRCNN_UPSTREAM_TERMS_URL",
    "PreparedCheckpoint",
    "prepare_torchvision_maskrcnn_checkpoint",
]


MASKRCNN_CHECKPOINT_URL = (
    "https://download.pytorch.org/models/"
    "maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"
)
MASKRCNN_CHECKPOINT_SHA256 = (
    "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e"
)
MASKRCNN_CHECKPOINT_SIZE_BYTES = 185_828_065
MASKRCNN_UPSTREAM_TERMS_URL = (
    "https://github.com/pytorch/vision#pre-trained-model-license"
)
MASKRCNN_ARTIFACT_CACHE_KEY = (
    "torchvision/maskrcnn-resnet50-fpn-v2-coco-v1/model.safetensors"
)
MASKRCNN_SAFETENSORS_SHA256 = (
    "59f39b25a05a7130ebb754f82451e44b791c684601a15cefc64f79462e1e8187"
)
MASKRCNN_SAFETENSORS_SIZE_BYTES = 185_728_220


@dataclass(frozen=True)
class PreparedCheckpoint:
    artifact_path: Path
    provenance_path: Path
    provenance: Mapping[str, Any]
    artifact_reused: bool


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _verified_source_handle(path: Path):
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("checkpoint preparation requires O_NOFOLLOW")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    handle = os.fdopen(descriptor, "rb")
    try:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("source checkpoint must be a regular file")
        if before.st_size != MASKRCNN_CHECKPOINT_SIZE_BYTES:
            raise ValueError(
                "source checkpoint size mismatch: expected "
                f"{MASKRCNN_CHECKPOINT_SIZE_BYTES}, got {before.st_size}"
            )
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != MASKRCNN_CHECKPOINT_SHA256:
            raise ValueError(
                "source checkpoint sha256 mismatch: expected "
                f"{MASKRCNN_CHECKPOINT_SHA256}, got {observed_sha256}"
            )
        after = os.fstat(handle.fileno())
        identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
            raise ValueError("source checkpoint changed while it was being verified")
        handle.seek(0)
        return handle
    except Exception:
        handle.close()
        raise


def _convert_verified_checkpoint(source_handle: Any, destination: Path) -> None:
    try:
        import torch
        from safetensors.torch import save_file as save_safetensors_file
    except ImportError as exc:
        raise RuntimeError(
            "checkpoint preparation requires torch and safetensors; "
            "install the YOLOZU demo extra"
        ) from exc

    descriptor_path = f"/dev/fd/{source_handle.fileno()}"
    state = torch.load(
        descriptor_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(state, Mapping) or not state:
        raise ValueError("source checkpoint must contain a non-empty state dictionary")
    converted: dict[str, Any] = {}
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not name:
            raise ValueError("source checkpoint state keys must be non-empty strings")
        if not torch.is_tensor(value):
            raise ValueError(f"source checkpoint value is not a tensor: {name}")
        converted[name] = value.contiguous()
    save_safetensors_file(converted, str(destination))


def _ensure_private_directory(path: Path) -> Path:
    selected = path.expanduser()
    if selected.exists() and selected.is_symlink():
        raise ValueError("artifact root must not be a symlink")
    selected.mkdir(mode=0o700, parents=True, exist_ok=True)
    selected = selected.resolve(strict=True)
    expected_uid = getattr(os, "getuid", lambda: selected.stat().st_uid)()
    if selected.stat().st_uid != expected_uid:
        raise ValueError("artifact root must be owned by the current user")
    current = selected
    for component in Path(MASKRCNN_ARTIFACT_CACHE_KEY).parts[:-1]:
        current = current / component
        if current.exists():
            if current.is_symlink() or not current.is_dir():
                raise ValueError("artifact cache-key parent must be a real directory")
        else:
            current.mkdir(mode=0o700)
        if current.stat().st_uid != expected_uid:
            raise ValueError("artifact cache-key parent must be owned by the current user")
    return selected


def _verify_installed_artifact(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("prepared artifact path must be a regular non-symlink file")
    observed_size = path.stat().st_size
    if observed_size != MASKRCNN_SAFETENSORS_SIZE_BYTES:
        raise ValueError(
            "prepared artifact size mismatch: expected "
            f"{MASKRCNN_SAFETENSORS_SIZE_BYTES}, got {observed_size}"
        )
    observed_sha256 = _sha256_path(path)
    if observed_sha256 != MASKRCNN_SAFETENSORS_SHA256:
        raise ValueError(
            "prepared artifact sha256 mismatch: expected "
            f"{MASKRCNN_SAFETENSORS_SHA256}, got {observed_sha256}"
        )


def _install_exclusive(path: Path, temporary: Path) -> bool:
    """Install one verified temporary file without overwriting."""
    if path.exists() or path.is_symlink():
        _verify_installed_artifact(path)
        return True
    _verify_installed_artifact(temporary)
    try:
        os.link(temporary, path)
    except FileExistsError:
        _verify_installed_artifact(path)
        return True
    return False


def _provenance_payload(
    *, prepared_at: str, conversion_performed: bool
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "yolozu_local_torchvision_checkpoint_preparation",
        "model_id": "torchvision-maskrcnn-r50-fpn-v2-coco-v1",
        "prepared_at": prepared_at,
        "source": {
            "url": MASKRCNN_CHECKPOINT_URL,
            "sha256": MASKRCNN_CHECKPOINT_SHA256,
            "size_bytes": MASKRCNN_CHECKPOINT_SIZE_BYTES,
            "provided_by_user": True,
            "downloaded_by_yolozu": False,
            "local_path_recorded": False,
        },
        "upstream_terms": {
            "url": MASKRCNN_UPSTREAM_TERMS_URL,
            "acknowledged_by_user": True,
        },
        "conversion": {
            "performed_by_this_invocation": conversion_performed,
            "source_load": (
                "torch.load(weights_only=True,map_location=cpu,mmap=True)"
            ),
            "format": "safetensors",
            "recipe": (
                "safetensors.torch.save_file({name: state[name].contiguous() "
                "for name in sorted(state)})"
            ),
            "runtime_versions": {
                "torch": _dependency_version("torch"),
                "safetensors": _dependency_version("safetensors"),
            },
        },
        "artifact": {
            "cache_key": MASKRCNN_ARTIFACT_CACHE_KEY,
            "sha256": MASKRCNN_SAFETENSORS_SHA256,
            "size_bytes": MASKRCNN_SAFETENSORS_SIZE_BYTES,
        },
        "licensing": {
            "yolozu_source_license": "Apache-2.0",
            "checkpoint_license_expression": "NOASSERTION",
            "checkpoint_relicensed_by_yolozu": False,
            "source_checkpoint_redistributed_by_yolozu": False,
            "converted_artifact_redistributed_by_yolozu": False,
        },
    }


def _provenance_matches(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    source = payload.get("source")
    terms = payload.get("upstream_terms")
    conversion = payload.get("conversion")
    artifact = payload.get("artifact")
    licensing = payload.get("licensing")
    return bool(
        payload.get("schema_version") == 1
        and payload.get("kind") == "yolozu_local_torchvision_checkpoint_preparation"
        and isinstance(source, Mapping)
        and source.get("url") == MASKRCNN_CHECKPOINT_URL
        and source.get("sha256") == MASKRCNN_CHECKPOINT_SHA256
        and source.get("size_bytes") == MASKRCNN_CHECKPOINT_SIZE_BYTES
        and source.get("provided_by_user") is True
        and source.get("downloaded_by_yolozu") is False
        and source.get("local_path_recorded") is False
        and isinstance(terms, Mapping)
        and terms.get("url") == MASKRCNN_UPSTREAM_TERMS_URL
        and terms.get("acknowledged_by_user") is True
        and isinstance(conversion, Mapping)
        and conversion.get("format") == "safetensors"
        and conversion.get("source_load")
        == "torch.load(weights_only=True,map_location=cpu,mmap=True)"
        and conversion.get("recipe")
        == (
            "safetensors.torch.save_file({name: state[name].contiguous() "
            "for name in sorted(state)})"
        )
        and isinstance(conversion.get("performed_by_this_invocation"), bool)
        and isinstance(artifact, Mapping)
        and artifact.get("cache_key") == MASKRCNN_ARTIFACT_CACHE_KEY
        and artifact.get("sha256") == MASKRCNN_SAFETENSORS_SHA256
        and artifact.get("size_bytes") == MASKRCNN_SAFETENSORS_SIZE_BYTES
        and isinstance(licensing, Mapping)
        and licensing.get("yolozu_source_license") == "Apache-2.0"
        and licensing.get("checkpoint_license_expression") == "NOASSERTION"
        and licensing.get("checkpoint_relicensed_by_yolozu") is False
        and licensing.get("source_checkpoint_redistributed_by_yolozu") is False
        and licensing.get("converted_artifact_redistributed_by_yolozu") is False
    )


def _write_provenance(
    path: Path, payload: Mapping[str, Any]
) -> Mapping[str, Any]:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError("provenance path must be a regular non-symlink file")
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not _provenance_matches(existing):
            raise ValueError("existing provenance does not match the exact checkpoint")
        return existing
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=".yolozu-maskrcnn-provenance-",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        os.chmod(handle.name, 0o600)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not _provenance_matches(existing):
                raise ValueError("existing provenance does not match the exact checkpoint")
            return existing
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def prepare_torchvision_maskrcnn_checkpoint(
    *,
    checkpoint_path: str | Path,
    artifact_root: str | Path | None = None,
    accept_upstream_terms: bool,
) -> PreparedCheckpoint:
    """Verify and locally convert the exact pinned checkpoint.

    No network operation is performed. The source path is not written to the
    provenance record, and existing destination bytes are never overwritten.
    """
    if not accept_upstream_terms:
        raise PermissionError(
            "explicit upstream-terms acknowledgement is required; review "
            f"{MASKRCNN_UPSTREAM_TERMS_URL} and re-run with --accept-upstream-terms"
        )
    source = Path(checkpoint_path).expanduser()
    try:
        source_handle = _verified_source_handle(source)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"source checkpoint not found: {source}") from exc
    with source_handle:
        root = _ensure_private_directory(
            Path(artifact_root)
            if artifact_root is not None
            else Path.home() / ".cache" / "yolozu" / "models"
        )
        artifact_path = root.joinpath(*Path(MASKRCNN_ARTIFACT_CACHE_KEY).parts)
        provenance_path = artifact_path.with_suffix(".safetensors.provenance.json")
        artifact_reused = False
        if artifact_path.exists() or artifact_path.is_symlink():
            _verify_installed_artifact(artifact_path)
            artifact_reused = True
        else:
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=".yolozu-maskrcnn-",
                    suffix=".tmp",
                    dir=artifact_path.parent,
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    os.chmod(handle.name, 0o600)
                _convert_verified_checkpoint(source_handle, temporary)
                artifact_reused = _install_exclusive(artifact_path, temporary)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    provenance = _provenance_payload(
        prepared_at=datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        conversion_performed=not artifact_reused,
    )
    provenance = _write_provenance(provenance_path, provenance)
    return PreparedCheckpoint(
        artifact_path=artifact_path,
        provenance_path=provenance_path,
        provenance=provenance,
        artifact_reused=artifact_reused,
    )
