"""Bounded single-tenant image-job service for AI-facing MCP clients.

The service accepts image bytes rather than caller-controlled paths, builds the
typed adaptive request itself, and exposes only opaque asset/job identifiers.
Model selection and execution remain delegated to the existing qualified
adaptive pipeline.  An unavailable qualified pipeline therefore produces an
explicit abstention instead of falling back to an arbitrary backend.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import shutil
import stat
import threading
import time
import uuid
import warnings
import fcntl
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, UnidentifiedImageError

from .image_job_execution import run_image_job

from .layers.jobs import JobManager
from .image_service_http import HTTP_UPLOAD_TIMEOUT_SECONDS, MAX_HTTP_BODY_BYTES
from .manifest_resources import workspace_root as resolved_workspace_root

MAX_ASSET_BYTES = 8 * 1024 * 1024
MAX_IMAGE_DIMENSION = 16_384
MAX_IMAGE_PIXELS = 64_000_000
MAX_LABELS = 128
MAX_LABEL_BYTES = 256
MAX_ASSETS_PER_TENANT = 128
MAX_ACTIVE_JOBS = 16
MAX_JOB_RECORDS = 1024
MIN_RETENTION_SECONDS = 300
MAX_RETENTION_SECONDS = 7 * 24 * 60 * 60
DEFAULT_RETENTION_SECONDS = 24 * 60 * 60
CLEANUP_INTERVAL_SECONDS = 60
RATE_WINDOW_SECONDS = 60
REQUEST_LIMITS = {"capabilities": 120, "upload": 12, "submit": 12, "get": 120, "cancel": 30}

_TENANT_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_ASSET_RE = re.compile(r"asset_[0-9a-f]{24}\Z")
_JOB_RE = re.compile(r"job_[0-9a-f]{12}\Z")
_MEDIA_BY_FORMAT = {
    "JPEG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
    "WEBP": ("image/webp", ".webp"),
}


class ImageServiceError(ValueError):
    """Stable public rejection from the image service boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.public_message = message


def _fail(code: str, message: str) -> ImageServiceError:
    return ImageServiceError(code, message)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_json_read(path: Path, *, maximum_bytes: int = 2 * 1024 * 1024) -> Any:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        handle = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= maximum_bytes:
            raise _fail("stored_result_invalid", "stored service data exceeds its bound")
        data = handle.read(maximum_bytes + 1)
        if len(data) != info.st_size:
            raise _fail("stored_result_invalid", "stored service data changed")
    return json.loads(data)


def _write_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = json.dumps(dict(payload), ensure_ascii=False, indent=2).encode("utf-8")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            # A successful replace has already consumed the temporary file.
            pass


def _normalized_labels(values: list[str]) -> list[str]:
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_LABELS:
        raise _fail("invalid_labels", f"fixed_classes must contain 1..{MAX_LABELS} labels")
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise _fail("invalid_labels", "each fixed class must be a string")
        item = value.strip()
        if (
            not item
            or len(item.encode("utf-8")) > MAX_LABEL_BYTES
            or any(ord(character) < 32 or ord(character) == 127 for character in item)
        ):
            raise _fail("invalid_labels", "a fixed class is empty, too long, or contains controls")
        key = item.casefold()
        if key in seen:
            raise _fail("invalid_labels", "fixed classes must be unique")
        seen.add(key)
        normalized.append(item)
    return normalized


class ImageService:
    """One credential-bound tenant service rooted below a workspace."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        tenant_id: str = "local",
        retention_seconds: int = DEFAULT_RETENTION_SECONDS,
        service_root: str | Path = "runs/mcp_image_service",
    ) -> None:
        if _TENANT_RE.fullmatch(tenant_id) is None:
            raise _fail("invalid_tenant", "tenant_id must be a bounded lowercase identifier")
        if (
            isinstance(retention_seconds, bool)
            or not isinstance(retention_seconds, int)
            or not MIN_RETENTION_SECONDS <= retention_seconds <= MAX_RETENTION_SECONDS
        ):
            raise _fail(
                "invalid_retention",
                f"retention_seconds must be in {MIN_RETENTION_SECONDS}..{MAX_RETENTION_SECONDS}",
            )
        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir():
            raise _fail("invalid_workspace", "workspace must be a directory")
        root_value = Path(service_root)
        if root_value.is_absolute():
            raise _fail("invalid_service_root", "service_root must be workspace-relative")
        if not root_value.parts or any(part == ".." for part in root_value.parts):
            raise _fail("invalid_service_root", "service_root must name a workspace subdirectory")
        root = self.workspace / root_value
        try:
            root.relative_to(self.workspace)
        except ValueError as exc:
            raise _fail("invalid_service_root", "service_root escaped the workspace") from exc
        current = self.workspace
        for component in root.relative_to(self.workspace).parts:
            current = current / component
            if current.is_symlink():
                raise _fail("invalid_service_root", "service_root contains a symlink")
        self.tenant_id = tenant_id
        self.retention_seconds = retention_seconds
        self.root = root
        self.tenant_root = root / "tenants" / tenant_id
        self.assets_root = self.tenant_root / "assets"
        self.jobs_root = self.tenant_root / "jobs"
        self.outputs_root = self.tenant_root / "outputs"
        self._jobs: JobManager | None = None
        self._state_lock = threading.RLock()
        self._active_assets: dict[str, int] = {}
        self._job_assets: dict[str, tuple[str, str]] = {}
        self._request_times: dict[str, deque[float]] = {key: deque() for key in REQUEST_LIMITS}
        self._maintenance_stop = threading.Event()
        self._maintenance_thread: threading.Thread | None = None
        self._maintenance_failed = False
        self._ownership = None
        self._closed = False

    def _admit(self, operation: str) -> None:
        with self._state_lock:
            if self._closed:
                raise _fail("service_closed", "image service is closed")
            if self._maintenance_failed:
                raise _fail("retention_unavailable", "retention cleanup needs operator attention")
            now = time.monotonic()
            attempts = self._request_times[operation]
            while attempts and attempts[0] <= now - RATE_WINDOW_SECONDS:
                attempts.popleft()
            if len(attempts) >= REQUEST_LIMITS[operation]:
                raise _fail("rate_limited", "tenant request limit reached; retry after 60 seconds")
            attempts.append(now)

    def start_maintenance(self) -> None:
        """Start one idle-time retention worker; no model or network is used."""
        with self._state_lock:
            if self._closed:
                raise _fail("service_closed", "image service is closed")
            if self._maintenance_thread is not None:
                return
            self.purge_expired()
            self._maintenance_stop.clear()

            def maintain() -> None:
                while not self._maintenance_stop.wait(CLEANUP_INTERVAL_SECONDS):
                    try:
                        self.purge_expired()
                        self._maintenance_failed = False
                    except (OSError, ValueError):
                        self._maintenance_failed = True

            self._maintenance_thread = threading.Thread(
                target=maintain, name="yolozu-image-retention", daemon=True,
            )
            self._maintenance_thread.start()

    def close(self) -> None:
        """Stop admission, retention, and queued/running owned jobs before unlocking."""
        with self._state_lock:
            self._closed = True
        self._maintenance_stop.set()
        if self._maintenance_thread is not None:
            self._maintenance_thread.join(timeout=5)
            if self._maintenance_thread.is_alive():
                raise RuntimeError("retention shutdown exceeded its cleanup deadline")
            self._maintenance_thread = None
        # Workers release their asset references under _state_lock. Never wait
        # for a worker while holding that lock.
        if self._jobs is not None:
            self._jobs.shutdown(timeout=5)
        with self._state_lock:
            for job_id in list(self._job_assets):
                self._release_job(job_id)
            if self._ownership is not None:
                self._ownership.close()
                self._ownership = None

    def _ensure_storage(self) -> None:
        with self._state_lock:
            if self._closed:
                raise _fail("service_closed", "image service is closed")
            self._ensure_owned_storage()

    def _ensure_owned_storage(self) -> None:
        for directory in (
            self.tenant_root,
            self.assets_root,
            self.jobs_root,
            self.outputs_root,
        ):
            current = self.workspace
            for component in directory.relative_to(self.workspace).parts:
                current = current / component
                if current.is_symlink():
                    raise _fail("unsafe_service_storage", "service storage contains a symlink")
                current.mkdir(exist_ok=True, mode=0o700)
            os.chmod(directory, 0o700)
        if self._ownership is None:
            descriptor = os.open(self.tenant_root / ".owner.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise _fail("unsafe_service_storage", "tenant lock is not a regular file")
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._ownership = os.fdopen(descriptor, "rb")
            except BaseException as exc:
                os.close(descriptor)
                if isinstance(exc, BlockingIOError):
                    raise _fail("tenant_in_use", "another service owns this tenant directory") from exc
                raise
            # Keep the lock inode: unlinking it would let two owners lock
            # different files with the same name.

    def _job_manager(self) -> JobManager:
        with self._state_lock:
            self._ensure_storage()
            if self._jobs is None:
                self._jobs = JobManager(max_workers=1, storage_dir=self.jobs_root)
            return self._jobs

    def _release_job(self, job_id: str) -> None:
        with self._state_lock:
            binding = self._job_assets.pop(job_id, None)
            if binding is None:
                return
            asset_id, _output_token = binding
            directory = self._asset_directory(asset_id)
            try:
                if directory.is_dir() and not directory.is_symlink():
                    os.utime(directory, None)
            except OSError:
                # A failed retention timestamp refresh must not pin an asset
                # forever after its job has already stopped.
                pass
            count = self._active_assets.get(asset_id, 0)
            if count <= 1:
                self._active_assets.pop(asset_id, None)
            else:
                self._active_assets[asset_id] = count - 1

    def _asset_directory(self, asset_id: str) -> Path:
        if _ASSET_RE.fullmatch(asset_id) is None:
            raise _fail("invalid_asset_id", "asset_id is invalid")
        return self.assets_root / asset_id

    def _job_output_directory(self, output_token: str) -> Path:
        if re.fullmatch(r"run_[0-9a-f]{24}", output_token) is None:
            raise _fail("stored_result_invalid", "stored output identity is invalid")
        return self.outputs_root / output_token

    def _remove_expired_tree(self, path: Path, *, now: float) -> bool:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return False
        if path.is_symlink() or not path.is_dir():
            return False
        if now - info.st_mtime < self.retention_seconds:
            return False
        try:
            path.relative_to(self.tenant_root)
        except ValueError:
            return False
        shutil.rmtree(path)
        return True

    def purge_expired(self) -> dict[str, int]:
        """Remove only expired service-owned asset and output directories."""

        if not self.tenant_root.exists():
            return {"assets": 0, "outputs": 0, "jobs": 0}
        self._ensure_storage()
        now = time.time()
        with self._state_lock:
            active_assets = set(self._active_assets)
            active_outputs = {value[1] for value in self._job_assets.values()}
            removed_assets = sum(
                self._remove_expired_tree(path, now=now)
                for path in self.assets_root.glob("asset_*")
                if _ASSET_RE.fullmatch(path.name) and path.name not in active_assets
            )
            removed_outputs = sum(
                self._remove_expired_tree(path, now=now)
                for path in self.outputs_root.glob("run_*")
                if re.fullmatch(r"run_[0-9a-f]{24}", path.name) and path.name not in active_outputs
            )
            # SIGKILL cannot execute the transaction's Python finally block.
            # Expire only exact service-owned transaction siblings, protecting
            # active output tokens just as for their published directories.
            for path in self.outputs_root.glob(".run_*"):
                match = re.fullmatch(r"\.(run_[0-9a-f]{24})\.(?:(?:stage|backup)\.[0-9a-f]{32}|yolozu-output-transaction\.json)", path.name)
                if match is None or match[1] in active_outputs:
                    continue
                if path.name.endswith(".json"):
                    info = path.lstat()
                    if stat.S_ISREG(info.st_mode) and now - info.st_mtime >= self.retention_seconds:
                        path.unlink()
                        removed_outputs += 1
                else:
                    removed_outputs += self._remove_expired_tree(path, now=now)
            removed_jobs = self._job_manager().purge_terminal_before(
                now - self.retention_seconds
            )
        return {
            "assets": removed_assets,
            "outputs": removed_outputs,
            "jobs": removed_jobs,
        }

    def capabilities(self) -> dict[str, Any]:
        self._admit("capabilities")
        return {
            "schema_version": 1,
            "ok": True,
            "tool": "image_service_capabilities",
            "summary": "reported the bounded image-service interface contract",
            "service": {
                "scope": "single_credential_single_tenant",
                "tasks": ["object_detection", "instance_segmentation"],
                "prompt_modes": ["fixed_classes"],
                "input_media_types": sorted(value[0] for value in _MEDIA_BY_FORMAT.values()),
                "max_asset_bytes": MAX_ASSET_BYTES,
                "max_http_body_bytes": MAX_HTTP_BODY_BYTES,
                "http_upload_timeout_seconds": HTTP_UPLOAD_TIMEOUT_SECONDS,
                "max_image_dimension": MAX_IMAGE_DIMENSION,
                "max_image_pixels": MAX_IMAGE_PIXELS,
                "max_images_per_job": 1,
                "max_assets_per_tenant": MAX_ASSETS_PER_TENANT,
                "max_active_jobs": MAX_ACTIVE_JOBS,
                "max_job_records": MAX_JOB_RECORDS,
                "network_policy": "deny",
                "os_network_isolation": False,
                "execution_default": False,
                "selection_policy": "qualified_registered_pipeline_or_abstain",
                "retention_seconds": self.retention_seconds,
                "cleanup_interval_seconds": CLEANUP_INTERVAL_SECONDS,
                "request_limits_per_60_seconds": dict(REQUEST_LIMITS),
            },
        }

    def put_asset(
        self,
        *,
        content_base64: str,
        media_type: str,
    ) -> dict[str, Any]:
        self._admit("upload")
        self._ensure_storage()
        self.purge_expired()
        if not isinstance(content_base64, str) or not content_base64:
            raise _fail("invalid_asset", "content_base64 is required")
        if len(content_base64) > ((MAX_ASSET_BYTES + 2) // 3) * 4 + 8:
            raise _fail("asset_too_large", "encoded image exceeds the service cap")
        try:
            data = base64.b64decode(content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise _fail("invalid_asset", "content_base64 is not strict base64") from exc
        if not data or len(data) > MAX_ASSET_BYTES:
            raise _fail("asset_too_large", "decoded image exceeds the service cap")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as image:
                    detected_format = str(image.format or "")
                    if detected_format not in _MEDIA_BY_FORMAT:
                        raise _fail("unsupported_media_type", "only JPEG, PNG, and WebP are accepted")
                    if bool(getattr(image, "is_animated", False)):
                        raise _fail("unsupported_image", "animated images are not accepted")
                    width, height = image.size
                    if (
                        width < 1
                        or height < 1
                        or width > MAX_IMAGE_DIMENSION
                        or height > MAX_IMAGE_DIMENSION
                        or width * height > MAX_IMAGE_PIXELS
                    ):
                        raise _fail("image_dimensions_exceeded", "decoded image exceeds dimension limits")
                    image.verify()
        except ImageServiceError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise _fail("image_dimensions_exceeded", "decoded image exceeds Pillow safety limits") from exc
        except (OSError, SyntaxError, UnidentifiedImageError) as exc:
            raise _fail("invalid_asset", "image decoding failed") from exc
        expected_media_type, suffix = _MEDIA_BY_FORMAT[detected_format]
        if media_type != expected_media_type:
            raise _fail("media_type_mismatch", "declared media_type does not match decoded image")

        with self._state_lock:
            self.purge_expired()
            stored_assets = sum(
                1
                for path in self.assets_root.glob("asset_*")
                if path.is_dir() and not path.is_symlink()
            )
            if stored_assets >= MAX_ASSETS_PER_TENANT:
                raise _fail(
                    "asset_capacity_reached",
                    "tenant asset capacity is full until retained assets expire",
                )
            asset_id = f"asset_{uuid.uuid4().hex[:24]}"
            directory = self._asset_directory(asset_id)
            directory.mkdir(mode=0o700)
            image_path = directory / f"source{suffix}"
            descriptor = os.open(
                image_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                digest = hashlib.sha256(data).hexdigest()
                _write_private_json(
                    directory / "asset.json",
                    {
                        "schema_version": 1,
                        "asset_id": asset_id,
                        "created_at": _utc_now(),
                        "created_unix": time.time(),
                        "media_type": expected_media_type,
                        "suffix": suffix,
                        "size_bytes": len(data),
                        "sha256": digest,
                        "width": width,
                        "height": height,
                    },
                )
            except Exception:
                shutil.rmtree(directory, ignore_errors=True)
                raise
        return {
            "schema_version": 1,
            "ok": True,
            "tool": "put_image_asset",
            "summary": "stored one bounded private image asset",
            "asset": {
                "asset_id": asset_id,
                "media_type": expected_media_type,
                "size_bytes": len(data),
                "sha256": digest,
                "width": width,
                "height": height,
                "expires_in_seconds": self.retention_seconds,
            },
        }

    def _load_asset(self, asset_id: str) -> tuple[dict[str, Any], Path]:
        directory = self._asset_directory(asset_id)
        if directory.is_symlink() or not directory.is_dir():
            raise _fail("asset_not_found", "asset_id was not found")
        try:
            metadata = _safe_json_read(directory / "asset.json", maximum_bytes=32 * 1024)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ImageServiceError) as exc:
            raise _fail("asset_not_found", "asset metadata is unavailable") from exc
        if not isinstance(metadata, dict) or metadata.get("asset_id") != asset_id:
            raise _fail("asset_not_found", "asset metadata is invalid")
        suffix = metadata.get("suffix")
        if suffix not in {item[1] for item in _MEDIA_BY_FORMAT.values()}:
            raise _fail("asset_not_found", "asset metadata is invalid")
        image_path = directory / f"source{suffix}"
        if image_path.is_symlink() or not image_path.is_file():
            raise _fail("asset_not_found", "asset image is unavailable")
        image_size = image_path.stat().st_size
        if image_size < 1 or image_size > MAX_ASSET_BYTES:
            raise _fail("asset_changed", "stored asset size changed")
        data = image_path.read_bytes()
        if (
            len(data) != metadata.get("size_bytes")
            or hashlib.sha256(data).hexdigest() != metadata.get("sha256")
        ):
            raise _fail("asset_changed", "stored asset identity changed")
        os.utime(directory, None)
        return metadata, image_path

    def submit_job(
        self,
        *,
        asset_id: str,
        fixed_classes: list[str],
        task: str = "object_detection",
        execute: bool = False,
        timeout_seconds: int = 300,
    ) -> dict[str, Any]:
        self._admit("submit")
        self._ensure_storage()
        self.purge_expired()
        if task not in {"object_detection", "instance_segmentation"}:
            raise _fail("unsupported_task", "task is not supported by the image service")
        if not isinstance(execute, bool):
            raise _fail("invalid_execution", "execute must be a boolean")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 30 <= timeout_seconds <= 3600
        ):
            raise _fail("invalid_timeout", "timeout_seconds must be in 30..3600")
        labels = _normalized_labels(fixed_classes)
        _metadata, image_path = self._load_asset(asset_id)
        input_relative = str(image_path.relative_to(self.workspace))
        output_token = f"run_{uuid.uuid4().hex[:24]}"
        output_path = self._job_output_directory(output_token)
        output_relative = str(output_path.relative_to(self.workspace))
        job_spec = {
            "schema_version": 1,
            "task": task,
            "prompt_mode": "fixed_classes",
            "fixed_classes": labels,
            "input_mode": "single_image",
            "execution_mode": "batch",
            "batch_size": 1,
            "concurrency": 1,
            "max_images": 1,
            "max_results_per_image": 1000,
            "job_timeout_seconds": timeout_seconds,
            "ranking_policy": "latency_first",
            "allowed_maturities": ["Stable", "Experimental"],
            "network_policy": "deny",
            "compute_policy": "auto",
        }

        cancel = threading.Event()
        deadline = time.monotonic() + timeout_seconds
        request = {
            "job_spec": job_spec, "workspace": str(self.workspace),
            "input": input_relative, "output": output_relative,
            "output_token": output_token, "execute": execute,
        }

        def run() -> dict[str, Any]:
            return run_image_job(request, cancel, deadline)

        manager = self._job_manager()
        with self._state_lock:
            self._load_asset(asset_id)
            jobs = manager.list()
            active_count = sum(
                item["status"] in {"queued", "running"} for item in jobs
            )
            if active_count >= MAX_ACTIVE_JOBS:
                raise _fail("job_capacity_reached", "tenant active-job capacity is full")
            if len(jobs) >= MAX_JOB_RECORDS:
                raise _fail(
                    "job_record_capacity_reached",
                    "tenant job-record capacity is full until retained records expire",
                )
            self._active_assets[asset_id] = self._active_assets.get(asset_id, 0) + 1
            try:
                job_id = manager.submit("image_pipeline", run, on_done=self._release_job, cancel_event=cancel, deadline=deadline)
                self._job_assets[job_id] = (asset_id, output_token)
            except Exception:
                count = self._active_assets.get(asset_id, 0)
                if count <= 1:
                    self._active_assets.pop(asset_id, None)
                else:
                    self._active_assets[asset_id] = count - 1
                raise
        return {
            "schema_version": 1,
            "ok": True,
            "tool": "submit_image_job",
            "summary": "queued one bounded qualified image job",
            "job": {
                "job_id": job_id,
                "status": "queued",
                "asset_id": asset_id,
                "task": task,
                "execute": execute,
            },
        }

    @staticmethod
    def _public_decision(decision: Any) -> dict[str, Any] | None:
        if not isinstance(decision, Mapping):
            return None
        result: dict[str, Any] = {"status": decision.get("status")}
        for key in (
            "decision_id",
            "reason_codes",
            "selected_bundle",
            "environment_fingerprint",
            "qualification_workload_fingerprint",
            "local_input_digest",
        ):
            if key in decision:
                result[key] = decision[key]
        return result

    def _public_result(self, result: Any) -> dict[str, Any] | None:
        if not isinstance(result, Mapping):
            return None
        public: dict[str, Any] = {
            "ok": bool(result.get("ok")),
            "outcome": result.get("outcome"),
            "executed": bool(result.get("executed", False)),
        }
        decision = self._public_decision(result.get("decision"))
        if decision is not None:
            public["decision"] = decision
        error = result.get("error")
        if isinstance(error, Mapping):
            public["error"] = {
                "code": error.get("code"),
                "message": error.get("message"),
            }
        output_token = result.get("_output_token")
        if result.get("outcome") == "completed" and isinstance(output_token, str):
            output = self._job_output_directory(output_token)
            try:
                if output.is_symlink():
                    raise _fail("stored_result_invalid", "managed output is a symlink")
                public["predictions"] = _safe_json_read(
                    output / "predictions.json",
                    maximum_bytes=16 * 1024 * 1024,
                )
                public["provenance"] = _safe_json_read(
                    output / "provenance.json",
                    maximum_bytes=2 * 1024 * 1024,
                )
                public["checksums"] = _safe_json_read(
                    output / "checksums.json",
                    maximum_bytes=2 * 1024 * 1024,
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ImageServiceError):
                public = {
                    "ok": False,
                    "outcome": "result_unavailable",
                    "executed": True,
                    "error": {
                        "code": "stored_result_invalid",
                        "message": "managed output could not be read safely",
                    },
                }
        return public

    def get_job(self, *, job_id: str) -> dict[str, Any]:
        self._admit("get")
        self.purge_expired()
        if _JOB_RE.fullmatch(job_id) is None:
            raise _fail("invalid_job_id", "job_id is invalid")
        if not self.jobs_root.is_dir():
            raise _fail("job_not_found", "job_id was not found")
        with self._state_lock:
            status = self._job_manager().status(job_id)
            if status is None:
                raise _fail("job_not_found", "job_id was not found")
            public_result = self._public_result(status.get("result"))
        return {
            "schema_version": 1,
            "ok": True,
            "tool": "get_image_job",
            "summary": "returned one image job",
            "job": {
                "job_id": job_id,
                "status": status["status"],
                "created_at": status["created_at"],
                "started_at": status["started_at"],
                "finished_at": status["finished_at"],
                "result": public_result,
            },
        }

    def cancel_job(self, *, job_id: str) -> dict[str, Any]:
        self._admit("cancel")
        self.purge_expired()
        if _JOB_RE.fullmatch(job_id) is None:
            raise _fail("invalid_job_id", "job_id is invalid")
        if not self.jobs_root.is_dir():
            raise _fail("job_not_found", "job_id was not found")
        with self._state_lock:
            result = self._job_manager().cancel(job_id)
            if result and result.get("cancelled"):
                self._release_job(job_id)
        if result is None:
            raise _fail("job_not_found", "job_id was not found")
        return {
            "schema_version": 1,
            "ok": True,
            "tool": "cancel_image_job",
            "summary": "processed one bounded cancellation request",
            "job": result,
        }


_SERVICE_CONFIGURATION: dict[str, Any] = {
    "workspace": None,
    "tenant_id": "local",
    "retention_seconds": DEFAULT_RETENTION_SECONDS,
    "service_root": "runs/mcp_image_service",
}
_CONFIGURATION_LOCK = threading.RLock()
_SERVICE_INSTANCE: ImageService | None = None


def configure_image_service(
    *,
    workspace: str | Path | None = None,
    tenant_id: str = "local",
    retention_seconds: int = DEFAULT_RETENTION_SECONDS,
    service_root: str | Path = "runs/mcp_image_service",
) -> None:
    with _CONFIGURATION_LOCK:
        close_image_service()
        _SERVICE_CONFIGURATION.update(
            {
                "workspace": workspace,
                "tenant_id": tenant_id,
                "retention_seconds": retention_seconds,
                "service_root": service_root,
            }
        )


def close_image_service() -> None:
    global _SERVICE_INSTANCE
    with _CONFIGURATION_LOCK:
        if _SERVICE_INSTANCE is not None:
            _SERVICE_INSTANCE.close()
            _SERVICE_INSTANCE = None


def _configured_service() -> ImageService:
    global _SERVICE_INSTANCE
    with _CONFIGURATION_LOCK:
        if _SERVICE_INSTANCE is None:
            workspace = _SERVICE_CONFIGURATION["workspace"]
            if workspace is None:
                workspace = resolved_workspace_root()
            service = ImageService(
                workspace=workspace,
                tenant_id=_SERVICE_CONFIGURATION["tenant_id"],
                retention_seconds=_SERVICE_CONFIGURATION["retention_seconds"],
                service_root=_SERVICE_CONFIGURATION["service_root"],
            )
            _SERVICE_INSTANCE = service
        return _SERVICE_INSTANCE


def start_image_service() -> ImageService:
    service = _configured_service()
    service.start_maintenance()
    return service


def image_service_capabilities() -> dict[str, Any]:
    return _configured_service().capabilities()


def put_image_asset(*, content_base64: str, media_type: str) -> dict[str, Any]:
    return start_image_service().put_asset(
        content_base64=content_base64,
        media_type=media_type,
    )


def submit_image_job(
    *,
    asset_id: str,
    fixed_classes: list[str],
    task: str = "object_detection",
    execute: bool = False,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    return start_image_service().submit_job(
        asset_id=asset_id,
        fixed_classes=fixed_classes,
        task=task,
        execute=execute,
        timeout_seconds=timeout_seconds,
    )


def get_image_job(*, job_id: str) -> dict[str, Any]:
    return start_image_service().get_job(job_id=job_id)


def cancel_image_job(*, job_id: str) -> dict[str, Any]:
    return start_image_service().cancel_job(job_id=job_id)


def public_service_call(fn: Any, /, **kwargs: Any) -> dict[str, Any]:
    """Convert stable service rejections into MCP-safe structured responses."""

    try:
        return fn(**kwargs)
    except ImageServiceError as exc:
        return {
            "schema_version": 1,
            "ok": False,
            "tool": getattr(fn, "__name__", "image_service"),
            "summary": "image service request rejected",
            "error": {
                "code": exc.code,
                "message": exc.public_message,
            },
        }
