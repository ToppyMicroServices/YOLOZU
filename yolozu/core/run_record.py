"""Reproducibility-oriented run-record builder.

Captures host info, accelerator details, git state, dependency
versions, and runtime config into a structured JSON record.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
import logging

__all__ = [
    "host_info",
    "accelerator_info",
    "git_info",
    "versions",
    "dependency_lock_info",
    "preprocess_config",
    "runtime_info",
    "validate_run_record_contract",
    "build_run_record",
]
import hashlib
import shlex
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    importlib_metadata = None


def _safe_version(module_name: str) -> str | None:
    try:
        mod = __import__(module_name)
    except Exception as exc:
        logger.debug("module import for version probe failed: %s: %s", module_name, exc)
        return None
    version = getattr(mod, "__version__", None)
    if version is None:
        return None
    # Some libraries (notably PyTorch) use a custom string subclass for __version__.
    # Cast to a plain str so it remains safe/portable when saved in torch checkpoints.
    return str(version)


def host_info() -> dict[str, Any]:
    """Best-effort host metadata (never raises)."""

    try:
        hostname = socket.gethostname()
    except OSError:
        hostname = None
    try:
        fqdn = socket.getfqdn()
    except OSError:
        fqdn = None
    try:
        node = platform.node()
    except OSError:
        node = None

    return {
        "hostname": hostname,
        "fqdn": fqdn,
        "node": node,
        "pid": int(os.getpid()),
    }


def accelerator_info() -> dict[str, Any]:
    """Best-effort accelerator metadata (cuda/mps), safe without torch."""

    try:
        import torch  # type: ignore
    except Exception as exc:
        logger.debug("torch accelerator probe import failed: %s", exc)
        return {"torch_available": False, "cuda": {"available": False}, "mps": {"available": False}}

    cuda_available = False
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        logger.debug("torch.cuda.is_available probe failed: %s", exc)
        cuda_available = False

    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    cudnn_version = None
    try:
        cudnn_version = int(torch.backends.cudnn.version()) if hasattr(torch.backends, "cudnn") else None
    except Exception as exc:
        logger.debug("torch cudnn version probe failed: %s", exc)
        cudnn_version = None

    device_count = 0
    try:
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
    except Exception as exc:
        logger.debug("torch device_count probe failed: %s", exc)
        device_count = 0

    devices: list[dict[str, Any]] = []
    if cuda_available and device_count > 0:
        for idx in range(device_count):
            name = None
            total_mem = None
            capability = None
            try:
                name = str(torch.cuda.get_device_name(idx))
            except Exception as exc:
                logger.debug("torch device name probe failed for index %s: %s", idx, exc)
                name = None
            try:
                props = torch.cuda.get_device_properties(idx)
                total_mem = int(getattr(props, "total_memory", 0) or 0)
                major = getattr(props, "major", None)
                minor = getattr(props, "minor", None)
                if major is not None and minor is not None:
                    capability = [int(major), int(minor)]
            except Exception as exc:
                logger.debug("torch device properties probe failed for index %s: %s", idx, exc)
                total_mem = None
                capability = None
            devices.append(
                {
                    "index": int(idx),
                    "name": name,
                    "total_memory_bytes": total_mem,
                    "capability": capability,
                }
            )

    mps_available = False
    try:
        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception as exc:
        logger.debug("torch MPS probe failed: %s", exc)
        mps_available = False

    return {
        "torch_available": True,
        "cuda": {
            "available": bool(cuda_available),
            "version": str(cuda_version) if cuda_version is not None else None,
            "cudnn": cudnn_version,
            "device_count": int(device_count),
            "devices": devices,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "mps": {"available": bool(mps_available)},
    }


def git_info(repo_root: str | Path) -> dict[str, Any]:
    """Return best-effort git metadata for repo_root.

    Never raises; returns empty dict if git is unavailable or repo_root is not a git repo.
    """

    root = Path(repo_root)
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.call(
            ["git", "-C", str(root), "diff", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # diff --quiet returns 1 when there are changes
        is_dirty = bool(dirty != 0)
        return {"sha": sha, "dirty": is_dirty}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _safe_version("numpy"),
        "torch": _safe_version("torch"),
        "Pillow": _safe_version("PIL"),
        "PyYAML": _safe_version("yaml"),
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _collect_installed_packages() -> list[dict[str, str]]:
    packages: list[dict[str, str]] = []
    if importlib_metadata is None:
        return packages
    try:
        for dist in importlib_metadata.distributions():
            name = str(dist.metadata.get("Name") or dist.metadata.get("Summary") or "").strip()
            version = str(getattr(dist, "version", "") or "").strip()
            if not name:
                continue
            packages.append({"name": name, "version": version})
    except Exception as exc:
        logger.debug("installed packages probe failed: %s", exc)
        return []
    packages.sort(key=lambda item: item["name"].lower())
    return packages


def dependency_lock_info(repo_root: str | Path) -> dict[str, Any]:
    packages = _collect_installed_packages()
    package_lines = [f"{pkg['name']}=={pkg['version']}" for pkg in packages]
    package_blob = "\n".join(package_lines)

    requirements_hashes: dict[str, str] = {}
    root = Path(repo_root)
    for rel in ("requirements.txt", "requirements-dev.txt", "requirements-test.txt"):
        p = root / rel
        if not p.exists() or not p.is_file():
            continue
        try:
            requirements_hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            continue

    return {
        "python_version": platform.python_version(),
        "package_count": int(len(packages)),
        "package_set_sha256": _sha256_text(package_blob),
        "requirements_files_sha256": requirements_hashes,
    }


def preprocess_config(args: dict[str, Any] | None) -> dict[str, Any]:
    args = args or {}
    image_size = args.get("image_size", args.get("imgsz"))
    try:
        image_size = int(image_size) if image_size is not None else None
    except (TypeError, ValueError):
        image_size = None

    scale_min = args.get("scale_min")
    scale_max = args.get("scale_max")
    try:
        scale_min = float(scale_min) if scale_min is not None else None
    except (TypeError, ValueError):
        scale_min = None
    try:
        scale_max = float(scale_max) if scale_max is not None else None
    except (TypeError, ValueError):
        scale_max = None

    return {
        "image_size": image_size,
        "multiscale": bool(args.get("multiscale", False)),
        "scale_min": scale_min,
        "scale_max": scale_max,
    }


def _command_info(argv: list[str]) -> dict[str, Any]:
    command = [str(sys.executable), *[str(x) for x in argv]]
    return {
        "argv": list(argv),
        "command": command,
        "command_str": shlex.join(command),
        "python_executable": str(sys.executable),
        "cwd": str(Path.cwd()),
    }


def runtime_info() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "python_executable": str(sys.executable),
    }


def validate_run_record_contract(record: Any, *, require_git_sha: bool = True) -> None:
    if not isinstance(record, dict):
        raise ValueError("run_meta must be an object")

    schema_version = record.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version < 1:
        raise ValueError("run_meta.schema_version must be a positive integer")

    command = record.get("command")
    if not isinstance(command, dict):
        raise ValueError("run_meta.command must be an object")
    argv = command.get("argv")
    if not isinstance(argv, list):
        raise ValueError("run_meta.command.argv must be a list")
    if not isinstance(command.get("command_str"), str) or not str(command.get("command_str")).strip():
        raise ValueError("run_meta.command.command_str must be a non-empty string")

    runtime = record.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("run_meta.runtime must be an object")
    for key in ("python_version", "platform", "python_executable"):
        value = runtime.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"run_meta.runtime.{key} must be a non-empty string")

    hardware = record.get("hardware")
    if not isinstance(hardware, dict):
        raise ValueError("run_meta.hardware must be an object")
    if not isinstance(hardware.get("host"), dict):
        raise ValueError("run_meta.hardware.host must be an object")
    if not isinstance(hardware.get("accelerator"), dict):
        raise ValueError("run_meta.hardware.accelerator must be an object")

    preprocess = record.get("preprocess")
    if not isinstance(preprocess, dict):
        raise ValueError("run_meta.preprocess must be an object")
    image_size = preprocess.get("image_size")
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
        raise ValueError("run_meta.preprocess.image_size must be a positive integer")

    dependency_lock = record.get("dependency_lock")
    if not isinstance(dependency_lock, dict):
        raise ValueError("run_meta.dependency_lock must be an object")
    package_hash = dependency_lock.get("package_set_sha256")
    if not isinstance(package_hash, str) or len(package_hash) != 64:
        raise ValueError("run_meta.dependency_lock.package_set_sha256 must be a sha256 hex string")

    git = record.get("git")
    if not isinstance(git, dict):
        raise ValueError("run_meta.git must be an object")
    sha = git.get("sha")
    if require_git_sha:
        if not isinstance(sha, str) or not sha.strip():
            raise ValueError("run_meta.git.sha must be a non-empty string")


def build_run_record(
    *,
    repo_root: str | Path,
    argv: list[str] | None = None,
    args: dict[str, Any] | None = None,
    dataset_root: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a reproducibility record for logs/checkpoints.

    This is intentionally best-effort (never raises) so it can run in CI and non-git envs.
    """

    record: dict[str, Any] = {
        "host": host_info(),
        "versions": versions(),
        "accelerator": accelerator_info(),
        "git": git_info(repo_root),
    }

    if argv is None:
        argv = sys.argv[1:]
    record["argv"] = list(argv)

    pp = preprocess_config(args)
    dep = dependency_lock_info(repo_root)
    cmd = _command_info(argv)
    run_rt = runtime_info()

    record.update(
        {
            "schema_version": 1,
            "command": cmd,
            "runtime": run_rt,
            "hardware": {"host": record.get("host"), "accelerator": record.get("accelerator")},
            "preprocess": pp,
            "dependency_lock": dep,
        }
    )

    if args is not None:
        record["args"] = dict(args)

    if dataset_root is not None:
        record["dataset_root"] = str(dataset_root)

    if extra:
        record["extra"] = dict(extra)

    # Fallbacks for detached/no-git environments where git CLI is unavailable.
    git_obj = record.get("git") if isinstance(record.get("git"), dict) else {}
    if not isinstance(git_obj.get("sha"), str) or not str(git_obj.get("sha", "")).strip():
        for env_key in ("GITHUB_SHA", "GIT_COMMIT", "CI_COMMIT_SHA"):
            env_sha = os.environ.get(env_key)
            if isinstance(env_sha, str) and env_sha.strip():
                git_obj["sha"] = env_sha.strip()
                break
    record["git"] = git_obj

    return record
