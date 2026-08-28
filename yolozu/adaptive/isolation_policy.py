"""Fail-closed candidate-isolation policy and host capability observation."""

from __future__ import annotations

import os
import platform
import shutil
from datetime import datetime, timezone
from typing import Callable, Mapping

from .canonical import canonical_sha256_v1

__all__ = [
    "CANDIDATE_ISOLATION_DECISION",
    "candidate_isolation_policy",
    "probe_candidate_isolation",
]


CANDIDATE_ISOLATION_DECISION = "none_supported"
_POLICY_ID = "candidate-isolation-v1"
_DOCUMENTATION_COLLECTED_ON = "2026-08-28"

_BACKEND_ROWS = (
    {
        "backend_id": "linux_rootless_podman",
        "os_family": "Linux",
        "matrix_status": "not_supported",
        "executable": "podman",
        "enforcement_boundary": "kernel_and_linux_vm_when_applicable",
        "reason_code": "isolation_backend_unreviewed",
    },
    {
        "backend_id": "macos_podman_machine",
        "os_family": "Darwin",
        "matrix_status": "not_supported",
        "executable": "podman",
        "enforcement_boundary": "linux_vm",
        "reason_code": "isolation_backend_unreviewed",
    },
    {
        "backend_id": "macos_apple_virtualization",
        "os_family": "Darwin",
        "matrix_status": "not_supported",
        "executable": None,
        "enforcement_boundary": "hypervisor",
        "reason_code": "isolation_backend_unimplemented",
    },
    {
        "backend_id": "macos_sandbox_exec",
        "os_family": "Darwin",
        "matrix_status": "rejected",
        "executable": "/usr/bin/sandbox-exec",
        "enforcement_boundary": "platform_native_sandbox",
        "reason_code": "isolation_controls_unproven",
    },
    {
        "backend_id": "in_process_python",
        "os_family": "Any",
        "matrix_status": "rejected",
        "executable": None,
        "enforcement_boundary": "none",
        "reason_code": "isolation_boundary_absent",
    },
)

_MANDATORY_CONTROLS = (
    ("acquisition", "source_allowlist", "probe_acquisition_source", "isolation_acquisition_source_unsupported"),
    ("acquisition", "outer_size_limits", "probe_acquisition_size", "isolation_acquisition_size_unsupported"),
    ("acquisition", "outer_hash_verification", "probe_acquisition_hash", "isolation_acquisition_hash_unsupported"),
    ("extraction", "archive_media_allowlist", "probe_archive_type", "isolation_archive_type_unsupported"),
    ("extraction", "traversal_safe_extraction", "probe_archive_entries", "isolation_archive_entries_unsupported"),
    ("extraction", "entry_depth_and_size_limits", "probe_archive_limits", "isolation_archive_limits_unsupported"),
    ("build", "reviewed_dependency_lock", "probe_dependency_lock", "isolation_dependency_lock_unsupported"),
    ("build", "secret_free_disposable_sandbox", "probe_build_secret_mounts", "isolation_build_secrets_unsupported"),
    ("build", "network_off_after_acquisition", "probe_build_network", "isolation_build_network_unsupported"),
    ("build", "resource_limits", "probe_build_resources", "isolation_build_resources_unsupported"),
    ("build", "digest_and_sbom_provenance", "probe_build_provenance", "isolation_build_provenance_unsupported"),
    ("inference", "image_digest_readback", "probe_image_digest", "isolation_image_missing"),
    ("inference", "pull_never", "probe_image_pull_policy", "isolation_image_pull_unsupported"),
    ("inference", "fresh_empty_backend_config", "probe_backend_config", "isolation_backend_config_unsupported"),
    ("inference", "backend_logging_disabled", "probe_backend_logging", "isolation_logging_unsupported"),
    ("inference", "network_egress_denied", "probe_network_egress", "isolation_network_unsupported"),
    ("inference", "dns_denied", "probe_dns", "isolation_dns_unsupported"),
    ("inference", "loopback_denied", "probe_loopback", "isolation_loopback_unsupported"),
    ("inference", "read_only_pinned_mounts", "probe_mounts", "isolation_mount_unsupported"),
    ("inference", "workspace_home_secrets_absent", "probe_secret_mounts", "isolation_secrets_unsupported"),
    ("inference", "uid_privileges_capabilities", "probe_privileges", "isolation_privilege_unsupported"),
    ("inference", "subprocess_pid_limit", "probe_pid_limit", "isolation_pid_limit_unsupported"),
    ("inference", "cpu_and_wall_time_limits", "probe_cpu_time", "isolation_cpu_time_unsupported"),
    ("inference", "memory_and_oom_limit", "probe_memory_oom", "isolation_memory_unsupported"),
    ("inference", "disk_and_temp_limit", "probe_disk_temp", "isolation_disk_unsupported"),
    ("inference", "output_count_and_bytes", "probe_output_limits", "isolation_output_unsupported"),
    ("inference", "syscall_and_device_policy", "probe_syscalls_devices", "isolation_syscall_device_unsupported"),
    ("inference", "termination_and_cleanup", "probe_cleanup", "isolation_cleanup_unsupported"),
)


def candidate_isolation_policy() -> dict[str, object]:
    """Return the canonical code-owned isolation decision record."""

    record: dict[str, object] = {
        "schema_version": 1,
        "policy_id": _POLICY_ID,
        "decision": CANDIDATE_ISOLATION_DECISION,
        "documentation_collected_on": _DOCUMENTATION_COLLECTED_ON,
        "execution_fallback": "forbidden",
        "backend_rows": [dict(row) for row in _BACKEND_ROWS],
        "mandatory_controls": [
            {
                "phase": phase,
                "control_id": control_id,
                "probe_id": probe_id,
                "failure_code": failure_code,
            }
            for phase, control_id, probe_id, failure_code in _MANDATORY_CONTROLS
        ],
        "acquisition": {
            "approved_https_artifacts": [],
            "allowed_archive_media_types": [],
            "limits": {
                "maximum_artifact_bytes": 17_179_869_184,
                "maximum_total_artifact_bytes": 68_719_476_736,
                "maximum_archive_entries": 8_192,
                "maximum_path_depth": 16,
                "maximum_decompressed_bytes": 68_719_476_736,
            },
        },
        "dependency_lock": {
            "status": "unselected",
            "lock_digest": "unknown",
            "sbom_digest": "unknown",
        },
        "base_image": {
            "status": "unselected",
            "source_reference": "unknown",
            "image_digest": "unknown",
            "config_digest": "unknown",
            "layer_digests": "unknown",
            "compressed_size_bytes": "unknown",
            "license_review": "unknown",
            "sbom_digest": "unknown",
            "manual_import_procedure": "unavailable",
            "post_import_readback": "unavailable",
            "failure_code": "isolation_image_unreviewed",
        },
    }
    record["policy_digest"] = canonical_sha256_v1(
        record, own_digest_field="policy_digest"
    )
    return record


def _bounded_platform_value(value: str) -> str:
    cleaned = "".join(character for character in value if character.isprintable()).strip()
    return cleaned[:128] or "unknown"


def _default_executable_probe(executable: str) -> bool:
    if os.path.isabs(executable):
        return os.path.isfile(executable) and os.access(executable, os.X_OK)
    return shutil.which(executable) is not None


def probe_candidate_isolation(
    *,
    platform_values: Mapping[str, str] | None = None,
    executable_probe: Callable[[str], bool] | None = None,
    collected_at: datetime | None = None,
) -> dict[str, object]:
    """Observe backend presence without executing candidate or backend code.

    Executable presence is diagnostic only. The code-owned decision has no
    supported matrix row, so this function cannot return a supported capability.
    """

    supplied = platform_values or {}
    os_family = _bounded_platform_value(supplied.get("os", platform.system()))
    release = _bounded_platform_value(supplied.get("release", platform.release()))
    architecture = _bounded_platform_value(
        supplied.get("architecture", platform.machine())
    )
    is_executable = executable_probe or _default_executable_probe

    observations: list[dict[str, object]] = []
    for row in _BACKEND_ROWS:
        applicable = row["os_family"] in {"Any", os_family}
        executable = row["executable"]
        if not applicable:
            executable_status = "not_applicable"
        elif executable is None:
            executable_status = "not_probed"
        else:
            executable_status = "present" if is_executable(executable) else "absent"
        observations.append(
            {
                "backend_id": row["backend_id"],
                "matrix_status": row["matrix_status"],
                "applicable": applicable,
                "executable_status": executable_status,
                "backend_version": "unknown",
                "reason_code": row["reason_code"],
            }
        )

    instant = collected_at or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise ValueError("collected_at must be timezone-aware")
    instant = instant.astimezone(timezone.utc).replace(microsecond=0)
    policy = candidate_isolation_policy()
    return {
        "schema_version": 1,
        "policy_id": policy["policy_id"],
        "policy_digest": policy["policy_digest"],
        "decision": CANDIDATE_ISOLATION_DECISION,
        "capability_status": "unsupported",
        "enforcement_boundary": "none",
        "reason_codes": ["isolation_no_supported_backend"],
        "collected_at": instant.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": {
            "os": os_family,
            "release": release,
            "architecture": architecture,
        },
        "backend_observations": observations,
        "control_results": [
            {
                "phase": phase,
                "control_id": control_id,
                "probe_id": probe_id,
                "status": "not_run",
                "reason_code": failure_code,
            }
            for phase, control_id, probe_id, failure_code in _MANDATORY_CONTROLS
        ],
        "candidate_execution_attempted": False,
    }
