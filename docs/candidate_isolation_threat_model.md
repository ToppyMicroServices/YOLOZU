# Candidate isolation threat model and backend decision

## Decision

The v1 decision is `none_supported`. YOLOZU does not currently have an OS/backend
combination that may execute third-party candidate code. There is no fallback to a
host subprocess, an in-process import, a mocked socket policy, or an application-only
network promise.

`python3 tools/probe_candidate_isolation.py` prints a bounded JSON observation. It
checks only platform facts and the presence of fixed backend executables. It does not
start a backend or execute candidate code. Executable presence cannot change the
code-owned `none_supported` decision. Every mandatory enforcement probe therefore
stays `not_run`, and the capability is `unsupported`.

Exit status zero means only that the observation was produced. Consumers must gate
on `capability_status`; zero is never a candidate-execution approval.

This decision covers candidate acquisition, extraction, dependency build/install,
and inference. It does not change the audited host-runner path for code owned by this
repository.

## Scope and trust boundaries

The candidate is untrusted. Assume its source, archives, dependencies, install
scripts, native extensions, model assets, and runtime code can be buggy or hostile.
It may try to read mounted files, discover credentials, use DNS or loopback, open
network connections, spawn descendants, exhaust CPU/memory/disk/PIDs, access
devices, escape through syscalls, forge output, or retain data in backend logs.

The operator, reviewed YOLOZU code, a selected kernel or hypervisor boundary, and a
reviewed backend configuration would be trusted. No such complete backend
configuration is selected today. A host administrator remains outside the promised
threat boundary.

Protected assets are the workspace, home directory, credentials and registry
configuration, input data, verified artifacts, host network, other processes and
devices, host availability, and the integrity and bounded size of published output.

A future implementation must keep four phases separate:

1. Credential-free acquisition admits only exact code-owned HTTPS URLs and immutable
   revisions, verifies outer sizes and SHA-256 values before parsing, and stores only
   verified cache entries.
2. Extraction accepts only reviewed media types and rejects absolute paths, `..`,
   excessive depth, symlinks, hardlinks, devices, FIFOs, nested archives, excess
   entries, and compressed or decompressed byte overruns.
3. Build/install runs from that cache in a disposable, secret-free sandbox with a
   reviewed exact name/version/hash dependency lock. Network remains off. The result,
   base image, backend, lock, dependency inventory/SBOM, and built output are bound by
   digest.
4. Inference uses a fresh separate network-off sandbox. Only verified built outputs,
   pinned inputs, and pinned assets are mounted read-only. The host accepts only typed,
   bounded output.

Crossing any boundary without all matching live probes returns
`isolation_unsupported` before host installation, import, asset parsing, or candidate
execution.

## Backend and OS matrix

“Not supported” means that documentation describes useful primitives, but YOLOZU has
not selected and live-verified the complete configuration. “Rejected” means that the
boundary cannot count as isolation evidence for this policy.

| OS and version | Backend and version | Boundary | Decision | Current reason |
|---|---|---|---|---|
| Linux, no selected release | `linux_rootless_podman`; no selected version | Linux kernel; a VM may also apply | Not supported | No reviewed image, backend version, complete configuration, or real control probes |
| macOS 26.6.1 (25G76), arm64 | `macos_podman_machine`; absent on the observed host | Linux VM | Not supported | No local backend or image; no real control probes |
| macOS, no supported-version range | `macos_apple_virtualization`; no adapter/version | Hypervisor | Not supported | Framework primitives do not provide a YOLOZU backend or complete policy probes |
| macOS 26.6.1 (25G76), arm64 | `macos_sandbox_exec`; `/usr/bin/sandbox-exec` version unverified | Platform-native sandbox | Rejected | Presence does not prove every required network, mount, PID, memory, disk, device, log, and cleanup control |
| Any | `in_process_python`; hooks or ordinary subprocess | None | Rejected | No OS or hypervisor boundary |

The host observation was collected at `2026-08-28T22:08:12+09:00` in Asia/Tokyo.
`podman`, `docker`, `nerdctl`, `limactl`, and `qemu-system-aarch64` were
absent. `/usr/bin/sandbox-exec` was present. These are activity and presence facts,
not containment evidence.

Podman documents that macOS uses a Linux virtual machine and documents individual
run controls such as no network, read-only filesystems, memory limits, PID limits,
disabled logging, and `pull=never`. Those documented flags are not evidence that
one exact host/backend/image configuration enforces the complete YOLOZU policy.
Apple documents virtualization and hypervisor primitives, but YOLOZU has no adapter
or accepted probe suite for them.

Primary documentation was collected on 2026-08-28:

- [Podman machine](https://docs.podman.io/en/stable/markdown/podman-machine.1.html)
- [Podman run](https://docs.podman.io/en/latest/markdown/podman-run.1.html)
- [Apple Virtualization](https://developer.apple.com/documentation/virtualization)
- [Apple Hypervisor](https://developer.apple.com/documentation/hypervisor)
- [Apple vmnet](https://developer.apple.com/documentation/vmnet)

## Admission limits and empty allowlists

The code-owned policy has no approved artifact URL, revision, archive media type,
dependency lock, base image, or manual image-import procedure. The empty allowlists
deny acquisition rather than treating an unknown as approved.

A future reviewed row may choose smaller limits, but it may not exceed these
admission ceilings:

| Limit | Maximum |
|---|---:|
| One acquired artifact | 17,179,869,184 bytes |
| All acquired artifacts | 68,719,476,736 bytes |
| Archive entries | 8,192 |
| Relative path depth | 16 components |
| Total decompressed bytes | 68,719,476,736 bytes |

These are policy ceilings, not current enforcement claims. The current result remains
unsupported because there is no implementation or backend row that proves them.

## Required live probes and failures

Each probe below is mandatory for a future supported row. A mocked result, unit-test
fixture, in-process guard, executable-presence observation, or documentation claim
cannot mark one as passed. With the current decision, all return `not_run` in the
machine-readable observation and retain the listed failure code.

| Phase | Control | Live probe ID | Failure code |
|---|---|---|---|
| Acquisition | Exact source/revision allowlist | `probe_acquisition_source` | `isolation_acquisition_source_unsupported` |
| Acquisition | Per-artifact and total outer size | `probe_acquisition_size` | `isolation_acquisition_size_unsupported` |
| Acquisition | Outer SHA-256 before parsing | `probe_acquisition_hash` | `isolation_acquisition_hash_unsupported` |
| Extraction | Archive/media allowlist | `probe_archive_type` | `isolation_archive_type_unsupported` |
| Extraction | Absolute path, traversal, link, device, FIFO, and nested-archive rejection | `probe_archive_entries` | `isolation_archive_entries_unsupported` |
| Extraction | Entry, depth, and decompressed-byte ceilings | `probe_archive_limits` | `isolation_archive_limits_unsupported` |
| Build | Exact reviewed dependency lock | `probe_dependency_lock` | `isolation_dependency_lock_unsupported` |
| Build | Disposable sandbox without host secrets | `probe_build_secret_mounts` | `isolation_build_secrets_unsupported` |
| Build | Network disabled after acquisition | `probe_build_network` | `isolation_build_network_unsupported` |
| Build | CPU, time, memory, PID, and disk ceilings | `probe_build_resources` | `isolation_build_resources_unsupported` |
| Build | Image/backend/lock/SBOM/output digest binding | `probe_build_provenance` | `isolation_build_provenance_unsupported` |
| Inference | Local image digest readback | `probe_image_digest` | `isolation_image_missing` |
| Inference | No pull or implicit image fetch | `probe_image_pull_policy` | `isolation_image_pull_unsupported` |
| Inference | Fresh empty backend configuration | `probe_backend_config` | `isolation_backend_config_unsupported` |
| Inference | Disabled or ephemeral backend logs | `probe_backend_logging` | `isolation_logging_unsupported` |
| Inference | External network egress denial | `probe_network_egress` | `isolation_network_unsupported` |
| Inference | DNS denial | `probe_dns` | `isolation_dns_unsupported` |
| Inference | Loopback denial | `probe_loopback` | `isolation_loopback_unsupported` |
| Inference | Only pinned read-only mounts | `probe_mounts` | `isolation_mount_unsupported` |
| Inference | Workspace, home, credentials, and secrets absent | `probe_secret_mounts` | `isolation_secrets_unsupported` |
| Inference | UID, privilege, and capability policy | `probe_privileges` | `isolation_privilege_unsupported` |
| Inference | Descendant process/PID ceiling | `probe_pid_limit` | `isolation_pid_limit_unsupported` |
| Inference | CPU and outer wall-time ceiling | `probe_cpu_time` | `isolation_cpu_time_unsupported` |
| Inference | Memory limit and observed OOM behavior | `probe_memory_oom` | `isolation_memory_unsupported` |
| Inference | Writable disk and temporary-space ceiling | `probe_disk_temp` | `isolation_disk_unsupported` |
| Inference | Output file-count and byte ceilings | `probe_output_limits` | `isolation_output_unsupported` |
| Inference | Syscall and device policy | `probe_syscalls_devices` | `isolation_syscall_device_unsupported` |
| Inference | Termination, descendant reap, and cleanup | `probe_cleanup` | `isolation_cleanup_unsupported` |

A supported result would require all probes to pass on the exact OS version, backend
version, image digest, and configuration under review. One absent, failed, skipped,
or indeterminate control returns `isolation_unsupported`.

## Base image and logging preconditions

No base image is selected. Its official immutable source reference, expected image,
config and layer digests, sizes, license review, SBOM digest, independently reviewed
manual import procedure, and post-import local digest readback are all `unknown` or
unavailable. YOLOZU has no image pull, fetch, or import workflow in this roadmap.

A future backend must require the exact reviewed image to exist locally before it
starts. Every launch must use `pull=never` or an equivalent, an empty configuration,
and no host registry, credential, home, or workspace mount. A missing, mismatched, or
unreviewed image returns `isolation_image_missing` before network or candidate
execution. If backend stdout/stderr can persist in a daemon log, journald, container
log file, or VM console artifact and that retention cannot be disabled and probed,
the result is `isolation_logging_unsupported`.

## Residual risks and change trigger

Even a future passing row would not protect against an adversarial host
administrator, vulnerabilities in the selected kernel/hypervisor/backend, malicious
firmware, denial of service within configured limits, or sensitive content that the
operator intentionally includes in a pinned input.

The decision can change only through reviewed repository code and evidence that
pins one exact OS/backend/image configuration and passes every real mandatory probe.
The two implementation Beads, `YOLOZU-ll2.81.3.6` and
`YOLOZU-ll2.81.3.7`, remain deferred to `2099-12-31`. Dependency closure alone
must not make them ready.
