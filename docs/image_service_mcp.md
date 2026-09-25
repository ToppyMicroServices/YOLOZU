# Bounded MCP image service for OpenAI and Claude

YOLOZU provides a narrow MCP surface for clients that need local CNN image
processing. The AI client interprets natural language. YOLOZU accepts only a
typed image asset and job request, then selects a qualified registered pipeline
or abstains.

The service surface contains only these tools:

- `image_service_capabilities`
- `put_image_asset`
- `submit_image_job`
- `get_image_job`
- `cancel_image_job`

A [local OpenAI plugin](openai_image_service_plugin.md) packages this same
surface and its image-request workflow. Plugin installation does not qualify
or activate a model.

It does not accept a backend name, model path, local input path, remote URL,
shell argument, or arbitrary output path.

## Local private service

Install the MCP dependency and start the dedicated stdio surface:

```bash
python3 -m pip install 'yolozu[mcp]'
yolozu-mcp --surface image-service
```

The `mcp` extra installs the transport surface. A deployment that will qualify
and execute the Torchvision runner also needs the exact pinned Torch,
Torchvision, and safetensors runtime; the `full` extra includes those packages.
Installing them does not qualify or promote a bundle.

For a local Streamable HTTP endpoint:

```bash
yolozu-mcp \
  --transport streamable-http \
  --surface image-service \
  --host 127.0.0.1 \
  --port 8000 \
  --http-path /mcp
```

The endpoint is `http://127.0.0.1:8000/mcp`. Loopback is the default. DNS
rebinding protection remains enabled.

OpenAI can connect a local or private endpoint through its Secure MCP Tunnel.
See the current [OpenAI MCP documentation](https://developers.openai.com/api/docs/guides/tools-connectors-mcp).

## Public endpoint behind TLS

The built-in server does not terminate TLS. Put it behind a reviewed HTTPS
reverse proxy, load a bearer token from the deployment secret manager, and use
one process and credential for one tenant:

```bash
export YOLOZU_MCP_AUTH_TOKEN='<at least 32 random bytes from the secret manager>'
yolozu-mcp \
  --transport streamable-http \
  --surface image-service \
  --host 0.0.0.0 \
  --port 8000 \
  --http-path /mcp \
  --public-url https://vision.example/mcp \
  --tenant-id customer-a
```

A non-loopback bind fails unless all of these conditions hold:

- the surface is exactly `image-service`;
- `--public-url` is HTTPS;
- the configured token environment variable contains 32--4096 UTF-8 bytes.

The token is never accepted as a command-line value. This is a bounded
single-credential service boundary, not a multi-user OAuth authorization
server. Use an external identity-aware gateway when several users or tenants
share one deployment.

OpenAI remote MCP supports Streamable HTTP. Claude's hosted MCP connector also
requires a public HTTPS Streamable HTTP or SSE server; local stdio must instead
be connected by a local Claude client or SDK helper. Check the current
[Claude MCP connector documentation](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector),
including its Beta and data-retention limits, before sending sensitive data.

### Provider request settings

These are request fragments for an application's existing API client, not
deployment credentials or tested provider connections. Keep the service token in
the backend secret store. The application must handle image upload and job
polling; an image attached to a chat does not automatically become an `asset_id`.

For the OpenAI Responses API, configure the MCP tool with explicit approvals:

```python
yolozu_tool = {
    "type": "mcp",
    "server_label": "yolozu",
    "server_url": service_https_url,
    "authorization": service_token,
    "allowed_tools": [
        "image_service_capabilities", "put_image_asset", "submit_image_job",
        "get_image_job", "cancel_image_job",
    ],
    "require_approval": "always",
}
# Pass tools=[yolozu_tool] to your Responses request.
```

The application handles `mcp_approval_request` before sending its approval
response. See the [official Responses MCP guide](https://developers.openai.com/api/docs/guides/tools-connectors-mcp).

For Claude Messages, use the current connector toolset format. This initial
configuration enables only inspection; enable each write tool after the
application has obtained the user's consent for the intended operation:

```python
yolozu_settings = {
    "betas": ["mcp-client-2025-11-20"],
    "mcp_servers": [{
        "type": "url", "name": "yolozu", "url": service_https_url,
        "authorization_token": service_token,
    }],
    "tools": [{
        "type": "mcp_toolset", "mcp_server_name": "yolozu",
        "default_config": {"enabled": False},
        "configs": {
            "image_service_capabilities": {"enabled": True},
            "get_image_job": {"enabled": True},
        },
    }],
}
# Pass these fields to your client's beta Messages request.
```

The service's `execute=true` is a caller instruction, not proof of human
approval. Access to `submit_image_job` permits execution requests, so the calling
application owns the approval boundary. The [Claude connector guide](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector)
defines the beta header and tool allowlist format. These settings were checked
against official documentation on 2026-09-20; no paid provider request was made.

## Request flow

1. Call `image_service_capabilities` and check the current bounds.
2. Call `put_image_asset` with strict base64 and the exact decoded media type.
3. Call `submit_image_job` with the returned `asset_id`, fixed class names, and
   `execute=false` for selection and preflight.
4. Poll `get_image_job` using the opaque `job_id`.
5. After an application-level approval, submit with `execute=true` when actual
   model execution is intended.

`put_image_asset` accepts one JPEG, PNG, or WebP up to 8 MiB, 16,384 pixels on
either dimension, and 64 million decoded pixels. Animated images, archives,
media-type mismatches, path input, and URL input fail closed. Base64 upload is
the initial bounded MCP path; deployments handling larger payloads should add a
separate authenticated upload plane rather than increasing tool-argument limits.

Each service process uses a private tenant directory with mode `0700`; image and
job-state files are written with mode `0600`. Queued jobs use one supervisor worker;
each image pipeline runs in a separately spawned process. A
background worker checks expiry every 60 seconds while the server is running,
including when no new requests arrive. Startup and job lookup also check expiry.
Inactive assets, terminal job state, and managed outputs expire under the configured
retention policy. A stopped server cannot delete data; expired data is cleaned
on restart. Each tenant is capped at 128 retained assets, 16 active jobs, and 1,024
retained job records. An asset referenced by a queued or running job is not
removed by retention cleanup. Terminal paths release the asset reference, including
exceptions before the worker body starts and cancellation of queued or running work;
cleanup also preserves active output directories. Cleanup is restricted to validated
service-owned directories. Symlink ancestors are rejected, including aliases within
the workspace. Cleanup failure blocks new tool requests until cleanup succeeds or
an operator repairs the storage.
Exact service-owned staging/backup directories and transaction markers left by
forced termination also expire after retention; active job tokens remain protected.

Per-tenant, per-process sliding 60-second limits are: 12 upload attempts, 12 job
submissions, 120 capability requests, 120 status lookups, and 30 cancellation
requests. Invalid attempts also consume the relevant allowance. Rejections return
`error.code=rate_limited`; wait 60 seconds before retrying. Poll status no faster
than once per second. These in-memory counters reset on process restart. They are
not a distributed quota or an HTTP denial-of-service control. The dedicated
HTTP surface bounds each POST body to 16 MiB and its upload to 30 seconds before
MCP JSON parsing, including chunked requests (`413` for excess bytes, `408` for
timeout). The deployment gateway still needs connection-rate and concurrency limits.
An OS advisory lock rejects a second service using the same tenant directory with
`tenant_in_use`. The owner holds the lock until workers and cleanup have stopped.
The lock file is intentionally retained; do not unlink it to bypass ownership.
Use a local filesystem with working POSIX advisory locks.

`cancel_image_job` cancels queued work immediately. For running work it returns
`cancelled=false, reason=cancellation_requested`; poll until the job is terminal.
The supervisor stops and reaps the owned process before releasing its asset.
`timeout_seconds` (30..3600) starts at queue admission and includes selection,
preflight, and execution. Expiry reports `timed_out`, including in the queue.
Independent lifetime guards stop job, probe, and inference process groups if
their owner dies, even when native code blocks the owner's threads. Shutdown
rejects new work and bounds retention/job cleanup waits to five seconds each;
a cleanup failure raises rather than claiming that storage ownership was released.
These are application-level bounds, not a guarantee against an uninterruptible
OS/filesystem operation or an OS scheduling stall. No OS network sandbox is added.

Private directories and a terminable child process are not a container or an OS
network sandbox. The code-owned runner declares no inference network use; the
service reports `os_network_isolation=false`. A public deployment still needs its
own OS/container isolation and egress policy. No public host, TLS termination,
DNS record, credential, or production container was provisioned by this change.

## Current model boundary

The repository includes a code-owned Torchvision Mask R-CNN runner. It accepts
only a hash-pinned safetensors artifact and performs network-free CPU inference;
it never loads the original pickle checkpoint during service inference. A local
one-image smoke run confirmed real detections for one exact checkpoint,
runtime, and image. The bounded record is
[`mcp_image_service_runner_smoke_2026-09-20.json`](../reports/mcp_image_service_runner_smoke_2026-09-20.json).

Users can prepare that exact artifact without giving YOLOZU a download role:

```bash
yolozu prepare-torchvision-maskrcnn \
  --checkpoint /path/to/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth \
  --accept-upstream-terms
```

The command performs no network request. It checks the pinned source size and
SHA-256 before a restricted `weights_only=True`, read-only memory-mapped load,
writes the exact converted artifact under the local YOLOZU model cache, and records a path-free provenance
file. It records the checkpoint license as `NOASSERTION` and both source and
converted redistribution as false. Apache-2.0 applies to YOLOZU source, not to
the checkpoint. Existing destination bytes are never overwritten.

That smoke result is not qualification, license approval, or promotion
evidence. No packaged bundle currently combines this runner with completed
license review, support profiles, activated qualification evidence, and an
Experimental or Stable lifecycle assignment. Therefore the default installed
service still abstains instead of executing a CNN. This boundary must remain in
place until the existing governance gates are completed for an exact bundle.

The [candidate review](image_service_candidate_review.md) records a validated,
unregistered bundle proposal, exact runtime/component identities, a repeated
one-image smoke, the approved local-preparation boundary, and the remaining
review gates. The historical proposal retains `license_expression=unknown`; a
future managed spec must use `NOASSERTION` unless separate rights evidence is
approved. Preparation changes no packaged registry or lifecycle record.
