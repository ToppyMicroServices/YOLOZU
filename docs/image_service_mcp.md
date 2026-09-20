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
job-state files are written with mode `0600`. Queued jobs use one worker. Assets,
terminal job state, and managed outputs expire under the configured retention
policy. Each tenant is capped at 128 retained assets, 16 active jobs, and 1,024
retained job records. An asset referenced by a queued or running job is not
removed by retention cleanup. Cleanup is restricted to validated service-owned
directories.

`cancel_image_job` cancels a queued job. It does not claim that a model already
running in the worker was interrupted. The adaptive runner has its own bounded
child-process timeout when an eligible pipeline reaches execution.

## Current model boundary

The repository includes a code-owned Torchvision Mask R-CNN runner. It accepts
only a hash-pinned safetensors artifact and performs network-free CPU inference;
it never loads the original pickle checkpoint during service inference. A local
one-image smoke run confirmed real detections for one exact checkpoint,
runtime, and image. The bounded record is
[`mcp_image_service_runner_smoke_2026-09-20.json`](../reports/mcp_image_service_runner_smoke_2026-09-20.json).

That smoke result is not qualification, license approval, or promotion
evidence. No packaged bundle currently combines this runner with completed
license review, support profiles, activated qualification evidence, and an
Experimental or Stable lifecycle assignment. Therefore the default installed
service still abstains instead of executing a CNN. This boundary must remain in
place until the existing governance gates are completed for an exact bundle.
