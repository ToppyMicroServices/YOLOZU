---
name: yolozu-image-service
description: Use YOLOZU to inspect local image-service capabilities or submit bounded object-detection image jobs. Apply when the user asks to use YOLOZU for an image, not for general image discussion or model training.
---

# YOLOZU image service

This experimental plugin connects to a local service. Installation does not
qualify a CNN or enable one. No qualified registered model means abstention,
not an empty detection result.

## Choose the input route

- For capabilities, call `image_service_capabilities`. This does not run inference.
- For an existing asset or job from this live service, use the five MCP tools
  below. Do not invent identifiers or reuse identifiers from a different session.
- For a user-selected local image, use the bundled
  [image client](../../scripts/image_service_client.py) with the Python runtime
  configured for this plugin. It transfers bytes programmatically over local
  stdio, waits for the result, and prints JSON without exposing image base64 to
  the conversation. Run its `--help` for the available options.
- A chat attachment is not automatically a local path or an `asset_id`. If the
  client cannot access it, ask for an accessible local file. Do not fetch URLs,
  search unrelated directories, or claim an attachment was processed.

## Submit a bounded request

Identify the target classes from the user's request; ask if they are unclear.
The local client takes `--image /absolute/image.png --class cat` (repeat
`--class` for more labels). Without `--execute`, it previews selection only.
It creates a separate temporary service session; its asset/job IDs are not
usable by the plugin's long-lived MCP connection. The temporary copy and job
records are removed when the client exits; the original image is unchanged.

For the long-lived MCP connection:

1. Check `image_service_capabilities` for supported tasks and current limits.
2. Use `put_image_asset` only with real image bytes and the matching MIME type.
   Do not fabricate or transcribe large base64 strings. Prefer the local client
   when a direct programmatic transfer is unavailable.
3. Call `submit_image_job` with the returned `asset_id`, explicit
   `fixed_classes`, and `execute=false` for a preview.
4. Use `get_image_job` at most once per second and stop at a terminal status.
   After two minutes, report the job ID and current state instead of polling
   indefinitely. Use `cancel_image_job` when the user requests cancellation.

Use `execute=true` or the client's `--execute` only when the user has requested
actual processing of that image. Installing the plugin or asking for a preview
does not authorize execution. This flag does not prove human approval; preserve
the host's permission checks. Never change tool-approval settings to make a job run.

## Interpret the result

Report the returned state: preview, completed, abstained, failed, cancelled, or
timed out. A completed job wrapper can contain an abstained result. Show reason
codes when available. Do not turn abstention into "no objects found" or
substitute the language model's own vision analysis without explaining that it
is a different operation.

The service does not accept model paths, arbitrary URLs, backend names, shell
commands, or output paths. Do not work around these limits with the full MCP
surface. Do not download weights, edit the registry, approve licenses, qualify,
activate, or promote a model as part of an image request. Those are separate
operator workflows. Do not claim OS-level isolation or public ChatGPT availability.
