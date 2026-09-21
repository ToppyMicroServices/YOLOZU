# Local OpenAI image-service plugin

`plugins/yolozu-image-service` packages the existing five-tool MCP image service
and one workflow skill for local Codex use. It uses the supported
`.codex-plugin/plugin.json` compatibility format. It is not a public-directory
release, an authenticated multi-user service, or a qualified CNN distribution.

The plugin adds no model, weights, dependency installer, lifecycle hooks, remote
endpoint, or provider API call. CNN license review, quality evaluation, and
activation remain separate under `YOLOZU-0rp4`; see
[the candidate review](image_service_candidate_review.md).

## Prepare a local copy

Use a trusted Python environment where **this source revision** and the optional
MCP dependency are already installed. A published PyPI version is not assumed
to contain the current image-service implementation. When preparing a new
environment yourself, review the dependencies before installing the checkout:

```bash
python3 -m venv .venv-plugin
.venv-plugin/bin/python -m pip install -e '.[mcp]'
python3 tools/prepare_image_service_plugin.py --help
python3 tools/prepare_image_service_plugin.py \
  --python .venv-plugin/bin/python \
  --output /absolute/staging/plugins/yolozu-image-service
```

The preparation command checks the real five-tool surface and copies only the
manifest, MCP settings, skill, client script, and repository license. It pins
the local copy's command to that interpreter without resolving away its
virtualenv. It makes no network requests and refuses to overwrite a destination.
The source `.mcp.json` uses `python3`; use a prepared copy when the desktop
application has a different Python on its PATH. The copy depends on the selected
runtime remaining installed. Model weights are not covered by the code license.

## Install and try it locally

Use OpenAI's `plugin-creator` workflow to add the prepared plugin to a personal
marketplace, preserving existing entries. The default catalog is
`~/.agents/plugins/marketplace.json`; its `./plugins/yolozu-image-service` entry
resolves to `~/plugins/yolozu-image-service`. Registration and installation are
separate from preparing the package. Do not commit machine-specific interpreter
paths or edit unrelated global settings.

Install it from the personal source in the desktop plugin browser, then start
a new task. The local MCP connection launches only:

```text
<selected-python> -I -m yolozu.integrations.mcp_cli --transport stdio --surface image-service
```

First ask: "Show the YOLOZU image service capabilities without running inference."
Only `image_service_capabilities`, `put_image_asset`, `submit_image_job`,
`get_image_job`, and `cancel_image_job` should be available. No full-surface tool
is added. The long-lived connection uses the host's working directory and the
existing `runs/mcp_image_service` retention rules; it is not an OS sandbox.
Only one service may own a workspace/tenant at a time. A second connection is
rejected with `tenant_in_use` when it tries to access storage; the bundled
temporary-session client uses a separate workspace and does not share this lock.

For a local image, the bundled client avoids copying base64 through chat:

```bash
<selected-python> /absolute/plugin/scripts/image_service_client.py --help
<selected-python> /absolute/plugin/scripts/image_service_client.py --capabilities
<selected-python> /absolute/plugin/scripts/image_service_client.py \
  --image /absolute/image.png --class cat
```

The default is a selection preview. Add `--execute` only for an authorized
processing request; it does not override qualification. A chat attachment must
first be available as a user-selected local file. Automatic ChatGPT attachment
transfer is not implemented.

The client starts a separate temporary stdio session, uploads one bounded image,
submits the job, and polls no faster than once per second. `--timeout` is 1..120
seconds (default 60); expiry requests termination of queued or running work.
The client also sets the server deadline (minimum 30 seconds). Server deadlines
include queueing and selection, and terminal `timed_out` is an unsuccessful result.
Transport initialization and cleanup have separate bounds; `--timeout` is not
a hard wall-clock guarantee for the entire command. Independent guards reclaim
owned process groups after owner death. Original images are
unchanged, and temporary copies/job records are removed on normal exit. Forced
termination can leave an OS temporary directory. IDs from this session cannot
be reused on the plugin's long-lived connection.

Exit 0 means the request returned successfully, **not that inference ran**:
inspect `job.result.outcome` and reason codes for abstention. Exit 2 means invalid
input, protocol/runtime failure, failed/cancelled/timed-out job; exit 130 means
interruption. The client rejects symlinks at the input filename, non-regular
files, images over 8 MiB, unsupported formats, and oversized dimensions.

## Verification and public-release boundary

```bash
python3 tools/validate_tool_manifest.py --manifest tools/manifest.json --require-declarative
python3 -m unittest tests.test_packaged_tools_manifest tests.test_manifest_docs_references
<selected-python> -m unittest tests.test_image_service_plugin
```

Tests cover metadata, a relocated prepared copy, the live stdio surface, real
image upload/job/abstention, invalid inputs, and mocked polling/cancellation.
These checks do not establish host UI installation, CNN quality, or public
ChatGPT connectivity. Record those states separately.

For a future public plugin, reuse the MCP backend but separately implement and
test the image attachment handoff, HTTPS hosting, OAuth/user isolation, tool
annotations, and data-handling disclosures. Public submission requires review;
local packaging does not grant directory approval.

Official guidance checked 2026-09-21:
[packaging](https://developers.openai.com/plugins/build/plugins),
[authentication](https://developers.openai.com/plugins/build/auth), and
[public submission](https://developers.openai.com/plugins/deploy/submission).
