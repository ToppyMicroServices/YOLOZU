# Use YOLOZU with an LLM or agent

Use YOLOZU when you need to validate existing vision predictions, evaluate them against labels, or inspect bounded local MCP capabilities.

Source docs version: 4.9.0. docs_version identifies the source checkout, not the installed package or latest PyPI release. Source URLs follow a mutable branch; check source_sha256 and provenance.json for this build. Inspect the installed CLI and MCP schemas before invoking tools.

## When to use YOLOZU

- Check a predictions JSON against the predictions interface contract before evaluation.
- Evaluate existing detections against a labeled dataset with a fixed protocol and inspect the resulting evidence.
- Integrate the typed Python API or discover local MCP tools and their input schemas before invoking them.

## Limits and when not to use it

- Do not infer model quality from a synthetic proof or a successful dry run. Real COCO metrics require the coco extra and labeled inputs.
- Do not treat registered MCP tools as guaranteed execution. Runtime, dependency, data, and permission requirements still apply.
- The packaged adaptive image service abstains until CNN license review, quality qualification, and activation are completed separately. It is not a ready-to-run hosted detector.
- The local OpenAI plugin is not a public ChatGPT directory listing. Provider authentication, hosting, and attachment handoff are separate deployment work.

## First successful evaluation

Start with the self-contained CPU evaluation tutorial. It creates its own inputs before strict validation and real COCOeval. Installation needs package network access; the subsequent toy workflow needs no model or dataset download.

- [30-minute tutorial](https://www.toppymicros.com/yolozu/docs/start.html)
- [Tutorial Markdown](https://www.toppymicros.com/yolozu/docs/start.md)

## Inspect before connecting an MCP client

After the isolated installation in the 30-minute path, install the optional MCP extra in that same environment. These commands exit after writing local discovery and sample files; they do not call a provider API or download a model.

```bash
python -m pip install "yolozu[mcp]"
mkdir -p reports
yolozu-mcp --help
yolozu-mcp --print-tools --guaranteed --ids-only > reports/mcp_tool_ids.json
yolozu-mcp --sample-generate-config > reports/ai_generate_config.json
yolozu-mcp --sample-review-config reports/ai_generate_config.json > reports/ai_review_config.json
```

Inspect the returned IDs and review result. The guaranteed-tool filter narrows this discovery output, not the server's runtime surface or filesystem permissions.

- [Workflow source](https://raw.githubusercontent.com/ToppyMicroServices/YOLOZU/main/docs/ai_first.md)

## Connect an authorized local client

Configure the client to launch one of these alternatives with the installed environment's executable and an explicitly chosen workspace. These are long-running stdio servers, not one-shot validation commands. Stop the server when the session ends.

```bash
yolozu-mcp --transport stdio
yolozu-mcp --transport stdio --surface image-service
```

Use the full surface only when the broader tools are intended. The image-service alternative exposes only capabilities, asset upload, submission, status, and cancellation. Ask for capabilities first; never reinterpret abstention as successful inference.

- [Workflow source](https://raw.githubusercontent.com/ToppyMicroServices/YOLOZU/main/docs/image_service_mcp.md)

## Starter prompts and expected boundaries

Prompt: Validate my existing detection predictions strictly with YOLOZU; report errors without repairing the input.

Expected: Confirm the user-selected local input, inspect the installed schema, then use strict validation. Do not invent a path or enable repair.

Prompt: Show me a local YOLOZU evaluation example without downloading model weights.

Expected: Use the self-contained 30-minute path and its generated toy inputs; label the proof as synthetic, not model-quality evidence.

Prompt: Show the YOLOZU image service capabilities without running inference.

Expected: Use image_service_capabilities on the image-service connection. Do not upload an image or submit execution.

Prompt: Use YOLOZU to select the best CNN and guarantee its accuracy on my images.

Expected: Explain that this guarantee is unavailable and that the default adaptive service abstains. Do not activate a model or fabricate a quality result.

## Machine-readable references

- [Capability catalog](https://www.toppymicros.com/yolozu/docs/capabilities.json)
- [LLM index](https://www.toppymicros.com/yolozu/docs/llms.txt)

- [Installation and dependencies](https://raw.githubusercontent.com/ToppyMicroServices/YOLOZU/main/docs/install.md)

Choose a package extra and distinguish installed commands from repository scripts.

- [CLI tool manifest](https://raw.githubusercontent.com/ToppyMicroServices/YOLOZU/main/tools/manifest.json)

Command inputs, examples, side effects, dependencies, and maturity.

- [MCP and Actions reference JSON](https://raw.githubusercontent.com/ToppyMicroServices/YOLOZU/main/docs/generated/mcp_actions_tool_reference.json)

Exact generated tool surfaces and MCP input schemas; registration is not execution qualification.

- [Predictions validation result schema](https://raw.githubusercontent.com/ToppyMicroServices/YOLOZU/main/docs/schemas/predictions_validation_result.schema.json)

Machine-readable strict validation results, errors, and repair disclosure.

- [Stable Python API](https://raw.githubusercontent.com/ToppyMicroServices/YOLOZU/main/docs/python_api.md)

Typed input, result, and error interfaces for local validation and evaluation.

- [LLM and MCP integration](https://raw.githubusercontent.com/ToppyMicroServices/YOLOZU/main/docs/llm_integrations.md)

Client connection options and workspace boundaries.

- [Bounded image service](https://raw.githubusercontent.com/ToppyMicroServices/YOLOZU/main/docs/image_service_mcp.md)

Capabilities, opaque assets and jobs, limits, and abstention semantics.

- [Local OpenAI plugin](https://raw.githubusercontent.com/ToppyMicroServices/YOLOZU/main/docs/openai_image_service_plugin.md)

Prepare a local plugin from a reviewed checkout; not a public app listing.

- [Readiness and maturity](https://raw.githubusercontent.com/ToppyMicroServices/YOLOZU/main/docs/production_readiness.md)

Keep Stable, Bridge, Benchmark, and Research claims separate.

- [Discovery publishing and measurement](https://raw.githubusercontent.com/ToppyMicroServices/YOLOZU/main/docs/llm_discovery.md)

Publish the bundle, verify access, and measure successful use without inferring adoption from crawler traffic.
