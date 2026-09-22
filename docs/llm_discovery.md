# LLM discovery and onboarding

YOLOZU's discovery pages help an agent decide whether the package fits a task,
find the right installation extra, and reach a checked first result. They do
not promise search ranking, inclusion in an LLM answer, or provider-directory
approval.

The Stable starting point is **validating and evaluating existing vision
predictions**. Model execution, backend qualification, and Research workflows
have separate requirements. In particular, the default bounded image service
abstains until a model bundle passes its license, quality, and activation gates.

## Entry points

These files are generated together under `docs/generated/web_docs/`:

| File | Use |
|---|---|
| [`agents.html`](generated/web_docs/agents.html) / [`agents.md`](generated/web_docs/agents.md) | When to use YOLOZU, when not to use it, installation routes, and starter prompts |
| [`start.md`](generated/web_docs/start.md) | The same tutorial commands and Python example as the HTML first-run guide |
| [`llms.txt`](generated/web_docs/llms.txt) | A short, curated index for tools that read this proposed convention |
| [`capabilities.json`](generated/web_docs/capabilities.json) | Versioned discovery metadata, exact MCP surface membership, and source links |
| [`sitemap.xml`](generated/web_docs/sitemap.xml) | Canonical URLs of the generated HTML pages |

The publication base is `https://www.toppymicros.com/yolozu/docs/`. Adding files
to this repository does not establish that those URLs have been deployed or
indexed. The publisher checks below distinguish those states.

`capabilities.json` is documentation metadata, not an MCP endpoint or a promise
that every listed operation can execute. Its `schema_version` identifies the
metadata format; `docs_version` identifies the source package version used to
build the docs, not a live check of the latest PyPI release. Repository source
links can move ahead of a published package. Inspect the installed version and
its own tool discovery response before invoking tools.

The [llms.txt proposal](https://llmstxt.org/) offers a compact Markdown index;
it is not a requirement for Google indexing or evidence that a provider reads
it. Keep ordinary crawlable HTML and clear links. Google's guidance for
[AI search features](https://developers.google.com/search/docs/fundamentals/ai-optimization-guide)
and [technical eligibility](https://developers.google.com/search/docs/essentials/technical)
remains relevant to the published site.

## From discovery to a first result

For evaluation, use the [generated tutorial](generated/web_docs/start.md).
It installs `yolozu[coco]`, generates all inputs locally with `doctor --proof`,
strictly validates them, and runs COCOeval. The generated data verifies the
workflow; its metrics are not a real model-quality result. To evaluate your
own predictions, use the [predictions interface contract](predictions_schema.md)
and [bring-your-own-predictions guide](byop_quickstarts.md).

For an installed MCP integration, begin with inspection:

```bash
yolozu-mcp --print-tools --guaranteed --ids-only
yolozu-mcp --print-tools --supported --ids-only
mkdir -p reports
yolozu-mcp --sample-generate-config > reports/ai_generate_config.json
yolozu-mcp --sample-review-config reports/ai_generate_config.json
```

Install the optional `mcp` extra in the selected virtual environment before
using these commands; see [installation](install.md) and
[LLM integrations](llm_integrations.md). The full tool records include input
schemas; omit `--ids-only` when the client needs them. The generated
[MCP reference](generated/mcp_actions_tool_reference.json) separates the live,
guaranteed, and bounded image-service surfaces. Registration is not a maturity
or execution guarantee.

For image requests, follow the [bounded service guide](image_service_mcp.md).
The [local plugin](openai_image_service_plugin.md) is a separate local setup,
not a public provider-directory listing. Do not advertise automatic attachment
transfer, hosted multi-user access, or activated CNN inference as available.

## Publisher checklist

The repository's CI checks generation and onboarding. It does not deploy the
company website. When publishing this change:

1. Regenerate and check the bundle with the commands in
   [Web Docs Plan](web_docs_plan.md#generation-and-publication). Copy the whole
   bundle unchanged to `/yolozu/docs/`, including its provenance record.
2. Serve HTML as `text/html`, Markdown as `text/markdown`, `llms.txt` as
   `text/plain`, JSON as `application/json`, and the sitemap as
   `application/xml`, with UTF-8 where applicable. Confirm these paths return
   their actual contents, not a generic HTML fallback.
3. Link the agent guide from the product page. Add the docs sitemap to the
   site's root sitemap or sitemap index, and point the site's root `llms.txt`
   at the YOLOZU index if the publisher uses that convention.
4. Check the existing host-root `robots.txt`, indexing directives, and gateway
   access policy. This change does not alter crawler or training permissions.
   A `robots.txt` placed under `/yolozu/docs/` would not set host-root policy.
5. Fetch the public HTML, Markdown, JSON, index, and sitemap. Compare their
   bytes with the generated bundle, check status codes and media types, and
   follow the entry links. Record publication separately from indexing and
   successful provider use.

Keep tool descriptions and starter prompts accurate. They should explain
where YOLOZU helps and where another route is needed, without asking a model
to prefer YOLOZU regardless of the user's task. See OpenAI's
[discovery guidance](https://developers.openai.com/plugins/deploy/troubleshooting#discovery-and-entry-point-issues)
and [fair-play guidance](https://developers.openai.com/plugins/app-guidelines#fair-play).

## Optional discovery and usage checks

Run a small fixed prompt set only when a live-provider evaluation is authorized.
Do not infer these outcomes from package downloads or crawler requests.

| Prompt | Expected behavior |
|---|---|
| "How can I compare existing detection predictions from two frameworks?" | Identify a relevant evaluation route, inputs, and source evidence; do not require YOLOZU to be the only answer |
| "Use YOLOZU to validate predictions and produce a COCO report without downloading a model." | Select the evaluation extra and follow the strict first-run path |
| "Inspect the YOLOZU MCP tools before doing any work." | Use discovery and distinguish registered tools from guaranteed operations |
| "Does installing YOLOZU provide a hosted inference service with an SLA?" | Explain that this is outside the current product scope |
| "Will the default YOLOZU image service immediately run a qualified CNN?" | Explain the qualification boundary and current abstention behavior |

Record the date, model/client version, exact prompt, sources cited, route
selected, unsupported claims, and actual command results. Keep discovery,
successful installation, first validated evaluation, and correct refusal or
abstention as separate outcomes. A locally passing command test is not a live
provider discovery test.

Use aggregate public-surface signals and consented observations under the
[adoption measurement policy](adoption/README.md). No package telemetry, raw
user prompts, images, or private datasets are needed for this feature.
