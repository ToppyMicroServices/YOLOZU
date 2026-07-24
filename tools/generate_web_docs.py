#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "tools" / "manifest.json"
DEFAULT_SCHEMAS = REPO_ROOT / "docs" / "schemas"
DEFAULT_CONTENT = REPO_ROOT / "docs" / "web_docs_content.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "generated" / "web_docs"
DEFAULT_ASSETS = REPO_ROOT / "docs" / "web_docs_assets"
LANE_IDS = ("stable", "bridge", "benchmark", "research")
PAGE_NAV = (
    ("index", "Overview", "index.html"),
    ("start", "30-minute path", "start.html"),
    ("commands", "Commands", "commands.html"),
    ("schemas", "Schemas", "schemas.html"),
    ("examples", "Examples", "examples.html"),
    ("glossary", "Glossary", "glossary.html"),
    ("troubleshooting", "What can go wrong", "troubleshooting.html"),
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the searchable YOLOZU onboarding web-docs bundle from "
            "the tool manifest, JSON Schemas, and curated source content."
        )
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Source tools manifest JSON.",
    )
    parser.add_argument(
        "--schemas",
        default=str(DEFAULT_SCHEMAS),
        help="Directory containing source JSON Schemas.",
    )
    parser.add_argument(
        "--content",
        default=str(DEFAULT_CONTENT),
        help="Curated web-docs content JSON.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Generated static-site output directory.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail without writing when the generated bundle is stale.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the generation or drift-check result as JSON.",
    )
    return parser.parse_args(argv)


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _load_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"{label} not found: {_repo_rel(path)}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} is not valid JSON: {_repo_rel(path)}: {exc}") from exc


def _as_nonempty_string(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{where} must be a non-empty string")
    return value.strip()


def _as_list(value: Any, *, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise SystemExit(f"{where} must be a list")
    return value


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "entry"


def _source_url(content: dict[str, Any], rel: str) -> str:
    base = _as_nonempty_string(
        content["site"].get("repository_base"),
        where="site.repository_base",
    )
    return base.rstrip("/") + "/" + rel.lstrip("/")


def _source_link(content: dict[str, Any], rel: str, label: str = "Source") -> str:
    return (
        f'<a href="{html.escape(_source_url(content, rel), quote=True)}" '
        f'target="_blank" rel="noopener noreferrer">{html.escape(label)}</a>'
    )


def _nav(active: str) -> str:
    links = []
    for page_id, label, href in PAGE_NAV:
        current = ' aria-current="page"' if page_id == active else ""
        links.append(f'<a href="{href}"{current}>{html.escape(label)}</a>')
    return "\n".join(links)


def _search_shell(label: str) -> str:
    return f"""
<div class="search-shell" role="search">
  <label for="docs-search">{html.escape(label)}</label>
  <input id="docs-search" class="search-input" type="search"
    placeholder="Try validate, predictions, protocol, schema, or TTT"
    autocomplete="off" data-docs-search />
  <div class="search-results" aria-live="polite" data-docs-search-results></div>
</div>
"""


def _layout(
    *,
    content: dict[str, Any],
    page_id: str,
    title: str,
    description: str,
    body: str,
) -> str:
    site = content["site"]
    canonical_base = _as_nonempty_string(
        site.get("canonical_base"),
        where="site.canonical_base",
    ).rstrip("/")
    filename = "index.html" if page_id == "index" else f"{page_id}.html"
    canonical = f"{canonical_base}/" if page_id == "index" else f"{canonical_base}/{filename}"
    full_title = f"{title} — YOLOZU Docs"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(full_title)}</title>
  <meta name="description" content="{html.escape(description, quote=True)}" />
  <meta name="robots" content="index,follow" />
  <meta name="theme-color" content="#07111f" />
  <link rel="canonical" href="{html.escape(canonical, quote=True)}" />
  <meta property="og:type" content="website" />
  <meta property="og:title" content="{html.escape(full_title, quote=True)}" />
  <meta property="og:description" content="{html.escape(description, quote=True)}" />
  <meta property="og:url" content="{html.escape(canonical, quote=True)}" />
  <meta property="og:site_name" content="ToppyMicroServices" />
  <meta name="twitter:card" content="summary" />
  <link rel="stylesheet" href="assets/styles.css" />
  <script defer data-domain="toppymicros.com" data-auto="false"
    src="https://plausible.io/js/script.tag"></script>
  <script defer src="assets/docs.js"></script>
</head>
<body data-page="{html.escape(page_id, quote=True)}">
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="site-header">
    <nav class="container nav" aria-label="YOLOZU docs">
      <a class="brand" href="/yolozu/">YOLOZU · ToppyMicroServices</a>
      <div class="nav-links">
        {_nav(page_id)}
      </div>
    </nav>
  </header>
  <main id="main" class="container">
    {body}
  </main>
  <footer class="site-footer">
    <div class="container footer-row">
      <span>YOLOZU is a free-of-charge commercial ToppyMicroServices product.</span>
      <a href="/yolozu/">Product page</a>
      <a href="https://github.com/ToppyMicroServices/YOLOZU"
        target="_blank" rel="noopener noreferrer">Repository</a>
      <a href="/privacy-policy.html">Privacy</a>
    </div>
  </footer>
</body>
</html>
"""


def _lane_cards(content: dict[str, Any]) -> str:
    cards = []
    for lane in content["lanes"]:
        lane_id = _as_nonempty_string(lane.get("id"), where="lanes[].id")
        title = _as_nonempty_string(lane.get("title"), where=f"lanes[{lane_id}].title")
        summary = _as_nonempty_string(
            lane.get("summary"),
            where=f"lanes[{lane_id}].summary",
        )
        start_page = _as_nonempty_string(
            lane.get("start_page"),
            where=f"lanes[{lane_id}].start_page",
        )
        stable_return = ""
        if lane_id == "research":
            return_href = _as_nonempty_string(
                lane.get("stable_return"),
                where="lanes[research].stable_return",
            )
            stable_return = (
                f'<p><a href="{html.escape(return_href, quote=True)}">'
                "Return to the stable artifact</a></p>"
            )
        cards.append(
            f"""
<article class="card lane-card lane-{lane_id}" data-search-card
  data-search-text="{html.escape(f'{title} {summary} {lane_id}', quote=True)}">
  <span class="badge badge-{lane_id}">{html.escape(lane_id)}</span>
  <h3>{html.escape(title)}</h3>
  <p>{html.escape(summary)}</p>
  <p><a href="{html.escape(start_page, quote=True)}">Open this lane</a></p>
  {stable_return}
</article>
"""
        )
    return "\n".join(cards)


def _render_index(content: dict[str, Any]) -> str:
    body = f"""
<section class="hero" aria-labelledby="page-title">
  <span class="eyebrow">Generated onboarding surface</span>
  <h1 id="page-title">Evaluate existing predictions</h1>
  <p class="lead">Start with one wrapped predictions artifact, validate the
  predictions interface contract, and produce a comparable report. Search the
  generated command and schema references without navigating the full repository
  or PDF manual.</p>
  <div class="hero-actions">
    <a class="button primary" href="start.html" data-docs-event="YOLOZU docs entry"
      data-event-target="30-minute-path">Start the 30-minute path</a>
    <a class="button" href="commands.html">Browse commands</a>
    <a class="button" href="schemas.html">Browse schemas</a>
  </div>
  {_search_shell("Search every generated page")}
</section>
<section class="section" aria-labelledby="lanes-title">
  <h2 id="lanes-title">Choose the lane after the stable artifact works</h2>
  <p class="section-copy">The four lanes are intentionally separate. Bridge,
  Benchmark, and Research work extend or qualify the Stable evaluation boundary;
  they do not silently promote themselves to Stable support.</p>
  <div class="grid four">
    {_lane_cards(content)}
  </div>
</section>
<section class="section" aria-labelledby="surfaces-title">
  <h2 id="surfaces-title">Short paths, repository-backed details</h2>
  <div class="grid two">
    <article class="card" data-search-card data-search-text="tutorial install doctor proof validate eval">
      <h3>30-minute tutorial</h3>
      <p>Install, produce CPU-only proof artifacts, validate both sides, evaluate,
      and read the report.</p>
      <a href="start.html">Follow the tutorial</a>
    </article>
    <article class="card" data-search-card data-search-text="generated commands manifest cli reference">
      <h3>Generated command reference</h3>
      <p>Every entry comes from <code>tools/manifest.json</code>, including
      maturity, inputs, examples, and source links.</p>
      <a href="commands.html">Search commands</a>
    </article>
    <article class="card" data-search-card data-search-text="schema browser json predictions reports training research">
      <h3>Schema browser</h3>
      <p>Inspect the checked-in prediction, evaluation, training, and Research
      JSON Schema surfaces.</p>
      <a href="schemas.html">Search schemas</a>
    </article>
    <article class="card" data-search-card data-search-text="examples reports overlays expected files">
      <h3>Examples and report reading</h3>
      <p>See report and overlay examples, expected companion files, and the
      checklist for fair comparison.</p>
      <a href="examples.html">Open examples</a>
    </article>
    <article class="card" data-search-card data-search-text="glossary interface contract protocol lane">
      <h3>Glossary</h3>
      <p>Use the same terms for artifacts, protocol identity, maturity, and
      skipped lanes.</p>
      <a href="glossary.html">Open glossary</a>
    </article>
    <article class="card" data-search-card data-search-text="troubleshooting failures doctor validation drift">
      <h3>What can go wrong</h3>
      <p>Start from observed symptoms, verify the report, and avoid attributing
      drift before inputs and protocol align.</p>
      <a href="troubleshooting.html">Open the failure guide</a>
    </article>
  </div>
</section>
"""
    return _layout(
        content=content,
        page_id="index",
        title="Evaluate existing predictions",
        description=content["site"]["description"],
        body=body,
    )


def _commands_block(commands: list[str]) -> str:
    if not commands:
        return ""
    rendered = "\n".join(html.escape(command) for command in commands)
    return f"<pre><code>{rendered}</code></pre>"


def _render_start(content: dict[str, Any]) -> str:
    steps = []
    for step in content["tutorial"]["thirty_minute"]:
        minutes = _as_nonempty_string(step.get("minutes"), where="tutorial step minutes")
        title = _as_nonempty_string(step.get("title"), where="tutorial step title")
        expected = _as_nonempty_string(step.get("expected"), where="tutorial step expected")
        commands = [
            _as_nonempty_string(command, where=f"tutorial[{title}].commands[]")
            for command in _as_list(step.get("commands"), where=f"tutorial[{title}].commands")
        ]
        search_text = f"{minutes} {title} {expected} {' '.join(commands)}"
        steps.append(
            f"""
<article class="step" data-search-card
  data-search-text="{html.escape(search_text, quote=True)}">
  <div class="meta"><span>{html.escape(minutes)} minutes</span></div>
  <h3>{html.escape(title)}</h3>
  {_commands_block(commands)}
  <p><strong>Expected:</strong> {html.escape(expected)}</p>
</article>
"""
        )
    two_hour = "\n".join(
        f"<li>{html.escape(_as_nonempty_string(item, where='tutorial.two_hour[]'))}</li>"
        for item in content["tutorial"]["two_hour"]
    )
    body = f"""
<section class="hero" aria-labelledby="page-title">
  <span class="eyebrow">Stable lane</span>
  <h1 id="page-title">A 30-minute path to a checked report</h1>
  <p class="lead">This path is CPU-only and begins with generated proof artifacts,
  so the dataset and prediction paths come from evidence rather than memory.</p>
  {_search_shell("Filter steps or search all docs")}
</section>
<section class="section" aria-labelledby="steps-title">
  <h2 id="steps-title">Proof → visible demo → validate → evaluate</h2>
  <div class="steps">
    {''.join(steps)}
  </div>
  <div class="hero-actions">
    <a class="button primary" href="examples.html"
      data-docs-event="YOLOZU docs completion"
      data-event-target="checked-report">I have a checked report</a>
    <a class="button" href="troubleshooting.html">Something failed</a>
  </div>
  <p class="source-note">Canonical procedure:
    {_source_link(content, "docs/cpu_only_dod.md", "CPU-only DoD")}.
  </p>
</section>
<section class="section" aria-labelledby="two-hour-title">
  <h2 id="two-hour-title">Continue to the two-hour path</h2>
  <ol class="check-list">{two_hour}</ol>
  <p class="source-note">Use
    {_source_link(content, "docs/evaluation_protocol_template.md", "the evaluation protocol template")}
    before comparing artifacts.</p>
</section>
"""
    return _layout(
        content=content,
        page_id="start",
        title="30-minute path",
        description="CPU-only proof, visible demo, strict validation, and pinned evaluation.",
        body=body,
    )


def _render_input_list(inputs: Any) -> str:
    if not isinstance(inputs, list) or not inputs:
        return "<p>No declared CLI inputs.</p>"
    rows = []
    for item in inputs:
        if not isinstance(item, dict):
            continue
        flag = str(item.get("flag") or item.get("name") or "").strip()
        if not flag:
            continue
        details = []
        if item.get("required") is True:
            details.append("required")
        if "default" in item:
            details.append(f"default={item['default']!s}")
        kind = str(item.get("kind") or "").strip()
        if kind:
            details.append(kind)
        suffix = f" — {', '.join(details)}" if details else ""
        rows.append(f"<li><code>{html.escape(flag)}</code>{html.escape(suffix)}</li>")
    return f'<ul class="check-list">{"".join(rows)}</ul>' if rows else "<p>No declared CLI inputs.</p>"


def _render_commands(content: dict[str, Any], tools: list[dict[str, Any]]) -> str:
    cards = []
    for tool in tools:
        tool_id = _as_nonempty_string(tool.get("id"), where="tools[].id")
        summary = _as_nonempty_string(tool.get("summary"), where=f"tools[{tool_id}].summary")
        maturity = _as_nonempty_string(
            tool.get("maturity"),
            where=f"tools[{tool_id}].maturity",
        )
        entrypoint = _as_nonempty_string(
            tool.get("entrypoint"),
            where=f"tools[{tool_id}].entrypoint",
        )
        examples = []
        for example in tool.get("examples") or []:
            if isinstance(example, dict) and isinstance(example.get("command"), str):
                examples.append(example["command"])
        docs_links = []
        for rel in tool.get("docs") or []:
            if isinstance(rel, str):
                docs_links.append(_source_link(content, rel, Path(rel).name))
        docs_html = " · ".join(docs_links) if docs_links else "No dedicated prose page."
        search_text = (
            f"{tool_id} {summary} {maturity} {entrypoint} {' '.join(examples)}"
        )
        cards.append(
            f"""
<details class="card" id="tool-{_slug(tool_id)}" data-tool-id="{html.escape(tool_id, quote=True)}"
  data-search-card data-search-text="{html.escape(search_text, quote=True)}">
  <summary><code>{html.escape(tool_id)}</code> — {html.escape(summary)}</summary>
  <div class="meta">
    <span class="badge badge-{html.escape(maturity, quote=True)}">{html.escape(maturity)}</span>
    <span><code>{html.escape(entrypoint)}</code></span>
  </div>
  <h3>Inputs</h3>
  {_render_input_list(tool.get("inputs"))}
  <h3>Examples</h3>
  {_commands_block(examples) if examples else "<p>No declared example command.</p>"}
  <p class="source-note">{_source_link(content, entrypoint, "Implementation")} · {docs_html}</p>
</details>
"""
        )
    body = f"""
<section class="hero" aria-labelledby="page-title">
  <span class="eyebrow">Generated from tools/manifest.json</span>
  <h1 id="page-title">Command reference</h1>
  <p class="lead">{len(tools)} manifest entries with declared maturity, inputs,
  examples, implementation paths, and documentation links. The manifest remains
  the source of truth.</p>
  {_search_shell("Filter command IDs, summaries, flags, and examples")}
</section>
<section class="section" aria-labelledby="commands-title">
  <h2 id="commands-title">Manifested tools</h2>
  <div class="tool-list">{''.join(cards)}</div>
  <p class="source-note">{_source_link(content, "tools/manifest.json", "Open the source manifest")}.</p>
</section>
"""
    return _layout(
        content=content,
        page_id="commands",
        title="Command reference",
        description="Generated reference for every YOLOZU tool manifest entry.",
        body=body,
    )


def _schema_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    title = str(payload.get("title") or path.stem).strip()
    description = str(payload.get("description") or "Checked-in YOLOZU JSON Schema.").strip()
    properties = payload.get("properties")
    property_names = sorted(properties) if isinstance(properties, dict) else []
    required = payload.get("required")
    required_names = sorted(str(value) for value in required) if isinstance(required, list) else []
    return {
        "path": path,
        "title": title,
        "description": description,
        "schema_id": str(payload.get("$id") or "").strip(),
        "properties": property_names,
        "required": required_names,
    }


def _render_schemas(content: dict[str, Any], schemas: list[dict[str, Any]]) -> str:
    cards = []
    for schema in schemas:
        rel = _repo_rel(schema["path"])
        property_text = ", ".join(schema["properties"]) or "No root properties declared"
        required_text = ", ".join(schema["required"]) or "No root required list"
        schema_id = (
            f"<dt>Schema ID</dt><dd><code>{html.escape(schema['schema_id'])}</code></dd>"
            if schema["schema_id"]
            else ""
        )
        cards.append(
            f"""
<article class="card" id="schema-{_slug(schema['path'].stem)}"
  data-schema-path="{html.escape(rel, quote=True)}" data-search-card
  data-search-text="{html.escape(f"{schema['title']} {schema['description']} {rel} {property_text} {required_text}", quote=True)}">
  <h3>{html.escape(schema['title'])}</h3>
  <p>{html.escape(schema['description'])}</p>
  <dl class="kv">
    {schema_id}
    <dt>Root properties</dt><dd><code>{html.escape(property_text)}</code></dd>
    <dt>Root required</dt><dd><code>{html.escape(required_text)}</code></dd>
  </dl>
  <p class="source-note">{_source_link(content, rel, "Open the complete JSON Schema")}</p>
</article>
"""
        )
    body = f"""
<section class="hero" aria-labelledby="page-title">
  <span class="eyebrow">Generated from docs/schemas/*.json</span>
  <h1 id="page-title">Schema browser</h1>
  <p class="lead">{len(schemas)} checked-in JSON Schemas covering predictions,
  evaluation reports, training handoff, registry records, and Research artifacts.
  Open the source schema for complete field constraints.</p>
  {_search_shell("Filter schema titles, paths, properties, or descriptions")}
</section>
<section class="section" aria-labelledby="schemas-title">
  <h2 id="schemas-title">Checked-in schema surfaces</h2>
  <div class="schema-list">{''.join(cards)}</div>
  <p class="source-note">{_source_link(content, "docs/schema_governance.md", "Schema governance and lifecycle")}.</p>
</section>
"""
    return _layout(
        content=content,
        page_id="schemas",
        title="Schema browser",
        description="Generated browser for YOLOZU prediction, report, training, and Research schemas.",
        body=body,
    )


def _render_examples(content: dict[str, Any]) -> str:
    cards = []
    for example in content["examples"]:
        example_id = _as_nonempty_string(example.get("id"), where="examples[].id")
        title = _as_nonempty_string(example.get("title"), where=f"examples[{example_id}].title")
        lane = _as_nonempty_string(example.get("lane"), where=f"examples[{example_id}].lane")
        description = _as_nonempty_string(
            example.get("description"),
            where=f"examples[{example_id}].description",
        )
        expected = "\n".join(
            f"<li><code>{html.escape(_as_nonempty_string(item, where='expected_files[]'))}</code></li>"
            for item in example["expected_files"]
        )
        image = ""
        if example.get("image_output"):
            image = (
                f'<img class="example-image" src="assets/examples/'
                f'{html.escape(example["image_output"], quote=True)}" '
                f'alt="{html.escape(title, quote=True)}" loading="lazy" decoding="async" />'
            )
        stable_artifact = ""
        if example.get("stable_artifact"):
            stable_artifact = (
                " · "
                + _source_link(
                    content,
                    example["stable_artifact"],
                    "Stable baseline artifact",
                )
            )
        search_text = (
            f"{title} {lane} {description} {' '.join(example['expected_files'])}"
        )
        cards.append(
            f"""
<article class="card lane-card lane-{lane}" id="example-{_slug(example_id)}"
  data-search-card data-search-text="{html.escape(search_text, quote=True)}">
  <span class="badge badge-{lane}">{html.escape(lane)}</span>
  <h3>{html.escape(title)}</h3>
  <p>{html.escape(description)}</p>
  {image}
  <h3>Expected companion files or fields</h3>
  <ul class="check-list">{expected}</ul>
  <p class="source-note">{_source_link(content, example["source"], "Source artifact")}{stable_artifact}</p>
</article>
"""
        )
    checklist = "\n".join(
        f"<li>{html.escape(item)}</li>" for item in content["report_checklist"]
    )
    body = f"""
<section class="hero" aria-labelledby="page-title">
  <span class="eyebrow">Artifact-backed examples</span>
  <h1 id="page-title">Examples and report reading</h1>
  <p class="lead">Each example points back to checked-in repository evidence and
  lists the companion files or fields needed to interpret it.</p>
  {_search_shell("Filter examples, lanes, and expected files")}
</section>
<section class="section" aria-labelledby="gallery-title">
  <h2 id="gallery-title">Examples gallery</h2>
  <div class="grid two">{''.join(cards)}</div>
</section>
<section class="section" aria-labelledby="checklist-title">
  <h2 id="checklist-title">Before comparing two reports</h2>
  <div class="notice">
    <ul class="check-list">{checklist}</ul>
  </div>
  <div class="hero-actions">
    <a class="button primary" href="start.html"
      data-docs-event="YOLOZU docs stable return"
      data-event-target="stable-tutorial">Return to stable evaluation</a>
    <a class="button" href="troubleshooting.html">Resolve a mismatch</a>
  </div>
</section>
"""
    return _layout(
        content=content,
        page_id="examples",
        title="Examples and report reading",
        description="Repository-backed YOLOZU reports, overlays, expected files, and comparison checklist.",
        body=body,
    )


def _render_glossary(content: dict[str, Any]) -> str:
    cards = []
    for entry in content["glossary"]:
        term = _as_nonempty_string(entry.get("term"), where="glossary[].term")
        definition = _as_nonempty_string(
            entry.get("definition"),
            where=f"glossary[{term}].definition",
        )
        cards.append(
            f"""
<article class="card" id="term-{_slug(term)}" data-search-card
  data-search-text="{html.escape(f'{term} {definition}', quote=True)}">
  <h3>{html.escape(term)}</h3>
  <p>{html.escape(definition)}</p>
  <p class="source-note">{_source_link(content, entry["source"], "Canonical context")}</p>
</article>
"""
        )
    body = f"""
<section class="hero" aria-labelledby="page-title">
  <span class="eyebrow">Curated from repository SSOT</span>
  <h1 id="page-title">Glossary</h1>
  <p class="lead">Terms for the predictions boundary, protocol identity, maturity
  lanes, and comparison evidence.</p>
  {_search_shell("Filter glossary terms and definitions")}
</section>
<section class="section" aria-labelledby="terms-title">
  <h2 id="terms-title">Core terms</h2>
  <div class="grid two">{''.join(cards)}</div>
</section>
"""
    return _layout(
        content=content,
        page_id="glossary",
        title="Glossary",
        description="YOLOZU glossary for interface contracts, protocol identity, lanes, and evidence.",
        body=body,
    )


def _render_troubleshooting(content: dict[str, Any]) -> str:
    cards = []
    for index, failure in enumerate(content["failures"], start=1):
        symptom = _as_nonempty_string(failure.get("symptom"), where="failures[].symptom")
        check = _as_nonempty_string(failure.get("check"), where=f"failures[{symptom}].check")
        next_step = _as_nonempty_string(
            failure.get("next"),
            where=f"failures[{symptom}].next",
        )
        cards.append(
            f"""
<article class="card" id="failure-{index}" data-search-card
  data-search-text="{html.escape(f'{symptom} {check} {next_step}', quote=True)}">
  <span class="badge badge-danger">Observed symptom</span>
  <h3>{html.escape(symptom)}</h3>
  <dl class="kv">
    <dt>Verify</dt><dd>{html.escape(check)}</dd>
    <dt>Next action</dt><dd>{html.escape(next_step)}</dd>
  </dl>
  <p class="source-note">{_source_link(content, failure["source"], "Detailed guidance")}</p>
</article>
"""
        )
    body = f"""
<section class="hero" aria-labelledby="page-title">
  <span class="eyebrow">Fail visibly</span>
  <h1 id="page-title">What can go wrong</h1>
  <p class="lead">Start from observed output, verify the relevant artifact, and
  change only the condition supported by evidence. A skipped or unavailable lane
  is not a successful result.</p>
  {_search_shell("Filter symptoms, checks, and next actions")}
</section>
<section class="section" aria-labelledby="failures-title">
  <h2 id="failures-title">Symptom → verification → next action</h2>
  <div class="grid two">{''.join(cards)}</div>
</section>
"""
    return _layout(
        content=content,
        page_id="troubleshooting",
        title="What can go wrong",
        description="Evidence-first troubleshooting for YOLOZU installation, validation, evaluation, and lane status.",
        body=body,
    )


def _search_index(
    content: dict[str, Any],
    tools: list[dict[str, Any]],
    schemas: list[dict[str, Any]],
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = [
        {
            "title": "Evaluate existing predictions",
            "kind": "page",
            "summary": "Start with the stable predictions evaluation lane.",
            "href": "index.html",
            "search_text": "evaluate existing predictions stable validation report",
        },
        {
            "title": "30-minute path",
            "kind": "tutorial",
            "summary": "CPU proof, visible demo, strict validation, and evaluation.",
            "href": "start.html",
            "search_text": "install doctor proof demo validate eval tutorial",
        },
    ]
    for lane in content["lanes"]:
        entries.append(
            {
                "title": lane["title"],
                "kind": "lane",
                "summary": lane["summary"],
                "href": lane["start_page"],
                "search_text": f"{lane['id']} {lane['title']} {lane['summary']}",
            }
        )
    for tool in tools:
        tool_id = tool["id"]
        flags = " ".join(
            str(item.get("flag") or "")
            for item in tool.get("inputs") or []
            if isinstance(item, dict)
        )
        entries.append(
            {
                "title": tool_id,
                "kind": "command",
                "summary": tool["summary"],
                "href": f"commands.html#tool-{_slug(tool_id)}",
                "search_text": (
                    f"{tool_id} {tool['summary']} {tool['maturity']} "
                    f"{tool['entrypoint']} {flags}"
                ),
            }
        )
    for schema in schemas:
        entries.append(
            {
                "title": schema["title"],
                "kind": "schema",
                "summary": schema["description"],
                "href": f"schemas.html#schema-{_slug(schema['path'].stem)}",
                "search_text": (
                    f"{schema['title']} {schema['description']} "
                    f"{' '.join(schema['properties'])} {_repo_rel(schema['path'])}"
                ),
            }
        )
    for example in content["examples"]:
        entries.append(
            {
                "title": example["title"],
                "kind": "example",
                "summary": example["description"],
                "href": f"examples.html#example-{_slug(example['id'])}",
                "search_text": (
                    f"{example['title']} {example['lane']} {example['description']} "
                    f"{' '.join(example['expected_files'])}"
                ),
            }
        )
    for entry in content["glossary"]:
        entries.append(
            {
                "title": entry["term"],
                "kind": "glossary",
                "summary": entry["definition"],
                "href": f"glossary.html#term-{_slug(entry['term'])}",
                "search_text": f"{entry['term']} {entry['definition']}",
            }
        )
    for index, failure in enumerate(content["failures"], start=1):
        entries.append(
            {
                "title": failure["symptom"],
                "kind": "troubleshooting",
                "summary": failure["check"],
                "href": f"troubleshooting.html#failure-{index}",
                "search_text": (
                    f"{failure['symptom']} {failure['check']} {failure['next']}"
                ),
            }
        )
    return entries


def _validate_content(content: Any) -> dict[str, Any]:
    if not isinstance(content, dict):
        raise SystemExit("web docs content root must be an object")
    site = content.get("site")
    if not isinstance(site, dict):
        raise SystemExit("web docs content requires site object")
    for key in ("title", "description", "canonical_base", "repository_base"):
        _as_nonempty_string(site.get(key), where=f"site.{key}")

    lanes = _as_list(content.get("lanes"), where="lanes")
    lane_ids = [
        _as_nonempty_string(item.get("id"), where="lanes[].id")
        for item in lanes
        if isinstance(item, dict)
    ]
    if lane_ids != list(LANE_IDS):
        raise SystemExit(f"lanes must be ordered exactly as {LANE_IDS!r}")

    tutorial = content.get("tutorial")
    if not isinstance(tutorial, dict):
        raise SystemExit("web docs content requires tutorial object")
    if not _as_list(tutorial.get("thirty_minute"), where="tutorial.thirty_minute"):
        raise SystemExit("tutorial.thirty_minute must not be empty")
    if not _as_list(tutorial.get("two_hour"), where="tutorial.two_hour"):
        raise SystemExit("tutorial.two_hour must not be empty")

    for key in ("examples", "glossary", "failures", "report_checklist"):
        if not _as_list(content.get(key), where=key):
            raise SystemExit(f"{key} must not be empty")

    source_paths = []
    for lane in lanes:
        source_paths.append(lane.get("source"))
    for group in ("examples", "glossary", "failures"):
        for item in content[group]:
            source_paths.append(item.get("source"))
            if item.get("stable_artifact"):
                source_paths.append(item["stable_artifact"])
            if item.get("image_source"):
                source_paths.append(item["image_source"])
    for rel in source_paths:
        rel_path = _as_nonempty_string(rel, where="content source path")
        if not (REPO_ROOT / rel_path).is_file():
            raise SystemExit(f"web docs source file not found: {rel_path}")
    return content


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path, label="tools manifest")
    if not isinstance(payload, dict) or not isinstance(payload.get("tools"), list):
        raise SystemExit("tools manifest requires a tools list")
    tools = payload["tools"]
    if not tools or any(not isinstance(tool, dict) for tool in tools):
        raise SystemExit("tools manifest entries must be objects")
    ids = [str(tool.get("id") or "") for tool in tools]
    if len(ids) != len(set(ids)) or any(not tool_id for tool_id in ids):
        raise SystemExit("tools manifest ids must be non-empty and unique")
    return sorted(tools, key=lambda item: item["id"])


def _load_schemas(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        raise SystemExit(f"schemas directory not found: {_repo_rel(directory)}")
    schemas = []
    for path in sorted(directory.glob("*.json")):
        payload = _load_json(path, label="JSON Schema")
        if not isinstance(payload, dict):
            raise SystemExit(f"JSON Schema root must be an object: {_repo_rel(path)}")
        schemas.append(_schema_summary(path, payload))
    if not schemas:
        raise SystemExit(f"no JSON Schemas found in {_repo_rel(directory)}")
    return schemas


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _build_bundle(
    *,
    manifest_path: Path,
    schemas_dir: Path,
    content_path: Path,
) -> tuple[dict[str, bytes], dict[str, int]]:
    content = _validate_content(_load_json(content_path, label="web docs content"))
    tools = _load_manifest(manifest_path)
    schemas = _load_schemas(schemas_dir)

    bundle: dict[str, bytes] = {
        "index.html": _render_index(content).encode("utf-8"),
        "start.html": _render_start(content).encode("utf-8"),
        "commands.html": _render_commands(content, tools).encode("utf-8"),
        "schemas.html": _render_schemas(content, schemas).encode("utf-8"),
        "examples.html": _render_examples(content).encode("utf-8"),
        "glossary.html": _render_glossary(content).encode("utf-8"),
        "troubleshooting.html": _render_troubleshooting(content).encode("utf-8"),
        "search-index.json": _json_bytes(_search_index(content, tools, schemas)),
    }
    source_files = [manifest_path, content_path]
    for asset_name in ("styles.css", "docs.js"):
        source = DEFAULT_ASSETS / asset_name
        if not source.is_file():
            raise SystemExit(f"web docs asset not found: {_repo_rel(source)}")
        bundle[f"assets/{asset_name}"] = source.read_bytes()
        source_files.append(source)

    for example in content["examples"]:
        image_source = example.get("image_source")
        image_output = example.get("image_output")
        if not image_source and not image_output:
            continue
        if not image_source or not image_output:
            raise SystemExit(
                f"example {example.get('id')} must set both image_source and image_output"
            )
        source = REPO_ROOT / image_source
        bundle[f"assets/examples/{image_output}"] = source.read_bytes()
        source_files.append(source)

    source_files.extend(schema["path"] for schema in schemas)
    provenance = {
        "schema_version": 1,
        "generator": "tools/generate_web_docs.py",
        "source_hashes": {
            _repo_rel(path): _sha256(path.read_bytes()) for path in sorted(set(source_files))
        },
        "generated_files": sorted([*bundle, "provenance.json"]),
        "counts": {
            "tools": len(tools),
            "schemas": len(schemas),
            "examples": len(content["examples"]),
            "glossary_terms": len(content["glossary"]),
            "failure_guides": len(content["failures"]),
        },
    }
    bundle["provenance.json"] = _json_bytes(provenance)
    return bundle, provenance["counts"]


def _compare_bundle(output: Path, bundle: dict[str, bytes]) -> dict[str, list[str]]:
    actual = (
        {
            path.relative_to(output).as_posix(): path
            for path in output.rglob("*")
            if path.is_file()
        }
        if output.is_dir()
        else {}
    )
    expected_names = set(bundle)
    actual_names = set(actual)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    stale = sorted(
        name
        for name in expected_names & actual_names
        if actual[name].read_bytes() != bundle[name]
    )
    return {"missing": missing, "stale": stale, "extra": extra}


def _write_bundle(output: Path, bundle: dict[str, bytes]) -> None:
    resolved = output.resolve()
    if resolved == REPO_ROOT or REPO_ROOT == resolved.parent:
        raise SystemExit("refusing to replace the repository root or a top-level directory")
    if output.exists():
        shutil.rmtree(output)
    for rel, data in bundle.items():
        destination = output / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    manifest_path = Path(args.manifest)
    schemas_dir = Path(args.schemas)
    content_path = Path(args.content)
    output = Path(args.output)
    bundle, counts = _build_bundle(
        manifest_path=manifest_path,
        schemas_dir=schemas_dir,
        content_path=content_path,
    )

    if args.check:
        drift = _compare_bundle(output, bundle)
        ok = not any(drift.values())
        result = {
            "ok": ok,
            "mode": "check",
            "output": _repo_rel(output),
            "counts": counts,
            **drift,
        }
        print(
            json.dumps(result, sort_keys=True)
            if args.json
            else (
                f"OK: {_repo_rel(output)}"
                if ok
                else f"STALE: {_repo_rel(output)}: {drift}"
            )
        )
        return 0 if ok else 1

    _write_bundle(output, bundle)
    result = {
        "ok": True,
        "mode": "write",
        "output": _repo_rel(output),
        "file_count": len(bundle),
        "counts": counts,
    }
    print(
        json.dumps(result, sort_keys=True)
        if args.json
        else f"Wrote {len(bundle)} files to {_repo_rel(output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
