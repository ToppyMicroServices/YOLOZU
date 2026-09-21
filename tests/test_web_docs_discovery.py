from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from unittest import TestCase, main, mock
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree


class _Page(HTMLParser):
    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[dict[str, str | None]] = []
        self.metadata: list[dict[str, str | None]] = []
        self.scripts: list[tuple[dict[str, str | None], str]] = []
        self.titles: list[str] = []
        self.headlines: list[str] = []
        self.visible: list[str] = []
        self._hidden = 0
        self._headline: list[str] | None = None
        self._title: list[str] | None = None
        self._script: tuple[dict[str, str | None], list[str]] | None = None
        self.feed(source)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "link":
            self.links.append(attributes)
        if tag == "meta":
            self.metadata.append(attributes)
        if tag in {"head", "script", "style"}:
            self._hidden += 1
        if tag == "h1":
            self._headline = []
        if tag == "title":
            self._title = []
        if tag == "script":
            self._script = (attributes, [])

    def handle_endtag(self, tag: str) -> None:
        if tag in {"head", "script", "style"}:
            self._hidden -= 1
        if tag == "h1" and self._headline is not None:
            self.headlines.append("".join(self._headline).strip())
            self._headline = None
        if tag == "title" and self._title is not None:
            self.titles.append("".join(self._title).strip())
            self._title = None
        if tag == "script" and self._script is not None:
            attributes, text = self._script
            self.scripts.append((attributes, "".join(text)))
            self._script = None

    def handle_data(self, text: str) -> None:
        if self._script is not None:
            self._script[1].append(text)
        if self._headline is not None:
            self._headline.append(text)
        if self._title is not None:
            self._title.append(text)
        if not self._hidden:
            self.visible.append(text)

    @property
    def text(self) -> str:
        return " ".join(" ".join(self.visible).split())


class TestWebDocsDiscovery(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.content_path = cls.repo_root / "docs" / "web_docs_content.json"
        cls.content = json.loads(cls.content_path.read_text(encoding="utf-8"))
        spec = importlib.util.spec_from_file_location(
            "yolozu_test_web_docs_discovery",
            cls.repo_root / "tools" / "generate_web_docs.py",
        )
        if spec is None or spec.loader is None:
            raise AssertionError("web docs generator is not importable")
        cls.generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.generator)
        cls.bundle = cls._build_bundle()

    @classmethod
    def _build_bundle(cls) -> dict[str, bytes]:
        bundle, _ = cls.generator._build_bundle(
            manifest_path=cls.repo_root / "tools" / "manifest.json",
            schemas_dir=cls.repo_root / "docs" / "schemas",
            content_path=cls.content_path,
        )
        return bundle

    def _text(self, name: str) -> str:
        return self.bundle[name].decode("utf-8")

    def _assert_visible_and_markdown(
        self, expected: str, page: _Page, markdown: str
    ) -> None:
        self.assertIn(" ".join(expected.split()), page.text)
        self.assertIn(expected, markdown)

    def test_agent_guide_html_and_markdown_share_curated_content(self) -> None:
        guide = self.content["agent_guide"]
        page = _Page(self._text("agents.html"))
        markdown = self._text("agents.md")
        for value in [guide["title"], guide["summary"], *guide["use_when"], *guide["not_for"]]:
            with self.subTest(value=value):
                self._assert_visible_and_markdown(value, page, markdown)
        for workflow in guide["workflows"]:
            for key in ("title", "description", "expected"):
                with self.subTest(workflow=workflow["id"], key=key):
                    self._assert_visible_and_markdown(workflow[key], page, markdown)
            for command in workflow["commands"]:
                with self.subTest(command=command):
                    self._assert_visible_and_markdown(command, page, markdown)
        for prompt in guide["prompts"]:
            for key in ("prompt", "expected"):
                with self.subTest(prompt=prompt["prompt"], key=key):
                    self._assert_visible_and_markdown(prompt[key], page, markdown)
        for resource in guide["resources"]:
            for key in ("title", "description"):
                with self.subTest(resource=resource["id"], key=key):
                    self._assert_visible_and_markdown(resource[key], page, markdown)
            self.assertIn(resource["source"], markdown)
            self.assertIn(resource["source"], self._text("agents.html"))

    def test_quickstart_markdown_keeps_exact_commands_and_limitations(self) -> None:
        page = _Page(self._text("start.html"))
        markdown = self._text("start.md")
        for step in self.content["tutorial"]["thirty_minute"]:
            for text in (step["title"], step["expected"], *step["commands"]):
                with self.subTest(text=text):
                    self._assert_visible_and_markdown(text, page, markdown)
        api = self.content["python_api"]
        for key in ("title", "description", "example", "expected"):
            with self.subTest(api=key):
                self._assert_visible_and_markdown(api[key], page, markdown)
        fallback = self.content["tutorial"]["dry_run_fallback"]
        for key in ("title", "description", "command"):
            with self.subTest(fallback=key):
                self._assert_visible_and_markdown(fallback[key], page, markdown)

    def test_llms_is_compact_and_links_to_resolvable_https_entrypoints(self) -> None:
        llms = self._text("llms.txt")
        self.assertTrue(llms.startswith("# YOLOZU\n"))
        self.assertLess(len(llms.encode("utf-8")), 12_000)
        urls = re.findall(r"\[[^\]]+\]\(([^\s)]+)\)", llms)
        self.assertTrue(urls)
        base = self.content["site"]["canonical_base"].rstrip("/") + "/"
        for expected in ("agents.md", "start.md", "capabilities.json"):
            self.assertIn(base + expected, urls)
        for url in urls:
            with self.subTest(url=url):
                parsed = urlsplit(url)
                self.assertEqual(parsed.scheme, "https")
                self.assertTrue(parsed.netloc)
                self.assertIsNone(parsed.username)
                self.assertIsNone(parsed.password)
                if url.startswith(base):
                    relative = urlsplit(url[len(base):]).path or "index.html"
                    self.assertIn(relative, self.bundle)
        raw_base = self.content["site"]["raw_repository_base"].rstrip("/") + "/"
        for resource in self.content["agent_guide"]["resources"]:
            self.assertIn(raw_base + resource["source"], urls)

    def test_padded_base_urls_are_normalized_before_link_generation(self) -> None:
        content = copy.deepcopy(self.content)
        for key in ("canonical_base", "repository_base", "raw_repository_base"):
            content["site"][key] = " " + content["site"][key] + " "
        validated = self.generator._validate_content(content)
        self.assertEqual(validated["site"], self.content["site"])
        self.assertEqual(
            self.generator._render_llms(validated),
            self.generator._render_llms(self.content),
        )

    def test_capabilities_use_checkout_version_exact_surfaces_and_source_hashes(self) -> None:
        capabilities = json.loads(self._text("capabilities.json"))
        self.assertEqual(capabilities["schema_version"], 1)
        self.assertEqual(capabilities["project"], "YOLOZU")
        init = (self.repo_root / "yolozu" / "__init__.py").read_text(encoding="utf-8")
        version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', init, re.MULTILINE)
        self.assertIsNotNone(version)
        self.assertEqual(capabilities["docs_version"], version.group(1))
        scope = capabilities["version_scope"].lower()
        self.assertIn("checkout", scope)
        self.assertIn("installed", scope)
        self.assertIn("pypi", scope)
        reference_path = self.repo_root / "docs" / "generated" / "mcp_actions_tool_reference.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        self.assertEqual(capabilities["surfaces"], reference["surfaces"])
        guide = self.content["agent_guide"]
        self.assertEqual(capabilities["use_when"], guide["use_when"])
        self.assertEqual(capabilities["limitations"], guide["not_for"])
        by_id = {entry["id"]: entry for entry in capabilities["resources"]}
        self.assertEqual(len(by_id), len(capabilities["resources"]))
        self.assertEqual(set(by_id), {entry["id"] for entry in guide["resources"]})
        provenance = json.loads(self._text("provenance.json"))["source_hashes"]
        raw_base = self.content["site"]["raw_repository_base"].rstrip("/") + "/"
        for resource in guide["resources"]:
            with self.subTest(resource=resource["id"]):
                entry = by_id[resource["id"]]
                for field in ("title", "description"):
                    self.assertEqual(entry[field], resource[field])
                self.assertEqual(entry["url"], raw_base + resource["source"])
                digest = hashlib.sha256((self.repo_root / resource["source"]).read_bytes()).hexdigest()
                self.assertEqual(entry["source_sha256"], digest)
                self.assertEqual(provenance[resource["source"]], digest)
        base = self.content["site"]["canonical_base"].rstrip("/") + "/"
        self.assertEqual(
            capabilities["entrypoints"],
            {
                "overview": base,
                "quickstart": base + "start.md",
                "agent_guide": base + "agents.md",
                "llms": base + "llms.txt",
            },
        )

    def test_html_discovery_links_and_jsonld_match_page_metadata(self) -> None:
        base = self.content["site"]["canonical_base"].rstrip("/") + "/"
        for name in sorted(self.bundle):
            if not name.endswith(".html"):
                continue
            with self.subTest(page=name):
                page = _Page(self._text(name))
                describedby = [link for link in page.links if link.get("rel") == "describedby"]
                self.assertEqual(len(describedby), 1)
                self.assertEqual(urljoin(base + name, describedby[0]["href"]), base + "llms.txt")
                if name in {"start.html", "agents.html"}:
                    alternate = [
                        link for link in page.links
                        if link.get("rel") == "alternate" and link.get("type") == "text/markdown"
                    ]
                    self.assertEqual(len(alternate), 1)
                    self.assertEqual(
                        urljoin(base + name, alternate[0]["href"]),
                        base + name.removesuffix(".html") + ".md",
                    )
                canonical = [link["href"] for link in page.links if link.get("rel") == "canonical"]
                self.assertEqual(canonical, [base if name == "index.html" else base + name])
                scripts = [text for attrs, text in page.scripts if attrs.get("type") == "application/ld+json"]
                self.assertEqual(len(scripts), 1)
                data = json.loads(scripts[0])
                self.assertEqual(data["@context"], "https://schema.org")
                self.assertEqual(data["@type"], "TechArticle")
                self.assertEqual(page.titles, [data["headline"] + " — YOLOZU Docs"])
                self.assertEqual(data["url"], canonical[0])
                description = [meta["content"] for meta in page.metadata if meta.get("name") == "description"]
                self.assertEqual(description, [data["description"]])

    def test_sitemap_contains_each_canonical_html_page_once(self) -> None:
        root = ElementTree.fromstring(self._text("sitemap.xml"))
        namespace = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
        self.assertEqual(root.tag, namespace + "urlset")
        locations = [element.text for element in root.findall(f"{namespace}url/{namespace}loc")]
        base = self.content["site"]["canonical_base"].rstrip("/") + "/"
        expected = {
            base if name == "index.html" else base + name
            for name in self.bundle if name.endswith(".html")
        }
        self.assertEqual(len(expected), 8)
        self.assertEqual(len(locations), len(expected))
        self.assertEqual(set(locations), expected)

    def test_jsonld_escapes_html_script_terminators_without_changing_values(self) -> None:
        content = copy.deepcopy(self.content)
        attack = '</script><script>alert("discovery")</script>'
        content["agent_guide"]["title"] = attack
        content["agent_guide"]["summary"] = "Example " + attack
        original = self.generator._load_json

        def load_json(path: Path, *, label: str):
            if path == self.content_path:
                return content
            return original(path, label=label)

        with mock.patch.object(self.generator, "_load_json", side_effect=load_json):
            bundle = self._build_bundle()
        page = _Page(bundle["agents.html"].decode("utf-8"))
        structured = [text for attrs, text in page.scripts if attrs.get("type") == "application/ld+json"]
        self.assertEqual(len(structured), 1)
        self.assertNotIn("</script", structured[0].lower())
        data = json.loads(structured[0])
        self.assertEqual(data["headline"], attack)
        self.assertEqual(data["description"], "Example " + attack)
        self.assertEqual(page.headlines, [attack])
        self.assertFalse(any('alert("discovery")' in text for attrs, text in page.scripts if attrs.get("type") != "application/ld+json"))

    def test_base_urls_reject_unsafe_schemes_authority_query_and_fragment(self) -> None:
        for field in ("canonical_base", "repository_base", "raw_repository_base"):
            for value in (
                "http://example.com/docs/",
                "//example.com/docs/",
                "javascript:alert(1)",
                "https://user:pass@example.com/docs/",
                "https://example.com/docs/?branch=other",
                "https://example.com/docs/#fragment",
                "https://example.com/a/../docs/",
                "https://example.com/docs/%0a",
            ):
                with self.subTest(field=field, value=value):
                    content = copy.deepcopy(self.content)
                    content["site"][field] = value
                    with self.assertRaisesRegex(SystemExit, re.escape("site." + field)):
                        self.generator._validate_content(content)

    def test_agent_resources_reject_unsafe_sources(self) -> None:
        for value in (
            "../README.md",
            "docs/../README.md",
            "/etc/passwd",
            "https://example.com/install.md",
            r"docs\install.md",
            "docs/install.md?version=old",
            "docs/install.md#fragment",
        ):
            with self.subTest(source=value):
                content = copy.deepcopy(self.content)
                content["agent_guide"]["resources"][0]["source"] = value
                with self.assertRaisesRegex(SystemExit, "agent_guide"):
                    self.generator._validate_content(content)

    def test_agent_resources_reject_duplicate_ids_and_missing_mcp_reference(self) -> None:
        content = copy.deepcopy(self.content)
        content["agent_guide"]["resources"][1]["id"] = content["agent_guide"]["resources"][0]["id"]
        with self.assertRaisesRegex(SystemExit, "agent_guide"):
            self.generator._validate_content(content)
        content = copy.deepcopy(self.content)
        content["agent_guide"]["resources"] = [
            resource for resource in content["agent_guide"]["resources"]
            if resource["id"] != "mcp_reference"
        ]
        with self.assertRaisesRegex(SystemExit, "mcp"):
            self.generator._validate_content(content)

    def test_agent_text_rejects_control_characters(self) -> None:
        paths = (
            ("title",),
            ("summary",),
            ("use_when", 0),
            ("not_for", 0),
            ("workflows", 0, "description"),
            ("workflows", 0, "commands", 0),
            ("workflows", 0, "expected"),
            ("prompts", 0, "prompt"),
            ("prompts", 0, "expected"),
            ("resources", 0, "description"),
        )
        for path in paths:
            for control in ("\x00", "\x1b", "\x7f"):
                with self.subTest(path=path, control=repr(control)):
                    content = copy.deepcopy(self.content)
                    target = content["agent_guide"]
                    for key in path[:-1]:
                        target = target[key]
                    target[path[-1]] += control
                    with self.assertRaisesRegex(SystemExit, "agent_guide"):
                        self.generator._validate_content(content)


if __name__ == "__main__":
    main()
