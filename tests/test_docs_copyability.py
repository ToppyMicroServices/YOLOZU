import re
import unittest
from pathlib import Path


PUBLIC_COPYABILITY_PATHS = [
    "README.md",
    "Readme_jp.md",
    "docs/README.md",
    "docs/install.md",
    "docs/cpu_only_dod.md",
    "docs/production_readiness.md",
    "docs/evaluation_protocol_template.md",
    "docs/schema_governance.md",
    "docs/web_docs_plan.md",
    "manual/chapters/01_overview.tex",
    "manual/chapters/04_cli_reference.tex",
    "manual/chapters/05_workflows_eval_export.tex",
    "manual/chapters/11_troubleshooting.tex",
    "manual/chapters/18_manifest_driven_docs.tex",
]


class TestDocsCopyability(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def _read(self, rel_path: str) -> str:
        return (self.repo_root / rel_path).read_text(encoding="utf-8")

    def test_public_entry_docs_avoid_copy_paste_punctuation(self):
        forbidden = {
            "smart double quote": r"[“”]",
            "smart single quote": r"[‘’]",
            "en dash": "–",
            "em dash": "—",
            "concatenated python command": "python3tools",
        }
        hits = []
        for rel_path in PUBLIC_COPYABILITY_PATHS:
            text = self._read(rel_path)
            for label, pattern in forbidden.items():
                if re.search(pattern, text):
                    hits.append(f"{rel_path}: {label}")
        self.assertEqual(hits, [], "copyability punctuation drift:\n" + "\n".join(hits))

    def test_markdown_links_do_not_include_tracking_query_params(self):
        hits = []
        for path in (self.repo_root / "docs").rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            rel_path = path.relative_to(self.repo_root)
            if "utm_source" in text:
                hits.append(f"{rel_path}: utm_source")
            if re.search(r"https://arxiv\.org/abs/[^\s)]+\?", text):
                hits.append(f"{rel_path}: arxiv query parameter")
        self.assertEqual(hits, [], "tracking/query links found:\n" + "\n".join(hits))

    def test_markdown_docs_do_not_leak_visible_latex_commands(self):
        latex_patterns = [
            r"\\cmd\{",
            r"\\path\{",
            r"\\texttt\{",
            r"\\begin\{",
            r"\\end\{",
        ]
        hits = []
        for rel_path in PUBLIC_COPYABILITY_PATHS:
            if not rel_path.endswith(".md"):
                continue
            text = self._read(rel_path)
            for pattern in latex_patterns:
                if re.search(pattern, text):
                    hits.append(f"{rel_path}: {pattern}")
        self.assertEqual(hits, [], "visible LaTeX commands in Markdown:\n" + "\n".join(hits))

    def test_manual_tables_stay_below_dense_column_threshold(self):
        hits = []
        table_re = re.compile(r"\\begin\{(?:tabular|longtable)\}\{([^}]*)\}")
        for path in (self.repo_root / "manual" / "chapters").glob("*.tex"):
            text = path.read_text(encoding="utf-8")
            for match in table_re.finditer(text):
                spec = match.group(1)
                column_count = len(re.findall(r"[lcr]|p\{", spec))
                if column_count > 5:
                    rel_path = path.relative_to(self.repo_root)
                    hits.append(f"{rel_path}: {column_count} columns")
        self.assertEqual(hits, [], "dense manual tables found:\n" + "\n".join(hits))


if __name__ == "__main__":
    unittest.main()
