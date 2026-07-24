from __future__ import annotations

import re
import unittest
from pathlib import Path


DISALLOWED_PATTERNS = {
    "bare contract-first": re.compile(
        r"(?<!interface-)(?<!interface )\bcontract[- ]first\b",
        re.IGNORECASE,
    ),
    "bare contract metadata": re.compile(
        r"(?<!interface )\bcontract metadata\b",
        re.IGNORECASE,
    ),
    "bare contract boundary heading": re.compile(
        r"^#{1,6}\s+contract boundary\b",
        re.IGNORECASE | re.MULTILINE,
    ),
    "bare contract-oriented": re.compile(
        r"(?<!interface-)\bcontract-oriented\b",
        re.IGNORECASE,
    ),
    "bare contract/eval": re.compile(
        r"(?<!interface-)\bcontract/eval\b",
        re.IGNORECASE,
    ),
    "bare contract surface": re.compile(
        r"(?<!interface )\bcontract surface\b",
        re.IGNORECASE,
    ),
    "ambiguous version-the-contract wording": re.compile(
        r"\bversion the contract\b",
        re.IGNORECASE,
    ),
    "ambiguous non-contract wording": re.compile(
        r"\bnon-contract experimental\b",
        re.IGNORECASE,
    ),
    "bare common interface category": re.compile(
        r"(?<!interface )\b(?:artifact|output|projection|preprocessing|"
        r"evaluation|bbox|status|response|metadata|intake) contracts?\b",
        re.IGNORECASE,
    ),
    "bare contract gates": re.compile(
        r"(?<!interface )\bcontract gates\b",
        re.IGNORECASE,
    ),
    "bare contract reference": re.compile(
        r"(?<!interface )\bcontract reference\b",
        re.IGNORECASE,
    ),
    "bare tooling-or-harness contracts": re.compile(
        r"\b(?:tooling|harness) and contracts\b",
        re.IGNORECASE,
    ),
}

# These are intentionally not rewritten by this terminology audit.
# They are named training/data surfaces, repository identifiers, or CLI fields.
ALLOWED_STANDALONE_TERMS = {
    "Run Contract": "canonical training-run artifact title",
    "Dataset Contract": "canonical dataset format title",
    "run_contract": "configuration and schema identifier",
    "contracts.consumes": "manifest field name",
    "contracts.produces": "manifest field name",
    "--contract": "existing CLI flag",
}


class TestInterfaceContractTerminology(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def _public_text_files(self) -> list[Path]:
        files = [
            self.repo_root / "README.md",
            self.repo_root / "Readme_jp.md",
            self.repo_root / "tools" / "manifest.json",
            self.repo_root / "yolozu" / "data" / "manifest" / "tools_manifest.json",
        ]
        files.extend((self.repo_root / "docs").rglob("*.md"))
        files.extend((self.repo_root / "manual" / "chapters").rglob("*.tex"))
        return sorted(set(files))

    def test_ambiguous_software_contract_wording_is_absent(self) -> None:
        failures: list[str] = []
        for path in self._public_text_files():
            text = path.read_text(encoding="utf-8")
            for label, pattern in DISALLOWED_PATTERNS.items():
                for match in pattern.finditer(text):
                    line = text.count("\n", 0, match.start()) + 1
                    failures.append(
                        f"{path.relative_to(self.repo_root)}:{line}: {label}: "
                        f"{match.group(0)!r}"
                    )
        self.assertEqual(failures, [], "\n".join(failures))

    def test_allowlisted_terms_remain_explicit_and_documented(self) -> None:
        corpus = "\n".join(
            path.read_text(encoding="utf-8") for path in self._public_text_files()
        )
        for term, reason in ALLOWED_STANDALONE_TERMS.items():
            with self.subTest(term=term, reason=reason):
                self.assertIn(term, corpus)


if __name__ == "__main__":
    unittest.main()
