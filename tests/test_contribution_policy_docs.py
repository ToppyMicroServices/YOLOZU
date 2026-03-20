from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestContributionPolicyDocs(unittest.TestCase):
    def test_contributing_mentions_test_policy(self) -> None:
        text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        self.assertIn("## Test policy", text)
        self.assertIn("major new functionality", text)
        self.assertIn("automated tests", text)

    def test_security_crypto_scope_doc_exists(self) -> None:
        text = (ROOT / "docs" / "security_crypto_scope.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("not a cryptography product", text)
        self.assertIn("dedicated, publicly reviewed libraries", text)
        self.assertIn("insecure transport path", text)

    def test_pr_templates_reference_tests_and_docs_sync(self) -> None:
        expected_snippets = [
            "Added or updated automated tests for major new functionality / behavior changes",
            "Manifest/manual were updated when machine-readable tool docs or published operator guidance changed.",
            "If no docs/manual/manifest updates were needed, the PR description explains why.",
        ]
        for relpath in (
            ".github/PULL_REQUEST_TEMPLATE.md",
            ".github/pull_request_template.md",
        ):
            text = (ROOT / relpath).read_text(encoding="utf-8")
            for snippet in expected_snippets:
                self.assertIn(snippet, text, msg=f"{relpath} missing: {snippet}")
