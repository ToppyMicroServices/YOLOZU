from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ISSUE_TEMPLATE_DIR = ROOT / ".github" / "ISSUE_TEMPLATE"


class FeedbackSupportDocsTests(unittest.TestCase):
    def test_structured_issue_forms_capture_the_shared_intake(self) -> None:
        expected = {
            "bug_report.yml": ("[Bug] ", "bug"),
            "first_run_failure.yml": ("[First run] ", "bug"),
            "integration_request.yml": ("[Integration] ", "enhancement"),
            "evaluation_question.yml": ("[Evaluation] ", "question"),
            "feature_demand.yml": ("[Feature] ", "enhancement"),
        }
        required_ids = {
            "goal",
            "source_framework",
            "source_framework_version",
            "environment",
            "commands",
            "artifacts",
            "impact",
            "aggregate_consent",
            "consent",
        }
        allowed_body_types = {"markdown", "input", "textarea", "dropdown", "checkboxes"}
        allowed_labels = {
            "bug",
            "documentation",
            "duplicate",
            "enhancement",
            "good first issue",
            "help wanted",
            "invalid",
            "question",
            "wontfix",
        }

        for filename, (title_prefix, label) in expected.items():
            with self.subTest(filename=filename):
                path = ISSUE_TEMPLATE_DIR / filename
                self.assertTrue(path.is_file(), f"missing structured issue form: {filename}")
                form = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertIsInstance(form.get("name"), str)
                self.assertTrue(form["name"].strip())
                self.assertIsInstance(form.get("description"), str)
                self.assertTrue(form["description"].strip())
                self.assertEqual(form["title"], title_prefix)
                self.assertIsInstance(form.get("labels"), list)
                self.assertTrue(form["labels"])
                self.assertTrue(all(isinstance(item, str) for item in form["labels"]))
                self.assertTrue(set(form["labels"]).issubset(allowed_labels))
                self.assertIn(label, form["labels"])
                self.assertIsInstance(form.get("body"), list)
                self.assertTrue(form["body"])
                self.assertLessEqual(
                    len(form["body"]),
                    10,
                    f"{filename}: GitHub issue forms allow at most 10 body elements",
                )
                body_types = {item.get("type") for item in form["body"]}
                self.assertTrue(body_types.issubset(allowed_body_types))
                fields = {
                    item.get("id"): item
                    for item in form["body"]
                    if isinstance(item, dict) and item.get("id")
                }
                ids = [
                    item["id"]
                    for item in form["body"]
                    if isinstance(item, dict) and item.get("id")
                ]
                self.assertEqual(len(ids), len(set(ids)), f"{filename}: duplicate field id")
                self.assertTrue(
                    all(re.fullmatch(r"[A-Za-z0-9_-]+", field_id) for field_id in ids),
                    f"{filename}: invalid field id",
                )
                for item in form["body"]:
                    self.assertIsInstance(item, dict)
                    self.assertIn("type", item)
                    self.assertIsInstance(item.get("attributes"), dict)
                    if item["type"] != "markdown":
                        self.assertIsInstance(item["attributes"].get("label"), str)
                        self.assertTrue(item["attributes"]["label"].strip())
                self.assertTrue(required_ids.issubset(fields), f"{filename}: shared intake drifted")
                for field_id in required_ids - {"consent"}:
                    self.assertTrue(
                        fields[field_id].get("validations", {}).get("required"),
                        f"{filename}: {field_id} must be required",
                    )

                aggregate_options = fields["aggregate_consent"]["attributes"]["options"]
                self.assertEqual(len(aggregate_options), 3)
                self.assertTrue(str(aggregate_options[0]).startswith("Yes"))
                self.assertTrue(str(aggregate_options[1]).startswith("Count only"))
                self.assertTrue(str(aggregate_options[2]).startswith("No"))
                consent_options = fields["consent"]["attributes"]["options"]
                self.assertEqual(len(consent_options), 2)
                self.assertTrue(all(option.get("required") for option in consent_options))
                consent_text = " ".join(option["label"] for option in consent_options)
                self.assertIn("no confidential", consent_text)
                self.assertIn("follow-up", consent_text)

                form_text = path.read_text(encoding="utf-8")
                self.assertIn("SECURITY.md", form_text)
                self.assertNotIn("customer name", form_text.lower())
                support = (ROOT / "docs" / "support.md").read_text(encoding="utf-8")
                self.assertIn(f"template={filename}", support)

    def test_issue_chooser_uses_verified_public_routes(self) -> None:
        config = yaml.safe_load(
            (ISSUE_TEMPLATE_DIR / "config.yml").read_text(encoding="utf-8")
        )
        self.assertFalse(config["blank_issues_enabled"])
        urls = {item["url"] for item in config["contact_links"]}
        self.assertIn(
            "https://github.com/ToppyMicroServices/YOLOZU/discussions/categories/q-a",
            urls,
        )
        self.assertIn(
            "https://github.com/ToppyMicroServices/YOLOZU/discussions/categories/ideas",
            urls,
        )
        self.assertIn(
            "https://github.com/ToppyMicroServices/YOLOZU/security/advisories/new",
            urls,
        )

    def test_support_and_monthly_review_stay_linked(self) -> None:
        support = (ROOT / "docs" / "support.md").read_text(encoding="utf-8")
        adoption = (ROOT / "docs" / "adoption" / "README.md").read_text(
            encoding="utf-8"
        )
        monthly_path = (
            ROOT / "docs" / "adoption" / "monthly_feedback_review_template.md"
        )
        monthly = monthly_path.read_text(encoding="utf-8")

        self.assertIn("within **5 business days**", support)
        self.assertIn("not a service-level guarantee", support)
        self.assertIn("Europe/Tallinn", support)
        self.assertIn("Automated acknowledgements do", support)
        self.assertIn("public-citation option", support)
        self.assertIn("Public triage labels", support)
        self.assertIn("monthly_feedback_review_template.md", support)
        self.assertIn("monthly_feedback_review_template.md", adoption)
        self.assertIn("Category | Frequency | Highest non-security impact", monthly)
        self.assertIn("bd create", monthly)
        self.assertIn("Do not create a second item", monthly)
        self.assertIn("bd list --all --limit 0 --json", monthly)
        self.assertIn("Qualifying consented non-security requests", monthly)
        self.assertIn("bd update <id> --append-notes", monthly)
        self.assertIn("bd close <id>", monthly)
        self.assertIn("--parent YOLOZU-ll2", monthly)
        self.assertIn("Security severity is never assessed here", monthly)
        self.assertIn("SECURITY.md", monthly)

    def test_entry_docs_link_structured_support(self) -> None:
        expected = {
            "README.md": "Structured support and feedback",
            "Readme_jp.md": "構造化された support / feedback",
            "Readme_zh.md": "结构化支持、反馈与法务说明",
            "docs/README.md": "Structured support and feedback",
        }
        for relative_path, phrase in expected.items():
            with self.subTest(relative_path=relative_path):
                text = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn(phrase, text)
                self.assertIn("docs/support.md" if relative_path != "docs/README.md" else "support.md", text)


if __name__ == "__main__":
    unittest.main()
