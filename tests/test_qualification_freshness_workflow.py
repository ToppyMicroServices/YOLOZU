from pathlib import Path
from unittest import TestCase, main


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "qualification_freshness.yml"


class TestQualificationFreshnessWorkflow(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_schedule_permissions_and_public_scope_are_exact(self) -> None:
        self.assertIn("- cron: '30 0 * * 1'", self.text)
        self.assertIn("timeout-minutes: 10", self.text)
        self.assertIn("group: qualification-freshness", self.text)
        self.assertIn("cancel-in-progress: false", self.text)
        self.assertIn("contents: read", self.text)
        self.assertIn("actions: read", self.text)
        self.assertIn("issues: write", self.text)
        self.assertNotIn("--evidence-root", self.text)
        self.assertIn("repository_owned_public_ids_only", self.text)
        self.assertIn("upload_eligible", self.text)

    def test_exit_artifact_retention_and_no_backfill_are_explicit(self) -> None:
        self.assertIn("yolozu check-qualification-freshness", self.text)
        self.assertIn('echo "exit_code=${code}" >> "${GITHUB_OUTPUT}"', self.text)
        self.assertIn("steps.freshness.outputs.exit_code == '0'", self.text)
        self.assertIn("steps.freshness.outputs.exit_code == '3'", self.text)
        self.assertIn("retention-days: 30", self.text)
        self.assertIn("actions/artifacts", self.text)
        self.assertIn("missed_arg=${previous_date}", self.text)
        self.assertIn('--missed-run-date "${MISSED_RUN_DATE}"', self.text)

    def test_issue_is_exact_title_deduplicated_and_bounded_by_renderer(self) -> None:
        self.assertIn("Qualification evidence freshness action required", self.text)
        self.assertIn("select(.title == $title)", self.text)
        self.assertIn("render_qualification_freshness_issue_body", self.text)
        self.assertEqual(self.text.count("actions/upload-artifact@"), 1)
        self.assertIn('exit "${code}"', self.text)


if __name__ == "__main__":
    main()
