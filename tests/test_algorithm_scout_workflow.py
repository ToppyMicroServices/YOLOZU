from pathlib import Path
from unittest import TestCase, main


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "algorithm_scout.yml"


class TestAlgorithmScoutWorkflow(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_schedule_permissions_concurrency_and_time_budget_are_exact(self) -> None:
        self.assertIn("- cron: '0 0 * * 1'", self.text)
        self.assertIn("workflow_dispatch:", self.text)
        self.assertIn("group: algorithm-scout", self.text)
        self.assertIn("cancel-in-progress: false", self.text)
        self.assertIn("timeout-minutes: 15", self.text)
        self.assertIn("contents: read", self.text)
        self.assertIn("actions: read", self.text)
        self.assertIn("issues: write", self.text)
        self.assertNotIn("artifacts:", self.text)

    def test_checked_out_cli_installs_required_runtime_dependencies(self) -> None:
        self.assertIn("run: python3 -m pip install .", self.text)
        self.assertNotIn("pip install --no-deps .", self.text)

    def test_collection_invocation_and_exit_control_flow_are_bounded(self) -> None:
        invocation = '''yolozu scout-algorithms \\
            --sources docs/algorithm_intake/sources.json \\
            --output-dir reports/algorithm_scout \\
            --collection-date "${COLLECTION_DATE}" \\
            --trigger "${EVENT_NAME}"'''
        self.assertIn(invocation, self.text)
        self.assertIn('echo "exit_code=${code}" >> "${GITHUB_OUTPUT}"', self.text)
        self.assertIn("steps.scout.outputs.exit_code == '0'", self.text)
        self.assertIn("steps.scout.outputs.exit_code == '3'", self.text)
        self.assertIn("steps.stage.outcome == 'success'", self.text)
        self.assertIn('exit "${code}"', self.text)

    def test_only_safe_reports_are_uploaded_and_failures_are_deduplicated(self) -> None:
        self.assertIn("prepare_scout_workflow_artifact", self.text)
        self.assertIn("retention-days: 30", self.text)
        self.assertIn("if-no-files-found: error", self.text)
        self.assertEqual(self.text.count("actions/upload-artifact@"), 1)
        upload_block = self.text.split("- name: Upload only the validated report artifact", 1)[1].split(
            "- name: Update the deduplicated failure issue", 1
        )[0]
        self.assertNotIn("reports/algorithm_scout", upload_block)
        self.assertNotIn("runner.temp", upload_block)
        self.assertIn("Algorithm scout scheduled run failed", self.text)
        self.assertIn("select(.title == $title)", self.text)
        self.assertIn("This bounded summary contains no fetched text", self.text)

    def test_prior_artifact_lookup_records_missed_without_backfill(self) -> None:
        self.assertIn("actions/artifacts", self.text)
        self.assertIn("expired == false", self.text)
        self.assertIn("missed_arg=${previous_date}", self.text)
        self.assertIn('--missed-collection-date "${MISSED_COLLECTION_DATE}"', self.text)


if __name__ == "__main__":
    main()
