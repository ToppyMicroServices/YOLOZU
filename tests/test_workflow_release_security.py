import json
import unittest
from pathlib import Path


class TestWorkflowReleaseSecurity(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_publish_workflow_requires_manual_version_and_changelog_alignment(self):
        publish = (self.repo_root / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", publish)
        self.assertIn("expected_version:", publish)
        self.assertIn("release_tag:", publish)
        self.assertIn("macOS wheel build check", publish)
        self.assertIn("python -m build --wheel", publish)
        self.assertIn("Validate version and changelog alignment", publish)
        self.assertIn("CHANGELOG.md", publish)
        self.assertIn("EXPECTED_VERSION_INPUT", publish)
        self.assertIn("wheel/manual version mismatch", publish)
        self.assertIn("Verify published release on PyPI", publish)
        self.assertIn('https://pypi.org/pypi/yolozu/{version}/json', publish)
        self.assertIn('required_types = {"bdist_wheel", "sdist"}', publish)

    def test_manual_doi_uses_one_automatic_trigger_and_idempotency_guard(self):
        release_tool = (self.repo_root / "tools" / "release.py").read_text(encoding="utf-8")
        workflow = (self.repo_root / ".github" / "workflows" / "manual_doi.yml").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("zenodo_workflow_dispatch", release_tool)
        self.assertIn("zenodo_manual_doi_via_release_event", release_tool)
        self.assertIn("types: [published]", workflow)
        self.assertIn("concurrency:", workflow)
        self.assertIn("cancel-in-progress: false", workflow)
        self.assertIn("collect_record_search_pages(", workflow)
        self.assertIn("&page={page}", workflow)
        self.assertIn("find_matching_record(records, version)", workflow)
        self.assertIn('state = "already_published"', workflow)
        self.assertIn("create_first_deposition:", workflow)
        self.assertIn('create_first_deposition}" != "true"', workflow)
        self.assertIn("Missing vars.YOLOZU_MANUAL_CONCEPTRECID", workflow)
        self.assertLess(
            workflow.index("find_matching_record(records, version)"),
            workflow.index("actions/newversion"),
        )

    def test_manual_doi_version_guard_matches_normalized_versions(self):
        from tools.manual_doi_guard import (
            collect_record_search_pages,
            find_matching_record,
            latest_record_id,
            normalize_manual_version,
        )

        records = [
            {"id": 22, "metadata": {"version": "v2.0.0"}},
            {"id": 21, "metadata": {"version": "1.9.0"}},
        ]
        self.assertEqual(normalize_manual_version("v2.0.0"), "2.0.0")
        self.assertEqual(find_matching_record(records, "2.0.0"), records[0])
        self.assertEqual(find_matching_record(records, "v1.9.0"), records[1])
        self.assertIsNone(find_matching_record(records, "1.8.0"))
        self.assertEqual(latest_record_id(records), "22")

        pages = {
            1: {
                "hits": {
                    "hits": [{"id": number, "metadata": {"version": f"1.0.{number}"}} for number in range(1, 101)],
                    "total": 101,
                }
            },
            2: {
                "hits": {
                    "hits": [{"id": 101, "metadata": {"version": "2.0.0"}}],
                    "total": 101,
                }
            },
        }
        calls = []

        def fetch_page(page, size):
            calls.append((page, size))
            return pages[page]

        paginated = collect_record_search_pages(fetch_page)
        self.assertEqual(len(paginated), 101)
        self.assertEqual(find_matching_record(paginated, "2.0.0"), pages[2]["hits"]["hits"][0])
        self.assertEqual(calls, [(1, 100), (2, 100)])

    def test_manual_doi_version_guard_fails_closed_on_incomplete_pagination(self):
        from tools.manual_doi_guard import collect_record_search_pages

        def fetch_page(page, _size):
            if page == 1:
                return {
                    "hits": {
                        "hits": [{"id": number} for number in range(1, 101)],
                        "total": {"value": 101},
                    }
                }
            return {"hits": {"hits": [], "total": {"value": 101}}}

        with self.assertRaisesRegex(ValueError, "before every published record"):
            collect_record_search_pages(fetch_page)

    def test_container_workflow_runs_on_tag_or_manual_only(self):
        container = (self.repo_root / ".github" / "workflows" / "container.yml").read_text(encoding="utf-8")
        self.assertNotIn("pull_request:", container)
        self.assertIn("release_tag:", container)
        self.assertNotIn("branches:", container)
        self.assertIn("deploy/docker/Dockerfile", container)
        self.assertIn("deploy/runpod/Dockerfile", container)
        self.assertIn("push:", container)
        self.assertIn("NGC_REGISTRY: nvcr.io", container)
        self.assertIn("NGC_NAMESPACE: yolozu", container)
        self.assertIn("github.event.inputs.release_tag", container)
        self.assertIn("nvcr.io", container)
        self.assertIn("github.ref_type == 'tag' || github.event_name == 'workflow_dispatch'", container)
        self.assertIn("Mirror to NVIDIA NGC (optional)", container)
        self.assertIn("continue-on-error: true", container)
        self.assertIn("steps.tags.outputs.ghcr_tags", container)
        self.assertIn("steps.tags.outputs.ngc_tags", container)

    def test_scorecard_runs_on_main_and_codeql_covers_pr_commits(self):
        scorecard = (self.repo_root / ".github" / "workflows" / "scorecard.yml").read_text(encoding="utf-8")
        codeql = (self.repo_root / ".github" / "workflows" / "codeql.yml").read_text(encoding="utf-8")
        self.assertNotIn("pull_request:", scorecard)
        self.assertIn("branches:", scorecard)
        self.assertIn("main", scorecard)
        self.assertIn("actions: read", scorecard)
        self.assertIn("contents: read", scorecard)
        self.assertIn("security-events: write", scorecard)
        self.assertIn("pull_request:", codeql)
        self.assertIn("branches:", codeql)
        self.assertIn("main", codeql)

    def test_default_ci_keeps_pr_lightweight_and_main_full(self):
        ci = (self.repo_root / ".github" / "workflows" / "build_and_test.yml").read_text(encoding="utf-8")
        self.assertIn("pull_request:", ci)
        self.assertIn("branches:\n      - main", ci)
        self.assertIn("github.event_name == 'push' && github.ref == 'refs/heads/main'", ci)

    def test_expensive_gpu_and_full_sweep_workflows_are_manual_only(self):
        workflow_names = [
            "gpu_smoke_machine.yml",
            "gpu_practical_suite_machine.yml",
            "gpu_zisn_pipeline.yml",
            "pytest_gpu_machine.yml",
            "reference_adapter_full.yml",
            "cflite_batch.yml",
            "Debug4TensorRT",
        ]
        for name in workflow_names:
            with self.subTest(workflow=name):
                workflow = (self.repo_root / ".github" / "workflows" / name).read_text(encoding="utf-8")
                self.assertIn("workflow_dispatch:", workflow)
                self.assertNotIn("pull_request:", workflow)
                self.assertNotIn("\n  push:", workflow)
                self.assertNotIn("\n  schedule:", workflow)

    def test_reference_adapter_full_workflow_uses_matching_full_baseline(self):
        workflow = (
            self.repo_root / ".github" / "workflows" / "reference_adapter_full.yml"
        ).read_text(encoding="utf-8")
        baseline_path = (
            self.repo_root
            / "baselines"
            / "reference_adapter"
            / "rtdetr_pose"
            / "torch"
            / "cpu"
            / "v1"
            / "full.json"
        )
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        self.assertIn("runs-on: ubuntu-24.04", workflow)
        self.assertIn('default: "baselines/reference_adapter"', workflow)
        self.assertIn('default: "v1"', workflow)
        self.assertIn("--baseline-layout", workflow)
        self.assertIn("matrix", workflow)
        self.assertIn("--profile", workflow)
        self.assertIn("full", workflow)
        self.assertIn("--repro-policy", workflow)
        self.assertIn("strict", workflow)
        self.assertIn("COMMON_ARGS=(", workflow)
        self.assertIn('"${COMMON_ARGS[@]}"', workflow)
        self.assertIn("--write-baseline", workflow)
        self.assertIn("reference_adapter_regression_full_baseline_write.json", workflow)
        self.assertIn("reference_adapter_regression_full_ci.json", workflow)
        self.assertIn("if: ${{ always() }}", workflow)
        self.assertIn("rtdetr_pose/torch/cpu/${{ inputs.baseline_version || 'v1' }}/full.json", workflow)
        self.assertEqual(baseline.get("profile"), "full")
        self.assertEqual(baseline.get("baseline_layout"), "matrix")
        self.assertEqual((baseline.get("baseline_meta") or {}).get("repro_policy"), "strict")

    def test_workflow_only_changes_still_run_release_and_security_regressions(self):
        ci = (self.repo_root / ".github" / "workflows" / "build_and_test.yml").read_text(encoding="utf-8")
        self.assertIn("workflows fast path (release/security regression)", ci)
        self.assertIn("tests.test_release_readiness_docs", ci)
        self.assertIn("tests.test_workflow_release_security", ci)

    def test_full_ci_checks_generated_cli_and_ssot_coverage(self):
        ci = (self.repo_root / ".github" / "workflows" / "build_and_test.yml").read_text(encoding="utf-8")
        self.assertIn("tests/test_generated_cli_reference.py", ci)
        self.assertIn("tests/test_ssot_capability_coverage.py", ci)

    def test_scorecard_governance_tracks_sast_and_ci_history_findings(self):
        governance = (self.repo_root / "docs" / "security_scorecard_governance.md").read_text(encoding="utf-8")
        audit_tool = (self.repo_root / "tools" / "check_repo_governance.py").read_text(encoding="utf-8")
        for finding in ["SASTID", "CITestsID"]:
            with self.subTest(finding=finding):
                self.assertIn(f"### `{finding}`", governance)
                self.assertIn(f'"id": "{finding}"', audit_tool)
        self.assertIn("actions: read", governance)
        self.assertIn("torch>=2.10.0", governance)

    def test_optional_torch_bounds_start_at_patched_envelope(self):
        pyproject = (self.repo_root / "pyproject.toml").read_text(encoding="utf-8")
        requirements_test = (self.repo_root / "requirements-test.txt").read_text(encoding="utf-8")
        osv_config = (self.repo_root / "osv-scanner.toml").read_text(encoding="utf-8")

        self.assertNotIn("torch>=2.8.0", pyproject)
        self.assertNotIn("torchvision>=0.23.0", pyproject)
        self.assertIn("torch>=2.10.0", pyproject)
        self.assertIn("torchvision>=0.25.0", pyproject)
        self.assertIn("torch>=2.10.0", requirements_test)
        self.assertIn("GHSA-rrmf-rvhw-rf47", osv_config)
        self.assertIn("PYSEC-2026-139", osv_config)

    def test_container_bootstrap_locks_keep_incident_fix_versions(self):
        runtime_lock = (self.repo_root / "requirements-locks" / "requirements-runtime.lock").read_text(encoding="utf-8")
        pose_lock = (
            self.repo_root / "requirements-locks" / "requirements-rtdetr-pose-image-extra.lock"
        ).read_text(encoding="utf-8")
        self.assertIn('numpy==2.2.6 ; python_version < "3.14"', runtime_lock)
        self.assertIn('numpy==2.4.4 ; python_version >= "3.14"', runtime_lock)
        self.assertIn("cuda-python==12.9.4", pose_lock)


if __name__ == "__main__":
    unittest.main()
