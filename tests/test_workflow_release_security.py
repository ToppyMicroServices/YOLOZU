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
