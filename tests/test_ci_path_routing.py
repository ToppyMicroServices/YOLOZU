"""Exercise the pinned paths-filter bundle, not a substitute glob matcher.

Set YOLOZU_PATHS_FILTER_ENTRYPOINT to its dist/index.js to run the behavioral
tests. The workflow regression job checks out that exact action revision.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github/workflows/build_and_test.yml"


def action_filters():
    result = []
    for step in re.split(r"^      - ", CI.read_text(encoding="utf-8"), flags=re.M):
        if "uses: dorny/paths-filter@" not in step:
            continue
        block = re.search(r"          filters: \|\n((?:            .+\n)+)", step)
        if block is None:
            raise AssertionError("Keep inline filter YAML available to routing tests")
        quantifier = re.search(r"predicate-quantifier: (\w+)", step)
        result.append(
            (textwrap.dedent(block[1]), quantifier[1] if quantifier else "some")
        )
    if not result:
        raise AssertionError("No paths-filter steps found")
    return result


class TestCiPathRoutingStructure(unittest.TestCase):
    def test_runtime_exclusions_use_every_without_changing_other_filters(self):
        filters = action_filters()
        self.assertEqual(len(filters), 2)
        runtime, surfaces = filters
        self.assertIn("full_ci:", runtime[0])
        self.assertEqual(runtime[1], "every")
        self.assertIn("docs_mcp_ci:", surfaces[0])
        self.assertEqual(surfaces[1], "some")

    def test_workflow_tests_run_for_mixed_changes_and_use_the_same_action_pin(self):
        ci = CI.read_text(encoding="utf-8")
        job = ci.split("  workflows_meta:")[1].split("  manual_build:")[0]
        self.assertIn("if: ${{ needs.changes.outputs.workflows_ci == 'true' }}", job)
        self.assertIn("tests.test_ci_path_routing", job)
        self.assertIn("YOLOZU_PATHS_FILTER_ENTRYPOINT:", job)
        pins = set(re.findall(r"uses: dorny/paths-filter@([a-f0-9]{40})", ci))
        self.assertEqual(len(pins), 1)
        self.assertIn("repository: dorny/paths-filter", job)
        self.assertIn(f"ref: {pins.pop()}", job)

    def test_main_runs_do_not_share_a_cancellation_group(self):
        ci = CI.read_text(encoding="utf-8")
        concurrency = ci.split("concurrency:\n")[1].split("\njobs:")[0]
        self.assertIn("github.run_id", concurrency)
        self.assertIn(
            "cancel-in-progress: ${{ github.event_name == 'pull_request' }}",
            concurrency,
        )


@unittest.skipUnless(
    os.environ.get("YOLOZU_PATHS_FILTER_ENTRYPOINT"),
    "pinned action bundle not supplied",
)
class TestCiPathRoutingBundle(unittest.TestCase):
    def run_filters(self, changed, *, operation="modify"):
        entrypoint = Path(os.environ["YOLOZU_PATHS_FILTER_ENTRYPOINT"]).resolve()
        self.assertTrue(entrypoint.is_file(), entrypoint)
        self.assertIsNotNone(
            shutil.which("node"), "Node is required for the actual action bundle"
        )
        with tempfile.TemporaryDirectory(prefix="yolozu-ci-paths-") as temp:
            root = Path(temp)

            def git(*args):
                subprocess.run(
                    ["git", *args], cwd=root, check=True, capture_output=True
                )

            git("init", "-q")
            git("config", "user.email", "ci-routing@example.invalid")
            git("config", "user.name", "CI routing test")
            for name in [] if operation == "add" else changed:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("before\n", encoding="utf-8")
            git("add", ".")
            git(
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "fixture",
                "--allow-empty",
            )
            for name in changed:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if operation == "delete":
                    path.unlink()
                elif operation == "rename":
                    target = root / "reports/renamed" / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    path.rename(target)
                else:
                    path.write_text("after\n", encoding="utf-8")
            git("add", "-A")
            result = {}
            for index, (filters, quantifier) in enumerate(action_filters()):
                output = root / f"action-output-{index}"
                output.touch()
                env = {
                    key: value
                    for key, value in os.environ.items()
                    if not key.startswith(("GITHUB_", "INPUT_", "GIT_"))
                }
                env.update(
                    {
                        "INPUT_FILTERS": filters,
                        "INPUT_PREDICATE-QUANTIFIER": quantifier,
                        "INPUT_BASE": "HEAD",
                        "INPUT_LIST-FILES": "json",
                        "GITHUB_OUTPUT": str(output),
                        "GITHUB_EVENT_NAME": "workflow_dispatch",
                    }
                )
                run = subprocess.run(
                    ["node", str(entrypoint)],
                    cwd=root,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                values = {}
                for match in re.finditer(
                    r"^(\w+)<<(\S+)\n(.*?)\n\2\n", output.read_text(), re.M | re.S
                ):
                    values[match[1]] = match[3]
                for name, value in values.items():
                    if name.endswith("_files"):
                        route = name.removesuffix("_files")
                        result[route] = set(json.loads(value))
                        self.assertEqual(
                            values[route], str(bool(result[route])).lower()
                        )
            self.assertEqual(
                set(result), {"full_ci", "docs_mcp_ci", "manual_ci", "workflows_ci"}
            )
            return {name: paths for name, paths in result.items() if paths}

    def test_metadata_and_evidence_do_not_select_heavy_jobs(self):
        self.assertEqual(
            self.run_filters(
                [
                    "reports/audit/result.json",
                    ".beads/interactions.jsonl",
                    "CHANGELOG.md",
                ]
            ),
            {},
        )
        self.assertEqual(self.run_filters([]), {})

    def test_runtime_includes_locks_and_mixed_reports(self):
        runtime = {
            "yolozu/cli.py",
            "tools/validate_predictions.py",
            "scripts/check.sh",
            "rtdetr_pose/model.py",
            "tests/test_dataset.py",
            "configs/sample.json",
            "data/smoke/a.txt",
            "baselines/ref.json",
            "requirements-test.txt",
            "requirements.lock",
            "requirements-locks/requirements-ci.lock",
            "pyproject.toml",
            "pytest.ini",
            "ruff.toml",
            "MANIFEST.in",
            "setup.py",
            "setup.cfg",
        }
        for operation in ("modify", "add", "delete", "rename"):
            with self.subTest(operation=operation):
                self.assertEqual(
                    self.run_filters(
                        sorted(runtime | {"reports/audit.json"}), operation=operation
                    ),
                    {"full_ci": runtime},
                )

    def test_docs_mcp_exclusions_and_mixed_runtime(self):
        docs = {
            "README.md",
            "Readme_jp.md",
            "docs/quickstart.md",
            "reports/adaptive_vision_roadmap.md",
            "reports/adaptive_qualification_foundation_cpu.md",
            "yolozu/integrations/mcp_server.py",
            "yolozu/integrations/actions_api.py",
            "tools/run_mcp_server.py",
            "tools/run_actions_api.py",
            "tools/check_mcp_settings.py",
            "tests/test_ai_first_mcp_surface.py",
            "tests/test_candidate_artifact_ai_surface.py",
            "tests/test_integrations_mcp_actions_parity.py",
            "tests/test_check_mcp_settings_tool.py",
            "tests/test_generated_integration_tool_reference.py",
        }
        self.assertEqual(self.run_filters(sorted(docs)), {"docs_mcp_ci": docs})
        self.assertEqual(
            self.run_filters(["yolozu/cli.py", "docs/quickstart.md"]),
            {"full_ci": {"yolozu/cli.py"}, "docs_mcp_ci": {"docs/quickstart.md"}},
        )

    def test_manual_workflow_and_routing_test_paths(self):
        workflows = {
            ".github/workflows/build_and_test.yml",
            "tests/test_ci_path_routing.py",
        }
        self.assertEqual(
            self.run_filters(sorted(workflows)), {"workflows_ci": workflows}
        )
        self.assertEqual(
            self.run_filters(
                [
                    "manual/main.tex",
                    "docs/quickstart.md",
                    ".github/workflows/publish.yml",
                ]
            ),
            {
                "manual_ci": {"manual/main.tex"},
                "docs_mcp_ci": {"docs/quickstart.md"},
                "workflows_ci": {".github/workflows/publish.yml"},
            },
        )

    def test_docs_dependency_locks_select_both_runtime_and_docs(self):
        locks = {
            "requirements-locks/requirements-docs-actions.lock",
            "requirements-locks/requirements-web-docs.lock",
        }
        self.assertEqual(
            self.run_filters(sorted(locks)),
            {"full_ci": locks, "docs_mcp_ci": locks},
        )


if __name__ == "__main__":
    unittest.main()
