from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "gpu_zisn_pipeline.yml"


class GpuZisnWorkflowTests(unittest.TestCase):
    def test_workflow_uses_checked_in_ci_scripts(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("bash /workspace/tools/ci/run_gpu_zisn1.sh", text)
        self.assertIn("bash /workspace/tools/ci/run_gpu_zisn2.sh", text)
        self.assertIn("bash /workspace/tools/ci/run_gpu_zisn3.sh", text)
        self.assertNotIn("bash -lc '", text)

    def test_ci_stage_scripts_parse_with_bash(self) -> None:
        scripts = [
            REPO_ROOT / "tools" / "ci" / "run_gpu_zisn1.sh",
            REPO_ROOT / "tools" / "ci" / "run_gpu_zisn2.sh",
            REPO_ROOT / "tools" / "ci" / "run_gpu_zisn3.sh",
        ]
        for script in scripts:
            self.assertTrue(script.is_file(), f"missing script: {script}")
            completed = subprocess.run(
                ["bash", "-n", str(script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"bash parse failed for {script}: {completed.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
