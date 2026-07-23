import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestDocsExamplesDrift(unittest.TestCase):
    def test_docs_examples_drift_audit_passes_repo_docs(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "audit_docs_examples_drift.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--json"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"docs examples drift audit failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        payload = json.loads(proc.stdout)
        self.assertTrue(payload.get("ok"))
        self.assertGreater(int(payload.get("checked_examples") or 0), 0)
        self.assertEqual(
            payload.get("docs"),
            [
                "README.md",
                "Readme_jp.md",
                "Readme_zh.md",
                "docs/README.md",
                "docs/cpu_only_dod.md",
                "docs/external_inference.md",
                "docs/interop_detectron2_mmdet.md",
                "docs/interop_yolox.md",
                "examples/infer_cpp/README.md",
                "examples/infer_rust/README.md",
            ],
        )

    def test_docs_examples_drift_audit_fails_stale_flag(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "audit_docs_examples_drift.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            doc = Path(td) / "stale.md"
            doc.write_text("```bash\nyolozu doctor --not-a-real-flag\n```\n", encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--docs",
                    str(doc),
                    "--skip-manual",
                    "--skip-manifest",
                    "--json",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        self.assertNotEqual(proc.returncode, 0)
        payload = json.loads(proc.stdout)
        self.assertFalse(payload.get("ok"))
        self.assertIn("--not-a-real-flag", json.dumps(payload))

    def test_dynamic_and_delegated_help_keep_unknown_flags_strict(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "audit_docs_examples_drift.py"
        commands = (
            "python3 -m yolozu train --external-backend yolox "
            "configs/examples/finetune_external/yolox_s_finetune_smoke.py "
            "--dataset data/smoke --not-a-backend-flag",
            "bash scripts/ttt_compare.sh --boilerplate tent --dataset data/smoke "
            "--checkpoint /path/to.ckpt --not-a-wrapper-flag",
        )
        for command in commands:
            with self.subTest(command=command), tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
                doc = Path(td) / "stale.md"
                doc.write_text(f"```bash\n{command}\n```\n", encoding="utf-8")
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "--docs",
                        str(doc),
                        "--skip-manual",
                        "--skip-manifest",
                        "--json",
                    ],
                    cwd=str(repo_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
            self.assertNotEqual(proc.returncode, 0)
            payload = json.loads(proc.stdout)
            self.assertFalse(payload.get("ok"))
            self.assertIn("--not-", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
