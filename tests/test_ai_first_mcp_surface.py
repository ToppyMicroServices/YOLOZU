import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from yolozu.integrations.ai_surface import generate_config, review_config


class TestAiFirstMcpSurface(unittest.TestCase):
    def test_generate_config_shape_is_stable(self):
        cfg = generate_config()
        self.assertEqual(cfg.get("schema_version"), 1)
        self.assertIn("goal", cfg)
        self.assertIn("tool", cfg)
        self.assertIn("arguments", cfg)
        self.assertIn("safety", cfg)
        self.assertIsInstance(cfg.get("recommended_sequence"), list)
        self.assertGreaterEqual(len(cfg.get("recommended_sequence") or []), 1)

    def test_review_config_accepts_safe_config(self):
        cfg = generate_config()
        out = review_config(cfg, workspace_root=".")
        self.assertTrue(bool(out.get("ok")))
        self.assertEqual(out.get("schema_version"), 1)
        self.assertIn("issues", out)
        self.assertIn("warnings", out)

    def test_review_config_rejects_workspace_escape(self):
        cfg = generate_config(output="/tmp/escape.json")
        out = review_config(cfg, workspace_root=".")
        self.assertFalse(bool(out.get("ok")))
        issues = out.get("issues") or []
        codes = {str(item.get("code")) for item in issues if isinstance(item, dict)}
        self.assertIn("unsafe_output_path", codes)

    def test_run_mcp_server_help_and_samples(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_mcp_server.py"
        self.assertTrue(script.is_file())

        help_proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(help_proc.returncode, 0, msg=f"--help failed:\n{help_proc.stdout}\n{help_proc.stderr}")
        self.assertIn("--print-tools", help_proc.stdout)
        self.assertIn("--sample-generate-config", help_proc.stdout)
        self.assertIn("--sample-review-config", help_proc.stdout)

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            cfg_path = root / "ai_generate_config.json"
            gen_proc = subprocess.run(
                [sys.executable, str(script), "--sample-generate-config"],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(gen_proc.returncode, 0, msg=f"generate sample failed:\n{gen_proc.stdout}\n{gen_proc.stderr}")
            cfg_path.write_text(gen_proc.stdout, encoding="utf-8")
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.assertEqual(cfg.get("schema_version"), 1)

            review_proc = subprocess.run(
                [sys.executable, str(script), "--sample-review-config", str(cfg_path)],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(review_proc.returncode, 0, msg=f"review sample failed:\n{review_proc.stdout}\n{review_proc.stderr}")
            review_doc = json.loads(review_proc.stdout)
            self.assertIn("ok", review_doc)
            self.assertIn("issues", review_doc)

    def test_run_actions_api_help(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_actions_api.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"run_actions_api --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("--host", proc.stdout)
        self.assertIn("--port", proc.stdout)
        self.assertIn("--workers", proc.stdout)


if __name__ == "__main__":
    unittest.main()
