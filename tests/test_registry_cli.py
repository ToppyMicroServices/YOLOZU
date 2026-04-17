import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestRegistryCLI(unittest.TestCase):
    def test_registry_list_show_validate(self):
        repo_root = Path(__file__).resolve().parents[1]

        # validate
        proc = subprocess.run(
            [sys.executable, "-m", "yolozu", "registry", "validate"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"registry validate failed:\n{proc.stdout}\n{proc.stderr}")

        # list --json
        proc = subprocess.run(
            [sys.executable, "-m", "yolozu", "registry", "list", "--json"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"registry list --json failed:\n{proc.stdout}\n{proc.stderr}")
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("kind"), "yolozu_tool_registry")
        self.assertEqual(payload.get("schema_version"), 1)
        tools = payload.get("tools")
        self.assertTrue(isinstance(tools, list) and tools)
        self.assertIn("export_predictions", {t.get("id") for t in tools if isinstance(t, dict)})

        # show --json
        proc = subprocess.run(
            [sys.executable, "-m", "yolozu", "registry", "show", "export_predictions", "--json"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"registry show --json failed:\n{proc.stdout}\n{proc.stderr}")
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("kind"), "yolozu_tool_spec")
        self.assertEqual(payload.get("schema_version"), 1)
        tool = payload.get("tool")
        self.assertTrue(isinstance(tool, dict))
        self.assertEqual(tool.get("id"), "export_predictions")

    def test_registry_run_normalize_predictions(self):
        repo_root = Path(__file__).resolve().parents[1]

        reports = repo_root / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        in_path = reports / "_tmp_registry_in.json"
        out_path = reports / "_tmp_registry_out.json"

        # Minimal predictions payload (list form) accepted by normalize_predictions.
        in_path.write_text(
            json.dumps(
                [
                    {
                        "image": "dummy.jpg",
                        "detections": [
                            {
                                "class_id": 0,
                                "score": 0.9,
                                "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                            }
                        ],
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "yolozu",
                "registry",
                "run",
                "--allow-write-root",
                "reports",
                "normalize_predictions",
                "--",
                "--input",
                str(in_path.relative_to(repo_root)),
                "--output",
                str(out_path.relative_to(repo_root)),
            ],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"registry run normalize_predictions failed:\n{proc.stdout}\n{proc.stderr}")

        self.assertTrue(out_path.is_file(), "expected output file to be created")


if __name__ == "__main__":
    unittest.main()
