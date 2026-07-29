import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestQualifyFinetuneLanesCli(unittest.TestCase):
    def _script(self) -> Path:
        return Path(__file__).resolve().parents[1] / "tools" / "qualify_finetune_lanes.py"

    def _module(self):
        spec = importlib.util.spec_from_file_location("qualify_finetune_lanes", self._script())
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(self._script()), "--help"],
            cwd=str(self._script().parents[1]),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
        self.assertIn("--output-dir", proc.stdout)
        self.assertIn("--dataset-root", proc.stdout)
        self.assertIn("--max-steps", proc.stdout)

    def test_output_must_be_fresh(self) -> None:
        module = self._module()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit) as ctx:
                module._fresh_output(td)
        self.assertIn("output already exists", str(ctx.exception))

    def test_dataset_must_be_repo_confined(self) -> None:
        module = self._module()
        with self.assertRaises(SystemExit) as ctx:
            module._confined_repo_path("/tmp", kind="dataset root")
        self.assertIn("inside the repository", str(ctx.exception))

    def test_schema_requires_protocol_completion_signal(self) -> None:
        schema_path = self._script().parents[1] / "docs" / "schemas" / "finetune_lane_qualification.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertIn("protocol_complete", schema["required"])
        self.assertEqual(schema["properties"]["protocol_complete"]["type"], "boolean")


if __name__ == "__main__":
    unittest.main()
