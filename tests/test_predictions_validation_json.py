from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from yolozu.predictions import validate_predictions_path


class TestPredictionsValidationJson(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.fixture = (
            self.repo_root
            / "data"
            / "smoke"
            / "predictions"
            / "predictions_dummy.json"
        )

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def test_module_cli_json_success_and_failure_are_stable(self) -> None:
        success = self._run(
            [
                sys.executable,
                "-m",
                "yolozu",
                "validate",
                "predictions",
                str(self.fixture),
                "--strict",
                "--json",
            ]
        )
        self.assertEqual(success.returncode, 0, msg=success.stderr)
        success_payload = json.loads(success.stdout)
        self.assertEqual(success_payload["schema_version"], 1)
        self.assertTrue(success_payload["ok"])
        self.assertEqual(success_payload["tool"], "validate_predictions")
        self.assertIsInstance(success_payload["warnings"], list)
        self.assertEqual(success_payload["errors"], [])

        with tempfile.TemporaryDirectory() as td:
            invalid = Path(td) / "invalid.json"
            invalid.write_text(json.dumps("not predictions"), encoding="utf-8")
            failure = self._run(
                [
                    sys.executable,
                    "-m",
                    "yolozu",
                    "validate",
                    "predictions",
                    str(invalid),
                    "--strict",
                    "--json",
                ]
            )
        self.assertEqual(failure.returncode, 1)
        failure_payload = json.loads(failure.stdout)
        self.assertFalse(failure_payload["ok"])
        self.assertEqual(
            failure_payload["errors"][0]["code"],
            "invalid_predictions",
        )
        self.assertEqual(failure_payload["warnings"], [])

    def test_standalone_cli_json_and_human_modes(self) -> None:
        script = self.repo_root / "tools" / "validate_predictions.py"
        machine = self._run(
            [
                sys.executable,
                str(script),
                str(self.fixture),
                "--strict",
                "--json",
            ]
        )
        self.assertEqual(machine.returncode, 0, msg=machine.stderr)
        self.assertTrue(json.loads(machine.stdout)["ok"])

        human = self._run(
            [
                sys.executable,
                str(script),
                str(self.fixture),
                "--strict",
            ]
        )
        self.assertEqual(human.returncode, 0, msg=human.stderr)
        self.assertIn("OK:", human.stdout)

    def test_python_result_bounds_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "many.json"
            path.write_text(
                json.dumps(
                    [
                        {"image": f"{index}.jpg", "detections": []}
                        for index in range(150)
                    ]
                ),
                encoding="utf-8",
            )
            result, exit_code = validate_predictions_path(path, strict=True)

        self.assertEqual(exit_code, 0)
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["warnings"]), 100)
        self.assertEqual(result["limits"]["warnings_truncated"], 50)

    def test_source_and_packaged_result_schemas_match(self) -> None:
        source = (
            self.repo_root
            / "docs"
            / "schemas"
            / "predictions_validation_result.schema.json"
        )
        packaged = (
            self.repo_root
            / "yolozu"
            / "data"
            / "schemas"
            / "predictions_validation_result.schema.json"
        )
        self.assertEqual(
            json.loads(source.read_text(encoding="utf-8")),
            json.loads(packaged.read_text(encoding="utf-8")),
        )


if __name__ == "__main__":
    unittest.main()
