from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from yolozu.predictions import validate_predictions_path


def _assert_schema_value(value, schema, *, where="$") -> None:
    if "const" in schema:
        if value != schema["const"]:
            raise AssertionError(
                f"{where}: expected const {schema['const']!r}, got {value!r}"
            )
    if "enum" in schema and value not in schema["enum"]:
        raise AssertionError(
            f"{where}: expected one of {schema['enum']!r}, got {value!r}"
        )
    expected_type = schema.get("type")
    if expected_type is not None:
        names = (
            list(expected_type)
            if isinstance(expected_type, list)
            else [expected_type]
        )

        def matches(name):
            return {
                "object": isinstance(value, dict),
                "array": isinstance(value, list),
                "string": isinstance(value, str),
                "boolean": isinstance(value, bool),
                "integer": isinstance(value, int)
                and not isinstance(value, bool),
                "null": value is None,
            }.get(name, True)

        if not any(matches(name) for name in names):
            raise AssertionError(
                f"{where}: expected type {names!r}, got {type(value).__name__}"
            )
    if isinstance(value, dict):
        required = list(schema.get("required") or [])
        missing = [name for name in required if name not in value]
        if missing:
            raise AssertionError(f"{where}: missing required keys {missing}")
        properties = dict(schema.get("properties") or {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise AssertionError(
                    f"{where}: unexpected keys {extra}"
                )
        for name, item in value.items():
            child = properties.get(name)
            if isinstance(child, dict):
                _assert_schema_value(item, child, where=f"{where}.{name}")
    if isinstance(value, list):
        maximum = schema.get("maxItems")
        if isinstance(maximum, int) and len(value) > maximum:
            raise AssertionError(
                f"{where}: {len(value)} items exceeds maxItems={maximum}"
            )
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _assert_schema_value(
                    item,
                    item_schema,
                    where=f"{where}[{index}]",
                )
    minimum = schema.get("minimum")
    if (
        isinstance(minimum, (int, float))
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value < minimum
    ):
        raise AssertionError(
            f"{where}: {value} is below minimum={minimum}"
        )


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
        self.assertEqual(success_payload["mode"], "strict")
        self.assertFalse(success_payload["repair_enabled"])
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

    def test_strict_rejects_and_non_strict_explicitly_reports_repair(self) -> None:
        payload = {
            "schema_version": 1,
            "predictions": [
                {
                    "schema_version": 2,
                    "image": "image.jpg",
                    "detections": [
                        {
                            "class_id": 0,
                            "score": 1.5,
                            "bbox": {
                                "cx": 1.2,
                                "cy": 0.5,
                                "w": 0.2,
                                "h": 0.2,
                            },
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "repairable.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            strict_result, strict_exit = validate_predictions_path(
                path,
                strict=True,
            )
            repair_result, repair_exit = validate_predictions_path(
                path,
                strict=False,
            )

        self.assertEqual(strict_exit, 1)
        self.assertFalse(strict_result["ok"])
        self.assertEqual(strict_result["mode"], "strict")
        self.assertFalse(strict_result["repair_enabled"])
        self.assertEqual(
            strict_result["errors"][0]["code"],
            "invalid_predictions",
        )
        self.assertEqual(repair_exit, 0)
        self.assertTrue(repair_result["ok"])
        self.assertEqual(repair_result["mode"], "repair")
        self.assertTrue(repair_result["repair_enabled"])
        self.assertTrue(
            any("score: out of range" in item for item in repair_result["warnings"])
        )
        self.assertTrue(
            any("bbox.cx: out of range" in item for item in repair_result["warnings"])
        )

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

    def test_success_and_failure_match_packaged_result_schema(self) -> None:
        schema = json.loads(
            (
                self.repo_root
                / "yolozu"
                / "data"
                / "schemas"
                / "predictions_validation_result.schema.json"
            ).read_text(encoding="utf-8")
        )
        success, success_exit = validate_predictions_path(
            self.fixture,
            strict=True,
        )
        with tempfile.TemporaryDirectory() as td:
            invalid = Path(td) / "invalid.json"
            invalid.write_text(json.dumps("not predictions"), encoding="utf-8")
            failure, failure_exit = validate_predictions_path(
                invalid,
                strict=True,
            )

        self.assertEqual(success_exit, 0)
        self.assertEqual(failure_exit, 1)
        _assert_schema_value(success, schema)
        _assert_schema_value(failure, schema)

        missing_mode = dict(success)
        missing_mode.pop("mode")
        with self.assertRaisesRegex(AssertionError, "missing required"):
            _assert_schema_value(missing_mode, schema)


if __name__ == "__main__":
    unittest.main()
