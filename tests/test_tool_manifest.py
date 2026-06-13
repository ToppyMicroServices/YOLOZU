import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import json


class TestToolManifest(unittest.TestCase):
    def _run_validator(self, manifest_obj: dict, *, require_declarative: bool = False) -> subprocess.CompletedProcess[str]:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "validate_tool_manifest.py"

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fp:
            json.dump(manifest_obj, fp)
            tmp_path = fp.name

        args = [sys.executable, str(script), "--manifest", tmp_path]
        if require_declarative:
            args.append("--require-declarative")

        try:
            return subprocess.run(
                args,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_validate_tool_manifest(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "validate_tool_manifest.py"
        self.assertTrue(script.is_file(), "missing tools/validate_tool_manifest.py")

        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"tool manifest validation failed:\n{proc.stdout}\n{proc.stderr}")

    def test_validate_tool_manifest_declarative_mode_fails_missing_fields(self):
        bad_manifest = {
            "manifest_version": 1,
            "tools": [
                {
                    "id": "dummy_tool",
                    "entrypoint": "tools/validate_tool_manifest.py",
                    "runner": "python3",
                    "summary": "dummy",
                    "maturity": "stable",
                    "platform": {
                        "cpu_ok": True,
                        "gpu_required": False,
                        "macos_ok": True,
                        "linux_ok": True,
                    },
                    "effects": {"writes": []},
                    "examples": [{"description": "x", "command": "python3 tools/validate_tool_manifest.py"}],
                }
            ],
        }

        proc = self._run_validator(bad_manifest, require_declarative=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("required in declarative mode", proc.stderr)

    def test_validate_tool_manifest_declarative_mode_accepts_compliant(self):
        ok_manifest = {
            "manifest_version": 1,
            "tools": [
                {
                    "id": "dummy_tool_ok",
                    "entrypoint": "tools/validate_tool_manifest.py",
                    "runner": "python3",
                    "summary": "dummy",
                    "maturity": "stable",
                    "platform": {
                        "cpu_ok": True,
                        "gpu_required": False,
                        "macos_ok": True,
                        "linux_ok": True,
                    },
                    "inputs": [{"name": "manifest", "kind": "file", "required": False, "flag": "--manifest"}],
                    "effects": {"writes": [], "fixed_writes": []},
                    "outputs": [{"name": "status", "kind": "stdout", "description": "Validation status on stdout."}],
                    "examples": [{"description": "x", "command": "python3 tools/validate_tool_manifest.py --help"}],
                }
            ],
        }

        proc = self._run_validator(ok_manifest, require_declarative=True)
        self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

    def test_validate_tool_manifest_fails_duplicate_tool_id(self):
        manifest = {
            "manifest_version": 1,
            "tools": [
                {
                    "id": "dup_id",
                    "entrypoint": "tools/validate_tool_manifest.py",
                    "runner": "python3",
                    "summary": "a",
                    "maturity": "stable",
                },
                {
                    "id": "dup_id",
                    "entrypoint": "tools/validate_tool_manifest.py",
                    "runner": "python3",
                    "summary": "b",
                    "maturity": "stable",
                },
            ],
        }
        proc = self._run_validator(manifest)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("duplicate id", proc.stderr)

    def test_validate_tool_manifest_fails_effect_flag_not_declared(self):
        manifest = {
            "manifest_version": 1,
            "tools": [
                {
                    "id": "flag_mismatch",
                    "entrypoint": "tools/validate_tool_manifest.py",
                    "runner": "python3",
                    "summary": "x",
                    "maturity": "stable",
                    "inputs": [{"name": "output", "kind": "file", "required": False, "flag": "--output"}],
                    "effects": {
                        "writes": [
                            {
                                "flag": "--other-output",
                                "kind": "file",
                                "scope": "path",
                                "description": "mismatch",
                            }
                        ],
                        "fixed_writes": [],
                    },
                }
            ],
        }
        proc = self._run_validator(manifest)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("not declared in tool.inputs", proc.stderr)

    def test_validate_tool_manifest_fails_unknown_contract_reference(self):
        manifest = {
            "manifest_version": 1,
            "contracts": {
                "known_contract": {"summary": "k"},
            },
            "tools": [
                {
                    "id": "unknown_contract_ref",
                    "entrypoint": "tools/validate_tool_manifest.py",
                    "runner": "python3",
                    "summary": "x",
                    "maturity": "stable",
                    "contracts": {"produces": ["missing_contract"]},
                }
            ],
        }
        proc = self._run_validator(manifest)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("unknown contract id", proc.stderr)

    def test_validate_tool_manifest_fails_non_bool_platform_in_declarative_mode(self):
        manifest = {
            "manifest_version": 1,
            "tools": [
                {
                    "id": "bad_platform_bool",
                    "entrypoint": "tools/validate_tool_manifest.py",
                    "runner": "python3",
                    "summary": "x",
                    "maturity": "stable",
                    "platform": {
                        "cpu_ok": "yes",
                        "gpu_required": False,
                        "macos_ok": True,
                        "linux_ok": True,
                    },
                    "inputs": [],
                    "effects": {"writes": [], "fixed_writes": []},
                    "outputs": [],
                    "examples": [{"description": "x", "command": "python3 tools/validate_tool_manifest.py --help"}],
                }
            ],
        }
        proc = self._run_validator(manifest, require_declarative=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("platform.cpu_ok: must be bool", proc.stderr)

    def test_validate_tool_manifest_fails_missing_output_description(self):
        manifest = {
            "manifest_version": 1,
            "tools": [
                {
                    "id": "missing_output_description",
                    "entrypoint": "tools/validate_tool_manifest.py",
                    "runner": "python3",
                    "summary": "x",
                    "maturity": "stable",
                    "platform": {
                        "cpu_ok": True,
                        "gpu_required": False,
                        "macos_ok": True,
                        "linux_ok": True,
                    },
                    "inputs": [],
                    "effects": {"writes": [], "fixed_writes": []},
                    "outputs": [{"name": "status", "kind": "stdout"}],
                    "examples": [{"description": "x", "command": "python3 tools/validate_tool_manifest.py --help"}],
                }
            ],
        }
        proc = self._run_validator(manifest, require_declarative=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("outputs[0].description: required in declarative mode", proc.stderr)

    def test_validate_tool_manifest_fails_missing_file_output_default(self):
        manifest = {
            "manifest_version": 1,
            "tools": [
                {
                    "id": "missing_file_output_default",
                    "entrypoint": "tools/validate_tool_manifest.py",
                    "runner": "python3",
                    "summary": "x",
                    "maturity": "stable",
                    "platform": {
                        "cpu_ok": True,
                        "gpu_required": False,
                        "macos_ok": True,
                        "linux_ok": True,
                    },
                    "inputs": [],
                    "effects": {"writes": [], "fixed_writes": []},
                    "outputs": [{"name": "report_json", "kind": "file", "description": "Report JSON."}],
                    "examples": [{"description": "x", "command": "python3 tools/validate_tool_manifest.py --help"}],
                }
            ],
        }
        proc = self._run_validator(manifest, require_declarative=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("outputs[0].default: required for file/dir outputs", proc.stderr)

    def test_validate_tool_manifest_fails_missing_maturity(self):
        manifest = {
            "manifest_version": 1,
            "tools": [
                {
                    "id": "missing_maturity",
                    "entrypoint": "tools/validate_tool_manifest.py",
                    "runner": "python3",
                    "summary": "x",
                }
            ],
        }
        proc = self._run_validator(manifest)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(".maturity: must be one of", proc.stderr)

    def test_validate_tool_manifest_fails_invalid_maturity(self):
        manifest = {
            "manifest_version": 1,
            "tools": [
                {
                    "id": "bad_maturity",
                    "entrypoint": "tools/validate_tool_manifest.py",
                    "runner": "python3",
                    "summary": "x",
                    "maturity": "beta",
                }
            ],
        }
        proc = self._run_validator(manifest)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(".maturity: must be one of", proc.stderr)


if __name__ == "__main__":
    unittest.main()
