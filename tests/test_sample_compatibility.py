import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/ci/check_sample_compatibility.py"
SPEC = importlib.util.spec_from_file_location("sample_compatibility", SCRIPT)
compat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compat)


class SampleCompatibilityTests(unittest.TestCase):
    def _installed_wheel(self, root):
        installed = root / "installed/yolozu"
        installed.mkdir(parents=True)
        source = b'__version__ = "4.7.0"\n'
        module = installed / "__init__.py"
        module.write_bytes(source)
        wheel = root / "yolozu-4.7.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("yolozu/__init__.py", source)
            archive.writestr("yolozu-4.7.0.dist-info/METADATA", "Name: yolozu\nVersion: 4.7.0\n")
        probe = {"editable": False, "module_file": str(module), "distribution_module_file": str(module),
                 "version": "4.7.0", "module_version": "4.7.0", "python": "3.10.19",
                 "system": "Linux", "machine": "x86_64", "dependencies": {"yolozu": "4.7.0"}}
        return wheel, probe

    def _sample(self, root):
        root.mkdir()
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            for index in range(4):
                (root / "images" / split / f"{index}.png").write_bytes(b"test image")
                (root / "labels" / split / f"{index}.txt").write_text(f"{index % 2} 0.5 0.5 0.2 0.2\n")
        (root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames: [box, circle]\n")
        (root / "predictions.json").write_text('{"predictions": []}\n')
        (root / "sample_manifest.json").write_text("{}\n")
        (root / "preview.png").write_bytes(b"test preview")

    def test_same_version_different_code_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel, probe = self._installed_wheel(Path(tmp))
            result = compat._verify_wheel_identity(wheel, probe)
            self.assertEqual(result["installed_package_files_verified"], 1)
            Path(probe["module_file"]).write_text('__version__ = "4.7.0"\n# different code\n')
            with self.assertRaisesRegex(ValueError, "differs from candidate wheel"):
                compat._verify_wheel_identity(wheel, probe)

    def test_editable_and_mismatched_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel, probe = self._installed_wheel(Path(tmp))
            with self.assertRaisesRegex(ValueError, "non-editably"):
                compat._verify_wheel_identity(wheel, {**probe, "editable": True})
            with self.assertRaisesRegex(ValueError, "metadata"):
                compat._verify_wheel_identity(wheel, {**probe, "distribution_module_file": str(Path(tmp) / "other.py")})
            with self.assertRaisesRegex(ValueError, "versions differ"):
                compat._verify_wheel_identity(wheel, {**probe, "version": "1.0.4"})

    def test_metrics_require_actual_finite_perfect_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            for dry_run, value in ((True, 1.0), (False, None), (False, float("nan")),
                                   (False, True), (False, 0.9)):
                with self.subTest(dry_run=dry_run, value=value):
                    path.write_text(json.dumps({"dry_run": dry_run,
                        "metrics": {"map50": value, "map50_95": 1.0, "map75": 1.0}}))
                    with self.assertRaises(ValueError):
                        compat._check_metrics(path)

    def _run_mocked(self, root, *, pip_failure=False, mutate=False, changed_recall=False):
        wheel, probe = self._installed_wheel(root)
        calls = []

        def run(args, **kwargs):
            self.assertEqual(args[1], "-I")
            self.assertNotEqual(Path(kwargs["cwd"]), root)
            calls.append(args)
            stdout = ""
            if args[2] == "-c":
                stdout = json.dumps(probe)
            elif args[2:] == ["-m", "pip", "check"] and pip_failure:
                return subprocess.CompletedProcess(args, 1, "", "dependency mismatch")
            elif "--run-dir" in args:
                self._sample(Path(args[args.index("--run-dir") + 1]))
            elif "eval-coco" in args:
                dataset = Path(args[args.index("--dataset") + 1])
                destination = Path(args[args.index("--output") + 1])
                metrics = {"map50": 1.0, "map50_95": 1.0, "map75": 1.0, "ar100": 1.0}
                if changed_recall and "relocated" in str(dataset):
                    metrics["ar100"] = 0.9
                destination.write_text(json.dumps({"dry_run": False, "metrics": metrics}))
                if mutate:
                    (dataset / "labels/val/0.txt").write_text("0 0.1 0.1 0.1 0.1\n")
            return subprocess.CompletedProcess(args, 0, stdout, "")

        with patch.object(compat.subprocess, "run", side_effect=run):
            report = compat.run_check(wheel, root / "report", source_revision="abc123")
        self.assertTrue((root / "report/compatibility-report.json").is_file())
        return report, calls

    def test_journey_checks_descriptor_relocation_and_unchanged_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, calls = self._run_mocked(Path(tmp))
            self.assertTrue(report["ok"], report.get("error"))
            self.assertTrue(report["source_files_unchanged"])
            self.assertEqual(report["metrics"], report["relocated_metrics"])
            self.assertEqual(report["source_revision"], "abc123")
            self.assertEqual(len([args for args in calls if "eval-coco" in args]), 2)
            self.assertTrue(any("relocated sample with spaces" in arg for args in calls for arg in args))
            self.assertEqual(len([args for args in calls if any(arg.endswith("data.yaml") for arg in args)]), 2)

    def test_dependency_failure_does_not_generate_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, calls = self._run_mocked(Path(tmp), pip_failure=True)
            self.assertFalse(report["ok"])
            self.assertIn("pip-check failed", report["error"])
            self.assertFalse(any("--run-dir" in args for args in calls))

    def test_sample_mutation_and_metric_drift_fail_the_journey(self):
        for option in ("mutate", "changed_recall"):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as tmp:
                report, _ = self._run_mocked(Path(tmp), **{option: True})
                self.assertFalse(report["ok"])
                self.assertIn("changed", report["error"])

    def test_help(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--wheel", result.stdout)


class SampleCompatibilityWorkflowTests(unittest.TestCase):
    def test_candidate_matrix_has_all_eight_scoped_lanes(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/sample_compatibility.yml").read_text())
        rows = workflow["jobs"]["candidate"]["strategy"]["matrix"]["include"]
        actual = {(row["os"], row["python"], row["dependencies"]) for row in rows}
        expected = {("ubuntu-latest", version, "latest") for version in ("3.10", "3.11", "3.12", "3.13", "3.14")}
        expected.update({("ubuntu-latest", "3.10", "minimum"), ("macos-14", "3.14", "latest"),
                         ("windows-latest", "3.12", "latest")})
        self.assertEqual(actual, expected)
        self.assertEqual(len(rows), 8)
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        steps = workflow["jobs"]["candidate"]["steps"]
        install = next(step["run"] for step in steps if step.get("id") == "install")
        self.assertIn("venv.EnvBuilder(with_pip=True)", install)
        self.assertIn('"--only-binary=:all:"', install)
        for requirement in ("numpy==1.24.0", "PyYAML==6.0", "Pillow==12.2.0", "typing_extensions==4.8.0", "pycocotools==2.0.7"):
            self.assertIn(requirement, install)
        self.assertTrue(any("check_sample_compatibility.py" in step.get("run", "") for step in steps))


if __name__ == "__main__":
    unittest.main()
