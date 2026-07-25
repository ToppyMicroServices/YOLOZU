from __future__ import annotations

import ast
import json
import os
import shlex
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import unittest
from pathlib import Path


class TestWebDocsCandidateWheel(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def _run(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        expect: int = 0,
    ) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            expect,
            (
                f"command returned {proc.returncode}, expected {expect}: {command!r}\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            ),
        )
        return proc

    def test_candidate_wheel_runs_tutorial_outside_checkout_without_pythonpath(
        self,
    ) -> None:
        content = json.loads(
            (self.repo_root / "docs" / "web_docs_content.json").read_text(
                encoding="utf-8"
            )
        )
        documented_commands = [
            command
            for step in content["tutorial"]["thirty_minute"]
            for command in step["commands"]
        ]

        def documented_command(snippet: str) -> str:
            matches = [
                command for command in documented_commands if snippet in command
            ]
            self.assertEqual(len(matches), 1, (snippet, matches))
            return matches[0]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_archive = root / "source.tar"
            source_root = root / "source"
            wheel_dir = root / "wheelhouse"
            venv_dir = root / "venv"
            work_dir = root / "journey"
            source_root.mkdir()
            wheel_dir.mkdir()
            work_dir.mkdir()

            self._run(
                ["git", "archive", "--format=tar", "-o", str(source_archive), "HEAD"],
                cwd=self.repo_root,
            )
            with tarfile.open(source_archive) as archive:
                archive.extractall(source_root)

            self._run(
                [
                    sys.executable,
                    "-c",
                    "import setuptools, wheel",
                ],
                cwd=source_root,
            )
            self._run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--disable-pip-version-check",
                    "--no-cache-dir",
                    "--no-deps",
                    "--no-build-isolation",
                    "--wheel-dir",
                    str(wheel_dir),
                    ".",
                ],
                cwd=source_root,
            )
            wheels = list(wheel_dir.glob("yolozu-*.whl"))
            self.assertEqual(len(wheels), 1, wheels)

            self._run(
                [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
                cwd=root,
            )
            if os.name == "nt":
                python = venv_dir / "Scripts" / "python.exe"
                cli_path = venv_dir / "Scripts" / "yolozu.exe"
            else:
                python = venv_dir / "bin" / "python"
                cli_path = venv_dir / "bin" / "yolozu"
            runner_site = Path(sysconfig.get_paths()["purelib"]).resolve()
            nested_site_proc = self._run(
                [
                    str(python),
                    "-c",
                    "import sysconfig; print(sysconfig.get_paths()['purelib'])",
                ],
                cwd=root,
            )
            nested_site = Path(nested_site_proc.stdout.strip()).resolve()
            if runner_site != nested_site:
                (nested_site / "yolozu-ci-runner-dependencies.pth").write_text(
                    str(runner_site) + "\n",
                    encoding="utf-8",
                )
            self._run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-deps",
                    str(wheels[0]),
                ],
                cwd=root,
            )

            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            # The nested environment reuses the test runner's already-installed
            # dependencies so this contract test stays offline. Isolation here
            # is for the YOLOZU wheel/import path, not dependency resolution.
            location_proc = self._run(
                [
                    str(python),
                    "-c",
                    (
                        "from pathlib import Path; import yolozu, yolozu.api; "
                        "print(Path(yolozu.__file__).resolve())"
                    ),
                ],
                cwd=work_dir,
                env=env,
            )
            installed_location = Path(location_proc.stdout.strip())
            self.assertNotEqual(installed_location, Path())
            self.assertNotIn(self.repo_root, installed_location.parents)
            self.assertIn(venv_dir.resolve(), installed_location.parents)

            self.assertTrue(cli_path.is_file(), f"missing console script: {cli_path}")

            def installed_command(command: str) -> list[str]:
                parts = shlex.split(command)
                if parts[0] == "yolozu":
                    parts[0] = str(cli_path)
                elif parts[0] == "python":
                    parts[0] = str(python)
                return parts

            doctor_proc = subprocess.run(
                installed_command(documented_command("yolozu doctor --proof")),
                cwd=work_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            doctor_report = json.loads(
                (work_dir / "reports" / "quickstart" / "doctor.json").read_text(
                    encoding="utf-8"
                )
            )
            proof_report = json.loads(
                (
                    work_dir
                    / "reports"
                    / "quickstart"
                    / "proof"
                    / "proof_report.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                doctor_proc.returncode,
                0,
                (
                    f"doctor failed:\nstdout:\n{doctor_proc.stdout}\n"
                    f"stderr:\n{doctor_proc.stderr}\n"
                    f"doctor report:\n{json.dumps(doctor_report, indent=2)}\n"
                    f"proof report:\n{json.dumps(proof_report, indent=2)}"
                ),
            )
            dataset = "reports/quickstart/proof/toy_dataset"
            predictions = "reports/quickstart/proof/known_predictions.json"
            self._run(
                installed_command(
                    documented_command("yolozu validate dataset")
                ),
                cwd=work_dir,
                env=env,
            )
            self._run(
                installed_command(
                    documented_command("yolozu validate predictions")
                ),
                cwd=work_dir,
                env=env,
            )

            dry_report = work_dir / "reports" / "quickstart" / "eval_coco_dry_run.json"
            self._run(
                installed_command(
                    content["tutorial"]["dry_run_fallback"]["command"]
                ),
                cwd=work_dir,
                env=env,
            )
            dry_payload = json.loads(dry_report.read_text(encoding="utf-8"))
            self.assertEqual(dry_payload["status"], "ok")
            self.assertTrue(dry_payload["dry_run"])
            self.assertEqual(
                dry_payload["validation"],
                {"mode": "strict", "repair_enabled": False},
            )
            self.assertTrue(
                all(value is None for value in dry_payload["metrics"].values())
            )

            api_script = work_dir / "api_check.py"
            api_script.write_text(
                "\n".join(
                    [
                        "import json",
                        "from pathlib import Path",
                        "from yolozu.api import evaluate_coco",
                        "workspace = Path.cwd().resolve()",
                        "result = evaluate_coco(",
                        '    dataset=workspace / "reports/quickstart/proof/toy_dataset",',
                        '    predictions=workspace / "reports/quickstart/proof/known_predictions.json",',
                        '    split="val2017",',
                        "    dry_run=True,",
                        ")",
                        "print(json.dumps(result.to_dict(), sort_keys=True))",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            api_proc = self._run(
                [str(python), str(api_script)],
                cwd=work_dir,
                env=env,
            )
            api_payload = json.loads(api_proc.stdout)
            self.assertEqual(api_payload["status"], "ok")
            self.assertEqual(
                api_payload["validation"],
                {"mode": "strict", "repair_enabled": False},
            )

            invalid_predictions = work_dir / "invalid_predictions.json"
            invalid_payload = json.loads(
                (work_dir / predictions).read_text(encoding="utf-8")
            )
            invalid_payload["predictions"][0]["detections"][0]["score"] = 1.5
            invalid_predictions.write_text(
                json.dumps(invalid_payload),
                encoding="utf-8",
            )
            rejected_report = work_dir / "reports" / "quickstart" / "rejected.json"
            rejected_report.write_text('{"status": "ok"}\n', encoding="utf-8")
            reject_proc = subprocess.run(
                [
                    str(cli_path),
                    "eval-coco",
                    "-d",
                    dataset,
                    "-p",
                    str(invalid_predictions),
                    "-s",
                    "val2017",
                    "--dry-run",
                    "-o",
                    str(rejected_report),
                ],
                cwd=work_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(reject_proc.returncode, 0)
            rejected_payload = json.loads(
                rejected_report.read_text(encoding="utf-8")
            )
            self.assertEqual(rejected_payload["status"], "failed")
            self.assertFalse(rejected_payload["ok"])

            pycocotools_available = subprocess.run(
                [str(python), "-c", "import pycocotools"],
                cwd=work_dir,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode == 0
            require_real_coco = os.environ.get("YOLOZU_REQUIRE_REAL_COCO") == "1"
            if require_real_coco and not pycocotools_available:
                self.fail(
                    "YOLOZU_REQUIRE_REAL_COCO=1 but pycocotools is unavailable"
                )
            if pycocotools_available:
                real_report = work_dir / "reports" / "quickstart" / "eval_coco.json"
                self._run(
                    installed_command(
                        documented_command("yolozu eval-coco")
                    ),
                    cwd=work_dir,
                    env=env,
                )
                real_payload = json.loads(real_report.read_text(encoding="utf-8"))
                self.assertEqual(real_payload["status"], "ok")
                self.assertFalse(real_payload["dry_run"])
                self.assertIsInstance(real_payload["metrics"]["map50"], float)
                report_inspection = documented_command("python -m json.tool")
                self._run(
                    installed_command(report_inspection),
                    cwd=work_dir,
                    env=env,
                )

                documented_api_script = work_dir / "documented_api_check.py"
                documented_api_script.write_text(
                    content["python_api"]["example"] + "\n",
                    encoding="utf-8",
                )
                documented_api_proc = self._run(
                    [str(python), str(documented_api_script)],
                    cwd=work_dir,
                    env=env,
                )
                documented_metrics = ast.literal_eval(
                    documented_api_proc.stdout.strip()
                )
                self.assertIsInstance(
                    documented_metrics["map50"],
                    float,
                )


if __name__ == "__main__":
    unittest.main()
