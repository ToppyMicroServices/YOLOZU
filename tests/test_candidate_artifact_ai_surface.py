from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


class TestCandidateArtifactAiSurface(unittest.TestCase):
    def _run(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def test_git_archive_to_clean_venv_outside_checkout(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is not installed")
        if importlib.util.find_spec("setuptools") is None:
            self.skipTest("setuptools build backend is not installed")

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive_path = root / "candidate-source.tar"
            archive_source = root / "archive-source"
            sdist_source = root / "sdist-source"
            wheel_dir = root / "wheel"
            venv_dir = root / "venv"
            consumer = root / "outside-consumer"
            for path in (
                archive_source,
                sdist_source,
                wheel_dir,
                consumer,
            ):
                path.mkdir()

            archived = self._run(
                [
                    "git",
                    "archive",
                    "--format=tar",
                    "--output",
                    str(archive_path),
                    "HEAD",
                ],
                cwd=repo_root,
            )
            self.assertEqual(
                archived.returncode,
                0,
                msg=f"git archive failed:\n{archived.stderr}",
            )
            with tarfile.open(archive_path, mode="r") as archive:
                archive.extractall(archive_source, filter="data")

            sdist_build = self._run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from setuptools.build_meta import build_sdist; "
                        "print(build_sdist('dist'))"
                    ),
                ],
                cwd=archive_source,
            )
            self.assertEqual(
                sdist_build.returncode,
                0,
                msg=(
                    "sdist build from git archive failed:\n"
                    f"{sdist_build.stdout}\n{sdist_build.stderr}"
                ),
            )
            sdists = sorted((archive_source / "dist").glob("yolozu-*.tar.gz"))
            self.assertEqual(len(sdists), 1)
            with tarfile.open(sdists[0], mode="r:gz") as archive:
                archive.extractall(sdist_source, filter="data")
            projects = [
                path
                for path in sdist_source.iterdir()
                if path.is_dir()
            ]
            self.assertEqual(len(projects), 1)

            wheel_build = self._run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from setuptools.build_meta import build_wheel; "
                        f"print(build_wheel({str(wheel_dir)!r}))"
                    ),
                ],
                cwd=projects[0],
            )
            self.assertEqual(
                wheel_build.returncode,
                0,
                msg=(
                    "wheel build from sdist failed:\n"
                    f"{wheel_build.stdout}\n{wheel_build.stderr}"
                ),
            )
            wheels = sorted(wheel_dir.glob("yolozu-*.whl"))
            self.assertEqual(len(wheels), 1)

            created = self._run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                cwd=root,
            )
            self.assertEqual(
                created.returncode,
                0,
                msg=f"venv creation failed:\n{created.stderr}",
            )
            executable_dir = "Scripts" if os.name == "nt" else "bin"
            venv_python = venv_dir / executable_dir / (
                "python.exe" if os.name == "nt" else "python"
            )
            installed = self._run(
                [
                    str(venv_python),
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    str(wheels[0]),
                ],
                cwd=root,
            )
            self.assertEqual(
                installed.returncode,
                0,
                msg=(
                    "candidate wheel install failed:\n"
                    f"{installed.stdout}\n{installed.stderr}"
                ),
            )

            shutil.copytree(
                archive_source / "data" / "smoke",
                consumer / "data" / "smoke",
            )
            clean_env = os.environ.copy()
            clean_env.pop("PYTHONPATH", None)
            clean_env["PYTHONNOUSERSITE"] = "1"
            probe = self._run(
                [
                    str(venv_python),
                    "-c",
                    """
import importlib.util
import json
import sys
from importlib.resources import files
from pathlib import Path

import yolozu
from yolozu.integrations.ai_surface import list_manifest_tools

package = files("yolozu")
data = files("yolozu.data")
ids = list_manifest_tools(guaranteed=True, ids_only=True)
print(json.dumps({
    "executable": sys.executable,
    "module": yolozu.__file__,
    "ids": ids,
    "manifest": data.joinpath("manifest").joinpath("tools_manifest.json").is_file(),
    "schema": data.joinpath("schemas").joinpath("predictions_validation_result.schema.json").is_file(),
    "mcp_reference": data.joinpath("integrations").joinpath("mcp_actions_tool_reference.json").is_file(),
    "py_typed": package.joinpath("py.typed").is_file(),
    "numpy_available": importlib.util.find_spec("numpy") is not None,
}, sort_keys=True))
""",
                ],
                cwd=consumer,
                env=clean_env,
            )
            self.assertEqual(
                probe.returncode,
                0,
                msg=(
                    "outside-checkout import failed:\n"
                    f"{probe.stdout}\n{probe.stderr}"
                ),
            )
            payload = json.loads(probe.stdout)
            self.assertEqual(
                Path(payload["executable"]).resolve(),
                venv_python.resolve(),
            )
            self.assertTrue(
                Path(payload["module"]).resolve().is_relative_to(
                    venv_dir.resolve()
                )
            )
            self.assertFalse(
                Path(payload["module"]).resolve().is_relative_to(
                    repo_root.resolve()
                )
            )
            self.assertEqual(
                set(payload["ids"]),
                {
                    "doctor",
                    "generate_config",
                    "review_config",
                    "validate_predictions",
                },
            )
            for key in (
                "manifest",
                "schema",
                "mcp_reference",
                "py_typed",
            ):
                self.assertTrue(payload[key], key)
            self.assertFalse(payload["numpy_available"])

            mcp_entry = venv_dir / executable_dir / (
                "yolozu-mcp.exe" if os.name == "nt" else "yolozu-mcp"
            )
            yolozu_entry = venv_dir / executable_dir / (
                "yolozu.exe" if os.name == "nt" else "yolozu"
            )
            strict_validation = self._run(
                [
                    str(yolozu_entry),
                    "validate",
                    "predictions",
                    "data/smoke/predictions/predictions_dummy.json",
                    "--strict",
                    "--json",
                ],
                cwd=consumer,
                env=clean_env,
            )
            self.assertEqual(
                strict_validation.returncode,
                0,
                msg=(
                    "installed strict validation failed:\n"
                    f"{strict_validation.stdout}\n"
                    f"{strict_validation.stderr}"
                ),
            )
            validation_payload = json.loads(strict_validation.stdout)
            self.assertTrue(validation_payload["ok"])
            self.assertEqual(validation_payload["mode"], "strict")
            self.assertFalse(validation_payload["repair_enabled"])

            helped = self._run(
                [str(mcp_entry), "--help"],
                cwd=consumer,
                env=clean_env,
            )
            self.assertEqual(
                helped.returncode,
                0,
                msg=f"installed help failed:\n{helped.stderr}",
            )
            self.assertIn("--print-tools", helped.stdout)

            discovered = self._run(
                [
                    str(mcp_entry),
                    "--print-tools",
                    "--guaranteed",
                    "--ids-only",
                ],
                cwd=consumer,
                env=clean_env,
            )
            self.assertEqual(
                discovered.returncode,
                0,
                msg=f"installed discovery failed:\n{discovered.stderr}",
            )
            discovery = json.loads(discovered.stdout)
            self.assertEqual(set(discovery["selected_tool_ids"]), set(payload["ids"]))
            self.assertLess(len(discovered.stdout.encode("utf-8")), 1_500)
            self.assertEqual(discovered.stdout.count("\n"), 1)

            missing_extra = self._run(
                [str(mcp_entry)],
                cwd=consumer,
                env=clean_env,
            )
            self.assertEqual(missing_extra.returncode, 2)
            self.assertIn("yolozu[mcp]", missing_extra.stderr)
            self.assertFalse((consumer / "runs").exists())


if __name__ == "__main__":
    unittest.main()
