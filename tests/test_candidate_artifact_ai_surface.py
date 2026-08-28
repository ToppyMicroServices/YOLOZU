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
                sdist_names = {
                    member.name.partition("/")[2]
                    for member in archive.getmembers()
                    if "/" in member.name
                }
                for required in (
                    "yolozu/adaptive/isolation.py",
                    "yolozu/adaptive/isolation_policy.py",
                    "yolozu/adaptive/lifecycle.py",
                    "yolozu/adaptive/promotion.py",
                    "yolozu/adaptive/processing.py",
                    "yolozu/adaptive/algorithm_scout.py",
                    "yolozu/adaptive/freshness.py",
                    "yolozu/adaptive/screening.py",
                    "yolozu/adaptive/support_profiles.py",
                    "yolozu/adaptive/control_stream.py",
                    "yolozu/adaptive/safe_https.py",
                    "yolozu/data/adaptive_routing/bundle_specs.json",
                    "yolozu/data/adaptive_routing/bundle_lifecycle.jsonl",
                    "yolozu/data/adaptive_routing/candidate_screening.jsonl",
                    "yolozu/data/adaptive_routing/support_profiles.jsonl",
                    "yolozu/data/adaptive_routing/evidence_activation.jsonl",
                    "yolozu/data/schemas/image_job_spec.schema.json",
                    "yolozu/data/schemas/qualification_workload_profile.schema.json",
                    "yolozu/data/schemas/environment_profile.schema.json",
                    "yolozu/data/schemas/selection_decision.schema.json",
                    "yolozu/data/schemas/algorithm_scout_sources.schema.json",
                    "yolozu/data/schemas/algorithm_scout_report.schema.json",
                    "yolozu/data/schemas/qualification_freshness_report.schema.json",
                    "yolozu/data/schemas/candidate_screening_record.schema.json",
                    "yolozu/data/schemas/ocr_bundle_interface.schema.json",
                    "yolozu/data/schemas/ocr_result.schema.json",
                    "yolozu/data/schemas/candidate_isolation_probe.schema.json",
                    "yolozu/data/schemas/support_profile_set_proposal.schema.json",
                    "yolozu/data/schemas/lifecycle_rollback_bindings.schema.json",
                    "yolozu/data/schemas/bundle_lifecycle_record.schema.json",
                    "yolozu/data/schemas/support_profile_spec.schema.json",
                    "yolozu/data/integrations/mcp_actions_tool_reference.json",
                ):
                    self.assertIn(required, sdist_names)
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
    "adaptive_processing": package.joinpath("adaptive").joinpath("processing.py").is_file(),
    "adaptive_promotion": package.joinpath("adaptive").joinpath("promotion.py").is_file(),
    "adaptive_isolation": package.joinpath("adaptive").joinpath("isolation.py").is_file(),
    "adaptive_isolation_policy": package.joinpath("adaptive").joinpath("isolation_policy.py").is_file(),
    "adaptive_algorithm_scout": package.joinpath("adaptive").joinpath("algorithm_scout.py").is_file(),
    "adaptive_freshness": package.joinpath("adaptive").joinpath("freshness.py").is_file(),
    "adaptive_screening": package.joinpath("adaptive").joinpath("screening.py").is_file(),
    "adaptive_safe_https": package.joinpath("adaptive").joinpath("safe_https.py").is_file(),
    "adaptive_registry": data.joinpath("adaptive_routing").joinpath("bundle_specs.json").is_file(),
    "adaptive_lifecycle": data.joinpath("adaptive_routing").joinpath("bundle_lifecycle.jsonl").is_file(),
    "adaptive_screening_stream": data.joinpath("adaptive_routing").joinpath("candidate_screening.jsonl").is_file(),
    "adaptive_support": data.joinpath("adaptive_routing").joinpath("support_profiles.jsonl").is_file(),
    "adaptive_evidence": data.joinpath("adaptive_routing").joinpath("evidence_activation.jsonl").is_file(),
    "adaptive_job_schema": data.joinpath("schemas").joinpath("image_job_spec.schema.json").is_file(),
    "adaptive_workload_schema": data.joinpath("schemas").joinpath("qualification_workload_profile.schema.json").is_file(),
    "adaptive_environment_schema": data.joinpath("schemas").joinpath("environment_profile.schema.json").is_file(),
    "adaptive_selection_schema": data.joinpath("schemas").joinpath("selection_decision.schema.json").is_file(),
    "adaptive_scout_sources_schema": data.joinpath("schemas").joinpath("algorithm_scout_sources.schema.json").is_file(),
    "adaptive_scout_report_schema": data.joinpath("schemas").joinpath("algorithm_scout_report.schema.json").is_file(),
    "adaptive_freshness_schema": data.joinpath("schemas").joinpath("qualification_freshness_report.schema.json").is_file(),
    "adaptive_screening_schema": data.joinpath("schemas").joinpath("candidate_screening_record.schema.json").is_file(),
    "ocr_bundle_schema": data.joinpath("schemas").joinpath("ocr_bundle_interface.schema.json").is_file(),
    "ocr_result_schema": data.joinpath("schemas").joinpath("ocr_result.schema.json").is_file(),
    "adaptive_isolation_probe_schema": data.joinpath("schemas").joinpath("candidate_isolation_probe.schema.json").is_file(),
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
                "adaptive_environment_schema",
                "adaptive_algorithm_scout",
                "adaptive_freshness",
                "adaptive_freshness_schema",
                "adaptive_evidence",
                "adaptive_isolation",
                "adaptive_isolation_policy",
                "adaptive_isolation_probe_schema",
                "adaptive_job_schema",
                "adaptive_lifecycle",
                "adaptive_processing",
                "adaptive_promotion",
                "adaptive_registry",
                "adaptive_selection_schema",
                "adaptive_scout_report_schema",
                "adaptive_scout_sources_schema",
                "adaptive_safe_https",
                "adaptive_screening",
                "adaptive_screening_schema",
                "adaptive_screening_stream",
                "adaptive_support",
                "adaptive_workload_schema",
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
            scout_help = self._run(
                [str(yolozu_entry), "scout-algorithms", "--help"],
                cwd=consumer,
                env=clean_env,
            )
            self.assertEqual(
                scout_help.returncode,
                0,
                msg=f"installed scout help failed:\n{scout_help.stdout}\n{scout_help.stderr}",
            )
            self.assertIn("--collect", scout_help.stdout)
            freshness_help = self._run(
                [str(yolozu_entry), "check-qualification-freshness", "--help"],
                cwd=consumer,
                env=clean_env,
            )
            self.assertEqual(
                freshness_help.returncode,
                0,
                msg=(
                    "installed freshness help failed:\n"
                    f"{freshness_help.stdout}\n{freshness_help.stderr}"
                ),
            )
            self.assertIn("--evidence-root", freshness_help.stdout)
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
            self.assertEqual(
                validation_payload["limits"],
                {
                    "warnings_max": 100,
                    "warnings_truncated": 0,
                },
            )

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

            require_ttt = os.environ.get(
                "YOLOZU_REQUIRE_TTT_CANDIDATE",
            ) == "1"
            torch_available = importlib.util.find_spec("torch") is not None
            if require_ttt:
                self.assertTrue(
                    torch_available,
                    "required candidate TTT test needs torch in the CI runner",
                )
            if torch_available:
                purelib_probe = self._run(
                    [
                        str(venv_python),
                        "-c",
                        (
                            "import sysconfig; "
                            "print(sysconfig.get_paths()['purelib'])"
                        ),
                    ],
                    cwd=consumer,
                    env=clean_env,
                )
                self.assertEqual(
                    purelib_probe.returncode,
                    0,
                    msg=purelib_probe.stderr,
                )
                host_site_dirs = [
                    Path(entry).resolve()
                    for entry in sys.path
                    if "site-packages" in entry
                    and Path(entry).is_dir()
                ]
                self.assertTrue(
                    host_site_dirs,
                    "host dependency site-packages could not be located",
                )
                dependency_bridge = (
                    Path(purelib_probe.stdout.strip())
                    / "_yolozu_candidate_host_dependencies.pth"
                )
                dependency_bridge.write_text(
                    "".join(f"{path}\n" for path in host_site_dirs),
                    encoding="utf-8",
                )

                live_jobs = self._run(
                    [
                        str(venv_python),
                        "-c",
                        """
import json
import sys
import time
from pathlib import Path

import torch
import yolozu
from yolozu.adapter import RTDETRPoseAdapter
from yolozu.integrations import tool_runner

prefix = Path(sys.prefix).resolve()
module = Path(yolozu.__file__).resolve()
if not module.is_relative_to(prefix):
    raise SystemExit(f"candidate package escaped venv: {module}")

config = Path("tiny_rtdetr.json")
config.write_text(json.dumps({
    "dataset": {"root": ".", "split": "val", "format": "yolo"},
    "model": {
        "num_classes": 1,
        "hidden_dim": 64,
        "num_queries": 10,
        "stem_channels": 8,
        "backbone_channels": [16, 32, 64],
        "stage_blocks": [1, 1, 1],
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "nhead": 4,
        "encoder_dim_feedforward": 128,
        "decoder_dim_feedforward": 128,
    },
    "train": {"batch_size": 1, "lr": 0.0001, "epochs": 1},
}), encoding="utf-8")
adapter = RTDETRPoseAdapter(
    config_path=str(config),
    device="cpu",
    image_size=(32, 32),
)
checkpoint = Path("compatible.pt")
torch.save(adapter.get_model().state_dict(), checkpoint)

results = {}
cases = (
    ("ttt", tool_runner.ttt_job, "tent", "sample"),
    ("ctta", tool_runner.ctta_job, "cotta", "stream"),
)
for lane, submit, method, reset in cases:
    output = f"runs/{lane}/predictions.json"
    report = f"runs/{lane}/report.json"
    queued = submit(
        "data/smoke",
        str(checkpoint),
        output,
        config=str(config),
        report=report,
        method=method,
        reset=reset,
        max_images=1,
    )
    if not queued.get("ok"):
        raise SystemExit(json.dumps(queued, sort_keys=True))
    deadline = time.monotonic() + 60
    terminal = {}
    while time.monotonic() < deadline:
        terminal = tool_runner.jobs_status(queued["job_id"])
        status = (terminal.get("job") or {}).get("status")
        if status in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.02)
    job = terminal.get("job") or {}
    result = job.get("result") or {}
    if job.get("status") != "completed" or not result.get("ok"):
        raise SystemExit(json.dumps(terminal, sort_keys=True))
    if not Path(output).is_file() or not Path(report).is_file():
        raise SystemExit(f"missing {lane} artifacts")
    results[lane] = {
        "job_status": job["status"],
        "exit_code": result["exit_code"],
        "preflight": queued["preflight"]["status"],
        "method": json.loads(
            Path(report).read_text(encoding="utf-8")
        )["ttt"]["method"],
    }
print(json.dumps({
    "module": str(module),
    "results": results,
}, sort_keys=True))
""",
                    ],
                    cwd=consumer,
                    env=clean_env,
                )
                self.assertEqual(
                    live_jobs.returncode,
                    0,
                    msg=(
                        "installed TTT/CTTA jobs failed outside checkout:\n"
                        f"{live_jobs.stdout}\n{live_jobs.stderr}"
                    ),
                )
                live_payload = json.loads(live_jobs.stdout)
                self.assertTrue(
                    Path(live_payload["module"]).resolve().is_relative_to(
                        venv_dir.resolve()
                    )
                )
                self.assertEqual(
                    live_payload["results"],
                    {
                        "ctta": {
                            "exit_code": 0,
                            "job_status": "completed",
                            "method": "cotta",
                            "preflight": "full",
                        },
                        "ttt": {
                            "exit_code": 0,
                            "job_status": "completed",
                            "method": "tent",
                            "preflight": "full",
                        },
                    },
                )


if __name__ == "__main__":
    unittest.main()
