import json
import os
import sys
import time
import unittest
import unittest.mock

from yolozu.adaptive import build_environment_profile, validate_environment_profile
from yolozu.adaptive import environment as environment_mod
from yolozu.core import doctor as doctor_mod


def _ok_text(value: str) -> environment_mod._ProbeRun:
    return environment_mod._ProbeRun("ok", stdout=value.encode("utf-8"))


def _ok_json(value: object) -> environment_mod._ProbeRun:
    return _ok_text(json.dumps(value, separators=(",", ":")))


class _FakeRunner:
    def __init__(self, results: dict[str, environment_mod._ProbeRun] | None = None):
        self.results = results or {}
        self.calls: list[tuple[str, float]] = []

    def __call__(
        self, spec: environment_mod._ProbeSpec, timeout_seconds: float
    ) -> environment_mod._ProbeRun:
        self.calls.append((spec.probe_id, timeout_seconds))
        if spec.probe_id in self.results:
            return self.results[spec.probe_id]
        if spec.probe_id == "runtime_trtexec":
            return environment_mod._ProbeRun("unsupported", code="executable_unavailable")
        if spec.probe_id.startswith("runtime_"):
            return _ok_json({"status": "absent"})
        return environment_mod._ProbeRun("unsupported", code="executable_unavailable")


def _facts(system: str, **updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "system": system,
        "release": "1.2.3",
        "machine": "test_arch",
        "processor": "Test CPU",
        "logical_cores": 8,
        "total_memory_bytes": 16 * 1024**3,
    }
    record.update(updates)
    return record


def _linux_results() -> dict[str, environment_mod._ProbeRun]:
    return {
        "linux_cpu": _ok_json(
            {
                "lscpu": [
                    {"field": "Model name:", "data": "Fixture CPU"},
                    {"field": "CPU(s):", "data": "8"},
                    {"field": "Core(s) per socket:", "data": "4"},
                    {"field": "Socket(s):", "data": "1"},
                ]
            }
        ),
        "linux_accelerators": _ok_json([]),
        "nvidia_accelerators": _ok_text(""),
        "linux_power_mode": _ok_text("balanced\n"),
    }


class TestAdaptiveEnvironmentProfile(unittest.TestCase):
    def test_confirmed_linux_cpu_only_profile_is_valid(self) -> None:
        runner = _FakeRunner(_linux_results())
        profile = build_environment_profile(
            probe_runner=runner,
            platform_facts=_facts("Linux"),
            collected_at="2026-08-25T00:00:00Z",
        ).to_dict()

        self.assertEqual(validate_environment_profile(profile).to_dict(), profile)
        self.assertEqual(profile["cpu"]["model"], "Fixture CPU")
        self.assertEqual(profile["cpu"]["physical_cores"]["value"], 4)
        self.assertFalse(
            any(item["probe_status"] == "present" for item in profile["accelerators"])
        )
        self.assertTrue(
            any(item["probe_status"] == "absent" for item in profile["accelerators"])
        )
        self.assertEqual(profile["power_performance_mode"]["mode"], "balanced")
        self.assertLessEqual(len(runner.calls), environment_mod.MAX_PROBES)
        self.assertTrue(
            all(timeout <= environment_mod.PROBE_TIMEOUT_SECONDS for _, timeout in runner.calls)
        )

    def test_darwin_apple_silicon_fixture(self) -> None:
        results = {
            "darwin_cpu_model": _ok_text("Apple M4 Max\n"),
            "darwin_physical_cores": _ok_text("12\n"),
            "darwin_total_memory": _ok_text(str(64 * 1024**3)),
            "darwin_accelerators": _ok_json(
                {
                    "SPDisplaysDataType": [
                        {
                            "sppci_model": "Apple M4 Max",
                            "spdisplays_vram_shared": "64 GB",
                        }
                    ]
                }
            ),
            "darwin_power_mode": _ok_text("Currently in use:\n lowpowermode 1\n"),
        }
        profile = build_environment_profile(
            probe_runner=_FakeRunner(results),
            platform_facts=_facts("Darwin", total_memory_bytes=None),
            collected_at="2026-08-25T00:00:00Z",
        ).to_dict()

        self.assertEqual(profile["cpu"]["model"], "Apple M4 Max")
        self.assertEqual(profile["cpu"]["physical_cores"]["value"], 12)
        self.assertEqual(profile["total_memory"]["value_bytes"], 64 * 1024**3)
        self.assertEqual(profile["accelerators"][0]["vendor"], "Apple")
        self.assertEqual(profile["power_performance_mode"]["mode"], "low_power")

    def test_nvidia_and_runtime_fixture(self) -> None:
        results = _linux_results()
        results.update(
            {
                "linux_accelerators": _ok_json(
                    [
                        {
                            "class": "display",
                            "vendor": "NVIDIA Corporation",
                            "product": "Test GPU",
                            "size": 8 * 1024**3,
                        }
                    ]
                ),
                "nvidia_accelerators": _ok_text("Test GPU, 8192\n"),
                "runtime_torch": _ok_json(
                    {
                        "status": "present",
                        "version": "2.9.0",
                        "providers": ["cpu", "cuda"],
                        "accelerators": [
                            {
                                "kind": "gpu",
                                "vendor": "NVIDIA",
                                "model": "Test GPU",
                                "memory_bytes": 8 * 1024**3,
                            }
                        ],
                    }
                ),
            }
        )
        profile = build_environment_profile(
            probe_runner=_FakeRunner(results),
            platform_facts=_facts("Linux"),
            collected_at="2026-08-25T00:00:00Z",
        ).to_dict()

        self.assertTrue(
            any(item.get("vendor") == "NVIDIA" for item in profile["accelerators"])
        )
        torch = next(item for item in profile["runtimes"] if item["runtime_id"] == "torch")
        self.assertEqual(torch["provider_ids"], ["cpu", "cuda"])

    def test_unsupported_os_and_runtime_import_failure_stay_unknown(self) -> None:
        runner = _FakeRunner(
            {"runtime_torch": _ok_json({"status": "failed", "code": "import_failed"})}
        )
        profile = build_environment_profile(
            probe_runner=runner,
            platform_facts=_facts("OtherOS"),
            collected_at="2026-08-25T00:00:00Z",
        ).to_dict()

        inventory = next(
            item
            for item in profile["accelerators"]
            if item["accelerator_id"] == "host_accelerator_inventory"
        )
        self.assertEqual(inventory["probe_status"], "unsupported")
        self.assertFalse(
            any(item["probe_status"] == "absent" for item in profile["accelerators"])
        )
        torch = next(item for item in profile["runtimes"] if item["runtime_id"] == "torch")
        self.assertEqual(torch["probe_status"], "failed")
        self.assertIn(
            {"probe_id": "runtime_torch", "status": "failed", "code": "import_failed"},
            profile["probe_issues"],
        )

    def test_malformed_and_exception_probes_become_bounded_failures(self) -> None:
        results = _linux_results()
        results["linux_accelerators"] = _ok_text("not-json")

        class RaisingRunner(_FakeRunner):
            def __call__(self, spec, timeout_seconds):  # type: ignore[no-untyped-def]
                if spec.probe_id == "runtime_cv2":
                    raise RuntimeError("private raw exception")
                return super().__call__(spec, timeout_seconds)

        profile = build_environment_profile(
            probe_runner=RaisingRunner(results),
            platform_facts=_facts("Linux"),
            collected_at="2026-08-25T00:00:00Z",
        ).to_dict()
        serialized = json.dumps(profile)

        self.assertNotIn("private raw exception", serialized)
        self.assertIn("malformed_output", serialized)
        self.assertIn("probe_exception", serialized)
        self.assertLessEqual(len(serialized.encode("utf-8")), 65_536)

    def test_total_deadline_marks_unstarted_probes_failed(self) -> None:
        values = iter([0.0] + [31.0] * 64)
        profile = build_environment_profile(
            probe_runner=_FakeRunner(_linux_results()),
            platform_facts=_facts("Linux"),
            collected_at="2026-08-25T00:00:00Z",
            monotonic=lambda: next(values),
        ).to_dict()

        self.assertTrue(
            any(issue["code"] == "total_deadline" for issue in profile["probe_issues"])
        )
        self.assertEqual(
            next(item for item in profile["runtimes"] if item["runtime_id"] == "torch")[
                "probe_status"
            ],
            "failed",
        )

    def test_fingerprint_excludes_collection_time_and_tracks_power_mode(self) -> None:
        results = _linux_results()
        first = build_environment_profile(
            probe_runner=_FakeRunner(results),
            platform_facts=_facts("Linux"),
            collected_at="2026-08-25T00:00:00Z",
        ).to_dict()
        second = build_environment_profile(
            probe_runner=_FakeRunner(results),
            platform_facts=_facts("Linux"),
            collected_at="2026-08-26T00:00:00Z",
        ).to_dict()
        changed = dict(results)
        changed["linux_power_mode"] = _ok_text("performance\n")
        third = build_environment_profile(
            probe_runner=_FakeRunner(changed),
            platform_facts=_facts("Linux"),
            collected_at="2026-08-26T00:00:00Z",
        ).to_dict()

        self.assertEqual(first["environment_fingerprint"], second["environment_fingerprint"])
        self.assertNotEqual(first["environment_fingerprint"], third["environment_fingerprint"])

    def test_sensitive_probe_text_is_not_returned(self) -> None:
        results = _linux_results()
        results["linux_cpu"] = _ok_json(
            {"lscpu": [{"field": "Model name:", "data": "host.example/Users/alice"}]}
        )
        profile = build_environment_profile(
            probe_runner=_FakeRunner(results),
            platform_facts=_facts("Linux"),
            collected_at="2026-08-25T00:00:00Z",
        ).to_dict()
        serialized = json.dumps(profile)

        self.assertNotIn("alice", serialized)
        self.assertNotIn("host.example", serialized)

    def test_doctor_additively_emits_valid_profile(self) -> None:
        profile = build_environment_profile(
            probe_runner=_FakeRunner(_linux_results()),
            platform_facts=_facts("Linux"),
            collected_at="2026-08-25T00:00:00Z",
        ).to_dict()
        with unittest.mock.patch.object(doctor_mod, "build_environment_profile", return_value=profile):
            with unittest.mock.patch.object(
                doctor_mod,
                "_gather_required_runtime",
                return_value=({"numpy": {"available": True}}, []),
            ):
                with unittest.mock.patch.object(doctor_mod, "_gather_gpu_info", return_value={}):
                    with unittest.mock.patch.object(
                        doctor_mod, "_gather_runtime_capabilities", return_value={}
                    ):
                        with unittest.mock.patch.object(
                            doctor_mod, "_gather_git_info", return_value={"head": None, "dirty": None}
                        ):
                            report, exit_code = doctor_mod.build_doctor_report()

        self.assertEqual(exit_code, 0)
        self.assertEqual(validate_environment_profile(report["environment_profile"]).to_dict(), profile)
        for existing_key in ("timestamp", "gpu", "env", "runtime_capabilities", "drift_hints"):
            self.assertIn(existing_key, report)

    def test_doctor_legacy_runtime_meanings_use_bounded_diagnostics(self) -> None:
        results = _linux_results()
        results["nvidia_accelerators"] = _ok_text("Test GPU, 8192\n")
        results["runtime_torch"] = _ok_json(
            {
                "status": "present",
                "version": "2.9.0",
                "providers": ["cpu", "cuda"],
                "accelerators": [],
                "diagnostics": {
                    "cuda_available": True,
                    "cuda_version": "13.0",
                    "cudnn_version": 91000,
                    "device_count": 1,
                    "mps_built": False,
                    "mps_available": False,
                },
            }
        )
        profile, diagnostics = environment_mod._build_environment_observation(
            probe_runner=_FakeRunner(results),
            platform_facts=_facts("Linux"),
            collected_at="2026-08-25T00:00:00Z",
        )
        payload = profile.to_dict()
        gpu = doctor_mod._gather_gpu_info(
            environment_profile=payload,
            runtime_diagnostics=diagnostics,
        )
        runtime = doctor_mod._gather_runtime_capabilities(
            tools={"nvidia_smi": True, "trtexec": False},
            gpu=gpu,
            environment_profile=payload,
            runtime_diagnostics=diagnostics,
        )

        self.assertEqual(gpu["nvidia_smi_list"], ["GPU 0: Test GPU"])
        self.assertEqual(runtime["cuda"]["gpu_count_from_nvidia_smi"], 1)
        self.assertTrue(runtime["torch"]["installed"])
        self.assertTrue(runtime["torch"]["cuda_available"])
        self.assertEqual(runtime["torch"]["cuda_version"], "13.0")
        self.assertEqual(runtime["torch"]["cudnn_version"], 91000)
        self.assertEqual(runtime["torch"]["device_count"], 1)


class TestBoundedEnvironmentProbeProcess(unittest.TestCase):
    def _spec(self, probe_id: str) -> environment_mod._ProbeSpec:
        return next(
            spec for spec in environment_mod._LIMIT_TEST_SPECS if spec.probe_id == probe_id
        )

    def test_timeout_and_process_tree_deadlines_terminate(self) -> None:
        for probe_id in ("limit_test_timeout", "limit_test_process_tree"):
            started = time.monotonic()
            result = environment_mod._run_bounded_probe(self._spec(probe_id), 0.1)
            elapsed = time.monotonic() - started
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.code, "timeout")
            self.assertLess(elapsed, 2.0)

    def test_output_flood_is_bounded(self) -> None:
        result = environment_mod._run_bounded_probe(self._spec("limit_test_output"), 2.0)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.code, "stdout_limit")
        self.assertEqual(result.stdout, b"")

    def test_disallowed_command_or_arguments_are_rejected(self) -> None:
        disallowed = environment_mod._ProbeSpec(
            "runtime_torch",
            ((str(sys.executable), "-c", "print('caller-controlled')"),),
        )
        with self.assertRaisesRegex(ValueError, "not code-owned"):
            environment_mod._run_bounded_probe(disallowed, 1.0)

    def test_subprocess_does_not_inherit_caller_environment(self) -> None:
        original = environment_mod.subprocess.Popen
        captured_env: dict[str, str] = {}

        def recording_popen(*args, **kwargs):  # type: ignore[no-untyped-def]
            captured_env.update(kwargs.get("env") or {})
            return original(*args, **kwargs)

        with unittest.mock.patch.dict(os.environ, {"YOLOZU_PRIVATE_TEST_VALUE": "secret"}):
            with unittest.mock.patch.object(environment_mod.subprocess, "Popen", recording_popen):
                result = environment_mod._run_bounded_probe(
                    self._spec("limit_test_output"), 2.0
                )
        self.assertEqual(result.code, "stdout_limit")
        self.assertNotIn("YOLOZU_PRIVATE_TEST_VALUE", captured_env)
        self.assertNotIn("secret", captured_env.values())


if __name__ == "__main__":
    unittest.main()
