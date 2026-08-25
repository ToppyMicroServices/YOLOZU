from __future__ import annotations

import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

from PIL import Image

from tests.test_adaptive_bundle_contracts import _bundle_payload
from yolozu.adaptive.artifact_resolver import ArtifactResolver
from yolozu.adaptive.bundles import validate_algorithm_bundle_spec
from yolozu.adaptive.contracts import validate_environment_profile, validate_image_job_spec
from yolozu.adaptive.inventory import pin_decoded_inputs
from yolozu.adaptive.qualification import (
    _ForkedRunnerSession,
    _collect_report,
    _sustained_summary,
    nanoseconds_to_milliseconds,
    nearest_rank_nanoseconds,
    qualification_input_schedule,
    qualify_image_pipeline,
)
from yolozu.adaptive.bundle_registry import RunnerProbeResult


class _Clock:
    def __init__(self) -> None:
        self.value = 1_000_000_000

    def now(self) -> int:
        return self.value

    def advance(self, value: int) -> None:
        self.value += value


class _Session:
    runner_id = "onnxruntime"
    runner_version = "1.23.0"

    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.predicted_indices: list[int] = []
        self.warmup_indices: list[int] = []
        self.closed = False

    def probe(self, timeout_seconds: int) -> RunnerProbeResult:
        self.clock.advance(1_000_000)
        return RunnerProbeResult("supported")

    def load(self, timeout_seconds: int) -> None:
        self.clock.advance(2_000_000)

    def warmup(self, index: int, timeout_seconds: int) -> None:
        self.warmup_indices.append(index)
        self.clock.advance(500_000)

    def predict(self, index: int, timeout_seconds: int) -> tuple[Mapping[str, Any], ...]:
        self.predicted_indices.append(index)
        self.clock.advance(1_000_000)
        return (
            {
                "native_class_index": 0,
                "score": "0.9",
                "bbox": ["0", "0", "1", "1"],
            },
        )

    def close(self, timeout_seconds: int) -> None:
        self.closed = True


class _SoakSession(_Session):
    def predict(self, index: int, timeout_seconds: int) -> tuple[Mapping[str, Any], ...]:
        self.predicted_indices.append(index)
        self.clock.advance(
            300_000_000_000 if len(self.predicted_indices) > 601 else 1_000_000
        )
        return (
            {
                "native_class_index": 0,
                "score": "0.9",
                "bbox": ["0", "0", "1", "1"],
            },
        )


class _WrongRunnerSession(_Session):
    runner_id = "wrong-runner"


class _Evaluator:
    evaluator_id = "fixture-evaluator"
    evaluator_version = "1"
    source_digest = "9" * 64
    metric_id = "fixture_metric"
    direction = "higher_is_better"
    evaluation_dataset_id = "fixture-dataset"
    evaluation_dataset_sha256 = "a" * 64
    evaluation_protocol_sha256 = "b" * 64
    evaluation_vocabulary_id = "example-classes"

    def __init__(self) -> None:
        self.prediction_count = 0

    def evaluate(self, *, predictions: tuple[bytes, ...], job: Any, bundle: Any) -> str:
        self.prediction_count = len(predictions)
        return "0.6"


class _BlockingRunner:
    runner_id = "fixture-blocking"
    runner_version = "1"

    def probe(self, *, bundle: Any, environment: Any) -> RunnerProbeResult:
        return RunnerProbeResult("supported")

    def load(self, *, bundle: Any, artifacts: Any) -> None:
        return None

    def warmup(self, *, input_item: Any) -> None:
        time.sleep(5)

    def predict(
        self,
        *,
        input_item: Any,
        requested_labels: tuple[str, ...],
    ) -> tuple[Mapping[str, Any], ...]:
        return ()

    def close(self) -> None:
        return None


def _blocking_runner_factory() -> _BlockingRunner:
    return _BlockingRunner()


class TestAdaptiveQualification(unittest.TestCase):
    def _job(
        self,
        *,
        input_mode: str,
        max_images: int,
        quality: bool = False,
        execution_mode: str = "batch",
        memory_gate: bool = False,
    ) -> Any:
        value = {
            "schema_version": 1,
            "task": "object_detection",
            "prompt_mode": "fixed_classes",
            "fixed_classes": ["cat"],
            "input_mode": input_mode,
            "execution_mode": execution_mode,
            "batch_size": 1,
            "concurrency": 1,
            "max_images": max_images,
            "max_results_per_image": 100,
            "job_timeout_seconds": 60,
            "ranking_policy": "latency_first",
            "allowed_maturities": ["Experimental"],
            "network_policy": "deny",
            "compute_policy": "auto",
        }
        if execution_mode == "batch":
            value["min_repeat_throughput_fps"] = "1"
        if quality:
            value["quality_requirement"] = {
                "metric_id": "fixture_metric",
                "direction": "higher_is_better",
                "threshold": "0.5",
                "evaluation_dataset_id": "fixture-dataset",
                "evaluation_dataset_sha256": "a" * 64,
                "evaluation_protocol_sha256": "b" * 64,
                "evaluation_vocabulary_id": "example-classes",
            }
        if memory_gate:
            value["max_runner_tree_peak_rss_bytes"] = 1_000_000
        return validate_image_job_spec(value)

    def _environment(self) -> Any:
        from tests.test_adaptive_image_contracts import TestAdaptiveImageContracts

        helper = TestAdaptiveImageContracts()
        return validate_environment_profile(helper._environment_payload())

    def test_nearest_rank_known_vectors_and_units(self) -> None:
        values = list(range(1, 101))
        self.assertEqual(nearest_rank_nanoseconds(values, 50), 50)
        self.assertEqual(nearest_rank_nanoseconds(values, 95), 95)
        self.assertEqual(nearest_rank_nanoseconds(values, 99), 99)
        self.assertEqual(nearest_rank_nanoseconds([4, 1, 3, 2], 50), 2)
        self.assertEqual(nanoseconds_to_milliseconds(0), "0")
        self.assertEqual(nanoseconds_to_milliseconds(1_234_567), "1.234567")
        with self.assertRaises(ValueError):
            nearest_rank_nanoseconds([1], 90)
        with self.assertRaises(ValueError):
            nearest_rank_nanoseconds([True], 50)

    def test_frozen_schedule_covers_1_2_20_and_100_inputs(self) -> None:
        for input_count in (1, 2, 20, 100):
            schedule = qualification_input_schedule(input_count, 200)
            self.assertEqual(schedule[0], 0)
            counts = [schedule.count(index) for index in range(input_count)]
            self.assertEqual(sum(counts), 200)
            self.assertTrue(all(count >= 2 for count in counts))
        with self.assertRaises(ValueError):
            qualification_input_schedule(100, 199)

    def test_collects_exact_three_repeat_unactivated_report_with_injected_clock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            for index in range(2):
                Image.new("RGB", (4, 3), color=(index, 0, 0)).save(images / f"{index}.png")
            job = self._job(input_mode="bounded_directory", max_images=2)
            bundle = validate_algorithm_bundle_spec(_bundle_payload())
            environment = self._environment()
            clock = _Clock()
            session = _Session(clock)
            with pin_decoded_inputs(
                images,
                input_mode="bounded_directory",
                workspace_root=root,
                max_images=2,
            ) as inputs:
                report = _collect_report(
                    session=session,
                    bundle=bundle,
                    job=job,
                    environment=environment,
                    inputs=inputs,
                    workload_fingerprint="7" * 64,
                    artifact_state_fingerprint="5" * 64,
                    started=datetime(2026, 8, 25, tzinfo=timezone.utc),
                    smoke=False,
                    monotonic_ns=clock.now,
                    utc_now=lambda: datetime(2026, 8, 25, 0, 1, tzinfo=timezone.utc),
                    cold_started_ns=clock.now(),
                )
            value = report.to_dict()
            self.assertEqual(value["status"], "qualified")
            self.assertEqual(len(value["repeats"]), 3)
            self.assertEqual(
                [item["input_coverage_counts"] for item in value["repeats"]],
                [[100, 100], [100, 100], [100, 100]],
            )
            self.assertEqual(value["conservative_aggregates"]["p95_latency_ms"], "1")
            self.assertEqual(session.warmup_indices, [index % 2 for index in range(20)])
            self.assertEqual(len(session.predicted_indices), 601)
            self.assertTrue(session.closed)
            self.assertIn("unactivated", value["limitations"][-1])

    def test_smoke_never_becomes_qualified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "one.png"
            Image.new("RGB", (2, 2)).save(image_path)
            job = self._job(input_mode="single_image", max_images=1)
            bundle = validate_algorithm_bundle_spec(_bundle_payload())
            environment = self._environment()
            clock = _Clock()
            with pin_decoded_inputs(
                image_path,
                input_mode="single_image",
                workspace_root=root,
                max_images=1,
            ) as inputs:
                report = _collect_report(
                    session=_Session(clock),
                    bundle=bundle,
                    job=job,
                    environment=environment,
                    inputs=inputs,
                    workload_fingerprint="7" * 64,
                    artifact_state_fingerprint="5" * 64,
                    started=datetime(2026, 8, 25, tzinfo=timezone.utc),
                    smoke=True,
                    monotonic_ns=clock.now,
                    utc_now=lambda: datetime(2026, 8, 25, 0, 1, tzinfo=timezone.utc),
                    cold_started_ns=clock.now(),
                )
            self.assertEqual(report.to_dict()["status"], "smoke")
            self.assertIsNone(report.to_dict()["conservative_aggregates"])

    def test_quality_uses_one_prediction_per_unique_input_outside_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            for index in range(2):
                Image.new("RGB", (4, 3), color=(index, 0, 0)).save(images / f"{index}.png")
            job = self._job(input_mode="bounded_directory", max_images=2, quality=True)
            bundle = validate_algorithm_bundle_spec(_bundle_payload())
            environment = self._environment()
            clock = _Clock()
            session = _Session(clock)
            evaluator = _Evaluator()
            with pin_decoded_inputs(
                images,
                input_mode="bounded_directory",
                workspace_root=root,
                max_images=2,
            ) as inputs:
                report = _collect_report(
                    session=session,
                    bundle=bundle,
                    job=job,
                    environment=environment,
                    inputs=inputs,
                    workload_fingerprint="7" * 64,
                    artifact_state_fingerprint="5" * 64,
                    started=datetime(2026, 8, 25, tzinfo=timezone.utc),
                    smoke=False,
                    monotonic_ns=clock.now,
                    utc_now=lambda: datetime(2026, 8, 25, 0, 1, tzinfo=timezone.utc),
                    cold_started_ns=clock.now(),
                    evaluator=evaluator,
                )
            self.assertEqual(report.to_dict()["quality"]["status"], "known")
            self.assertEqual(evaluator.prediction_count, 2)
            self.assertEqual(len(session.predicted_indices), 603)

    def test_missing_evaluator_keeps_quality_unknown_and_holds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "one.png"
            Image.new("RGB", (2, 2)).save(image_path)
            clock = _Clock()
            with pin_decoded_inputs(
                image_path,
                input_mode="single_image",
                workspace_root=root,
                max_images=1,
            ) as inputs:
                report = _collect_report(
                    session=_Session(clock),
                    bundle=validate_algorithm_bundle_spec(_bundle_payload()),
                    job=self._job(input_mode="single_image", max_images=1, quality=True),
                    environment=self._environment(),
                    inputs=inputs,
                    workload_fingerprint="7" * 64,
                    artifact_state_fingerprint="5" * 64,
                    started=datetime(2026, 8, 25, tzinfo=timezone.utc),
                    smoke=False,
                    monotonic_ns=clock.now,
                    utc_now=lambda: datetime(2026, 8, 25, 0, 1, tzinfo=timezone.utc),
                    cold_started_ns=clock.now(),
                )
            self.assertEqual(report.to_dict()["status"], "hold")
            self.assertEqual(report.to_dict()["quality"]["status"], "unknown")
            self.assertIn("quality_unknown", report.to_dict()["failures"])

    def test_unknown_complete_memory_cannot_pass_a_hard_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "one.png"
            Image.new("RGB", (2, 2)).save(image_path)
            clock = _Clock()
            with pin_decoded_inputs(
                image_path,
                input_mode="single_image",
                workspace_root=root,
                max_images=1,
            ) as inputs:
                report = _collect_report(
                    session=_Session(clock),
                    bundle=validate_algorithm_bundle_spec(_bundle_payload()),
                    job=self._job(
                        input_mode="single_image",
                        max_images=1,
                        memory_gate=True,
                    ),
                    environment=self._environment(),
                    inputs=inputs,
                    workload_fingerprint="7" * 64,
                    artifact_state_fingerprint="5" * 64,
                    started=datetime(2026, 8, 25, tzinfo=timezone.utc),
                    smoke=False,
                    monotonic_ns=clock.now,
                    utc_now=lambda: datetime(2026, 8, 25, 0, 1, tzinfo=timezone.utc),
                    cold_started_ns=clock.now(),
                )
            self.assertEqual(report.to_dict()["status"], "hold")
            self.assertIn("runner_tree_memory_unknown", report.to_dict()["failures"])

    def test_runner_identity_mismatch_fails_before_probe_or_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "one.png"
            Image.new("RGB", (2, 2)).save(image_path)
            clock = _Clock()
            session = _WrongRunnerSession(clock)
            with pin_decoded_inputs(
                image_path,
                input_mode="single_image",
                workspace_root=root,
                max_images=1,
            ) as inputs:
                report = _collect_report(
                    session=session,
                    bundle=validate_algorithm_bundle_spec(_bundle_payload()),
                    job=self._job(input_mode="single_image", max_images=1),
                    environment=self._environment(),
                    inputs=inputs,
                    workload_fingerprint="7" * 64,
                    artifact_state_fingerprint="5" * 64,
                    started=datetime(2026, 8, 25, tzinfo=timezone.utc),
                    smoke=False,
                    monotonic_ns=clock.now,
                    utc_now=lambda: datetime(2026, 8, 25, 0, 1, tzinfo=timezone.utc),
                    cold_started_ns=clock.now(),
                )
            self.assertEqual(report.to_dict()["status"], "failed")
            self.assertEqual(report.to_dict()["failures"], ["runner_identity_mismatch"])
            self.assertEqual(session.predicted_indices, [])

    def test_soft_realtime_soak_uses_injected_clock_without_wall_wait(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            for index in range(2):
                Image.new("RGB", (4, 3), color=(index, 0, 0)).save(images / f"{index}.png")
            job = self._job(
                input_mode="bounded_directory",
                max_images=2,
                execution_mode="soft_realtime",
            )
            bundle = validate_algorithm_bundle_spec(_bundle_payload())
            environment = self._environment()
            clock = _Clock()
            session = _SoakSession(clock)
            with pin_decoded_inputs(
                images,
                input_mode="bounded_directory",
                workspace_root=root,
                max_images=2,
            ) as inputs:
                report = _collect_report(
                    session=session,
                    bundle=bundle,
                    job=job,
                    environment=environment,
                    inputs=inputs,
                    workload_fingerprint="7" * 64,
                    artifact_state_fingerprint="5" * 64,
                    started=datetime(2026, 8, 25, tzinfo=timezone.utc),
                    smoke=False,
                    monotonic_ns=clock.now,
                    utc_now=lambda: datetime(2026, 8, 25, 0, 1, tzinfo=timezone.utc),
                    cold_started_ns=clock.now(),
                )
            sustained = report.to_dict()["sustained_section"]
            self.assertEqual(sustained["status"], "completed")
            self.assertEqual(sustained["duration_ns"], 600_000_000_000)
            self.assertEqual(sustained["processed_count"], 2)
            self.assertEqual(session.predicted_indices[-2:], [0, 1])

    def test_soft_realtime_fails_if_sample_cap_arrives_before_duration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "one.png"
            Image.new("RGB", (2, 2)).save(image_path)
            clock = _Clock()
            with pin_decoded_inputs(
                image_path,
                input_mode="single_image",
                workspace_root=root,
                max_images=1,
            ) as inputs, patch(
                "yolozu.adaptive.qualification.MAX_SUSTAINED_SAMPLES", 2
            ):
                with self.assertRaisesRegex(ValueError, "sustained_sample_limit"):
                    _sustained_summary(
                        session=_Session(clock),
                        inputs=inputs,
                        job=self._job(
                            input_mode="single_image",
                            max_images=1,
                            execution_mode="soft_realtime",
                        ),
                        bundle=validate_algorithm_bundle_spec(_bundle_payload()),
                        environment=self._environment(),
                        monotonic_ns=clock.now,
                    )

    def test_unbound_packaged_candidate_fails_without_dummy_evidence(self) -> None:
        job = self._job(input_mode="single_image", max_images=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "bundle_ineligible"):
                qualify_image_pipeline(
                    job=job,
                    input_path="missing.png",
                    bundle_id="yolox-s-coco",
                    bundle_version="0.1.1rc0",
                    output_dir="reports/qualification",
                    workspace_root=root,
                )
            self.assertFalse((root / "reports").exists())

    def test_pinned_input_rejects_directory_entry_swap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "one.png"
            Image.new("RGB", (2, 2), color=(1, 2, 3)).save(image_path)
            with pin_decoded_inputs(
                image_path,
                input_mode="single_image",
                workspace_root=root,
                max_images=1,
            ) as inputs:
                image_path.unlink()
                Image.new("RGB", (2, 2), color=(3, 2, 1)).save(image_path)
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    inputs[0].read_source_bytes()

    def test_pinned_artifact_rejects_cache_entry_swap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_root = root / "artifacts"
            artifact_path = artifact_root / "weights" / "model.onnx"
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_bytes(b"model")
            bundle = validate_algorithm_bundle_spec(_bundle_payload())
            with ArtifactResolver(workspace=root, artifact_root=artifact_root) as resolver:
                with resolver.pin(bundle) as artifacts:
                    artifact_path.unlink()
                    artifact_path.write_bytes(b"model")
                    with self.assertRaisesRegex(ValueError, "identity changed"):
                        artifacts.read_artifact_chunk(
                            "model", offset_bytes=0, maximum_bytes=5
                        )

    def test_phase_timeout_terminates_and_reaps_runner_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "one.png"
            Image.new("RGB", (2, 2)).save(image_path)
            artifact_root = root / "artifacts"
            artifact_path = artifact_root / "weights" / "model.onnx"
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_bytes(b"model")
            bundle = validate_algorithm_bundle_spec(_bundle_payload())
            with (
                pin_decoded_inputs(
                    image_path,
                    input_mode="single_image",
                    workspace_root=root,
                    max_images=1,
                ) as inputs,
                ArtifactResolver(workspace=root, artifact_root=artifact_root) as resolver,
                resolver.pin(bundle) as artifacts,
            ):
                session = _ForkedRunnerSession(
                    factory=_blocking_runner_factory,
                    bundle=bundle,
                    environment=self._environment(),
                    artifacts=artifacts,
                    inputs=inputs,
                    labels=("cat",),
                    outer_deadline_ns=time.monotonic_ns() + 10_000_000_000,
                )
                with self.assertRaisesRegex(ValueError, "phase_timeout"):
                    session.warmup(0, 0)
                self.assertFalse(session._process.is_alive())
                session.close(1)


if __name__ == "__main__":
    unittest.main()
