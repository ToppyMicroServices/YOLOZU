from __future__ import annotations

import copy
import hashlib
import json
import unittest
from collections.abc import Iterator
from pathlib import Path

from tests.test_adaptive_image_contracts import _schema_accepts
from yolozu.adaptive.canonical import canonical_json_v1, canonical_sha256_v1
from yolozu.adaptive.streaming import (
    StreamContractError,
    build_stream_workload_profile,
    compute_max_consecutive_drops,
    compute_stream_source_digest,
    drop_fraction_within_limit,
    sustained_fps_meets_minimum,
    validate_frame_result,
    validate_stream_job_spec,
    validate_stream_output_artifacts,
    validate_stream_qualification_report,
    validate_stream_selection_decision,
    validate_stream_summary,
    validate_stream_workload_profile,
)


def _decoder_policy() -> dict:
    record = {
        "policy_id": "bounded_h264_decoder_v1",
        "policy_version": 1,
        "backend_id": "fixture_decoder",
        "backend_version": "1",
        "enforcement_id": "bounded_process_tree_hard_limit_v1",
        "allowed_profiles": ["constrained_baseline", "main", "high"],
        "allowed_levels": ["3.0", "3.1", "3.2", "4.0", "4.1", "4.2", "5.0", "5.1"],
        "pixel_format": "yuv420p8",
        "max_reference_frames": 16,
        "max_dpb_bytes": 536_870_912,
        "max_gop_frames": 600,
        "max_sample_bytes": 67_108_864,
        "max_nal_unit_bytes": 33_554_432,
        "probe_input_bytes": 67_108_864,
        "probe_box_count": 100_000,
        "probe_nesting_depth": 32,
        "policy_digest": "0" * 64,
    }
    record["policy_digest"] = canonical_sha256_v1(
        record, own_digest_field="policy_digest"
    )
    return record


def _job(*, task: str = "object_detection", policy: str = "block") -> dict:
    return {
        "schema_version": 1,
        "task": task,
        "source": {
            "source_kind": "local_mp4",
            "width": 640,
            "height": 480,
            "source_rate_num": 30,
            "source_rate_den": 1,
            "container": "mp4",
            "codec": "h264_avc",
            "pixel_format": "yuv420p8",
            "provider_id": None,
            "capability_format": None,
        },
        "required_duration_seconds": 600,
        "warmup_frame_count": 0,
        "min_sustained_fps": "1",
        "max_p95_latency_ms": "100",
        "queue_capacity_frames": 4,
        "max_queued_decoded_bytes": 2_000_000,
        "drop_policy": policy,
        "max_drop_num": 0 if policy == "block" else 1,
        "max_drop_den": 1 if policy == "block" else 3,
        "max_consecutive_drops": 0 if policy == "block" else 2,
        "max_duration_seconds": 600,
        "max_frames": 18_000,
        "job_timeout_seconds": 600,
        "max_results_per_frame": 100,
        "max_total_results": 1_000_000,
        "max_mask_artifacts": 0 if task == "object_detection" else 10_000,
        "max_output_files": 4 if task == "object_detection" else 10_004,
        "max_output_bytes": 4_294_967_296,
        "max_mask_bytes": 67_108_864,
        "mask_chunk_bytes": 1_048_576,
        "callback_max_items": 64,
        "callback_max_bytes": 67_108_864,
        "max_decoder_rss_bytes": 134_217_728,
        "max_stream_job_peak_rss_bytes": 1_073_741_824,
        "max_accelerator_process_tree_peak_bytes": None,
        "camera_pool_capacity_frames": None,
        "camera_pool_bytes": None,
        "stride_policy_id": "width_times_3_to_4_v1",
        "decoded_stride_min_bytes": 1_920,
        "decoded_stride_max_bytes": 2_560,
        "source_admission_policy_id": "rational_monotonic_due_v1",
        "source_admission_policy_version": 1,
        "output_cadence_id": "one_frame_result_per_processed_frame_v1",
        "latency_interval_id": "stream_due_to_callback_enqueue_v1",
        "decoded_layout_id": "uint8_rgb_bgr_strided_v1",
        "camera_pool_policy_id": "caller_preallocated_fixed_pool_v1",
        "camera_pool_policy_version": 1,
        "camera_eligibility_policy_id": "exactly_one_reenumerate_before_open_v1",
        "mask_encoding_id": "png_binary_mask_v1",
        "mask_encoding_version": 1,
        "decoder_policy": _decoder_policy(),
        "memory_collector": {
            "collector_id": "fixture_tree_memory",
            "collector_version": "1",
            "source": "fixture_process_tree",
            "scope": "whole_stream_job_process_tree_v1",
        },
        "quality_requirement": None,
        "network_policy": "deny",
        "source_probe_timeout_seconds": 10,
        "camera_open_timeout_seconds": 10,
        "camera_first_frame_timeout_seconds": 10,
        "runner_probe_timeout_seconds": 30,
        "runner_load_timeout_seconds": 600,
        "frame_decode_timeout_seconds": 5,
        "frame_predict_timeout_seconds": 30,
        "output_step_timeout_seconds": 30,
        "close_timeout_seconds": 10,
        "cancellation_grace_seconds": 5,
    }


def _frame(*, task: str = "object_detection", index: int = 0) -> dict:
    due_num = 0 if index == 0 else index * 100_000_000
    due_den = 1 if index == 0 else 3
    mask = None
    if task == "instance_segmentation":
        mask = {
            "relative_path": f"artifacts/masks/{index:06d}.png",
            "sha256": "8" * 64,
            "size_bytes": 100,
            "width": 640,
            "height": 480,
            "encoding_id": "png_binary_mask_v1",
        }
    record = {
        "schema_version": 1,
        "source_frame_index": index,
        "scheduled_due_offset_num_ns": due_num,
        "scheduled_due_offset_den": due_den,
        "processing_completed_offset_ns": (due_num + due_den - 1) // due_den + 1_000_000,
        "device_timestamp_ns": None,
        "task": task,
        "decoded_width": 640,
        "decoded_height": 480,
        "task_results": [
            {
                "class_id": 1,
                "score": "0.9",
                "bbox": {"x1": "0.1", "y1": "0.2", "x2": "0.8", "y2": "0.9"},
                "mask": mask,
            }
        ],
        "frame_result_digest": "0" * 64,
    }
    record["frame_result_digest"] = canonical_sha256_v1(
        record, own_digest_field="frame_result_digest"
    )
    return record


def _summary(*, failed: int = 0) -> dict:
    record = {
        "schema_version": 1,
        "status": "failed" if failed else "completed",
        "task": "object_detection",
        "source_kind": "local_mp4",
        "scheduled_frame_count": 18_000 + failed,
        "processed_frame_count": 18_000,
        "dropped_frame_count": 0,
        "failed_unaccounted_frame_count": failed,
        "max_consecutive_drops": 0,
        "frame_queue_count_high_watermark": 4,
        "frame_queue_decoded_bytes_high_watermark": 1_228_800,
        "callback_item_high_watermark": 8,
        "callback_bytes_high_watermark": 16_384,
        "duration_ns": 600_000_000_000,
        "p50_latency_ms": "10",
        "p95_latency_ms": "20",
        "p99_latency_ms": "30",
        "drop_fraction_display": "0",
        "result_count": 0,
        "mask_artifact_count": 0,
        "output_file_count": 4,
        "output_bytes": 1_024,
        "termination_reason": "runner_failure" if failed else "normal_eof",
        "bundle_spec_digest": "a" * 64,
        "evidence_report_digest": None,
        "summary_digest": "0" * 64,
    }
    record["summary_digest"] = canonical_sha256_v1(
        record, own_digest_field="summary_digest"
    )
    return record


def _selection(job: dict, workload: dict) -> dict:
    validated_job = validate_stream_job_spec(job)
    preimage = {
        "source_kind": "local_mp4",
        "byte_length": 1_024,
        "source_sha256": "1" * 64,
    }
    record = {
        "schema_version": 1,
        "decision_kind": "local_stream",
        "decision_id": "decision-1",
        "decision_digest": "0" * 64,
        "status": "selected",
        "decided_at": "2026-08-29T00:00:00Z",
        "local_job_digest": validated_job.local_job_digest,
        "stream_source_preimage": preimage,
        "stream_source_digest": compute_stream_source_digest(
            preimage, expected_source=validated_job.to_dict()["source"]
        ),
        "artifact_resolver_state_digest": "2" * 64,
        "environment_fingerprint": "3" * 64,
        "protocol_fingerprint": "4" * 64,
        "stream_job_spec": validated_job.to_dict(),
        "stream_workload_profile": workload,
        "selected_bundle": {
            "bundle_id": "bundle-a",
            "bundle_version": "1",
            "bundle_spec_digest": "a" * 64,
        },
        "selected_evidence": {
            "report_id": "report-a",
            "report_digest": "b" * 64,
            "trust_domain": "site_managed",
        },
        "selected_artifact_state_fingerprint": "c" * 64,
        "support_scope": "site_qualified",
        "reason_codes": [],
        "candidate_evaluations": [
            {
                "bundle_id": "bundle-a",
                "bundle_version": "1",
                "rank_state": "selected",
                "reason_codes": [],
            }
        ],
    }
    record["decision_digest"] = canonical_sha256_v1(
        record, own_digest_field="decision_digest"
    )
    return record


def _checksum_manifest(outputs: dict[str, bytes]) -> bytes:
    paths = sorted(outputs, key=lambda path: path.encode("utf-8"))
    files = [
        {
            "path": path,
            "size_bytes": len(outputs[path]),
            "sha256": hashlib.sha256(outputs[path]).hexdigest(),
        }
        for path in paths
    ]
    return canonical_json_v1(
        {
            "schema_version": 1,
            "files": files,
            "expected_paths": paths,
            "file_count": len(files),
            "total_bytes": sum(item["size_bytes"] for item in files),
        }
    )


def _stream_output_bundle(
    *, task: str = "object_detection", empty: bool = False
) -> tuple[dict, dict, dict, dict[str, bytes], bytes]:
    job = _job(task=task)
    workload = build_stream_workload_profile(
        job, collector_id="stream_preflight", collector_version="1"
    ).to_dict()
    decision = _selection(job, workload)
    outputs: dict[str, bytes] = {}
    frame = None if empty else _frame(task=task)
    if task == "instance_segmentation" and frame is not None:
        mask_bytes = b"M" * 100
        mask = frame["task_results"][0]["mask"]
        mask["size_bytes"] = len(mask_bytes)
        mask["sha256"] = hashlib.sha256(mask_bytes).hexdigest()
        frame["frame_result_digest"] = canonical_sha256_v1(
            frame, own_digest_field="frame_result_digest"
        )
        outputs[mask["relative_path"]] = mask_bytes
    outputs["stream_results.jsonl"] = (
        b"" if frame is None else canonical_json_v1(frame) + b"\n"
    )

    summary = {
        "schema_version": 1,
        "status": "completed",
        "task": task,
        "source_kind": "local_mp4",
        "scheduled_frame_count": 0 if empty else 1,
        "processed_frame_count": 0 if empty else 1,
        "dropped_frame_count": 0,
        "failed_unaccounted_frame_count": 0,
        "max_consecutive_drops": 0,
        "frame_queue_count_high_watermark": 0 if empty else 1,
        "frame_queue_decoded_bytes_high_watermark": 0 if empty else 1_228_800,
        "callback_item_high_watermark": 0 if empty else 1,
        "callback_bytes_high_watermark": len(outputs["stream_results.jsonl"]),
        "duration_ns": 0 if empty else 1_000_000,
        "p50_latency_ms": None if empty else "1",
        "p95_latency_ms": None if empty else "1",
        "p99_latency_ms": None if empty else "1",
        "drop_fraction_display": None if empty else "0",
        "result_count": 0 if empty else 1,
        "mask_artifact_count": (
            1 if task == "instance_segmentation" and not empty else 0
        ),
        "output_file_count": (
            5 if task == "instance_segmentation" and not empty else 4
        ),
        "output_bytes": 0,
        "termination_reason": "normal_eof",
        "bundle_spec_digest": "a" * 64,
        "evidence_report_digest": "b" * 64,
        "summary_digest": "0" * 64,
    }
    for _attempt in range(8):
        summary["summary_digest"] = canonical_sha256_v1(
            summary, own_digest_field="summary_digest"
        )
        outputs["stream_summary.json"] = canonical_json_v1(summary)
        outputs["provenance.json"] = canonical_json_v1(
            {
                "schema_version": 1,
                "output_kind": "local_stream",
                "local_job_digest": validate_stream_job_spec(job).local_job_digest,
                "stream_workload_fingerprint": workload["workload_fingerprint"],
                "stream_selection_decision_digest": decision["decision_digest"],
                "stream_source_digest": decision["stream_source_digest"],
                "bundle_spec_digest": "a" * 64,
                "evidence_report_digest": "b" * 64,
                "artifact_state_fingerprint": "c" * 64,
                "summary_digest": summary["summary_digest"],
                "frame_result_count": 0 if empty else 1,
            }
        )
        declared_bytes = sum(len(data) for data in outputs.values())
        if summary["output_bytes"] == declared_bytes:
            break
        summary["output_bytes"] = declared_bytes
    else:  # pragma: no cover - fixed-width digests make this converge quickly
        raise AssertionError("output-byte fixture did not converge")
    manifest = _checksum_manifest(outputs)
    return job, workload, decision, outputs, manifest


def _declared_outputs(outputs: dict[str, bytes]) -> list[tuple[str, bytes]]:
    return [
        (path, outputs[path])
        for path in sorted(outputs, key=lambda path: path.encode("utf-8"))
    ]


class TestStreamingContract(unittest.TestCase):
    def test_job_and_workload_bind_every_frozen_policy(self) -> None:
        job = validate_stream_job_spec(_job())
        self.assertEqual(job.to_dict()["source"]["source_rate_num"], 30)
        self.assertEqual(len(job.local_job_digest), 64)
        workload = build_stream_workload_profile(
            job, collector_id="stream_preflight", collector_version="1"
        )
        self.assertEqual(
            validate_stream_workload_profile(workload.to_dict()).to_dict(),
            workload.to_dict(),
        )
        changed = _job()
        changed["queue_capacity_frames"] = 5
        self.assertNotEqual(
            validate_stream_job_spec(changed).local_job_digest, job.local_job_digest
        )

    def test_job_rejects_unbounded_network_rate_and_capacity_variants(self) -> None:
        cases = []
        network = _job()
        network["network_policy"] = "allow"
        cases.append(network)
        url = _job()
        url["source"]["url"] = "rtsp://example.invalid/live"
        cases.append(url)
        unreduced = _job()
        unreduced["source"]["source_rate_num"] = 60
        unreduced["source"]["source_rate_den"] = 2
        cases.append(unreduced)
        sampled = _job()
        sampled["max_frames"] = 17_999
        cases.append(sampled)
        too_small = _job()
        too_small["max_queued_decoded_bytes"] = 1_228_799
        cases.append(too_small)
        hidden_drop = _job()
        hidden_drop["max_drop_num"] = 1
        hidden_drop["max_drop_den"] = 3
        cases.append(hidden_drop)
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(StreamContractError):
                validate_stream_job_spec(payload)

    def test_camera_is_capability_bound_without_device_identity(self) -> None:
        payload = _job()
        payload["source"] = {
            "source_kind": "local_camera",
            "width": 640,
            "height": 480,
            "source_rate_num": 30,
            "source_rate_den": 1,
            "container": None,
            "codec": None,
            "pixel_format": "rgb24",
            "provider_id": "contract_fixture_camera_v1",
            "capability_format": "rgb24",
        }
        payload["camera_pool_capacity_frames"] = 4
        payload["camera_pool_bytes"] = 4 * 640 * 4 * 480
        checked = validate_stream_job_spec(payload).to_dict()
        self.assertNotIn("device_id", json.dumps(checked, sort_keys=True))
        payload["source"]["device_id"] = "camera-serial"
        with self.assertRaises(StreamContractError):
            validate_stream_job_spec(payload)

    def test_frame_result_own_digest_canonical_line_and_context(self) -> None:
        frame = validate_frame_result(
            _frame(index=1),
            source_rate_num=30,
            source_rate_den=1,
            expected_task="object_detection",
            expected_width=640,
            expected_height=480,
        )
        self.assertEqual(frame.canonical_line(), frame.canonical_bytes() + b"\n")
        self.assertEqual(frame.to_dict()["task_results"][0]["class_id"], 1)
        tampered = _frame(index=1)
        tampered["task_results"][0]["score"] = "0.8"
        with self.assertRaises(StreamContractError):
            validate_frame_result(tampered)

    def test_frame_result_empty_success_and_mask_boundaries(self) -> None:
        empty = _frame()
        empty["task_results"] = []
        empty["frame_result_digest"] = canonical_sha256_v1(
            empty, own_digest_field="frame_result_digest"
        )
        self.assertEqual(validate_frame_result(empty).to_dict()["task_results"], [])
        segmentation = validate_frame_result(_frame(task="instance_segmentation"))
        self.assertEqual(
            segmentation.to_dict()["task_results"][0]["mask"]["encoding_id"],
            "png_binary_mask_v1",
        )
        oversized = _frame(task="instance_segmentation")
        oversized["task_results"][0]["mask"]["size_bytes"] = 67_108_865
        oversized["frame_result_digest"] = canonical_sha256_v1(
            oversized, own_digest_field="frame_result_digest"
        )
        with self.assertRaises(StreamContractError):
            validate_frame_result(oversized)

    def test_exact_drop_and_sustained_arithmetic(self) -> None:
        self.assertEqual(compute_max_consecutive_drops([1, 2, 5, 6, 7]), 3)
        self.assertTrue(
            drop_fraction_within_limit(
                dropped_count=1,
                processed_count=2,
                maximum_numerator=1,
                maximum_denominator=3,
            )
        )
        self.assertFalse(
            drop_fraction_within_limit(
                dropped_count=2,
                processed_count=3,
                maximum_numerator=1,
                maximum_denominator=3,
            )
        )
        self.assertTrue(
            sustained_fps_meets_minimum(
                processed_count=600, duration_ns=600_000_000_000, minimum_fps="1"
            )
        )
        with self.assertRaises(StreamContractError):
            drop_fraction_within_limit(
                dropped_count=0,
                processed_count=0,
                maximum_numerator=0,
                maximum_denominator=1,
            )

    def test_summary_requires_complete_accounting_and_exact_digest(self) -> None:
        summary = validate_stream_summary(_summary())
        self.assertEqual(summary.to_dict()["drop_fraction_display"], "0")
        bad = _summary()
        bad["scheduled_frame_count"] += 1
        bad["summary_digest"] = canonical_sha256_v1(
            bad, own_digest_field="summary_digest"
        )
        with self.assertRaises(StreamContractError):
            validate_stream_summary(bad)

    def test_qualified_report_uses_stream_native_gates(self) -> None:
        workload = build_stream_workload_profile(
            _job(), collector_id="stream_preflight", collector_version="1"
        ).to_dict()
        report = {
            "schema_version": 1,
            "report_kind": "stream_qualification",
            "report_id": "fixture-report",
            "report_digest": "0" * 64,
            "status": "qualified",
            "started_at": "2026-08-29T00:00:00Z",
            "completed_at": "2026-08-29T00:10:00Z",
            "valid_until": "2026-08-30T00:10:00Z",
            "bundle_spec_digest": "a" * 64,
            "artifact_set_digest": "b" * 64,
            "artifact_state_fingerprint": "c" * 64,
            "environment_fingerprint": "d" * 64,
            "protocol_fingerprint": "e" * 64,
            "stream_workload_profile": workload,
            "summary": _summary(),
            "sustained_section": {
                "start_source_frame_index": 0,
                "end_source_frame_index_exclusive": 18_000,
                "scheduled_frame_count": 18_000,
                "duration_ns": 600_000_000_000,
                "processed_frame_count": 18_000,
                "dropped_frame_count": 0,
                "failed_unaccounted_frame_count": 0,
                "p50_latency_ms": "10",
                "p95_latency_ms": "20",
                "p99_latency_ms": "30",
                "sustained_fps_display": "30",
            },
            "memory": {
                "collector": _job()["memory_collector"],
                "coverage_complete": True,
                "stream_job_peak_rss_bytes": 268_435_456,
                "accelerator_process_tree_peak_bytes": None,
                "thermal_status": "nominal",
                "power_status": "known",
            },
            "quality": None,
            "limitations": ["fixture contract validation only"],
            "failures": [],
        }
        report["report_digest"] = canonical_sha256_v1(
            report, own_digest_field="report_digest"
        )
        checked = validate_stream_qualification_report(report)
        self.assertEqual(checked.to_dict()["status"], "qualified")
        self.assertNotIn("stream_source_digest", json.dumps(checked.to_dict()))
        self.assertNotIn("stream_source_preimage", json.dumps(checked.to_dict()))
        unknown_memory = copy.deepcopy(report)
        unknown_memory["memory"]["coverage_complete"] = False
        unknown_memory["memory"]["stream_job_peak_rss_bytes"] = None
        unknown_memory["report_digest"] = canonical_sha256_v1(
            unknown_memory, own_digest_field="report_digest"
        )
        with self.assertRaises(StreamContractError):
            validate_stream_qualification_report(unknown_memory)

        queue_overflow = copy.deepcopy(report)
        queue_overflow["summary"]["frame_queue_count_high_watermark"] = 5
        queue_overflow["summary"]["summary_digest"] = canonical_sha256_v1(
            queue_overflow["summary"], own_digest_field="summary_digest"
        )
        queue_overflow["report_digest"] = canonical_sha256_v1(
            queue_overflow, own_digest_field="report_digest"
        )
        with self.assertRaisesRegex(StreamContractError, "queue-count HWM"):
            validate_stream_qualification_report(queue_overflow)

        cancelled = copy.deepcopy(report)
        cancelled["summary"]["status"] = "failed"
        cancelled["summary"]["termination_reason"] = "cancelled"
        cancelled["summary"]["summary_digest"] = canonical_sha256_v1(
            cancelled["summary"], own_digest_field="summary_digest"
        )
        cancelled["report_digest"] = canonical_sha256_v1(
            cancelled, own_digest_field="report_digest"
        )
        with self.assertRaisesRegex(StreamContractError, "cannot contain failures"):
            validate_stream_qualification_report(cancelled)

        early = copy.deepcopy(report)
        early["summary"]["scheduled_frame_count"] = 17_970
        early["summary"]["processed_frame_count"] = 17_970
        early["summary"]["duration_ns"] = 599_000_000_000
        early["summary"]["summary_digest"] = canonical_sha256_v1(
            early["summary"], own_digest_field="summary_digest"
        )
        early["sustained_section"].update(
            {
                "end_source_frame_index_exclusive": 17_970,
                "scheduled_frame_count": 17_970,
                "duration_ns": 599_000_000_000,
                "processed_frame_count": 17_970,
                "sustained_fps_display": "30",
            }
        )
        early["report_digest"] = canonical_sha256_v1(
            early, own_digest_field="report_digest"
        )
        with self.assertRaisesRegex(StreamContractError, "short sustained"):
            validate_stream_qualification_report(early)

        wrong_window = copy.deepcopy(report)
        wrong_window["sustained_section"]["start_source_frame_index"] = 1
        wrong_window["sustained_section"]["end_source_frame_index_exclusive"] = 18_001
        wrong_window["report_digest"] = canonical_sha256_v1(
            wrong_window, own_digest_field="report_digest"
        )
        with self.assertRaisesRegex(StreamContractError, "source window mismatch"):
            validate_stream_qualification_report(wrong_window)

        wrong_cadence_count = copy.deepcopy(report)
        wrong_cadence_count["sustained_section"].update(
            {
                "end_source_frame_index_exclusive": 18_001,
                "scheduled_frame_count": 18_001,
                "dropped_frame_count": 1,
            }
        )
        wrong_cadence_count["report_digest"] = canonical_sha256_v1(
            wrong_cadence_count, own_digest_field="report_digest"
        )
        with self.assertRaisesRegex(StreamContractError, "count/cadence mismatch"):
            validate_stream_qualification_report(wrong_cadence_count)

        quality_job = _job()
        quality_job["quality_requirement"] = {
            "metric_id": "fixture_metric",
            "direction": "higher_is_better",
            "threshold": "-0.5",
            "evaluation_dataset_id": "fixture_dataset",
            "evaluation_dataset_sha256": "5" * 64,
            "evaluation_protocol_sha256": "6" * 64,
            "evaluation_vocabulary_id": "fixture_vocabulary",
            "task": "object_detection",
        }
        quality_report = copy.deepcopy(report)
        quality_report["stream_workload_profile"] = build_stream_workload_profile(
            quality_job, collector_id="stream_preflight", collector_version="1"
        ).to_dict()
        quality_report["quality"] = {
            **quality_job["quality_requirement"],
            "status": "passed",
            "measured_value": "-0.4",
        }
        quality_report["report_digest"] = canonical_sha256_v1(
            quality_report, own_digest_field="report_digest"
        )
        validate_stream_qualification_report(quality_report)
        divergent_quality = copy.deepcopy(quality_report)
        divergent_quality["quality"]["status"] = "failed"
        divergent_quality["report_digest"] = canonical_sha256_v1(
            divergent_quality, own_digest_field="report_digest"
        )
        with self.assertRaisesRegex(StreamContractError, "diverges"):
            validate_stream_qualification_report(divergent_quality)

    def test_qualified_sustained_section_excludes_warmup(self) -> None:
        job = _job()
        job["warmup_frame_count"] = 30
        job["max_duration_seconds"] = 601
        job["max_frames"] = 18_030
        job["job_timeout_seconds"] = 601
        workload = build_stream_workload_profile(
            job, collector_id="stream_preflight", collector_version="1"
        ).to_dict()
        summary = _summary()
        summary["scheduled_frame_count"] = 18_030
        summary["processed_frame_count"] = 18_030
        summary["duration_ns"] = 601_000_000_000
        summary["summary_digest"] = canonical_sha256_v1(
            summary, own_digest_field="summary_digest"
        )
        report = {
            "schema_version": 1,
            "report_kind": "stream_qualification",
            "report_id": "warmup-report",
            "report_digest": "0" * 64,
            "status": "qualified",
            "started_at": "2026-08-29T00:00:00Z",
            "completed_at": "2026-08-29T00:10:01Z",
            "valid_until": "2026-08-30T00:10:01Z",
            "bundle_spec_digest": "a" * 64,
            "artifact_set_digest": "b" * 64,
            "artifact_state_fingerprint": "c" * 64,
            "environment_fingerprint": "d" * 64,
            "protocol_fingerprint": "e" * 64,
            "stream_workload_profile": workload,
            "summary": summary,
            "sustained_section": {
                "start_source_frame_index": 30,
                "end_source_frame_index_exclusive": 18_030,
                "scheduled_frame_count": 18_000,
                "duration_ns": 600_000_000_000,
                "processed_frame_count": 18_000,
                "dropped_frame_count": 0,
                "failed_unaccounted_frame_count": 0,
                "p50_latency_ms": "10",
                "p95_latency_ms": "20",
                "p99_latency_ms": "30",
                "sustained_fps_display": "30",
            },
            "memory": {
                "collector": job["memory_collector"],
                "coverage_complete": True,
                "stream_job_peak_rss_bytes": 268_435_456,
                "accelerator_process_tree_peak_bytes": None,
                "thermal_status": "nominal",
                "power_status": "known",
            },
            "quality": None,
            "limitations": ["fixture contract validation only"],
            "failures": [],
        }
        report["report_digest"] = canonical_sha256_v1(
            report, own_digest_field="report_digest"
        )
        checked = validate_stream_qualification_report(report).to_dict()
        self.assertEqual(checked["sustained_section"]["start_source_frame_index"], 30)
        self.assertNotEqual(
            checked["sustained_section"]["processed_frame_count"],
            checked["summary"]["processed_frame_count"],
        )

    def test_stream_selection_is_distinct_and_source_digest_is_local_only(self) -> None:
        job = validate_stream_job_spec(_job())
        workload = build_stream_workload_profile(
            job, collector_id="stream_preflight", collector_version="1"
        ).to_dict()
        decision = {
            "schema_version": 1,
            "decision_kind": "local_stream",
            "decision_id": "decision-1",
            "decision_digest": "0" * 64,
            "status": "selected",
            "decided_at": "2026-08-29T00:00:00Z",
            "local_job_digest": job.local_job_digest,
            "stream_source_preimage": {
                "source_kind": "local_mp4",
                "byte_length": 1_024,
                "source_sha256": "1" * 64,
            },
            "stream_source_digest": compute_stream_source_digest(
                {
                    "source_kind": "local_mp4",
                    "byte_length": 1_024,
                    "source_sha256": "1" * 64,
                },
                expected_source=job.to_dict()["source"],
            ),
            "artifact_resolver_state_digest": "2" * 64,
            "environment_fingerprint": "3" * 64,
            "protocol_fingerprint": "4" * 64,
            "stream_job_spec": job.to_dict(),
            "stream_workload_profile": workload,
            "selected_bundle": {
                "bundle_id": "bundle-a",
                "bundle_version": "1",
                "bundle_spec_digest": "a" * 64,
            },
            "selected_evidence": {
                "report_id": "report-a",
                "report_digest": "b" * 64,
                "trust_domain": "site_managed",
            },
            "selected_artifact_state_fingerprint": "c" * 64,
            "support_scope": "site_qualified",
            "reason_codes": [],
            "candidate_evaluations": [
                {
                    "bundle_id": "bundle-a",
                    "bundle_version": "1",
                    "rank_state": "selected",
                    "reason_codes": [],
                }
            ],
        }
        decision["decision_digest"] = canonical_sha256_v1(
            decision, own_digest_field="decision_digest"
        )
        self.assertEqual(
            validate_stream_selection_decision(decision).to_dict()["decision_kind"],
            "local_stream",
        )
        mismatched_candidate = copy.deepcopy(decision)
        mismatched_candidate["candidate_evaluations"][0]["bundle_id"] = "bundle-z"
        mismatched_candidate["decision_digest"] = canonical_sha256_v1(
            mismatched_candidate, own_digest_field="decision_digest"
        )
        with self.assertRaisesRegex(StreamContractError, "contradictory selected"):
            validate_stream_selection_decision(mismatched_candidate)

        rejected_selected = copy.deepcopy(decision)
        rejected_selected["candidate_evaluations"][0]["reason_codes"] = [
            "evidence_not_qualified"
        ]
        rejected_selected["decision_digest"] = canonical_sha256_v1(
            rejected_selected, own_digest_field="decision_digest"
        )
        with self.assertRaisesRegex(StreamContractError, "rejection reasons"):
            validate_stream_selection_decision(rejected_selected)

        public_site_evidence = copy.deepcopy(decision)
        public_site_evidence["support_scope"] = "public_qualified"
        public_site_evidence["decision_digest"] = canonical_sha256_v1(
            public_site_evidence, own_digest_field="decision_digest"
        )
        with self.assertRaisesRegex(StreamContractError, "public scope"):
            validate_stream_selection_decision(public_site_evidence)

        selected_without_scope = copy.deepcopy(decision)
        selected_without_scope["support_scope"] = "none"
        selected_without_scope["decision_digest"] = canonical_sha256_v1(
            selected_without_scope, own_digest_field="decision_digest"
        )
        with self.assertRaisesRegex(StreamContractError, "qualified support scope"):
            validate_stream_selection_decision(selected_without_scope)

        static_kind = copy.deepcopy(decision)
        static_kind["decision_kind"] = "image"
        static_kind["decision_digest"] = canonical_sha256_v1(
            static_kind, own_digest_field="decision_digest"
        )
        with self.assertRaises(StreamContractError):
            validate_stream_selection_decision(static_kind)

    def test_stream_output_artifacts_bind_every_durable_byte(self) -> None:
        job, workload, decision, outputs, manifest = _stream_output_bundle()
        summary = validate_stream_output_artifacts(
            _declared_outputs(outputs),
            manifest,
            job=job,
            workload=workload,
            selection_decision=decision,
            dropped_source_frame_indices=[],
        )
        self.assertEqual(summary.to_dict()["processed_frame_count"], 1)

        split_chunks = _declared_outputs(outputs)
        stream_index = next(
            index
            for index, (path, _chunks) in enumerate(split_chunks)
            if path == "stream_results.jsonl"
        )
        stream_bytes = outputs["stream_results.jsonl"]
        split_chunks[stream_index] = (
            "stream_results.jsonl",
            (stream_bytes[:7], stream_bytes[7:91], stream_bytes[91:]),
        )
        validate_stream_output_artifacts(
            split_chunks,
            manifest,
            job=job,
            workload=workload,
            selection_decision=decision,
            dropped_source_frame_indices=[],
        )

        overrun_chunks = copy.deepcopy(split_chunks)
        overrun_chunks[stream_index] = (
            "stream_results.jsonl",
            (stream_bytes, b"x"),
        )
        with self.assertRaisesRegex(StreamContractError, "size exceeds"):
            validate_stream_output_artifacts(
                overrun_chunks,
                manifest,
                job=job,
                workload=workload,
                selection_decision=decision,
                dropped_source_frame_indices=[],
            )

        tampered = copy.deepcopy(outputs)
        tampered["provenance.json"] = b"X" * len(tampered["provenance.json"])
        with self.assertRaisesRegex(StreamContractError, "SHA-256"):
            validate_stream_output_artifacts(
                _declared_outputs(tampered),
                manifest,
                job=job,
                workload=workload,
                selection_decision=decision,
                dropped_source_frame_indices=[],
            )

        nonprogress = _declared_outputs(outputs)
        provenance_index = next(
            index
            for index, (path, _chunks) in enumerate(nonprogress)
            if path == "provenance.json"
        )
        nonprogress[provenance_index] = (
            "provenance.json",
            (b"", outputs["provenance.json"]),
        )
        with self.assertRaisesRegex(StreamContractError, "cannot progress"):
            validate_stream_output_artifacts(
                nonprogress,
                manifest,
                job=job,
                workload=workload,
                selection_decision=decision,
                dropped_source_frame_indices=[],
            )

        with self.assertRaisesRegex(StreamContractError, "exactly match"):
            validate_stream_output_artifacts(
                _declared_outputs(outputs)[:-1],
                manifest,
                job=job,
                workload=workload,
                selection_decision=decision,
                dropped_source_frame_indices=[],
            )

        with self.assertRaisesRegex(StreamContractError, "drop aggregate"):
            validate_stream_output_artifacts(
                _declared_outputs(outputs),
                manifest,
                job=job,
                workload=workload,
                selection_decision=decision,
                dropped_source_frame_indices=[0],
            )

        pretty = json.dumps(json.loads(manifest), indent=2).encode("utf-8")
        with self.assertRaisesRegex(StreamContractError, "canonical_json_v1"):
            validate_stream_output_artifacts(
                _declared_outputs(outputs),
                pretty,
                job=job,
                workload=workload,
                selection_decision=decision,
                dropped_source_frame_indices=[],
            )

        self_entry = json.loads(manifest)
        self_entry["files"].append(
            {
                "path": "checksums.json",
                "size_bytes": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            }
        )
        self_entry["files"].sort(key=lambda item: item["path"].encode("utf-8"))
        self_entry["expected_paths"] = [item["path"] for item in self_entry["files"]]
        self_entry["file_count"] += 1
        with self.assertRaisesRegex(StreamContractError, "must not list itself"):
            validate_stream_output_artifacts(
                _declared_outputs(outputs),
                canonical_json_v1(self_entry),
                job=job,
                workload=workload,
                selection_decision=decision,
                dropped_source_frame_indices=[],
            )

    def test_stream_output_accepts_finite_empty_jsonl_not_empty_chunk_iterators(self) -> None:
        job, workload, decision, outputs, manifest = _stream_output_bundle(empty=True)
        summary = validate_stream_output_artifacts(
            _declared_outputs(outputs),
            manifest,
            job=job,
            workload=workload,
            selection_decision=decision,
            dropped_source_frame_indices=[],
        )
        self.assertEqual(summary.to_dict()["processed_frame_count"], 0)

        def nonprogress() -> Iterator[bytes]:
            while True:
                yield b""

        declared = _declared_outputs(outputs)
        stream_index = next(
            index
            for index, (path, _chunks) in enumerate(declared)
            if path == "stream_results.jsonl"
        )
        declared[stream_index] = ("stream_results.jsonl", nonprogress())
        with self.assertRaisesRegex(StreamContractError, "empty chunk cannot progress"):
            validate_stream_output_artifacts(
                declared,
                manifest,
                job=job,
                workload=workload,
                selection_decision=decision,
                dropped_source_frame_indices=[],
            )

    def test_stream_output_artifacts_bind_mask_bytes_and_frame_reference(self) -> None:
        job, workload, decision, outputs, manifest = _stream_output_bundle(
            task="instance_segmentation"
        )
        validate_stream_output_artifacts(
            _declared_outputs(outputs),
            manifest,
            job=job,
            workload=workload,
            selection_decision=decision,
            dropped_source_frame_indices=[],
        )
        mask_path = next(path for path in outputs if path.startswith("artifacts/masks/"))
        tampered_mask = copy.deepcopy(outputs)
        tampered_mask[mask_path] = b"X" * len(tampered_mask[mask_path])
        with self.assertRaisesRegex(StreamContractError, "SHA-256"):
            validate_stream_output_artifacts(
                _declared_outputs(tampered_mask),
                manifest,
                job=job,
                workload=workload,
                selection_decision=decision,
                dropped_source_frame_indices=[],
            )

        mismatched_reference = copy.deepcopy(outputs)
        frame = json.loads(mismatched_reference["stream_results.jsonl"])
        frame["task_results"][0]["mask"]["sha256"] = "f" * 64
        frame["frame_result_digest"] = canonical_sha256_v1(
            frame, own_digest_field="frame_result_digest"
        )
        mismatched_reference["stream_results.jsonl"] = canonical_json_v1(frame) + b"\n"
        mismatched_manifest = _checksum_manifest(mismatched_reference)
        with self.assertRaisesRegex(StreamContractError, "mask reference"):
            validate_stream_output_artifacts(
                _declared_outputs(mismatched_reference),
                mismatched_manifest,
                job=job,
                workload=workload,
                selection_decision=decision,
                dropped_source_frame_indices=[],
            )

    def test_schema_pairs_are_valid_and_byte_identical(self) -> None:
        root = Path(__file__).resolve().parents[1]
        basenames = (
            "stream_job_spec.schema.json",
            "stream_workload_profile.schema.json",
            "frame_result.schema.json",
            "stream_summary.schema.json",
            "stream_qualification_report.schema.json",
            "stream_selection_decision.schema.json",
        )
        for basename in basenames:
            canonical = root / "docs" / "schemas" / basename
            packaged = root / "yolozu" / "data" / "schemas" / basename
            with self.subTest(schema=basename):
                self.assertEqual(canonical.read_bytes(), packaged.read_bytes())
                self.assertEqual(json.loads(canonical.read_text())["$schema"], "https://json-schema.org/draft/2020-12/schema")

    def test_schema_transport_bounds_and_python_cross_field_semantics(self) -> None:
        root = Path(__file__).resolve().parents[1] / "docs" / "schemas"
        job_schema = json.loads((root / "stream_job_spec.schema.json").read_text())
        frame_schema = json.loads((root / "frame_result.schema.json").read_text())
        report_schema = json.loads(
            (root / "stream_qualification_report.schema.json").read_text()
        )
        decision_schema = json.loads(
            (root / "stream_selection_decision.schema.json").read_text()
        )

        job = _job()
        self.assertTrue(_schema_accepts(job, job_schema, root=job_schema))
        self.assertEqual(validate_stream_job_spec(job).to_dict(), job)

        quality_job = _job()
        quality_job["quality_requirement"] = {
            "metric_id": "fixture_metric",
            "direction": "higher_is_better",
            "threshold": "-0.5",
            "evaluation_dataset_id": "fixture_dataset",
            "evaluation_dataset_sha256": "5" * 64,
            "evaluation_protocol_sha256": "6" * 64,
            "evaluation_vocabulary_id": "fixture_vocabulary",
            "task": "object_detection",
        }
        self.assertTrue(
            _schema_accepts(quality_job, job_schema, root=job_schema)
        )
        validate_stream_job_spec(quality_job)

        negative_slo = _job()
        negative_slo["min_sustained_fps"] = "-1"
        self.assertFalse(
            _schema_accepts(negative_slo, job_schema, root=job_schema)
        )
        with self.assertRaises(StreamContractError):
            validate_stream_job_spec(negative_slo)

        below_minimum_slo = _job()
        below_minimum_slo["min_sustained_fps"] = "0.01"
        self.assertFalse(
            _schema_accepts(below_minimum_slo, job_schema, root=job_schema)
        )
        with self.assertRaises(StreamContractError):
            validate_stream_job_spec(below_minimum_slo)

        unreduced_rate = _job()
        unreduced_rate["source"]["source_rate_num"] = 60
        unreduced_rate["source"]["source_rate_den"] = 2
        self.assertTrue(
            _schema_accepts(unreduced_rate, job_schema, root=job_schema)
        )
        self.assertIn("normative semantic validator", job_schema["$defs"]["source"]["description"])
        with self.assertRaisesRegex(StreamContractError, "reduced"):
            validate_stream_job_spec(unreduced_rate)

        excessive_pixels = _job()
        excessive_pixels["source"]["width"] = 8_192
        excessive_pixels["source"]["height"] = 8_192
        excessive_pixels["decoded_stride_min_bytes"] = 24_576
        excessive_pixels["decoded_stride_max_bytes"] = 32_768
        self.assertTrue(
            _schema_accepts(excessive_pixels, job_schema, root=job_schema)
        )
        with self.assertRaisesRegex(StreamContractError, "pixel cap"):
            validate_stream_job_spec(excessive_pixels)

        mixed_source = _job()
        mixed_source["source"]["provider_id"] = "contract_fixture_camera_v1"
        self.assertFalse(
            _schema_accepts(mixed_source, job_schema, root=job_schema)
        )
        with self.assertRaises(StreamContractError):
            validate_stream_job_spec(mixed_source)

        frame = _frame()
        self.assertTrue(_schema_accepts(frame, frame_schema, root=frame_schema))
        invalid_score = _frame()
        invalid_score["task_results"][0]["score"] = "-0.1"
        invalid_score["frame_result_digest"] = canonical_sha256_v1(
            invalid_score, own_digest_field="frame_result_digest"
        )
        self.assertFalse(
            _schema_accepts(invalid_score, frame_schema, root=frame_schema)
        )
        with self.assertRaises(StreamContractError):
            validate_frame_result(invalid_score)

        oversized_frame = _frame()
        oversized_frame["decoded_width"] = 8_192
        oversized_frame["decoded_height"] = 8_192
        oversized_frame["frame_result_digest"] = canonical_sha256_v1(
            oversized_frame, own_digest_field="frame_result_digest"
        )
        self.assertTrue(
            _schema_accepts(oversized_frame, frame_schema, root=frame_schema)
        )
        self.assertIn("normative semantic validator", frame_schema["description"])
        with self.assertRaisesRegex(StreamContractError, "pixel cap"):
            validate_frame_result(oversized_frame)

        preimage_schema = decision_schema["$defs"]["stream_source_preimage"]
        mp4_preimage = {
            "source_kind": "local_mp4",
            "byte_length": 1,
            "source_sha256": "1" * 64,
        }
        camera_preimage = {
            "source_kind": "local_camera",
            "provider_id": "contract_fixture_camera_v1",
            "capability_format": "rgb24",
            "width": 640,
            "height": 480,
            "source_rate_num": 30,
            "source_rate_den": 1,
            "eligible_device_count": 1,
        }
        self.assertTrue(
            _schema_accepts(mp4_preimage, preimage_schema, root=decision_schema)
        )
        self.assertTrue(
            _schema_accepts(camera_preimage, preimage_schema, root=decision_schema)
        )
        wrong_count = {**camera_preimage, "eligible_device_count": 2}
        self.assertFalse(
            _schema_accepts(wrong_count, preimage_schema, root=decision_schema)
        )

        quality_schema = report_schema["$defs"]["quality"]
        quality = {
            **quality_job["quality_requirement"],
            "status": "passed",
            "measured_value": "-0.4",
        }
        self.assertTrue(
            _schema_accepts(quality, quality_schema, root=report_schema)
        )


if __name__ == "__main__":
    unittest.main()
