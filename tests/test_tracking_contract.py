from __future__ import annotations

import copy
import hashlib
import json
import unittest
from fractions import Fraction
from pathlib import Path
from typing import Any

from tests.test_adaptive_image_contracts import _schema_accepts
from tests.test_streaming_contract import (
    _job as _stream_job_record,
    _summary as _stream_summary_record,
)
from yolozu.adaptive.canonical import canonical_json_v1, canonical_sha256_v1
from yolozu.adaptive.streaming import (
    FrameResult,
    StreamJobSpec,
    validate_frame_result,
    validate_stream_job_spec,
    validate_stream_summary,
)
from yolozu.contracts.tracking import (
    JS_SAFE_TRACK_ID_MAX,
    MAX_ACTIVE_TRACKS,
    MAX_JOB_ROW_STATE_BUDGET,
    MAX_SESSION_UNIQUE_TRACKS,
    TrackingContractError,
    TrackingStateMachine,
    validate_tracking_output_artifacts,
    validate_tracking_output_interface,
    validate_tracking_output_provenance,
    validate_tracking_output_record,
    validate_tracking_output_streams,
)
from yolozu.predictions import validate_predictions_payload


def _interface(
    *,
    task: str = "object_detection",
    detector_interval: int = 1,
    max_results_per_frame: int = 1_000,
    max_prediction_frames: int = 240,
    max_lost_frames: int = 240,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "kind": "tracking_output_interface",
        "task": task,
        "detector_interval": detector_interval,
        "max_results_per_frame": max_results_per_frame,
        "max_prediction_frames": max_prediction_frames,
        "max_lost_frames": max_lost_frames,
        "max_active_tracks": 1_000,
        "max_session_unique_tracks": 1_000_000,
        "max_job_row_state_budget": 1_000_000,
        "identity_scope": "session_only",
        "biometric_inference": False,
        "cross_session_identity_linking": False,
        "persistent_identity_database": False,
        "tracking_output_interface_digest": "0" * 64,
    }
    record["tracking_output_interface_digest"] = canonical_sha256_v1(
        record, own_digest_field="tracking_output_interface_digest"
    )
    return record


def _bbox(*, offset: str = "0") -> dict[str, str]:
    if offset == "0":
        return {"x1": "0.1", "y1": "0.2", "x2": "0.7", "y2": "0.8"}
    return {"x1": "0.2", "y1": "0.3", "x2": "0.8", "y2": "0.9"}


def _mask(*, index: int = 0) -> dict[str, Any]:
    return {
        "relative_path": f"artifacts/masks/{index:06d}.png",
        "sha256": f"{index + 1:064x}",
        "size_bytes": 128,
        "width": 640,
        "height": 480,
        "encoding_id": "png_binary_mask_v1",
    }


def _task_result(
    *, index: int = 0, task: str = "object_detection"
) -> dict[str, Any]:
    return {
        "class_id": index,
        "score": "0.9",
        "bbox": _bbox(offset="0" if index % 2 == 0 else "1"),
        "mask": None if task == "object_detection" else _mask(index=index),
    }


def _due(index: int) -> tuple[int, int]:
    fraction = Fraction(index * 1_000_000_000, 30)
    return fraction.numerator, fraction.denominator


def _frame(
    index: int,
    *,
    task: str = "object_detection",
    results: list[dict[str, Any]] | None = None,
) -> FrameResult:
    due_num, due_den = _due(index)
    record: dict[str, Any] = {
        "schema_version": 1,
        "source_frame_index": index,
        "scheduled_due_offset_num_ns": due_num,
        "scheduled_due_offset_den": due_den,
        "processing_completed_offset_ns": (due_num + due_den - 1) // due_den + 1_000,
        "device_timestamp_ns": None,
        "task": task,
        "decoded_width": 640,
        "decoded_height": 480,
        "task_results": results if results is not None else [_task_result(task=task)],
        "frame_result_digest": "0" * 64,
    }
    record["frame_result_digest"] = canonical_sha256_v1(
        record, own_digest_field="frame_result_digest"
    )
    return validate_frame_result(record, expected_task=task)


def _estimate(*, source: str, task: str = "object_detection") -> dict[str, Any]:
    return {
        "source_semantics": source,
        "confidence": "0.8",
        "bbox": _bbox(offset="1"),
        "mask": None if task == "object_detection" else _mask(index=99),
    }


def _common_row(
    frame: FrameResult,
    *,
    interface: dict[str, Any],
    track_id: int = 1,
    state: str,
    session_index: int = 0,
    session_frame_index: int | None = None,
    prediction_age: int = 0,
    lost_age: int = 0,
) -> dict[str, Any]:
    source = frame.to_dict()
    if session_frame_index is None:
        session_frame_index = source["source_frame_index"]
    return {
        "schema_version": 1,
        "kind": "tracking_result",
        "tracking_output_interface_digest": interface[
            "tracking_output_interface_digest"
        ],
        "source_frame_index": source["source_frame_index"],
        "session_index": session_index,
        "session_frame_index": session_frame_index,
        "track_id": track_id,
        "state": state,
        "row_source": {
            "observed": "detector_observation_with_tracker_estimate",
            "predicted": "tracker_prediction",
            "lost": "lifecycle_only",
            "ended": "lifecycle_only",
        }[state],
        "source_scheduled_due_offset_num_ns": source[
            "scheduled_due_offset_num_ns"
        ],
        "source_scheduled_due_offset_den": source["scheduled_due_offset_den"],
        "tracking_completed_offset_ns": source["processing_completed_offset_ns"]
        + 1_000,
        "consecutive_prediction_frames": prediction_age,
        "consecutive_lost_frames": lost_age,
    }


def _observed(
    frame: FrameResult,
    *,
    interface: dict[str, Any],
    track_id: int = 1,
    source_result_index: int = 0,
    session_index: int = 0,
    session_frame_index: int | None = None,
    include_copy: bool = True,
) -> dict[str, Any]:
    row = _common_row(
        frame,
        interface=interface,
        track_id=track_id,
        state="observed",
        session_index=session_index,
        session_frame_index=session_frame_index,
    )
    source = frame.to_dict()
    row["observation_ref"] = {
        "source_frame_result_digest": source["frame_result_digest"],
        "source_result_index": source_result_index,
    }
    if include_copy:
        row["observation_copy"] = copy.deepcopy(
            source["task_results"][source_result_index]
        )
    row["track_estimate"] = _estimate(
        source="observation_adjusted", task=interface["task"]
    )
    return row


def _predicted(
    frame: FrameResult,
    *,
    interface: dict[str, Any],
    age: int,
    track_id: int = 1,
    session_index: int = 0,
    session_frame_index: int | None = None,
) -> dict[str, Any]:
    row = _common_row(
        frame,
        interface=interface,
        track_id=track_id,
        state="predicted",
        session_index=session_index,
        session_frame_index=session_frame_index,
        prediction_age=age,
    )
    row["track_estimate"] = _estimate(
        source="tracker_prediction", task=interface["task"]
    )
    return row


def _lost(
    frame: FrameResult,
    *,
    interface: dict[str, Any],
    age: int,
    track_id: int = 1,
    session_index: int = 0,
    session_frame_index: int | None = None,
) -> dict[str, Any]:
    return _common_row(
        frame,
        interface=interface,
        track_id=track_id,
        state="lost",
        session_index=session_index,
        session_frame_index=session_frame_index,
        lost_age=age,
    )


def _ended(
    frame: FrameResult,
    *,
    interface: dict[str, Any],
    track_id: int = 1,
    session_index: int = 0,
    session_frame_index: int | None = None,
) -> dict[str, Any]:
    return _common_row(
        frame,
        interface=interface,
        track_id=track_id,
        state="ended",
        session_index=session_index,
        session_frame_index=session_frame_index,
    )


def _termination(
    *,
    interface: dict[str, Any],
    reason: str,
    session_index: int,
    last_source_frame_index: int | None,
    last_session_frame_index: int | None,
    offset_ns: int,
    active_count: int,
    lost_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "tracking_session_termination",
        "tracking_output_interface_digest": interface[
            "tracking_output_interface_digest"
        ],
        "session_index": session_index,
        "last_source_frame_index": last_source_frame_index,
        "last_session_frame_index": last_session_frame_index,
        "termination_reason": reason,
        "termination_offset_ns": offset_ns,
        "active_track_count": active_count,
        "lost_track_count": lost_count,
    }


def _line(record: dict[str, Any]) -> bytes:
    return canonical_json_v1(record) + b"\n"


def _tracking_stream_job(
    *,
    task: str = "object_detection",
    max_results_per_frame: int = 100,
    max_total_results: int = 1_000_000,
    max_mask_artifacts: int | None = None,
    max_output_files: int | None = None,
) -> StreamJobSpec:
    record = _stream_job_record(task=task)
    if max_mask_artifacts is None:
        max_mask_artifacts = 0 if task == "object_detection" else 10_000
    if max_output_files is None:
        max_output_files = 5 if task == "object_detection" else 10_004
    record.update(
        {
            "max_results_per_frame": max_results_per_frame,
            "max_total_results": max_total_results,
            "max_mask_artifacts": max_mask_artifacts,
            "max_output_files": max_output_files,
        }
    )
    return validate_stream_job_spec(record)


def _artifact_bundle(
    frames: list[FrameResult],
    tracking_records: list[dict[str, Any]],
    *,
    interface: dict[str, Any],
    stream_job: StreamJobSpec,
    extra_outputs: dict[str, bytes] | None = None,
) -> tuple[list[tuple[str, bytes | tuple[bytes, ...]]], bytes, int]:
    detector_lines = [frame.canonical_line() for frame in frames]
    tracking_lines = [_line(record) for record in tracking_records]
    outputs: dict[str, bytes | tuple[bytes, ...]] = {
        "detector_frame_results.jsonl": tuple(detector_lines),
        "tracking_results.jsonl": tuple(tracking_lines),
    }
    outputs.update(extra_outputs or {})
    job = stream_job.to_dict()
    result_count = sum(len(frame.to_dict()["task_results"]) for frame in frames)
    mask_count = sum(
        path.startswith("artifacts/masks/") for path in (extra_outputs or {})
    )
    scheduled = 0 if not frames else max(
        frame.to_dict()["source_frame_index"] for frame in frames
    ) + 1
    summary = _stream_summary_record()
    summary.update(
        {
            "status": "completed",
            "task": job["task"],
            "source_kind": job["source"]["source_kind"],
            "scheduled_frame_count": scheduled,
            "processed_frame_count": len(frames),
            "dropped_frame_count": 0,
            "failed_unaccounted_frame_count": scheduled - len(frames),
            "max_consecutive_drops": 0,
            "duration_ns": len(frames) * 33_333_334,
            "p50_latency_ms": "1" if frames else None,
            "p95_latency_ms": "1" if frames else None,
            "p99_latency_ms": "1" if frames else None,
            "drop_fraction_display": "0" if frames else None,
            "result_count": result_count,
            "mask_artifact_count": mask_count,
            "output_file_count": len(outputs) + 3,
            "termination_reason": "normal_eof",
            "summary_digest": "0" * 64,
        }
    )

    detector_bytes = b"".join(detector_lines)
    tracking_bytes = b"".join(tracking_lines)
    for _iteration in range(8):
        summary["summary_digest"] = canonical_sha256_v1(
            summary, own_digest_field="summary_digest"
        )
        summary_bytes = canonical_json_v1(summary)
        provenance = {
            "schema_version": 1,
            "kind": "tracking_output_provenance",
            "tracking_output_interface_digest": interface[
                "tracking_output_interface_digest"
            ],
            "stream_job_digest": stream_job.local_job_digest,
            "summary_digest": summary["summary_digest"],
            "detector_frame_results_sha256": hashlib.sha256(
                detector_bytes
            ).hexdigest(),
            "tracking_results_sha256": hashlib.sha256(tracking_bytes).hexdigest(),
            "identity_scope": "session_only",
            "contains_frame_or_identity_data": False,
        }
        provenance_bytes = canonical_json_v1(provenance)
        outputs["provenance.json"] = provenance_bytes
        outputs["stream_summary.json"] = summary_bytes
        declared_payload_bytes = sum(
            len(value) if isinstance(value, bytes) else sum(map(len, value))
            for value in outputs.values()
        )
        if summary.get("output_bytes") == declared_payload_bytes:
            break
        summary["output_bytes"] = declared_payload_bytes
    else:
        raise AssertionError("summary output_bytes did not converge")
    validate_stream_summary(summary)

    paths = sorted(outputs, key=lambda path: path.encode("utf-8"))
    files: list[dict[str, Any]] = []
    for path in paths:
        value = outputs[path]
        data = value if isinstance(value, bytes) else b"".join(value)
        files.append(
            {
                "path": path,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = canonical_json_v1(
        {
            "schema_version": 1,
            "files": files,
            "expected_paths": paths,
            "file_count": len(files),
            "total_bytes": sum(item["size_bytes"] for item in files),
        }
    )
    declared = [(path, outputs[path]) for path in paths]
    return declared, manifest, sum(item["size_bytes"] for item in files) + len(
        manifest
    )


def _manifest_value(data: bytes) -> dict[str, Any]:
    return json.loads(data)


def _manifest_for_declared(
    declared: list[tuple[str, bytes | tuple[bytes, ...]]],
) -> tuple[bytes, int]:
    files: list[dict[str, Any]] = []
    for path, chunks in declared:
        data = chunks if isinstance(chunks, bytes) else b"".join(chunks)
        files.append(
            {
                "path": path,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    value = {
        "schema_version": 1,
        "files": files,
        "expected_paths": [item["path"] for item in files],
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
    }
    manifest = canonical_json_v1(value)
    return manifest, value["total_bytes"] + len(manifest)


class _ReportedLengthSet(set[int]):
    def __init__(self, reported_length: int) -> None:
        super().__init__()
        self._reported_length = reported_length

    def __len__(self) -> int:
        return self._reported_length


class TestTrackingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.interface_schema = json.loads(
            (cls.root / "docs/schemas/tracking_output_interface.schema.json").read_text(
                encoding="utf-8"
            )
        )
        cls.record_schema = json.loads(
            (cls.root / "docs/schemas/tracking_output_record.schema.json").read_text(
                encoding="utf-8"
            )
        )

    def test_interface_digest_fixed_caps_privacy_and_schema(self) -> None:
        payload = _interface()
        validated = validate_tracking_output_interface(payload)
        self.assertEqual(validated.to_dict(), payload)
        self.assertTrue(
            _schema_accepts(payload, self.interface_schema, root=self.interface_schema)
        )
        for field, value in (
            ("detector_interval", 0),
            ("detector_interval", 864_001),
            ("max_results_per_frame", 1_001),
            ("max_prediction_frames", 241),
            ("max_lost_frames", -1),
            ("max_active_tracks", 999),
            ("identity_scope", "global"),
            ("biometric_inference", True),
            ("cross_session_identity_linking", True),
            ("persistent_identity_database", True),
        ):
            changed = copy.deepcopy(payload)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(TrackingContractError):
                validate_tracking_output_interface(changed)
        tampered = copy.deepcopy(payload)
        tampered["task"] = "instance_segmentation"
        with self.assertRaisesRegex(TrackingContractError, "digest"):
            validate_tracking_output_interface(tampered)

    def test_observed_reference_copy_estimate_and_schema(self) -> None:
        interface = _interface()
        frame = _frame(0)
        row = _observed(frame, interface=interface)
        checked = validate_tracking_output_record(
            row, interface=interface, frame_result=frame
        )
        self.assertEqual(checked, row)
        self.assertTrue(
            _schema_accepts(checked, self.record_schema, root=self.record_schema)
        )

        no_copy = _observed(frame, interface=interface, include_copy=False)
        validate_tracking_output_record(no_copy, interface=interface, frame_result=frame)
        for mutation in (
            lambda item: item["observation_ref"].__setitem__(
                "source_frame_result_digest", "f" * 64
            ),
            lambda item: item["observation_ref"].__setitem__("source_result_index", 1),
            lambda item: item["observation_copy"].__setitem__("score", "0.8"),
            lambda item: item["track_estimate"].__setitem__(
                "source_semantics", "tracker_prediction"
            ),
            lambda item: item.__setitem__("track_id", JS_SAFE_TRACK_ID_MAX + 1),
            lambda item: item.__setitem__("track_id", True),
        ):
            changed = copy.deepcopy(row)
            mutation(changed)
            with self.assertRaises(TrackingContractError):
                validate_tracking_output_record(
                    changed, interface=interface, frame_result=frame
                )

        for invalid_id in (0, -1, 1.5, "1", True, JS_SAFE_TRACK_ID_MAX + 1):
            changed = _observed(frame, interface=interface)
            changed["track_id"] = invalid_id
            with self.subTest(track_id=invalid_id), self.assertRaises(
                TrackingContractError
            ):
                validate_tracking_output_record(
                    changed, interface=interface, frame_result=frame
                )

        cross_frame = _observed(frame, interface=interface)
        cross_frame["observation_ref"]["source_frame_result_digest"] = _frame(
            1
        ).to_dict()["frame_result_digest"]
        with self.assertRaisesRegex(TrackingContractError, "cross-frame"):
            validate_tracking_output_record(
                cross_frame, interface=interface, frame_result=frame
            )

    def test_state_specific_required_and_forbidden_fields(self) -> None:
        interface = _interface()
        frame = _frame(0)
        invalid: list[dict[str, Any]] = []
        observed = _observed(frame, interface=interface)
        del observed["track_estimate"]
        invalid.append(observed)
        predicted = _predicted(frame, interface=interface, age=1)
        predicted["observation_ref"] = {
            "source_frame_result_digest": frame.to_dict()["frame_result_digest"],
            "source_result_index": 0,
        }
        invalid.append(predicted)
        lost = _lost(frame, interface=interface, age=1)
        lost["track_estimate"] = _estimate(source="tracker_prediction")
        invalid.append(lost)
        ended = _ended(frame, interface=interface)
        ended["observation_copy"] = copy.deepcopy(frame.to_dict()["task_results"][0])
        invalid.append(ended)
        wrong_source = _lost(frame, interface=interface, age=1)
        wrong_source["row_source"] = "tracker_prediction"
        invalid.append(wrong_source)
        for row in invalid:
            with self.subTest(state=row["state"]), self.assertRaises(
                TrackingContractError
            ):
                validate_tracking_output_record(
                    row, interface=interface, frame_result=frame
                )
            self.assertFalse(
                _schema_accepts(row, self.record_schema, root=self.record_schema)
            )

    def test_allowed_transitions_reappearance_ended_and_reuse(self) -> None:
        interface = _interface()
        machine = TrackingStateMachine(interface)
        sequence = [
            _observed(_frame(0), interface=interface),
            _predicted(_frame(1), interface=interface, age=1),
            _observed(_frame(2), interface=interface),
            _lost(_frame(3), interface=interface, age=1),
            _observed(_frame(4), interface=interface),
            _ended(_frame(5), interface=interface),
        ]
        for index, row in enumerate(sequence):
            machine.validate_frame_batch(_frame(index), [row])
        with self.assertRaisesRegex(TrackingContractError, "cannot be reused"):
            machine.validate_frame_batch(
                _frame(6), [_observed(_frame(6), interface=interface)]
            )

        for state in ("predicted", "lost", "ended"):
            fresh = TrackingStateMachine(interface)
            frame = _frame(0)
            row = {
                "predicted": _predicted(frame, interface=interface, age=1),
                "lost": _lost(frame, interface=interface, age=1),
                "ended": _ended(frame, interface=interface),
            }[state]
            with self.subTest(state=state), self.assertRaises(TrackingContractError):
                fresh.validate_frame_batch(frame, [row])

    def test_prediction_and_lost_ages_zero_240_and_max_plus_one(self) -> None:
        interface = _interface()
        predicted_machine = TrackingStateMachine(interface)
        predicted_machine.validate_frame_batch(
            _frame(0), [_observed(_frame(0), interface=interface)]
        )
        for age in range(1, 241):
            frame = _frame(age)
            predicted_machine.validate_frame_batch(
                frame, [_predicted(frame, interface=interface, age=age)]
            )
        frame = _frame(241)
        with self.assertRaises(TrackingContractError):
            predicted_machine.validate_frame_batch(
                frame, [_predicted(frame, interface=interface, age=240)]
            )
        predicted_machine.validate_frame_batch(
            frame, [_ended(frame, interface=interface)]
        )

        lost_machine = TrackingStateMachine(interface)
        lost_machine.validate_frame_batch(
            _frame(0), [_observed(_frame(0), interface=interface)]
        )
        for age in range(1, 241):
            frame = _frame(age)
            lost_machine.validate_frame_batch(
                frame, [_lost(frame, interface=interface, age=age)]
            )
        frame = _frame(241)
        with self.assertRaises(TrackingContractError):
            lost_machine.validate_frame_batch(
                frame, [_lost(frame, interface=interface, age=240)]
            )

        zero = _interface(max_prediction_frames=0, max_lost_frames=0)
        for row_factory in (
            lambda item: _predicted(item, interface=zero, age=1),
            lambda item: _lost(item, interface=zero, age=1),
        ):
            machine = TrackingStateMachine(zero)
            machine.validate_frame_batch(
                _frame(0), [_observed(_frame(0), interface=zero)]
            )
            frame = _frame(1)
            with self.assertRaises(TrackingContractError):
                machine.validate_frame_batch(frame, [row_factory(frame)])

    def test_indices_cadence_reset_and_same_id_restart(self) -> None:
        interface = _interface(detector_interval=2)
        machine = TrackingStateMachine(interface)
        frame0 = _frame(0)
        machine.validate_frame_batch(frame0, [_observed(frame0, interface=interface)])
        frame1 = _frame(1)
        with self.assertRaisesRegex(TrackingContractError, "cadence"):
            machine.validate_frame_batch(frame1, [_observed(frame1, interface=interface)])
        wrong_session_frame = _predicted(
            frame1, interface=interface, age=1, session_frame_index=2
        )
        with self.assertRaisesRegex(TrackingContractError, "session_frame_index"):
            machine.validate_frame_batch(frame1, [wrong_session_frame])
        machine.validate_frame_batch(
            frame1, [_predicted(frame1, interface=interface, age=1)]
        )
        frame2 = _frame(4)
        machine.validate_frame_batch(
            frame2,
            [
                _observed(
                    frame2,
                    interface=interface,
                    session_frame_index=2,
                )
            ],
        )
        machine.terminate_session(
            _termination(
                interface=interface,
                reason="reset",
                session_index=0,
                last_source_frame_index=4,
                last_session_frame_index=2,
                offset_ns=frame2.to_dict()["processing_completed_offset_ns"] + 2_000,
                active_count=1,
                lost_count=0,
            )
        )
        frame3 = _frame(7)
        wrong_session = _observed(
            frame3,
            interface=interface,
            track_id=1,
            session_index=0,
            session_frame_index=0,
        )
        with self.assertRaisesRegex(TrackingContractError, "session_index"):
            machine.validate_frame_batch(frame3, [wrong_session])
        machine.validate_frame_batch(
            frame3,
            [
                _observed(
                    frame3,
                    interface=interface,
                    track_id=1,
                    session_index=1,
                    session_frame_index=0,
                )
            ],
        )
        self.assertEqual(machine.session_index, 1)

    def test_row_order_active_coverage_duplicate_links_and_atomicity(self) -> None:
        interface = _interface()
        results = [_task_result(index=0), _task_result(index=1)]
        frame0 = _frame(0, results=results)
        machine = TrackingStateMachine(interface)
        machine.validate_frame_batch(
            frame0,
            [
                _observed(frame0, interface=interface, track_id=1, source_result_index=0),
                _observed(frame0, interface=interface, track_id=2, source_result_index=1),
            ],
        )
        budget = machine.job_budget_used
        frame1 = _frame(1, results=results)
        with self.assertRaisesRegex(TrackingContractError, "active track omitted"):
            machine.validate_frame_batch(
                frame1, [_observed(frame1, interface=interface, track_id=1)]
            )
        self.assertEqual(machine.job_budget_used, budget)
        with self.assertRaisesRegex(TrackingContractError, "ascending"):
            machine.validate_frame_batch(
                frame1,
                [
                    _observed(frame1, interface=interface, track_id=2, source_result_index=1),
                    _observed(frame1, interface=interface, track_id=1, source_result_index=0),
                ],
            )
        self.assertEqual(machine.job_budget_used, budget)

        duplicate_machine = TrackingStateMachine(interface)
        with self.assertRaisesRegex(TrackingContractError, "duplicate source-result"):
            duplicate_machine.validate_frame_batch(
                frame0,
                [
                    _observed(frame0, interface=interface, track_id=1, source_result_index=0),
                    _observed(frame0, interface=interface, track_id=2, source_result_index=0),
                ],
            )
        self.assertEqual(duplicate_machine.job_budget_used, 0)

    def test_per_frame_session_unique_and_nonresetting_budget_caps(self) -> None:
        one = _interface(max_results_per_frame=1)
        frame = _frame(0, results=[_task_result(index=0), _task_result(index=1)])
        machine = TrackingStateMachine(one)
        with self.assertRaisesRegex(TrackingContractError, "per-frame"):
            machine.validate_frame_batch(
                frame,
                [
                    _observed(frame, interface=one, track_id=1, source_result_index=0),
                    _observed(frame, interface=one, track_id=2, source_result_index=1),
                ],
            )
        self.assertEqual(machine.job_budget_used, 0)

        interface = _interface()
        unique_machine = TrackingStateMachine(interface)
        unique_machine._seen_ids = _ReportedLengthSet(MAX_SESSION_UNIQUE_TRACKS)  # type: ignore[attr-defined]
        with self.assertRaisesRegex(TrackingContractError, "unique-ID"):
            unique_machine.validate_frame_batch(
                _frame(0), [_observed(_frame(0), interface=interface)]
            )
        self.assertEqual(unique_machine.job_budget_used, 0)

        budget_machine = TrackingStateMachine(interface)
        budget_machine._job_budget_used = MAX_JOB_ROW_STATE_BUDGET - 1  # type: ignore[attr-defined]
        with self.assertRaisesRegex(TrackingContractError, "nonresetting"):
            budget_machine.validate_frame_batch(
                _frame(0), [_observed(_frame(0), interface=interface)]
            )
        self.assertEqual(budget_machine.job_budget_used, MAX_JOB_ROW_STATE_BUDGET - 1)
        self.assertEqual(MAX_ACTIVE_TRACKS, 1_000)

        exact_machine = TrackingStateMachine(interface)
        first = _frame(0)
        exact_machine.validate_frame_batch(
            first, [_observed(first, interface=interface)]
        )
        exact_machine._job_budget_used = MAX_JOB_ROW_STATE_BUDGET - 1  # type: ignore[attr-defined]
        second = _frame(1)
        exact_machine.validate_frame_batch(
            second, [_predicted(second, interface=interface, age=1)]
        )
        self.assertEqual(exact_machine.job_budget_used, MAX_JOB_ROW_STATE_BUDGET)
        third = _frame(2)
        with self.assertRaisesRegex(TrackingContractError, "nonresetting"):
            exact_machine.validate_frame_batch(
                third, [_predicted(third, interface=interface, age=2)]
            )

    def test_exact_active_track_max_and_timestamp_monotonicity(self) -> None:
        interface = _interface()
        results = [_task_result(index=index) for index in range(MAX_ACTIVE_TRACKS)]
        frame = _frame(0, results=results)
        rows = [
            _observed(
                frame,
                interface=interface,
                track_id=index + 1,
                source_result_index=index,
            )
            for index in range(MAX_ACTIVE_TRACKS)
        ]
        machine = TrackingStateMachine(interface)
        machine.validate_frame_batch(frame, rows)
        self.assertEqual(machine.job_budget_used, MAX_ACTIVE_TRACKS * 2)

        regressing_due = _frame(1).to_dict()
        regressing_due["scheduled_due_offset_num_ns"] = 0
        regressing_due["scheduled_due_offset_den"] = 1
        regressing_due["frame_result_digest"] = canonical_sha256_v1(
            regressing_due, own_digest_field="frame_result_digest"
        )
        regressing_frame = validate_frame_result(regressing_due)
        with self.assertRaisesRegex(TrackingContractError, "due offset"):
            machine.validate_frame_batch(
                regressing_frame,
                [
                    _predicted(
                        regressing_frame,
                        interface=interface,
                        age=1,
                        track_id=index + 1,
                    )
                    for index in range(MAX_ACTIVE_TRACKS)
                ],
            )

    def test_session_termination_counts_reasons_and_no_row_explosion(self) -> None:
        interface = _interface()
        for reason in ("eof", "cancelled", "terminal_failure"):
            machine = TrackingStateMachine(interface)
            frame = _frame(0)
            machine.validate_frame_batch(
                frame, [_observed(frame, interface=interface)]
            )
            termination = _termination(
                interface=interface,
                reason=reason,
                session_index=0,
                last_source_frame_index=0,
                last_session_frame_index=0,
                offset_ns=frame.to_dict()["processing_completed_offset_ns"] + 2_000,
                active_count=1,
                lost_count=0,
            )
            self.assertTrue(
                _schema_accepts(
                    termination, self.record_schema, root=self.record_schema
                )
            )
            self.assertEqual(machine.terminate_session(termination), termination)
            self.assertTrue(machine.final)
            self.assertEqual(machine.tracking_row_count, 1)
            with self.assertRaises(TrackingContractError):
                machine.terminate_session(termination)

        empty = TrackingStateMachine(interface)
        empty.terminate_session(
            _termination(
                interface=interface,
                reason="eof",
                session_index=0,
                last_source_frame_index=None,
                last_session_frame_index=None,
                offset_ns=0,
                active_count=0,
                lost_count=0,
            )
        )
        self.assertTrue(empty.final)

        mismatched_pair = _termination(
            interface=interface,
            reason="eof",
            session_index=0,
            last_source_frame_index=None,
            last_session_frame_index=0,
            offset_ns=0,
            active_count=0,
            lost_count=0,
        )
        self.assertFalse(
            _schema_accepts(
                mismatched_pair, self.record_schema, root=self.record_schema
            )
        )
        with self.assertRaises(TrackingContractError):
            validate_tracking_output_record(mismatched_pair, interface=interface)

    def test_incremental_stream_validation_and_content_free_summary(self) -> None:
        interface = _interface()
        frame0 = _frame(0)
        frame1 = _frame(1, results=[])
        observed = _observed(frame0, interface=interface)
        lost = _lost(frame1, interface=interface, age=1)
        termination = _termination(
            interface=interface,
            reason="eof",
            session_index=0,
            last_source_frame_index=1,
            last_session_frame_index=1,
            offset_ns=frame1.to_dict()["processing_completed_offset_ns"] + 2_000,
            active_count=0,
            lost_count=1,
        )
        summary = validate_tracking_output_streams(
            [frame0.canonical_line(), frame1.canonical_line()],
            [_line(observed), _line(lost), _line(termination)],
            interface=interface,
            expected_frame_result_count=2,
            expected_task_result_count=1,
        )
        value = summary.to_dict()
        self.assertEqual(value["detector_frame_result_count"], 2)
        self.assertEqual(value["detector_task_result_count"], 1)
        self.assertEqual(value["tracking_row_count"], 2)
        self.assertEqual(value["job_row_state_budget_used"], 3)
        self.assertFalse(value["contains_frame_or_identity_data"])
        self.assertNotIn("track_id", json.dumps(value))

    def test_stream_rejects_tamper_noncanonical_dangling_duplicate_and_missing_end(self) -> None:
        interface = _interface()
        frame = _frame(0)
        observed = _observed(frame, interface=interface)
        termination = _termination(
            interface=interface,
            reason="eof",
            session_index=0,
            last_source_frame_index=0,
            last_session_frame_index=0,
            offset_ns=frame.to_dict()["processing_completed_offset_ns"] + 2_000,
            active_count=1,
            lost_count=0,
        )
        tampered = frame.to_dict()
        tampered["task_results"][0]["score"] = "0.8"
        cases = (
            ([_line(tampered)], [_line(observed), _line(termination)]),
            ([b" " + frame.canonical_line()], [_line(observed), _line(termination)]),
            ([frame.canonical_bytes() + b"\r\n"], [_line(observed), _line(termination)]),
            ([frame.canonical_line()], [_line(observed)]),
            ([], [_line(observed), _line(termination)]),
            (
                [frame.canonical_line(), frame.canonical_line()],
                [_line(observed), _line(termination)],
            ),
        )
        for detector, tracking in cases:
            with self.subTest(detector_lines=len(detector)), self.assertRaises(
                TrackingContractError
            ):
                validate_tracking_output_streams(
                    detector,
                    tracking,
                    interface=interface,
                    expected_frame_result_count=1,
                    expected_task_result_count=1,
                )

        with self.assertRaisesRegex(TrackingContractError, "processed-frame count"):
            validate_tracking_output_streams(
                [frame.canonical_line()],
                [_line(observed), _line(termination)],
                interface=interface,
                expected_frame_result_count=2,
                expected_task_result_count=1,
            )

    def test_aggregate_derives_real_summary_and_enforces_stream_job(self) -> None:
        interface = _interface()
        exact_job = _tracking_stream_job(
            max_results_per_frame=2, max_total_results=2
        )
        frame = _frame(0, results=[_task_result(index=0), _task_result(index=1)])
        termination = _termination(
            interface=interface,
            reason="eof",
            session_index=0,
            last_source_frame_index=0,
            last_session_frame_index=0,
            offset_ns=frame.to_dict()["processing_completed_offset_ns"] + 2_000,
            active_count=0,
            lost_count=0,
        )
        declared, manifest, total_bytes = _artifact_bundle(
            [frame], [termination], interface=interface, stream_job=exact_job
        )
        summary = validate_tracking_output_artifacts(
            declared,
            manifest,
            interface=interface,
            stream_job=exact_job,
        )
        self.assertEqual(summary.detector_frame_result_count, 1)
        self.assertEqual(summary.detector_task_result_count, 2)
        self.assertLess(total_bytes, exact_job.to_dict()["max_output_bytes"])
        output_by_path = dict(declared)
        validated_summary = validate_stream_summary(
            json.loads(output_by_path["stream_summary.json"])
        )
        provenance_value = json.loads(output_by_path["provenance.json"])
        self.assertTrue(
            _schema_accepts(
                provenance_value,
                {"$ref": "#/$defs/trackingProvenance"},
                root=self.record_schema,
            )
        )
        self.assertEqual(
            validate_tracking_output_provenance(
                provenance_value,
                interface=interface,
                stream_job=exact_job,
                stream_summary=validated_summary,
                detector_frame_results_sha256=hashlib.sha256(
                    b"".join(output_by_path["detector_frame_results.jsonl"])
                ).hexdigest(),
                tracking_results_sha256=hashlib.sha256(
                    b"".join(output_by_path["tracking_results.jsonl"])
                ).hexdigest(),
            )["identity_scope"],
            "session_only",
        )

        total_one_over = _tracking_stream_job(
            max_results_per_frame=2, max_total_results=1
        )
        with self.assertRaisesRegex(TrackingContractError, "total result cap"):
            validate_tracking_output_artifacts(
                declared,
                manifest,
                interface=interface,
                stream_job=total_one_over,
            )

        frame1 = _frame(1, results=[])
        termination2 = _termination(
            interface=interface,
            reason="eof",
            session_index=0,
            last_source_frame_index=1,
            last_session_frame_index=1,
            offset_ns=frame1.to_dict()["processing_completed_offset_ns"] + 2_000,
            active_count=0,
            lost_count=0,
        )
        per_frame_job = _tracking_stream_job(
            max_results_per_frame=1, max_total_results=2
        )
        per_frame_outputs, per_frame_manifest, _ = _artifact_bundle(
            [frame, frame1],
            [termination2],
            interface=interface,
            stream_job=per_frame_job,
        )
        with self.assertRaisesRegex(TrackingContractError, "per-frame detector"):
            validate_tracking_output_artifacts(
                per_frame_outputs,
                per_frame_manifest,
                interface=interface,
                stream_job=per_frame_job,
            )

        wrong_width = frame.to_dict()
        wrong_width["decoded_width"] = 641
        wrong_width["frame_result_digest"] = canonical_sha256_v1(
            wrong_width, own_digest_field="frame_result_digest"
        )
        wrong_width_frame = validate_frame_result(wrong_width)
        width_outputs, width_manifest, _ = _artifact_bundle(
            [wrong_width_frame],
            [termination],
            interface=interface,
            stream_job=exact_job,
        )
        with self.assertRaisesRegex(TrackingContractError, "decoded width"):
            validate_tracking_output_artifacts(
                width_outputs,
                width_manifest,
                interface=interface,
                stream_job=exact_job,
            )

        rate_record = _stream_job_record()
        rate_record["max_output_files"] = 5
        rate_record["source"]["source_rate_num"] = 25
        rate_job = validate_stream_job_spec(rate_record)
        rate_outputs, rate_manifest, _ = _artifact_bundle(
            [_frame(0), _frame(1)],
            [termination2],
            interface=interface,
            stream_job=rate_job,
        )
        with self.assertRaisesRegex(TrackingContractError, "scheduled due"):
            validate_tracking_output_artifacts(
                rate_outputs,
                rate_manifest,
                interface=interface,
                stream_job=rate_job,
            )

    def test_aggregate_keeps_independent_mask_cap_and_uses_available_file_slots(self) -> None:
        interface = _interface(task="instance_segmentation")
        mask_bytes = b"M" * 128
        result = _task_result(task="instance_segmentation")
        result["mask"]["sha256"] = hashlib.sha256(mask_bytes).hexdigest()
        result["mask"]["size_bytes"] = len(mask_bytes)
        frame = _frame(0, task="instance_segmentation", results=[result])
        termination = _termination(
            interface=interface,
            reason="eof",
            session_index=0,
            last_source_frame_index=0,
            last_session_frame_index=0,
            offset_ns=frame.to_dict()["processing_completed_offset_ns"] + 2_000,
            active_count=0,
            lost_count=0,
        )
        legal_max_job = _tracking_stream_job(task="instance_segmentation")
        self.assertEqual(legal_max_job.to_dict()["max_mask_artifacts"], 10_000)
        declared, manifest, _ = _artifact_bundle(
            [frame],
            [termination],
            interface=interface,
            stream_job=legal_max_job,
            extra_outputs={result["mask"]["relative_path"]: mask_bytes},
        )
        validate_tracking_output_artifacts(
            declared,
            manifest,
            interface=interface,
            stream_job=legal_max_job,
        )

        no_tracking_mask_slot = _tracking_stream_job(
            task="instance_segmentation",
            max_mask_artifacts=1,
            max_output_files=5,
        )
        slot_outputs, slot_manifest, _ = _artifact_bundle(
            [frame],
            [termination],
            interface=interface,
            stream_job=no_tracking_mask_slot,
            extra_outputs={result["mask"]["relative_path"]: mask_bytes},
        )
        with self.assertRaisesRegex(TrackingContractError, "output-file"):
            validate_tracking_output_artifacts(
                slot_outputs,
                slot_manifest,
                interface=interface,
                stream_job=no_tracking_mask_slot,
            )

        second_mask_bytes = b"N" * 128
        second_result = _task_result(index=1, task="instance_segmentation")
        second_result["mask"]["sha256"] = hashlib.sha256(
            second_mask_bytes
        ).hexdigest()
        second_result["mask"]["size_bytes"] = len(second_mask_bytes)
        mask_cap_job = _tracking_stream_job(
            task="instance_segmentation",
            max_results_per_frame=2,
            max_total_results=2,
            max_mask_artifacts=1,
            max_output_files=7,
        )
        two_mask_frame = _frame(
            0,
            task="instance_segmentation",
            results=[result, second_result],
        )
        two_mask_outputs, two_mask_manifest, _ = _artifact_bundle(
            [two_mask_frame],
            [termination],
            interface=interface,
            stream_job=mask_cap_job,
            extra_outputs={
                result["mask"]["relative_path"]: mask_bytes,
                second_result["mask"]["relative_path"]: second_mask_bytes,
            },
        )
        with self.assertRaisesRegex(TrackingContractError, "mask-artifact"):
            validate_tracking_output_artifacts(
                two_mask_outputs,
                two_mask_manifest,
                interface=interface,
                stream_job=mask_cap_job,
            )

        mismatched_result = copy.deepcopy(result)
        mismatched_result["mask"]["sha256"] = "f" * 64
        mismatched_frame = _frame(
            0, task="instance_segmentation", results=[mismatched_result]
        )
        mismatch_outputs, mismatch_manifest, _ = _artifact_bundle(
            [mismatched_frame],
            [termination],
            interface=interface,
            stream_job=legal_max_job,
            extra_outputs={result["mask"]["relative_path"]: mask_bytes},
        )
        with self.assertRaisesRegex(TrackingContractError, "mask reference"):
            validate_tracking_output_artifacts(
                mismatch_outputs,
                mismatch_manifest,
                interface=interface,
                stream_job=legal_max_job,
            )

    def test_aggregate_rejects_manifest_content_and_provenance_tamper(self) -> None:
        interface = _interface()
        job = _tracking_stream_job(max_mask_artifacts=1, max_output_files=6)
        frame = _frame(0)
        termination = _termination(
            interface=interface,
            reason="eof",
            session_index=0,
            last_source_frame_index=0,
            last_session_frame_index=0,
            offset_ns=frame.to_dict()["processing_completed_offset_ns"] + 2_000,
            active_count=0,
            lost_count=0,
        )
        declared, manifest, _ = _artifact_bundle(
            [frame], [termination], interface=interface, stream_job=job
        )
        kwargs = {"interface": interface, "stream_job": job}

        tampered_outputs = copy.deepcopy(declared)
        provenance_index = next(
            index
            for index, (path, _chunks) in enumerate(tampered_outputs)
            if path == "provenance.json"
        )
        original = tampered_outputs[provenance_index][1]
        self.assertIsInstance(original, bytes)
        tampered_outputs[provenance_index] = (
            "provenance.json",
            b"X" * len(original),
        )
        with self.assertRaisesRegex(TrackingContractError, "SHA-256"):
            validate_tracking_output_artifacts(tampered_outputs, manifest, **kwargs)

        with self.assertRaisesRegex(TrackingContractError, "exactly match"):
            validate_tracking_output_artifacts(declared[:-1], manifest, **kwargs)

        extra_outputs, extra_manifest, _ = _artifact_bundle(
            [frame],
            [termination],
            interface=interface,
            stream_job=job,
            extra_outputs={"artifacts/masks/unreferenced.png": b"mask"},
        )
        with self.assertRaisesRegex(TrackingContractError, "exactly match mask references"):
            validate_tracking_output_artifacts(
                extra_outputs, extra_manifest, **kwargs
            )

        value = _manifest_value(manifest)
        duplicate = copy.deepcopy(value)
        duplicate["files"].insert(1, copy.deepcopy(duplicate["files"][0]))
        duplicate["expected_paths"].insert(1, duplicate["expected_paths"][0])
        duplicate["file_count"] += 1
        duplicate["total_bytes"] += duplicate["files"][0]["size_bytes"]
        reordered = copy.deepcopy(value)
        reordered["files"][0], reordered["files"][1] = (
            reordered["files"][1], reordered["files"][0],
        )
        reordered["expected_paths"][0], reordered["expected_paths"][1] = (
            reordered["expected_paths"][1], reordered["expected_paths"][0],
        )
        missing = copy.deepcopy(value)
        removed = missing["files"].pop(1)
        missing["expected_paths"].pop(1)
        missing["file_count"] -= 1
        missing["total_bytes"] -= removed["size_bytes"]
        self_entry = copy.deepcopy(value)
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
        mismatch = copy.deepcopy(value)
        mismatch["files"][1]["sha256"] = "f" * 64
        byte_one_over = copy.deepcopy(value)
        byte_one_over["files"][0]["size_bytes"] = job.to_dict()[
            "max_output_bytes"
        ]
        byte_one_over["total_bytes"] = sum(
            item["size_bytes"] for item in byte_one_over["files"]
        )
        for name, changed in {
            "duplicate": duplicate,
            "reordered": reordered,
            "missing": missing,
            "self-entry": self_entry,
            "mismatched": mismatch,
            "byte-one-over": byte_one_over,
        }.items():
            with self.subTest(name=name), self.assertRaises(TrackingContractError):
                validate_tracking_output_artifacts(
                    declared, canonical_json_v1(changed), **kwargs
                )

        semantic_provenance = copy.deepcopy(declared)
        provenance = json.loads(semantic_provenance[provenance_index][1])
        provenance["stream_job_digest"] = "f" * 64
        semantic_provenance[provenance_index] = (
            "provenance.json",
            canonical_json_v1(provenance),
        )
        semantic_manifest, _ = _manifest_for_declared(semantic_provenance)
        with self.assertRaisesRegex(TrackingContractError, "privacy binding"):
            validate_tracking_output_artifacts(
                semantic_provenance, semantic_manifest, **kwargs
            )

        summary_index = next(
            index
            for index, (path, _chunks) in enumerate(declared)
            if path == "stream_summary.json"
        )
        noncanonical_summary = copy.deepcopy(declared)
        summary_value = json.loads(noncanonical_summary[summary_index][1])
        noncanonical_summary[summary_index] = (
            "stream_summary.json",
            json.dumps(summary_value, indent=2).encode("utf-8"),
        )
        noncanonical_manifest, _ = _manifest_for_declared(noncanonical_summary)
        with self.assertRaisesRegex(TrackingContractError, "canonical_json_v1"):
            validate_tracking_output_artifacts(
                noncanonical_summary, noncanonical_manifest, **kwargs
            )

        nonprogress = copy.deepcopy(declared)
        nonprogress[provenance_index] = (
            "provenance.json",
            (b"", original),
        )
        with self.assertRaisesRegex(TrackingContractError, "non-progress"):
            validate_tracking_output_artifacts(nonprogress, manifest, **kwargs)

        pretty = json.dumps(value, indent=2).encode("utf-8")
        with self.assertRaisesRegex(TrackingContractError, "canonical_json_v1"):
            validate_tracking_output_artifacts(declared, pretty, **kwargs)

    def test_aggregate_accepts_direct_empty_detector_bytes_for_zero_frames(self) -> None:
        interface = _interface()
        job = _tracking_stream_job()
        termination = _termination(
            interface=interface,
            reason="eof",
            session_index=0,
            last_source_frame_index=None,
            last_session_frame_index=None,
            offset_ns=0,
            active_count=0,
            lost_count=0,
        )
        declared, _manifest, _ = _artifact_bundle(
            [], [termination], interface=interface, stream_job=job
        )
        detector_index = next(
            index
            for index, (path, _chunks) in enumerate(declared)
            if path == "detector_frame_results.jsonl"
        )
        declared[detector_index] = ("detector_frame_results.jsonl", b"")
        manifest, _ = _manifest_for_declared(declared)
        summary = validate_tracking_output_artifacts(
            declared, manifest, interface=interface, stream_job=job
        )
        self.assertEqual(summary.detector_frame_result_count, 0)

    def test_instance_segmentation_mask_copy_and_estimate_are_exact(self) -> None:
        interface = _interface(task="instance_segmentation")
        frame = _frame(0, task="instance_segmentation")
        row = _observed(frame, interface=interface)
        validate_tracking_output_record(row, interface=interface, frame_result=frame)
        changed = copy.deepcopy(row)
        changed["observation_copy"]["mask"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(TrackingContractError, "mismatch"):
            validate_tracking_output_record(
                changed, interface=interface, frame_result=frame
            )

    def test_existing_predictions_interface_contract_remains_compatible(self) -> None:
        payload = [
            {
                "schema_version": 2,
                "image": "inputs/000000",
                "task": "object_detection",
                "detections": [
                    {
                        "class_id": 0,
                        "score": 0.9,
                        "bbox": {"cx": 0.4, "cy": 0.5, "w": 0.2, "h": 0.3},
                    }
                ],
            }
        ]
        result = validate_predictions_payload(payload, strict=True)
        self.assertEqual(result.warnings, [])
        self.assertNotIn("track_id", json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
