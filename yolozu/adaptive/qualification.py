"""Measured local image-pipeline qualification.

This Experimental module measures only an exact registered bundle on pinned
local inputs and artifacts. A report is evidence for one configuration; it is
unactivated and is not a Stable support or human-adoption claim.
"""

from __future__ import annotations

import io
import multiprocessing
import os
import pickle
import signal
import time
from array import array
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from PIL import Image, UnidentifiedImageError

from .artifact_resolver import ArtifactResolver, PinnedVerifiedArtifactSet
from .bundle_registry import AlgorithmRunner, RunnerProbeResult, load_algorithm_bundle_registry
from .bundles import (
    CODE_OWNED_RUNNER_IDS,
    AlgorithmBundleSpec,
    build_fixed_class_mapping,
    map_fixed_class_outputs,
    map_text_prompt_outputs,
)
from .canonical import canonical_decimal_v1, canonical_json_v1, canonical_sha256_v1
from .contracts import (
    ACCELERATOR_PROVIDER_IDS,
    CPU_PROVIDER_IDS,
    HANDOFF_MAX_MASK_ARTIFACTS,
    HANDOFF_MAX_OUTPUT_BYTES,
    HANDOFF_MAX_OUTPUT_FILES,
    HANDOFF_SCRATCH_CAP_BYTES,
    EnvironmentProfile,
    ImageJobSpec,
    build_qualification_workload_profile,
    validate_image_job_spec,
)
from .environment import build_environment_profile
from .evidence import (
    HANDOFF_ID,
    HANDOFF_VERSION,
    LATENCY_INTERVAL_ID,
    LATENCY_PHASES,
    MAX_SUSTAINED_SAMPLES,
    MIN_SUSTAINED_DURATION_NS,
    QualificationReport,
    compute_artifact_state_fingerprint,
    validate_local_artifact_inventory,
    validate_qualification_report,
)
from .inventory import PinnedDecodedInput, PinnedDecodedInputSet, pin_decoded_inputs
from .managed_output import ManagedOutputLimits, ManagedOutputTransaction

__all__ = [
    "QUALIFICATION_PROTOCOL_FINGERPRINT",
    "QualificationError",
    "QualificationEvaluator",
    "QualificationReport",
    "qualification_input_schedule",
    "nearest_rank_nanoseconds",
    "nanoseconds_to_milliseconds",
    "qualify_image_pipeline",
    "qualification_report_has_code_owned_issuer",
]


QUALIFICATION_TIMEOUT_MIN_SECONDS = 60
QUALIFICATION_TIMEOUT_MAX_SECONDS = 14_400
SOFT_REALTIME_MIN_TIMEOUT_SECONDS = 1_260
PROBE_TIMEOUT_SECONDS = 30
LOAD_TIMEOUT_SECONDS = 600
PREDICT_TIMEOUT_SECONDS = 300
CLOSE_TIMEOUT_SECONDS = 30
WARMUP_ITERATIONS = 20
REPEAT_COUNT = 3
REPEAT_ITERATIONS = 200
MAX_RUNNER_MESSAGE_BYTES = HANDOFF_SCRATCH_CAP_BYTES

_RSS_COLLECTOR = {
    "id": "process_tree_memory",
    "version": "1",
    "source_digest": canonical_sha256_v1(
        {
            "id": "process_tree_memory",
            "version": "1",
            "claim": "complete_runner_tree_or_unknown",
        }
    ),
}
_ACCELERATOR_COLLECTOR = {
    "id": "accelerator_memory",
    "version": "1",
    "source_digest": canonical_sha256_v1(
        {
            "id": "accelerator_memory",
            "version": "1",
            "claim": "all_runner_pids_all_declared_devices_or_unknown",
        }
    ),
}
_COLLECTOR_IDENTITY = {
    "id": "yolozu_qualifier",
    "version": "1",
    "source_digest": canonical_sha256_v1(
        {"module": "yolozu.adaptive.qualification", "interface_version": 1}
    ),
}
_ISSUER_IDENTITY = {
    "id": "yolozu_qualification_workflow",
    "version": "1",
    "source_digest": canonical_sha256_v1(
        {"workflow": "local_unactivated_qualification", "interface_version": 1}
    ),
}


def qualification_report_has_code_owned_issuer(
    report: QualificationReport | Mapping[str, Any],
) -> bool:
    """Return whether a validated report names the exact code-owned workflow.

    This checks the typed report identity only.  Activation trust is derived
    separately from the retained workflow and file boundary.
    """

    validated = (
        report
        if isinstance(report, QualificationReport)
        else validate_qualification_report(report)
    )
    payload = validated.to_dict()
    return (
        payload["collector"] == _COLLECTOR_IDENTITY
        and payload["issuer"] == _ISSUER_IDENTITY
    )

QUALIFICATION_PROTOCOL = {
    "schema_version": 1,
    "protocol_id": "image-pipeline-qualification-v1",
    "runner_policy": {
        "execution_trust_class": "code_owned_audited",
        "network_required": False,
        "os_network_isolation_claim": False,
        "blocking_work_boundary": "terminable_child_process_group",
    },
    "timeouts_seconds": {
        "probe": PROBE_TIMEOUT_SECONDS,
        "load": LOAD_TIMEOUT_SECONDS,
        "predict_each": PREDICT_TIMEOUT_SECONDS,
        "close": CLOSE_TIMEOUT_SECONDS,
        "qualification_min": QUALIFICATION_TIMEOUT_MIN_SECONDS,
        "qualification_max": QUALIFICATION_TIMEOUT_MAX_SECONDS,
        "soft_realtime_min": SOFT_REALTIME_MIN_TIMEOUT_SECONDS,
    },
    "cold_start": {
        "input_index": 0,
        "fresh_runner": True,
        "os_cache_state": "uncontrolled",
        "starts_before": "runner_process_creation",
        "ends_after": "first_strict_validated_handoff",
    },
    "warmup": {
        "iteration_count": WARMUP_ITERATIONS,
        "schedule": "i_mod_input_count",
    },
    "repeats": {
        "repeat_count": REPEAT_COUNT,
        "iteration_count": REPEAT_ITERATIONS,
        "schedule": "reset_zero_then_i_mod_input_count",
        "percentile": "exact_nearest_rank",
        "raw_timing_retention": False,
    },
    "latency_interval": {
        "interval_id": LATENCY_INTERVAL_ID,
        "handoff_id": HANDOFF_ID,
        "handoff_version": HANDOFF_VERSION,
        "included_phases": list(LATENCY_PHASES),
        "publication_boundary": "managed_output_transaction_after_interval",
    },
    "memory": {
        "rss": _RSS_COLLECTOR,
        "accelerator": _ACCELERATOR_COLLECTOR,
        "incomplete_coverage": "unknown",
    },
    "soft_realtime": {
        "minimum_duration_ns": MIN_SUSTAINED_DURATION_NS,
        "maximum_samples": MAX_SUSTAINED_SAMPLES,
        "sample_type": "uint64",
        "sample_storage_bytes_max": MAX_SUSTAINED_SAMPLES * 8,
        "schedule": "reset_zero_then_i_mod_input_count",
        "aggregation": "exact_nearest_rank_all_samples",
    },
    "quality": {
        "schedule": "exactly_once_each_unique_input_outside_timed_samples",
        "identity": "request_preregistered_exact_match",
        "private_identity_export": False,
    },
    "managed_output": {
        "required_files": ["qualification_report.json", "checksums.json"],
        "optional_private_prefix": "local_private",
        "maximum_masks": HANDOFF_MAX_MASK_ARTIFACTS,
        "maximum_files": HANDOFF_MAX_OUTPUT_FILES,
        "maximum_bytes": HANDOFF_MAX_OUTPUT_BYTES,
    },
}
QUALIFICATION_PROTOCOL_FINGERPRINT = canonical_sha256_v1(QUALIFICATION_PROTOCOL)


class QualificationError(ValueError):
    """One stable fail-closed qualification error."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> QualificationError:
    return QualificationError(code, detail)


class QualificationEvaluator(Protocol):
    """Code-owned evaluator bound to one preregistered quality identity."""

    evaluator_id: str
    evaluator_version: str
    source_digest: str
    metric_id: str
    direction: str
    evaluation_dataset_id: str
    evaluation_dataset_sha256: str
    evaluation_protocol_sha256: str
    evaluation_vocabulary_id: str

    def evaluate(
        self,
        *,
        predictions: tuple[bytes, ...],
        job: ImageJobSpec,
        bundle: AlgorithmBundleSpec,
    ) -> str:
        """Return one CanonicalDecimalV1 value for unique-input predictions."""


def qualification_input_schedule(input_count: int, iterations: int) -> tuple[int, ...]:
    """Return the frozen reset-at-zero cyclic input schedule."""

    if isinstance(input_count, bool) or not isinstance(input_count, int) or not 1 <= input_count <= 100:
        raise ValueError("input_count must be in 1..100")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < input_count * 2:
        raise ValueError("iterations must cover every input at least twice")
    return tuple(index % input_count for index in range(iterations))


def nearest_rank_nanoseconds(values: Sequence[int], percentile: int) -> int:
    """Compute the exact nearest-rank percentile over unsigned nanoseconds."""

    if percentile not in {50, 95, 99}:
        raise ValueError("percentile must be one of 50, 95, or 99")
    if not values:
        raise ValueError("nearest-rank requires at least one sample")
    checked: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**64 - 1:
            raise ValueError("latency sample must be an unsigned 64-bit integer")
        checked.append(value)
    checked.sort()
    rank = (percentile * len(checked) + 99) // 100
    return checked[rank - 1]


def nanoseconds_to_milliseconds(value: int) -> str:
    """Convert integer nanoseconds to an exact CanonicalDecimalV1 millisecond."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("nanoseconds must be a nonnegative integer")
    whole, fraction = divmod(value, 1_000_000)
    text = str(whole) if fraction == 0 else f"{whole}.{fraction:06d}".rstrip("0")
    return canonical_decimal_v1(text, field="latency_ms", nonnegative=True)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _unknown_rss() -> dict[str, Any]:
    return {
        "status": "unknown",
        "value_bytes": None,
        "collector_id": _RSS_COLLECTOR["id"],
        "collector_version": _RSS_COLLECTOR["version"],
        "collector_source_digest": _RSS_COLLECTOR["source_digest"],
        "scope": "unknown",
        "covered_processes": "unknown",
        "covered_devices": "not_applicable",
    }


def _accelerator_memory(environment: EnvironmentProfile) -> dict[str, Any]:
    present = any(
        item.get("probe_status") == "present"
        for item in environment.to_dict().get("accelerators", [])
    )
    if not present:
        return {
            "status": "not_applicable",
            "value_bytes": None,
            "collector_id": _ACCELERATOR_COLLECTOR["id"],
            "collector_version": _ACCELERATOR_COLLECTOR["version"],
            "collector_source_digest": _ACCELERATOR_COLLECTOR["source_digest"],
            "scope": "not_applicable",
            "covered_processes": "not_applicable",
            "covered_devices": "not_applicable",
        }
    return {
        "status": "unknown",
        "value_bytes": None,
        "collector_id": _ACCELERATOR_COLLECTOR["id"],
        "collector_version": _ACCELERATOR_COLLECTOR["version"],
        "collector_source_digest": _ACCELERATOR_COLLECTOR["source_digest"],
        "scope": "unknown",
        "covered_processes": "unknown",
        "covered_devices": "unknown",
    }


def _coverage(schedule: Sequence[int], input_count: int) -> list[int]:
    counts = [0] * input_count
    for index in schedule:
        counts[index] += 1
    if any(count < 2 for count in counts) or sum(counts) != len(schedule):
        raise _fail("schedule_coverage_failed", "timed repeat did not cover every input twice")
    return counts


def _score(value: Any) -> str:
    score = canonical_decimal_v1(value, field="runner_result.score", nonnegative=True)
    if Decimal(score) > Decimal(1):
        raise _fail("invalid_runner_output", "score exceeds 1")
    return score


def _bbox(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise _fail("invalid_runner_output", "bbox must contain four coordinates")
    coords = [
        canonical_decimal_v1(item, field="runner_result.bbox[]", nonnegative=True)
        for item in value
    ]
    if Decimal(coords[0]) > Decimal(coords[2]) or Decimal(coords[1]) > Decimal(coords[3]):
        raise _fail("invalid_runner_output", "bbox corners are reversed")
    return coords


def _validated_mask(value: Any, *, width: int, height: int) -> bytes:
    if not isinstance(value, bytes) or not value or len(value) > 16 * 1024 * 1024:
        raise _fail("invalid_runner_output", "mask_png must be bounded PNG bytes")
    try:
        with Image.open(io.BytesIO(value)) as mask:
            if mask.format != "PNG" or bool(getattr(mask, "is_animated", False)):
                raise _fail("invalid_runner_output", "mask must be one PNG frame")
            mask.load()
            if mask.size != (width, height) or mask.mode not in {"1", "L"}:
                raise _fail("invalid_runner_output", "mask size or mode is invalid")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise _fail("invalid_runner_output", "mask PNG is malformed") from exc
    return value


@dataclass(frozen=True)
class _Handoff:
    metadata: bytes
    masks: tuple[bytes, ...]


def _strict_handoff(
    *,
    raw_results: Sequence[Mapping[str, Any]],
    job: ImageJobSpec,
    bundle: AlgorithmBundleSpec,
    input_item: PinnedDecodedInput,
    input_width: int,
    input_height: int,
) -> _Handoff:
    job_record = job.to_dict()
    if not isinstance(raw_results, (list, tuple)):
        raise _fail("invalid_runner_output", "runner result must be a bounded sequence")
    if len(raw_results) > job_record["max_results_per_image"]:
        raise _fail("invalid_runner_output", "runner result count exceeds request cap")
    prompt_mode = job_record["prompt_mode"]
    mapping = (
        build_fixed_class_mapping(bundle, job.prompt_phrases)
        if prompt_mode == "fixed_classes"
        else None
    )
    native_indices: list[Any] = []
    prompt_indices: list[Any] = []
    records: list[dict[str, Any]] = []
    mask_blobs: list[bytes] = []
    for position, raw in enumerate(raw_results):
        if not isinstance(raw, Mapping):
            raise _fail("invalid_runner_output", "each runner result must be an object")
        item = dict(raw)
        class_key = "native_class_index" if prompt_mode == "fixed_classes" else "request_prompt_index"
        required = {class_key, "score", "bbox"}
        if job_record["task"] == "instance_segmentation":
            required.add("mask_png")
        if set(item) != required:
            raise _fail("invalid_runner_output", "runner result keys do not match the v1 handoff")
        if prompt_mode == "fixed_classes":
            native_indices.append(item[class_key])
        else:
            prompt_indices.append(item[class_key])
        normalized: dict[str, Any] = {
            "result_index": position,
            "score": _score(item["score"]),
            "bbox": _bbox(item["bbox"]),
        }
        if job_record["task"] == "instance_segmentation":
            mask = _validated_mask(item["mask_png"], width=input_width, height=input_height)
            if len(mask_blobs) >= HANDOFF_MAX_MASK_ARTIFACTS:
                raise _fail("handoff_limit_exceeded", "mask count exceeds 1000")
            normalized["mask_index"] = len(mask_blobs)
            mask_blobs.append(mask)
        records.append(normalized)

    labels = (
        map_fixed_class_outputs(mapping, native_indices)
        if mapping is not None
        else map_text_prompt_outputs(job.prompt_phrases, prompt_indices)
    )
    if mapping is not None:
        retained_native = set(mapping["request_to_bundle_class_index"])
        retained_positions = [
            index for index, value in enumerate(native_indices) if value in retained_native
        ]
        records = [records[index] for index in retained_positions]
        if job_record["task"] == "instance_segmentation":
            mask_blobs = [mask_blobs[index] for index in retained_positions]
            for index, record in enumerate(records):
                record["mask_index"] = index
    if len(labels) != len(records):
        raise _fail("invalid_runner_output", "class mapping and retained results disagree")
    for record, label in zip(records, labels):
        record.update(label)
    payload = canonical_json_v1(
        {
            "schema_version": 1,
            "handoff_id": HANDOFF_ID,
            "handoff_version": HANDOFF_VERSION,
            "input_index": input_item.input_index,
            "task": job_record["task"],
            "results": records,
        }
    )
    scratch = io.BytesIO()
    scratch.write(payload)
    for mask in mask_blobs:
        scratch.write(mask)
    if scratch.tell() > HANDOFF_SCRATCH_CAP_BYTES:
        raise _fail("handoff_limit_exceeded", "handoff scratch exceeds 512 MiB")
    scratch.flush()
    return _Handoff(metadata=payload, masks=tuple(mask_blobs))


class _RunnerSession(Protocol):
    runner_id: str
    runner_version: str

    def probe(self, timeout_seconds: int) -> RunnerProbeResult:
        raise NotImplementedError

    def load(self, timeout_seconds: int) -> None:
        raise NotImplementedError

    def warmup(self, index: int, timeout_seconds: int) -> None:
        raise NotImplementedError

    def predict(
        self, index: int, timeout_seconds: int
    ) -> tuple[Mapping[str, Any], ...]:
        raise NotImplementedError

    def close(self, timeout_seconds: int) -> None:
        raise NotImplementedError


def _runner_worker(
    connection: Any,
    factory: Callable[[], AlgorithmRunner],
    bundle: AlgorithmBundleSpec,
    environment: EnvironmentProfile,
    artifacts: PinnedVerifiedArtifactSet,
    inputs: PinnedDecodedInputSet,
    labels: tuple[str, ...],
) -> None:
    runner: AlgorithmRunner | None = None
    try:
        os.setsid()
        runner = factory()
        connection.send_bytes(
            pickle.dumps(
                {"ok": True, "runner_id": runner.runner_id, "runner_version": runner.runner_version},
                protocol=5,
            )
        )
        while True:
            request = pickle.loads(connection.recv_bytes(4096))
            operation = request["operation"]
            if operation == "probe":
                value: Any = runner.probe(bundle=bundle, environment=environment)
            elif operation == "load":
                runner.load(bundle=bundle, artifacts=artifacts)
                value = None
            elif operation == "warmup":
                runner.warmup(input_item=inputs[int(request["index"])])
                value = None
            elif operation == "predict":
                value = tuple(
                    runner.predict(
                        input_item=inputs[int(request["index"])],
                        requested_labels=labels,
                    )
                )
            elif operation == "close":
                runner.close()
                connection.send_bytes(pickle.dumps({"ok": True, "value": None}, protocol=5))
                return
            else:
                raise ValueError("unknown runner operation")
            encoded = pickle.dumps({"ok": True, "value": value}, protocol=5)
            if len(encoded) > MAX_RUNNER_MESSAGE_BYTES:
                raise ValueError("runner response exceeds bounded handoff")
            connection.send_bytes(encoded)
    except Exception as exc:
        try:
            message = str(exc)
            if len(message.encode("utf-8", "replace")) > 512:
                message = message.encode("utf-8", "replace")[:512].decode("utf-8", "ignore")
            connection.send_bytes(
                pickle.dumps(
                    {"ok": False, "error_type": type(exc).__name__, "detail": message},
                    protocol=5,
                )
            )
        except Exception:
            return
    finally:
        connection.close()


class _ForkedRunnerSession:
    """Run blocking/native runner work in one terminable POSIX process group."""

    def __init__(
        self,
        *,
        factory: Callable[[], AlgorithmRunner],
        bundle: AlgorithmBundleSpec,
        environment: EnvironmentProfile,
        artifacts: PinnedVerifiedArtifactSet,
        inputs: PinnedDecodedInputSet,
        labels: tuple[str, ...],
        outer_deadline_ns: int,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if os.name != "posix":
            raise _fail("platform_unsupported", "runner cancellation requires POSIX process groups")
        self._clock = monotonic_ns
        self._deadline = outer_deadline_ns
        context = multiprocessing.get_context("fork")
        parent, child = context.Pipe(duplex=True)
        self._connection = parent
        self._process = context.Process(
            target=_runner_worker,
            args=(child, factory, bundle, environment, artifacts, inputs, labels),
            daemon=False,
        )
        self._process.start()
        child.close()
        try:
            ready = self._receive(PROBE_TIMEOUT_SECONDS, phase="runner_start")
            if set(ready) != {"ok", "runner_id", "runner_version"} or not all(
                isinstance(ready[name], str)
                and 1 <= len(ready[name].encode("utf-8")) <= 128
                for name in ("runner_id", "runner_version")
            ):
                raise _fail("runner_failed", "runner startup identity is invalid")
        except Exception:
            self._terminate()
            self._connection.close()
            raise
        self.runner_id = ready["runner_id"]
        self.runner_version = ready["runner_version"]

    def _remaining_seconds(self, phase_seconds: int) -> float:
        remaining_ns = self._deadline - self._clock()
        if remaining_ns <= 0:
            self._terminate()
            raise _fail("qualification_timeout", "outer qualification watchdog expired")
        return min(float(phase_seconds), remaining_ns / 1_000_000_000)

    def _receive(self, timeout_seconds: int, *, phase: str) -> dict[str, Any]:
        timeout = self._remaining_seconds(timeout_seconds)
        if not self._connection.poll(timeout):
            self._terminate()
            raise _fail("phase_timeout", f"{phase} exceeded its timeout")
        try:
            payload = pickle.loads(self._connection.recv_bytes(MAX_RUNNER_MESSAGE_BYTES))
        except (EOFError, OSError, pickle.UnpicklingError) as exc:
            self._terminate()
            raise _fail("runner_failed", f"{phase} runner process ended") from exc
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            detail = payload.get("detail", "runner operation failed") if isinstance(payload, dict) else "invalid runner response"
            raise _fail("runner_failed", f"{phase}: {detail}")
        return payload

    def _call(self, operation: str, timeout_seconds: int, **payload: Any) -> Any:
        try:
            self._connection.send_bytes(
                pickle.dumps({"operation": operation, **payload}, protocol=5)
            )
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._terminate()
            raise _fail("runner_failed", f"{operation}: runner process is unavailable") from exc
        return self._receive(timeout_seconds, phase=operation).get("value")

    def probe(self, timeout_seconds: int) -> RunnerProbeResult:
        value = self._call("probe", timeout_seconds)
        if not isinstance(value, RunnerProbeResult):
            raise _fail("runner_failed", "probe returned an invalid typed result")
        return value

    def load(self, timeout_seconds: int) -> None:
        self._call("load", timeout_seconds)

    def warmup(self, index: int, timeout_seconds: int) -> None:
        self._call("warmup", timeout_seconds, index=index)

    def predict(self, index: int, timeout_seconds: int) -> tuple[Mapping[str, Any], ...]:
        value = self._call("predict", timeout_seconds, index=index)
        if not isinstance(value, tuple):
            raise _fail("invalid_runner_output", "predict did not return a tuple")
        return value

    def _terminate(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or not process.is_alive():
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                process.kill()
            process.join(timeout=1)

    def close(self, timeout_seconds: int) -> None:
        if self._process.is_alive():
            try:
                self._call("close", timeout_seconds)
            except QualificationError:
                self._terminate()
                raise
        self._process.join(timeout=1)
        if self._process.is_alive():
            self._terminate()
            raise _fail("phase_timeout", "runner close did not reap the child process")
        self._connection.close()


# Factories are populated only by repository-owned adapter modules. There are
# intentionally no generic import strings, entry points, or caller arguments.
_CODE_OWNED_RUNNER_FACTORIES: dict[str, Callable[[], AlgorithmRunner]] = {}
_CODE_OWNED_EVALUATOR_FACTORIES: dict[
    str, Callable[[Path, Path], QualificationEvaluator]
] = {}


def _select_bundle(
    *,
    bundle_id: str,
    bundle_version: str,
    channel: str,
) -> AlgorithmBundleSpec:
    loaded = load_algorithm_bundle_registry()
    if not loaded.bundles:
        raise _fail(
            "registry_empty",
            "the packaged bundle registry has no qualified bundle; register and review a real bundle first",
        )
    matches = [
        item
        for item in loaded.bundles
        if item.to_dict()["bundle_id"] == bundle_id
        and item.to_dict()["bundle_version"] == bundle_version
    ]
    if len(matches) != 1:
        raise _fail("bundle_not_found", "exact bundle ID/version is not in the packaged registry")
    bundle = matches[0]
    record = bundle.to_dict()
    pointer = loaded.lifecycle.channel_pointers.get((record["family_id"], channel))
    if (
        not loaded.is_lifecycle_eligible(family_id=record["family_id"], channel=channel)
        or pointer is None
        or pointer["bundle_spec_digest"] != bundle.spec_digest
    ):
        raise _fail("bundle_ineligible", "exact bundle is not the current enabled license-approved channel target")
    return bundle


def _preflight_bundle(bundle: AlgorithmBundleSpec, job: ImageJobSpec, *, channel: str) -> None:
    bundle_record = bundle.to_dict()
    job_record = job.to_dict()
    if bundle_record["execution_binding"]["status"] != "bound":
        raise _fail(
            "runner_unavailable",
            "bundle metadata is registered without a complete adaptive runner binding",
        )
    if channel not in job_record["allowed_maturities"]:
        raise _fail("maturity_not_allowed", "request does not allow the selected channel")
    if job_record["task"] not in bundle_record["tasks"]:
        raise _fail("task_unsupported", "bundle does not declare the requested task")
    if job_record["prompt_mode"] not in bundle_record["prompt_modes"]:
        raise _fail("prompt_mode_unsupported", "bundle does not declare the requested prompt mode")
    if job_record["network_policy"] != "deny" or bundle_record["execution_network_required"]:
        raise _fail("network_forbidden", "P0 qualification permits only declared network-free execution")
    if bundle_record["execution_trust_class"] != "code_owned_audited":
        raise _fail("runner_untrusted", "P0 qualification permits only audited code-owned runners")
    if bundle_record["runner_id"] not in CODE_OWNED_RUNNER_IDS:
        raise _fail("runner_untrusted", "bundle runner is not in the code-owned allowlist")
    runtime = bundle_record["runtime"]
    if job_record["compute_policy"] == "cpu_only" and runtime["provider_id"] not in CPU_PROVIDER_IDS:
        raise _fail("compute_policy_mismatch", "cpu_only rejects the bundle provider")
    if job_record["compute_policy"] == "accelerator_required" and runtime["provider_id"] not in ACCELERATOR_PROVIDER_IDS:
        raise _fail("compute_policy_mismatch", "accelerator_required rejects the bundle provider")
    if job_record.get("provider_allowlist") and runtime["provider_id"] not in job_record["provider_allowlist"]:
        raise _fail("provider_not_allowed", "bundle provider is outside the request allowlist")
    if job_record.get("precision_allowlist") and runtime["precision"] not in job_record["precision_allowlist"]:
        raise _fail("precision_not_allowed", "bundle precision is outside the request allowlist")
    spdx = set(job_record.get("spdx_allowlist", []))
    if spdx and any(item["license_expression"] not in spdx for item in bundle_record["artifacts"]):
        raise _fail("license_not_allowed", "one or more bundle artifacts are outside the request SPDX allowlist")
    if job_record["prompt_mode"] == "fixed_classes":
        build_fixed_class_mapping(bundle, job.prompt_phrases)


def _preflight_environment(
    bundle: AlgorithmBundleSpec,
    environment: EnvironmentProfile,
) -> None:
    runtime = bundle.to_dict()["runtime"]
    observed = environment.to_dict()
    matches = [
        item
        for item in observed["runtimes"]
        if item["runtime_id"] == runtime["runtime_id"]
    ]
    if len(matches) != 1 or matches[0]["probe_status"] != "present":
        raise _fail("runtime_unavailable", "exact bundle runtime was not observed as present")
    if (
        matches[0]["version"] != runtime["runtime_version"]
        or runtime["provider_id"] not in matches[0]["provider_ids"]
    ):
        raise _fail("runtime_mismatch", "observed runtime version/provider does not match the bundle")
    architecture = runtime["architecture"]
    if architecture != "any" and observed["os"].get("architecture") != architecture:
        raise _fail("architecture_mismatch", "observed architecture does not match the bundle")
    if runtime["accelerator_requirement"] == "required":
        accelerators = [
            item
            for item in observed["accelerators"]
            if item["probe_status"] == "present"
        ]
        if not accelerators:
            raise _fail("accelerator_unavailable", "bundle requires an observed accelerator")
        required_bytes = runtime["minimum_accelerator_memory_bytes"]
        known_memory = [
            item["memory"]["value_bytes"]
            for item in accelerators
            if item.get("memory", {}).get("probe_status") == "present"
        ]
        if not known_memory:
            raise _fail("accelerator_memory_unknown", "required accelerator memory is unknown")
        if max(known_memory) < required_bytes:
            raise _fail("accelerator_memory_insufficient", "observed accelerator memory is below the bundle minimum")


def _artifact_inventory(
    *,
    pinned: PinnedVerifiedArtifactSet,
    bundle: AlgorithmBundleSpec,
    verified_at: str,
) -> tuple[dict[str, Any], str]:
    by_id = {item["artifact_id"]: item for item in pinned.iter_local_observations()}
    observations = []
    for artifact in bundle.to_dict()["artifacts"]:
        observed = by_id[artifact["artifact_id"]]
        observations.append(
            {
                "artifact_id": artifact["artifact_id"],
                "role": artifact["role"],
                "order": artifact["order"],
                "expected_size_bytes": artifact["expected_size_bytes"],
                "expected_sha256": artifact["sha256"],
                "presence_status": "present",
                "path_type_status": "regular_file",
                "read_status": "readable",
                "observed_size_bytes": observed["size_bytes"],
                "observed_sha256": observed["sha256"],
                "verified_at": verified_at,
                "error_status": "none",
            }
        )
    value: dict[str, Any] = {
        "schema_version": 1,
        "inventory_id": f"inventory-{bundle.spec_digest[:16]}",
        "bundle_spec_digest": bundle.spec_digest,
        "artifact_set_digest": bundle.artifact_set_digest,
        "observations": observations,
        "artifact_state_fingerprint": "0" * 64,
        "inventory_digest": "0" * 64,
    }
    value["artifact_state_fingerprint"] = compute_artifact_state_fingerprint(value)
    value["inventory_digest"] = canonical_sha256_v1(value, own_digest_field="inventory_digest")
    validated = validate_local_artifact_inventory(value, bundle)
    return validated.to_dict(), validated.artifact_state_fingerprint


def _pipeline_identities(bundle: AlgorithmBundleSpec) -> dict[str, dict[str, str]]:
    record = bundle.to_dict()

    def identity(name: str) -> dict[str, str]:
        value = record[name]
        return {"id": value["id"], "version": value["version"], "source_digest": value["digest"]}

    return {
        "decoder": identity("decoder"),
        "model_input": {
            "id": "bundle_model_input_shapes",
            "version": "1",
            "source_digest": canonical_sha256_v1(record["model_input_shapes"]),
        },
        "preprocess": identity("preprocess"),
        "postprocess": identity("postprocess"),
    }


def _failed_repeat(index: int, code: str, environment: EnvironmentProfile) -> dict[str, Any]:
    return {
        "repeat_index": index,
        "status": "failed",
        "failure_code": code,
        "sample_count": None,
        "duration_ns": None,
        "p50_latency_ms": None,
        "p95_latency_ms": None,
        "p99_latency_ms": None,
        "throughput_processed_count": None,
        "throughput_duration_ns": None,
        "input_coverage_counts": None,
        "runner_tree_peak_rss": _unknown_rss(),
        "accelerator_process_tree_peak": _accelerator_memory(environment),
    }


def _sustained_failed(code: str, environment: EnvironmentProfile) -> dict[str, Any]:
    return {
        "status": "failed",
        "failure_code": code,
        "schedule_reset_index": 0,
        "duration_ns": None,
        "processed_count": None,
        "sample_count": None,
        "max_sustained_samples": MAX_SUSTAINED_SAMPLES,
        "sample_storage_bytes": None,
        "aggregation_method": "exact_nearest_rank_all_samples",
        "p95_latency_ms": None,
        "p99_latency_ms": None,
        "throughput_processed_count": None,
        "throughput_duration_ns": None,
        "runner_tree_peak_rss": _unknown_rss(),
        "accelerator_process_tree_peak": _accelerator_memory(environment),
        "queue_status": "not_applicable",
        "drop_status": "not_applicable",
        "power_observation": {"status": "unknown", "value": None},
        "thermal_observation": {"status": "unknown", "value": None},
        "warmup_excluded": True,
        "cold_start_excluded": True,
        "repeat_samples_excluded": True,
    }


def _base_report(
    *,
    status: str,
    bundle: AlgorithmBundleSpec,
    job: ImageJobSpec,
    environment: EnvironmentProfile,
    workload_fingerprint: str,
    artifact_state_fingerprint: str,
    started: datetime,
    completed: datetime,
    repeats: list[dict[str, Any]],
    conservative: dict[str, Any] | None,
    cold_start: dict[str, Any],
    warmup: dict[str, Any],
    sustained: dict[str, Any],
    quality: dict[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    bundle_record = bundle.to_dict()
    runtime = bundle_record["runtime"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_id": f"qualification-{completed.strftime('%Y%m%dT%H%M%SZ')}-{bundle.spec_digest[:12]}",
        "report_digest": "0" * 64,
        "collector": _COLLECTOR_IDENTITY,
        "issuer": _ISSUER_IDENTITY,
        "status": status,
        "task": job.to_dict()["task"],
        "execution_mode": job.to_dict()["execution_mode"],
        "bundle_spec_digest": bundle.spec_digest,
        "artifact_set_digest": bundle.artifact_set_digest,
        "artifact_state_fingerprint": artifact_state_fingerprint,
        "environment_fingerprint": environment.environment_fingerprint,
        "qualification_workload_fingerprint": workload_fingerprint,
        "protocol_fingerprint": QUALIFICATION_PROTOCOL_FINGERPRINT,
        "latency_interval": {
            "interval_id": LATENCY_INTERVAL_ID,
            "handoff_id": HANDOFF_ID,
            "handoff_version": HANDOFF_VERSION,
            "included_phases": list(LATENCY_PHASES),
            "publication_boundary": "managed_output_transaction_after_interval",
        },
        "started_at": _utc_text(started),
        "completed_at": _utc_text(completed),
        "valid_until": _utc_text(completed + timedelta(days=90)),
        "repeats": repeats,
        "conservative_aggregates": conservative,
        "cold_start": cold_start,
        "warmup": warmup,
        "lifetime_memory": {
            "interval_scope": "fresh_runner_creation_through_close",
            "runner_tree_peak_rss": _unknown_rss(),
            "accelerator_process_tree_peak": _accelerator_memory(environment),
        },
        "sustained_section": sustained,
        "quality": quality,
        "resolved_pipeline": _pipeline_identities(bundle),
        "source_runtime_provenance": {
            "model_source_id": bundle_record["model_source_id"],
            "model_revision": bundle_record["model_revision"],
            "runtime_id": runtime["runtime_id"],
            "runtime_version": runtime["runtime_version"],
            "provider_id": runtime["provider_id"],
            "provider_version": runtime["provider_version"],
        },
        "limitations": [
            "Measurements apply only to this immutable bundle, environment, workload, and protocol.",
            "OS filesystem cache state was uncontrolled.",
            "P0 enforces a declared network-free runner but does not claim OS-level network isolation.",
            "Complete runner-tree or accelerator peak memory is unknown unless a full-coverage collector is available.",
            "This report is unactivated and does not establish Stable support or human adoption.",
        ],
        "failures": failures,
    }
    report["report_digest"] = canonical_sha256_v1(report, own_digest_field="report_digest")
    return report


def _predict_handoff(
    *,
    session: _RunnerSession,
    inputs: PinnedDecodedInputSet,
    input_index: int,
    job: ImageJobSpec,
    bundle: AlgorithmBundleSpec,
    monotonic_ns: Callable[[], int],
) -> tuple[int, _Handoff]:
    item = inputs[input_index]
    observation = inputs.inventory.inputs[input_index]
    started = monotonic_ns()
    item.read_source_bytes()
    raw = session.predict(input_index, PREDICT_TIMEOUT_SECONDS)
    handoff = _strict_handoff(
        raw_results=raw,
        job=job,
        bundle=bundle,
        input_item=item,
        input_width=observation.width,
        input_height=observation.height,
    )
    ended = monotonic_ns()
    duration = ended - started
    if duration < 0 or duration > 2**64 - 1:
        raise _fail("clock_invalid", "monotonic latency interval is invalid")
    return duration, handoff


def _repeat_summary(
    *,
    repeat_index: int,
    session: _RunnerSession,
    inputs: PinnedDecodedInputSet,
    job: ImageJobSpec,
    bundle: AlgorithmBundleSpec,
    environment: EnvironmentProfile,
    monotonic_ns: Callable[[], int],
) -> dict[str, Any]:
    schedule = qualification_input_schedule(len(inputs), REPEAT_ITERATIONS)
    samples: list[int] = []
    started = monotonic_ns()
    for input_index in schedule:
        prediction = _predict_handoff(
            session=session,
            inputs=inputs,
            input_index=input_index,
            job=job,
            bundle=bundle,
            monotonic_ns=monotonic_ns,
        )
        samples.append(prediction[0])
    ended = monotonic_ns()
    total = ended - started
    if total <= 0 or total > 2**64 - 1:
        raise _fail("clock_invalid", "repeat duration is invalid")
    return {
        "repeat_index": repeat_index,
        "status": "completed",
        "failure_code": None,
        "sample_count": REPEAT_ITERATIONS,
        "duration_ns": total,
        "p50_latency_ms": nanoseconds_to_milliseconds(nearest_rank_nanoseconds(samples, 50)),
        "p95_latency_ms": nanoseconds_to_milliseconds(nearest_rank_nanoseconds(samples, 95)),
        "p99_latency_ms": nanoseconds_to_milliseconds(nearest_rank_nanoseconds(samples, 99)),
        "throughput_processed_count": REPEAT_ITERATIONS,
        "throughput_duration_ns": total,
        "input_coverage_counts": _coverage(schedule, len(inputs)),
        "runner_tree_peak_rss": _unknown_rss(),
        "accelerator_process_tree_peak": _accelerator_memory(environment),
    }


def _conservative(repeats: Sequence[dict[str, Any]], environment: EnvironmentProfile) -> dict[str, Any]:
    worst = min(
        range(len(repeats)),
        key=lambda index: Decimal(repeats[index]["throughput_processed_count"])
        / Decimal(repeats[index]["throughput_duration_ns"]),
    )
    return {
        "repeat_throughput_source_index": worst + 1,
        "repeat_throughput_processed_count": repeats[worst]["throughput_processed_count"],
        "repeat_throughput_duration_ns": repeats[worst]["throughput_duration_ns"],
        "p50_latency_ms": max((item["p50_latency_ms"] for item in repeats), key=Decimal),
        "p95_latency_ms": max((item["p95_latency_ms"] for item in repeats), key=Decimal),
        "p99_latency_ms": max((item["p99_latency_ms"] for item in repeats), key=Decimal),
        "runner_tree_peak_rss": _unknown_rss(),
        "accelerator_process_tree_peak": _accelerator_memory(environment),
    }


def _sustained_summary(
    *,
    session: _RunnerSession,
    inputs: PinnedDecodedInputSet,
    job: ImageJobSpec,
    bundle: AlgorithmBundleSpec,
    environment: EnvironmentProfile,
    monotonic_ns: Callable[[], int],
) -> dict[str, Any]:
    samples = array("Q", [0]) * MAX_SUSTAINED_SAMPLES
    if samples.itemsize != 8:
        raise _fail("sustained_storage_unsupported", "uint64 samples require exactly eight bytes")
    started = monotonic_ns()
    count = 0
    while True:
        if count >= MAX_SUSTAINED_SAMPLES:
            raise _fail("sustained_sample_limit", "sample cap reached before ten-minute section completed")
        prediction = _predict_handoff(
            session=session,
            inputs=inputs,
            input_index=count % len(inputs),
            job=job,
            bundle=bundle,
            monotonic_ns=monotonic_ns,
        )
        samples[count] = prediction[0]
        count += 1
        elapsed = monotonic_ns() - started
        if elapsed >= MIN_SUSTAINED_DURATION_NS:
            break
    used = samples[:count]
    return {
        "status": "completed",
        "failure_code": None,
        "schedule_reset_index": 0,
        "duration_ns": elapsed,
        "processed_count": count,
        "sample_count": count,
        "max_sustained_samples": MAX_SUSTAINED_SAMPLES,
        "sample_storage_bytes": count * 8,
        "aggregation_method": "exact_nearest_rank_all_samples",
        "p95_latency_ms": nanoseconds_to_milliseconds(nearest_rank_nanoseconds(used, 95)),
        "p99_latency_ms": nanoseconds_to_milliseconds(nearest_rank_nanoseconds(used, 99)),
        "throughput_processed_count": count,
        "throughput_duration_ns": elapsed,
        "runner_tree_peak_rss": _unknown_rss(),
        "accelerator_process_tree_peak": _accelerator_memory(environment),
        "queue_status": "not_applicable",
        "drop_status": "not_applicable",
        "power_observation": {"status": "unknown", "value": None},
        "thermal_observation": {"status": "unknown", "value": None},
        "warmup_excluded": True,
        "cold_start_excluded": True,
        "repeat_samples_excluded": True,
    }


def _apply_gates(
    *,
    job: ImageJobSpec,
    repeats: Sequence[dict[str, Any]],
    conservative: dict[str, Any],
    cold_start: dict[str, Any],
    sustained: dict[str, Any],
    quality: dict[str, Any],
) -> list[str]:
    record = job.to_dict()
    failures: list[str] = []
    if record.get("max_cold_start_ms") is not None and Decimal(cold_start["cold_start_ms"]) > Decimal(record["max_cold_start_ms"]):
        failures.append("max_cold_start_exceeded")
    if record.get("max_p95_latency_ms") is not None and Decimal(conservative["p95_latency_ms"]) > Decimal(record["max_p95_latency_ms"]):
        failures.append("max_p95_latency_exceeded")
    if record.get("max_runner_tree_peak_rss_bytes") is not None:
        failures.append("runner_tree_memory_unknown")
    if record.get("max_accelerator_process_tree_peak_bytes") is not None:
        failures.append("accelerator_memory_unknown")
    if record["execution_mode"] == "batch" and record.get("min_repeat_throughput_fps") is not None:
        count = conservative["repeat_throughput_processed_count"]
        duration = conservative["repeat_throughput_duration_ns"]
        required = Decimal(record["min_repeat_throughput_fps"])
        if Decimal(count) * Decimal(1_000_000_000) < required * Decimal(duration):
            failures.append("min_repeat_throughput_not_met")
    if record["execution_mode"] == "soft_realtime" and record.get("min_sustained_fps") is not None:
        required = Decimal(record["min_sustained_fps"])
        if Decimal(sustained["processed_count"]) * Decimal(1_000_000_000) < required * Decimal(sustained["duration_ns"]):
            failures.append("min_sustained_fps_not_met")
    requirement = record.get("quality_requirement")
    if requirement is not None:
        if quality["status"] != "known":
            failures.append("quality_unknown")
        elif quality["direction"] == "higher_is_better" and Decimal(quality["measured_value"]) < Decimal(requirement["threshold"]):
            failures.append("quality_threshold_not_met")
        elif quality["direction"] == "lower_is_better" and Decimal(quality["measured_value"]) > Decimal(requirement["threshold"]):
            failures.append("quality_threshold_not_met")
    return failures


def _evaluate_quality(
    *,
    evaluator: QualificationEvaluator,
    session: _RunnerSession,
    inputs: PinnedDecodedInputSet,
    job: ImageJobSpec,
    bundle: AlgorithmBundleSpec,
    monotonic_ns: Callable[[], int],
) -> dict[str, Any]:
    requirement = job.to_dict().get("quality_requirement")
    if requirement is None:
        raise _fail("evaluator_unexpected", "quality evaluator requires a quality requirement")
    identity_fields = (
        "metric_id",
        "direction",
        "evaluation_dataset_id",
        "evaluation_dataset_sha256",
        "evaluation_protocol_sha256",
        "evaluation_vocabulary_id",
    )
    for field in identity_fields:
        if getattr(evaluator, field) != requirement[field]:
            reason = "vocabulary_mismatch" if field == "evaluation_vocabulary_id" else "no_matching_evaluator"
            return {"status": "unknown", "reason": reason}
    predictions: list[bytes] = []
    for input_index in range(len(inputs)):
        prediction = _predict_handoff(
            session=session,
            inputs=inputs,
            input_index=input_index,
            job=job,
            bundle=bundle,
            monotonic_ns=monotonic_ns,
        )
        predictions.append(prediction[1].metadata)
    measured = canonical_decimal_v1(
        evaluator.evaluate(
            predictions=tuple(predictions),
            job=job,
            bundle=bundle,
        ),
        field="quality.measured_value",
    )
    return {
        "status": "known",
        "metric_id": requirement["metric_id"],
        "direction": requirement["direction"],
        "measured_value": measured,
        "threshold_context": requirement["threshold"],
        "evaluation_dataset_id": requirement["evaluation_dataset_id"],
        "evaluation_dataset_sha256": requirement["evaluation_dataset_sha256"],
        "evaluation_protocol_sha256": requirement["evaluation_protocol_sha256"],
        "evaluation_vocabulary_id": requirement["evaluation_vocabulary_id"],
        "predictions_source": "same_qualification_run",
    }


def _collect_report(
    *,
    session: _RunnerSession,
    bundle: AlgorithmBundleSpec,
    job: ImageJobSpec,
    environment: EnvironmentProfile,
    inputs: PinnedDecodedInputSet,
    workload_fingerprint: str,
    artifact_state_fingerprint: str,
    started: datetime,
    smoke: bool,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    utc_now: Callable[[], datetime] = _utc_now,
    cold_started_ns: int | None = None,
    evaluator: QualificationEvaluator | None = None,
) -> QualificationReport:
    job_record = job.to_dict()
    failure: QualificationError | None = None
    cold: dict[str, Any] = {
        "status": "failed",
        "cold_start_ms": None,
        "failure_code": "not_started",
        "fresh_runner": True,
        "os_cache_state": "uncontrolled",
        "interval_id": LATENCY_INTERVAL_ID,
    }
    warmup: dict[str, Any] = {"status": "failed", "iteration_count": None, "failure_code": "not_started"}
    repeats = [_failed_repeat(index, "not_started", environment) for index in range(1, 4)]
    sustained = (
        {"status": "not_required", "reason": "batch_profile"}
        if job_record["execution_mode"] == "batch"
        else _sustained_failed("not_started", environment)
    )
    quality = (
        {"status": "not_required", "reason": "request_has_no_quality_requirement"}
        if job_record.get("quality_requirement") is None
        else {"status": "unknown", "reason": "no_matching_evaluator"}
    )
    conservative: dict[str, Any] | None = None
    cold_started = monotonic_ns() if cold_started_ns is None else cold_started_ns
    try:
        bundle_record = bundle.to_dict()
        if (
            session.runner_id != bundle_record["runner_id"]
            or session.runner_version != bundle_record["runner_version"]
        ):
            raise _fail(
                "runner_identity_mismatch",
                "runner ID/version does not match the immutable bundle",
            )
        probe = session.probe(PROBE_TIMEOUT_SECONDS)
        if probe.status != "supported":
            raise _fail("runner_probe_failed", probe.reason_code or probe.status)
        session.load(LOAD_TIMEOUT_SECONDS)
        _predict_handoff(
            session=session,
            inputs=inputs,
            input_index=0,
            job=job,
            bundle=bundle,
            monotonic_ns=monotonic_ns,
        )
        cold_elapsed = monotonic_ns() - cold_started
        cold = {
            "status": "known",
            "cold_start_ms": nanoseconds_to_milliseconds(cold_elapsed),
            "failure_code": None,
            "fresh_runner": True,
            "os_cache_state": "uncontrolled",
            "interval_id": LATENCY_INTERVAL_ID,
        }
        warmup_count = 1 if smoke else WARMUP_ITERATIONS
        for index in range(warmup_count):
            session.warmup(index % len(inputs), PREDICT_TIMEOUT_SECONDS)
        if smoke:
            warmup = {"status": "failed", "iteration_count": None, "failure_code": "smoke_mode"}
            for index in range(min(3, len(inputs))):
                _predict_handoff(
                    session=session,
                    inputs=inputs,
                    input_index=index,
                    job=job,
                    bundle=bundle,
                    monotonic_ns=monotonic_ns,
                )
            repeats = [_failed_repeat(index, "smoke_mode", environment) for index in range(1, 4)]
            if job_record["execution_mode"] == "soft_realtime":
                sustained = _sustained_failed("smoke_mode", environment)
        else:
            warmup = {"status": "completed", "iteration_count": WARMUP_ITERATIONS, "failure_code": None}
            repeats = [
                _repeat_summary(
                    repeat_index=index,
                    session=session,
                    inputs=inputs,
                    job=job,
                    bundle=bundle,
                    environment=environment,
                    monotonic_ns=monotonic_ns,
                )
                for index in range(1, REPEAT_COUNT + 1)
            ]
            conservative = _conservative(repeats, environment)
            if job_record["execution_mode"] == "soft_realtime":
                sustained = _sustained_summary(
                    session=session,
                    inputs=inputs,
                    job=job,
                    bundle=bundle,
                    environment=environment,
                    monotonic_ns=monotonic_ns,
                )
            if evaluator is not None:
                quality = _evaluate_quality(
                    evaluator=evaluator,
                    session=session,
                    inputs=inputs,
                    job=job,
                    bundle=bundle,
                    monotonic_ns=monotonic_ns,
                )
    except QualificationError as exc:
        failure = exc
    except (MemoryError, OSError, ValueError) as exc:
        failure = _fail("qualification_failed", str(exc)[:400])
    finally:
        try:
            session.close(CLOSE_TIMEOUT_SECONDS)
        except QualificationError as exc:
            failure = failure or exc

    completed = utc_now().astimezone(timezone.utc).replace(microsecond=0)
    if failure is not None:
        code = failure.code
        if cold["status"] != "known":
            cold["failure_code"] = code
        if warmup["status"] != "completed":
            warmup["failure_code"] = code
        repeats = [_failed_repeat(index, code, environment) for index in range(1, 4)]
        conservative = None
        if job_record["execution_mode"] == "soft_realtime":
            sustained = _sustained_failed(code, environment)
        status = "failed"
        failures = [code]
    elif smoke:
        status = "smoke"
        failures = []
    else:
        assert conservative is not None
        failures = _apply_gates(
            job=job,
            repeats=repeats,
            conservative=conservative,
            cold_start=cold,
            sustained=sustained,
            quality=quality,
        )
        status = "qualified" if not failures else "hold"
    report = _base_report(
        status=status,
        bundle=bundle,
        job=job,
        environment=environment,
        workload_fingerprint=workload_fingerprint,
        artifact_state_fingerprint=artifact_state_fingerprint,
        started=started,
        completed=completed,
        repeats=repeats,
        conservative=conservative,
        cold_start=cold,
        warmup=warmup,
        sustained=sustained,
        quality=quality,
        failures=failures,
    )
    return validate_qualification_report(report, as_of=completed)


def _workspace_output_destination(output_dir: Path, workspace: Path) -> str:
    workspace_lexical = Path(os.path.abspath(workspace))
    output = Path(output_dir)
    if not output.is_absolute():
        output = workspace_lexical / output
    lexical = Path(os.path.abspath(output))
    try:
        relative = lexical.relative_to(workspace_lexical)
    except ValueError as exc:
        raise _fail("output_outside_workspace", "output directory must stay inside workspace") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise _fail("output_invalid", "output directory must be a non-root workspace path")
    return relative.as_posix()


def qualify_image_pipeline(
    *,
    job: ImageJobSpec | Mapping[str, Any],
    input_path: str | Path,
    bundle_id: str,
    bundle_version: str,
    output_dir: str | Path,
    workspace_root: str | Path,
    artifact_root: str | Path | None = None,
    channel: str = "Experimental",
    qualification_timeout_seconds: int = 3_600,
    smoke: bool = False,
    force: bool = False,
    ground_truth_path: str | Path | None = None,
    evaluator_id: str | None = None,
) -> QualificationReport:
    """Measure one exact trusted bundle and atomically publish its report.

    The runner is selected only from the private repository-owned factory map.
    Callers cannot supply an import path, executable, arbitrary runner options,
    or performance values.
    """

    if (
        isinstance(qualification_timeout_seconds, bool)
        or not isinstance(qualification_timeout_seconds, int)
        or not QUALIFICATION_TIMEOUT_MIN_SECONDS
        <= qualification_timeout_seconds
        <= QUALIFICATION_TIMEOUT_MAX_SECONDS
    ):
        raise _fail("timeout_invalid", "qualification timeout must be in 60..14400 seconds")
    normalized_job = job if isinstance(job, ImageJobSpec) else validate_image_job_spec(job)
    if normalized_job.to_dict()["execution_mode"] == "soft_realtime" and qualification_timeout_seconds < SOFT_REALTIME_MIN_TIMEOUT_SECONDS:
        raise _fail("timeout_invalid", "soft_realtime requires at least 1260 seconds for setup and soak")
    if channel not in {"Experimental", "Stable"}:
        raise _fail("channel_invalid", "channel must be Experimental or Stable")
    if (ground_truth_path is None) != (evaluator_id is None):
        raise _fail("evaluator_invalid", "ground truth and evaluator ID must be supplied together")
    if evaluator_id is not None and normalized_job.to_dict().get("quality_requirement") is None:
        raise _fail("evaluator_invalid", "an evaluator requires a preregistered quality requirement")
    workspace = Path(workspace_root).resolve(strict=True)
    if not workspace.is_dir():
        raise _fail("workspace_invalid", "workspace root must be a directory")
    bundle = _select_bundle(bundle_id=bundle_id, bundle_version=bundle_version, channel=channel)
    _preflight_bundle(bundle, normalized_job, channel=channel)
    runner_id = bundle.to_dict()["runner_id"]
    factory = _CODE_OWNED_RUNNER_FACTORIES.get(runner_id)
    if factory is None:
        raise _fail(
            "runner_unavailable",
            f"registered runner {runner_id!r} has no audited code-owned adapter in this build",
        )
    evaluator: QualificationEvaluator | None = None
    if evaluator_id is not None:
        evaluator_factory = _CODE_OWNED_EVALUATOR_FACTORIES.get(evaluator_id)
        if evaluator_factory is None:
            raise _fail(
                "evaluator_unavailable",
                f"evaluator {evaluator_id!r} is not registered as code-owned in this build",
            )
        evaluator = evaluator_factory(Path(ground_truth_path), workspace)
        if evaluator.evaluator_id != evaluator_id:
            raise _fail(
                "evaluator_invalid",
                "evaluator factory identity does not match the requested evaluator ID",
            )

    started = _utc_now()
    environment = build_environment_profile(collected_at=_utc_text(started))
    _preflight_environment(bundle, environment)
    destination = _workspace_output_destination(Path(output_dir), workspace)
    with pin_decoded_inputs(
        input_path,
        input_mode=normalized_job.to_dict()["input_mode"],
        workspace_root=workspace,
        max_images=normalized_job.to_dict()["max_images"],
    ) as inputs:
        workload = build_qualification_workload_profile(normalized_job, inputs.inventory)
        with ArtifactResolver(
            workspace=workspace,
            artifact_root=None if artifact_root is None else Path(artifact_root),
        ) as resolver:
            with resolver.pin(bundle) as artifacts:
                _inventory_record, artifact_state = _artifact_inventory(
                    pinned=artifacts,
                    bundle=bundle,
                    verified_at=_utc_text(started),
                )
                deadline = time.monotonic_ns() + qualification_timeout_seconds * 1_000_000_000
                cold_started_ns = time.monotonic_ns()
                session = _ForkedRunnerSession(
                    factory=factory,
                    bundle=bundle,
                    environment=environment,
                    artifacts=artifacts,
                    inputs=inputs,
                    labels=normalized_job.prompt_phrases,
                    outer_deadline_ns=deadline,
                )
                report = _collect_report(
                    session=session,
                    bundle=bundle,
                    job=normalized_job,
                    environment=environment,
                    inputs=inputs,
                    workload_fingerprint=workload.workload_fingerprint,
                    artifact_state_fingerprint=artifact_state,
                    started=started,
                    smoke=bool(smoke),
                    cold_started_ns=cold_started_ns,
                    evaluator=evaluator,
                )
                if report.to_dict()["status"] != "failed":
                    for input_index in range(len(inputs)):
                        inputs[input_index].read_source_bytes()
                    tuple(artifacts.iter_local_observations())

    report_bytes = canonical_json_v1(report.to_dict())
    with ManagedOutputTransaction(
        root=workspace,
        destination=destination,
        declared_paths=("qualification_report.json",),
        limits=ManagedOutputLimits(
            max_files=HANDOFF_MAX_OUTPUT_FILES,
            max_file_bytes=HANDOFF_MAX_OUTPUT_BYTES,
            max_total_bytes=HANDOFF_MAX_OUTPUT_BYTES,
        ),
        force=bool(force),
    ) as transaction:
        transaction.write_bytes("qualification_report.json", report_bytes)
        transaction.commit()
    return report
