"""Pinned local execution for a previously selected adaptive image pipeline."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping

from yolozu.predictions.predictions import validate_predictions_payload

from .artifact_resolver import ArtifactResolver, PinnedVerifiedArtifactSet
from .bundle_registry import AlgorithmRunner, load_algorithm_bundle_registry
from .bundles import CODE_OWNED_RUNNER_IDS, AlgorithmBundleSpec
from .canonical import canonical_json_v1
from .contracts import (
    HANDOFF_MAX_MASK_ARTIFACTS,
    HANDOFF_MAX_OUTPUT_BYTES,
    HANDOFF_MAX_OUTPUT_FILES,
    ImageJobSpec,
    build_qualification_workload_profile,
    validate_image_job_spec,
)
from .environment import build_environment_profile
from .inventory import PinnedDecodedInputSet, pin_decoded_inputs
from .isolation import (
    IsolatedRunnerCapability,
    IsolatedRunnerService,
    _CODE_OWNED_ISOLATED_SERVICES,
    _RunnerSession,
)
from .managed_output import (
    ManagedOutputError,
    ManagedOutputLimits,
    ManagedOutputTransaction,
    validate_managed_output_destination,
)
from .qualification import (
    CLOSE_TIMEOUT_SECONDS,
    LOAD_TIMEOUT_SECONDS,
    PREDICT_TIMEOUT_SECONDS,
    PROBE_TIMEOUT_SECONDS,
    QualificationError,
    _CODE_OWNED_RUNNER_FACTORIES,
    _ForkedRunnerSession,
    _strict_handoff,
)
from .recommendation import (
    RecommendationError,
    _artifact_inventory,
    _confined_directory,
    _load_support_profiles,
    _workspace_root,
    recommend_image_pipeline,
)
from .selection import SelectionDecision, validate_selection_decision

__all__ = [
    "IsolatedRunnerCapability",
    "IsolatedRunnerService",
    "ProcessingError",
    "process_images",
]


_OUTPUT_LIMITS = ManagedOutputLimits(
    max_files=HANDOFF_MAX_OUTPUT_FILES,
    max_file_bytes=HANDOFF_MAX_OUTPUT_BYTES,
    max_total_bytes=HANDOFF_MAX_OUTPUT_BYTES,
)
_STABLE_DECISION_FIELDS = (
    "local_job_digest",
    "local_input_digest",
    "artifact_resolver_state_digest",
    "environment_fingerprint",
    "qualification_workload_fingerprint",
    "protocol_fingerprint",
    "advertised_gates_digest",
    "registry_id",
    "registry_digest",
    "registry_trust_domain",
    "lifecycle_projection_digest",
    "lifecycle_trust_domain",
    "ranking_policy",
    "prompt_mode",
    "registry_bundle_count",
    "selected_bundle",
    "selected_evidence",
    "selected_artifact_state_fingerprint",
    "selected_class_mapping",
    "support_scope",
)


class ProcessingError(ValueError):
    """A safe public processing failure with one stable error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message


@dataclass(frozen=True)
class _ExecutionRoute:
    kind: str
    host_factory: Callable[[], AlgorithmRunner] | None = None
    isolated_service: IsolatedRunnerService | None = None


def _fail(code: str, message: str) -> ProcessingError:
    return ProcessingError(code, message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _output_destination(value: str | Path, workspace: Path) -> str:
    workspace_lexical = Path(os.path.abspath(workspace))
    output = Path(value)
    if not output.is_absolute():
        output = workspace_lexical / output
    lexical = Path(os.path.abspath(output))
    try:
        relative = lexical.relative_to(workspace_lexical)
    except ValueError as exc:
        raise _fail(
            "output_outside_workspace",
            "output_dir must stay inside the workspace",
        ) from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise _fail(
            "output_invalid",
            "output_dir must be a non-root workspace path",
        )
    return relative.as_posix()


def _same_stable_decision(
    pinned: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    return all(pinned.get(field) == current.get(field) for field in _STABLE_DECISION_FIELDS)


def _resolve_execution_route(
    bundle: AlgorithmBundleSpec,
    *,
    host_factories: Mapping[str, Callable[[], AlgorithmRunner]] | None = None,
    isolated_services: Mapping[str, IsolatedRunnerService] | None = None,
) -> _ExecutionRoute:
    """Resolve only code-owned routes; tests may call this below the public gate."""

    record = bundle.to_dict()
    if record["execution_network_required"]:
        raise _fail("network_forbidden", "the selected runner declares network use")
    runner_id = record["runner_id"]
    if runner_id not in CODE_OWNED_RUNNER_IDS:
        raise _fail("runner_untrusted", "the selected runner is not code-owned")
    trust_class = record["execution_trust_class"]
    if trust_class == "code_owned_audited":
        available_hosts = (
            _CODE_OWNED_RUNNER_FACTORIES
            if host_factories is None
            else host_factories
        )
        factory = available_hosts.get(runner_id)
        if factory is None:
            raise _fail(
                "runner_unavailable",
                "the selected audited host runner is not installed in this build",
            )
        return _ExecutionRoute("code_owned_audited", host_factory=factory)
    if trust_class != "third_party_isolated":
        raise _fail("runner_untrusted", "the selected execution trust class is invalid")
    available_isolation = (
        _CODE_OWNED_ISOLATED_SERVICES
        if isolated_services is None
        else isolated_services
    )
    service = available_isolation.get(runner_id)
    if service is None:
        raise _fail(
            "isolation_unsupported",
            "the selected bundle requires an unavailable isolated runner",
        )
    capability = service.capability
    expected_policy = record["execution_isolation_policy_digest"]
    if capability.runner_id != runner_id or capability.status != "available":
        raise _fail(
            "isolation_required",
            "the isolated runner capability is not currently available",
        )
    if capability.image_present is not True:
        raise _fail(
            "isolation_image_missing",
            "the isolated runner image is not present",
        )
    if capability.policy_digest != expected_policy:
        raise _fail(
            "isolation_policy_mismatch",
            "the isolated runner policy does not match the selected bundle",
        )
    return _ExecutionRoute("third_party_isolated", isolated_service=service)


def _decimal(value: Any, *, field: str) -> Decimal:
    if not isinstance(value, str):
        raise _fail("invalid_runner_output", f"{field} is not canonical decimal text")
    try:
        result = Decimal(value)
    except Exception as exc:
        raise _fail("invalid_runner_output", f"{field} is invalid") from exc
    if not result.is_finite():
        raise _fail("invalid_runner_output", f"{field} is not finite")
    return result


def _output_from_handoffs(
    *,
    handoffs: list[tuple[bytes, tuple[bytes, ...]]],
    job: ImageJobSpec,
    decision: SelectionDecision,
    bundle: AlgorithmBundleSpec,
    runner_id: str,
    runner_version: str,
    completed_at: str,
) -> tuple[bytes, bytes, dict[str, bytes]]:
    entries: list[dict[str, Any]] = []
    masks: dict[str, bytes] = {}
    mask_count = 0
    for input_index, (metadata, mask_blobs) in enumerate(handoffs):
        try:
            handoff = json.loads(metadata.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _fail("invalid_runner_output", "validated handoff JSON is invalid") from exc
        if handoff.get("input_index") != input_index:
            raise _fail("invalid_runner_output", "handoff input order changed")
        detections: list[dict[str, Any]] = []
        referenced_masks: set[int] = set()
        for result in handoff.get("results", []):
            bbox = result["bbox"]
            x1, y1, x2, y2 = (
                _decimal(item, field="result.bbox") for item in bbox
            )
            if not (
                Decimal(0) <= x1 < x2 <= Decimal(1)
                and Decimal(0) <= y1 < y2 <= Decimal(1)
            ):
                raise _fail(
                    "invalid_runner_output",
                    "runner bbox must be normalized xyxy coordinates",
                )
            score = _decimal(result["score"], field="result.score")
            detection: dict[str, Any] = {
                "class_id": result["request_index"],
                "score": float(score),
                "bbox": {
                    "cx": float((x1 + x2) / Decimal(2)),
                    "cy": float((y1 + y2) / Decimal(2)),
                    "w": float(x2 - x1),
                    "h": float(y2 - y1),
                },
                "meta": {"requested_label": result["label"]},
            }
            if "mask_index" in result:
                local_index = result["mask_index"]
                if (
                    isinstance(local_index, bool)
                    or not isinstance(local_index, int)
                    or not 0 <= local_index < len(mask_blobs)
                    or local_index in referenced_masks
                ):
                    raise _fail("invalid_runner_output", "mask reference is invalid")
                referenced_masks.add(local_index)
                if mask_count >= HANDOFF_MAX_MASK_ARTIFACTS:
                    raise _fail("output_limit_exceeded", "mask count exceeds 1000")
                path = f"artifacts/masks/{input_index:06d}-{local_index:04d}.png"
                masks[path] = mask_blobs[local_index]
                mask_count += 1
                detection["mask"] = path
            detections.append(detection)
        if len(referenced_masks) != len(mask_blobs):
            raise _fail("invalid_runner_output", "handoff contains an unreferenced mask")
        entries.append(
            {
                "schema_version": 2,
                "image": f"inputs/{input_index:06d}",
                "detections": detections,
                "task": job.to_dict()["task"],
            }
        )
    try:
        validate_predictions_payload(entries, strict=True)
    except ValueError as exc:
        raise _fail(
            "invalid_runner_output",
            "predictions do not satisfy the strict predictions interface contract",
        ) from exc
    predictions = json.dumps(
        entries,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    decision_record = decision.to_dict()
    selected = decision_record["selected_bundle"]
    provenance = canonical_json_v1(
        {
            "schema_version": 1,
            "kind": "adaptive_pinned_image_processing",
            "maturity": "experimental",
            "completed_at": completed_at,
            "selection_decision_digest": decision.decision_digest,
            "local_job_digest": decision_record["local_job_digest"],
            "local_input_digest": decision_record["local_input_digest"],
            "bundle": {
                "family_id": selected["family_id"],
                "bundle_id": selected["bundle_id"],
                "bundle_version": selected["bundle_version"],
                "spec_digest": selected["spec_digest"],
                "artifact_set_digest": selected["artifact_set_digest"],
                "effective_channel": selected["effective_channel"],
            },
            "runner": {"id": runner_id, "version": runner_version},
            "network_used": False,
            "input_count": len(entries),
            "result_count": sum(len(entry["detections"]) for entry in entries),
            "mask_count": len(masks),
            "limitations": [
                "Execution is Experimental and applies only to the pinned decision and local state.",
                "Audited host execution does not claim OS-enforced isolation.",
            ],
        }
    )
    if len(masks) + 3 > HANDOFF_MAX_OUTPUT_FILES:
        raise _fail("output_limit_exceeded", "managed output file count exceeds 1003")
    if len(predictions) + len(provenance) + sum(map(len, masks.values())) > HANDOFF_MAX_OUTPUT_BYTES:
        raise _fail("output_limit_exceeded", "managed output exceeds 4 GiB")
    return predictions, provenance, masks


def _execute_pinned_pipeline(
    *,
    job: ImageJobSpec,
    decision: SelectionDecision,
    bundle: AlgorithmBundleSpec,
    inputs: PinnedDecodedInputSet,
    artifacts: PinnedVerifiedArtifactSet,
    workspace: Path,
    destination: str,
    dry_run: bool,
    force: bool,
    session_factory: Callable[[], _RunnerSession],
    outer_deadline_ns: int,
) -> dict[str, Any]:
    """Execute below the public routing gate; tests inject only at this boundary."""

    for input_index in range(len(inputs)):
        inputs[input_index].read_source_bytes()
    tuple(artifacts.iter_local_observations())
    if dry_run:
        return {
            "schema_version": 1,
            "ok": True,
            "tool": "process_images",
            "summary": "pinned execution preflight passed without model execution",
            "exit_code": 0,
            "maturity": "experimental",
            "availability": "mcp_live",
            "dry_run": True,
            "executed": False,
            "writes_performed": False,
            "network_used": False,
            "selection_decision_digest": decision.decision_digest,
        }

    session: _RunnerSession | None = None
    close_error: Exception | None = None
    failure: Exception | None = None
    handoffs: list[tuple[bytes, tuple[bytes, ...]]] = []
    try:
        session = session_factory()
        bundle_record = bundle.to_dict()
        if (
            session.runner_id != bundle_record["runner_id"]
            or session.runner_version != bundle_record["runner_version"]
        ):
            raise _fail(
                "runner_identity_mismatch",
                "runner identity does not match the selected bundle",
            )
        probe = session.probe(PROBE_TIMEOUT_SECONDS)
        if probe.status != "supported":
            raise _fail("runner_probe_failed", "runner probe did not report support")
        session.load(LOAD_TIMEOUT_SECONDS)
        for input_index in range(len(inputs)):
            if time.monotonic_ns() >= outer_deadline_ns:
                raise _fail("runner_timeout", "the image job deadline expired")
            item = inputs[input_index]
            observation = inputs.inventory.inputs[input_index]
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
            handoffs.append((handoff.metadata, handoff.masks))
    except Exception as exc:
        failure = exc
    finally:
        if session is not None:
            try:
                session.close(CLOSE_TIMEOUT_SECONDS)
            except Exception as exc:
                close_error = exc
    if failure is not None:
        raise failure
    if close_error is not None:
        raise close_error
    assert session is not None

    for input_index in range(len(inputs)):
        inputs[input_index].read_source_bytes()
    tuple(artifacts.iter_local_observations())
    predictions, provenance, masks = _output_from_handoffs(
        handoffs=handoffs,
        job=job,
        decision=decision,
        bundle=bundle,
        runner_id=session.runner_id,
        runner_version=session.runner_version,
        completed_at=_utc_now(),
    )
    declared = ("predictions.json", "provenance.json", *sorted(masks))
    try:
        with ManagedOutputTransaction(
            root=workspace,
            destination=destination,
            declared_paths=declared,
            limits=_OUTPUT_LIMITS,
            force=force,
        ) as transaction:
            transaction.write_bytes("predictions.json", predictions)
            transaction.write_bytes("provenance.json", provenance)
            for path in sorted(masks):
                transaction.write_bytes(path, masks[path])
            capabilities = transaction.commit()
    except ManagedOutputError as exc:
        raise _fail(exc.code, "managed output publication failed safely") from exc
    return {
        "schema_version": 1,
        "ok": True,
        "tool": "process_images",
        "summary": "executed the pinned image pipeline and published managed output",
        "exit_code": 0,
        "maturity": "experimental",
        "availability": "mcp_live",
        "dry_run": False,
        "executed": True,
        "writes_performed": True,
        "network_used": False,
        "selection_decision_digest": decision.decision_digest,
        "output": {
            "predictions": "predictions.json",
            "provenance": "provenance.json",
            "checksums": "checksums.json",
            "mask_count": len(masks),
            "atomic_visibility": capabilities.same_filesystem_atomic_visibility,
            "power_loss_durability": capabilities.power_loss_durability,
        },
    }


def process_images(
    job_spec: Mapping[str, Any],
    selection_decision: Mapping[str, Any],
    input_path: str,
    output_dir: str,
    *,
    workspace_root: str | Path = ".",
    registry_root: str | None = None,
    evidence_root: str | None = None,
    artifact_root: str | None = None,
    dry_run: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Revalidate and optionally execute one complete selected decision locally."""

    if not isinstance(dry_run, bool) or not isinstance(force, bool):
        raise _fail("invalid_options", "dry_run and force must be booleans")
    started_ns = time.monotonic_ns()
    now = _utc_now()
    workspace = _workspace_root(workspace_root)
    destination = _output_destination(output_dir, workspace)
    try:
        validate_managed_output_destination(
            root=workspace,
            destination=destination,
            limits=_OUTPUT_LIMITS,
            force=force,
        )
    except ManagedOutputError as exc:
        raise _fail(exc.code, "output_dir failed read-only managed-output preflight") from exc
    try:
        job = validate_image_job_spec(job_spec)
        pinned_decision = validate_selection_decision(selection_decision, as_of=now)
    except (TypeError, ValueError) as exc:
        raise _fail(
            "invalid_execution_request",
            "job_spec or selection_decision failed its interface contract",
        ) from exc
    pinned_record = pinned_decision.to_dict()
    if pinned_record["status"] != "selected":
        raise _fail(
            "selection_required",
            "process_images requires a complete selected decision",
        )
    try:
        current_response = recommend_image_pipeline(
            job.to_dict(),
            input_path,
            workspace_root=workspace,
            registry_root=registry_root,
            evidence_root=evidence_root,
            artifact_root=artifact_root,
            decided_at=now,
        )
    except RecommendationError as exc:
        raise _fail(exc.code, exc.public_message) from exc
    current_record = current_response["decision"]
    if current_record["status"] != "selected" or not _same_stable_decision(
        pinned_record, current_record
    ):
        raise _fail(
            "selection_stale",
            "the selected decision no longer matches current local state",
        )
    expected_selected_resolver_state = current_response[
        "recommendation_metadata"
    ]["selected_artifact_resolver_state_digest"]
    if not isinstance(expected_selected_resolver_state, str):
        raise _fail(
            "selection_stale",
            "the selected artifact resolver state is unavailable",
        )

    support_profiles = _load_support_profiles()
    custom_registry = (
        None
        if registry_root is None
        else _confined_directory(
            registry_root,
            workspace=workspace,
            label="registry_root",
        )
    )
    try:
        registry = load_algorithm_bundle_registry(
            workspace_root=workspace if custom_registry is not None else None,
            custom_registry_root=custom_registry,
            support_profiles=support_profiles if custom_registry is None else None,
        )
        validate_selection_decision(
            selection_decision,
            expected_registry=registry.registry,
            as_of=now,
        )
    except (TypeError, ValueError) as exc:
        raise _fail("selection_stale", "the decision registry no longer matches") from exc
    selected = pinned_record["selected_bundle"]
    bundle = registry.by_spec_digest().get(selected["spec_digest"])
    if bundle is None:
        raise _fail("selection_stale", "the selected bundle is no longer registered")
    route = _resolve_execution_route(bundle)

    try:
        environment = build_environment_profile(collected_at=now)
    except (OSError, RuntimeError, ValueError) as exc:
        raise _fail(
            "environment_probe_failed",
            "the local environment profile could not be collected",
        ) from exc
    if environment.environment_fingerprint != pinned_record["environment_fingerprint"]:
        raise _fail("selection_stale", "the current environment fingerprint changed")
    job_timeout = job.to_dict()["job_timeout_seconds"]
    outer_deadline_ns = started_ns + job_timeout * 1_000_000_000
    if time.monotonic_ns() >= outer_deadline_ns:
        raise _fail("job_timeout", "the image job deadline expired during preflight")

    explicit_artifact_root = (
        None
        if artifact_root is None
        else _confined_directory(
            artifact_root,
            workspace=workspace,
            label="artifact_root",
        )
    )
    try:
        with pin_decoded_inputs(
            input_path,
            input_mode=job.to_dict()["input_mode"],
            workspace_root=workspace,
            max_images=job.to_dict()["max_images"],
        ) as inputs:
            workload = build_qualification_workload_profile(job, inputs.inventory)
            if (
                inputs.inventory.local_input_digest != pinned_record["local_input_digest"]
                or workload.workload_fingerprint
                != pinned_record["qualification_workload_fingerprint"]
            ):
                raise _fail("selection_stale", "the pinned input or workload changed")
            with ArtifactResolver(
                workspace=workspace,
                artifact_root=explicit_artifact_root,
            ) as resolver:
                with resolver.pin(bundle) as artifacts:
                    if (
                        artifacts.artifact_resolver_state_digest
                        != expected_selected_resolver_state
                    ):
                        raise _fail(
                            "selection_stale",
                            "the selected artifact resolver state changed",
                        )
                    inventory = _artifact_inventory(
                        pinned=artifacts,
                        bundle=bundle,
                        verified_at=now,
                    )
                    if (
                        inventory.artifact_state_fingerprint
                        != pinned_record["selected_artifact_state_fingerprint"]
                    ):
                        raise _fail("selection_stale", "the selected artifact state changed")

                    def session_factory() -> _RunnerSession:
                        if route.kind == "code_owned_audited":
                            assert route.host_factory is not None
                            return _ForkedRunnerSession(
                                factory=route.host_factory,
                                bundle=bundle,
                                environment=environment,
                                artifacts=artifacts,
                                inputs=inputs,
                                labels=job.prompt_phrases,
                                outer_deadline_ns=outer_deadline_ns,
                            )
                        assert route.isolated_service is not None
                        return route.isolated_service.open_session(
                            bundle=bundle,
                            environment=environment,
                            artifacts=artifacts,
                            inputs=inputs,
                            labels=job.prompt_phrases,
                            outer_deadline_ns=outer_deadline_ns,
                        )

                    return _execute_pinned_pipeline(
                        job=job,
                        decision=pinned_decision,
                        bundle=bundle,
                        inputs=inputs,
                        artifacts=artifacts,
                        workspace=workspace,
                        destination=destination,
                        dry_run=dry_run,
                        force=force,
                        session_factory=session_factory,
                        outer_deadline_ns=outer_deadline_ns,
                    )
    except ProcessingError:
        raise
    except QualificationError as exc:
        code = "runner_timeout" if "timeout" in exc.code else exc.code
        raise _fail(code, "bounded runner execution failed safely") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise _fail("execution_preflight_failed", "pinned execution preflight failed") from exc
