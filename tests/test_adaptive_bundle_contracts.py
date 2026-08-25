from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tests.test_adaptive_image_contracts import _schema_accepts
from yolozu.adaptive.artifact_resolver import ArtifactResolver
from yolozu.adaptive.bundles import (
    EMPTY_PROFILE_SET_DIGEST,
    ZERO_DIGEST,
    build_fixed_class_mapping,
    map_fixed_class_outputs,
    map_text_prompt_outputs,
    project_bundle_lifecycle,
    project_support_profiles,
    validate_algorithm_bundle_registry,
    validate_algorithm_bundle_spec,
    validate_bundle_lifecycle_record,
    validate_support_profile_spec,
)
from yolozu.adaptive.canonical import canonical_sha256_v1
from yolozu.adaptive.control_records import (
    MAX_CONTROL_RECORD_BYTES,
    load_bounded_json_bytes,
    load_bounded_jsonl_bytes,
)


def _artifact(*, cache_key: str = "weights/model.onnx", data: bytes = b"model") -> dict[str, Any]:
    return {
        "artifact_id": "model",
        "order": 0,
        "role": "weight",
        "source_id": "https://example.invalid/model",
        "source_revision": "revision-1",
        "expected_size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "license_expression": "Apache-2.0",
        "license_source": "https://example.invalid/license",
        "license_revision": "revision-1",
        "cache_key": cache_key,
    }


def _bundle_payload(
    *,
    data: bytes = b"model",
    cache_key: str = "weights/model.onnx",
    version: str = "1.0-rc01",
) -> dict[str, Any]:
    artifacts = [_artifact(cache_key=cache_key, data=data)]
    labels = ["cat", "dog"]
    vocabulary = {
        "id": "example-classes",
        "digest": canonical_sha256_v1({"id": "example-classes", "labels": labels}),
        "labels": labels,
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "family_id": "example-detector",
        "bundle_id": "example-detector-onnx",
        "bundle_version": version,
        "spec_digest": "a" * 64,
        "provenance_class": "existing_code_owned",
        "test_only": False,
        "tasks": ["object_detection"],
        "prompt_modes": ["fixed_classes", "text"],
        "adapter_backend_id": "onnxruntime",
        "execution_binding": {
            "status": "bound",
            "artifact_scope": "runner_consumed",
            "reason_code": None,
        },
        "runner_id": "onnxruntime",
        "runner_version": "1.23.0",
        "execution_trust_class": "code_owned_audited",
        "loader_format": "onnx",
        "unsafe_deserialization_required": False,
        "model_source_id": "https://example.invalid/model-card",
        "model_revision": "revision-1",
        "runtime": {
            "runtime_id": "onnxruntime",
            "runtime_version": "1.23.0",
            "provider_id": "cpu",
            "provider_version": "1",
            "precision": "fp32",
            "architecture": "any",
            "accelerator_requirement": "none",
        },
        "model_input_shapes": [
            {"name": "images", "layout": "NCHW", "dimensions": [1, 3, 640, 640]}
        ],
        "decoder": {"id": "pillow", "version": "12.3.0", "digest": "b" * 64},
        "preprocess": {"id": "letterbox", "version": "1", "digest": "c" * 64},
        "postprocess": {"id": "class-map", "version": "1", "digest": "d" * 64},
        "execution_network_required": False,
        "runner_options": {"device_policy": "job_controlled"},
        "artifacts": artifacts,
        "artifact_set_digest": canonical_sha256_v1(artifacts),
        "class_vocabulary": vocabulary,
        "text_prompt_support": {
            "mode": "dynamic_text",
            "output_label_semantics": "request_prompt_index_v1",
        },
    }
    record["spec_digest"] = canonical_sha256_v1(record, own_digest_field="spec_digest")
    return record


def _registry_payload(*bundles: dict[str, Any]) -> dict[str, Any]:
    record = {
        "schema_version": 1,
        "registry_id": "yolozu-bundle-registry-v1",
        "bundles": list(bundles),
        "registry_digest": "a" * 64,
    }
    record["registry_digest"] = canonical_sha256_v1(
        record, own_digest_field="registry_digest"
    )
    return record


def _support_profile_payload() -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "profile_id": "cpu-batch",
        "profile_digest": "a" * 64,
        "task": "object_detection",
        "environment_fingerprint": "1" * 64,
        "qualification_workload_fingerprint": "2" * 64,
        "protocol_fingerprint": "3" * 64,
        "advertised_constraints": {
            "execution_mode": "batch",
            "max_p95_latency_ms": "50",
            "min_repeat_throughput_fps": "1",
        },
        "public_limitations": ["Exact measured CPU configuration only."],
    }
    record["profile_digest"] = canonical_sha256_v1(
        record, own_digest_field="profile_digest"
    )
    return record


def _support_record(
    *,
    sequence: int,
    previous: str,
    record_id: str,
    kind: str,
    variant: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "schema_version": 1,
        "stream_id": "support-profiles-v1",
        "sequence": sequence,
        "previous_record_digest": previous,
        "record_id": record_id,
        "kind": kind,
        "reviewer_role_id": "repo_maintainer",
        "review_reference": {"kind": "public_repository_id", "value": "gh-1"},
        "issuer_claim": "repository_source",
        "reason": "Reviewed support scope.",
        "occurred_at": f"2026-08-24T00:00:0{sequence}Z",
        "record_digest": "a" * 64,
        **variant,
    }
    record["record_digest"] = canonical_sha256_v1(
        record, own_digest_field="record_digest"
    )
    return record


def _lifecycle_event(
    *,
    sequence: int,
    previous: str,
    scope: str,
    event_type: str,
    variant: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "schema_version": 1,
        "stream_id": "bundle-lifecycle-v1",
        "sequence": sequence,
        "previous_event_digest": previous,
        "event_scope": scope,
        "event_type": event_type,
        "reviewer_role_id": "repo_maintainer",
        "review_reference": {"kind": "public_repository_id", "value": "gh-1"},
        "issuer_claim": "repository_source",
        "reason": f"Lifecycle step {sequence}.",
        "occurred_at": f"2026-08-24T00:01:{sequence:02d}Z",
        "event_digest": "a" * 64,
        **variant,
    }
    record["event_digest"] = canonical_sha256_v1(
        record, own_digest_field="event_digest"
    )
    return record


class TestAdaptiveBundleContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.schemas = {
            name: json.loads(
                (root / "docs" / "schemas" / f"{name}.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            for name in (
                "algorithm_bundle_spec",
                "algorithm_bundle_registry",
                "bundle_lifecycle_record",
                "support_profile_spec",
                "support_profile_record",
            )
        }

    def test_bundle_and_registry_validate_with_exact_version_bytes(self) -> None:
        first = validate_algorithm_bundle_spec(_bundle_payload(version="1.0-rc01"))
        second = validate_algorithm_bundle_spec(_bundle_payload(version="1.00"))
        self.assertNotEqual(first.spec_digest, second.spec_digest)
        self.assertLess(b"1.0-rc01", b"1.00")
        registry = validate_algorithm_bundle_registry(_registry_payload(first.to_dict()))
        self.assertEqual(len(registry.bundles), 1)
        self.assertTrue(
            _schema_accepts(
                first.to_dict(),
                self.schemas["algorithm_bundle_spec"],
                root=self.schemas["algorithm_bundle_spec"],
            )
        )
        self.assertEqual(
            self.schemas["algorithm_bundle_registry"]["properties"]["bundles"]["items"][
                "$ref"
            ],
            "algorithm_bundle_spec.schema.json",
        )

    def test_unbound_bundle_is_fetchable_metadata_not_an_execution_claim(self) -> None:
        payload = _bundle_payload()
        execution_fields = {
            "runner_id",
            "runner_version",
            "execution_trust_class",
            "execution_isolation_policy_digest",
            "loader_format",
            "unsafe_deserialization_required",
            "runtime",
            "model_input_shapes",
            "decoder",
            "preprocess",
            "postprocess",
            "execution_network_required",
            "runner_options",
        }
        for field in execution_fields:
            payload.pop(field, None)
        payload["execution_binding"] = {
            "status": "unbound",
            "artifact_scope": "fetchable_model_assets",
            "reason_code": "runner_artifact_set_incomplete",
        }
        payload["spec_digest"] = canonical_sha256_v1(
            payload, own_digest_field="spec_digest"
        )

        checked = validate_algorithm_bundle_spec(payload).to_dict()
        self.assertEqual(checked["execution_binding"]["status"], "unbound")
        self.assertFalse(execution_fields.intersection(checked))
        self.assertTrue(
            _schema_accepts(
                checked,
                self.schemas["algorithm_bundle_spec"],
                root=self.schemas["algorithm_bundle_spec"],
            )
        )

        payload["runner_id"] = "onnxruntime"
        payload["spec_digest"] = canonical_sha256_v1(
            payload, own_digest_field="spec_digest"
        )
        with self.assertRaisesRegex(ValueError, "unbound execution forbids"):
            validate_algorithm_bundle_spec(payload)
        self.assertFalse(
            _schema_accepts(
                payload,
                self.schemas["algorithm_bundle_spec"],
                root=self.schemas["algorithm_bundle_spec"],
            )
        )

    def test_bundle_rejects_mutable_pathlike_and_unsafe_state(self) -> None:
        cases: list[dict[str, Any]] = []
        for key, value in (
            ("bundle_version", "/tmp/model"),
            ("enabled", True),
            ("source_trust_domain", "yolozu_managed"),
            ("command", "python setup.py"),
        ):
            item = _bundle_payload()
            item[key] = value
            cases.append(item)
        path_artifact = _bundle_payload()
        path_artifact["artifacts"][0]["artifact_id"] = "../model"
        cases.append(path_artifact)
        deep_key = _bundle_payload()
        deep_key["artifacts"][0]["cache_key"] = "/absolute/model"
        cases.append(deep_key)
        optional = _bundle_payload()
        optional["artifacts"][0]["required"] = False
        cases.append(optional)
        unsafe_host = _bundle_payload()
        unsafe_host["loader_format"] = "pytorch_pickle"
        unsafe_host["unsafe_deserialization_required"] = True
        cases.append(unsafe_host)
        duplicate = _bundle_payload()
        duplicate["class_vocabulary"]["labels"] = ["A", "A"]
        cases.append(duplicate)
        for case in cases:
            with self.subTest(keys=sorted(case)), self.assertRaises(ValueError):
                validate_algorithm_bundle_spec(case)

    def test_screened_code_archive_requires_isolation_binding(self) -> None:
        payload = _bundle_payload()
        payload["provenance_class"] = "screened_candidate"
        payload["screening_binding"] = {
            "stream_key": "candidate-example",
            "pass_record_id": "screen-pass-1",
            "pass_record_digest": "e" * 64,
            "source_revision": "revision-1",
        }
        payload["execution_trust_class"] = "third_party_isolated"
        payload["execution_isolation_policy_digest"] = "f" * 64
        payload["loader_format"] = "python_archive"
        payload["unsafe_deserialization_required"] = True
        payload["artifacts"][0]["role"] = "code_archive"
        payload["artifact_set_digest"] = canonical_sha256_v1(payload["artifacts"])
        payload["spec_digest"] = canonical_sha256_v1(
            payload, own_digest_field="spec_digest"
        )
        self.assertEqual(
            validate_algorithm_bundle_spec(payload).to_dict()["execution_trust_class"],
            "third_party_isolated",
        )

    def test_fixed_and_text_output_mapping_never_guess_labels(self) -> None:
        bundle = validate_algorithm_bundle_spec(_bundle_payload())
        mapping = build_fixed_class_mapping(bundle, ["dog"])
        self.assertEqual(mapping["request_to_bundle_class_index"], [1])
        self.assertEqual(
            map_fixed_class_outputs(mapping, [0, 1]),
            [{"request_index": 0, "label": "dog"}],
        )
        with self.assertRaises(ValueError):
            build_fixed_class_mapping(bundle, ["canine"])
        with self.assertRaises(ValueError):
            map_fixed_class_outputs(mapping, ["dog"])
        with self.assertRaises(ValueError):
            map_fixed_class_outputs(mapping, [2])
        self.assertEqual(
            map_text_prompt_outputs(["red car"], [0]),
            [{"request_index": 0, "label": "red car"}],
        )
        for invalid in (-1, 1, "red car", True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                map_text_prompt_outputs(["red car"], [invalid])

    def test_support_profile_chain_is_immutable_and_ordered(self) -> None:
        profile = _support_profile_payload()
        definition = _support_record(
            sequence=1,
            previous=ZERO_DIGEST,
            record_id="define-cpu-batch",
            kind="profile_definition",
            variant={"profile": profile},
        )
        refs = [{"profile_id": profile["profile_id"], "profile_digest": profile["profile_digest"]}]
        assignment = _support_record(
            sequence=2,
            previous=definition["record_digest"],
            record_id="assign-cpu-batch",
            kind="profile_set_assignment",
            variant={
                "family_id": "example-detector",
                "channel": "Experimental",
                "profiles": refs,
                "profile_set_digest": canonical_sha256_v1(refs),
            },
        )
        projection = project_support_profiles(
            [definition, assignment], source_trust_domain="yolozu_managed"
        )
        self.assertEqual(projection.head_digest, assignment["record_digest"])
        self.assertEqual(
            projection.assignments[("example-detector", "Experimental")][
                "profile_set_digest"
            ],
            canonical_sha256_v1(refs),
        )
        self.assertTrue(
            _schema_accepts(
                validate_support_profile_spec(profile).to_dict(),
                self.schemas["support_profile_spec"],
                root=self.schemas["support_profile_spec"],
            )
        )
        self.assertEqual(
            self.schemas["support_profile_record"]["$defs"]["profile_definition"][
                "allOf"
            ][1]["properties"]["profile"]["$ref"],
            "support_profile_spec.schema.json",
        )
        broken = copy.deepcopy(assignment)
        broken["previous_record_digest"] = "9" * 64
        broken["record_digest"] = canonical_sha256_v1(
            broken, own_digest_field="record_digest"
        )
        with self.assertRaises(ValueError):
            project_support_profiles([definition, broken])
        with self.assertRaises(ValueError):
            project_support_profiles([definition, definition])

    def test_lifecycle_projection_channel_and_global_scopes_are_distinct(self) -> None:
        bundle_payload = _bundle_payload()
        bundle = validate_algorithm_bundle_spec(bundle_payload)
        registry = validate_algorithm_bundle_registry(_registry_payload(bundle_payload))
        reviews = [{"artifact_id": "model", "review_state": "approved"}]

        profile = _support_profile_payload()
        definition = _support_record(
            sequence=1,
            previous=ZERO_DIGEST,
            record_id="define-cpu-batch",
            kind="profile_definition",
            variant={"profile": profile},
        )
        refs = [{"profile_id": profile["profile_id"], "profile_digest": profile["profile_digest"]}]
        assignment = _support_record(
            sequence=2,
            previous=definition["record_digest"],
            record_id="assign-cpu-batch",
            kind="profile_set_assignment",
            variant={
                "family_id": "example-detector",
                "channel": "Experimental",
                "profiles": refs,
                "profile_set_digest": canonical_sha256_v1(refs),
            },
        )
        stable_assignment = _support_record(
            sequence=3,
            previous=assignment["record_digest"],
            record_id="assign-cpu-stable",
            kind="profile_set_assignment",
            variant={
                "family_id": "example-detector",
                "channel": "Stable",
                "profiles": refs,
                "profile_set_digest": canonical_sha256_v1(refs),
            },
        )
        support = project_support_profiles(
            [definition, assignment, stable_assignment],
            source_trust_domain="yolozu_managed",
        )

        global_register = _lifecycle_event(
            sequence=1,
            previous=ZERO_DIGEST,
            scope="bundle_global",
            event_type="register_global",
            variant={
                "family_id": "example-detector",
                "bundle_spec_digest": bundle.spec_digest,
                "artifact_set_digest": bundle.artifact_set_digest,
                "bundle_state": "enabled",
                "artifact_license_reviews": reviews,
            },
        )
        candidate = _lifecycle_event(
            sequence=2,
            previous=global_register["event_digest"],
            scope="channel_assignment",
            event_type="candidate_registration",
            variant={
                "family_id": "example-detector",
                "channel": "Candidate",
                "target_bundle_spec_digest": bundle.spec_digest,
                "target_artifact_set_digest": bundle.artifact_set_digest,
                "target_artifact_license_reviews": reviews,
                "support_profile_index_head": support.head_digest,
                "profile_set_record_id": None,
                "profile_set_record_digest": None,
                "profile_set_digest": EMPTY_PROFILE_SET_DIGEST,
                "profiles": [],
                "evidence_bindings": [],
            },
        )

        def public_event(
            *, sequence: int, previous: str, channel: str, set_record: dict[str, Any]
        ) -> dict[str, Any]:
            return _lifecycle_event(
                sequence=sequence,
                previous=previous,
                scope="channel_assignment",
                event_type="public_assignment",
                variant={
                    "family_id": "example-detector",
                    "channel": channel,
                    "target_bundle_spec_digest": bundle.spec_digest,
                    "target_artifact_set_digest": bundle.artifact_set_digest,
                    "target_artifact_license_reviews": reviews,
                    "support_profile_index_head": support.head_digest,
                    "profile_set_record_id": set_record["record_id"],
                    "profile_set_record_digest": set_record["record_digest"],
                    "profile_set_digest": canonical_sha256_v1(refs),
                    "profiles": refs,
                    "evidence_bindings": [
                        {
                            "profile_id": profile["profile_id"],
                            "profile_digest": profile["profile_digest"],
                            "activation_id": f"activation-{channel.lower()}",
                            "activation_digest": "8" * 64,
                            "trust_domain_claim": "yolozu_managed",
                        }
                    ],
                },
            )

        experimental = public_event(
            sequence=3,
            previous=candidate["event_digest"],
            channel="Experimental",
            set_record=assignment,
        )
        stable = public_event(
            sequence=4,
            previous=experimental["event_digest"],
            channel="Stable",
            set_record=stable_assignment,
        )
        rollback = _lifecycle_event(
            sequence=5,
            previous=stable["event_digest"],
            scope="channel_none",
            event_type="channel_none",
            variant={
                "family_id": "example-detector",
                "channel": "Experimental",
                "profile_set_digest": EMPTY_PROFILE_SET_DIGEST,
                "profiles": [],
                "evidence_bindings": [],
                "prior_bundle_spec_digest": bundle.spec_digest,
                "prior_artifact_set_digest": bundle.artifact_set_digest,
                "prior_support_profile_index_head": support.head_digest,
                "prior_profile_set_record_digest": assignment["record_digest"],
                "prior_profile_set_digest": canonical_sha256_v1(refs),
            },
        )
        revoked = _lifecycle_event(
            sequence=6,
            previous=rollback["event_digest"],
            scope="bundle_global",
            event_type="revoke",
            variant={
                "family_id": "example-detector",
                "bundle_spec_digest": bundle.spec_digest,
                "artifact_set_digest": bundle.artifact_set_digest,
                "bundle_state": "revoked",
                "artifact_license_reviews": reviews,
            },
        )
        projection = project_bundle_lifecycle(
            registry,
            [global_register, candidate, experimental, stable, rollback],
            source_trust_domain="yolozu_managed",
            support_profiles=support,
        )
        self.assertIsNone(
            projection.channel_pointers[("example-detector", "Experimental")]
        )
        self.assertTrue(
            projection.is_lifecycle_eligible(
                family_id="example-detector", channel="Stable"
            )
        )
        revoked_projection = project_bundle_lifecycle(
            registry,
            [global_register, candidate, experimental, stable, rollback, revoked],
            source_trust_domain="yolozu_managed",
            support_profiles=support,
        )
        self.assertFalse(
            revoked_projection.is_lifecycle_eligible(
                family_id="example-detector", channel="Stable"
            )
        )
        un_revoke = copy.deepcopy(revoked)
        un_revoke["sequence"] = 7
        un_revoke["previous_event_digest"] = revoked["event_digest"]
        un_revoke["event_type"] = "enable"
        un_revoke["bundle_state"] = "enabled"
        un_revoke["event_digest"] = canonical_sha256_v1(
            un_revoke, own_digest_field="event_digest"
        )
        with self.assertRaises(ValueError):
            project_bundle_lifecycle(
                registry,
                [
                    global_register,
                    candidate,
                    experimental,
                    stable,
                    rollback,
                    revoked,
                    un_revoke,
                ],
                support_profiles=support,
            )
        self.assertTrue(
            _schema_accepts(
                validate_bundle_lifecycle_record(
                    global_register, registry=registry
                ).to_dict(),
                self.schemas["bundle_lifecycle_record"],
                root=self.schemas["bundle_lifecycle_record"],
            )
        )

    def test_unknown_license_history_validates_but_public_assignment_fails(self) -> None:
        bundle = validate_algorithm_bundle_spec(_bundle_payload())
        registry = validate_algorithm_bundle_registry(_registry_payload(bundle.to_dict()))
        unknown = [{"artifact_id": "model", "review_state": "unknown"}]
        registered = _lifecycle_event(
            sequence=1,
            previous=ZERO_DIGEST,
            scope="bundle_global",
            event_type="register_global",
            variant={
                "family_id": "example-detector",
                "bundle_spec_digest": bundle.spec_digest,
                "artifact_set_digest": bundle.artifact_set_digest,
                "bundle_state": "enabled",
                "artifact_license_reviews": unknown,
            },
        )
        candidate = _lifecycle_event(
            sequence=2,
            previous=registered["event_digest"],
            scope="channel_assignment",
            event_type="candidate_registration",
            variant={
                "family_id": "example-detector",
                "channel": "Candidate",
                "target_bundle_spec_digest": bundle.spec_digest,
                "target_artifact_set_digest": bundle.artifact_set_digest,
                "target_artifact_license_reviews": unknown,
                "support_profile_index_head": ZERO_DIGEST,
                "profile_set_record_id": None,
                "profile_set_record_digest": None,
                "profile_set_digest": EMPTY_PROFILE_SET_DIGEST,
                "profiles": [],
                "evidence_bindings": [],
            },
        )
        self.assertEqual(
            project_bundle_lifecycle(registry, [registered, candidate]).bundle_states[
                bundle.spec_digest
            ]["artifact_license_reviews"],
            unknown,
        )
        profile = _support_profile_payload()
        refs = [
            {
                "profile_id": profile["profile_id"],
                "profile_digest": profile["profile_digest"],
            }
        ]
        public = _lifecycle_event(
            sequence=3,
            previous=candidate["event_digest"],
            scope="channel_assignment",
            event_type="public_assignment",
            variant={
                "family_id": "example-detector",
                "channel": "Experimental",
                "target_bundle_spec_digest": bundle.spec_digest,
                "target_artifact_set_digest": bundle.artifact_set_digest,
                "target_artifact_license_reviews": unknown,
                "support_profile_index_head": "7" * 64,
                "profile_set_record_id": "set-1",
                "profile_set_record_digest": "6" * 64,
                "profile_set_digest": canonical_sha256_v1(refs),
                "profiles": refs,
                "evidence_bindings": [
                    {
                        "profile_id": profile["profile_id"],
                        "profile_digest": profile["profile_digest"],
                        "activation_id": "activation-1",
                        "activation_digest": "8" * 64,
                        "trust_domain_claim": "yolozu_managed",
                    }
                ],
            },
        )
        with self.assertRaises(ValueError):
            project_bundle_lifecycle(registry, [registered, candidate, public])

    def test_disable_enable_and_license_review_are_append_only(self) -> None:
        bundle = validate_algorithm_bundle_spec(_bundle_payload())
        registry = validate_algorithm_bundle_registry(_registry_payload(bundle.to_dict()))
        unknown = [{"artifact_id": "model", "review_state": "unknown"}]
        approved = [{"artifact_id": "model", "review_state": "approved"}]
        registered = _lifecycle_event(
            sequence=1,
            previous=ZERO_DIGEST,
            scope="bundle_global",
            event_type="register_global",
            variant={
                "family_id": "example-detector",
                "bundle_spec_digest": bundle.spec_digest,
                "artifact_set_digest": bundle.artifact_set_digest,
                "bundle_state": "enabled",
                "artifact_license_reviews": unknown,
            },
        )
        reviewed = _lifecycle_event(
            sequence=2,
            previous=registered["event_digest"],
            scope="bundle_global",
            event_type="license_review",
            variant={
                "family_id": "example-detector",
                "bundle_spec_digest": bundle.spec_digest,
                "artifact_set_digest": bundle.artifact_set_digest,
                "bundle_state": "enabled",
                "artifact_license_reviews": approved,
            },
        )
        disabled = _lifecycle_event(
            sequence=3,
            previous=reviewed["event_digest"],
            scope="bundle_global",
            event_type="disable",
            variant={
                "family_id": "example-detector",
                "bundle_spec_digest": bundle.spec_digest,
                "artifact_set_digest": bundle.artifact_set_digest,
                "bundle_state": "disabled",
                "artifact_license_reviews": approved,
            },
        )
        enabled = _lifecycle_event(
            sequence=4,
            previous=disabled["event_digest"],
            scope="bundle_global",
            event_type="enable",
            variant={
                "family_id": "example-detector",
                "bundle_spec_digest": bundle.spec_digest,
                "artifact_set_digest": bundle.artifact_set_digest,
                "bundle_state": "enabled",
                "artifact_license_reviews": approved,
            },
        )
        original_bytes = json.dumps(registered, sort_keys=True).encode()
        projection = project_bundle_lifecycle(
            registry, [registered, reviewed, disabled, enabled]
        )
        self.assertEqual(
            projection.bundle_states[bundle.spec_digest]["bundle_state"], "enabled"
        )
        self.assertEqual(json.dumps(registered, sort_keys=True).encode(), original_bytes)
        personal = copy.deepcopy(enabled)
        personal["reviewer_role_id"] = "akira@example.invalid"
        personal["event_digest"] = canonical_sha256_v1(
            personal, own_digest_field="event_digest"
        )
        with self.assertRaises(ValueError):
            validate_bundle_lifecycle_record(personal, registry=registry)

    def test_only_current_versioned_pointer_is_lifecycle_eligible(self) -> None:
        old_payload = _bundle_payload(version="1.0")
        new_payload = _bundle_payload(version="1.00")
        old = validate_algorithm_bundle_spec(old_payload)
        new = validate_algorithm_bundle_spec(new_payload)
        registry = validate_algorithm_bundle_registry(
            _registry_payload(old_payload, new_payload)
        )
        reviews = [{"artifact_id": "model", "review_state": "approved"}]
        profile = _support_profile_payload()
        refs = [
            {
                "profile_id": profile["profile_id"],
                "profile_digest": profile["profile_digest"],
            }
        ]
        profile_definition = _support_record(
            sequence=1,
            previous=ZERO_DIGEST,
            record_id="define-current-version-profile",
            kind="profile_definition",
            variant={"profile": profile},
        )
        profile_assignment = _support_record(
            sequence=2,
            previous=profile_definition["record_digest"],
            record_id="assign-current-version-profile",
            kind="profile_set_assignment",
            variant={
                "family_id": "example-detector",
                "channel": "Experimental",
                "profiles": refs,
                "profile_set_digest": canonical_sha256_v1(refs),
            },
        )
        support = project_support_profiles(
            [profile_definition, profile_assignment],
            source_trust_domain="yolozu_managed",
        )

        events: list[dict[str, Any]] = []

        def append(scope: str, event_type: str, variant: dict[str, Any]) -> None:
            previous = ZERO_DIGEST if not events else events[-1]["event_digest"]
            events.append(
                _lifecycle_event(
                    sequence=len(events) + 1,
                    previous=previous,
                    scope=scope,
                    event_type=event_type,
                    variant=variant,
                )
            )

        def global_variant(bundle: Any) -> dict[str, Any]:
            return {
                "family_id": "example-detector",
                "bundle_spec_digest": bundle.spec_digest,
                "artifact_set_digest": bundle.artifact_set_digest,
                "bundle_state": "enabled",
                "artifact_license_reviews": reviews,
            }

        def candidate_variant(bundle: Any) -> dict[str, Any]:
            return {
                "family_id": "example-detector",
                "channel": "Candidate",
                "target_bundle_spec_digest": bundle.spec_digest,
                "target_artifact_set_digest": bundle.artifact_set_digest,
                "target_artifact_license_reviews": reviews,
                "support_profile_index_head": support.head_digest,
                "profile_set_record_id": None,
                "profile_set_record_digest": None,
                "profile_set_digest": EMPTY_PROFILE_SET_DIGEST,
                "profiles": [],
                "evidence_bindings": [],
            }

        def public_variant(bundle: Any, suffix: str) -> dict[str, Any]:
            return {
                "family_id": "example-detector",
                "channel": "Experimental",
                "target_bundle_spec_digest": bundle.spec_digest,
                "target_artifact_set_digest": bundle.artifact_set_digest,
                "target_artifact_license_reviews": reviews,
                "support_profile_index_head": support.head_digest,
                "profile_set_record_id": profile_assignment["record_id"],
                "profile_set_record_digest": profile_assignment["record_digest"],
                "profile_set_digest": canonical_sha256_v1(refs),
                "profiles": refs,
                "evidence_bindings": [
                    {
                        "profile_id": profile["profile_id"],
                        "profile_digest": profile["profile_digest"],
                        "activation_id": f"activation-{suffix}",
                        "activation_digest": ("8" if suffix == "old" else "9") * 64,
                        "trust_domain_claim": "yolozu_managed",
                    }
                ],
            }

        append("bundle_global", "register_global", global_variant(old))
        append("channel_assignment", "candidate_registration", candidate_variant(old))
        append("channel_assignment", "public_assignment", public_variant(old, "old"))
        append("bundle_global", "register_global", global_variant(new))
        append("channel_assignment", "candidate_registration", candidate_variant(new))
        append("channel_assignment", "public_assignment", public_variant(new, "new"))
        with self.assertRaisesRegex(ValueError, "projection"):
            project_bundle_lifecycle(registry, events)
        projection = project_bundle_lifecycle(
            registry,
            events,
            source_trust_domain="yolozu_managed",
            support_profiles=support,
        )
        current = projection.channel_pointers[("example-detector", "Experimental")]
        self.assertIsNotNone(current)
        self.assertEqual(current["bundle_spec_digest"], new.spec_digest)
        self.assertNotEqual(current["bundle_spec_digest"], old.spec_digest)

    def test_bounded_control_parser_rejects_duplicate_float_depth_and_blank_jsonl(self) -> None:
        self.assertEqual(load_bounded_json_bytes(b'{"a":[1,true,null]}'), {"a": [1, True, None]})
        for raw in (b'{"a":1,"a":2}', b'{"a":1.5}', b'{"a":1} trailing'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                load_bounded_json_bytes(raw)
        for too_deep in (
            (b"[" * 65) + (b"]" * 65),
            (b"[" * 65) + b"0" + (b"]" * 65),
        ):
            with self.subTest(depth_case=too_deep[-4:]), self.assertRaises(ValueError):
                load_bounded_json_bytes(too_deep)
        with self.assertRaises(ValueError):
            load_bounded_json_bytes(b'"' + b"x" * MAX_CONTROL_RECORD_BYTES + b'"')
        with self.assertRaises(ValueError):
            load_bounded_jsonl_bytes(b"{}\n\n{}\n", max_records=3)

    def test_artifact_resolver_verifies_store_and_rejects_symlink(self) -> None:
        data = b"model"
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            store = workspace / "artifacts"
            (store / "weights").mkdir(parents=True)
            (store / "weights" / "model.onnx").write_bytes(data)
            bundle = validate_algorithm_bundle_spec(
                _bundle_payload(data=data, cache_key="weights/model.onnx")
            )
            with ArtifactResolver(workspace=workspace, artifact_root=store) as resolver:
                verified = resolver.verify(bundle)
            self.assertEqual(len(verified.artifacts), 1)
            self.assertEqual(verified.artifacts[0].sha256, hashlib.sha256(data).hexdigest())
            self.assertEqual(len(verified.artifact_resolver_state_digest), 64)

            target = store / "weights" / "model.onnx"
            target.unlink()
            external = workspace / "outside.onnx"
            external.write_bytes(data)
            target.symlink_to(external)
            with ArtifactResolver(workspace=workspace, artifact_root=store) as resolver:
                with self.assertRaises(ValueError):
                    resolver.verify(bundle)

    def test_artifact_resolver_preflights_all_members_before_hashing(self) -> None:
        first_data = b"first"
        second_expected = b"second"
        payload = _bundle_payload(
            data=first_data,
            cache_key="weights/first.onnx",
        )
        second = _artifact(
            data=second_expected,
            cache_key="configs/second.json",
        )
        second.update({"artifact_id": "config", "order": 1, "role": "config"})
        payload["artifacts"].append(second)
        payload["artifact_set_digest"] = canonical_sha256_v1(payload["artifacts"])
        payload["spec_digest"] = canonical_sha256_v1(
            payload, own_digest_field="spec_digest"
        )
        bundle = validate_algorithm_bundle_spec(payload)

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            store = workspace / "artifacts"
            (store / "weights").mkdir(parents=True)
            (store / "configs").mkdir()
            (store / "weights" / "first.onnx").write_bytes(first_data)
            (store / "configs" / "second.json").write_bytes(b"bad")
            with ArtifactResolver(workspace=workspace, artifact_root=store) as resolver:
                with patch("yolozu.adaptive.artifact_resolver.os.read") as read:
                    with self.assertRaisesRegex(ValueError, "stat size mismatch"):
                        resolver.verify(bundle)
                    read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
