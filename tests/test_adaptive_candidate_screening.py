from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_adaptive_bundle_contracts import _bundle_payload
from tests.test_adaptive_image_contracts import _schema_accepts
from tests.test_adaptive_selector import _context, _select
from yolozu.adaptive.bundles import (
    validate_algorithm_bundle_registry,
    validate_algorithm_bundle_spec,
)
from yolozu.adaptive.canonical import canonical_json_v1, canonical_sha256_v1
from yolozu.adaptive.screening import (
    MAX_SCREENING_RECORDS,
    MAX_SCREENING_STREAM_BYTES,
    ZERO_DIGEST,
    _validate_screening_stream_envelope,
    build_screening_eligibility_observation,
    compute_candidate_screening_stream_key,
    load_candidate_screening_jsonl_bytes,
    project_candidate_screening_records,
    validate_candidate_screening_record,
)


def _reference() -> dict[str, str]:
    return {"kind": "public_repository_id", "value": "review-YOLOZU-ll2.81.3.4"}


def _record() -> dict:
    reference = _reference()
    candidate = {
        "candidate_id": "example-detector-v1",
        "source": {
            "scheme": "https",
            "host": "github.com",
            "path": "/example/detector/releases/tag/v1.0.0",
        },
        "immutable_revision": "sha256:source-revision-1",
        "requested_capability": {
            "task": "object_detection",
            "prompt_modes": ["fixed_classes"],
            "output_interface_contract_id": "predictions-v2",
        },
    }
    payload = {
        "schema_version": 1,
        "stream_key": compute_candidate_screening_stream_key(candidate),
        "sequence": 1,
        "previous_record_digest": ZERO_DIGEST,
        "record_id": "screening-example-1",
        "record_digest": "a" * 64,
        "candidate": candidate,
        "mechanical_checks": {
            "source_provenance": {
                "result": "pass",
                "revision_kind": "immutable",
                "reference": reference,
            },
            "source_integrity": {
                "result": "pass",
                "procedure_id": "source-checksum-v1",
                "expected_sha256": "1" * 64,
                "reference": reference,
            },
            "code_license": {
                "result": "pass",
                "spdx_id": "Apache-2.0",
                "reference": reference,
            },
            "weight_license": {
                "result": "pass",
                "spdx_id": "Apache-2.0",
                "reference": reference,
            },
            "dataset_evaluation_license": {
                "result": "pass",
                "spdx_id": "CC-BY-4.0",
                "reference": reference,
            },
            "weight_source_integrity": {
                "result": "pass",
                "source_kind": "official_release",
                "procedure_id": "weight-checksum-v1",
                "expected_sha256": "2" * 64,
                "reference": reference,
            },
            "local_availability": {
                "result": "pass",
                "mode": "downloadable_artifacts",
                "reference": reference,
            },
            "task_prompt_output_fit": {
                "result": "pass",
                "reference": reference,
            },
            "predictions_interface_mapping": {
                "result": "pass",
                "interface_contract_id": "predictions-v2",
                "mapping_revision": "mapping-v1",
                "reference": reference,
            },
            "runtime_provider": {
                "result": "pass",
                "runtime_ids": ["onnxruntime"],
                "provider_ids": ["cpu"],
                "requires_new_surface": False,
                "reference": reference,
            },
            "compute_memory": {
                "result": "pass",
                "estimated_compute_operations": 1000000,
                "estimated_peak_memory_bytes": 1048576,
                "reference": reference,
            },
            "maintenance": {
                "result": "pass",
                "status": "maintained",
                "reference": reference,
            },
            "security_supply_chain": {
                "result": "pass",
                "status": "no_known_concern",
                "reference": reference,
            },
        },
        "human_review": {
            "status": "reviewed",
            "reviewer_role_id": "release_reviewer",
            "review_reference": reference,
        },
        "overall_status": "pass",
        "reason_codes": [],
        "issuer_claim": "repository_source",
        "reviewed_at": "2026-08-26T00:00:00Z",
        "supersedes_record_id": None,
        "supersedes_record_digest": None,
    }
    return _finalize(payload)


def _finalize(payload: dict) -> dict:
    reasons = []
    failed = False
    unknown = False
    for name, assessment in payload["mechanical_checks"].items():
        if assessment["result"] == "fail":
            failed = True
            reasons.append(f"{name}_failed")
        elif assessment["result"] == "unknown":
            unknown = True
            if name == "runtime_provider" and assessment.get(
                "requires_new_surface"
            ):
                reasons.append("runtime_provider_new_surface_hold")
            else:
                reasons.append(f"{name}_unknown")
    if payload["human_review"]["status"] == "unreviewed":
        unknown = True
        reasons.append("human_review_unreviewed")
    payload["overall_status"] = "reject" if failed else ("hold" if unknown else "pass")
    payload["reason_codes"] = reasons
    payload["record_digest"] = canonical_sha256_v1(
        payload,
        own_digest_field="record_digest",
    )
    return payload


def _supersede(prior: dict, *, record_id: str = "screening-example-2") -> dict:
    current = copy.deepcopy(prior)
    current["sequence"] = prior["sequence"] + 1
    current["previous_record_digest"] = prior["record_digest"]
    current["record_id"] = record_id
    current["supersedes_record_id"] = prior["record_id"]
    current["supersedes_record_digest"] = prior["record_digest"]
    current["reviewed_at"] = "2026-08-26T00:01:00Z"
    return current


def _screened_bundle(record: dict) -> dict:
    bundle = _bundle_payload()
    bundle["provenance_class"] = "screened_candidate"
    bundle["screening_binding"] = {
        "stream_key": record["stream_key"],
        "pass_record_id": record["record_id"],
        "pass_record_digest": record["record_digest"],
        "source_revision": record["candidate"]["immutable_revision"],
    }
    bundle["spec_digest"] = canonical_sha256_v1(
        bundle,
        own_digest_field="spec_digest",
    )
    return bundle


class TestAdaptiveCandidateScreening(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.schema = json.loads(
            (root / "docs" / "schemas" / "candidate_screening_record.schema.json").read_text(
                encoding="utf-8"
            )
        )

    def test_pass_is_schema_valid_and_keeps_license_facts_separate(self) -> None:
        payload = _record()
        checked = validate_candidate_screening_record(
            payload,
            source_trust_domain="yolozu_managed",
        )
        self.assertEqual(checked.overall_status, "pass")
        self.assertTrue(_schema_accepts(checked.to_dict(), self.schema, root=self.schema))
        checks = checked.to_dict()["mechanical_checks"]
        self.assertEqual(
            {checks[name]["spdx_id"] for name in (
                "code_license",
                "weight_license",
                "dataset_evaluation_license",
            )},
            {"Apache-2.0", "CC-BY-4.0"},
        )

    def test_mandatory_unknowns_hold_and_hard_failures_reject(self) -> None:
        cases = []

        unknown_license = _record()
        unknown_license["mechanical_checks"]["weight_license"].update(
            {"result": "unknown", "spdx_id": None}
        )
        cases.append((unknown_license, "hold", "weight_license_unknown"))

        missing_hash = _record()
        missing_hash["mechanical_checks"]["source_integrity"].update(
            {"result": "unknown", "procedure_id": None, "expected_sha256": None}
        )
        cases.append((missing_hash, "hold", "source_integrity_unknown"))

        new_runtime = _record()
        new_runtime["mechanical_checks"]["runtime_provider"].update(
            {
                "result": "unknown",
                "runtime_ids": ["new-runtime"],
                "provider_ids": ["new-provider"],
                "requires_new_surface": True,
            }
        )
        cases.append((new_runtime, "hold", "runtime_provider_new_surface_hold"))

        unreviewed = _record()
        unreviewed["human_review"]["status"] = "unreviewed"
        cases.append((unreviewed, "hold", "human_review_unreviewed"))

        remote_only = _record()
        remote_only["mechanical_checks"]["local_availability"].update(
            {"result": "fail", "mode": "hosted_only"}
        )
        cases.append((remote_only, "reject", "local_availability_failed"))

        mutable = _record()
        mutable["mechanical_checks"]["source_provenance"].update(
            {"result": "fail", "revision_kind": "mutable"}
        )
        cases.append((mutable, "reject", "source_provenance_failed"))

        output_mismatch = _record()
        output_mismatch["mechanical_checks"]["predictions_interface_mapping"].update(
            {
                "result": "fail",
                "interface_contract_id": None,
                "mapping_revision": None,
            }
        )
        cases.append(
            (
                output_mismatch,
                "reject",
                "predictions_interface_mapping_failed",
            )
        )

        for payload, status, reason in cases:
            with self.subTest(reason=reason):
                checked = validate_candidate_screening_record(
                    _finalize(payload),
                    source_trust_domain="yolozu_managed",
                ).to_dict()
                self.assertEqual(checked["overall_status"], status)
                self.assertIn(reason, checked["reason_codes"])

    def test_trust_comes_from_loader_path_and_role_is_non_personal(self) -> None:
        payload = _record()
        managed = validate_candidate_screening_record(
            payload,
            source_trust_domain="yolozu_managed",
        )
        asserted = validate_candidate_screening_record(
            payload,
            source_trust_domain="operator_asserted",
        )
        self.assertEqual(managed.source_trust_domain, "yolozu_managed")
        self.assertEqual(asserted.source_trust_domain, "operator_asserted")

        claimed = copy.deepcopy(payload)
        claimed["trust_domain"] = "yolozu_managed"
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            validate_candidate_screening_record(claimed)

        for role in ("reviewer@example.com", "Akira", "a"):
            private = copy.deepcopy(payload)
            private["human_review"]["reviewer_role_id"] = role
            private["record_digest"] = canonical_sha256_v1(
                private, own_digest_field="record_digest"
            )
            with self.subTest(role=role), self.assertRaisesRegex(
                ValueError, "reviewer_role_id"
            ):
                validate_candidate_screening_record(private)

    def test_projection_requires_one_exact_append_only_chain_per_key(self) -> None:
        first = _record()
        second = _supersede(first)
        second["mechanical_checks"]["maintenance"].update(
            {"result": "fail", "status": "stale"}
        )
        second = _finalize(second)
        projection = project_candidate_screening_records(
            [first, second],
            source_trust_domain="yolozu_managed",
        )
        self.assertEqual(
            projection.current_by_stream[first["stream_key"]].overall_status,
            "reject",
        )

        invalid_cases = []
        gap = copy.deepcopy(second)
        gap["sequence"] = 3
        gap = _finalize(gap)
        invalid_cases.append(gap)
        wrong = copy.deepcopy(second)
        wrong["previous_record_digest"] = "f" * 64
        wrong["supersedes_record_digest"] = "f" * 64
        wrong = _finalize(wrong)
        invalid_cases.append(wrong)
        for invalid in invalid_cases:
            with self.assertRaisesRegex(ValueError, "sequence gap|predecessor"):
                project_candidate_screening_records(
                    [first, invalid],
                    source_trust_domain="yolozu_managed",
                )
        with self.assertRaisesRegex(ValueError, "duplicate record_id"):
            project_candidate_screening_records(
                [first, first],
                source_trust_domain="yolozu_managed",
            )

    def test_stream_caps_partial_suffix_and_shared_parser_fail_closed(self) -> None:
        exact_bytes = b" " * (MAX_SCREENING_STREAM_BYTES - 1) + b"\n"
        _validate_screening_stream_envelope(exact_bytes)
        with self.assertRaisesRegex(ValueError, "control_stream_limit_exceeded"):
            _validate_screening_stream_envelope(exact_bytes + b"\n")

        exact_records = b"{}\n" * MAX_SCREENING_RECORDS
        _validate_screening_stream_envelope(exact_records)
        with self.assertRaisesRegex(ValueError, "control_stream_limit_exceeded"):
            _validate_screening_stream_envelope(exact_records + b"{}\n")

        encoded = canonical_json_v1(_record())
        with self.assertRaisesRegex(ValueError, "partial suffix"):
            load_candidate_screening_jsonl_bytes(encoded)
        with self.assertRaisesRegex(ValueError, "stream is invalid"):
            load_candidate_screening_jsonl_bytes(b'{"schema_version":1,"x":1,"x":2}\n')

    def test_provider_maps_all_states_without_trusting_custom_input(self) -> None:
        passed = _record()
        bundle = validate_algorithm_bundle_spec(_screened_bundle(passed))
        existing = validate_algorithm_bundle_spec(_bundle_payload())
        self.assertEqual(
            build_screening_eligibility_observation(existing, None).to_dict()["status"],
            "not_applicable",
        )
        self.assertEqual(
            build_screening_eligibility_observation(bundle, None).to_dict()["status"],
            "absent",
        )

        managed_pass = project_candidate_screening_records(
            [passed], source_trust_domain="yolozu_managed"
        )
        self.assertEqual(
            build_screening_eligibility_observation(bundle, managed_pass).to_dict()[
                "status"
            ],
            "current_pass",
        )
        custom_pass = project_candidate_screening_records(
            [passed], source_trust_domain="operator_asserted"
        )
        self.assertEqual(
            build_screening_eligibility_observation(bundle, custom_pass).to_dict()[
                "status"
            ],
            "untrusted",
        )

        for outcome, expected in (("hold", "current_hold"), ("reject", "current_reject")):
            current = _supersede(passed, record_id=f"screening-{outcome}-2")
            if outcome == "hold":
                current["mechanical_checks"]["weight_license"].update(
                    {"result": "unknown", "spdx_id": None}
                )
            else:
                current["mechanical_checks"]["maintenance"].update(
                    {"result": "fail", "status": "stale"}
                )
            projection = project_candidate_screening_records(
                [passed, _finalize(current)],
                source_trust_domain="yolozu_managed",
            )
            self.assertEqual(
                build_screening_eligibility_observation(bundle, projection).to_dict()[
                    "status"
                ],
                expected,
            )

        conflict_bundle = _screened_bundle(passed)
        conflict_bundle["screening_binding"]["pass_record_id"] = "different-pass"
        conflict_bundle["spec_digest"] = canonical_sha256_v1(
            conflict_bundle, own_digest_field="spec_digest"
        )
        self.assertEqual(
            build_screening_eligibility_observation(
                conflict_bundle, managed_pass
            ).to_dict()["status"],
            "conflict",
        )

        revision_bundle = _screened_bundle(passed)
        revision_bundle["screening_binding"]["source_revision"] = "different-revision"
        revision_bundle["spec_digest"] = canonical_sha256_v1(
            revision_bundle, own_digest_field="spec_digest"
        )
        self.assertEqual(
            build_screening_eligibility_observation(
                revision_bundle, managed_pass
            ).to_dict()["status"],
            "revision_mismatch",
        )

    def test_current_reject_immediately_excludes_promoted_bundle(self) -> None:
        passed = _record()
        bundle = _screened_bundle(passed)
        rejected = _supersede(passed)
        rejected["mechanical_checks"]["security_supply_chain"].update(
            {"result": "fail", "status": "concern_present"}
        )
        projection = project_candidate_screening_records(
            [passed, _finalize(rejected)],
            source_trust_domain="yolozu_managed",
        )
        context = _context(bundle)
        context.screening[bundle["spec_digest"]] = (
            build_screening_eligibility_observation(bundle, projection)
        )
        with patch("builtins.open", side_effect=AssertionError("unexpected file read")):
            decision = _select(context).to_dict()
        self.assertEqual(decision["status"], "abstained")
        self.assertIn(
            "screening_not_current_pass",
            decision["candidate_evaluations"][0]["reason_codes"],
        )

    def test_screening_record_is_not_a_bundle_registry(self) -> None:
        with self.assertRaises(ValueError):
            validate_algorithm_bundle_registry(_record())

    def test_packaged_holds_and_custom_stream_preserve_path_trust(self) -> None:
        empty = load_candidate_screening_jsonl_bytes(
            b"", source_trust_domain="yolozu_managed"
        )
        self.assertEqual(empty.current_by_stream, {})

        repo_root = Path(__file__).resolve().parents[1]
        packaged = load_candidate_screening_jsonl_bytes(
            (repo_root / "yolozu/data/adaptive_routing/candidate_screening.jsonl").read_bytes(),
            source_trust_domain="yolozu_managed",
        )
        self.assertEqual(len(packaged.current_by_stream), 2)
        current = [record.to_dict() for record in packaged.current_by_stream.values()]
        self.assertEqual({record["overall_status"] for record in current}, {"hold"})
        self.assertEqual(
            {record["record_id"] for record in current},
            {
                "screening-groundingdino-sam21-20260829",
                "screening-rfdetr-nano-1.9.4-20260829",
            },
        )
        self.assertTrue(
            all(
                "runtime_provider_new_surface_hold" in record["reason_codes"]
                for record in current
            )
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stream = root / "candidate_screening.jsonl"
            stream.write_bytes(canonical_json_v1(_record()) + b"\n")
            loaded = load_candidate_screening_jsonl_bytes(
                stream.read_bytes(), source_trust_domain="operator_asserted"
            )
        self.assertEqual(loaded.source_trust_domain, "operator_asserted")


if __name__ == "__main__":
    unittest.main()
