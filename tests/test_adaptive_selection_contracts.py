from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_adaptive_bundle_contracts import _bundle_payload, _registry_payload
from tests.test_adaptive_evidence_contracts import _report_payload
from tests.test_adaptive_image_contracts import _schema_accepts
from yolozu.adaptive.bundles import (
    build_fixed_class_mapping,
    validate_algorithm_bundle_registry,
    validate_algorithm_bundle_spec,
)
from yolozu.adaptive.canonical import canonical_sha256_v1
from yolozu.adaptive.selection import (
    CANDIDATE_REASON_CODES,
    MAX_SELECTION_DECISION_BYTES,
    validate_screening_eligibility_observation,
    validate_selection_decision,
    validate_support_profile_eligibility_observation,
)
from yolozu.adaptive.evidence import validate_qualification_report


AS_OF = "2026-08-25T12:00:00Z"


def _redigest(record: dict, field: str) -> None:
    record[field] = canonical_sha256_v1(record, own_digest_field=field)


def _screening(bundle: dict, *, status: str = "not_applicable") -> dict:
    if bundle["provenance_class"] == "existing_code_owned":
        record = {
            "schema_version": 1,
            "provider_id": "no_screening_required",
            "provider_version": "1",
            "provenance_class": "existing_code_owned",
            "screening_stream_key": None,
            "source_revision": None,
            "status": "not_applicable",
            "current_record_id": None,
            "current_record_digest": None,
            "projection_head_digest": None,
            "trust_domain": "unknown",
            "observation_digest": "a" * 64,
        }
    else:
        binding = bundle["screening_binding"]
        unique = status in {"current_pass", "current_hold", "current_reject"}
        record = {
            "schema_version": 1,
            "provider_id": "candidate-screening-projection",
            "provider_version": "1",
            "provenance_class": "screened_candidate",
            "screening_stream_key": binding["stream_key"],
            "source_revision": binding["source_revision"],
            "status": status,
            "current_record_id": binding["pass_record_id"] if unique else None,
            "current_record_digest": binding["pass_record_digest"] if unique else None,
            "projection_head_digest": "9" * 64 if unique else None,
            "trust_domain": "yolozu_managed" if status != "untrusted" else "unknown",
            "observation_digest": "a" * 64,
        }
    _redigest(record, "observation_digest")
    return record


def _support(
    bundle: dict,
    *,
    status: str = "matching_one",
    channel: str = "Experimental",
    trust_domain: str | None = None,
) -> dict:
    matching = status == "matching_one"
    if trust_domain is None:
        trust_domain = "yolozu_managed" if matching else "unknown"
    record = {
        "schema_version": 1,
        "provider_id": "support-profile-projection",
        "provider_version": "1",
        "family_id": bundle["family_id"],
        "bundle_spec_digest": bundle["spec_digest"],
        "channel": channel,
        "lifecycle_assignment_id": "assignment-1",
        "lifecycle_assignment_digest": "1" * 64,
        "support_profile_index_head_digest": "2" * 64,
        "profile_set_record_id": "profile-set-1",
        "profile_set_record_digest": "3" * 64,
        "profile_set_digest": "4" * 64,
        "status": status,
        "profile_id": "cpu-batch" if matching else None,
        "profile_digest": "5" * 64 if matching else None,
        "environment_fingerprint": "6" * 64 if matching else None,
        "qualification_workload_fingerprint": "7" * 64 if matching else None,
        "protocol_fingerprint": "8" * 64 if matching else None,
        "advertised_gates_digest": "9" * 64 if matching else None,
        "trust_domain": trust_domain,
        "observation_digest": "a" * 64,
    }
    _redigest(record, "observation_digest")
    return record


def _evidence(*, trust_domain: str = "yolozu_managed") -> dict:
    return {
        "activation_record_id": "activation-1",
        "activation_record_digest": "a" * 64,
        "report_id": "report-1",
        "report_digest": "b" * 64,
        "trust_domain": trust_domain,
    }


def _candidate(
    bundle: dict,
    *,
    rank_state: str = "selected",
    reason: str = "support_profile_mismatch",
) -> dict:
    eligible = rank_state != "excluded"
    return {
        "family_id": bundle["family_id"],
        "bundle_id": bundle["bundle_id"],
        "bundle_version": bundle["bundle_version"],
        "spec_digest": bundle["spec_digest"],
        "artifact_set_digest": bundle["artifact_set_digest"],
        "effective_channel": "Experimental",
        "pointed_channels": ["Experimental"],
        "matching_channels": ["Experimental"] if eligible else [],
        "screening_observation": _screening(bundle),
        "support_profile_observation": (
            _support(bundle) if eligible else _support(bundle, status="absent")
        ),
        "artifact_state_fingerprint": "c" * 64,
        "class_mapping": build_fixed_class_mapping(
            validate_algorithm_bundle_spec(bundle), ["cat"]
        ),
        "evidence": _evidence() if eligible else None,
        "support_scope": "public_qualified" if eligible else "none",
        "rank_state": rank_state,
        "rank_position": 1 if rank_state == "selected" else (2 if eligible else None),
        "reason_codes": [] if eligible else [reason],
        "reason_details": [],
        "human_summary": {
            "selected": "Selected after all required checks passed.",
            "eligible_not_selected": "Eligible, but another candidate ranked first.",
            "excluded": "Excluded because one or more required checks failed.",
        }[rank_state],
        "ranking_trace": (
            [{"step": 1, "status": "pass", "reason_code": None, "detail": None}]
            if eligible
            else [{"step": 1, "status": "failed", "reason_code": reason, "detail": None}]
        ),
    }


def _decision(bundle: dict, *, selected: bool = True) -> dict:
    candidate = _candidate(bundle, rank_state="selected" if selected else "excluded")
    selected_bundle = {
        key: candidate[key]
        for key in (
            "family_id",
            "bundle_id",
            "bundle_version",
            "spec_digest",
            "artifact_set_digest",
            "effective_channel",
        )
    }
    record = {
        "schema_version": 1,
        "decision_id": "decision-1",
        "status": "selected" if selected else "abstained",
        "decided_at": "2026-08-25T06:30:00Z",
        "local_job_digest": "1" * 64,
        "local_input_digest": "2" * 64,
        "artifact_resolver_state_digest": "3" * 64,
        "environment_fingerprint": "6" * 64,
        "qualification_workload_fingerprint": "7" * 64,
        "protocol_fingerprint": "8" * 64,
        "advertised_gates_digest": "9" * 64,
        "registry_id": "yolozu-bundle-registry-v1",
        "registry_digest": _registry_payload(bundle)["registry_digest"],
        "registry_trust_domain": "yolozu_managed",
        "lifecycle_projection_digest": "4" * 64,
        "lifecycle_trust_domain": "yolozu_managed",
        "ranking_policy": "latency_first",
        "prompt_mode": "fixed_classes",
        "registry_bundle_count": 1,
        "selected_bundle": selected_bundle if selected else None,
        "selected_evidence": candidate["evidence"] if selected else None,
        "selected_artifact_state_fingerprint": (
            candidate["artifact_state_fingerprint"] if selected else None
        ),
        "selected_class_mapping": candidate["class_mapping"] if selected else None,
        "support_scope": "public_qualified" if selected else "none",
        "reason_codes": [] if selected else ["no_eligible_candidate"],
        "human_summary": (
            "Selected one qualified bundle after all required checks passed."
            if selected
            else "No eligible qualified bundle matched all required checks."
        ),
        "candidate_evaluations": [candidate],
        "selection_trace": (
            [{"rank_position": 1, "spec_digest": bundle["spec_digest"]}]
            if selected
            else []
        ),
        "decision_digest": "d" * 64,
    }
    _redigest(record, "decision_digest")
    return record


class TestAdaptiveSelectionContracts(unittest.TestCase):
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
                "screening_eligibility_observation",
                "support_profile_eligibility_observation",
                "selection_decision",
            )
        }

    def test_selected_and_abstained_decisions_are_complete_and_schema_valid(self) -> None:
        bundle = _bundle_payload()
        registry = validate_algorithm_bundle_registry(_registry_payload(bundle))
        for selected in (True, False):
            validated = validate_selection_decision(
                _decision(bundle, selected=selected),
                expected_registry=registry,
                as_of=AS_OF,
            )
            payload = validated.to_dict()
            self.assertTrue(
                _schema_accepts(
                    payload,
                    self.schemas["selection_decision"],
                    root=self.schemas["selection_decision"],
                )
            )
            self.assertLessEqual(len(json.dumps(payload).encode()), MAX_SELECTION_DECISION_BYTES)

    def test_screening_observations_bind_exact_bundle_and_never_read_files(self) -> None:
        existing = _bundle_payload()
        observed = _screening(existing)
        with patch("builtins.open", side_effect=AssertionError("unexpected file read")):
            validated = validate_screening_eligibility_observation(
                observed,
                bundle=existing,
                source_trust_domain="unknown",
            )
        self.assertTrue(
            _schema_accepts(
                validated.to_dict(),
                self.schemas["screening_eligibility_observation"],
                root=self.schemas["screening_eligibility_observation"],
            )
        )

        screened = _bundle_payload()
        screened["provenance_class"] = "screened_candidate"
        screened["screening_binding"] = {
            "stream_key": "candidate-stream-1",
            "pass_record_id": "pass-1",
            "pass_record_digest": "e" * 64,
            "source_revision": "revision-1",
        }
        _redigest(screened, "spec_digest")
        for status in (
            "current_pass",
            "current_hold",
            "current_reject",
            "absent",
            "untrusted",
            "conflict",
            "revision_mismatch",
        ):
            validate_screening_eligibility_observation(
                _screening(screened, status=status), bundle=screened
            )
        mismatch = _screening(screened, status="current_pass")
        mismatch["source_revision"] = "different"
        _redigest(mismatch, "observation_digest")
        with self.assertRaisesRegex(ValueError, "bundle binding mismatch"):
            validate_screening_eligibility_observation(mismatch, bundle=screened)

    def test_support_observations_cover_public_and_site_semantics_without_reads(self) -> None:
        bundle = _bundle_payload()
        matching = _support(bundle)
        with patch("builtins.open", side_effect=AssertionError("unexpected file read")):
            validated = validate_support_profile_eligibility_observation(
                matching,
                source_trust_domain="yolozu_managed",
                expected_family_id=bundle["family_id"],
                expected_spec_digest=bundle["spec_digest"],
                expected_channel="Experimental",
                expected_environment_fingerprint="6" * 64,
                expected_workload_fingerprint="7" * 64,
                expected_protocol_fingerprint="8" * 64,
                expected_advertised_gates_digest="9" * 64,
            )
        self.assertTrue(
            _schema_accepts(
                validated.to_dict(),
                self.schemas["support_profile_eligibility_observation"],
                root=self.schemas["support_profile_eligibility_observation"],
            )
        )
        for status in ("no_match", "absent", "untrusted", "conflict"):
            validate_support_profile_eligibility_observation(_support(bundle, status=status))
        for field, expected in (
            ("environment_fingerprint", "6" * 64),
            ("qualification_workload_fingerprint", "7" * 64),
            ("protocol_fingerprint", "8" * 64),
            ("advertised_gates_digest", "9" * 64),
        ):
            mismatch = _support(bundle)
            mismatch[field] = "f" * 64
            _redigest(mismatch, "observation_digest")
            keyword = {
                "environment_fingerprint": "expected_environment_fingerprint",
                "qualification_workload_fingerprint": "expected_workload_fingerprint",
                "protocol_fingerprint": "expected_protocol_fingerprint",
                "advertised_gates_digest": "expected_advertised_gates_digest",
            }[field]
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                validate_support_profile_eligibility_observation(
                    mismatch, **{keyword: expected}
                )
        site = _support(bundle, status="not_required_site")
        validate_support_profile_eligibility_observation(
            site,
            evidence_trust_domain="site_managed",
            support_scope="site_qualified",
        )
        with self.assertRaisesRegex(ValueError, "site_managed"):
            validate_support_profile_eligibility_observation(site)

    def test_contradictory_selection_and_hidden_failures_are_rejected(self) -> None:
        bundle = _bundle_payload()
        registry = _registry_payload(bundle)
        for reason in (
            "evidence_expired",
            "evidence_superseded",
            "evidence_inactive",
            "evidence_revoked",
            "evidence_future_dated",
            "registry_untrusted",
            "lifecycle_untrusted",
            "evidence_untrusted",
            "evidence_conflict",
            "cold_start_unknown",
            "cold_start_above_requirement",
        ):
            case = _decision(bundle)
            candidate = case["candidate_evaluations"][0]
            candidate["reason_codes"] = [reason]
            candidate["ranking_trace"] = [
                {"step": 1, "status": "failed", "reason_code": reason, "detail": None}
            ]
            _redigest(case, "decision_digest")
            with self.assertRaisesRegex(ValueError, "hard-failure"):
                validate_selection_decision(case, expected_registry=registry, as_of=AS_OF)

        case = _decision(bundle, selected=False)
        case["selected_bundle"] = {
            key: _decision(bundle)["selected_bundle"][key]
            for key in _decision(bundle)["selected_bundle"]
        }
        _redigest(case, "decision_digest")
        with self.assertRaisesRegex(ValueError, "null selected identities"):
            validate_selection_decision(case, as_of=AS_OF)

    def test_registry_completeness_order_and_selected_identity_are_enforced(self) -> None:
        first = _bundle_payload(version="1.0")
        second = _bundle_payload(version="2.0")
        registry = _registry_payload(first, second)
        omitted = _decision(first)
        omitted["registry_digest"] = registry["registry_digest"]
        _redigest(omitted, "decision_digest")
        with self.assertRaisesRegex(ValueError, "incomplete expected registry"):
            validate_selection_decision(omitted, expected_registry=registry, as_of=AS_OF)

        case = _decision(first)
        case["selected_evidence"] = copy.deepcopy(case["selected_evidence"])
        case["selected_evidence"]["report_digest"] = "f" * 64
        _redigest(case, "decision_digest")
        with self.assertRaisesRegex(ValueError, "selected_evidence"):
            validate_selection_decision(case, as_of=AS_OF)

    def test_fixed_class_mapping_is_exact_and_inventory_digest_is_forbidden(self) -> None:
        bundle = _bundle_payload()
        case = _decision(bundle)
        mapping = case["candidate_evaluations"][0]["class_mapping"]
        mapping["requested_labels"] = ["kitty"]
        _redigest(case, "decision_digest")
        with self.assertRaises(ValueError):
            validate_selection_decision(
                case, expected_registry=_registry_payload(bundle), as_of=AS_OF
            )

        case = _decision(bundle)
        case["inventory_digest"] = "f" * 64
        _redigest(case, "decision_digest")
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            validate_selection_decision(case, as_of=AS_OF)

    def test_reason_detail_and_trace_boundaries_fail_without_truncation(self) -> None:
        bundle = _bundle_payload()
        reasons = sorted(CANDIDATE_REASON_CODES, key=lambda item: item.encode("ascii"))[:32]
        case = _decision(bundle, selected=False)
        candidate = case["candidate_evaluations"][0]
        candidate["reason_codes"] = reasons
        candidate["reason_details"] = [
            {"reason_code": reason, "detail": "x" * 256} for reason in reasons
        ]
        candidate["ranking_trace"] = [
            {
                "step": index + 1,
                "status": "failed" if index == 0 else "pass",
                "reason_code": reasons[0] if index == 0 else None,
                "detail": None,
            }
            for index in range(32)
        ]
        _redigest(case, "decision_digest")
        validate_selection_decision(case, as_of=AS_OF)

        too_many_reasons = copy.deepcopy(case)
        too_many_reasons["candidate_evaluations"][0]["reason_codes"] = sorted(
            CANDIDATE_REASON_CODES, key=lambda item: item.encode("ascii")
        )[:33]
        _redigest(too_many_reasons, "decision_digest")
        with self.assertRaisesRegex(ValueError, "0..32"):
            validate_selection_decision(too_many_reasons, as_of=AS_OF)

        oversized_detail = copy.deepcopy(case)
        oversized_detail["candidate_evaluations"][0]["reason_details"][0][
            "detail"
        ] = "x" * 257
        _redigest(oversized_detail, "decision_digest")
        with self.assertRaisesRegex(ValueError, "invalid identifier"):
            validate_selection_decision(oversized_detail, as_of=AS_OF)

        too_many_trace = copy.deepcopy(case)
        too_many_trace["candidate_evaluations"][0]["ranking_trace"].append(
            {"step": 32, "status": "pass", "reason_code": None, "detail": None}
        )
        _redigest(too_many_trace, "decision_digest")
        with self.assertRaisesRegex(ValueError, "0..32"):
            validate_selection_decision(too_many_trace, as_of=AS_OF)

        too_many_candidates = _decision(bundle, selected=False)
        too_many_candidates["registry_bundle_count"] = 128
        too_many_candidates["candidate_evaluations"] *= 129
        _redigest(too_many_candidates, "decision_digest")
        with self.assertRaisesRegex(ValueError, "0..128"):
            validate_selection_decision(too_many_candidates, as_of=AS_OF)

    def test_untrusted_prose_urls_and_paths_cannot_enter_explanations(self) -> None:
        bundle = _bundle_payload()
        for detail in (
            "runner said arbitrary prose",
            "https://example.invalid/error",
            "/Users/operator/private/file",
        ):
            case = _decision(bundle, selected=False)
            case["candidate_evaluations"][0]["reason_details"] = [
                {"reason_code": "support_profile_mismatch", "detail": detail}
            ]
            _redigest(case, "decision_digest")
            with self.assertRaisesRegex(ValueError, "invalid identifier"):
                validate_selection_decision(case, as_of=AS_OF)

        case = _decision(bundle, selected=False)
        case["human_summary"] = "Copied runner output"
        _redigest(case, "decision_digest")
        with self.assertRaisesRegex(ValueError, "fixed template"):
            validate_selection_decision(case, as_of=AS_OF)

    def test_sensitive_digests_are_required_and_bind_the_decision(self) -> None:
        bundle = _bundle_payload()
        original = _decision(bundle)
        validated = validate_selection_decision(original, as_of=AS_OF)
        for field in (
            "local_job_digest",
            "local_input_digest",
            "artifact_resolver_state_digest",
        ):
            missing = copy.deepcopy(original)
            missing.pop(field)
            with self.assertRaisesRegex(ValueError, "missing required keys"):
                validate_selection_decision(missing, as_of=AS_OF)
            changed = copy.deepcopy(original)
            changed[field] = "e" * 64
            _redigest(changed, "decision_digest")
            other = validate_selection_decision(changed, as_of=AS_OF)
            self.assertNotEqual(validated.decision_digest, other.decision_digest)

            exported = _report_payload()
            exported[field] = original[field]
            exported["report_digest"] = canonical_sha256_v1(
                exported, own_digest_field="report_digest"
            )
            with self.assertRaisesRegex(ValueError, "unknown keys"):
                validate_qualification_report(exported, as_of=AS_OF)

    def test_total_decision_limit_fails_instead_of_dropping_candidates(self) -> None:
        bundle = _bundle_payload()
        reasons = sorted(CANDIDATE_REASON_CODES, key=lambda item: item.encode("ascii"))[:32]
        candidates = []
        for index in range(128):
            current = copy.deepcopy(bundle)
            current["bundle_id"] = f"bundle-{index:03d}"
            _redigest(current, "spec_digest")
            candidate = _candidate(current, rank_state="excluded", reason=reasons[0])
            candidate["reason_codes"] = reasons
            candidate["reason_details"] = [
                {"reason_code": reason, "detail": "x" * 256} for reason in reasons
            ]
            candidates.append(candidate)
        case = _decision(bundle, selected=False)
        case["registry_bundle_count"] = 128
        case["candidate_evaluations"] = candidates
        _redigest(case, "decision_digest")
        with self.assertRaisesRegex(ValueError, "registry/evidence_limit_exceeded"):
            validate_selection_decision(case, as_of=AS_OF)


if __name__ == "__main__":
    unittest.main()
