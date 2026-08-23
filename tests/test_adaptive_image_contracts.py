from __future__ import annotations

import copy
import json
import re
import unittest
from pathlib import Path
from typing import Any

from yolozu.adaptive.canonical import (
    canonical_decimal_v1,
    canonical_json_v1,
    canonical_sha256_v1,
)
from yolozu.adaptive.contracts import (
    build_qualification_workload_profile,
    compute_environment_fingerprint,
    compute_workload_fingerprint,
    validate_environment_profile,
    validate_image_job_spec,
    validate_qualification_workload_profile,
)
from yolozu.adaptive.inventory import DecodedInputInventory, DecodedInputObservation


class _SchemaFailure(AssertionError):
    pass


def _schema_validate(value: Any, schema: dict[str, Any], *, root: dict[str, Any], where: str = "$") -> None:
    if "$ref" in schema:
        reference = schema["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/"):
            raise _SchemaFailure(f"{where}: unsupported test reference")
        target: Any = root
        for component in reference[2:].split("/"):
            target = target[component.replace("~1", "/").replace("~0", "~")]
        _schema_validate(value, target, root=root, where=where)

    for child in schema.get("allOf", []):
        _schema_validate(value, child, root=root, where=where)
    if "anyOf" in schema:
        if not any(_schema_accepts(value, child, root=root) for child in schema["anyOf"]):
            raise _SchemaFailure(f"{where}: no anyOf branch matched")
    if "oneOf" in schema:
        matches = sum(_schema_accepts(value, child, root=root) for child in schema["oneOf"])
        if matches != 1:
            raise _SchemaFailure(f"{where}: expected one oneOf branch, got {matches}")
    if "not" in schema and _schema_accepts(value, schema["not"], root=root):
        raise _SchemaFailure(f"{where}: matched forbidden schema")
    if "if" in schema and _schema_accepts(value, schema["if"], root=root):
        if "then" in schema:
            _schema_validate(value, schema["then"], root=root, where=where)
    elif "else" in schema:
        _schema_validate(value, schema["else"], root=root, where=where)

    if "const" in schema:
        expected_const = schema["const"]
        if value != expected_const or (
            isinstance(expected_const, (bool, int)) and type(value) is not type(expected_const)
        ):
            raise _SchemaFailure(f"{where}: const mismatch")
    if "enum" in schema and value not in schema["enum"]:
        raise _SchemaFailure(f"{where}: enum mismatch")

    expected_type = schema.get("type")
    if expected_type is not None:
        types = [expected_type] if isinstance(expected_type, str) else list(expected_type)

        def matches(name: str) -> bool:
            return {
                "object": isinstance(value, dict),
                "array": isinstance(value, list),
                "string": isinstance(value, str),
                "integer": isinstance(value, int) and not isinstance(value, bool),
                "boolean": isinstance(value, bool),
                "null": value is None,
            }.get(name, False)

        if not any(matches(name) for name in types):
            raise _SchemaFailure(f"{where}: type mismatch")

    if isinstance(value, dict):
        missing = sorted(set(schema.get("required", [])) - set(value))
        if missing:
            raise _SchemaFailure(f"{where}: missing {missing}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                raise _SchemaFailure(f"{where}: unknown {unknown}")
        for key, item in value.items():
            if key in properties:
                _schema_validate(item, properties[key], root=root, where=f"{where}.{key}")

    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            raise _SchemaFailure(f"{where}: too few items")
        maximum = schema.get("maxItems")
        if isinstance(maximum, int) and len(value) > maximum:
            raise _SchemaFailure(f"{where}: too many items")
        if schema.get("uniqueItems"):
            encoded = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in value]
            if len(encoded) != len(set(encoded)):
                raise _SchemaFailure(f"{where}: duplicate items")
        if isinstance(schema.get("items"), dict):
            for index, item in enumerate(value):
                _schema_validate(item, schema["items"], root=root, where=f"{where}[{index}]")
        if "contains" in schema and not any(
            _schema_accepts(item, schema["contains"], root=root) for item in value
        ):
            raise _SchemaFailure(f"{where}: contains did not match")

    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            raise _SchemaFailure(f"{where}: string too short")
        maximum = schema.get("maxLength")
        if isinstance(maximum, int) and len(value) > maximum:
            raise _SchemaFailure(f"{where}: string too long")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            raise _SchemaFailure(f"{where}: pattern mismatch")

    if isinstance(value, int) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, int) and value < minimum:
            raise _SchemaFailure(f"{where}: below minimum")
        if isinstance(maximum, int) and value > maximum:
            raise _SchemaFailure(f"{where}: above maximum")


def _schema_accepts(value: Any, schema: dict[str, Any], *, root: dict[str, Any]) -> bool:
    try:
        _schema_validate(value, schema, root=root)
    except _SchemaFailure:
        return False
    return True


class TestAdaptiveImageContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.schemas = {
            name: json.loads(
                (cls.repo_root / "docs" / "schemas" / f"{name}.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            for name in ("image_job_spec", "qualification_workload_profile", "environment_profile")
        }

    def _job_payload(self, *, phrase: str = "cat") -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task": "object_detection",
            "prompt_mode": "fixed_classes",
            "fixed_classes": [phrase],
            "input_mode": "single_image",
            "execution_mode": "batch",
            "batch_size": 1,
            "concurrency": 1,
            "max_images": 1,
            "max_results_per_image": 100,
            "job_timeout_seconds": 60,
            "ranking_policy": "latency_first",
            "allowed_maturities": ["Stable"],
            "network_policy": "deny",
            "compute_policy": "auto",
            "provider_allowlist": [],
            "precision_allowlist": [],
            "spdx_allowlist": [],
            "max_cold_start_ms": "500",
            "max_p95_latency_ms": "50.5",
            "min_repeat_throughput_fps": "1",
        }

    def _inventory(self) -> DecodedInputInventory:
        return DecodedInputInventory(
            input_mode="single_image",
            input_count=1,
            input_order="single_image_v1",
            inputs=(DecodedInputObservation(index=0, width=64, height=32, color_mode="RGB"),),
            decoder_id="pillow",
            decoder_version="12.3.0",
            source_total_bytes=100,
            local_input_digest="a" * 64,
        )

    def _environment_payload(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": 1,
            "collector_id": "yolozu_environment",
            "collector_version": "1",
            "collected_at": "2026-08-23T00:00:00Z",
            "os": {
                "probe_status": "present",
                "name": "Darwin",
                "version": "25.6.0",
                "architecture": "arm64",
            },
            "cpu": {
                "probe_status": "present",
                "model": "Apple M1",
                "logical_cores": {"probe_status": "present", "value": 8},
                "physical_cores": {"probe_status": "present", "value": 8},
            },
            "total_memory": {"probe_status": "present", "value_bytes": 17179869184},
            "accelerators": [
                {
                    "accelerator_id": "mps",
                    "probe_status": "present",
                    "kind": "gpu",
                    "vendor": "Apple",
                    "model": "M1",
                    "device_count": 1,
                    "memory": {"probe_status": "unsupported"},
                },
                {"accelerator_id": "cuda", "probe_status": "absent"},
            ],
            "runtimes": [
                {
                    "runtime_id": "torch",
                    "probe_status": "present",
                    "version": "2.10.0",
                    "provider_ids": ["cpu", "mps"],
                },
                {"runtime_id": "onnxruntime", "probe_status": "absent"},
            ],
            "power_performance_mode": {"probe_status": "unsupported"},
            "probe_issues": [
                {"probe_id": "power_mode", "status": "unsupported", "code": "not_exposed"}
            ],
        }
        fingerprint_record = copy.deepcopy(record)
        fingerprint_record["accelerators"].sort(key=lambda item: item["accelerator_id"])
        fingerprint_record["runtimes"].sort(key=lambda item: item["runtime_id"])
        for runtime in fingerprint_record["runtimes"]:
            if "provider_ids" in runtime:
                runtime["provider_ids"].sort()
        record["environment_fingerprint"] = compute_environment_fingerprint(fingerprint_record)
        return record

    def test_canonical_json_and_decimal_vectors(self) -> None:
        self.assertEqual(
            canonical_json_v1({"b": 1, "a": "é\n"}),
            b'{"a":"\xc3\xa9\\n","b":1}',
        )
        self.assertEqual(canonical_decimal_v1("0", field="value"), "0")
        self.assertEqual(canonical_decimal_v1("0.000000001", field="value"), "0.000000001")
        for invalid in ("-0", "+1", "01", "1.0", "1e2", "0.0000000001"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                canonical_decimal_v1(invalid, field="value")
        with self.assertRaises(ValueError):
            canonical_json_v1({"value": 1.0})
        with self.assertRaises(ValueError):
            canonical_json_v1({"value": "\ud800"})

    def test_job_normalizes_prompt_and_binds_sensitive_digest(self) -> None:
        payload = self._job_payload(phrase="  Ｃａｔ  ")
        job = validate_image_job_spec(payload)
        self.assertEqual(job.prompt_phrases, ("Cat",))
        self.assertEqual(len(job.local_job_digest), 64)
        self.assertNotIn("local_job_digest", job.to_dict())
        self.assertTrue(_schema_accepts(job.to_dict(), self.schemas["image_job_spec"], root=self.schemas["image_job_spec"]))

        same = validate_image_job_spec(self._job_payload(phrase="Cat"))
        changed = validate_image_job_spec(self._job_payload(phrase="Dog"))
        self.assertEqual(job.local_job_digest, same.local_job_digest)
        self.assertNotEqual(job.local_job_digest, changed.local_job_digest)

    def test_job_rejects_semantic_and_cross_mode_errors(self) -> None:
        cases: list[tuple[str, dict[str, Any]]] = []
        duplicate = self._job_payload()
        duplicate["fixed_classes"] = ["Ａ", "A"]
        cases.append(("duplicate", duplicate))
        control = self._job_payload(phrase="cat\n")
        cases.append(("control", control))
        too_long = self._job_payload(phrase="x" * 257)
        cases.append(("phrase bound", too_long))
        total = self._job_payload()
        total["fixed_classes"] = [f"{index:03d}" + "x" * 253 for index in range(17)]
        cases.append(("payload bound", total))
        wrong_mode = self._job_payload()
        wrong_mode["min_sustained_fps"] = "1"
        cases.append(("mode metric", wrong_mode))
        memory_auto = self._job_payload()
        memory_auto["ranking_policy"] = "memory_first"
        cases.append(("memory auto", memory_auto))
        accuracy = self._job_payload()
        accuracy["ranking_policy"] = "accuracy_first"
        cases.append(("accuracy identity", accuracy))
        cpu_cuda = self._job_payload()
        cpu_cuda["compute_policy"] = "cpu_only"
        cpu_cuda["provider_allowlist"] = ["cuda"]
        cases.append(("compute contradiction", cpu_cuda))
        nonfinite = self._job_payload()
        nonfinite["quality_requirement"] = {
            "metric_id": "map",
            "direction": "higher_is_better",
            "threshold": "NaN",
            "evaluation_dataset_id": "coco_val2017",
            "evaluation_dataset_sha256": "a" * 64,
            "evaluation_protocol_sha256": "b" * 64,
            "evaluation_vocabulary_id": "coco80",
        }
        cases.append(("nonfinite", nonfinite))
        invalid_direction = copy.deepcopy(nonfinite)
        invalid_direction["quality_requirement"]["threshold"] = "0.5"
        invalid_direction["quality_requirement"]["direction"] = "smaller"
        cases.append(("invalid direction", invalid_direction))
        for name, payload in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_image_job_spec(payload)

    def test_job_schema_and_python_share_structural_acceptance(self) -> None:
        schema = self.schemas["image_job_spec"]
        valid = validate_image_job_spec(self._job_payload()).to_dict()
        self.assertTrue(_schema_accepts(valid, schema, root=schema))
        invalid_records = []
        for mutation in (
            lambda item: item.__setitem__("schema_version", 2),
            lambda item: item.__setitem__("batch_size", True),
            lambda item: item.__setitem__("unknown", 1),
            lambda item: item.__setitem__("min_sustained_fps", "1"),
            lambda item: item.__setitem__("ranking_policy", "memory_first"),
            lambda item: item.update(prompt_mode="text"),
        ):
            item = copy.deepcopy(valid)
            mutation(item)
            invalid_records.append(item)
        for record in invalid_records:
            self.assertFalse(_schema_accepts(record, schema, root=schema), record)
            with self.assertRaises(ValueError):
                validate_image_job_spec(record)

    def test_workload_fingerprint_uses_characteristics_not_prompt_words(self) -> None:
        inventory = self._inventory()
        cat_job = validate_image_job_spec(self._job_payload(phrase="cat"))
        dog_job = validate_image_job_spec(self._job_payload(phrase="dog"))
        cat = build_qualification_workload_profile(cat_job, inventory)
        dog = build_qualification_workload_profile(dog_job, inventory)
        self.assertNotEqual(cat_job.local_job_digest, dog_job.local_job_digest)
        self.assertEqual(cat.workload_fingerprint, dog.workload_fingerprint)
        serialized = json.dumps(cat.to_dict(), sort_keys=True)
        self.assertNotIn('"cat"', serialized)
        self.assertNotIn(cat_job.local_job_digest, serialized)
        self.assertNotIn(inventory.local_input_digest, serialized)
        self.assertNotIn("source_sha256", serialized)
        self.assertTrue(
            _schema_accepts(
                cat.to_dict(),
                self.schemas["qualification_workload_profile"],
                root=self.schemas["qualification_workload_profile"],
            )
        )

        changed_payload = self._job_payload(phrase="cat")
        changed_payload["provider_allowlist"] = ["cpu"]
        changed = build_qualification_workload_profile(
            validate_image_job_spec(changed_payload), inventory
        )
        self.assertNotEqual(cat.workload_fingerprint, changed.workload_fingerprint)

        threshold_only = self._job_payload(phrase="cat")
        threshold_only["max_cold_start_ms"] = "900"
        same_profile = build_qualification_workload_profile(
            validate_image_job_spec(threshold_only), inventory
        )
        self.assertEqual(cat.workload_fingerprint, same_profile.workload_fingerprint)

    def test_workload_validator_rejects_tamper_and_cross_mode(self) -> None:
        profile = build_qualification_workload_profile(
            validate_image_job_spec(self._job_payload()), self._inventory()
        ).to_dict()
        tampered = copy.deepcopy(profile)
        tampered["decoded_inputs"][0]["width"] = 65
        with self.assertRaisesRegex(ValueError, "workload_fingerprint"):
            validate_qualification_workload_profile(tampered)

        wrong_mode = copy.deepcopy(profile)
        wrong_mode["max_sustained_samples"] = 1_000_000
        with self.assertRaisesRegex(ValueError, "soft_realtime"):
            validate_qualification_workload_profile(wrong_mode)

        for field, value, message in (
            ("collector_id", "another_collector", "collector_id"),
            ("collector_version", "2", "collector_version"),
        ):
            unsupported = copy.deepcopy(profile)
            unsupported[field] = value
            unsupported["workload_fingerprint"] = compute_workload_fingerprint(
                unsupported
            )
            with self.assertRaisesRegex(ValueError, message):
                validate_qualification_workload_profile(unsupported)

        unsupported_decoder = copy.deepcopy(profile)
        unsupported_decoder["decoder"]["id"] = "another_decoder"
        unsupported_decoder["workload_fingerprint"] = compute_workload_fingerprint(
            unsupported_decoder
        )
        with self.assertRaisesRegex(ValueError, "decoder.id"):
            validate_qualification_workload_profile(unsupported_decoder)

    def test_workload_fingerprint_binds_every_execution_relevant_field(self) -> None:
        profile = build_qualification_workload_profile(
            validate_image_job_spec(self._job_payload()), self._inventory()
        ).to_dict()
        baseline = compute_workload_fingerprint(profile)
        mutations = (
            lambda item: item.__setitem__("execution_mode", "soft_realtime"),
            lambda item: item.__setitem__("compute_policy", "cpu_only"),
            lambda item: item.__setitem__("provider_allowlist", ["cpu"]),
            lambda item: item.__setitem__("precision_allowlist", ["fp32"]),
            lambda item: item["decoded_inputs"][0].__setitem__("width", 65),
            lambda item: item["decoder"].__setitem__("version", "12.4.0"),
            lambda item: item.__setitem__("batch_size", 2),
            lambda item: item.__setitem__("concurrency", 2),
            lambda item: item.__setitem__("max_results_per_image", 101),
            lambda item: item.__setitem__("latency_interval_id", "different_interval_v1"),
            lambda item: item["handoff"].__setitem__("id", "different_handoff_v1"),
            lambda item: item["prompt_characteristics"].__setitem__(
                "maximum_codepoint_bucket", "17-32"
            ),
        )
        for mutate in mutations:
            changed = copy.deepcopy(profile)
            mutate(changed)
            self.assertNotEqual(baseline, compute_workload_fingerprint(changed))

    def test_quality_identity_is_complete_but_threshold_is_not_workload_identity(self) -> None:
        payload = self._job_payload()
        payload["ranking_policy"] = "accuracy_first"
        payload["quality_requirement"] = {
            "metric_id": "coco_map",
            "direction": "higher_is_better",
            "threshold": "0.5",
            "evaluation_dataset_id": "coco_val2017",
            "evaluation_dataset_sha256": "a" * 64,
            "evaluation_protocol_sha256": "b" * 64,
            "evaluation_vocabulary_id": "coco80",
        }
        first = build_qualification_workload_profile(
            validate_image_job_spec(payload), self._inventory()
        )
        self.assertNotIn("threshold", first.to_dict()["quality_identity"])

        threshold_changed = copy.deepcopy(payload)
        threshold_changed["quality_requirement"]["threshold"] = "0.6"
        second = build_qualification_workload_profile(
            validate_image_job_spec(threshold_changed), self._inventory()
        )
        self.assertEqual(first.workload_fingerprint, second.workload_fingerprint)

        dataset_changed = copy.deepcopy(payload)
        dataset_changed["quality_requirement"]["evaluation_dataset_sha256"] = "c" * 64
        third = build_qualification_workload_profile(
            validate_image_job_spec(dataset_changed), self._inventory()
        )
        self.assertNotEqual(first.workload_fingerprint, third.workload_fingerprint)

    def test_workload_schema_and_python_share_structural_acceptance(self) -> None:
        schema = self.schemas["qualification_workload_profile"]
        valid = build_qualification_workload_profile(
            validate_image_job_spec(self._job_payload()), self._inventory()
        ).to_dict()
        self.assertTrue(_schema_accepts(valid, schema, root=schema))
        for mutation in (
            lambda item: item.__setitem__("schema_version", 2),
            lambda item: item.__setitem__("batch_size", False),
            lambda item: item.__setitem__("unknown", 1),
            lambda item: item.__setitem__("max_sustained_samples", 1_000_000),
        ):
            invalid = copy.deepcopy(valid)
            mutation(invalid)
            self.assertFalse(_schema_accepts(invalid, schema, root=schema), invalid)
            with self.assertRaises(ValueError):
                validate_qualification_workload_profile(invalid)

    def test_environment_fingerprint_excludes_time_and_probe_issues(self) -> None:
        first_payload = self._environment_payload()
        first = validate_environment_profile(first_payload)
        second_payload = copy.deepcopy(first_payload)
        second_payload["collected_at"] = "2026-08-23T01:00:00Z"
        second_payload["accelerators"].reverse()
        second_payload["runtimes"].reverse()
        second_payload["probe_issues"] = [
            {"probe_id": "runtime", "status": "failed", "code": "timeout"}
        ]
        second = validate_environment_profile(second_payload)
        self.assertEqual(first.environment_fingerprint, second.environment_fingerprint)
        self.assertEqual(
            first.environment_fingerprint,
            compute_environment_fingerprint(second_payload),
        )

        changed = first.to_dict()
        changed["runtimes"][1]["version"] = "2.11.0"
        changed["environment_fingerprint"] = compute_environment_fingerprint(changed)
        third = validate_environment_profile(changed)
        self.assertNotEqual(first.environment_fingerprint, third.environment_fingerprint)
        self.assertTrue(
            _schema_accepts(
                first.to_dict(),
                self.schemas["environment_profile"],
                root=self.schemas["environment_profile"],
            )
        )

    def test_environment_rejects_private_or_partial_facts(self) -> None:
        cases = []
        hostname = self._environment_payload()
        hostname["hostname"] = "private-host"
        cases.append(hostname)
        partial = self._environment_payload()
        partial["total_memory"] = {"probe_status": "present"}
        cases.append(partial)
        ip_value = self._environment_payload()
        ip_value["os"]["version"] = "127.0.0.1"
        cases.append(ip_value)
        uuid_value = self._environment_payload()
        uuid_value["cpu"]["model"] = "123e4567-e89b-12d3-a456-426614174000"
        cases.append(uuid_value)
        for payload in cases:
            with self.assertRaises(ValueError):
                validate_environment_profile(payload)

    def test_environment_schema_and_python_share_structural_acceptance(self) -> None:
        schema = self.schemas["environment_profile"]
        valid = validate_environment_profile(self._environment_payload()).to_dict()
        self.assertTrue(_schema_accepts(valid, schema, root=schema))
        invalid_records = []
        for mutation in (
            lambda item: item.__setitem__("schema_version", 2),
            lambda item: item.__setitem__("unknown", 1),
            lambda item: item.__setitem__("total_memory", {"probe_status": "present"}),
            lambda item: item["cpu"]["logical_cores"].__setitem__("value", True),
        ):
            item = copy.deepcopy(valid)
            mutation(item)
            invalid_records.append(item)
        for record in invalid_records:
            self.assertFalse(_schema_accepts(record, schema, root=schema), record)
            with self.assertRaises(ValueError):
                validate_environment_profile(record)

    def test_schema_copies_are_byte_identical_and_packaged(self) -> None:
        for name in self.schemas:
            canonical = self.repo_root / "docs" / "schemas" / f"{name}.schema.json"
            packaged = self.repo_root / "yolozu" / "data" / "schemas" / f"{name}.schema.json"
            self.assertEqual(canonical.read_bytes(), packaged.read_bytes())

    def test_canonical_digest_omits_only_own_field(self) -> None:
        record = {"record_digest": "f" * 64, "child_digest": "a" * 64, "value": 2}
        expected = canonical_sha256_v1({"child_digest": "a" * 64, "value": 2})
        self.assertEqual(
            canonical_sha256_v1(record, own_digest_field="record_digest"), expected
        )


if __name__ == "__main__":
    unittest.main()
