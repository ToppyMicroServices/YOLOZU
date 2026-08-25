from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from importlib import resources
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tests.test_adaptive_bundle_contracts import (
    _bundle_payload,
    _lifecycle_event,
    _registry_payload,
    _support_profile_payload,
    _support_record,
)
from yolozu.adaptive import (
    CODE_OWNED_RUNNER_IDS,
    AlgorithmRunner,
    RunnerProbeResult,
    canonical_sha256_v1,
    load_algorithm_bundle_registry,
    project_support_profiles,
)
from yolozu.adaptive.control_records import (
    MAX_CONTROL_INTEGER_BYTES,
    MAX_CONTROL_KEY_BYTES,
    MAX_CONTROL_NODES,
    MAX_CONTROL_RECORD_BYTES,
    MAX_CONTROL_STRING_BYTES,
    load_bounded_json_bytes,
)


def _write_custom(
    workspace: Path,
    registry: dict[str, Any],
    lifecycle: list[dict[str, Any]] | None = None,
) -> Path:
    root = workspace / "adaptive-catalog"
    root.mkdir()
    root.joinpath("bundle_specs.json").write_bytes(
        json.dumps(registry, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    events = lifecycle or []
    root.joinpath("bundle_lifecycle.jsonl").write_bytes(
        b"".join(
            json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            + b"\n"
            for event in events
        )
    )
    return root


class TestAdaptiveBundleRegistry(unittest.TestCase):
    def test_packaged_empty_ssot_loads_without_selectable_bundle(self) -> None:
        loaded = load_algorithm_bundle_registry()
        self.assertEqual(loaded.source_kind, "packaged_ssot")
        self.assertEqual(loaded.registry_trust_domain, "yolozu_managed")
        self.assertEqual(loaded.lifecycle_trust_domain, "yolozu_managed")
        self.assertEqual(loaded.selection_trust_reason_codes, ())
        self.assertEqual(loaded.bundles, ())
        self.assertEqual(loaded.lifecycle.events, ())

        checkout = (
            Path(__file__).resolve().parents[1] / "yolozu" / "data" / "adaptive_routing"
        )
        packaged = resources.files("yolozu.data").joinpath("adaptive_routing")
        for basename in ("bundle_specs.json", "bundle_lifecycle.jsonl"):
            self.assertEqual(
                checkout.joinpath(basename).read_bytes(),
                packaged.joinpath(basename).read_bytes(),
            )

    def test_custom_catalog_is_deterministic_and_always_untrusted(self) -> None:
        first = _bundle_payload(version="1.0-rc01")
        second = _bundle_payload(version="1.00")
        registry = _registry_payload(second, first)
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            _write_custom(workspace, registry)
            loaded = load_algorithm_bundle_registry(
                workspace_root=workspace,
                custom_registry_root=Path("adaptive-catalog"),
            )

        self.assertEqual(loaded.source_kind, "workspace_custom")
        self.assertEqual(loaded.registry_trust_domain, "operator_asserted")
        self.assertEqual(loaded.lifecycle_trust_domain, "operator_asserted")
        self.assertEqual(
            loaded.selection_trust_reason_codes,
            ("registry_untrusted", "lifecycle_untrusted"),
        )
        self.assertEqual(
            [bundle.to_dict()["bundle_version"] for bundle in loaded.bundles],
            ["1.0-rc01", "1.00"],
        )
        changed = loaded.bundles[0].to_dict()
        changed["bundle_version"] = "changed"
        self.assertEqual(loaded.bundles[0].to_dict()["bundle_version"], "1.0-rc01")
        with self.assertRaises(FrozenInstanceError):
            loaded.source_kind = "packaged_ssot"  # type: ignore[misc]

    def test_custom_path_and_file_symlinks_fail_closed(self) -> None:
        registry = _registry_payload()
        with (
            tempfile.TemporaryDirectory() as temporary,
            tempfile.TemporaryDirectory() as outside,
        ):
            workspace = Path(temporary)
            external = Path(outside)
            external_root = _write_custom(external, registry)
            with self.assertRaisesRegex(ValueError, "inside workspace"):
                load_algorithm_bundle_registry(
                    workspace_root=workspace,
                    custom_registry_root=external_root,
                )

            linked_root = workspace / "linked-root"
            linked_root.symlink_to(external_root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                load_algorithm_bundle_registry(
                    workspace_root=workspace,
                    custom_registry_root=linked_root,
                )

            local_root = workspace / "local"
            local_root.mkdir()
            local_root.joinpath("bundle_specs.json").symlink_to(
                external_root / "bundle_specs.json"
            )
            local_root.joinpath("bundle_lifecycle.jsonl").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "unavailable"):
                load_algorithm_bundle_registry(
                    workspace_root=workspace,
                    custom_registry_root=local_root,
                )

    def test_entire_custom_registry_validates_before_return(self) -> None:
        valid = _bundle_payload()
        cases: list[dict[str, Any]] = []

        changed = copy.deepcopy(valid)
        changed["model_revision"] = "unbound-change"
        cases.append(_registry_payload(changed))

        unknown_runner = copy.deepcopy(valid)
        unknown_runner["runner_id"] = "caller.module:Runner"
        unknown_runner["spec_digest"] = canonical_sha256_v1(
            unknown_runner, own_digest_field="spec_digest"
        )
        cases.append(_registry_payload(unknown_runner))

        self.assertNotIn("caller.module:Runner", CODE_OWNED_RUNNER_IDS)
        for registry in cases:
            with self.subTest(registry_digest=registry["registry_digest"]):
                with tempfile.TemporaryDirectory() as temporary:
                    workspace = Path(temporary)
                    root = _write_custom(workspace, registry)
                    with self.assertRaises(ValueError):
                        load_algorithm_bundle_registry(
                            workspace_root=workspace,
                            custom_registry_root=root,
                        )

    def test_registry_and_lifecycle_share_one_total_input_cap(self) -> None:
        registry = _registry_payload()
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            root = _write_custom(workspace, registry)
            registry_size = root.joinpath("bundle_specs.json").stat().st_size
            root.joinpath("bundle_lifecycle.jsonl").write_bytes(b"{}\n")
            with (
                patch(
                    "yolozu.adaptive.bundle_registry.MAX_CONTROL_STREAM_BYTES",
                    registry_size,
                ),
                self.assertRaisesRegex(ValueError, "exceeds byte cap"),
            ):
                load_algorithm_bundle_registry(
                    workspace_root=workspace,
                    custom_registry_root=root,
                )

    def test_managed_lifecycle_rejects_untrusted_support_projection(self) -> None:
        profile = _support_profile_payload()
        definition = _support_record(
            sequence=1,
            previous="0" * 64,
            record_id="define-custom-profile",
            kind="profile_definition",
            variant={"profile": profile},
        )
        custom_support = project_support_profiles([definition])
        with self.assertRaisesRegex(ValueError, "managed support-profile trust"):
            load_algorithm_bundle_registry(support_profiles=custom_support)

        managed_support = project_support_profiles(
            [definition],
            source_trust_domain="yolozu_managed",
        )
        self.assertEqual(
            load_algorithm_bundle_registry(support_profiles=managed_support).bundles,
            (),
        )

    def test_lifecycle_failures_and_unknown_license_are_not_repaired(self) -> None:
        bundle = _bundle_payload()
        registry = _registry_payload(bundle)
        spec_digest = bundle["spec_digest"]
        artifact_set_digest = bundle["artifact_set_digest"]
        for review_state in ("unknown", "blocked"):
            reviews = [{"artifact_id": "model", "review_state": review_state}]
            registered = _lifecycle_event(
                sequence=1,
                previous="0" * 64,
                scope="bundle_global",
                event_type="register_global",
                variant={
                    "family_id": "example-detector",
                    "bundle_spec_digest": spec_digest,
                    "artifact_set_digest": artifact_set_digest,
                    "bundle_state": "enabled",
                    "artifact_license_reviews": reviews,
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
                    "target_bundle_spec_digest": spec_digest,
                    "target_artifact_set_digest": artifact_set_digest,
                    "target_artifact_license_reviews": reviews,
                    "support_profile_index_head": "0" * 64,
                    "profile_set_record_id": None,
                    "profile_set_record_digest": None,
                    "profile_set_digest": canonical_sha256_v1([]),
                    "profiles": [],
                    "evidence_bindings": [],
                },
            )
            with (
                self.subTest(review_state=review_state),
                tempfile.TemporaryDirectory() as temporary,
            ):
                workspace = Path(temporary)
                root = _write_custom(workspace, registry, [registered, candidate])
                loaded = load_algorithm_bundle_registry(
                    workspace_root=workspace,
                    custom_registry_root=root,
                )
                self.assertEqual(
                    loaded.lifecycle.bundle_states[spec_digest][
                        "artifact_license_reviews"
                    ],
                    reviews,
                )
                self.assertFalse(
                    loaded.is_lifecycle_eligible(
                        family_id="example-detector",
                        channel="Experimental",
                    )
                )
                self.assertEqual(
                    loaded.lifecycle.events[0].source_trust_domain,
                    "operator_asserted",
                )

        broken = copy.deepcopy(registered)
        broken["sequence"] = 2
        broken["event_digest"] = canonical_sha256_v1(
            broken, own_digest_field="event_digest"
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            root = _write_custom(workspace, registry, [broken])
            with self.assertRaises(ValueError):
                load_algorithm_bundle_registry(
                    workspace_root=workspace,
                    custom_registry_root=root,
                )

    def test_runner_protocol_has_no_dynamic_import_surface(self) -> None:
        class FakeRunner:
            runner_id = "onnxruntime"
            runner_version = "1"

            def probe(self, *, bundle: Any, environment: Any) -> RunnerProbeResult:
                return RunnerProbeResult("supported")

            def load(self, *, bundle: Any, artifacts: Any) -> None:
                return None

            def warmup(self, *, input_item: Any) -> None:
                return None

            def predict(
                self,
                *,
                input_item: Any,
                requested_labels: tuple[str, ...],
            ) -> tuple[dict[str, Any], ...]:
                return ()

            def close(self) -> None:
                return None

        self.assertIsInstance(FakeRunner(), AlgorithmRunner)
        self.assertEqual(
            RunnerProbeResult("unsupported", "runtime_absent").status, "unsupported"
        )
        with self.assertRaises(ValueError):
            RunnerProbeResult("supported", "should_not_exist")
        with self.assertRaises(ValueError):
            RunnerProbeResult("failed", "contains/path")

    def test_bounded_reader_enforces_exact_token_limits_before_dom_return(self) -> None:
        exact_key = "k" * MAX_CONTROL_KEY_BYTES
        self.assertEqual(
            load_bounded_json_bytes(json.dumps({exact_key: 1}).encode()),
            {exact_key: 1},
        )
        with self.assertRaisesRegex(ValueError, "object key exceeds"):
            load_bounded_json_bytes(
                json.dumps({"k" * (MAX_CONTROL_KEY_BYTES + 1): 1}).encode()
            )

        exact_string = b'"' + b"x" * MAX_CONTROL_STRING_BYTES + b'"'
        self.assertEqual(
            len(load_bounded_json_bytes(exact_string)), MAX_CONTROL_STRING_BYTES
        )
        with self.assertRaisesRegex(ValueError, "string exceeds"):
            load_bounded_json_bytes(b'"' + b"x" * (MAX_CONTROL_STRING_BYTES + 1) + b'"')

        exact_nodes = b"[" + b",".join([b"0"] * MAX_CONTROL_NODES) + b"]"
        self.assertEqual(len(load_bounded_json_bytes(exact_nodes)), MAX_CONTROL_NODES)
        with self.assertRaisesRegex(ValueError, "node limit"):
            load_bounded_json_bytes(
                b"[" + b",".join([b"0"] * (MAX_CONTROL_NODES + 1)) + b"]"
            )

        exact_integer = b"1" * MAX_CONTROL_INTEGER_BYTES
        self.assertEqual(load_bounded_json_bytes(exact_integer), int(exact_integer))
        with self.assertRaisesRegex(ValueError, "integer token"):
            load_bounded_json_bytes(b"1" * (MAX_CONTROL_INTEGER_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "negative zero"):
            load_bounded_json_bytes(b"-0")

        exact_record = b"{}" + b" " * (MAX_CONTROL_RECORD_BYTES - 2)
        self.assertEqual(load_bounded_json_bytes(exact_record), {})
        with self.assertRaisesRegex(ValueError, "exceeds 4 MiB"):
            load_bounded_json_bytes(exact_record + b" ")
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            load_bounded_json_bytes(b'"\xff"')

    def test_import_has_no_runtime_or_filesystem_side_effect(self) -> None:
        root = Path(__file__).resolve().parents[1]
        code = """
from pathlib import Path
import sys
import yolozu.adaptive.bundle_registry as registry
heavy = {'torch', 'onnxruntime', 'tensorrt', 'cv2', 'coremltools'}
assert not (heavy & set(sys.modules))
assert list(Path('.').iterdir()) == []
assert registry.load_algorithm_bundle_registry().bundles == ()
"""
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(root)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [sys.executable, "-B", "-c", code],
                cwd=temporary,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
