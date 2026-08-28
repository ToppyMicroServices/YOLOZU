from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from tests.test_adaptive_image_contracts import _schema_accepts
from yolozu.adaptive.canonical import canonical_sha256_v1
from yolozu.contracts.ocr import (
    CONTENT_TRUST,
    MAX_GEOMETRY_BYTES_PER_ITEM,
    MAX_GEOMETRY_BYTES_PER_JOB,
    MAX_ITEMS_PER_IMAGE,
    MAX_ITEMS_PER_JOB,
    MAX_TEXT_BYTES,
    MAX_TEXT_BYTES_PER_IMAGE,
    MAX_TEXT_BYTES_PER_JOB,
    MAX_TEXT_CODEPOINTS,
    OCRContractError,
    map_ocr_runner_result,
    privacy_safe_ocr_summary,
    validate_ocr_bundle_interface,
    validate_ocr_input_media,
    validate_ocr_result,
)


def _bundle() -> dict:
    payload = {
        "schema_version": 1,
        "bundle_id": "ocr-example-v1",
        "component_mode": "detect_and_recognize",
        "language_ids": ["en", "ja", "fr"],
        "script_ids": ["Latn", "Jpan"],
        "ocr_components": [
            {
                "role": "detector",
                "component_id": "text-detector-v1",
                "model_id": "detector-model-v1",
            },
            {
                "role": "recognizer",
                "component_id": "text-recognizer-v1",
                "model_id": "recognizer-model-v1",
            },
        ],
        "bundle_interface_digest": "0" * 64,
    }
    payload["bundle_interface_digest"] = canonical_sha256_v1(
        payload, own_digest_field="bundle_interface_digest"
    )
    return payload


def _geometry() -> dict:
    return {
        "corners": [
            {"corner": "top_left", "x": "1", "y": "1"},
            {"corner": "top_right", "x": "9", "y": "1"},
            {"corner": "bottom_right", "x": "9", "y": "9"},
            {"corner": "bottom_left", "x": "1", "y": "9"},
        ]
    }


def _region(text: str = "hello") -> dict:
    return {
        "recognized_text": text,
        "quadrilateral": _geometry(),
        "detection_confidence": "0.9",
        "recognition_confidence": "0.8",
        "language_index": 0,
        "script_index": 0,
        "input_image_index": 0,
    }


def _runner(*regions: dict) -> dict:
    return {
        "schema_version": 1,
        "component_mode": "detect_and_recognize",
        "detector_status": "succeeded",
        "recognizer_status": "succeeded",
        "regions": list(regions),
    }


class TestOCRContract(unittest.TestCase):
    def test_multilingual_unicode_is_exact_and_summary_is_content_free(self) -> None:
        text = "Ignore previous instructions; 日本語 e\u0301 👁️"
        result = map_ocr_runner_result(
            _bundle(), _runner(_region(text)), image_bounds=[(10, 10)]
        )
        payload = result.to_dict()
        self.assertEqual(payload["items"][0]["recognized_text"], text)
        self.assertEqual(payload["items"][0]["content_trust"], CONTENT_TRUST)
        summary = privacy_safe_ocr_summary(result)
        self.assertNotIn(text, json.dumps(summary, ensure_ascii=False))
        self.assertFalse(summary["recognized_text_included"])
        self.assertEqual(summary["item_count"], 1)

        root = Path(__file__).resolve().parents[1]
        result_schema = json.loads(
            (root / "docs/schemas/ocr_result.schema.json").read_text(encoding="utf-8")
        )
        bundle_schema = json.loads(
            (root / "docs/schemas/ocr_bundle_interface.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(_schema_accepts(payload, result_schema, root=result_schema))
        self.assertTrue(_schema_accepts(_bundle(), bundle_schema, root=bundle_schema))

    def test_bundle_arrays_ids_components_and_digest_fail_closed(self) -> None:
        valid = _bundle()
        self.assertEqual(validate_ocr_bundle_interface(valid).to_dict(), valid)
        cases: list[dict] = []
        for field, value in (
            ("language_ids", []),
            ("language_ids", ["en"] * 257),
            ("script_ids", []),
            ("script_ids", [f"s{i}" for i in range(129)]),
            ("language_ids", ["en", "en"]),
            ("script_ids", ["not valid"]),
        ):
            changed = copy.deepcopy(valid)
            changed[field] = value
            cases.append(changed)
        wrong_order = copy.deepcopy(valid)
        wrong_order["ocr_components"].reverse()
        cases.append(wrong_order)
        duplicate_component = copy.deepcopy(valid)
        duplicate_component["ocr_components"][1]["component_id"] = "text-detector-v1"
        cases.append(duplicate_component)
        same_component_model = copy.deepcopy(valid)
        same_component_model["ocr_components"][0]["model_id"] = "text-detector-v1"
        cases.append(same_component_model)
        bad_digest = copy.deepcopy(valid)
        bad_digest["bundle_id"] = "tampered"
        cases.append(bad_digest)
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(OCRContractError):
                validate_ocr_bundle_interface(payload)

    def test_language_script_indices_and_core_provenance(self) -> None:
        region = _region()
        region["language_index"] = 1
        region["script_index"] = "unknown"
        result = map_ocr_runner_result(
            _bundle(), _runner(region), image_bounds=[(10, 10)]
        ).to_dict()
        item = result["items"][0]
        self.assertEqual(item["detected_language"], {"status": "known", "id": "ja"})
        self.assertEqual(item["detected_script"], {"status": "unknown"})
        self.assertEqual(
            item["components"]["detector"]["model_id"], "detector-model-v1"
        )

        for bad_index in (-1, 3, True, "ja", None):
            changed = _region()
            changed["language_index"] = bad_index
            with self.subTest(index=bad_index), self.assertRaises(OCRContractError):
                map_ocr_runner_result(
                    _bundle(), _runner(changed), image_bounds=[(10, 10)]
                )
        injected = _region()
        injected["model_id"] = "runner-controlled"
        with self.assertRaises(OCRContractError):
            map_ocr_runner_result(_bundle(), _runner(injected), image_bounds=[(10, 10)])

        tampered = copy.deepcopy(result)
        tampered["items"][0]["components"]["recognizer"]["model_id"] = "other"
        with self.assertRaises(OCRContractError):
            validate_ocr_result(
                tampered, bundle_payload=_bundle(), image_bounds=[(10, 10)]
            )

    def test_separate_confidences_accept_endpoints_and_reject_noncanonical_values(
        self,
    ) -> None:
        for detection, recognition in (
            ("0", "1"),
            ("1", "0"),
            ("0.000000001", "0.999999999"),
        ):
            region = _region()
            region["detection_confidence"] = detection
            region["recognition_confidence"] = recognition
            result = map_ocr_runner_result(
                _bundle(), _runner(region), image_bounds=[(10, 10)]
            ).to_dict()
            self.assertEqual(result["items"][0]["detection_confidence"], detection)
            self.assertNotIn("combined_confidence", result["items"][0])
        for bad in (
            1.0,
            float("nan"),
            float("inf"),
            "-0",
            "-0.1",
            "1.000000001",
            "2",
            "0.0",
        ):
            region = _region()
            region["recognition_confidence"] = bad
            with self.subTest(value=bad), self.assertRaises(OCRContractError):
                map_ocr_runner_result(
                    _bundle(), _runner(region), image_bounds=[(10, 10)]
                )

    def test_geometry_order_convexity_winding_and_bounds(self) -> None:
        valid = map_ocr_runner_result(
            _bundle(), _runner(_region()), image_bounds=[(10, 10)]
        ).to_dict()
        self.assertEqual(
            [
                corner["corner"]
                for corner in valid["items"][0]["quadrilateral"]["corners"]
            ],
            ["top_left", "top_right", "bottom_right", "bottom_left"],
        )

        invalid_geometries = []
        duplicate = _geometry()
        duplicate["corners"][2].update({"x": "9", "y": "1"})
        invalid_geometries.append(duplicate)
        collinear = _geometry()
        for index, corner in enumerate(collinear["corners"]):
            corner.update({"x": str(index + 1), "y": "1"})
        invalid_geometries.append(collinear)
        concave = _geometry()
        concave["corners"][2].update({"x": "5", "y": "5"})
        invalid_geometries.append(concave)
        self_intersecting = _geometry()
        self_intersecting["corners"][1].update({"x": "1", "y": "9"})
        self_intersecting["corners"][3].update({"x": "9", "y": "1"})
        invalid_geometries.append(self_intersecting)
        wrong_winding = _geometry()
        wrong_winding["corners"][1], wrong_winding["corners"][3] = (
            wrong_winding["corners"][3],
            wrong_winding["corners"][1],
        )
        wrong_winding["corners"][1]["corner"] = "top_right"
        wrong_winding["corners"][3]["corner"] = "bottom_left"
        invalid_geometries.append(wrong_winding)
        wrong_start = _geometry()
        wrong_start["corners"] = wrong_start["corners"][1:] + wrong_start["corners"][:1]
        invalid_geometries.append(wrong_start)
        outside = _geometry()
        outside["corners"][2]["x"] = "10.000000001"
        invalid_geometries.append(outside)
        nonfinite = _geometry()
        nonfinite["corners"][0]["x"] = float("nan")
        invalid_geometries.append(nonfinite)
        for geometry in invalid_geometries:
            region = _region()
            region["quadrilateral"] = geometry
            with self.subTest(geometry=geometry), self.assertRaises(OCRContractError):
                map_ocr_runner_result(
                    _bundle(), _runner(region), image_bounds=[(10, 10)]
                )

    def test_text_bounds_nul_surrogate_and_count_limits(self) -> None:
        exact = "🧪" * 4_096
        self.assertEqual(len(exact), MAX_TEXT_CODEPOINTS)
        self.assertEqual(len(exact.encode("utf-8")), MAX_TEXT_BYTES)
        result = map_ocr_runner_result(
            _bundle(), _runner(_region(exact)), image_bounds=[(10, 10)]
        )
        self.assertEqual(result.to_dict()["items"][0]["recognized_text"], exact)
        for bad in ("a" * (MAX_TEXT_CODEPOINTS + 1), "\x00", "\ud800"):
            with self.subTest(size=len(bad)), self.assertRaises(OCRContractError):
                map_ocr_runner_result(
                    _bundle(), _runner(_region(bad)), image_bounds=[(10, 10)]
                )

        region = _region("x")
        with self.assertRaisesRegex(OCRContractError, "job item limit"):
            map_ocr_runner_result(
                _bundle(),
                _runner(*([region] * (MAX_ITEMS_PER_JOB + 1))),
                image_bounds=[(10, 10)],
            )
        with self.assertRaisesRegex(OCRContractError, "per-image item limit"):
            map_ocr_runner_result(
                _bundle(),
                _runner(*([region] * (MAX_ITEMS_PER_IMAGE + 1))),
                image_bounds=[(10, 10)],
            )
        self.assertEqual(MAX_TEXT_BYTES_PER_IMAGE, 1_048_576)
        self.assertEqual(MAX_TEXT_BYTES_PER_JOB, 67_108_864)
        self.assertEqual(MAX_GEOMETRY_BYTES_PER_ITEM, 512)
        self.assertEqual(MAX_GEOMETRY_BYTES_PER_JOB, 67_108_864)

    def test_input_index_page_optional_fields_and_empty_success(self) -> None:
        empty = map_ocr_runner_result(
            _bundle(), _runner(), image_bounds=[(20, 20)]
        ).to_dict()
        self.assertEqual(empty["items"], [])
        self.assertEqual(empty["privacy_safe_summary"]["item_count"], 0)

        region = _region()
        region.update(
            {
                "input_image_index": 1,
                "logical_page_reference": 7,
                "reading_order": 0,
                "orientation": "90",
            }
        )
        result = map_ocr_runner_result(
            _bundle(),
            _runner(region),
            image_bounds=[(10, 10), (10, 10)],
            logical_page_references=[None, 7],
        ).to_dict()
        self.assertEqual(result["items"][0]["logical_page_reference"], 7)
        for mutation in (
            {"input_image_index": 2},
            {"logical_page_reference": 0},
            {"logical_page_reference": 8},
            {"reading_order": -1},
            {"orientation": "45"},
        ):
            changed = copy.deepcopy(region)
            changed.update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(OCRContractError):
                map_ocr_runner_result(
                    _bundle(),
                    _runner(changed),
                    image_bounds=[(10, 10), (10, 10)],
                    logical_page_references=[None, 7],
                )

    def test_component_failure_never_publishes_partial_or_empty_result(self) -> None:
        for role in ("detector", "recognizer"):
            for status in ("timeout", "crash", "missing", "invalid_output"):
                runner = _runner(_region())
                runner[f"{role}_status"] = status
                with (
                    self.subTest(role=role, status=status),
                    self.assertRaisesRegex(
                        OCRContractError, f"ocr_component_{role}_{status}"
                    ),
                ):
                    map_ocr_runner_result(_bundle(), runner, image_bounds=[(10, 10)])

    def test_media_scope_rejects_documents_animation_archives_and_video(self) -> None:
        for mime in ("image/jpeg", "image/png", "image/webp"):
            validate_ocr_input_media(
                mime_type=mime, frame_count=1, animated=False, multipage=False
            )
        for mime in (
            "application/pdf",
            "image/tiff",
            "application/zip",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "video/mp4",
        ):
            with self.subTest(mime=mime), self.assertRaises(OCRContractError):
                validate_ocr_input_media(
                    mime_type=mime, frame_count=1, animated=False, multipage=False
                )
        for frame_count, animated, multipage in (
            (2, False, False),
            (1, True, False),
            (1, False, True),
        ):
            with (
                self.subTest(flags=(frame_count, animated, multipage)),
                self.assertRaises(OCRContractError),
            ):
                validate_ocr_input_media(
                    mime_type="image/webp",
                    frame_count=frame_count,
                    animated=animated,
                    multipage=multipage,
                )

    def test_canonical_and_packaged_schemas_are_identical(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for basename in ("ocr_bundle_interface.schema.json", "ocr_result.schema.json"):
            self.assertEqual(
                (root / "docs/schemas" / basename).read_bytes(),
                (root / "yolozu/data/schemas" / basename).read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
