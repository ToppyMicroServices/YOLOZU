import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import export_predictions


def _fallback_report():
    return {
        "requested": {
            "enabled": True,
            "backend": "missing",
            "mode": "default",
            "fullgraph": True,
            "dynamic": False,
            "allow_fallback": True,
        },
        "actual": {
            "status": "fallback",
            "backend": "eager",
            "mode": None,
            "fullgraph": False,
            "dynamic": None,
        },
        "evidence": {
            "compile_api_available": True,
            "setup_completed": False,
            "first_execution_completed": False,
            "fallback_execution_completed": True,
            "counter_source": None,
            "counter_delta": None,
            "graph_count": None,
            "graph_break_count": None,
            "captured_call_count": None,
        },
        "failure": {
            "phase": "setup",
            "type": "ValueError",
            "message": "invalid backend",
        },
    }


class _FallbackAdapter:
    init_kwargs = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def predict(self, records):
        return [
            {"schema_version": 2, "image": record["image"], "detections": []}
            for record in records
        ]

    def require_compile_established(self):
        return _fallback_report()


class _LazyFailureAdapter:
    def __init__(self, **_kwargs):
        pass

    def predict(self, _records):
        raise RuntimeError("lazy compile failure")


class _MismatchedEvidenceAdapter(_FallbackAdapter):
    def require_compile_established(self):
        report = _fallback_report()
        report["actual"]["backend"] = "inductor"
        return report


class TestCompileEvidenceExport(unittest.TestCase):
    def test_explicit_fallback_metadata_records_requested_and_actual(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "predictions.json"
            with (
                patch.object(
                    export_predictions,
                    "build_manifest",
                    return_value={"images": [{"image": "image.jpg"}]},
                ),
                patch.object(
                    export_predictions,
                    "RTDETRPoseAdapter",
                    _FallbackAdapter,
                ),
            ):
                export_predictions.main(
                    [
                        "--adapter",
                        "rtdetr_pose",
                        "--torch-compile",
                        "--torch-compile-backend",
                        "missing",
                        "--torch-compile-mode",
                        "default",
                        "--torch-compile-fullgraph",
                        "--torch-compile-dynamic",
                        "false",
                        "--allow-compile-fallback",
                        "--wrap",
                        "--output",
                        str(output),
                    ]
                )

            payload = json.loads(output.read_text())
            report = payload["meta"]["inference"]["torch_compile"]
            self.assertEqual(report["requested"]["backend"], "missing")
            self.assertTrue(report["requested"]["fullgraph"])
            self.assertFalse(report["requested"]["dynamic"])
            self.assertTrue(report["requested"]["allow_fallback"])
            self.assertEqual(report["actual"]["status"], "fallback")
            self.assertEqual(report["actual"]["backend"], "eager")
            self.assertEqual(report["failure"]["phase"], "setup")
            self.assertTrue(
                _FallbackAdapter.init_kwargs["allow_compile_fallback"]
            )

    def test_compile_failure_removes_stale_success_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "predictions.json"
            output.write_text('{"stale": true}')
            with (
                patch.object(
                    export_predictions,
                    "build_manifest",
                    return_value={"images": [{"image": "image.jpg"}]},
                ),
                patch.object(
                    export_predictions,
                    "RTDETRPoseAdapter",
                    _LazyFailureAdapter,
                ),
                self.assertRaisesRegex(RuntimeError, "lazy compile failure"),
            ):
                export_predictions.main(
                    [
                        "--adapter",
                        "rtdetr_pose",
                        "--torch-compile",
                        "--wrap",
                        "--output",
                        str(output),
                    ]
                )
            self.assertFalse(output.exists())

    def test_invalid_compile_evidence_is_not_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "predictions.json"
            with (
                patch.object(
                    export_predictions,
                    "build_manifest",
                    return_value={"images": [{"image": "image.jpg"}]},
                ),
                patch.object(
                    export_predictions,
                    "RTDETRPoseAdapter",
                    _MismatchedEvidenceAdapter,
                ),
                self.assertRaisesRegex(SystemExit, "fallback actual state mismatch"),
            ):
                export_predictions.main(
                    [
                        "--adapter",
                        "rtdetr_pose",
                        "--torch-compile",
                        "--torch-compile-backend",
                        "missing",
                        "--torch-compile-mode",
                        "default",
                        "--torch-compile-fullgraph",
                        "--torch-compile-dynamic",
                        "false",
                        "--allow-compile-fallback",
                        "--wrap",
                        "--output",
                        str(output),
                    ]
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
