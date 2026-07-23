import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _strict_json_loads(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_nonstandard_constant)


def _classification_labels() -> dict[str, Any]:
    return {
        "classes": ["cat", "dog"],
        "samples": [
            {"id": "img0", "label": "cat"},
            {"id": "img1", "label": "dog"},
        ],
    }


def _classification_predictions() -> dict[str, Any]:
    return {
        "classes": ["cat", "dog"],
        "predictions": [
            {"id": "img0", "scores": [0.9, 0.1]},
            {"id": "img1", "scores": [0.2, 0.8]},
        ],
    }


def _obb_labels() -> dict[str, Any]:
    return {
        "classes": ["ship"],
        "samples": [
            {
                "id": "img0",
                "objects": [
                    {
                        "class_id": 0,
                        "obb": {
                            "cx": 0.5,
                            "cy": 0.5,
                            "w": 0.4,
                            "h": 0.2,
                            "angle_deg": 10.0,
                        },
                    }
                ],
            },
            {"id": "img1", "objects": []},
        ],
    }


def _obb_predictions() -> dict[str, Any]:
    return {
        "classes": ["ship"],
        "predictions": [
            {
                "id": "img0",
                "detections": [
                    {
                        "class_id": 0,
                        "score": 0.9,
                        "obb": {
                            "cx": 0.5,
                            "cy": 0.5,
                            "w": 0.4,
                            "h": 0.2,
                            "angle_deg": 10.0,
                        },
                    }
                ],
            },
            {"id": "img1", "detections": []},
        ],
    }


class TestBenchmarkArtifactValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def _run_public_benchmark(
        self,
        root: Path,
        *,
        task: str,
        labels: Any,
        predictions: Any,
        onnx_predictions: Any | None = None,
        formats: str | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, Any], Path]:
        labels_path = root / "labels.json"
        predictions_path = root / "predictions.json"
        onnx_predictions_path = root / "predictions_onnx.json"
        report_path = root / "benchmark_report.json"
        history_path = root / "benchmark_history.jsonl"
        artifacts_path = root / "artifacts"
        labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
        predictions_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
        if onnx_predictions is not None:
            onnx_predictions_path.write_text(json.dumps(onnx_predictions, indent=2), encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "yolozu",
            "benchmark",
            "--task",
            task,
            "--model",
            str(predictions_path),
            "--data",
            str(labels_path),
            "--format",
            formats or ("torch,onnx" if onnx_predictions is not None else "torch"),
            "--latency-source",
            "artifact_eval",
            "--strict",
            "--history",
            str(history_path),
            "--predictions-output",
            str(artifacts_path),
            "--eval-output",
            str(artifacts_path),
            "--parity-output",
            str(artifacts_path),
            "--output",
            str(report_path),
        ]
        if onnx_predictions is not None:
            cmd.extend(["--onnx-model", str(onnx_predictions_path)])
        proc = subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        self.assertTrue(
            report_path.is_file(),
            f"benchmark report was not written:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )
        report = _strict_json_loads(report_path.read_text(encoding="utf-8"))
        self._assert_outputs_are_strict_json(report, history_path)
        return proc, report, report_path

    def _assert_outputs_are_strict_json(self, report: dict[str, Any], history_path: Path) -> None:
        for result in report["results"]:
            for path_text in result["artifacts"].values():
                path = Path(path_text)
                self.assertTrue(path.is_file(), f"missing benchmark artifact: {path}")
                _strict_json_loads(path.read_text(encoding="utf-8"))
        history_lines = history_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(history_lines), 1)
        _strict_json_loads(history_lines[0])

    def _assert_invalid_case(
        self,
        *,
        task: str,
        labels: Any,
        predictions: Any,
        expected_error: str,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            proc, report, _ = self._run_public_benchmark(
                Path(td),
                task=task,
                labels=labels,
                predictions=predictions,
            )
        self.assertEqual(proc.returncode, 2, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        self.assertEqual(report["status"], "failed")
        result = report["results"][0]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["skip_reason"], f"{task}_artifact_invalid")
        self.assertIn(expected_error, result["error"])

    def test_scalar_top_level_payloads_fail_through_public_cli(self) -> None:
        for task, valid_labels, valid_predictions in (
            ("classification", _classification_labels(), _classification_predictions()),
            ("obb", _obb_labels(), _obb_predictions()),
        ):
            for artifact_side in ("labels", "predictions"):
                with self.subTest(task=task, artifact_side=artifact_side):
                    self._assert_invalid_case(
                        task=task,
                        labels="not-an-object" if artifact_side == "labels" else valid_labels,
                        predictions=42 if artifact_side == "predictions" else valid_predictions,
                        expected_error="JSON top level must be an object or array",
                    )

    def test_optional_class_vocabulary_compatibility_through_public_cli(self) -> None:
        for task in ("classification", "obb"):
            for representation in ("missing", "null", "empty"):
                with self.subTest(task=task, representation=representation):
                    labels = _classification_labels() if task == "classification" else _obb_labels()
                    predictions = (
                        _classification_predictions()
                        if task == "classification"
                        else _obb_predictions()
                    )
                    if task == "classification":
                        labels["samples"][0]["label"] = 0
                        labels["samples"][1]["label"] = 1
                    if representation == "missing":
                        labels.pop("classes")
                        predictions.pop("classes")
                    else:
                        value = None if representation == "null" else []
                        labels["classes"] = value
                        predictions["classes"] = value
                    with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
                        proc, report, _ = self._run_public_benchmark(
                            Path(td),
                            task=task,
                            labels=labels,
                            predictions=predictions,
                        )
                    self.assertEqual(
                        proc.returncode,
                        0,
                        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
                    )
                    self.assertEqual(report["status"], "ok")

    def test_nonempty_class_vocabulary_requires_unique_strings(self) -> None:
        for task in ("classification", "obb"):
            for artifact_side in ("labels", "predictions"):
                for classes, expected_error in (
                    ([0], "class names must be strings"),
                    (["ship", "ship"], "duplicate class name"),
                    ([""], "class names must be non-empty"),
                ):
                    with self.subTest(
                        task=task,
                        artifact_side=artifact_side,
                        classes=classes,
                    ):
                        labels = (
                            _classification_labels()
                            if task == "classification"
                            else _obb_labels()
                        )
                        predictions = (
                            _classification_predictions()
                            if task == "classification"
                            else _obb_predictions()
                        )
                        target = labels if artifact_side == "labels" else predictions
                        target["classes"] = classes
                        self._assert_invalid_case(
                            task=task,
                            labels=labels,
                            predictions=predictions,
                            expected_error=expected_error,
                        )

    def test_sleep_s_rejects_nonfinite_and_negative_values_before_writes(self) -> None:
        for value in ("nan", "inf", "-inf", "-0.01"):
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
                    root = Path(td)
                    labels_path = root / "labels.json"
                    predictions_path = root / "predictions.json"
                    report_path = root / "benchmark_report.json"
                    labels_path.write_text(
                        json.dumps(_classification_labels()),
                        encoding="utf-8",
                    )
                    predictions_path.write_text(
                        json.dumps(_classification_predictions()),
                        encoding="utf-8",
                    )
                    proc = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "yolozu",
                            "benchmark",
                            "--task",
                            "classification",
                            "--model",
                            str(predictions_path),
                            "--data",
                            str(labels_path),
                            "--format",
                            "torch",
                            "--latency-source",
                            "artifact_eval",
                            f"--sleep-s={value}",
                            "--strict",
                            "--output",
                            str(report_path),
                        ],
                        cwd=str(self.repo_root),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                        text=True,
                    )
                    self.assertEqual(
                        proc.returncode,
                        1,
                        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
                    )
                    self.assertFalse(report_path.exists())
                    self.assertIn(
                        "--sleep-s must be a finite, non-negative number",
                        proc.stdout + proc.stderr,
                    )

    def test_classification_rejects_nonfinite_scores_through_public_cli(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                predictions = _classification_predictions()
                predictions["predictions"][0]["scores"][0] = value
                self._assert_invalid_case(
                    task="classification",
                    labels=_classification_labels(),
                    predictions=predictions,
                    expected_error="scores must be finite",
                )

    def test_classification_rejects_boolean_scores_through_public_cli(self) -> None:
        predictions = _classification_predictions()
        predictions["predictions"][0]["scores"][0] = True
        self._assert_invalid_case(
            task="classification",
            labels=_classification_labels(),
            predictions=predictions,
            expected_error="scores must be numeric",
        )

    def test_classification_rejects_duplicate_ids_through_public_cli(self) -> None:
        labels = _classification_labels()
        labels["samples"].append(copy.deepcopy(labels["samples"][0]))
        self._assert_invalid_case(
            task="classification",
            labels=labels,
            predictions=_classification_predictions(),
            expected_error="duplicate sample id",
        )

        predictions = _classification_predictions()
        predictions["predictions"].append(copy.deepcopy(predictions["predictions"][0]))
        self._assert_invalid_case(
            task="classification",
            labels=_classification_labels(),
            predictions=predictions,
            expected_error="duplicate sample id",
        )

    def test_classification_rejects_class_and_score_length_drift_through_public_cli(self) -> None:
        predictions = _classification_predictions()
        predictions["classes"] = ["dog", "cat"]
        self._assert_invalid_case(
            task="classification",
            labels=_classification_labels(),
            predictions=predictions,
            expected_error="classes must exactly match",
        )

        predictions = _classification_predictions()
        predictions["predictions"][1]["scores"] = [0.8]
        self._assert_invalid_case(
            task="classification",
            labels=_classification_labels(),
            predictions=predictions,
            expected_error="score count must match classes and other compared artifacts",
        )

    def test_classification_valid_multi_sample_artifact_emits_strict_json(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            proc, report, _ = self._run_public_benchmark(
                Path(td),
                task="classification",
                labels=_classification_labels(),
                predictions=_classification_predictions(),
            )
        self.assertEqual(proc.returncode, 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["results"][0]["eval_metrics"]["samples"], 2)

    def test_classification_rejects_cross_backend_class_and_length_drift(self) -> None:
        labels = _classification_labels()
        labels.pop("classes")
        labels["samples"][0]["label"] = 0
        labels["samples"][1]["label"] = 1

        for torch_predictions, onnx_predictions, expected_error in (
            (
                _classification_predictions(),
                {
                    **_classification_predictions(),
                    "classes": ["cat", "bird"],
                },
                "classes must exactly match",
            ),
            (
                {
                    "predictions": _classification_predictions()["predictions"],
                },
                {
                    "predictions": [
                        {"id": "img0", "scores": [0.8, 0.1, 0.1]},
                        {"id": "img1", "scores": [0.1, 0.8, 0.1]},
                    ],
                },
                "score count must match classes and other compared artifacts",
            ),
        ):
            with self.subTest(expected_error=expected_error):
                with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
                    proc, report, _ = self._run_public_benchmark(
                        Path(td),
                        task="classification",
                        labels=labels,
                        predictions=torch_predictions,
                        onnx_predictions=onnx_predictions,
                    )
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(report["status"], "partial")
                results = {item["format"]: item for item in report["results"]}
                self.assertEqual(results["torch"]["status"], "ok")
                self.assertEqual(results["onnx"]["status"], "failed")
                self.assertIn(expected_error, results["onnx"]["error"])
                if expected_error == "classes must exactly match":
                    self.assertIn(
                        "the first successfully evaluated backend artifact",
                        results["onnx"]["error"],
                    )

    def test_classification_invalid_backend_does_not_poison_shared_schema(self) -> None:
        labels = {
            "samples": [
                {"id": "img0", "label": 0},
                {"id": "img1", "label": 1},
            ]
        }
        incomplete_torch = {
            "predictions": [
                {"id": "img0", "scores": [0.9, 0.05, 0.05]},
            ]
        }
        valid_onnx = {
            "predictions": [
                {"id": "img0", "scores": [0.9, 0.1]},
                {"id": "img1", "scores": [0.1, 0.9]},
            ]
        }
        for formats in ("torch,onnx", "onnx,torch"):
            with self.subTest(formats=formats):
                with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
                    proc, report, _ = self._run_public_benchmark(
                        Path(td),
                        task="classification",
                        labels=labels,
                        predictions=incomplete_torch,
                        onnx_predictions=valid_onnx,
                        formats=formats,
                    )
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(report["status"], "partial")
                results = {item["format"]: item for item in report["results"]}
                self.assertEqual(results["torch"]["status"], "failed")
                self.assertEqual(results["onnx"]["status"], "ok")

    def test_obb_rejects_duplicate_image_ids_through_public_cli(self) -> None:
        labels = _obb_labels()
        labels["samples"].append(copy.deepcopy(labels["samples"][0]))
        self._assert_invalid_case(
            task="obb",
            labels=labels,
            predictions=_obb_predictions(),
            expected_error="duplicate image id",
        )

        predictions = _obb_predictions()
        predictions["predictions"].append(copy.deepcopy(predictions["predictions"][0]))
        self._assert_invalid_case(
            task="obb",
            labels=_obb_labels(),
            predictions=predictions,
            expected_error="duplicate image id",
        )

    def test_obb_rejects_nonfinite_and_out_of_range_scores_through_public_cli(self) -> None:
        for value, expected_error in (
            (math.nan, "score must be finite"),
            (math.inf, "score must be finite"),
            (-math.inf, "score must be finite"),
            (-0.01, "score must be in [0,1]"),
            (1.01, "score must be in [0,1]"),
        ):
            with self.subTest(value=value):
                predictions = _obb_predictions()
                predictions["predictions"][0]["detections"][0]["score"] = value
                self._assert_invalid_case(
                    task="obb",
                    labels=_obb_labels(),
                    predictions=predictions,
                    expected_error=expected_error,
                )

        predictions = _obb_predictions()
        predictions["predictions"][0]["detections"][0]["score"] = True
        self._assert_invalid_case(
            task="obb",
            labels=_obb_labels(),
            predictions=predictions,
            expected_error="score must be numeric",
        )

    def test_obb_rejects_nonfinite_geometry_through_public_cli(self) -> None:
        for artifact_side in ("labels", "predictions"):
            for field in ("cx", "cy", "w", "h", "angle_deg"):
                with self.subTest(artifact_side=artifact_side, field=field):
                    labels = _obb_labels()
                    predictions = _obb_predictions()
                    if artifact_side == "labels":
                        labels["samples"][0]["objects"][0]["obb"][field] = math.nan
                    else:
                        predictions["predictions"][0]["detections"][0]["obb"][field] = math.nan
                    self._assert_invalid_case(
                        task="obb",
                        labels=labels,
                        predictions=predictions,
                        expected_error=f"obb.{field} must be finite",
                    )

    def test_obb_rejects_boolean_geometry_through_public_cli(self) -> None:
        for field in ("cx", "cy", "w", "h", "angle_deg"):
            with self.subTest(field=field):
                predictions = _obb_predictions()
                predictions["predictions"][0]["detections"][0]["obb"][field] = True
                self._assert_invalid_case(
                    task="obb",
                    labels=_obb_labels(),
                    predictions=predictions,
                    expected_error=f"obb.{field} must be numeric",
                )

    def test_obb_rejects_class_list_drift_through_public_cli(self) -> None:
        predictions = _obb_predictions()
        predictions["classes"] = ["plane"]
        self._assert_invalid_case(
            task="obb",
            labels=_obb_labels(),
            predictions=predictions,
            expected_error="classes must exactly match",
        )

    def test_obb_rejects_boolean_and_out_of_range_class_ids(self) -> None:
        for artifact_side in ("labels", "predictions"):
            for value, expected_error in (
                (True, "class_id must be an int index or class name"),
                (1, "class_id must be smaller than the classes list length"),
            ):
                with self.subTest(artifact_side=artifact_side, value=value):
                    labels = _obb_labels()
                    predictions = _obb_predictions()
                    if artifact_side == "labels":
                        labels["samples"][0]["objects"][0]["class_id"] = value
                    else:
                        predictions["predictions"][0]["detections"][0]["class_id"] = value
                    self._assert_invalid_case(
                        task="obb",
                        labels=labels,
                        predictions=predictions,
                        expected_error=expected_error,
                    )

        labels = _obb_labels()
        labels.pop("classes")
        labels["samples"][0]["objects"][0]["class_id"] = 7
        self._assert_invalid_case(
            task="obb",
            labels=labels,
            predictions=_obb_predictions(),
            expected_error="OBB labels: class_id out of range",
        )

    def test_obb_valid_empty_detections_and_multi_sample_emit_strict_json(self) -> None:
        predictions = _obb_predictions()
        predictions["predictions"][0]["detections"] = []
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            proc, report, _ = self._run_public_benchmark(
                Path(td),
                task="obb",
                labels=_obb_labels(),
                predictions=predictions,
            )
            normalized_path = Path(report["results"][0]["artifacts"]["predictions"])
            normalized = _strict_json_loads(normalized_path.read_text(encoding="utf-8"))

        self.assertEqual(proc.returncode, 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["results"][0]["eval_metrics"]["samples"], 2)
        self.assertEqual(len(normalized["predictions"]), 2)
        self.assertEqual(normalized["predictions"][0]["detections"], [])


if __name__ == "__main__":
    unittest.main()
