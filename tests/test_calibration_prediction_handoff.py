"""Real CLI calibration -> strict validation -> evaluation handoff regressions."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from yolozu.predictions import validate_predictions_payload


REPO = Path(__file__).resolve().parents[1]
METHODS = ("fracal", "la", "norcal", "temperature")


class TestCalibrationPredictionHandoff(unittest.TestCase):
    def run_cli(self, *args):
        proc = subprocess.run(
            [sys.executable, "-m", "yolozu", *map(str, args)],
            cwd=REPO, capture_output=True, text=True, check=False, timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        return proc

    def exercise(self, *, input_shape, include_meta=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data"
            self.run_cli("demo", "dataset", "--run-dir", dataset)
            fixture = json.loads((dataset / "predictions.json").read_text())
            if include_meta:
                exported = root / "exported.json"
                self.run_cli("export", "--backend", "dummy", "--dataset", dataset,
                             "--split", "val", "--output", exported)
                # Retain the actual export's metadata when calibrating its output.
                fixture = json.loads(exported.read_text())
                self.assertIn("meta", fixture)
            entries = fixture["predictions"]
            if input_shape == "list":
                source_payload = entries
            elif input_shape == "mapping":
                source_payload = {entry["image"]: entry["detections"] for entry in entries}
            else:
                source_payload = fixture
            source = root / "input.json"
            source.write_text(json.dumps(source_payload))
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            self.run_cli("validate", "predictions", source, "--strict", "--json")
            source_geometry = [
                [(det["class_id"], det["bbox"]) for det in entry["detections"]]
                for entry in entries
            ]
            original_meta = copy.deepcopy(fixture.get("meta"))
            for method in METHODS:
                with self.subTest(shape=input_shape, meta=include_meta, method=method):
                    target = root / f"{method}.json"
                    self.run_cli("calibrate", "--method", method, "--task", "bbox",
                                 "--dataset", dataset, "--split", "val", "--predictions", source,
                                 "--output", target, "--output-report", root / f"{method}_report.json")
                    payload = json.loads(target.read_text())
                    validate_predictions_payload(payload, strict=True)
                    self.run_cli("validate", "predictions", target, "--strict", "--json")
                    if include_meta:
                        meta = payload["meta"].copy()
                        provenance = meta.pop("posthoc_calibration")
                        self.assertEqual(meta, original_meta)
                    else:
                        self.assertNotIn("meta", payload)
                        provenance = payload["posthoc_calibration"]
                    self.assertEqual(provenance["method"], method)
                    self.assertIn("timestamp", provenance)
                    self.assertEqual(
                        [[(d["class_id"], d["bbox"]) for d in e["detections"]]
                         for e in payload["predictions"]], source_geometry,
                    )
                    self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
                    output = root / f"{method}_eval.json"
                    args = ["eval-coco", "--dataset", dataset, "--split", "val",
                            "--predictions", target, "--output", output]
                    real_eval = importlib.util.find_spec("pycocotools") is not None
                    if not real_eval:
                        args.append("--dry-run")
                    self.run_cli(*args)
                    report = json.loads(output.read_text())
                    self.assertEqual(report["counts"]["images"], len(entries))
                    self.assertEqual(report["counts"]["detections"], sum(len(e["detections"]) for e in entries))
                    self.assertEqual(report["dry_run"], not real_eval)
                    if real_eval:
                        self.assertIsNotNone(report["metrics"]["map50_95"])

    def test_meta_free_wrapper_all_methods(self):
        self.exercise(input_shape="wrapper")

    def test_meta_free_list_all_methods(self):
        self.exercise(input_shape="list")

    def test_meta_free_mapping_all_methods(self):
        self.exercise(input_shape="mapping")

    def test_complete_export_metadata_all_methods(self):
        self.exercise(input_shape="wrapper", include_meta=True)


if __name__ == "__main__":
    unittest.main()
