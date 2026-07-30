import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


class TestPrepareExternalRuntimeSmokeDatasets(unittest.TestCase):
    def test_prepares_all_runtime_layouts(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            (source / "images" / "train").mkdir(parents=True)
            (source / "labels" / "train").mkdir(parents=True)
            Image.new("RGB", (20, 10), color="white").save(source / "images" / "train" / "a.jpg")
            (source / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.4 0.6\n")
            (source / "labels" / "train" / "classes.json").write_text('{"0": "object"}\n')
            output = root / "out"
            subprocess.run(
                [
                    sys.executable,
                    str(repo / "tools" / "prepare_external_runtime_smoke_datasets.py"),
                    "--source",
                    str(source),
                    "--output",
                    str(output),
                ],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )
            detection = json.loads(
                (output / "detection" / "annotations" / "instances_train2017.json").read_text()
            )
            keypoints = json.loads(
                (output / "keypoints" / "annotations" / "person_keypoints_train2017.json").read_text()
            )
            self.assertEqual(detection["annotations"][0]["bbox"], [6.0, 2.0, 8.0, 6.0])
            self.assertEqual(len(keypoints["annotations"][0]["keypoints"]), 51)
            self.assertTrue(
                (output / "segmentation" / "labels" / "train" / "a_gtFine_labelTrainIds.png").is_file()
            )
            report = json.loads((output / "preparation_report.json").read_text())
            self.assertIn("tree_sha256", report)

    def test_refuses_existing_output(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            output.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    str(repo / "tools" / "prepare_external_runtime_smoke_datasets.py"),
                    "--source",
                    str(root),
                    "--output",
                    str(output),
                ],
                cwd=repo,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to replace", result.stderr)


if __name__ == "__main__":
    unittest.main()
