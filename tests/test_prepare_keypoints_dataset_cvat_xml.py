from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestPrepareKeypointsDatasetCVATXML(unittest.TestCase):
    def test_prepare_keypoints_dataset_cvat_xml_minimal(self) -> None:
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:
            self.skipTest(f"Pillow is required for this test: {exc}")

        repo_root = Path(__file__).resolve().parents[1]
        tool = repo_root / "tools" / "prepare_keypoints_dataset.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            work = Path(td)
            source = work / "cvat"
            out = work / "out"
            images_dir = source / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            Image.new("RGB", (100, 80), color=(120, 120, 120)).save(images_dir / "img1.jpg")

            (source / "annotations.xml").write_text(
                """<?xml version=\"1.0\" encoding=\"utf-8\"?>
<annotations>
  <meta><task><labels>
    <label><name>nose</name></label>
    <label><name>left_eye</name></label>
  </labels></task></meta>
  <image id=\"0\" name=\"img1.jpg\" width=\"100\" height=\"80\">
    <box label=\"person\" group_id=\"1\" xtl=\"10\" ytl=\"20\" xbr=\"50\" ybr=\"60\"/>
    <points label=\"nose\" group_id=\"1\" points=\"30,30\" occluded=\"0\"/>
    <points label=\"left_eye\" group_id=\"1\" points=\"25,28\" occluded=\"1\"/>
  </image>
</annotations>
""",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(tool),
                    "--source",
                    str(source),
                    "--format",
                    "cvat_xml",
                    "--out",
                    str(out),
                    "--class-id",
                    "0",
                ],
                cwd=str(repo_root),
                check=True,
                capture_output=True,
                text=True,
            )

            dataset = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
            classes = json.loads((out / "labels" / "val2017" / "classes.json").read_text(encoding="utf-8"))
            label_line = (out / "labels" / "val2017" / "img1.txt").read_text(encoding="utf-8").strip()

            self.assertEqual(dataset["task"], "keypoints")
            self.assertEqual(dataset["split"], "val2017")
            self.assertEqual(int(dataset["num_keypoints"]), 2)
            self.assertEqual(int(classes["num_keypoints"]), 2)
            self.assertEqual(len(label_line.split()), 11)


if __name__ == "__main__":
    unittest.main()
