import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestExportDatasetCLI(unittest.TestCase):
    def _run(self, args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "yolozu", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )

    def test_end_to_end_coco_to_yolozu_to_yolo_coco_and_kitti(self):
        repo_root = Path(__file__).resolve().parents[1]
        sample_root = repo_root / "data" / "conversion_tiny_coco"
        self.assertTrue(sample_root.is_dir(), "missing data/conversion_tiny_coco fixture")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            wrapper = root / "wrapper"
            export_yolo = root / "export_yolo"
            export_coco = root / "export_coco"
            reimported = root / "reimported"
            export_kitti = root / "export_kitti"

            proc_migrate = self._run(
                [
                    "migrate",
                    "dataset",
                    "--from",
                    "coco",
                    "--coco-root",
                    str(sample_root),
                    "--split",
                    "val2017",
                    "--output",
                    str(wrapper),
                    "--force",
                ],
                cwd=repo_root,
            )
            if proc_migrate.returncode != 0:
                self.fail(f"migrate dataset --from coco failed:\n{proc_migrate.stdout}\n{proc_migrate.stderr}")

            proc_validate_wrapper = self._run(
                [
                    "validate",
                    "dataset",
                    str(wrapper),
                    "--split",
                    "val2017",
                    "--strict",
                    "--max-images",
                    "2",
                ],
                cwd=repo_root,
            )
            if proc_validate_wrapper.returncode != 0:
                self.fail(
                    "validate dataset on migrated wrapper failed:\n"
                    f"{proc_validate_wrapper.stdout}\n{proc_validate_wrapper.stderr}"
                )

            proc_export_yolo = self._run(
                [
                    "export-dataset",
                    "yolo",
                    "--dataset",
                    str(wrapper),
                    "--split",
                    "val2017",
                    "--out-dir",
                    str(export_yolo),
                    "--force",
                ],
                cwd=repo_root,
            )
            if proc_export_yolo.returncode != 0:
                self.fail(f"export-dataset yolo failed:\n{proc_export_yolo.stdout}\n{proc_export_yolo.stderr}")

            self.assertTrue((export_yolo / "data.yaml").is_file())
            self.assertTrue((export_yolo / "images" / "val2017" / "000000000009.jpg").is_file())
            self.assertTrue((export_yolo / "labels" / "val2017" / "000000000009.txt").is_file())

            proc_validate_yolo = self._run(
                [
                    "validate",
                    "dataset",
                    str(export_yolo),
                    "--split",
                    "val2017",
                    "--strict",
                    "--max-images",
                    "2",
                ],
                cwd=repo_root,
            )
            if proc_validate_yolo.returncode != 0:
                self.fail(f"validate dataset on YOLO export failed:\n{proc_validate_yolo.stdout}\n{proc_validate_yolo.stderr}")

            proc_reimport = self._run(
                [
                    "migrate",
                    "dataset",
                    "--from",
                    "ultralytics",
                    "--data",
                    str(export_yolo / "data.yaml"),
                    "--split",
                    "val2017",
                    "--output",
                    str(reimported),
                    "--force",
                ],
                cwd=repo_root,
            )
            if proc_reimport.returncode != 0:
                self.fail(f"re-import from data.yaml failed:\n{proc_reimport.stdout}\n{proc_reimport.stderr}")

            proc_export_coco = self._run(
                [
                    "export-dataset",
                    "coco",
                    "--dataset",
                    str(wrapper),
                    "--split",
                    "val2017",
                    "--out-dir",
                    str(export_coco),
                    "--force",
                ],
                cwd=repo_root,
            )
            if proc_export_coco.returncode != 0:
                self.fail(f"export-dataset coco failed:\n{proc_export_coco.stdout}\n{proc_export_coco.stderr}")

            self.assertTrue((export_coco / "annotations" / "instances_val2017.json").is_file())
            self.assertTrue((export_coco / "images" / "val2017" / "000000000009.jpg").is_file())

            proc_validate_coco = self._run(
                [
                    "validate",
                    "dataset",
                    str(export_coco),
                    "--split",
                    "val2017",
                    "--strict",
                    "--max-images",
                    "2",
                ],
                cwd=repo_root,
            )
            if proc_validate_coco.returncode != 0:
                self.fail(f"validate dataset on COCO export failed:\n{proc_validate_coco.stdout}\n{proc_validate_coco.stderr}")

            proc_export_kitti = self._run(
                [
                    "export-dataset",
                    "kitti",
                    "--dataset",
                    str(wrapper),
                    "--split",
                    "val2017",
                    "--out-dir",
                    str(export_kitti),
                    "--force",
                ],
                cwd=repo_root,
            )
            if proc_export_kitti.returncode != 0:
                self.fail(f"export-dataset kitti failed:\n{proc_export_kitti.stdout}\n{proc_export_kitti.stderr}")

            self.assertTrue((export_kitti / "image_2" / "000000000009.jpg").is_file())
            self.assertTrue((export_kitti / "label_2" / "000000000009.txt").is_file())
            self.assertTrue((export_kitti / "ImageSets" / "Main" / "val2017.txt").is_file())

            label_lines = (export_kitti / "label_2" / "000000000009.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(label_lines), 2)
            self.assertTrue(label_lines[0].startswith("crate "))
            self.assertTrue(label_lines[1].startswith("cone "))

            classes_payload = json.loads((export_yolo / "labels" / "val2017" / "classes.json").read_text(encoding="utf-8"))
            self.assertEqual(classes_payload.get("class_names"), ["crate", "cone"])

            coco_payload = json.loads((export_coco / "annotations" / "instances_val2017.json").read_text(encoding="utf-8"))
            self.assertEqual(len(coco_payload.get("images") or []), 2)
            self.assertEqual(len(coco_payload.get("annotations") or []), 3)
            self.assertEqual([cat.get("name") for cat in (coco_payload.get("categories") or [])], ["crate", "cone"])


if __name__ == "__main__":
    unittest.main()
