import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _write_tiny_png(path: Path, *, width: int = 64, height: int = 32) -> None:
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + int(width).to_bytes(4, "big") + int(height).to_bytes(4, "big")
    )


class TestDatasetAutoDetectionExtendedCLI(unittest.TestCase):
    def _run(self, args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "yolozu", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )

    def test_auto_detect_voc_segmentation_root_imports_and_validates(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td) / "VOC2012"
            (root / "JPEGImages").mkdir(parents=True, exist_ok=True)
            (root / "SegmentationClass").mkdir(parents=True, exist_ok=True)
            (root / "ImageSets" / "Segmentation").mkdir(parents=True, exist_ok=True)

            _write_tiny_png(root / "JPEGImages" / "0001.png")
            (root / "SegmentationClass" / "0001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (root / "ImageSets" / "Segmentation" / "val.txt").write_text("0001\n", encoding="utf-8")

            proc_doctor = self._run(
                [
                    "doctor",
                    "import",
                    "--dataset-from",
                    "auto",
                    "--dataset",
                    str(root),
                    "--split",
                    "val",
                    "--output",
                    "-",
                ],
                cwd=repo_root,
            )
            if proc_doctor.returncode != 0:
                self.fail(f"doctor import auto on VOC root failed:\n{proc_doctor.stdout}\n{proc_doctor.stderr}")
            payload = json.loads(proc_doctor.stdout)
            dataset = payload.get("dataset") or {}
            self.assertEqual(dataset.get("from"), "segmentation")
            self.assertEqual((dataset.get("layout") or {}).get("format"), "voc_segmentation_root")
            self.assertEqual(dataset.get("task_family"), "segmentation")
            readiness = dataset.get("reference_trainer") or {}
            self.assertFalse(bool(readiness.get("direct_train_ready")))
            self.assertFalse(bool(readiness.get("train_ready_after_migration")))

            proc_validate_root = self._run(
                [
                    "validate",
                    "dataset",
                    str(root),
                    "--split",
                    "val",
                ],
                cwd=repo_root,
            )
            if proc_validate_root.returncode != 0:
                self.fail(f"validate dataset on VOC root failed:\n{proc_validate_root.stdout}\n{proc_validate_root.stderr}")

            out_dir = Path(td) / "voc_wrapper"
            proc_import = self._run(
                [
                    "import",
                    "dataset",
                    "--from",
                    "auto",
                    "--dataset",
                    str(root),
                    "--split",
                    "val",
                    "--output",
                    str(out_dir),
                    "--force",
                ],
                cwd=repo_root,
            )
            if proc_import.returncode != 0:
                self.fail(f"import dataset auto on VOC root failed:\n{proc_import.stdout}\n{proc_import.stderr}")

            dataset_json = json.loads((out_dir / "dataset.json").read_text(encoding="utf-8"))
            self.assertEqual(dataset_json.get("task"), "semantic_segmentation")
            self.assertEqual(dataset_json.get("split"), "val")

            proc_validate_wrapper = self._run(
                [
                    "validate",
                    "dataset",
                    str(out_dir),
                ],
                cwd=repo_root,
            )
            if proc_validate_wrapper.returncode != 0:
                self.fail(f"validate dataset on VOC descriptor wrapper failed:\n{proc_validate_wrapper.stdout}\n{proc_validate_wrapper.stderr}")

    def test_auto_detect_coco_keypoints_root_imports_and_exports(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td) / "coco_kps"
            images_dir = root / "images" / "val2017"
            ann_dir = root / "annotations"
            images_dir.mkdir(parents=True, exist_ok=True)
            ann_dir.mkdir(parents=True, exist_ok=True)

            _write_tiny_png(images_dir / "0001.png")
            annotations = {
                "images": [{"id": 1, "file_name": "0001.png", "width": 64, "height": 32}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [8, 4, 24, 16],
                        "iscrowd": 0,
                        "keypoints": [16, 8, 2, 24, 12, 2],
                        "num_keypoints": 2,
                    }
                ],
                "categories": [{"id": 1, "name": "person", "keypoints": ["nose", "eye"], "skeleton": [[1, 2]]}],
            }
            (ann_dir / "person_keypoints_val2017.json").write_text(json.dumps(annotations), encoding="utf-8")

            proc_doctor = self._run(
                [
                    "doctor",
                    "import",
                    "--dataset-from",
                    "auto",
                    "--dataset",
                    str(root),
                    "--split",
                    "val2017",
                    "--output",
                    "-",
                ],
                cwd=repo_root,
            )
            if proc_doctor.returncode != 0:
                self.fail(f"doctor import auto on COCO keypoints root failed:\n{proc_doctor.stdout}\n{proc_doctor.stderr}")
            payload = json.loads(proc_doctor.stdout)
            dataset = payload.get("dataset") or {}
            self.assertEqual(dataset.get("from"), "coco")
            self.assertEqual((dataset.get("layout") or {}).get("format"), "coco_keypoints_root")
            self.assertEqual(dataset.get("task_family"), "keypoints")
            readiness = dataset.get("reference_trainer") or {}
            self.assertFalse(bool(readiness.get("direct_train_ready")))
            self.assertTrue(bool(readiness.get("train_ready_after_migration")))
            self.assertEqual(readiness.get("task_family"), "keypoints")

            proc_validate_root = self._run(
                [
                    "validate",
                    "dataset",
                    str(root),
                    "--split",
                    "val2017",
                    "--strict",
                ],
                cwd=repo_root,
            )
            if proc_validate_root.returncode != 0:
                self.fail(f"validate dataset on COCO keypoints root failed:\n{proc_validate_root.stdout}\n{proc_validate_root.stderr}")

            wrapper = Path(td) / "wrapper"
            proc_import = self._run(
                [
                    "import",
                    "dataset",
                    "--from",
                    "auto",
                    "--dataset",
                    str(root),
                    "--split",
                    "val2017",
                    "--output",
                    str(wrapper),
                    "--force",
                ],
                cwd=repo_root,
            )
            if proc_import.returncode != 0:
                self.fail(f"import dataset auto on COCO keypoints root failed:\n{proc_import.stdout}\n{proc_import.stderr}")

            dataset_json = json.loads((wrapper / "dataset.json").read_text(encoding="utf-8"))
            self.assertEqual(dataset_json.get("task"), "keypoints")
            self.assertEqual(dataset_json.get("keypoint_names"), ["nose", "eye"])

            from rtdetr_pose.dataset import build_manifest as build_rtdetr_manifest

            train_manifest = build_rtdetr_manifest(wrapper, split="val2017")
            train_records = train_manifest.get("images") or []
            self.assertEqual(len(train_records), 1)
            train_label = train_records[0]["labels"][0]
            self.assertEqual(train_label["class_id"], 0)
            self.assertEqual(len(train_label.get("keypoints") or []), 2)
            self.assertEqual((train_manifest.get("keypoints_meta") or {}).get("num_keypoints"), 2)

            export_yolo = Path(td) / "export_yolo"
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
                self.fail(f"export-dataset yolo on keypoints wrapper failed:\n{proc_export_yolo.stdout}\n{proc_export_yolo.stderr}")
            self.assertIn("kpt_shape: [2, 3]", (export_yolo / "data.yaml").read_text(encoding="utf-8"))

            export_coco = Path(td) / "export_coco"
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
                self.fail(f"export-dataset coco on keypoints wrapper failed:\n{proc_export_coco.stdout}\n{proc_export_coco.stderr}")
            self.assertTrue((export_coco / "annotations" / "person_keypoints_val2017.json").is_file())
            coco_payload = json.loads((export_coco / "annotations" / "person_keypoints_val2017.json").read_text(encoding="utf-8"))
            annotation = (coco_payload.get("annotations") or [])[0]
            self.assertEqual(annotation.get("num_keypoints"), 2)
            self.assertEqual(len(annotation.get("keypoints") or []), 6)

    def test_export_segmentation_layout_with_symlinks(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            images_dir = root / "source_images"
            masks_dir = root / "source_masks"
            images_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)
            _write_tiny_png(images_dir / "sample.png")
            (masks_dir / "sample.png").write_bytes(b"\x89PNG\r\n\x1a\n")

            descriptor_root = root / "seg_wrapper"
            descriptor_root.mkdir(parents=True, exist_ok=True)
            (descriptor_root / "dataset.json").write_text(
                json.dumps(
                    {
                        "dataset": "synthetic_seg",
                        "task": "semantic_segmentation",
                        "split": "val",
                        "mode": "manifest",
                        "path_type": "absolute",
                        "ignore_index": 255,
                        "classes": ["background", "thing"],
                        "samples": [
                            {
                                "id": "sample",
                                "image": str(images_dir / "sample.png"),
                                "mask": str(masks_dir / "sample.png"),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            export_root = root / "seg_export"
            proc_export = self._run(
                [
                    "export-dataset",
                    "segmentation",
                    "--dataset",
                    str(descriptor_root),
                    "--out-dir",
                    str(export_root),
                    "--image-mode",
                    "symlink",
                    "--force",
                ],
                cwd=repo_root,
            )
            if proc_export.returncode != 0:
                self.fail(f"export-dataset segmentation failed:\n{proc_export.stdout}\n{proc_export.stderr}")

            self.assertTrue((export_root / "dataset.json").is_file())
            self.assertTrue((export_root / "images" / "val" / "sample.png").is_symlink())
            self.assertTrue((export_root / "masks" / "val" / "sample.png").is_symlink())

            proc_validate = self._run(
                [
                    "validate",
                    "dataset",
                    str(export_root),
                ],
                cwd=repo_root,
            )
            if proc_validate.returncode != 0:
                self.fail(f"validate dataset on exported segmentation layout failed:\n{proc_validate.stdout}\n{proc_validate.stderr}")


if __name__ == "__main__":
    unittest.main()
