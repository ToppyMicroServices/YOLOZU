"""Filesystem/CLI regressions for dataset conversion identity and portability."""

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from yolozu.core.config import simple_yaml_load
from yolozu.datasets.dataset import build_manifest


class TestDatasetConversionSafetyCLI(unittest.TestCase):
    repo_root = Path(__file__).resolve().parents[1]

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-B", "-m", "yolozu", *map(str, args)],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    def _success(self, *args):
        proc = self._run(*args)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        return proc

    def _hashes(self, root):
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def _fixture(self, root, names=("a.png", "b.png", "negative.png")):
        doc = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 42, "name": "cone"}, {"id": 7, "name": "crate"}],
        }
        for index, name in enumerate(names, 1):
            path = root / "images" / "val2017" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 32), color=(index * 30, 10, 20)).save(path)
            doc["images"].append({"id": index, "file_name": name, "width": 64, "height": 32})
            if index <= 2:
                doc["annotations"].append({
                    "id": index,
                    "image_id": index,
                    "category_id": 7 if index == 1 else 42,
                    "bbox": [4, 2, 20, 10],
                    "area": 200,
                    "iscrowd": 0,
                })
        annotations = root / "annotations" / "instances_val2017.json"
        annotations.parent.mkdir(parents=True)
        annotations.write_text(json.dumps(doc), encoding="utf-8")
        return doc

    def _import(self, source, wrapper):
        self._success(
            "import", "dataset", "--from", "coco-instances",
            "--instances", source / "annotations" / "instances_val2017.json",
            "--images-dir", source / "images" / "val2017",
            "--split", "val2017", "--output", wrapper,
        )

    def test_existing_migration_refusal_is_read_only_for_all_modes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            original = self._fixture(source)
            annotations = source / "annotations" / "instances_val2017.json"
            for mode in ("manifest", "copy", "symlink"):
                with self.subTest(mode=mode):
                    annotations.write_text(json.dumps(original), encoding="utf-8")
                    output = root / mode
                    command = (
                        "migrate", "dataset", "--from", "coco", "--coco-root", source,
                        "--split", "val2017", "--mode", mode, "--output", output,
                    )
                    self._success(*command)
                    before = self._hashes(output)
                    modified = copy.deepcopy(original)
                    modified["annotations"][0]["bbox"] = [8, 2, 20, 10]
                    annotations.write_text(json.dumps(modified), encoding="utf-8")
                    proc = self._run(*command)
                    self.assertNotEqual(proc.returncode, 0)
                    self.assertIn("output already exists", proc.stderr)
                    self.assertEqual(self._hashes(output), before)
                    self._success(*command, "--force")
                    self.assertNotEqual(self._hashes(output)["labels/val2017/a.txt"], before["labels/val2017/a.txt"])

    def test_migration_rejects_flat_collisions_before_any_writes(self):
        cases = (
            ("camera_a/frame.png", "camera_b/frame.png"),
            ("東京/同じ.png", "大阪/同じ.png"),
            ("camera_a/frame.jpg", "camera_b/frame.png"),
            ("camera_a/Frame.png", "camera_b/frame.PNG"),
            ("camera_a/caf\u00e9.png", "camera_b/cafe\u0301.png"),
            ("frame.png", "frame.png"),
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for index, names in enumerate(cases):
                source = root / f"source_{index}"
                self._fixture(source, names)
                source_before = self._hashes(source)
                self._success("validate", "dataset", source, "--split", "val2017", "--strict")
                for mode in ("manifest", "copy", "symlink"):
                    with self.subTest(names=names, mode=mode):
                        new_output = root / f"new_{index}_{mode}"
                        command = (
                            "migrate", "dataset", "--from", "coco", "--coco-root", source,
                            "--split", "val2017", "--mode", mode,
                        )
                        proc = self._run(*command, "--output", new_output)
                        self.assertNotEqual(proc.returncode, 0)
                        self.assertIn("collision", proc.stderr)
                        self.assertIn("coco-instances", proc.stderr)
                        self.assertFalse(new_output.exists())
                        existing_output = root / f"existing_{index}_{mode}"
                        existing_output.mkdir()
                        (existing_output / "keep.txt").write_text("owned output to preserve", encoding="utf-8")
                        before = self._hashes(existing_output)
                        proc = self._run(*command, "--output", existing_output, "--force")
                        self.assertNotEqual(proc.returncode, 0)
                        self.assertIn("collision", proc.stderr)
                        self.assertEqual(self._hashes(existing_output), before)
                self.assertEqual(self._hashes(source), source_before)

    def test_exports_reject_collisions_before_force_replaces_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for index, names in enumerate((
                ("camera_a/frame.png", "camera_b/frame.png"),
                ("camera_a/Frame.png", "camera_b/frame.PNG"),
                ("frame.png", "frame.png"),
            )):
                source = root / f"source_{index}"
                self._fixture(source, names)
                wrapper = root / f"wrapper_{index}"
                self._import(source, wrapper)
                self._success("validate", "dataset", wrapper, "--split", "val2017", "--strict")
                self.assertEqual(len(build_manifest(wrapper, split="val2017")["images"]), 2)
                for target in ("yolo", "coco", "kitti"):
                    with self.subTest(names=names, target=target):
                        new_output = root / f"new_{index}_{target}"
                        command = (
                            "export-dataset", target, "--dataset", wrapper, "--split", "val2017",
                        )
                        proc = self._run(*command, "--out-dir", new_output)
                        self.assertNotEqual(proc.returncode, 0)
                        self.assertIn("collision", proc.stderr)
                        self.assertFalse(new_output.exists())
                        existing_output = root / f"existing_{index}_{target}"
                        existing_output.mkdir()
                        (existing_output / "keep.txt").write_text("keep", encoding="utf-8")
                        before = self._hashes(existing_output)
                        proc = self._run(*command, "--out-dir", existing_output, "--force")
                        self.assertNotEqual(proc.returncode, 0)
                        self.assertIn("collision", proc.stderr)
                        self.assertEqual(self._hashes(existing_output), before)

    def test_yolo_and_kitti_reject_same_stem_but_coco_preserves_extensions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            self._fixture(source, ("camera_a/frame.jpg", "camera_b/frame.png"))
            wrapper = root / "wrapper"
            self._import(source, wrapper)
            for target in ("yolo", "kitti"):
                output = root / target
                proc = self._run("export-dataset", target, "--dataset", wrapper,
                                 "--split", "val2017", "--out-dir", output)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("label stem collision", proc.stderr)
                self.assertFalse(output.exists())
            output = root / "coco"
            self._success("export-dataset", "coco", "--dataset", wrapper,
                          "--split", "val2017", "--out-dir", output)
            self._success("validate", "dataset", output, "--split", "val2017", "--strict")
            self.assertEqual(len(build_manifest(output, split="val2017")["images"]), 2)

    def test_unique_nested_names_still_materialize_and_export(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            self._fixture(source, ("東京/left.png", "大阪/right.png", "negative.png"))
            source_before = self._hashes(source)
            for mode in ("copy", "symlink"):
                output = root / mode
                self._success("migrate", "dataset", "--from", "coco", "--coco-root", source,
                              "--split", "val2017", "--mode", mode, "--output", output)
                self._success("validate", "dataset", output, "--split", "val2017", "--strict")
                records = build_manifest(output, split="val2017")["images"]
                self.assertEqual(len(records), 3)
                self.assertEqual(sum(len(r["labels"]) for r in records), 2)
                for target in ("yolo", "coco", "kitti"):
                    target_root = root / f"{mode}_{target}"
                    self._success("export-dataset", target, "--dataset", output,
                                  "--split", "val2017", "--out-dir", target_root)
                    image_dir = target_root / ("image_2" if target == "kitti" else "images/val2017")
                    self.assertEqual(len(list(image_dir.glob("*.png"))), 3)
                    for name in ("東京/left.png", "大阪/right.png", "negative.png"):
                        self.assertEqual((source / "images/val2017" / name).read_bytes(),
                                         (image_dir / Path(name).name).read_bytes())
            self.assertEqual(self._hashes(source), source_before)

    def test_copied_yolo_yaml_stays_usable_after_relocation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            self._fixture(source)
            wrapper = root / "wrapper"
            self._import(source, wrapper)
            output = root / "export"
            self._success("export-dataset", "yolo", "--dataset", wrapper,
                          "--split", "val2017", "--out-dir", output)
            yaml_data = simple_yaml_load((output / "data.yaml").read_text(encoding="utf-8"))
            self.assertNotIn("path", yaml_data)
            self.assertIsNone(yaml_data["train"])
            self.assertEqual(yaml_data["val"], "images/val2017")
            before = self._hashes(output)
            moved = root / "relocated dataset with spaces"
            shutil.move(output, moved)
            shutil.move(source, root / "source no longer available")
            self.assertEqual(before, self._hashes(moved))
            self._success("validate", "dataset", moved / "data.yaml", "--split", "val2017", "--strict")
            self._success("doctor", "train-dataset", "--from", "auto", "--dataset", moved / "data.yaml",
                          "--split", "val2017", "--output", root / "doctor.json")
            doctor = json.loads((root / "doctor.json").read_text(encoding="utf-8"))
            self.assertEqual(doctor["records"]["count"], 3)
            self.assertTrue(doctor["reference_trainer"]["direct_train_ready"])
            records = build_manifest(moved / "data.yaml", split="val2017")["images"]
            self.assertTrue(all(Path(record["image"]).resolve().is_relative_to(moved.resolve()) for record in records))
            reimported = root / "reimported"
            self._success("migrate", "dataset", "--from", "ultralytics", "--data", moved / "data.yaml",
                          "--split", "val2017", "--output", reimported)
            self._success("validate", "dataset", reimported, "--split", "val2017", "--strict")

    def test_yolo_train_export_keeps_unexported_validation_split_null(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            images = source / "images/train"
            labels = source / "labels/train"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB", (32, 16)).save(images / "train.png")
            (labels / "train.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
            output = root / "export"
            self._success("export-dataset", "yolo", "--dataset", source,
                          "--split", "train", "--out-dir", output)
            yaml_data = simple_yaml_load((output / "data.yaml").read_text(encoding="utf-8"))
            self.assertNotIn("path", yaml_data)
            self.assertEqual(yaml_data["train"], "images/train")
            self.assertIsNone(yaml_data["val"])
            moved = root / "relocated train dataset"
            shutil.move(output, moved)
            shutil.move(source, root / "source no longer available")
            self._success("validate", "dataset", moved / "data.yaml", "--strict")
            records = build_manifest(moved / "data.yaml")["images"]
            self.assertEqual(len(records), 1)
            self.assertTrue(Path(records[0]["image"]).resolve().is_relative_to(moved.resolve()))


if __name__ == "__main__":
    unittest.main()
