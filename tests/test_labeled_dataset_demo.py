import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path, PurePosixPath

from PIL import Image
import yaml

from yolozu import __version__
from yolozu.datasets.dataset import build_manifest
from yolozu.datasets.dataset_validator import validate_dataset_records
from yolozu.demos.labeled_dataset import generate_labeled_sample
from yolozu.eval.simple_map import evaluate_map
from yolozu.predictions import load_predictions_entries, validate_predictions_payload


class TestLabeledDatasetDemo(unittest.TestCase):
    def test_labels_match_visible_integer_geometry_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sample"
            manifest_path = generate_labeled_sample(run_dir=root, seed=7)
            self.assertEqual(manifest_path, root / "sample_manifest.json")
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["format_version"], 1)
            self.assertEqual(manifest["provenance"]["yolozu_version"], __version__)
            self.assertFalse(manifest["provenance"]["model_inference"])
            self.assertEqual(manifest["provenance"]["predictions_source"], "ground_truth_labels")
            self.assertEqual(manifest["counts"]["images"], 8)
            self.assertEqual(manifest["counts"]["objects"], 14)
            filenames = [Path(sample["image"]).name for sample in manifest["samples"]]
            self.assertEqual(len(set(filenames)), 8)
            empty_images = 0
            for sample in manifest["samples"]:
                labels = (root / sample["label"]).read_text().splitlines()
                self.assertEqual(len(labels), len(sample["objects"]))
                with Image.open(root / sample["image"]) as image:
                    self.assertEqual(image.size, (320, 240))
                    self.assertEqual(image.mode, "RGB")
                    for obj, line in zip(sample["objects"], labels):
                        class_id, cx, cy, width, height = [float(value) for value in line.split()]
                        self.assertEqual(class_id, obj["class_id"])
                        x0, y0, x1, y1 = obj["bbox_xyxy"]
                        for actual, expected in zip(
                            ((cx - width / 2) * 320, (cy - height / 2) * 240,
                             (cx + width / 2) * 320, (cy + height / 2) * 240),
                            (x0, y0, x1, y1),
                        ):
                            self.assertAlmostEqual(actual, expected, places=10)
                        color = tuple(manifest["recipe"]["class_colors_rgb"][int(class_id)])
                        mask = Image.new("1", image.size)
                        mask.putdata([pixel == color for pixel in image.get_flattened_data()])
                        self.assertEqual(mask.getbbox(), (x0, y0, x1, y1))
                        mask.close()
                    if not labels:
                        empty_images += 1
                        self.assertEqual(sample["split"], "val")
                        self.assertEqual(image.getextrema(), ((245, 245), (247, 247), (250, 250)))
            self.assertEqual(empty_images, 1)
            for relative, digest in manifest["sha256"].items():
                self.assertEqual(hashlib.sha256((root / relative).read_bytes()).hexdigest(), digest)
            with Image.open(root / manifest["paths"]["preview"]) as preview:
                self.assertEqual(preview.size, (640, 1112))

    def test_repeatable_semantics_and_seed_variation(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [generate_labeled_sample(run_dir=Path(temporary) / name, seed=seed)
                     for name, seed in (("first", 0), ("second", 0), ("third", 1))]
            first, second, third = [json.loads(path.read_text()) for path in paths]
            self.assertEqual(first, second)
            self.assertNotEqual(first["samples"], third["samples"])
            # Recipe-v1 geometry is stable even if PNG encoders or package versions change.
            geometry = json.dumps(first["samples"], sort_keys=True, separators=(",", ":")).encode()
            self.assertEqual(
                hashlib.sha256(geometry).hexdigest(),
                "b4c53efedfcd2345cf35f410267b883de9f77710289e85124235e72d944bab48",
            )

    def test_strict_validation_evaluation_and_portable_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = Path(temporary) / "original"
            manifest_path = generate_labeled_sample(run_dir=str(original))
            manifest = json.loads(manifest_path.read_text())
            yaml_data = yaml.safe_load((original / "data.yaml").read_text())
            self.assertEqual(yaml_data, {"path": ".", "train": "images/train", "val": "images/val", "names": {0: "circle", 1: "rectangle"}})
            payload = json.loads((original / "predictions.json").read_text())
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(validate_predictions_payload(payload, strict=True).warnings, [])
            self.assertEqual(len(payload["predictions"]), 4)
            for entry in payload["predictions"]:
                self.assertEqual(entry["schema_version"], 2)
            relative_paths = list(manifest["paths"].values()) + list(manifest["sha256"])
            relative_paths.extend(entry["image"] for entry in payload["predictions"])
            for value in relative_paths:
                path = PurePosixPath(value)
                self.assertFalse(path.is_absolute())
                self.assertNotIn("..", path.parts)
                self.assertNotIn("\\", value)
                self.assertTrue((original / value).exists())
            relocated = Path(temporary) / "moved sample"
            original.rename(relocated)
            for source in (relocated, relocated / "data.yaml"):
                for split in ("train", "val"):
                    records = build_manifest(source, split=split)["images"]
                    self.assertEqual(len(records), 4)
                    result = validate_dataset_records(records, strict=True)
                    self.assertTrue(result.ok(), result.errors)
                    classes = json.loads((relocated / "labels" / split / "classes.json").read_text())
                    self.assertEqual(classes["names"], ["circle", "rectangle"])
                    self.assertEqual(classes["class_to_category_id"], {"0": 1, "1": 2})
                    if split == "val":
                        metrics = evaluate_map(records, load_predictions_entries(relocated / "predictions.json"))
                        self.assertEqual(metrics.map50, 1.0)
                        self.assertEqual(metrics.map50_95, 1.0)

    def test_existing_paths_and_symlinks_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            existing_dir = base / "directory"
            existing_dir.mkdir()
            sentinel = existing_dir / "sentinel.txt"
            sentinel.write_text("keep me")
            existing_file = base / "file"
            existing_file.write_text("keep file")
            link = base / "link"
            link.symlink_to(existing_dir, target_is_directory=True)
            dangling = base / "dangling"
            dangling.symlink_to(base / "not-created", target_is_directory=True)
            for path in (existing_dir, existing_file, link, dangling):
                with self.subTest(path=path.name):
                    with self.assertRaises(FileExistsError):
                        generate_labeled_sample(run_dir=path)
            self.assertEqual(sentinel.read_text(), "keep me")
            self.assertEqual(list(existing_dir.iterdir()), [sentinel])
            self.assertEqual(existing_file.read_text(), "keep file")
            self.assertTrue(link.is_symlink())
            self.assertTrue(dangling.is_symlink())
            self.assertFalse((base / "not-created").exists())

    def test_relative_output_loads_via_absolute_yaml_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "relative-input"
            relative = Path(os.path.relpath(root, Path.cwd()))
            manifest_path = generate_labeled_sample(run_dir=relative)
            yaml_path = (manifest_path.parent / "data.yaml").resolve()
            self.assertTrue(yaml_path.is_absolute())
            for source in (manifest_path.parent / "data.yaml", yaml_path):
                for split in ("train", "val"):
                    records = build_manifest(source, split=split)["images"]
                    self.assertEqual(len(records), 4)
                    result = validate_dataset_records(records, strict=True)
                    self.assertTrue(result.ok(), result.errors)

    def test_invalid_seed_does_not_create_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sample"
            for seed in (True, 1.2, "0", None):
                with self.subTest(seed=seed):
                    with self.assertRaisesRegex(TypeError, "seed must be an integer"):
                        generate_labeled_sample(run_dir=root, seed=seed)
                    self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
