import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from yolozu.cli_commands import _validate_segmentation_layout
from yolozu.core.image_size import get_image_size
from yolozu.datasets.pascal_voc import iter_pascal_voc_seg_samples
from yolozu.datasets.registry import iter_samples, list_adapters, probe_format


class TestDatasetAdapterConformance(unittest.TestCase):
    def assert_readable_pair(
        self,
        image_path: Path,
        mask_path: Path,
        *,
        expected_values: set[int],
    ) -> None:
        self.assertEqual(get_image_size(image_path), get_image_size(mask_path))
        with Image.open(mask_path) as mask:
            values = {value for value, count in enumerate(mask.histogram()) if count}
            self.assertEqual(values, expected_values)

    def test_builtin_adapter_registry_is_available(self):
        self.assertEqual(
            list_adapters(),
            ["ade20k", "cityscapes", "coco", "pascal_voc"],
        )

    def test_coco_detection_handles_nested_unicode_path_and_crowd_policy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            images_dir = root / "images" / "val2017"
            annotations_dir = root / "annotations"
            image_path = images_dir / "東京" / "scene one.png"
            image_path.parent.mkdir(parents=True)
            annotations_dir.mkdir(parents=True)
            Image.new("RGB", (40, 20), color=(12, 34, 56)).save(image_path)

            annotations = {
                "images": [
                    {
                        "id": 1,
                        "file_name": "東京/scene one.png",
                        "width": 40,
                        "height": 20,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 17,
                        "bbox": [4, 2, 20, 10],
                        "iscrowd": 0,
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 5,
                        "bbox": [0, 0, 8, 4],
                        "iscrowd": 1,
                    },
                ],
                "categories": [
                    {"id": 17, "name": "cat"},
                    {"id": 5, "name": "airplane"},
                ],
            }
            (annotations_dir / "instances_val2017.json").write_text(
                json.dumps(annotations),
                encoding="utf-8",
            )

            info = probe_format(root)
            self.assertIsNotNone(info)
            self.assertEqual(info.format_name, "coco")
            self.assertIn("val2017", info.splits)

            samples = list(iter_samples(root, split="val2017"))
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].image_path, image_path)
            self.assertEqual(get_image_size(samples[0].image_path), (40, 20))
            self.assertEqual([label["class_id"] for label in samples[0].labels], [1])
            self.assertAlmostEqual(samples[0].labels[0]["cx"], 0.35)
            self.assertAlmostEqual(samples[0].labels[0]["cy"], 0.35)
            self.assertAlmostEqual(samples[0].labels[0]["w"], 0.5)
            self.assertAlmostEqual(samples[0].labels[0]["h"], 0.5)

            samples_with_crowd = list(
                iter_samples(root, split="val2017", include_crowd=True)
            )
            self.assertEqual(
                {label["class_id"] for label in samples_with_crowd[0].labels},
                {0, 1},
            )

    def test_pascal_voc_detection_and_segmentation_share_valid_assets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "VOCdevkit" / "VOC2012"
            images_dir = root / "JPEGImages"
            annotations_dir = root / "Annotations"
            masks_dir = root / "SegmentationClass"
            main_dir = root / "ImageSets" / "Main"
            seg_dir = root / "ImageSets" / "Segmentation"
            for path in (images_dir, annotations_dir, masks_dir, main_dir, seg_dir):
                path.mkdir(parents=True, exist_ok=True)

            sample_id = "sample_α"
            image_path = images_dir / f"{sample_id}.jpg"
            mask_path = masks_dir / f"{sample_id}.png"
            Image.new("RGB", (40, 20), color=(90, 80, 70)).save(image_path)
            mask = Image.new("L", (40, 20), color=0)
            mask.putpixel((1, 1), 15)
            mask.putpixel((2, 1), 255)
            mask.save(mask_path)
            (main_dir / "val.txt").write_text(f"{sample_id}\n", encoding="utf-8")
            (seg_dir / "val.txt").write_text(f"{sample_id}\n", encoding="utf-8")
            (annotations_dir / f"{sample_id}.xml").write_text(
                """<annotation>
  <filename>sample_α.jpg</filename>
  <size><width>40</width><height>20</height><depth>3</depth></size>
  <object>
    <name>person</name><difficult>0</difficult>
    <bndbox><xmin>4</xmin><ymin>2</ymin><xmax>24</xmax><ymax>12</ymax></bndbox>
  </object>
  <object>
    <name>dog</name><difficult>1</difficult>
    <bndbox><xmin>10</xmin><ymin>5</ymin><xmax>20</xmax><ymax>15</ymax></bndbox>
  </object>
</annotation>
""",
                encoding="utf-8",
            )

            info = probe_format(root)
            self.assertIsNotNone(info)
            self.assertEqual(info.format_name, "pascal_voc")
            self.assertEqual(info.task, "multi")

            detection = list(iter_samples(root, split="val"))
            self.assertEqual(len(detection), 1)
            self.assertEqual([label["class_id"] for label in detection[0].labels], [14])
            self.assertAlmostEqual(detection[0].labels[0]["cx"], 0.35)
            self.assertAlmostEqual(detection[0].labels[0]["cy"], 0.35)

            with_difficult = list(
                iter_samples(root, split="val", include_difficult=True)
            )
            self.assertEqual(
                {label["class_id"] for label in with_difficult[0].labels},
                {11, 14},
            )

            segmentation = list(iter_pascal_voc_seg_samples(root, split="val"))
            self.assertEqual(len(segmentation), 1)
            self.assert_readable_pair(
                segmentation[0].image_path,
                segmentation[0].mask_path,
                expected_values={0, 15, 255},
            )
            _, errors = _validate_segmentation_layout(
                dataset_path=root,
                layout_info={"format": "voc_segmentation_root", "split": "val"},
                max_images=None,
            )
            self.assertEqual(errors, [])

    def test_cityscapes_pairs_pixel_valid_mask_for_unicode_city(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            city = "東京"
            base = "東京_000000_000001"
            image_path = root / "leftImg8bit" / "val" / city / f"{base}_leftImg8bit.png"
            mask_path = root / "gtFine" / "val" / city / f"{base}_gtFine_labelTrainIds.png"
            image_path.parent.mkdir(parents=True)
            mask_path.parent.mkdir(parents=True)
            Image.new("RGB", (32, 16), color=(20, 30, 40)).save(image_path)
            mask = Image.new("L", (32, 16), color=0)
            mask.putpixel((1, 1), 13)
            mask.putpixel((2, 1), 255)
            mask.save(mask_path)

            info = probe_format(root)
            self.assertIsNotNone(info)
            self.assertEqual(info.format_name, "cityscapes")
            samples = list(iter_samples(root, split="val"))
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].extra["city"], city)
            self.assert_readable_pair(
                samples[0].image_path,
                samples[0].mask_path,
                expected_values={0, 13, 255},
            )
            _, errors = _validate_segmentation_layout(
                dataset_path=root,
                layout_info={"format": "cityscapes_segmentation_root", "split": "val"},
                max_images=None,
            )
            self.assertEqual(errors, [])

            Image.new("L", (31, 16), color=0).save(mask_path)
            _, errors = _validate_segmentation_layout(
                dataset_path=root,
                layout_info={"format": "cityscapes_segmentation_root", "split": "val"},
                max_images=None,
            )
            self.assertTrue(any("image/mask size mismatch" in error for error in errors))

    def test_ade20k_pairs_pixel_valid_mask_for_filename_with_spaces(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ADEChallengeData2016"
            images_dir = root / "images" / "validation"
            masks_dir = root / "annotations" / "validation"
            images_dir.mkdir(parents=True)
            masks_dir.mkdir(parents=True)
            image_path = images_dir / "scene ü.png"
            mask_path = masks_dir / "scene ü.png"
            Image.new("RGB", (24, 12), color=(50, 60, 70)).save(image_path)
            mask = Image.new("L", (24, 12), color=1)
            mask.putpixel((1, 1), 150)
            mask.putpixel((2, 1), 255)
            mask.save(mask_path)

            info = probe_format(root.parent)
            self.assertIsNotNone(info)
            self.assertEqual(info.format_name, "ade20k")
            samples = list(iter_samples(root.parent, split="val"))
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].sample_id, "scene ü")
            self.assert_readable_pair(
                samples[0].image_path,
                samples[0].mask_path,
                expected_values={1, 150, 255},
            )
            _, errors = _validate_segmentation_layout(
                dataset_path=root.parent,
                layout_info={"format": "ade20k_segmentation_root", "split": "val"},
                max_images=None,
            )
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
