import unittest
from pathlib import PurePosixPath

from yolozu.datasets import dataset_fetch
from yolozu.datasets.registry import list_adapters


class TestDatasetZooManifest(unittest.TestCase):
    def test_builtin_entries_are_unique_complete_and_resolvable(self):
        registry = dataset_fetch.load_dataset_registry()
        self.assertEqual(registry.get("schema_version"), 1)
        datasets = registry.get("datasets")
        self.assertIsInstance(datasets, list)
        self.assertTrue(datasets)

        ids = [str(item.get("id") or "") for item in datasets]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn("", ids)
        self.assertTrue(
            {
                "coco128",
                "coco-val2017",
                "coco-train2017",
                "voc2007",
                "voc2012",
                "cityscapes",
                "ade20k",
                "objects365-val-tiny",
                "lvis-val-v1",
            }.issubset(ids)
        )

        for item in datasets:
            dataset_id = str(item["id"])
            with self.subTest(dataset_id=dataset_id):
                spec = dataset_fetch.resolve_dataset_spec(dataset_id)
                self.assertTrue(spec.summary.strip())
                self.assertTrue(spec.format.strip())
                self.assertTrue(spec.task.strip())
                self.assertNotIn(spec.license.strip().upper(), {"", "UNKNOWN"})
                self.assertIsInstance(spec.num_classes, int)
                self.assertGreater(spec.num_classes, 0)
                self.assertTrue(spec.splits)
                self.assertTrue(spec.tags)
                self.assertEqual(len(spec.urls), len(spec.expected_sha256_by_url))

                source = item.get("source") or {}
                if spec.source_type == "manual":
                    self.assertEqual(spec.urls, [])
                    self.assertTrue(str(source.get("instructions") or "").strip())
                else:
                    self.assertTrue(spec.urls)
                    for url in spec.urls:
                        parsed = dataset_fetch._validated_download_url(url, allow_http=True)
                        self.assertIn(parsed.scheme, {"http", "https"})

                post_extract = spec.post_extract
                self.assertIn(
                    post_extract.get("layout"),
                    {"zip", "tar", "json", "manual"},
                )
                for key in ("root_subdir", "annotations_json", "images_dir"):
                    value = post_extract.get(key)
                    if value is None:
                        continue
                    path = PurePosixPath(str(value))
                    self.assertFalse(path.is_absolute(), f"{dataset_id}.{key} must be relative")
                    self.assertNotIn("..", path.parts, f"{dataset_id}.{key} must not escape its root")

    def test_fetch_formats_have_declared_adapter_or_native_yolo_path(self):
        format_to_adapter = {
            "coco_instances": "coco",
            "pascal_voc": "pascal_voc",
            "cityscapes": "cityscapes",
            "ade20k": "ade20k",
        }
        registered = set(list_adapters())
        for spec in dataset_fetch.list_datasets():
            with self.subTest(dataset_id=spec.dataset_id, format=spec.format):
                if spec.format == "yolo":
                    continue
                self.assertIn(spec.format, format_to_adapter)
                self.assertIn(format_to_adapter[spec.format], registered)


if __name__ == "__main__":
    unittest.main()
