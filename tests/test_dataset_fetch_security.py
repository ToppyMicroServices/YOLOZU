import json
import tempfile
import unittest
from pathlib import Path

from yolozu.datasets import dataset_fetch


class TestDatasetFetchSecurity(unittest.TestCase):
    def test_validated_download_url_accepts_https(self):
        parsed = dataset_fetch._validated_download_url("https://example.com/data.zip")
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.hostname, "example.com")

    def test_validated_download_url_rejects_private_ip(self):
        with self.assertRaises(ValueError):
            dataset_fetch._validated_download_url("https://127.0.0.1/data.zip")

    def test_fetch_dataset_mirror_urls_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src_json = root / "dataset_manifest.json"
            src_json.write_text(json.dumps({"ok": True}), encoding="utf-8")

            bad_uri = (root / "missing_manifest.json").resolve().as_uri()
            good_uri = src_json.resolve().as_uri()

            registry = root / "dataset_registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "datasets": [
                            {
                                "id": "toy-dataset",
                                "summary": "toy",
                                "format": "json",
                                "task": "detection",
                                "license": "Apache-2.0",
                                "source": {
                                    "type": "mirror_urls",
                                    "urls": [bad_uri, good_uri],
                                },
                                "splits": ["val"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            effective_root, meta_path = dataset_fetch.fetch_dataset(
                dataset_id="toy-dataset",
                out_dir=root / "datasets_out",
                cache_dir=root / "cache",
                accept_license=True,
                registry_path=registry,
            )

            self.assertTrue(effective_root.is_dir())
            copied = effective_root / src_json.name
            self.assertTrue(copied.is_file())
            self.assertEqual(json.loads(copied.read_text(encoding="utf-8")), {"ok": True})

            self.assertTrue(meta_path.is_file())
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta.get("source_type"), "mirror_urls")
            self.assertEqual(meta.get("urls"), [good_uri])


if __name__ == "__main__":
    unittest.main()
