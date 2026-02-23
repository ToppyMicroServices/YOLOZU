import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from yolozu import model_fetch


class TestModelFetchSecurity(unittest.TestCase):
    def test_validated_download_url_accepts_https(self):
        parsed = model_fetch._validated_download_url("https://example.com/model.bin")
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.hostname, "example.com")

    def test_validated_download_url_rejects_non_https_network_scheme(self):
        with self.assertRaises(ValueError):
            model_fetch._validated_download_url("http://example.com/model.bin")

    def test_validated_download_url_rejects_private_ip(self):
        with self.assertRaises(ValueError):
            model_fetch._validated_download_url("https://127.0.0.1/model.bin")

    def test_fetch_model_supports_file_uri_registry_entries(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "weights.bin"
            src.write_bytes(b"abc123")
            sha = hashlib.sha256(src.read_bytes()).hexdigest()

            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "models": [
                            {
                                "id": "toy-model",
                                "summary": "toy",
                                "family": "test",
                                "source": {"type": "official_url", "url": src.resolve().as_uri()},
                                "version": "v1",
                                "license": "Apache-2.0",
                                "sha256": sha,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            out_path, meta_path = model_fetch.fetch_model(
                model_id="toy-model",
                out_dir=root / "models",
                cache_dir=root / "cache",
                accept_license=True,
                registry_path=registry,
            )

            self.assertTrue(out_path.is_file())
            self.assertEqual(out_path.read_bytes(), b"abc123")
            self.assertTrue(meta_path.is_file())


if __name__ == "__main__":
    unittest.main()
