import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from yolozu.inference import model_fetch


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
                                "size_bytes": len(src.read_bytes()),
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
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["size_bytes"], len(b"abc123"))

    def test_fetch_model_rejects_size_mismatch_before_cache_publish(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "weights.bin"
            src.write_bytes(b"abc123")
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "models": [
                            {
                                "id": "wrong-size-model",
                                "summary": "wrong size",
                                "family": "test",
                                "source": {
                                    "type": "official_url",
                                    "url": src.resolve().as_uri(),
                                },
                                "version": "v1",
                                "license": "Apache-2.0",
                                "sha256": hashlib.sha256(
                                    src.read_bytes()
                                ).hexdigest(),
                                "size_bytes": len(src.read_bytes()) + 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                model_fetch.fetch_model(
                    model_id="wrong-size-model",
                    out_dir=root / "models",
                    cache_dir=root / "cache",
                    accept_license=True,
                    registry_path=registry,
                )

            self.assertFalse(
                (root / "cache" / "wrong-size-model" / "v1" / "weights.bin").exists()
            )
            self.assertFalse(
                (root / "models" / "wrong-size-model" / "meta.json").exists()
            )

    def test_fetch_model_mirror_urls_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "weights.bin"
            src.write_bytes(b"weights-from-mirror")
            sha = hashlib.sha256(src.read_bytes()).hexdigest()

            bad_uri = (root / "missing.bin").resolve().as_uri()
            good_uri = src.resolve().as_uri()

            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "models": [
                            {
                                "id": "toy-mirror-model",
                                "summary": "toy mirror",
                                "family": "test",
                                "source": {
                                    "type": "mirror_urls",
                                    "urls": [bad_uri, good_uri],
                                },
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
                model_id="toy-mirror-model",
                out_dir=root / "models",
                cache_dir=root / "cache",
                accept_license=True,
                registry_path=registry,
            )

            self.assertTrue(out_path.is_file())
            self.assertEqual(out_path.read_bytes(), b"weights-from-mirror")
            self.assertTrue(meta_path.is_file())
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta.get("source_url_used"), good_uri)


if __name__ == "__main__":
    unittest.main()
