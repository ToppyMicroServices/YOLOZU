import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from yolozu.datasets import dataset_fetch


class TestDatasetFetchSecurity(unittest.TestCase):
    @staticmethod
    def _write_registry(
        path: Path,
        *,
        source: dict[str, object],
        sha256: str | None = None,
    ) -> None:
        path.write_text(
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
                            "source": source,
                            "sha256": sha256,
                            "splits": ["val"],
                            "tags": ["test"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_validated_download_url_accepts_https(self):
        parsed = dataset_fetch._validated_download_url("https://example.com/data.zip")
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.hostname, "example.com")

    def test_validated_download_url_rejects_private_ip(self):
        with self.assertRaises(ValueError):
            dataset_fetch._validated_download_url("https://127.0.0.1/data.zip")

    def test_resolve_dataset_spec_rejects_invalid_sha256(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = root / "dataset_registry.json"
            self._write_registry(
                registry,
                source={"type": "url", "url": (root / "data.json").resolve().as_uri()},
                sha256="not-a-sha256",
            )

            with self.assertRaisesRegex(ValueError, "64-character hexadecimal sha256"):
                dataset_fetch.resolve_dataset_spec("toy-dataset", registry)

    def test_fetch_dataset_verifies_and_records_sha256(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src_json = root / "dataset_manifest.json"
            src_json.write_text(json.dumps({"ok": True}), encoding="utf-8")
            digest = hashlib.sha256(src_json.read_bytes()).hexdigest()
            registry = root / "dataset_registry.json"
            self._write_registry(
                registry,
                source={"type": "url", "url": src_json.resolve().as_uri()},
                sha256=digest.upper(),
            )

            effective_root, meta_path = dataset_fetch.fetch_dataset(
                dataset_id="toy-dataset",
                out_dir=root / "datasets_out",
                cache_dir=root / "cache",
                accept_license=True,
                registry_path=registry,
            )

            self.assertEqual((effective_root / src_json.name).read_bytes(), src_json.read_bytes())
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["sha256"], digest)
            self.assertEqual(
                meta["artifacts"],
                [
                    {
                        "url": src_json.resolve().as_uri(),
                        "sha256": digest,
                        "expected_sha256": digest,
                    }
                ],
            )

    def test_fetch_dataset_rejects_downloaded_sha256_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src_json = root / "dataset_manifest.json"
            src_json.write_text(json.dumps({"ok": True}), encoding="utf-8")
            registry = root / "dataset_registry.json"
            self._write_registry(
                registry,
                source={"type": "url", "url": src_json.resolve().as_uri()},
                sha256="0" * 64,
            )

            with self.assertRaisesRegex(RuntimeError, "sha256 mismatch"):
                dataset_fetch.fetch_dataset(
                    dataset_id="toy-dataset",
                    out_dir=root / "datasets_out",
                    cache_dir=root / "cache",
                    accept_license=True,
                    registry_path=registry,
                )
            self.assertFalse((root / "cache" / "toy-dataset" / src_json.name).exists())

    def test_fetch_dataset_rechecks_cached_sha256(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src_json = root / "dataset_manifest.json"
            src_json.write_text(json.dumps({"ok": True}), encoding="utf-8")
            digest = hashlib.sha256(src_json.read_bytes()).hexdigest()
            registry = root / "dataset_registry.json"
            self._write_registry(
                registry,
                source={"type": "url", "url": src_json.resolve().as_uri()},
                sha256=digest,
            )
            kwargs = {
                "dataset_id": "toy-dataset",
                "out_dir": root / "datasets_out",
                "cache_dir": root / "cache",
                "accept_license": True,
                "registry_path": registry,
            }
            dataset_fetch.fetch_dataset(**kwargs)
            (root / "cache" / "toy-dataset" / src_json.name).write_bytes(b"corrupt")

            with self.assertRaisesRegex(RuntimeError, "cached sha256 mismatch"):
                dataset_fetch.fetch_dataset(**kwargs)

    def test_multi_source_uses_per_part_sha256(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            parts: list[dict[str, str]] = []
            expected: list[str] = []
            for name in ("images.json", "annotations.json"):
                path = root / name
                path.write_text(json.dumps({"name": name}), encoding="utf-8")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                expected.append(digest)
                parts.append({"name": name, "url": path.resolve().as_uri(), "sha256": digest})
            registry = root / "dataset_registry.json"
            self._write_registry(registry, source={"type": "multi", "parts": parts})

            spec = dataset_fetch.resolve_dataset_spec("toy-dataset", registry)
            self.assertEqual(spec.expected_sha256_by_url, expected)
            effective_root, meta_path = dataset_fetch.fetch_dataset(
                dataset_id="toy-dataset",
                out_dir=root / "datasets_out",
                cache_dir=root / "cache",
                accept_license=True,
                registry_path=registry,
            )
            self.assertTrue((effective_root / "images.json").is_file())
            self.assertTrue((effective_root / "annotations.json").is_file())
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual([item["sha256"] for item in meta["artifacts"]], expected)

    def test_multi_source_rejects_ambiguous_dataset_level_sha256(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            parts = [
                {"name": name, "url": (root / name).resolve().as_uri()}
                for name in ("images.json", "annotations.json")
            ]
            registry = root / "dataset_registry.json"
            self._write_registry(
                registry,
                source={"type": "multi", "parts": parts},
                sha256="0" * 64,
            )

            with self.assertRaisesRegex(ValueError, "ambiguous for a multi-part source"):
                dataset_fetch.resolve_dataset_spec("toy-dataset", registry)

    def test_fetch_dataset_mirror_urls_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src_json = root / "dataset_manifest.json"
            src_json.write_text(json.dumps({"ok": True}), encoding="utf-8")
            digest = hashlib.sha256(src_json.read_bytes()).hexdigest()

            bad_json = root / "bad_manifest.json"
            bad_json.write_text(json.dumps({"ok": False}), encoding="utf-8")
            bad_uri = bad_json.resolve().as_uri()
            good_uri = src_json.resolve().as_uri()

            registry = root / "dataset_registry.json"
            self._write_registry(
                registry,
                source={"type": "mirror_urls", "urls": [bad_uri, good_uri]},
                sha256=digest,
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
            self.assertEqual(meta["artifacts"][0]["sha256"], digest)


if __name__ == "__main__":
    unittest.main()
