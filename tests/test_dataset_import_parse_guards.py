from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDatasetImportParseGuards(unittest.TestCase):
    def test_build_manifest_ignores_invalid_dataset_json_and_sidecar(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        from yolozu.dataset import build_manifest

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            (root / "dataset.json").write_text("{bad", encoding="utf-8")
            images = root / "images" / "train2017"
            labels = root / "labels" / "train2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            (images / "000000.jpg").write_bytes(b"")
            (labels / "000000.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (labels / "000000.json").write_text("{bad", encoding="utf-8")

            manifest = build_manifest(root, split="train2017")

            self.assertEqual(manifest.get("split"), "train2017")
            self.assertEqual(len(manifest.get("images") or []), 1)

    def test_load_coco_instances_dataset_skips_invalid_image_and_annotation_ids(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        from yolozu.datasets.dataset import load_coco_instances_dataset

        payload = {
            "images": [
                {"id": "bad", "file_name": "bad.jpg", "width": 10, "height": 10},
                {"id": 1, "file_name": "0001.jpg", "width": 100, "height": 80},
            ],
            "annotations": [
                {"image_id": "bad", "category_id": 7, "bbox": [0, 0, 10, 20]},
                {"image_id": 1, "category_id": "bad", "bbox": [0, 0, 10, 20]},
                {"image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 20]},
            ],
            "categories": [{"id": 7, "name": "thing"}],
        }

        records = load_coco_instances_dataset(payload, images_dir=repo_root, include_crowd=False)

        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0]["labels"]), 1)

    def test_project_yolox_exp_wraps_runtime_failure(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        from yolozu.datasets.imports import project_yolox_exp

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            config = Path(td) / "exp.py"
            config.write_text("raise NameError('boom')\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                project_yolox_exp(config=config)

    def test_load_config_yaml_falls_back_without_pyyaml(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        from yolozu.datasets import imports as module

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            path = Path(td) / "data.yaml"
            path.write_text("foo: bar\n", encoding="utf-8")

            original_import = __import__

            def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "yaml":
                    raise ImportError("yaml missing")
                return original_import(name, globals, locals, fromlist, level)

            import builtins

            with mock.patch.object(builtins, "__import__", side_effect=fake_import):
                loaded = module._load_config(path)

            self.assertEqual(loaded.get("foo"), "bar")

    def test_dataset_fetch_mirror_urls_retries_expected_errors(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        from yolozu.datasets import dataset_fetch

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
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
                                "id": "toy-dataset-2",
                                "summary": "toy",
                                "format": "json",
                                "task": "detection",
                                "license": "Apache-2.0",
                                "source": {"type": "mirror_urls", "urls": [bad_uri, good_uri]},
                                "splits": ["val"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            effective_root, meta_path = dataset_fetch.fetch_dataset(
                dataset_id="toy-dataset-2",
                out_dir=root / "datasets_out",
                cache_dir=root / "cache",
                accept_license=True,
                registry_path=registry,
            )

            self.assertTrue(effective_root.is_dir())
            self.assertTrue(meta_path.is_file())


if __name__ == "__main__":
    unittest.main()
