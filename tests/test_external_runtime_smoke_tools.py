import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestExternalRuntimeSmokeTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.hf_script = cls.repo_root / "tools" / "train_hf_detr_runtime_smoke.py"
        cls.detectron_script = cls.repo_root / "tools" / "train_detectron2_runtime_smoke.py"
        cls.hf_module = cls._load("train_hf_detr_runtime_smoke_test", cls.hf_script)
        cls.detectron_module = cls._load(
            "train_detectron2_runtime_smoke_test",
            cls.detectron_script,
        )

    @staticmethod
    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_help_does_not_import_optional_runtimes(self) -> None:
        for script in (self.hf_script, self.detectron_script):
            proc = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("usage:", proc.stdout)

    def test_hf_dataset_tree_hash_excludes_runtime_cache(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as temp_dir:
            root = Path(temp_dir)
            data = root / "dataset"
            data.mkdir()
            (data / "image.jpg").write_bytes(b"image")
            expected = self.hf_module._tree_sha256(data)

            (data / "labels.cache").write_bytes(b"runtime-cache")
            self.assertEqual(self.hf_module._tree_sha256(data), expected)

            (data / "label.txt").write_text("0 0.5 0.5 1 1\n", encoding="utf-8")
            self.assertNotEqual(self.hf_module._tree_sha256(data), expected)

    def test_hf_split_aliases_train2017_and_val2017(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as temp_dir:
            root = Path(temp_dir)
            for split in ("train2017", "val2017"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
            self.assertEqual(
                self.hf_module._resolve_split(root, "train"),
                (root / "images" / "train2017", root / "labels" / "train2017"),
            )
            self.assertEqual(
                self.hf_module._resolve_split(root, "val"),
                (root / "images" / "val2017", root / "labels" / "val2017"),
            )

    def test_detectron_requires_coco_wrapper_descriptor(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as temp_dir:
            root = Path(temp_dir)
            descriptor = root / "dataset.json"
            descriptor.write_text(
                json.dumps({"format": "yolo"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "requires a coco_instances wrapper"):
                self.detectron_module._dataset_descriptor(root)

            descriptor.write_text(
                json.dumps(
                    {
                        "format": "coco_instances",
                        "instances_json": str(root / "instances.json"),
                        "images_dir": str(root / "images"),
                    }
                ),
                encoding="utf-8",
            )
            loaded = self.detectron_module._dataset_descriptor(root)
            self.assertEqual(loaded["format"], "coco_instances")

    def test_manifest_declares_runtime_evidence_outputs(self) -> None:
        manifest = json.loads(
            (self.repo_root / "tools" / "manifest.json").read_text(encoding="utf-8")
        )
        tools = {item["id"]: item for item in manifest["tools"]}
        for tool_id in (
            "train_detectron2_runtime_smoke",
            "train_hf_detr_runtime_smoke",
        ):
            self.assertEqual(tools[tool_id]["maturity"], "experimental")
            output_names = {item["name"] for item in tools[tool_id]["outputs"]}
            self.assertIn("checkpoint", output_names)
            self.assertIn("training_evidence_json", output_names)
            self.assertTrue(tools[tool_id]["effects"]["writes"])


if __name__ == "__main__":
    unittest.main()
