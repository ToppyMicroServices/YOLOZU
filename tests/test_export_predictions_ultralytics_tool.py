import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class _FakeResult:
    def __init__(self, path: str) -> None:
        self.path = path
        self.boxes = None


class TestExportPredictionsUltralyticsTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        script = cls.repo_root / "tools" / "export_predictions_ultralytics.py"
        spec = importlib.util.spec_from_file_location("export_predictions_ultralytics_bounded", script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load {script}")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    @staticmethod
    def _write_dataset(root: Path, *, image_count: int = 3) -> Path:
        dataset = root / "dataset"
        images = dataset / "images" / "val"
        labels = dataset / "labels" / "val"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(image_count):
            stem = f"{index + 1:06d}"
            (images / f"{stem}.jpg").write_bytes(b"fake-image")
            (labels / f"{stem}.txt").write_text("", encoding="utf-8")
        return dataset

    @staticmethod
    def _fake_ultralytics(
        capture: dict[str, object],
        *,
        result_limit: int | None = None,
    ) -> types.ModuleType:
        fake_module = types.ModuleType("ultralytics")
        fake_module.__version__ = "test"

        class FakeYOLO:
            def __init__(self, model: str) -> None:
                capture["model"] = model

            def predict(self, **kwargs):
                sources = list(kwargs["source"])
                capture["sources"] = sources
                capture["predict_kwargs"] = kwargs
                selected = sources if result_limit is None else sources[:result_limit]
                return [_FakeResult(path) for path in selected]

        fake_module.YOLO = FakeYOLO
        return fake_module

    def _run(
        self,
        *,
        dataset: Path,
        output: Path,
        fake_module: types.ModuleType,
        extra_args: list[str],
    ) -> None:
        argv = [
            "--model",
            "fake.pt",
            "--dataset",
            str(dataset),
            "--split",
            "val",
            "--wrap",
            "--output",
            str(output),
            *extra_args,
        ]
        with mock.patch.dict(sys.modules, {"ultralytics": fake_module}):
            self.module.main(argv)

    def test_max_images_bounds_runtime_sources_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            root = Path(td)
            dataset = self._write_dataset(root, image_count=3)
            output = root / "predictions.json"
            capture: dict[str, object] = {}

            self._run(
                dataset=dataset,
                output=output,
                fake_module=self._fake_ultralytics(capture),
                extra_args=["--max-images", "2"],
            )

            expected_paths = [
                str((dataset / "images" / "val" / "000001.jpg").resolve()),
                str((dataset / "images" / "val" / "000002.jpg").resolve()),
            ]
            self.assertEqual(capture.get("sources"), expected_paths)

            payload = json.loads(output.read_text(encoding="utf-8"))
            predictions = payload["predictions"]
            extra = payload["meta"]["extra"]
            self.assertEqual(len(predictions), 2)
            self.assertEqual([item["image"] for item in predictions], extra["selected_inputs"])
            self.assertEqual(extra["selected_input_count"], 2)
            self.assertEqual(extra["result_count"], 2)
            self.assertEqual(extra["inference_calls"], 2)
            self.assertTrue(extra["runtime_executed"])
            self.assertEqual(extra["execution_status"], "completed")
            self.assertEqual(extra["source_mode"], "dataset_manifest")

    def test_manifest_path_prefers_dataset_over_same_named_cwd_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            root = Path(td)
            dataset = root / "dataset"
            dataset.mkdir()
            expected = dataset / "collision.jpg"
            expected.write_bytes(b"dataset-image")
            cwd = root / "cwd"
            cwd.mkdir()
            (cwd / "collision.jpg").write_bytes(b"unrelated-image")

            with mock.patch.object(self.module.Path, "cwd", return_value=cwd):
                resolved = self.module._manifest_image_path(
                    dataset=str(dataset),
                    image="collision.jpg",
                )

            self.assertEqual(resolved, expected.resolve())

    def test_manifest_path_accepts_build_manifest_relative_prefix(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            root = Path(td)
            expected = root / "dataset" / "images" / "val" / "image.jpg"
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"dataset-image")
            nested_decoy = root / "dataset" / "dataset" / "images" / "val" / "image.jpg"
            nested_decoy.parent.mkdir(parents=True)
            nested_decoy.write_bytes(b"unrelated-image")

            with mock.patch.object(self.module.Path, "cwd", return_value=root):
                resolved = self.module._manifest_image_path(
                    dataset="dataset",
                    image="dataset/images/val/image.jpg",
                )

            self.assertEqual(resolved, expected.resolve())

    def test_result_count_mismatch_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            root = Path(td)
            dataset = self._write_dataset(root, image_count=3)
            output = root / "predictions.json"
            capture: dict[str, object] = {}

            with self.assertRaisesRegex(SystemExit, "result count does not match selected input count"):
                self._run(
                    dataset=dataset,
                    output=output,
                    fake_module=self._fake_ultralytics(capture, result_limit=1),
                    extra_args=["--max-images", "2"],
                )

            self.assertEqual(len(capture.get("sources") or []), 2)
            self.assertFalse(output.exists())

    def test_zero_selected_inputs_fails_before_runtime(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            root = Path(td)
            dataset = self._write_dataset(root, image_count=2)
            output = root / "predictions.json"
            capture: dict[str, object] = {}

            with self.assertRaisesRegex(SystemExit, "no input images selected"):
                self._run(
                    dataset=dataset,
                    output=output,
                    fake_module=self._fake_ultralytics(capture),
                    extra_args=["--max-images", "0"],
                )

            self.assertNotIn("model", capture)
            self.assertFalse(output.exists())

    def test_explicit_source_rejects_max_images(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            root = Path(td)
            dataset = self._write_dataset(root, image_count=2)
            output = root / "predictions.json"
            capture: dict[str, object] = {}
            source = dataset / "images" / "val" / "000001.jpg"

            with self.assertRaisesRegex(SystemExit, "--source cannot be combined with --max-images"):
                self._run(
                    dataset=dataset,
                    output=output,
                    fake_module=self._fake_ultralytics(capture),
                    extra_args=["--source", str(source), "--max-images", "1"],
                )

            self.assertNotIn("model", capture)
            self.assertFalse(output.exists())

    def test_explicit_source_directory_is_expanded_deterministically(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            root = Path(td)
            dataset = self._write_dataset(root, image_count=1)
            source = root / "explicit"
            source.mkdir()
            (source / "z.jpg").write_bytes(b"fake-image")
            (source / "a.png").write_bytes(b"fake-image")
            (source / "ignored.txt").write_text("not an image", encoding="utf-8")
            output = root / "predictions.json"
            capture: dict[str, object] = {}

            self._run(
                dataset=dataset,
                output=output,
                fake_module=self._fake_ultralytics(capture),
                extra_args=["--source", str(source)],
            )

            expected_paths = [
                str((source / "a.png").resolve()),
                str((source / "z.jpg").resolve()),
            ]
            self.assertEqual(capture.get("sources"), expected_paths)
            payload = json.loads(output.read_text(encoding="utf-8"))
            extra = payload["meta"]["extra"]
            self.assertEqual(extra["source_mode"], "explicit_source")
            self.assertEqual(extra["selected_inputs"], expected_paths)
            self.assertEqual(extra["selected_input_count"], 2)
            self.assertEqual(extra["result_count"], 2)


if __name__ == "__main__":
    unittest.main()
