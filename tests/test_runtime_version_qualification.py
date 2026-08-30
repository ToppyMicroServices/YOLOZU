import importlib.util
import re
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]


class TestRuntimeVersionQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.versions = (cls.repo_root / "docs" / "versions.md").read_text(
            encoding="utf-8"
        )
        cls.production_readiness = (
            cls.repo_root / "docs" / "production_readiness.md"
        ).read_text(encoding="utf-8")
        cls.manual = (
            cls.repo_root / "manual" / "chapters" / "17_realtime_batch_inference.tex"
        ).read_text(encoding="utf-8")
        with (cls.repo_root / "pyproject.toml").open("rb") as handle:
            cls.pyproject = tomllib.load(handle)

    def _lock_pin(self, relative_path: str, package: str) -> str:
        text = (self.repo_root / relative_path).read_text(encoding="utf-8")
        match = re.search(
            rf"(?m)^{re.escape(package)}==([^;\s]+)",
            text,
        )
        self.assertIsNotNone(match, f"missing {package} pin in {relative_path}")
        return str(match.group(1))

    def _extra_requirements(self, package: str) -> list[str]:
        prefix = f"{package}>="
        return [
            item
            for requirements in self.pyproject["project"][
                "optional-dependencies"
            ].values()
            for item in requirements
            if item.startswith(prefix)
        ]

    def test_documented_package_floors_match_pyproject(self) -> None:
        expected = {
            "torch": "torch>=2.10.0",
            "torchvision": "torchvision>=0.25.0",
            "onnx": "onnx>=1.21.0",
            "onnxruntime": "onnxruntime>=1.17",
        }
        for package, requirement in expected.items():
            with self.subTest(package=package):
                matches = self._extra_requirements(package)
                self.assertGreater(len(matches), 0)
                self.assertEqual(set(matches), {requirement})
                self.assertIn(f"`{requirement}`", self.versions)

    def test_documented_pins_match_lock_files(self) -> None:
        expected = (
            ("requirements-locks/requirements-ci.lock", "torch", "2.10.0+cpu"),
            (
                "requirements-locks/requirements-demo-extra.lock",
                "torch",
                "2.10.0",
            ),
            (
                "requirements-locks/requirements-rtdetr-pose-image-extra.lock",
                "torch",
                "2.10.0",
            ),
            (
                "requirements-locks/requirements-demo-extra.lock",
                "torchvision",
                "0.25.0",
            ),
            ("requirements-locks/requirements-ci.lock", "onnx", "1.21.0"),
            ("requirements-locks/requirements-trt-tools.lock", "onnx", "1.21.0"),
            (
                "requirements-locks/requirements-rtdetr-pose-extra.lock",
                "onnx",
                "1.21.0",
            ),
            (
                "requirements-locks/requirements-rtdetr-pose-image-extra.lock",
                "onnx",
                "1.21.0",
            ),
            (
                "requirements-locks/requirements-ci.lock",
                "onnxruntime",
                "1.24.2",
            ),
            (
                "requirements-locks/requirements-trt-tools.lock",
                "onnxruntime",
                "1.24.2",
            ),
            (
                "requirements-locks/requirements-rtdetr-pose-extra.lock",
                "onnxruntime",
                "1.24.3",
            ),
            (
                "requirements-locks/requirements-rtdetr-pose-image-extra.lock",
                "onnxruntime-gpu",
                "1.24.4",
            ),
        )
        for relative_path, package, version in expected:
            with self.subTest(path=relative_path, package=package):
                self.assertEqual(
                    self._lock_pin(relative_path, package),
                    version,
                )
                self.assertIn(f"`{package}=={version}`", self.versions)

    def test_documented_opset_default_matches_cli(self) -> None:
        script = self.repo_root / "tools" / "export_trt.py"
        spec = importlib.util.spec_from_file_location(
            "_test_runtime_versions_export_trt",
            script,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        default_opset = module._parse_args([]).opset
        self.assertIn(f"defaults to ONNX opset `{default_opset}`", self.versions)
        self.assertIn("explicitly pin opset `17`", self.versions)

    def test_gpu_container_is_configured_not_claimed_as_completed(self) -> None:
        workflow = (
            self.repo_root / ".github" / "workflows" / "ngc_test.yml"
        ).read_text(encoding="utf-8")
        image = "nvcr.io/nvidia/tensorrt:24.08-py3"
        self.assertIn(image, workflow)
        self.assertIn(f"`{image}`", self.versions)
        self.assertIn("configured", self.versions)
        self.assertIn("Without those run artifacts", self.versions)

    def test_source_and_packaged_tensorrt_docs_share_version_boundary(self) -> None:
        source = (self.repo_root / "docs" / "tensorrt_pipeline.md").read_text(
            encoding="utf-8"
        )
        packaged = (
            self.repo_root / "yolozu" / "data" / "docs" / "tensorrt_pipeline.md"
        ).read_text(encoding="utf-8")
        expected = (
            "`tools/export_trt.py` itself defaults to opset 18. "
            "Do not treat either number as\n"
            "a universal TensorRT requirement"
        )
        self.assertIn(expected, source)
        self.assertIn(expected, packaged)
        checkpoint_boundary = (
            "Checkpoint loading is fail closed. The exporter requires a full "
            "name-and-shape\nmatch"
        )
        self.assertIn(checkpoint_boundary, source)
        self.assertIn(checkpoint_boundary, packaged)

    def test_production_and_manual_share_qualification_boundary(self) -> None:
        production = " ".join(self.production_readiness.split())
        manual = " ".join(self.manual.split())

        self.assertIn("`tools/export_trt.py` defaults to opset 18", production)
        self.assertIn("explicitly pin opset 17", production)
        self.assertIn("No universal static version pair", production)
        self.assertIn("run-specific container, driver/GPU, CUDA context", production)

        self.assertIn(
            r"\cmd{tools/export_trt.py} defaults to ONNX opset 18",
            manual,
        )
        self.assertIn("explicitly pin opset 17", manual)
        self.assertIn(
            "does not claim one universal TensorRT/CUDA version pair",
            manual,
        )
        self.assertIn(
            "exact environment or container, GPU/driver context, TensorRT version",
            manual,
        )


if __name__ == "__main__":
    unittest.main()
