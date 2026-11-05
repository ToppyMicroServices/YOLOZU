import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]


class TestPackagingMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        with (cls.repo_root / "pyproject.toml").open("rb") as handle:
            cls.pyproject = tomllib.load(handle)

    def test_license_uses_pep_639_metadata(self) -> None:
        project = self.pyproject["project"]
        self.assertEqual(project["license"], "Apache-2.0")
        self.assertEqual(project["license-files"], ["LICENSE"])
        self.assertFalse(
            any(item.startswith("License ::") for item in project["classifiers"])
        )

    def test_build_backend_supports_pep_639_metadata(self) -> None:
        self.assertIn("setuptools>=77", self.pyproject["build-system"]["requires"])

    def test_security_dependency_floor_requires_supported_python(self) -> None:
        project = self.pyproject["project"]
        self.assertEqual(project["requires-python"], ">=3.11")
        self.assertIn("Pillow>=13.0.0", project["dependencies"])
        self.assertNotIn("Programming Language :: Python :: 3.10", project["classifiers"])


if __name__ == "__main__":
    unittest.main()
