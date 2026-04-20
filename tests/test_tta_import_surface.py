import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestTTAImportSurface(unittest.TestCase):
    def test_tta_package_import_is_lightweight(self):
        sys.modules.pop("yolozu.tta", None)
        sys.modules.pop("yolozu.tta.tent", None)

        tta = importlib.import_module("yolozu.tta")

        self.assertTrue(hasattr(tta, "TTTConfig"))
        self.assertNotIn("yolozu.tta.tent", sys.modules)

    def test_lazy_tent_import_still_resolves_public_api(self):
        sys.modules.pop("yolozu.tta", None)
        sys.modules.pop("yolozu.tta.tent", None)

        tta = importlib.import_module("yolozu.tta")
        TentConfig = getattr(tta, "TentConfig")
        TentRunner = getattr(tta, "TentRunner")

        self.assertEqual(TentConfig.__name__, "TentConfig")
        self.assertEqual(TentRunner.__name__, "TentRunner")
        self.assertIn("yolozu.tta.tent", sys.modules)


if __name__ == "__main__":
    unittest.main()
