import importlib.util
import math
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "summarize_bop19_pose_evidence.py"
SPEC = importlib.util.spec_from_file_location("summarize_bop19_pose_evidence", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TestSummarizeBOP19PoseEvidence(unittest.TestCase):
    def test_rotation_and_translation_errors(self):
        identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.assertEqual(MODULE._rotation_error_deg(identity, identity), 0.0)
        self.assertEqual(MODULE._translation_error_mm([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.0)
        self.assertTrue(
            math.isclose(
                MODULE._translation_error_mm([4.0, 6.0, 3.0], [1.0, 2.0, 3.0]),
                5.0,
            )
        )

    def test_seed_parser(self):
        self.assertEqual(MODULE._seed_from_name("yolozu-rtdetrpose-s11_tless-test"), 11)
        self.assertEqual(MODULE._seed_from_name("method-seed22-test"), 22)
        with self.assertRaises(ValueError):
            MODULE._seed_from_name("missing")

    def test_mean_preserves_no_measurement_as_null(self):
        self.assertIsNone(MODULE._mean([]))
        self.assertIsNone(MODULE._mean([math.inf]))
        self.assertEqual(MODULE._mean([1.0, 3.0, math.inf]), 2.0)


if __name__ == "__main__":
    unittest.main()
