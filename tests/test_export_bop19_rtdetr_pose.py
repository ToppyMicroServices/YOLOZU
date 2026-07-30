import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


class TestExportBOP19RTDETRPose(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.script = cls.repo_root / "tools" / "export_bop19_rtdetr_pose.py"
        spec = importlib.util.spec_from_file_location("export_bop19_rtdetr_pose", cls.script)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_help(self) -> None:
        result = subprocess.run(
            [sys.executable, str(self.script), "--help"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("official BOP19", result.stdout)
        self.assertIn("--targets", result.stdout)

    def test_target_groups_are_deterministic_and_bounded(self) -> None:
        targets = [
            {"scene_id": 2, "im_id": 1, "obj_id": 3, "inst_count": 1},
            {"scene_id": 1, "im_id": 2, "obj_id": 2, "inst_count": 1},
            {"scene_id": 1, "im_id": 2, "obj_id": 1, "inst_count": 1},
        ]
        grouped = self.module._target_groups(targets, max_images=1)
        self.assertEqual(grouped[0][0], (1, 2))
        self.assertEqual(len(grouped[0][1]), 2)

    def test_translation_uses_millimetres(self) -> None:
        translation = self.module._translation_mm(
            bbox=[0.5, 0.5, 0.2, 0.2],
            offsets=[0.0, 0.0],
            log_z=0.0,
            k_delta=[0.0, 0.0, 0.0, 0.0],
            intrinsics=(100.0, 100.0, 50.0, 50.0),
            image_size=100,
        )
        self.assertEqual(translation, [0.0, 0.0, 1000.0])

    def test_rot6d_identity(self) -> None:
        rotation = self.module._rot6d_to_matrix([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertEqual(rotation, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
