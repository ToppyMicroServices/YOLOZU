import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from yolozu import cli as yolozu_cli


def _write_report(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class TestDemoCLIDispatch(unittest.TestCase):
    def test_demo_keypoints_dispatch(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "keypoints_report.json"
            run_dir = Path(td) / "keypoints_run"

            def run_keypoints_demo(**kwargs):
                self.assertEqual(kwargs["device"], "cpu")
                self.assertEqual(int(kwargs["max_persons"]), 2)
                return _write_report(
                    out,
                    {
                        "settings": {"image": kwargs.get("image"), "run_dir": str(run_dir)},
                        "result": {"num_persons": 1, "artifacts": {"overlay": str(run_dir / "overlay.png")}},
                    },
                )

            mod = types.ModuleType("yolozu.demos.keypoints")
            mod.run_keypoints_demo = run_keypoints_demo
            with patch.dict(
                sys.modules,
                {
                    "torch": types.ModuleType("torch"),
                    "torchvision": types.ModuleType("torchvision"),
                    "yolozu.demos.keypoints": mod,
                },
                clear=False,
            ):
                rc = yolozu_cli.main(
                    [
                        "demo",
                        "keypoints",
                        "--image",
                        "data/smoke/images/val/000001.jpg",
                        "--run-dir",
                        str(run_dir),
                        "--device",
                        "cpu",
                        "--max-persons",
                        "2",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())

    def test_demo_pose_dispatch(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "pose_report.json"
            run_dir = Path(td) / "pose_run"

            def run_pose6d_demo(**kwargs):
                self.assertEqual(kwargs["backend"], "chessboard")
                self.assertEqual(float(kwargs["square_size"]), 0.04)
                return _write_report(
                    out,
                    {
                        "settings": {"backend": kwargs["backend"], "run_dir": str(run_dir)},
                        "result": {"t_xyz": [0.1, 0.2, 0.3], "artifacts": {"overlay": str(run_dir / "overlay.png")}},
                    },
                )

            mod = types.ModuleType("yolozu.demos.pose6d")
            mod.run_pose6d_demo = run_pose6d_demo
            with patch.dict(
                sys.modules,
                {
                    "cv2": types.ModuleType("cv2"),
                    "numpy": types.ModuleType("numpy"),
                    "yolozu.demos.pose6d": mod,
                },
                clear=False,
            ):
                rc = yolozu_cli.main(
                    [
                        "demo",
                        "pose",
                        "--backend",
                        "chessboard",
                        "--run-dir",
                        str(run_dir),
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())

    def test_demo_depth_dispatch(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "depth_report.json"
            run_dir = Path(td) / "depth_run"

            def run_depth_demo(**kwargs):
                self.assertEqual(kwargs["device"], "cpu")
                self.assertEqual(kwargs["model"], "depth_anything")
                return _write_report(
                    out,
                    {
                        "settings": {"model": kwargs["model"], "run_dir": str(run_dir)},
                        "result": {"depth": {"min": 0.1, "max": 1.5}},
                    },
                )

            mod = types.ModuleType("yolozu.demos.depth")
            mod.run_depth_demo = run_depth_demo
            with patch.dict(
                sys.modules,
                {
                    "torch": types.ModuleType("torch"),
                    "yolozu.demos.depth": mod,
                },
                clear=False,
            ):
                rc = yolozu_cli.main(
                    [
                        "demo",
                        "depth",
                        "--run-dir",
                        str(run_dir),
                        "--device",
                        "cpu",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())

    def test_demo_train_dispatch(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "train_report.json"

            def run_train_demo(**kwargs):
                self.assertEqual(int(kwargs["epochs"]), 1)
                self.assertEqual(int(kwargs["max_steps"]), 5)
                return _write_report(
                    out,
                    {
                        "settings": {"run_dir": str(Path(td) / "train_run")},
                        "result": {"train": {"steps": 5, "loss_mean": 0.5}, "val": {"acc": 0.8}},
                    },
                )

            mod = types.ModuleType("yolozu.demos.train")
            mod.run_train_demo = run_train_demo
            with patch.dict(
                sys.modules,
                {
                    "torch": types.ModuleType("torch"),
                    "torchvision": types.ModuleType("torchvision"),
                    "yolozu.demos.train": mod,
                },
                clear=False,
            ):
                rc = yolozu_cli.main(
                    [
                        "demo",
                        "train",
                        "--output",
                        str(out),
                        "--epochs",
                        "1",
                        "--max-steps",
                        "5",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
