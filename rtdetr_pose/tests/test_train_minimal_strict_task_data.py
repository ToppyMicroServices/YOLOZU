import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _load_train_minimal_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "rtdetr_pose" / "tools" / "train_minimal.py"
    spec = importlib.util.spec_from_file_location("rtdetr_pose_tools_train_minimal", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipIf(torch is None, "torch not installed")
class TestTrainMinimalStrictTaskData(unittest.TestCase):
    def test_strict_task_data_rejects_missing_image_path(self):
        mod = _load_train_minimal_module()
        with tempfile.TemporaryDirectory() as td:
            records_path = Path(td) / "records.json"
            records = [
                {
                    "image_id": 1,
                    "labels": [{"class_id": 0, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.2}}],
                }
            ]
            records_path.write_text(json.dumps(records), encoding="utf-8")

            with self.assertRaises(SystemExit) as ctx:
                mod.main(
                    [
                        "--records-json",
                        str(records_path),
                        "--strict-task-data",
                        "--real-images",
                        "--epochs",
                        "1",
                        "--max-steps",
                        "1",
                        "--batch-size",
                        "1",
                        "--device",
                        "cpu",
                        "--no-export-onnx",
                    ]
                )
            self.assertIn("strict-task-data checks failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
