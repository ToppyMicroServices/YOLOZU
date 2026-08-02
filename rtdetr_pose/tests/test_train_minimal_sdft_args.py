import importlib.util
import unittest
from pathlib import Path


def _load_train_minimal_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "rtdetr_pose" / "tools" / "train_minimal.py"
    spec = importlib.util.spec_from_file_location("rtdetr_pose_tools_train_minimal", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestTrainMinimalSDFTArgs(unittest.TestCase):
    def test_sdft_args_present(self):
        mod = _load_train_minimal_module()
        args = mod.parse_args([])
        self.assertTrue(hasattr(args, "self_distill_from"))
        self.assertIsNone(args.self_distill_from)
        self.assertEqual(args.self_distill_keys, "logits,bbox")
        self.assertEqual(args.self_distill_kl, "reverse")
        self.assertEqual(args.self_distill_weight, 1.0)
        self.assertEqual(args.self_distill_temperature, 1.0)
        self.assertFalse(args.self_distill_response_selection)
        self.assertEqual(args.self_distill_response_conf_min, 0.2)
        self.assertEqual(args.self_distill_response_topk, 20)
        self.assertEqual(args.self_distill_response_min_selected, 1)


if __name__ == "__main__":
    unittest.main()
