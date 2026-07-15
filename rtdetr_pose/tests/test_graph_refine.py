import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from rtdetr_pose.config import ModelConfig, load_config
from rtdetr_pose.graph_refine import normalize_graph_refine_config


class TestGraphRefineConfig(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertEqual(normalize_graph_refine_config({}), {"enabled": False})

    def test_explicit_none_matches_omitted_default(self):
        self.assertEqual(normalize_graph_refine_config({"mode": "none"}), normalize_graph_refine_config({}))

    def test_rejects_unknown_version(self):
        with self.assertRaises(ValueError):
            normalize_graph_refine_config({"mode": "gcnv4"})

    def test_builtin_gcn_configs_load(self):
        cfg_v2 = load_config("builtin:gcnv2")
        cfg_v3 = load_config("builtin:gcnv3")
        self.assertEqual(cfg_v2.model.graph_refine["mode"], "gcnv2")
        self.assertEqual(cfg_v3.model.graph_refine["mode"], "gcnv3")


@unittest.skipIf(torch is None, "torch not installed")
class TestGraphRefineModel(unittest.TestCase):
    def test_gcnv2_refiner_preserves_query_shape(self):
        from rtdetr_pose.graph_refine import QueryGraphRefiner

        refiner = QueryGraphRefiner(hidden_dim=16, version="gcnv2", layers=2, topk=3)
        x = torch.randn(2, 5, 16)
        out = refiner(x)
        self.assertEqual(tuple(out.shape), (2, 5, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_gcnv3_refiner_preserves_query_shape(self):
        from rtdetr_pose.graph_refine import QueryGraphRefiner

        refiner = QueryGraphRefiner(hidden_dim=16, version="gcnv3", layers=1, topk=0)
        x = torch.randn(2, 5, 16)
        out = refiner(x)
        self.assertEqual(tuple(out.shape), (2, 5, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_rtdetr_pose_direct_gcnv2_forward(self):
        from rtdetr_pose.model import RTDETRPose

        model = RTDETRPose(
            num_classes=6,
            hidden_dim=32,
            num_queries=8,
            num_decoder_layers=1,
            nhead=4,
            stem_channels=16,
            backbone_channels=(32, 64, 128),
            stage_blocks=(1, 1, 1),
            graph_refine={"mode": "gcnv2", "layers": 1, "topk": 4},
        )
        x = torch.zeros(1, 3, 64, 64)
        out = model(x)
        self.assertEqual(out["logits"].shape, (1, 8, 6))
        self.assertEqual(out["bbox"].shape, (1, 8, 4))
        self.assertTrue(torch.isfinite(out["bbox"]).all())

    def test_factory_gcnv3_forward(self):
        from rtdetr_pose.factory import build_model

        cfg = ModelConfig(
            num_classes=5,
            hidden_dim=32,
            num_queries=8,
            num_decoder_layers=1,
            nhead=4,
            backbone_name="tiny_cnn",
            stem_channels=16,
            backbone_channels=[32, 64, 128],
            stage_blocks=[1, 1, 1],
            graph_refine={"mode": "gcnv3", "layers": 1, "topk": 4},
        )
        model = build_model(cfg)
        self.assertEqual(model.graph_refiner.version, "gcnv3")
        x = torch.zeros(1, 3, 64, 64)
        out = model(x)
        self.assertEqual(out["logits"].shape, (1, 8, 6))
        self.assertTrue(torch.isfinite(out["bbox"]).all())


if __name__ == "__main__":
    unittest.main()
