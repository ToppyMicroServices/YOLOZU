import importlib.util
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
class TestTrainMinimalMaskDepthCollate(unittest.TestCase):
    def test_mask_depth_loaded_and_collated(self):
        mod = _load_train_minimal_module()

        record = {
            "image_path": "",
            "labels": [
                {"class_id": 0, "bbox": {"cx": 0.2, "cy": 0.3, "w": 0.1, "h": 0.2}},
                {"class_id": 1, "bbox": {"cx": 0.6, "cy": 0.4, "w": 0.2, "h": 0.1}},
            ],
            "M": [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ],
            "D_obj": [
                [[0.5, 0.2], [0.1, 0.9]],
                [[0.6, 0.4], [0.2, 0.8]],
            ],
        }

        ds = mod.ManifestDataset(
            [record],
            num_queries=5,
            num_classes=80,
            image_size=2,
            seed=0,
            use_matcher=True,
            synthetic_pose=False,
            z_from_dobj=False,
            load_aux=True,
            real_images=False,
            multiscale=False,
            scale_min=1.0,
            scale_max=1.0,
            hflip_prob=0.0,
            intrinsics_jitter=False,
            jitter_dfx=0.0,
            jitter_dfy=0.0,
            jitter_dcx=0.0,
            jitter_dcy=0.0,
            sim_jitter=False,
            sim_jitter_profile=None,
            sim_jitter_extrinsics=False,
            extrinsics_jitter=False,
            jitter_dx=0.01,
            jitter_dy=0.01,
            jitter_dz=0.02,
            jitter_droll=1.0,
            jitter_dpitch=1.0,
            jitter_dyaw=2.0,
        )

        sample = ds[0]
        targets = sample["targets"]
        self.assertIn("gt_M", targets)
        self.assertIn("gt_D_obj", targets)
        self.assertEqual(tuple(targets["gt_M"].shape), (2, 2, 2))
        self.assertEqual(tuple(targets["gt_D_obj"].shape), (2, 2, 2))

        images, collated = mod.collate([sample, sample])
        padded = collated["padded"]
        self.assertEqual(tuple(images.shape), (2, 3, 2, 2))
        self.assertEqual(tuple(padded["gt_M"].shape), (2, 2, 2, 2))
        self.assertEqual(tuple(padded["gt_D_obj"].shape), (2, 2, 2, 2))

    def test_variable_aux_shapes_resize_before_batch_collate(self):
        mod = _load_train_minimal_module()
        records = [
            {
                "image_path": "",
                "labels": [{"class_id": 0, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.5}}],
                "M": [[[1.0, 0.0], [0.0, 1.0]]],
                "D_obj": [[[0.1, 0.2], [0.3, 0.4]]],
            },
            {
                "image_path": "",
                "labels": [{"class_id": 0, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.5}}],
                "M": [[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                "D_obj": [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]],
            },
        ]
        ds = mod.ManifestDataset(
            records,
            num_queries=5,
            num_classes=1,
            image_size=4,
            seed=0,
            use_matcher=True,
            load_aux=True,
        )

        images, collated = mod.collate([ds[0], ds[1]])
        padded = collated["padded"]
        self.assertEqual(tuple(images.shape), (2, 3, 4, 4))
        self.assertEqual(tuple(padded["gt_M"].shape), (2, 1, 4, 4))
        self.assertEqual(tuple(padded["gt_D_obj"].shape), (2, 1, 4, 4))

    def test_png_mask_sidecar_is_loaded_and_resized(self):
        from PIL import Image

        mod = _load_train_minimal_module()
        with tempfile.TemporaryDirectory() as td:
            mask_path = Path(td) / "mask.png"
            image = Image.new("L", (5, 3), color=0)
            image.putpixel((2, 1), 1)
            image.save(mask_path)
            record = {
                "image_path": "",
                "labels": [{"class_id": 0, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.5}}],
                "M": str(mask_path),
                "D_obj": [[[0.5] * 5 for _ in range(3)]],
            }
            ds = mod.ManifestDataset(
                [record],
                num_queries=5,
                num_classes=1,
                image_size=4,
                seed=0,
                use_matcher=True,
                load_aux=True,
            )

            targets = ds[0]["targets"]
            self.assertEqual(tuple(targets["gt_M"].shape), (1, 4, 4))
            self.assertEqual(tuple(targets["gt_D_obj"].shape), (1, 4, 4))
            self.assertGreater(float(targets["gt_M"].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
