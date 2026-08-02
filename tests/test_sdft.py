import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from yolozu.sdft import SdftConfig, compute_sdft_loss, kl_divergence_from_logits


class TestSDFT(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_kl_zero_when_equal(self):
        logits = torch.randn(2, 3, 5)
        loss_f = kl_divergence_from_logits(logits, logits, mode="forward")
        loss_r = kl_divergence_from_logits(logits, logits, mode="reverse")
        loss_s = kl_divergence_from_logits(logits, logits, mode="sym")
        self.assertLess(float(loss_f.detach().cpu()), 1e-7)
        self.assertLess(float(loss_r.detach().cpu()), 1e-7)
        self.assertLess(float(loss_s.detach().cpu()), 1e-7)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_compute_sdft_loss_parts(self):
        student = {
            "logits": torch.randn(2, 3, 5),
            "bbox": torch.randn(2, 3, 4),
        }
        teacher = {
            "logits": torch.randn(2, 3, 5),
            "bbox": torch.randn(2, 3, 4),
        }
        cfg = SdftConfig(
            weight=1.0,
            temperature=2.0,
            kl="reverse",
            keys=("logits", "bbox"),
            logits_weight=0.7,
            bbox_weight=0.3,
        )
        total, parts = compute_sdft_loss(student, teacher, cfg)
        self.assertIn("loss_sdft", parts)
        self.assertIn("loss_sdft_logits", parts)
        self.assertIn("loss_sdft_bbox", parts)
        self.assertTrue(torch.is_tensor(total))
        self.assertTrue(torch.allclose(total, parts["loss_sdft"]))

    @unittest.skipIf(torch is None, "torch not installed")
    def test_missing_keys_zero_on_reference_device(self):
        logits = torch.randn(1, 2, 3)
        student = {"logits": logits}
        teacher = {"logits": logits.clone()}
        cfg = SdftConfig(keys=("missing",))
        total, parts = compute_sdft_loss(student, teacher, cfg)
        self.assertEqual(total.device, logits.device)
        self.assertEqual(total.dtype, logits.dtype)
        self.assertIn("loss_sdft", parts)
        self.assertEqual(float(total.detach().cpu()), 0.0)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_weight_scales_total_and_parts(self):
        student = {
            "logits": torch.randn(2, 3, 5, requires_grad=True),
            "bbox": torch.randn(2, 3, 4, requires_grad=True),
        }
        teacher = {
            "logits": torch.randn(2, 3, 5),
            "bbox": torch.randn(2, 3, 4),
        }
        base_cfg = SdftConfig(weight=1.0, temperature=1.5, kl="reverse", keys=("logits", "bbox"))
        scaled_cfg = SdftConfig(weight=2.0, temperature=1.5, kl="reverse", keys=("logits", "bbox"))

        total_base, parts_base = compute_sdft_loss(student, teacher, base_cfg)
        total_scaled, parts_scaled = compute_sdft_loss(student, teacher, scaled_cfg)

        self.assertTrue(torch.allclose(total_scaled, total_base * 2.0))
        self.assertTrue(torch.allclose(parts_scaled["loss_sdft"], parts_base["loss_sdft"] * 2.0))
        self.assertTrue(torch.allclose(parts_scaled["loss_sdft_logits"], parts_base["loss_sdft_logits"] * 2.0))
        self.assertTrue(torch.allclose(parts_scaled["loss_sdft_bbox"], parts_base["loss_sdft_bbox"] * 2.0))

    @unittest.skipIf(torch is None, "torch not installed")
    def test_teacher_is_detached(self):
        student = {"logits": torch.randn(2, 3, 5, requires_grad=True)}
        teacher = {"logits": torch.randn(2, 3, 5, requires_grad=True)}
        cfg = SdftConfig(keys=("logits",))
        total, _ = compute_sdft_loss(student, teacher, cfg)
        total.backward()
        self.assertIsNone(teacher["logits"].grad)
        self.assertIsNotNone(student["logits"].grad)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_shape_mismatch_raises(self):
        student = {"logits": torch.randn(2, 3, 5)}
        teacher = {"logits": torch.randn(2, 3, 6)}
        cfg = SdftConfig(keys=("logits",))
        with self.assertRaises(ValueError):
            compute_sdft_loss(student, teacher, cfg)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_invalid_config_raises(self):
        student = {"logits": torch.randn(1, 2, 3)}
        teacher = {"logits": torch.randn(1, 2, 3)}
        with self.assertRaises(ValueError):
            compute_sdft_loss(student, teacher, SdftConfig(weight=-1.0))
        with self.assertRaises(ValueError):
            compute_sdft_loss(student, teacher, SdftConfig(temperature=0.0))

    @unittest.skipIf(torch is None, "torch not installed")
    def test_response_selection_limits_gradients_and_excludes_no_object(self):
        student_logits = torch.tensor(
            [[[2.0, 1.0, -1.0], [1.0, 0.0, 2.0]]], requires_grad=True
        )
        student_bbox = torch.ones(1, 2, 4, requires_grad=True)
        teacher = {
            "logits": torch.tensor([[[4.0, 0.0, -1.0], [0.0, 0.0, 3.0]]]),
            "bbox": torch.zeros(1, 2, 4),
        }
        total, parts = compute_sdft_loss(
            {"logits": student_logits, "bbox": student_bbox},
            teacher,
            SdftConfig(
                keys=("logits", "bbox"),
                response_selection=True,
                response_conf_min=0.5,
                response_topk=5,
            ),
        )
        total.backward()
        self.assertEqual(float(parts["sdft_selected_queries"]), 1.0)
        self.assertEqual(float(parts["sdft_used_queries"]), 1.0)
        self.assertEqual(float(parts["sdft_abstained"]), 0.0)
        self.assertTrue(torch.count_nonzero(student_logits.grad[0, 0, :2]))
        self.assertEqual(float(student_logits.grad[0, 0, -1]), 0.0)
        self.assertEqual(int(torch.count_nonzero(student_logits.grad[0, 1])), 0)
        self.assertTrue(torch.count_nonzero(student_bbox.grad[0, 0]))
        self.assertEqual(int(torch.count_nonzero(student_bbox.grad[0, 1])), 0)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_response_selection_zero_match_is_connected_zero(self):
        logits = torch.zeros(1, 2, 3, requires_grad=True)
        total, parts = compute_sdft_loss(
            {"logits": logits},
            {"logits": torch.tensor([[[0.0, 0.0, 4.0], [0.0, 0.0, 4.0]]])},
            SdftConfig(
                keys=("logits",),
                response_selection=True,
                response_conf_min=0.9,
            ),
        )
        total.backward()
        self.assertEqual(float(total.detach()), 0.0)
        self.assertEqual(float(parts["sdft_selected_queries"]), 0.0)
        self.assertEqual(float(parts["sdft_used_queries"]), 0.0)
        self.assertEqual(float(parts["sdft_abstained"]), 1.0)
        self.assertEqual(int(torch.count_nonzero(logits.grad)), 0)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_response_selection_abstains_below_minimum(self):
        logits = torch.tensor(
            [[[2.0, 1.0, -1.0], [1.0, 0.0, 2.0]]], requires_grad=True
        )
        total, parts = compute_sdft_loss(
            {"logits": logits},
            {"logits": torch.tensor([[[4.0, 0.0, -1.0], [0.0, 0.0, 3.0]]])},
            SdftConfig(
                keys=("logits",),
                response_selection=True,
                response_conf_min=0.5,
                response_min_selected=2,
            ),
        )
        total.backward()
        self.assertEqual(float(total.detach()), 0.0)
        self.assertEqual(float(parts["sdft_selected_queries"]), 1.0)
        self.assertEqual(float(parts["sdft_used_queries"]), 0.0)
        self.assertEqual(float(parts["sdft_abstained"]), 1.0)
        self.assertEqual(int(torch.count_nonzero(logits.grad)), 0)


if __name__ == "__main__":
    unittest.main()
