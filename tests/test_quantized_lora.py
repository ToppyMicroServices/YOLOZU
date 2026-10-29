import copy
import io
import unittest

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@unittest.skipIf(torch is None, "torch not installed")
class QuantizedLoRATest(unittest.TestCase):
    def setUp(self):
        from yolozu.tta import quantized_lora
        self.ql = quantized_lora
        torch.manual_seed(17)

    def test_physical_pack_odd_shape_and_error_bound(self):
        weight = torch.randn(3, 5)
        packed, scales = self.ql.pack_int4(weight)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.numel(), 8)
        reconstructed = self.ql.unpack_int4(packed, scales, tuple(weight.shape))
        self.assertTrue(torch.all((weight - reconstructed).abs() <= scales[:, None] / 2 + 1e-6))
        zero_packed, zero_scales = self.ql.pack_int4(torch.zeros(2, 3))
        self.assertTrue(torch.equal(self.ql.unpack_int4(zero_packed, zero_scales, (2, 3)), torch.zeros(2, 3)))

    def test_conv_matches_dequantized_oracle_and_input_gradient(self):
        for kernel, padding, groups in [(1, 0, 1), (3, 1, 1), (3, 1, 2)]:
            with self.subTest(kernel=kernel, groups=groups):
                base = nn.Conv2d(4, 6, kernel_size=kernel, padding=padding, stride=2, groups=groups)
                layer = self.ql.QuantizedLoRAConv2d(base, quantize=True)
                x = torch.randn(2, 4, 12, 12, requires_grad=True)
                expected = F.conv2d(x, layer.dequantized_weight(x.dtype), base.bias, 2, padding, 1, groups)
                actual = layer(x)
                self.assertTrue(torch.equal(expected, actual))
                actual.square().sum().backward()
                self.assertGreater(float(x.grad.abs().sum()), 0)

    def test_reflect_padding_and_nonquantized_parity(self):
        base = nn.Conv2d(3, 5, 3, padding=2, dilation=2, padding_mode="reflect")
        wrapper = self.ql.QuantizedLoRAConv2d(base, quantize=False, rank=2)
        x = torch.randn(1, 3, 8, 9)
        self.assertTrue(torch.equal(base(x), wrapper(x)))

    def test_only_adapters_update_and_base_stays_identical(self):
        model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 1))
        report = self.ql.prepare_model(model, quantize=True, rank=2, adapter_include=("0", "2"))
        self.assertEqual(report["quantized_layer_count"], 2)
        self.assertEqual(report["adapter_layer_count"], 2)
        self.assertTrue(all(name.endswith(("lora_A", "lora_B")) for name, p in model.named_parameters() if p.requires_grad))
        fingerprint = self.ql.base_fingerprint(model)
        adapters_before = self.ql.adapter_state_dict(model)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        x = torch.randn(2, 3, 8, 8, requires_grad=True)
        for _ in range(2):
            optimizer.zero_grad()
            model(x).square().mean().backward()
            optimizer.step()
        self.assertEqual(fingerprint, self.ql.base_fingerprint(model))
        self.assertGreater(float(x.grad.abs().sum()), 0)
        for name, p in model.named_parameters():
            self.assertGreater(float(p.grad.abs().sum()), 0, name)
        self.assertTrue(any(not torch.equal(adapters_before["adapter_tensors"][name], p) for name, p in model.named_parameters()))

    def test_actual_storage_reduction_and_no_retained_float_weight(self):
        model = nn.Sequential(nn.Linear(128, 128))
        report = self.ql.prepare_model(model, quantize=True, rank=2, adapter_include=("0",))
        self.assertLess(report["base_storage_bytes_after"], report["base_storage_bytes_before"] / 4)
        self.assertEqual(report["layers"][0]["packed_weight_bytes"], 128 * 128 // 2)
        self.assertFalse(hasattr(model[0], "base_weight"))
        self.assertFalse(any("weight" == name for name, _ in model[0].named_parameters()))
        all_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
        self.assertEqual(all_bytes, report["base_storage_bytes_after"] + report["adapter_storage_bytes"])

    def test_adapter_checkpoint_roundtrip_and_wrong_base_rejected(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8))
        self.ql.prepare_model(model, quantize=True, rank=2, adapter_include=("0",))
        model.eval()
        with torch.no_grad():
            model[0].lora_B.normal_()
        x = torch.randn(2, 8)
        prediction = model(x).detach().clone()
        envelope = self.ql.adapter_state_dict(model)
        stream = io.BytesIO()
        torch.save(envelope, stream)
        stream.seek(0)
        loaded = torch.load(stream, weights_only=True)
        with torch.no_grad():
            model[0].lora_B.zero_()
        self.ql.load_adapter_state_dict(model, loaded)
        self.assertTrue(torch.equal(prediction, model(x)))
        with torch.no_grad():
            model[1].running_mean.add_(1)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.ql.load_adapter_state_dict(model, loaded)

    def test_bad_envelope_does_not_partially_change_adapters(self):
        model = nn.Sequential(nn.Linear(4, 4))
        self.ql.prepare_model(model, quantize=True, rank=2, adapter_include=("0",))
        payload = self.ql.adapter_state_dict(model)
        bad = copy.deepcopy(payload)
        bad["adapter_tensors"]["0.lora_A"].fill_(1)
        bad["adapter_tensors"]["0.lora_B"].fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.ql.load_adapter_state_dict(model, bad)
        self.assertTrue(torch.equal(model[0].lora_A, payload["adapter_tensors"]["0.lora_A"]))
        self.assertTrue(torch.equal(model[0].lora_B, payload["adapter_tensors"]["0.lora_B"]))

    def test_adapter_rejects_same_weights_with_different_convolution_geometry(self):
        model = nn.Sequential(nn.Conv2d(3, 5, 3, padding=1))
        other = copy.deepcopy(model)
        other[0].stride = (2, 2)
        for candidate in (model, other):
            self.ql.prepare_model(candidate, quantize=True, rank=2, adapter_include=("0",))
        payload = self.ql.adapter_state_dict(model)
        with self.assertRaisesRegex(ValueError, "recipe"):
            self.ql.load_adapter_state_dict(other, payload)

    def test_missing_scope_and_grouped_adapter_reject_before_mutation(self):
        model = nn.Sequential(nn.Conv2d(4, 4, 1), nn.Conv2d(4, 4, 3, groups=4))
        for kwargs in [
            {"quantize": True, "rank": 1, "adapter_include": ("*",)},
            {"quantize": True, "rank": 1, "adapter_include": ("absent",)},
            {"quantize": True, "rank": 0, "adapter_include": (), "quant_include": ("absent",)},
        ]:
            with self.assertRaises(ValueError):
                self.ql.prepare_model(model, **kwargs)
            self.assertIsInstance(model[0], nn.Conv2d)
            self.assertTrue(model[0].weight.requires_grad)

    def test_quantization_and_adapter_scopes_are_independent(self):
        model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
        report = self.ql.prepare_model(model, quantize=True, rank=2, adapter_include=("1",), quant_include=("0",))
        self.assertTrue(model[0].quantized)
        self.assertFalse(model[1].quantized)
        self.assertEqual(report["quantized_layer_count"], 1)
        self.assertEqual(report["adapter_layer_count"], 1)
        with self.assertRaisesRegex(ValueError, "already"):
            self.ql.prepare_model(model, quantize=True, rank=2, adapter_include=("1",))

    def test_nonfinite_and_invalid_packed_shapes_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.ql.pack_int4(torch.tensor([[float("inf")]]))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.ql.pack_int4(torch.empty(0, 2))
        with self.assertRaisesRegex(ValueError, "shape"):
            self.ql.unpack_int4(torch.zeros(1, dtype=torch.uint8), torch.ones(2), (2, 2))
        model = nn.Sequential(nn.Linear(2, 2))
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            self.ql.prepare_model(model, quantize=True, rank=-1, adapter_include=("0",))


if __name__ == "__main__":
    unittest.main()
