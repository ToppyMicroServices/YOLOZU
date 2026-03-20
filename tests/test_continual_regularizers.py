import tempfile
import unittest


class TestContinualRegularizers(unittest.TestCase):
    def test_ewc_roundtrip_and_penalty(self):
        torch = None
        try:
            import torch
        except Exception:
            self.skipTest("torch not available")

        from yolozu.continual_regularizers import EwcAccumulator, ewc_penalty, load_ewc_state, save_ewc_state

        model = torch.nn.Linear(2, 1, bias=False)
        x = torch.tensor([[1.0, 2.0]])
        y = torch.tensor([[0.0]])

        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()

        acc = EwcAccumulator()
        acc.accumulate_from_grads(model)
        state = acc.finalize(model)

        self.assertEqual(state.schema_version, 1)
        self.assertGreaterEqual(state.steps, 1)
        self.assertIn("weight", state.fisher)
        self.assertIn("weight", state.theta_star)
        self.assertEqual(tuple(state.fisher["weight"].shape), tuple(model.weight.shape))

        with tempfile.TemporaryDirectory() as td:
            p = f"{td}/ewc_state.pt"
            save_ewc_state(p, state)
            loaded = load_ewc_state(p)
            self.assertIn("weight", loaded.fisher)
            self.assertIn("weight", loaded.theta_star)

        # Penalty should be zero at theta_star, positive after a change.
        penalty0 = ewc_penalty(model, state)
        self.assertAlmostEqual(float(penalty0.detach().cpu()), 0.0, places=7)

        with torch.no_grad():
            model.weight.add_(1.0)
        penalty1 = ewc_penalty(model, state)
        self.assertGreater(float(penalty1.detach().cpu()), 0.0)

    def test_si_roundtrip_and_penalty(self):
        torch = None
        try:
            import torch
        except Exception:
            self.skipTest("torch not available")

        from yolozu.continual_regularizers import SiAccumulator, load_si_state, save_si_state, si_penalty

        model = torch.nn.Linear(2, 1, bias=False)
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.tensor([[1.0, 2.0]])
        y = torch.tensor([[0.0]])

        acc = SiAccumulator(epsilon=1e-3)
        acc.begin_task(model)

        for _ in range(3):
            optim.zero_grad(set_to_none=True)
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            acc.capture_before_step(model)
            optim.step()
            acc.update_after_step(model)

        state = acc.finalize(model)
        self.assertEqual(state.schema_version, 1)
        self.assertIn("weight", state.omega)
        self.assertIn("weight", state.theta_star)
        self.assertGreaterEqual(state.steps, 1)

        with tempfile.TemporaryDirectory() as td:
            p = f"{td}/si_state.pt"
            save_si_state(p, state)
            loaded = load_si_state(p)
            self.assertAlmostEqual(float(loaded.epsilon), 1e-3, places=10)
            self.assertIn("weight", loaded.omega)

        penalty0 = si_penalty(model, state)
        self.assertAlmostEqual(float(penalty0.detach().cpu()), 0.0, places=7)

        with torch.no_grad():
            model.weight.add_(1.0)
        penalty1 = si_penalty(model, state)
        self.assertGreater(float(penalty1.detach().cpu()), 0.0)


if __name__ == "__main__":
    unittest.main()
