import hashlib
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

from yolozu.inference.checkpoint_compatibility import (
    CHECKPOINT_REPORT_FORMAT,
    CheckpointCompatibilityError,
    load_checkpoint_compatible,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class _Tensor:
    def __init__(self, *shape: int):
        self.shape = tuple(shape)
        self.dtype = "float32"

    def numel(self) -> int:
        total = 1
        for size in self.shape:
            total *= size
        return total


class _Model:
    def __init__(self):
        self._state = OrderedDict(
            [
                ("backbone.weight", _Tensor(2, 3)),
                ("head.weight", _Tensor(4, 2)),
                ("head.bias", _Tensor(4)),
                ("running_count", _Tensor()),
            ]
        )
        self.load_calls = []

    def state_dict(self):
        return self._state

    def named_parameters(self):
        return [
            ("backbone.weight", self._state["backbone.weight"]),
            ("head.weight", self._state["head.weight"]),
            ("head.bias", self._state["head.bias"]),
        ]

    def load_state_dict(self, state, *, strict):
        self.load_calls.append((dict(state), bool(strict)))


class _UnsafePayload:
    def __init__(self, sentinel: str):
        self.sentinel = sentinel

    def __reduce__(self):
        return _write_sentinel, (self.sentinel,)


def _write_sentinel(path: str) -> None:
    Path(path).write_text("executed", encoding="utf-8")


class TestCheckpointCompatibility(unittest.TestCase):
    def _run(self, checkpoint, *, allow_partial=False):
        model = _Model()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint_path = root / "checkpoint.pt"
            checkpoint_path.write_bytes(b"checkpoint fixture")
            config_path = root / "model.json"
            config_path.write_text('{"model":"tiny"}', encoding="utf-8")
            report = load_checkpoint_compatible(
                model,
                checkpoint_path,
                config_identity=config_path,
                allow_partial=allow_partial,
                checkpoint_loader=lambda _path: checkpoint,
            )
        return model, report

    @staticmethod
    def _exact_state():
        return OrderedDict(
            [
                ("backbone.weight", _Tensor(2, 3)),
                ("head.weight", _Tensor(4, 2)),
                ("head.bias", _Tensor(4)),
                ("running_count", _Tensor()),
            ]
        )

    def test_raw_state_dict_exact_match_loads_strictly(self):
        model, report = self._run(self._exact_state())

        self.assertEqual(report["format"], CHECKPOINT_REPORT_FORMAT)
        self.assertEqual(report["status"], "full")
        self.assertEqual(report["checkpoint"]["container"], "raw_state_dict")
        self.assertEqual(
            report["compatibility"]["tensor_count_coverage"]["model_ratio"],
            1.0,
        )
        self.assertEqual(
            report["compatibility"]["parameter_numel_coverage"]["model_ratio"],
            1.0,
        )
        self.assertTrue(report["checkpoint"]["sha256"])
        self.assertEqual(
            report["checkpoint"]["deserialization"]["policy"],
            "custom_loader",
        )
        self.assertTrue(report["model"]["config"]["sha256"])
        self.assertEqual(len(model.load_calls), 1)
        self.assertTrue(model.load_calls[0][1])

    def test_state_dict_wrapper_exact_match_loads_strictly(self):
        model, report = self._run({"state_dict": self._exact_state(), "epoch": 3})

        self.assertEqual(report["status"], "full")
        self.assertEqual(report["checkpoint"]["container"], "state_dict_wrapper")
        self.assertEqual(len(model.load_calls), 1)
        self.assertTrue(model.load_calls[0][1])

    def test_uniform_module_prefix_is_normalized(self):
        prefixed = OrderedDict(
            (f"module.{key}", value) for key, value in self._exact_state().items()
        )
        model, report = self._run(prefixed)

        self.assertEqual(report["status"], "full")
        applied = report["legacy_key_normalization"]["applied"]
        self.assertEqual([item["prefix"] for item in applied], ["module."])
        self.assertEqual(set(model.load_calls[0][0]), set(self._exact_state()))

    def test_name_mismatch_fails_before_model_mutation(self):
        state = self._exact_state()
        state["renamed.bias"] = state.pop("head.bias")
        model = _Model()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "checkpoint.pt"
            path.write_bytes(b"name mismatch")
            with self.assertRaises(CheckpointCompatibilityError) as ctx:
                load_checkpoint_compatible(
                    model,
                    path,
                    checkpoint_loader=lambda _path: state,
                )

        report = ctx.exception.report
        self.assertEqual(report["status"], "incompatible")
        self.assertIn("head.bias", report["compatibility"]["missing_keys"])
        self.assertIn("renamed.bias", report["compatibility"]["unexpected_keys"])
        self.assertEqual(model.load_calls, [])

    def test_shape_mismatch_fails_before_model_mutation(self):
        state = self._exact_state()
        state["head.weight"] = _Tensor(5, 2)
        model = _Model()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "checkpoint.pt"
            path.write_bytes(b"shape mismatch")
            with self.assertRaises(CheckpointCompatibilityError) as ctx:
                load_checkpoint_compatible(
                    model,
                    path,
                    checkpoint_loader=lambda _path: state,
                )

        report = ctx.exception.report
        mismatches = report["compatibility"]["shape_mismatches"]
        self.assertEqual([item["key"] for item in mismatches], ["head.weight"])
        self.assertEqual(mismatches[0]["checkpoint_shape"], [5, 2])
        self.assertEqual(mismatches[0]["model_shape"], [4, 2])
        self.assertEqual(model.load_calls, [])

    def test_explicit_partial_load_records_partial_status(self):
        state = OrderedDict(
            [
                ("backbone.weight", _Tensor(2, 3)),
                ("head.weight", _Tensor(99, 2)),
                ("extra.weight", _Tensor(1)),
            ]
        )
        model, report = self._run(state, allow_partial=True)

        self.assertEqual(report["status"], "partial")
        self.assertTrue(report["allow_partial"])
        self.assertEqual(report["compatibility"]["matched_keys"], ["backbone.weight"])
        self.assertEqual(set(model.load_calls[0][0]), {"backbone.weight"})
        self.assertFalse(model.load_calls[0][1])
        self.assertEqual(report["load"]["mode"], "name_and_shape_partial")

    def test_partial_opt_in_still_rejects_zero_matches(self):
        model = _Model()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "checkpoint.pt"
            path.write_bytes(b"zero match")
            with self.assertRaises(CheckpointCompatibilityError) as ctx:
                load_checkpoint_compatible(
                    model,
                    path,
                    allow_partial=True,
                    checkpoint_loader=lambda _path: {
                        "other.weight": _Tensor(1),
                    },
                )

        self.assertEqual(ctx.exception.report["status"], "incompatible")
        self.assertEqual(model.load_calls, [])

    @unittest.skipIf(torch is None, "torch not installed")
    def test_default_loader_rejects_unsafe_pickle_without_execution(self):
        model = _Model()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint_path = root / "unsafe.pt"
            sentinel = root / "unsafe-code-executed"
            torch.save(
                {
                    "state_dict": OrderedDict(
                        [
                            ("backbone.weight", torch.zeros(2, 3)),
                            ("head.weight", torch.zeros(4, 2)),
                            ("head.bias", torch.zeros(4)),
                            ("running_count", torch.zeros(())),
                        ]
                    ),
                    "unsafe": _UnsafePayload(str(sentinel)),
                },
                checkpoint_path,
            )

            with self.assertRaises(CheckpointCompatibilityError) as ctx:
                load_checkpoint_compatible(model, checkpoint_path)

            self.assertFalse(sentinel.exists())
            self.assertEqual(ctx.exception.report["status"], "incompatible")
            self.assertEqual(
                ctx.exception.report["checkpoint"]["deserialization"]["policy"],
                "torch_weights_only",
            )
            self.assertTrue(
                ctx.exception.report["checkpoint"]["deserialization"]["weights_only"]
            )
            self.assertEqual(model.load_calls, [])


class TestHistoricalCheckpointProvenance(unittest.TestCase):
    def test_historical_artifact_hashes_and_limitations_are_pinned(self):
        repo_root = Path(__file__).resolve().parents[1]
        report_path = (
            repo_root
            / "reports"
            / "rtdetr_pose_coco128_gpu_matcher_historical.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "historical")
        self.assertFalse(report["usable_as_current_full_checkpoint_evidence"])
        self.assertFalse(
            report["original_recipe"]["config_present_in_source_commit"]
        )
        self.assertFalse(
            (repo_root / report["original_recipe"]["claimed_config_path"]).exists()
        )
        for artifact in report["artifacts"]:
            path = repo_root / artifact["path"]
            self.assertTrue(path.is_file(), str(path))
            self.assertEqual(path.stat().st_size, artifact["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                artifact["sha256"],
            )

        audit = report["current_compatibility_audit"]
        self.assertEqual(audit["strict_status"], "incompatible")
        self.assertEqual(audit["matched_tensor_count"], 20)
        self.assertEqual(audit["model_state_tensor_count"], 308)
        self.assertEqual(audit["effective_unloaded_model_keys"], 288)
        self.assertEqual(audit["skipped_checkpoint_keys"], 263)
        self.assertLess(audit["model_parameter_numel_coverage"], 0.04)


if __name__ == "__main__":
    unittest.main()
