from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rtdetr_pose.config import load_config
from rtdetr_pose.torchao_integration import apply_torchao_quantization
from rtdetr_pose.train_rebalance import _record_label_ids
from rtdetr_pose.train_runtime import build_validation_loader, install_termination_handlers
from rtdetr_pose.validator import _load_array_from_path


class TestRuntimeParseGuards(unittest.TestCase):
    def test_install_termination_handlers_tolerates_signal_registration_error(self) -> None:
        with mock.patch("rtdetr_pose.train_runtime.signal.signal", side_effect=ValueError("unsupported")):
            flag = install_termination_handlers(is_main=False)
        self.assertEqual(flag, {"terminate": False})

    def test_build_validation_loader_invalid_val_batch_size_falls_back(self) -> None:
        args = argparse.Namespace(
            batch_size=4,
            val_batch_size="invalid",
            num_queries=10,
            num_classes=2,
            num_keypoints=0,
            image_size=64,
            seed=7,
            depth_mode="none",
            depth_unit="m",
            depth_scale=1.0,
            real_images=False,
        )

        class DummyDataset:
            def __init__(self, *dataset_args, **dataset_kwargs):
                self.dataset_args = dataset_args
                self.dataset_kwargs = dataset_kwargs

        def _make_loader(dataset, **kwargs):
            return {"dataset": dataset, "kwargs": kwargs}

        loader = build_validation_loader(
            args=args,
            val_records=[{"image_path": "x.jpg", "labels": []}],
            keypoint_flip_pairs=[],
            loader_kwargs={"num_workers": 0},
            manifest_dataset_cls=DummyDataset,
            dataloader_cls=_make_loader,
        )
        assert loader is not None
        self.assertEqual(loader["kwargs"]["batch_size"], 4)
        self.assertFalse(loader["kwargs"]["shuffle"])

    def test_load_config_builtin_missing_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_config("builtin:definitely_missing_config_for_tests")

    def test_record_label_ids_skips_invalid_numeric_values(self) -> None:
        record = {
            "labels": [
                {"class_id": 1},
                {"class_id": "2"},
                {"class_id": "nan"},
                {"class_id": None},
                {"class_id": -1},
            ]
        }
        self.assertEqual(_record_label_ids(record), [1, 2])

    def test_load_array_from_invalid_png_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.png"
            path.write_bytes(b"not-a-real-png")
            self.assertIsNone(_load_array_from_path(path))

    def test_apply_torchao_quantization_bad_config_factory_returns_report(self) -> None:
        fake_module = SimpleNamespace(
            int8_weight_only=lambda: (_ for _ in ()).throw(ValueError("bad config")),
            quantize_=lambda model, config: model,
        )
        with mock.patch("importlib.import_module", return_value=fake_module):
            obj = object()
            out, report = apply_torchao_quantization(obj, recipe="int8wo", required=False)
        self.assertIs(out, obj)
        self.assertEqual(report.reason, "config_not_found")
        self.assertFalse(report.applied)

    def test_load_config_from_path_still_reads_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "dataset": {"root": "data/sample"},
                        "model": {"num_classes": 3},
                    }
                ),
                encoding="utf-8",
            )
            cfg = load_config(path)
        self.assertEqual(cfg.model.num_classes, 3)


if __name__ == "__main__":
    unittest.main()
