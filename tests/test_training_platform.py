import json
import tempfile
import unittest
from pathlib import Path

from yolozu.core.canonical import TrainConfig
from yolozu.training.platform import (
    build_training_data_flow,
    build_training_run_summary,
    get_training_backend_spec,
    training_capability_matrix,
)


class TestTrainingPlatform(unittest.TestCase):
    def test_backend_spec_exposes_reference_lane(self) -> None:
        spec = get_training_backend_spec("reference-rtdetr-pose")
        self.assertEqual(spec.maturity, "stable")
        self.assertTrue(spec.supports_run_contract)
        self.assertEqual(spec.interface_contract_level, "full_run_contract")
        self.assertEqual(spec.training_family, "rtdetr")
        self.assertIn("AdamW", spec.optimizer_policy or "")

    def test_capability_matrix_lists_external_lanes(self) -> None:
        ids = {row["backend_id"] for row in training_capability_matrix()}
        self.assertIn("yolox", ids)
        self.assertIn("detectron2", ids)
        self.assertIn("mmdetection", ids)
        self.assertIn("mmpose", ids)
        self.assertIn("mmseg", ids)
        self.assertIn("tao", ids)
        self.assertIn("ultralytics", ids)
        self.assertIn("hf-detr", ids)
        spec = get_training_backend_spec("mmpose")
        self.assertTrue(spec.supports_export)
        self.assertTrue(spec.supports_parity)
        self.assertIn("keypoints", spec.supported_tasks)
        yolox = get_training_backend_spec("yolox")
        self.assertEqual(yolox.training_family, "yolo")
        self.assertIn("letterbox", yolox.preprocess_policy or "")

    def test_training_run_summary_has_shared_format(self) -> None:
        cfg = TrainConfig(backend="yolox", model="exp.py", batch=4, epochs=1)
        payload = build_training_run_summary(
            backend_id="yolox",
            report_path="reports/train_external_yolox.json",
            train_config=cfg,
            dataset_root="data/smoke",
            split="val",
            dry_run=True,
            work_dir="runs/support_external_training/yolox",
            steps={"train": {"status": "dry_run", "ok": True, "executed": False}},
            license_boundary={"repo_code": "Apache-2.0"},
        )
        self.assertEqual(payload["format"], "yolozu_training_run_summary_v1")
        self.assertEqual(payload["backend"]["backend_id"], "yolox")
        self.assertEqual(payload["canonical_train_config"]["backend"], "yolox")
        self.assertEqual(payload["run_output_contract"]["kind"], "external_run_contract")
        self.assertIn("reports/training_summary.json", payload["run_output_contract"]["stable_artifacts"])
        self.assertIn("reports/resume_handoff.json", payload["run_output_contract"]["stable_artifacts"])
        self.assertIn("reports/export_handoff.json", payload["run_output_contract"]["stable_artifacts"])
        flow = payload["training_data_flow"]
        self.assertEqual(flow["format"], "yolozu_training_data_flow_v1")
        self.assertEqual(
            flow["stages"],
            ["raw_dataset", "DatasetAdapter", "YOLOZU Dataset Contract", "TrainingBackend"],
        )
        self.assertEqual(flow["dataset_contract"]["bbox_storage_preference"], "xyxy_abs")
        self.assertEqual(flow["training_backend"]["backend_id"], "yolox")
        self.assertEqual(flow["training_backend"]["bbox_view"], "cxcywh_norm")

    def test_training_data_flow_uses_contract_before_detr_backend(self) -> None:
        flow = build_training_data_flow(
            backend_id="reference-rtdetr-pose",
            dataset_root="data/coco128",
            split="train2017",
            raw_dataset_format="YOLO data.yaml",
        )
        self.assertEqual(flow["raw_dataset"]["format"], "YOLO data.yaml")
        self.assertEqual(flow["dataset_contract"]["version"], "1")
        self.assertIn("xywh_abs", flow["dataset_contract"]["adapter_views"])
        self.assertEqual(flow["training_backend"]["family"], "rtdetr")
        self.assertEqual(flow["training_backend"]["bbox_view"], "xyxy_abs")

    def test_training_run_summary_is_json_serializable(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "training_summary.json"
            payload = build_training_run_summary(
                backend_id="ultralytics",
                report_path=out,
                train_config={"format": "yolozu_train_config_v1", "backend": "ultralytics"},
                dry_run=True,
                steps={"train": {"status": "dry_run", "ok": True, "executed": False}},
            )
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["format"], "yolozu_training_run_summary_v1")


if __name__ == "__main__":
    unittest.main()
