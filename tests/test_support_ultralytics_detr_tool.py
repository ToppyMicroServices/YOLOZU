import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestSupportUltralyticsDetrTool(unittest.TestCase):
    def test_help(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "support_ultralytics_detr.py"
        self.assertTrue(script.is_file(), "missing tools/support_ultralytics_detr.py")

        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"support_ultralytics_detr --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("train-ultralytics", proc.stdout)
        self.assertIn("train-hf-detr", proc.stdout)
        self.assertIn("export-onnx", proc.stdout)
        self.assertIn("predict-normalize", proc.stdout)

    def test_layers_json(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "support_ultralytics_detr.py"
        proc = subprocess.run(
            [sys.executable, str(script), "layers", "--json"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"layers --json failed:\n{proc.stdout}\n{proc.stderr}")
        payload = json.loads(proc.stdout)
        layers = payload.get("layers") or {}
        self.assertIn("trainer_runner", layers)
        self.assertIn("repo_impl", layers)
        self.assertIn("export_deploy", layers)

    def test_dataset_internal_and_dryrun_commands(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "support_ultralytics_detr.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)

            dataset_report = root / "dataset_report.json"
            proc_dataset = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "dataset",
                    "--from",
                    "internal",
                    "--dataset",
                    "data/smoke",
                    "--split",
                    "val",
                    "--output",
                    str(root / "dataset_cache"),
                    "--report",
                    str(dataset_report),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc_dataset.returncode != 0:
                self.fail(f"dataset command failed:\n{proc_dataset.stdout}\n{proc_dataset.stderr}")
            dataset_payload = json.loads(dataset_report.read_text(encoding="utf-8"))
            self.assertTrue(bool(dataset_payload.get("ok")))
            self.assertEqual(str(dataset_payload.get("source_format")), "internal")

            train_ultra_report = root / "train_ultra_report.json"
            proc_ultra = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "train-ultralytics",
                    "--model",
                    "yolo11n.pt",
                    "--dataset",
                    "data/smoke",
                    "--split",
                    "val",
                    "--dry-run",
                    "--output",
                    str(train_ultra_report),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc_ultra.returncode != 0:
                self.fail(f"train-ultralytics --dry-run failed:\n{proc_ultra.stdout}\n{proc_ultra.stderr}")
            ultra_payload = json.loads(train_ultra_report.read_text(encoding="utf-8"))
            self.assertTrue(bool(ultra_payload.get("ok")))
            self.assertTrue(bool(ultra_payload.get("dry_run")))
            self.assertIn("template_train_command", ultra_payload)

            train_hf_report = root / "train_hf_report.json"
            proc_hf = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "train-hf-detr",
                    "--model-id",
                    "facebook/detr-resnet-50",
                    "--dataset",
                    "data/smoke",
                    "--split",
                    "val",
                    "--dry-run",
                    "--output",
                    str(train_hf_report),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc_hf.returncode != 0:
                self.fail(f"train-hf-detr --dry-run failed:\n{proc_hf.stdout}\n{proc_hf.stderr}")
            hf_payload = json.loads(train_hf_report.read_text(encoding="utf-8"))
            self.assertTrue(bool(hf_payload.get("ok")))
            self.assertTrue(bool(hf_payload.get("dry_run")))

            export_report = root / "export_onnx_report.json"
            out_onnx = root / "model.onnx"
            proc_export = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "export-onnx",
                    "--provider",
                    "ultralytics",
                    "--model",
                    "yolo11n.pt",
                    "--output",
                    str(out_onnx),
                    "--dry-run",
                    "--report",
                    str(export_report),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc_export.returncode != 0:
                self.fail(f"export-onnx --dry-run failed:\n{proc_export.stdout}\n{proc_export.stderr}")
            export_payload = json.loads(export_report.read_text(encoding="utf-8"))
            self.assertTrue(bool(export_payload.get("ok")))
            self.assertTrue(bool(export_payload.get("dry_run")))

    def test_predict_normalize_from_input(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "support_ultralytics_detr.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            src = root / "raw_preds.json"
            src.write_text(
                json.dumps(
                    [
                        {
                            "image": "images/val/000001.jpg",
                            "detections": [
                                {"class_id": 1, "score": 0.2, "bbox": {"cx": 0.4, "cy": 0.5, "w": 0.2, "h": 0.2}},
                                {"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.6, "w": 0.2, "h": 0.2}},
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            out = root / "normalized.json"
            report = root / "normalize_report.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "predict-normalize",
                    "--input",
                    str(src),
                    "--output",
                    str(out),
                    "--report",
                    str(report),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"predict-normalize failed:\n{proc.stdout}\n{proc.stderr}")
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 1)
            dets = (payload[0] or {}).get("detections") or []
            self.assertEqual(len(dets), 2)
            self.assertGreaterEqual(float(dets[0]["score"]), float(dets[1]["score"]))


if __name__ == "__main__":
    unittest.main()
