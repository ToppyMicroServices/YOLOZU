import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestExternalFinetuneSmokeTool(unittest.TestCase):
    def _make_dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        images_train = dataset / "images" / "train"
        labels_train = dataset / "labels" / "train"
        images_val = dataset / "images" / "val"
        labels_val = dataset / "labels" / "val"
        images_train.mkdir(parents=True, exist_ok=True)
        labels_train.mkdir(parents=True, exist_ok=True)
        images_val.mkdir(parents=True, exist_ok=True)
        labels_val.mkdir(parents=True, exist_ok=True)

        (images_train / "000001.jpg").write_bytes(b"")
        (labels_train / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (images_val / "000002.jpg").write_bytes(b"")
        (labels_val / "000002.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (labels_train / "classes.json").write_text(json.dumps(["object"], ensure_ascii=False), encoding="utf-8")
        return dataset

    def _write_exec(self, path: Path, body: str) -> Path:
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
        return path

    def test_external_finetune_smoke_dry_run(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_external_finetune_smoke.py"
        self.assertTrue(script.is_file(), "missing tools/run_external_finetune_smoke.py")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root)
            out = root / "external_finetune_smoke.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-root",
                    str(dataset),
                    "--split",
                    "train",
                    "--output",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"run_external_finetune_smoke.py failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(bool(payload.get("ok")))
            self.assertEqual((payload.get("counts") or {}).get("frameworks"), 5)
            results = payload.get("results") or []
            self.assertEqual(len(results), 5)
            self.assertTrue(all(bool(row.get("dry_run", False)) for row in results))
            frameworks = {str(row.get("framework")) for row in results}
            self.assertEqual(frameworks, {"yolox", "yolov", "mmdetection", "detectron2", "rtdetr"})

    def test_external_finetune_smoke_require_non_dry_fails_without_selection(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_external_finetune_smoke.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root)
            out = root / "external_finetune_smoke.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-root",
                    str(dataset),
                    "--split",
                    "train",
                    "--output",
                    str(out),
                    "--require-non-dry",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(bool(payload.get("ok")))
            warnings = payload.get("warnings") or []
            self.assertTrue(any("require-non-dry" in str(w) for w in warnings))

    def test_external_finetune_template_files_exist(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        templates = [
            repo_root / "configs" / "examples" / "finetune_external" / "yolox_s_finetune_smoke.py",
            repo_root / "configs" / "examples" / "finetune_external" / "ultralytics_yolov8n_finetune_smoke.yaml",
            repo_root / "configs" / "examples" / "finetune_external" / "mmdetection_finetune_smoke.py",
            repo_root / "configs" / "examples" / "finetune_external" / "detectron2_finetune_smoke.yaml",
            repo_root / "configs" / "examples" / "finetune_external" / "rtdetr_pose_finetune_smoke.yaml",
        ]
        for path in templates:
            self.assertTrue(path.is_file(), f"missing template: {path}")

    def test_external_finetune_smoke_rtdetr_non_dry_reports_missing_torch(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_external_finetune_smoke.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root)
            out = root / "external_finetune_smoke.json"
            fake_python = self._write_exec(
                root / "fake_python_no_torch.py",
                """#!/usr/bin/env python3
import subprocess
import sys

if len(sys.argv) >= 3 and sys.argv[1] == "-c" and "import torch" in sys.argv[2]:
    sys.stderr.write("ModuleNotFoundError: No module named 'torch'\\n")
    raise SystemExit(1)

raise SystemExit(subprocess.call([sys.executable] + sys.argv[1:]))
""",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-root",
                    str(dataset),
                    "--split",
                    "train",
                    "--framework",
                    "rtdetr",
                    "--non-dry-framework",
                    "rtdetr",
                    "--python",
                    str(fake_python),
                    "--output",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            results = payload.get("results") or []
            self.assertEqual(len(results), 1)
            row = dict(results[0])
            self.assertEqual(str(row.get("framework")), "rtdetr")
            self.assertEqual(str(row.get("failure_code")), "E_DEP_TORCH_MISSING")
            self.assertIn("requires torch", str(row.get("runtime_error")))
            self.assertFalse(bool(row.get("training_executed")))

    def test_external_finetune_smoke_external_train_scripts_are_audited(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_external_finetune_smoke.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root)
            out = root / "external_finetune_smoke.json"
            mmdet_script = self._write_exec(
                root / "mmdet_train_stub.py",
                """#!/usr/bin/env python3
import sys
raise SystemExit(0)
""",
            )
            detectron2_script = self._write_exec(
                root / "detectron2_train_stub.py",
                """#!/usr/bin/env python3
import sys
raise SystemExit(0)
""",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-root",
                    str(dataset),
                    "--split",
                    "train",
                    "--framework",
                    "mmdetection",
                    "--framework",
                    "detectron2",
                    "--non-dry-framework",
                    "mmdetection",
                    "--non-dry-framework",
                    "detectron2",
                    "--mmdet-train-script",
                    str(mmdet_script),
                    "--detectron2-train-script",
                    str(detectron2_script),
                    "--require-training-execution",
                    "--output",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"run_external_finetune_smoke.py failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(bool(payload.get("ok")))
            self.assertEqual((payload.get("counts") or {}).get("training_executed"), 2)
            results = payload.get("results") or []
            self.assertEqual(len(results), 2)
            by_name = {str(row.get("framework")): row for row in results}
            self.assertEqual(set(by_name.keys()), {"mmdetection", "detectron2"})
            for row in by_name.values():
                self.assertTrue(bool(row.get("train_path_audited")))
                self.assertTrue(bool(row.get("train_script_configured")))
                self.assertTrue(bool(row.get("training_executed")))
                self.assertGreaterEqual(len(list(row.get("aux_commands") or [])), 1)
                self.assertIn("projection_executed", row)
                if not bool(row.get("projection_executed")):
                    self.assertTrue(str(row.get("projection_error")))

    def test_external_finetune_smoke_yolox_train_script_is_audited(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_external_finetune_smoke.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root)
            out = root / "external_finetune_smoke.json"
            yolox_script = self._write_exec(
                root / "yolox_train_stub.py",
                """#!/usr/bin/env python3
import sys
raise SystemExit(0)
""",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-root",
                    str(dataset),
                    "--split",
                    "train",
                    "--framework",
                    "yolox",
                    "--non-dry-framework",
                    "yolox",
                    "--yolox-train-script",
                    str(yolox_script),
                    "--require-training-execution",
                    "--output",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"run_external_finetune_smoke.py failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(bool(payload.get("ok")))
            results = payload.get("results") or []
            self.assertEqual(len(results), 1)
            row = dict(results[0])
            self.assertEqual(str(row.get("framework")), "yolox")
            self.assertTrue(bool(row.get("projection_executed")))
            self.assertTrue(bool(row.get("train_path_audited")))
            self.assertTrue(bool(row.get("train_script_configured")))
            self.assertTrue(bool(row.get("training_executed")))

    def test_non_dry_yolox_without_train_script_fails_closed(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_external_finetune_smoke.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root)
            out = root / "external_finetune_smoke.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-root",
                    str(dataset),
                    "--framework",
                    "yolox",
                    "--non-dry-framework",
                    "yolox",
                    "--output",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            row = json.loads(out.read_text(encoding="utf-8"))["results"][0]
            self.assertFalse(bool(row["ok"]))
            self.assertFalse(bool(row["training_executed"]))
            self.assertTrue(bool(row["projection_executed"]))
            self.assertEqual(row["failure_code"], "E_EXTERNAL_TRAIN_SCRIPT_REQUIRED")
            self.assertIn("projection is not training", row["runtime_error"])


if __name__ == "__main__":
    unittest.main()
