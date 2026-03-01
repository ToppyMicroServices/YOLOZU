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
            self.assertEqual((payload.get("counts") or {}).get("frameworks"), 4)
            results = payload.get("results") or []
            self.assertEqual(len(results), 4)
            self.assertTrue(all(bool(row.get("dry_run", False)) for row in results))
            frameworks = {str(row.get("framework")) for row in results}
            self.assertEqual(frameworks, {"yolov", "mmdetection", "detectron2", "rtdetr"})

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
            repo_root / "configs" / "examples" / "finetune_external" / "ultralytics_yolov8n_finetune_smoke.yaml",
            repo_root / "configs" / "examples" / "finetune_external" / "mmdetection_finetune_smoke.py",
            repo_root / "configs" / "examples" / "finetune_external" / "detectron2_finetune_smoke.yaml",
            repo_root / "configs" / "examples" / "finetune_external" / "rtdetr_pose_finetune_smoke.yaml",
        ]
        for path in templates:
            self.assertTrue(path.is_file(), f"missing template: {path}")


if __name__ == "__main__":
    unittest.main()
