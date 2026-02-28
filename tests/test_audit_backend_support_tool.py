import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestAuditBackendSupportTool(unittest.TestCase):
    def test_audit_backend_support_dry_run(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "audit_backend_support.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            images = dataset_root / "images" / "val2017"
            labels = dataset_root / "labels" / "val2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)

            (images / "000001.jpg").write_bytes(b"")
            (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            out = root / "backend_support_audit.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-root",
                    str(dataset_root),
                    "--split",
                    "val2017",
                    "--max-images",
                    "1",
                    "--output",
                    str(out),
                ],
                text=True,
                capture_output=True,
                check=False,
                cwd=str(repo_root),
            )
            if proc.returncode != 0:
                self.fail(f"audit_backend_support.py failed:\nstdout={proc.stdout}\nstderr={proc.stderr}")

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(bool(payload.get("ok")))
            results = payload.get("results") or []
            self.assertEqual(len(results), 4)
            self.assertTrue(all(bool(item.get("ok")) for item in results))
            self.assertTrue(all(bool(item.get("dry_run", True)) for item in results))

    def test_audit_backend_support_require_non_dry_fails_without_selection(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "audit_backend_support.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            images = dataset_root / "images" / "val2017"
            labels = dataset_root / "labels" / "val2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)

            (images / "000001.jpg").write_bytes(b"")
            (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            out = root / "backend_support_audit.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-root",
                    str(dataset_root),
                    "--split",
                    "val2017",
                    "--max-images",
                    "1",
                    "--output",
                    str(out),
                    "--require-non-dry",
                ],
                text=True,
                capture_output=True,
                check=False,
                cwd=str(repo_root),
            )
            self.assertNotEqual(proc.returncode, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(bool(payload.get("ok")))
            warnings = payload.get("warnings") or []
            self.assertTrue(any("require-non-dry" in str(w) for w in warnings))


if __name__ == "__main__":
    unittest.main()
