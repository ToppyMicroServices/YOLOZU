import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestPredictionsMigrateCli(unittest.TestCase):
    def _run(self, args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "yolozu", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def test_predictions_migrate_v1_to_v2(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            src = root / "pred_v1.json"
            dst = root / "pred_v2.json"
            src.write_text(
                json.dumps(
                    [
                        {
                            "schema_version": 1,
                            "image": "a.jpg",
                            "detections": [{"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}],
                        },
                        {
                            "image": "b.jpg",
                            "detections": [{"class_id": 1, "score": 0.8, "bbox": {"cx": 0.4, "cy": 0.4, "w": 0.2, "h": 0.2}}],
                        },
                    ]
                ),
                encoding="utf-8",
            )

            proc = self._run(
                [
                    "predictions",
                    "migrate",
                    "--input",
                    str(src),
                    "--output",
                    str(dst),
                    "--from",
                    "v1",
                    "--to",
                    "v2",
                    "--force",
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"predictions migrate failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

            migrated = json.loads(dst.read_text(encoding="utf-8"))
            self.assertEqual(len(migrated), 2)
            self.assertEqual(int(migrated[0]["schema_version"]), 2)
            self.assertEqual(int(migrated[1]["schema_version"]), 2)

    def test_predictions_migrate_strict_source_rejects_v2_input(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            src = root / "pred_v2.json"
            dst = root / "pred_out.json"
            src.write_text(
                json.dumps(
                    [
                        {
                            "schema_version": 2,
                            "image": "a.jpg",
                            "detections": [{"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            proc = self._run(
                [
                    "predictions",
                    "migrate",
                    "--input",
                    str(src),
                    "--output",
                    str(dst),
                    "--from",
                    "v1",
                    "--to",
                    "v2",
                    "--strict-source",
                    "--force",
                ],
                cwd=repo_root,
            )
            self.assertNotEqual(proc.returncode, 0, msg=f"expected non-zero exit, got stdout={proc.stdout} stderr={proc.stderr}")
            self.assertIn("from version policy", proc.stderr)


if __name__ == "__main__":
    unittest.main()
