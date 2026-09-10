import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from yolozu.predictions import validate_predictions_payload
from yolozu.predictions.export import export_dummy_predictions


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

    def test_wrapped_migration_without_meta_roundtrips(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "source.json"
            dst = Path(td) / "migrated.json"
            for versioned in (False, True):
                payload = {"predictions": [{"schema_version": 1, "image": "sample.jpg", "detections": []}]}
                if versioned:
                    payload["schema_version"] = 1
                validate_predictions_payload(payload, strict=True)
                src.write_text(json.dumps(payload), encoding="utf-8")
                proc = self._run(
                    ["predictions", "migrate", "--input", str(src), "--output", str(dst), "--from", "v1", "--to", "v2", "--force"],
                    cwd=repo_root,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                migrated = json.loads(dst.read_text(encoding="utf-8"))
                self.assertNotIn("meta", migrated)
                self.assertEqual(migrated["entry_schema_migration"]["migrated_candidates"], 1)
                self.assertEqual(migrated["predictions"][0]["schema_version"], 2)
                self.assertEqual("schema_version" in migrated, versioned)
                validate_predictions_payload(migrated, strict=True)
                checked = self._run(["validate", "predictions", str(dst), "--strict"], cwd=repo_root)
                self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_wrapped_migration_preserves_exporter_meta(self):
        repo_root = Path(__file__).resolve().parents[1]
        payload, _ = export_dummy_predictions(dataset_root=repo_root / "data/smoke", split="val", max_images=1)
        original_meta = dict(payload["meta"])
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td) / "source.json", Path(td) / "migrated.json"
            src.write_text(json.dumps(payload), encoding="utf-8")
            proc = self._run(
                ["predictions", "migrate", "--input", str(src), "--output", str(dst), "--from", "v1", "--to", "v2"],
                cwd=repo_root,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            migrated = json.loads(dst.read_text(encoding="utf-8"))
            self.assertIn("entry_schema_migration", migrated["meta"])
            self.assertEqual({k: v for k, v in migrated["meta"].items() if k != "entry_schema_migration"}, original_meta)
            validate_predictions_payload(migrated, strict=True)

    def test_migration_rejects_invalid_versions_without_replacing_output(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td) / "source.json", Path(td) / "migrated.json"
            for location, version in (("wrapper", 99), ("wrapper", None), ("entry", None), ("entry", 99)):
                entry = {"schema_version": 1, "image": "sample.jpg", "detections": []}
                payload = {"schema_version": 1, "predictions": [entry]}
                (payload if location == "wrapper" else entry)["schema_version"] = version
                src.write_text(json.dumps(payload), encoding="utf-8")
                dst.write_text("existing output\n", encoding="utf-8")
                with self.subTest(location=location, version=version):
                    proc = self._run(
                        ["predictions", "migrate", "--input", str(src), "--output", str(dst), "--from", "v1", "--to", "v2", "--force"],
                        cwd=repo_root,
                    )
                    self.assertNotEqual(proc.returncode, 0)
                    self.assertIn("schema_version" if location == "wrapper" else "from version policy", proc.stderr)
                    self.assertEqual(dst.read_text(encoding="utf-8"), "existing output\n")


if __name__ == "__main__":
    unittest.main()
