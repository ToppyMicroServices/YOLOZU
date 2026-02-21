import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _targets_doc() -> dict:
    return {
        "protocol_id": "yolo26",
        "metric_key": "map50_95",
        "imgsz": 640,
        "targets": {
            "yolo26n": 0.1,
            "yolo26s": 0.1,
            "yolo26m": 0.1,
            "yolo26l": 0.1,
            "yolo26x": 0.1,
        },
    }


class TestCheckMapTargetsContract(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.tool = _load_module(self.repo_root / "tools" / "check_map_targets.py", "check_map_targets")

    def _run_tool(self, suite_path: Path, targets_path: Path):
        out = io.StringIO()
        with redirect_stdout(out):
            try:
                self.tool.main(["--suite", str(suite_path), "--targets", str(targets_path)])
                return 0, json.loads(out.getvalue())
            except SystemExit as exc:
                return int(exc.code), json.loads(out.getvalue())

    def test_fails_with_clear_message_when_top_level_key_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            suite_path = tmp / "eval_suite_missing_results.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "timestamp": "2026-02-21T00:00:00Z",
                        "dataset": "data/coco128",
                        "bbox_format": "cxcywh_norm",
                        "images": 1,
                    }
                )
            )
            targets_path = tmp / "targets.json"
            targets_path.write_text(json.dumps(_targets_doc()))

            code, report = self._run_tool(suite_path, targets_path)
            self.assertEqual(code, 2)
            self.assertFalse(report.get("ok"))
            failures = report.get("failures") or []
            self.assertIn("suite contract: missing top-level key `results`", failures)

    def test_fails_with_clear_message_when_metrics_key_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            suite_path = tmp / "eval_suite_missing_metric.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "timestamp": "2026-02-21T00:00:00Z",
                        "dataset": "data/coco128",
                        "bbox_format": "cxcywh_norm",
                        "images": 1,
                        "results": [
                            {
                                "name": "pred_yolo26n",
                                "path": "reports/pred_yolo26n.json",
                                "metrics": {"map50": 0.2, "map75": 0.1, "ar100": 0.3},
                            }
                        ],
                    }
                )
            )
            targets_path = tmp / "targets.json"
            targets_path.write_text(json.dumps(_targets_doc()))

            code, report = self._run_tool(suite_path, targets_path)
            self.assertEqual(code, 2)
            self.assertFalse(report.get("ok"))
            failures = report.get("failures") or []
            self.assertIn("suite contract: missing key `results[0].metrics.map50_95`", failures)

    def test_passes_with_valid_report_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            suite_path = tmp / "eval_suite_valid.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "timestamp": "2026-02-21T00:00:00Z",
                        "dataset": "data/coco128",
                        "bbox_format": "cxcywh_norm",
                        "images": 1,
                        "results": [
                            {
                                "name": "pred_yolo26n",
                                "path": "reports/pred_yolo26n.json",
                                "warnings": [],
                                "metrics": {"map50_95": 0.2, "map50": 0.3, "map75": 0.1, "ar100": 0.4},
                                "stats": [],
                                "dry_run": False,
                                "counts": {"images": 1, "detections": 1},
                            }
                        ],
                    }
                )
            )
            targets_path = tmp / "targets.json"
            targets_path.write_text(json.dumps(_targets_doc()))

            code, report = self._run_tool(suite_path, targets_path)
            self.assertEqual(code, 0)
            self.assertTrue(report.get("ok"))
            self.assertEqual(report.get("failures"), [])


if __name__ == "__main__":
    unittest.main()