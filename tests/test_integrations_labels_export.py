"""Real label-oracle export across CLI, MCP and Actions; no mocked jobs."""
from __future__ import annotations

import argparse
import asyncio
from base64 import b64decode
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

REPO = Path(__file__).resolve().parents[1]
PNG = b64decode("iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAFElEQVR4nGNkYPjPgA0wYRUdtBIAy0MBD1Y0SxIAAAAASUVORK5CYII=")


def make_fixture(root: Path) -> None:
    images = root / "dataset/images/val"
    labels = root / "dataset/labels/val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for stem, content in (
        ("a", "0 0.3 0.3 0.2 0.2\n1 0.7 0.7 0.2 0.2\n"),
        ("b", "1 0.5 0.5 0.4 0.4\n"),
        ("c", ""),
    ):
        (images / f"{stem}.png").write_bytes(PNG)
        (labels / f"{stem}.txt").write_text(content, encoding="utf-8")
    (labels / "classes.json").write_text(json.dumps({
        "schema_version": 1, "names": ["first", "second"],
        "class_to_category_id": {"0": 1, "1": 2},
        "category_id_to_class_id": {"1": 0, "2": 1},
    }), encoding="utf-8")


def cli(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "yolozu", *args], cwd=root,
        env=dict(os.environ, PYTHONPATH=str(REPO)),
        capture_output=True, text=True, timeout=60,
    )


def assert_oracle(test: unittest.TestCase, root: Path, output: str) -> dict:
    payload = json.loads((root / output).read_text(encoding="utf-8"))
    test.assertEqual([len(row["detections"]) for row in payload["predictions"]], [2, 1, 0])
    test.assertEqual([[det["class_id"] for det in row["detections"]] for row in payload["predictions"]], [[0, 1], [1], []])
    test.assertTrue(all(det["score"] == 1.0 for row in payload["predictions"] for det in row["detections"]))
    test.assertEqual(payload["meta"]["adapter"], "labels")
    test.assertEqual(payload["meta"]["prediction_source"], "ground_truth_labels")
    test.assertIs(payload["meta"]["inference_executed"], False)
    test.assertEqual(payload["meta"]["purpose"], "evaluation_pipeline_oracle_self_check")
    checked = cli(root, "validate", "predictions", output, "--strict", "--json")
    test.assertEqual(checked.returncode, 0, checked.stderr)
    test.assertTrue(json.loads(checked.stdout)["ok"])
    return payload


class TestLabelsExportCLI(unittest.TestCase):
    def test_labels_help_all_labels_and_empty_images(self):
        help_result = cli(REPO, "export", "--help")
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("labels", help_result.stdout)
        self.assertIn("oracle self-check", help_result.stdout)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_fixture(root)
            result = cli(root, "export", "--backend", "labels", "--dataset", "dataset", "--split", "val", "--output", "labels.json")
            self.assertEqual(result.returncode, 0, result.stderr)
            assert_oracle(self, root, "labels.json")

    def test_labels_existing_output_and_changed_labels_fail_without_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_fixture(root)
            args = ("export", "--backend", "labels", "--dataset", "dataset", "--split", "val", "--output", "labels.json")
            self.assertEqual(cli(root, *args).returncode, 0)
            original = (root / "labels.json").read_bytes()
            self.assertEqual(cli(root, *args).returncode, 0)
            self.assertEqual((root / "labels.json").read_bytes(), original)
            (root / "dataset/labels/val/a.txt").write_text("0 0.4 0.4 0.2 0.2\n", encoding="utf-8")
            changed = cli(root, *args)
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn("config_hash", changed.stderr)
            self.assertEqual((root / "labels.json").read_bytes(), original)

    def test_labels_invalid_geometry_rejects_before_force_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_fixture(root)
            (root / "dataset/labels/val/a.txt").write_text("0 2.0 0.5 0.2 0.2\n", encoding="utf-8")
            (root / "protected.json").write_bytes(b"user-owned bytes\n")
            result = cli(root, "export", "--backend", "labels", "--dataset", "dataset", "--split", "val", "--output", "protected.json", "--force")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((root / "protected.json").read_bytes(), b"user-owned bytes\n")

    def test_labels_rejects_inference_and_adaptation_flags(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_fixture(root)
            for flags in (("--tta",), ("--ttt",), ("--ttt", "--ttt-lite-non-torch"), ("--lora-r", "1"), ("--torch-compile",), ("--infer-batch-size", "2"), ("--model", "model.onnx"), ("--checkpoint", "checkpoint.pt"), ("--device", "mps"), ("--image-size", "256"), ("--score-threshold", "0.2"), ("--max-detections", "1"), ("--config", "other.json"), ("--imgsz", "320"), ("--input-name", "custom"), ("--combined-output", "custom"), ("--boxes-scale", "norm"), ("--min-score", "0.5"), ("--topk", "1"), ("--score-thr", "0.5"), ("--keep-aspect",), ("--dnn-backend", "cuda"), ("--dnn-target", "cuda"), ("--nms-iou", "0.7"), ("--agnostic-nms",), ("--raw-format", "yolo_85_obj"), ("--decode", "rtdetr"), ("--preprocess", "rtdetr_resize_640"), ("--dump-io", "probe.json"), ("--dry-run",)):
                with self.subTest(flags=flags):
                    result = cli(root, "export", "--backend", "labels", "--dataset", "dataset", "--output", "rejected.json", *flags)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse((root / "rejected.json").exists())

    def test_labels_rejected_for_folder_inference(self):
        from yolozu.inference.export_orchestrator import parse_common_export_args

        parser = argparse.ArgumentParser()
        parse_common_export_args(parser, include_labels=False)
        backend_action = next(action for action in parser._actions if action.dest == "backend")
        self.assertNotIn("labels", backend_action.choices)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_fixture(root)
            result = cli(root, "predict-images", "--backend", "labels", "--input-dir", "dataset/images/val", "--output", "folder_labels.json")
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertFalse((root / "folder_labels.json").exists())


@unittest.skipUnless(importlib.util.find_spec("mcp") is not None, "optional MCP dependency not installed")
class TestLabelsExportLiveMCP(unittest.TestCase):
    def test_real_stdio_job_export_validate_and_evaluate(self):
        from mcp import Client, StdioServerParameters

        async def scenario(root: Path):
            params = StdioServerParameters(command=sys.executable, args=["-m", "yolozu.integrations.mcp_cli"], cwd=str(root), env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(REPO)})
            async with Client(params, mode="legacy") as client:
                async def call(name, arguments):
                    response = await asyncio.wait_for(client.call_tool(name, arguments), 60)
                    self.assertFalse(response.is_error)
                    return json.loads(response.content[0].text)
                queued = await call("export_predictions_job", {"dataset": "dataset", "split": "val", "output": "mcp_labels.json", "force": False})
                self.assertTrue(queued["ok"])
                for _ in range(60):
                    status = await call("jobs_status", {"job_id": queued["job_id"]})
                    if status["job"]["status"] in {"completed", "failed", "cancelled", "timed_out"}:
                        break
                    await asyncio.sleep(1)
                self.assertEqual(status["job"]["status"], "completed", status)
                output = status["job"]["result"]["artifacts"]["predictions"]
                assert_oracle(self, root, output)
                validation = await call("validate_predictions", {"path": output, "strict": True})
                self.assertTrue(validation["ok"])
                if importlib.util.find_spec("pycocotools") is not None:
                    evaluated = await call("eval_coco", {"dataset": "dataset", "split": "val", "predictions": output, "dry_run": False, "output": "mcp_eval.json"})
                    self.assertTrue(evaluated["ok"], evaluated)
                    evidence = json.loads((root / "mcp_eval.json").read_text())
                    self.assertEqual(evidence["counts"]["images"], 3)
                    self.assertEqual(evidence["counts"]["detections"], 3)
                    self.assertEqual(evidence["metrics"]["map50_95"], 1.0)
                failed = await call("export_predictions_job", {"dataset": "missing", "output": "missing.json", "force": False})
                for _ in range(60):
                    status = await call("jobs_status", {"job_id": failed["job_id"]})
                    if status["job"]["status"] in {"completed", "failed"}:
                        break
                    await asyncio.sleep(1)
                self.assertFalse(status["ok"])
                self.assertEqual(status["job"]["status"], "failed")
                self.assertEqual(status["job"]["result"]["artifacts"], {})
                self.assertFalse((root / "missing.json").exists())

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_fixture(root)
            asyncio.run(scenario(root))


@unittest.skipUnless(importlib.util.find_spec("fastapi") is not None, "optional Actions dependency not installed")
class TestLabelsExportActions(unittest.TestCase):
    def test_real_action_job_export_validate_and_evaluate(self):
        from fastapi.testclient import TestClient
        from yolozu.integrations.actions_api import app
        from yolozu.integrations import tool_runner

        previous = Path.cwd()
        previous_env = os.environ.get("PYTHONPATH")
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                make_fixture(root)
                os.chdir(root)
                os.environ["PYTHONPATH"] = str(REPO)
                tool_runner._job_manager.cache_clear()
                with TestClient(app) as client:
                    queued = client.post("/jobs/export-predictions", json={"dataset": "dataset", "split": "val", "output": "actions_labels.json", "force": False}).json()
                    self.assertTrue(queued["ok"])
                    for _ in range(60):
                        status = client.get(f"/jobs/{queued['job_id']}").json()
                        if status["job"]["status"] in {"completed", "failed", "cancelled", "timed_out"}:
                            break
                        time.sleep(1)
                    self.assertEqual(status["job"]["status"], "completed", status)
                    output = status["job"]["result"]["artifacts"]["predictions"]
                    assert_oracle(self, root, output)
                    self.assertTrue(client.post("/validate/predictions", json={"path": output}).json()["ok"])
                    if importlib.util.find_spec("pycocotools") is not None:
                        evaluated = client.post("/eval/coco", json={"dataset": "dataset", "split": "val", "predictions": output, "dry_run": False, "output": "actions_eval.json"}).json()
                        self.assertTrue(evaluated["ok"], evaluated)
                        evidence = json.loads((root / "actions_eval.json").read_text())
                        self.assertEqual(evidence["counts"]["images"], 3)
                        self.assertEqual(evidence["counts"]["detections"], 3)
                        self.assertEqual(evidence["metrics"]["map50_95"], 1.0)
                    failed = client.post("/jobs/export-predictions", json={"dataset": "missing", "output": "missing.json", "force": False}).json()
                    for _ in range(60):
                        status = client.get(f"/jobs/{failed['job_id']}").json()
                        if status["job"]["status"] in {"completed", "failed"}:
                            break
                        time.sleep(1)
                    self.assertFalse(status["ok"])
                    self.assertEqual(status["job"]["status"], "failed")
                    self.assertNotIn("stdout", status["job"]["result"])
                    self.assertNotIn("stderr", status["job"]["result"])
                    self.assertEqual(status["job"]["result"]["artifacts"], {})
                tool_runner._job_manager().shutdown(timeout=10)
                tool_runner._job_manager.cache_clear()
        finally:
            os.chdir(previous)
            if previous_env is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous_env


if __name__ == "__main__":
    unittest.main()
